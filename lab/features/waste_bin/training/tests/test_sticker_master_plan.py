"""Test master-bin planning and provenance with test doubles."""
import importlib
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
entry = importlib.import_module('generate_sticker_master_bin_v4')
NAMES = [f'item_{i}' for i in range(24)]


def state(cls, count=2):
    color = 'red' if cls is not None else None
    name = NAMES[cls] if cls is not None else None
    params = {'class_id': cls, 'bin_color': color, 'gain': .72, 'saturation': .65}
    return {'id': 'positive' if name else 'negative', 'planned_count': count, 'parameters': params,
            'background': {'kind': 'blender_render_pool', 'manifests': ['source/manifest.json']},
            'target': {'kind': 'sticker', 'class_id': cls, 'name': name} if name else {'kind': 'candidate_negative'},
            'labels': {'format': 'yolo_detection', 'class_id': cls},
            'augmentation': {'gain': .72, 'saturation': .65, 'variation': 'seeded_source_and_paper_placement'},
            'sources': ['source/manifest.json'] + ([f'Archive/Assets/working_sources/digital/waste_sticker/toolkit/SourcePic/{name}.png'] if name else [])}


def plan():
    return {'model_key': 'waste_sticker', 'target_split': 'train', 'constraints': dict(entry.CONSTRAINTS),
            'strata': [state(3), state(None, 1)]}


class MasterPlanTests(unittest.TestCase):
    def test_missing_plan_is_rejected_before_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'out'
            with mock.patch.object(sys, 'argv', ['master', '--out', str(out), '--source-manifest', 'missing.json']), \
                 mock.patch.object(entry, 'generation_guard') as guard, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    entry.main()
            guard.assert_not_called()
            self.assertFalse(out.exists())

    def test_state_and_constraint_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(entry, 'ROOT', root):
                manifest = root/'source/manifest.json'
                entry.validate_plan(plan(), NAMES, [manifest])
                for change in (lambda p: p['constraints'].update(min_paper_side_px=10),
                               lambda p: p['strata'][0]['parameters'].update(yaw=5),
                               lambda p: p['strata'][0]['sources'].append('unrelated'),
                               lambda p: p['strata'][0]['labels'].update(class_id=4)):
                    candidate = plan()
                    change(candidate)
                    with self.assertRaises(ValueError):
                        entry.validate_plan(candidate, NAMES, [manifest])

    def test_planned_jobs_record_source_and_preserve_negative_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root/'source/manifest.json'
            source_image, source_mask = root/'source/frame.png', root/'source/mask.png'
            report = {'scene_file': 'scene.blend', 'bin_source': 'bin.blend'}
            sample = {'camera_location_fitted': [1, 2, 3]}
            sources = [(manifest, report, sample, ['red'], 4, source_image, source_mask)]
            refs = {'source/frame.png': [{'manifest': 'source/manifest.json', 'record_index': 4}]}
            run = SimpleNamespace(plan=plan(), source_refs=refs, record_sample=mock.Mock())
            original = np.zeros((480, 640, 3), dtype=np.uint8)
            tones = []
            def tone(image, value_gain, saturation_gain):
                tones.append((value_gain, saturation_gain))
                return image
            def write(path, image):
                path.write_bytes(b'fixture')
            with mock.patch.object(entry, 'ROOT', root), mock.patch.object(entry.base, 'read_image', return_value=original), \
                 mock.patch.object(entry, 'decode_masks', return_value={'red': np.ones((480, 640), dtype=np.uint8)}), \
                 mock.patch.object(entry, 'place_avoiding_visible_graphics', return_value=(original, (20, 30, 50, 60))), \
                 mock.patch.object(entry.grouped, 'apply_tone', side_effect=tone), \
                 mock.patch.object(entry.grouped, 'write_png', side_effect=write):
                entry.produce(root/'out', 17, run, NAMES, [None]*24, sources)
            report = json.loads((root/'out/manifest.json').read_text(encoding='utf8'))
            self.assertEqual([r['state_id'] for r in report['records']], ['positive', 'positive', 'negative'])
            self.assertEqual(report['records'][-1]['status'], 'candidate_negative')
            self.assertEqual((root/'out/labels/train/train_000002.txt').read_text(encoding='utf8'), '')
            self.assertEqual(tones, [(.72, .65)] * 3)
            self.assertEqual(run.record_sample.call_count, 3)
            actual = [call.kwargs['actual'] for call in run.record_sample.call_args_list]
            self.assertEqual(len({item['seed'] for item in actual}), 3)
            self.assertTrue(all(item['source_record_index'] == 4 for item in actual))
            self.assertFalse((root/'out/images/val').exists())

    def test_missing_manifest_relation_cannot_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = (root/'source/manifest.json', {'scene_file': 'a', 'bin_source': 'b'}, {}, ['red'],
                      0, root/'source/frame.png', root/'source/mask.png')
            run = SimpleNamespace(plan={'target_split': 'train', 'strata': [state(None, 1)]},
                                  source_refs={}, record_sample=mock.Mock())
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            with mock.patch.object(entry, 'ROOT', root), mock.patch.object(entry.base, 'read_image', return_value=image), \
                 mock.patch.object(entry.grouped, 'apply_tone', return_value=image), \
                 mock.patch.object(entry.grouped, 'write_png') as write:
                with self.assertRaisesRegex(ValueError, 'missing manifest record_index'):
                    entry.produce(root/'out', 17, run, NAMES, [], [source])
            run.record_sample.assert_not_called()
            write.assert_not_called()


if __name__ == '__main__':
    unittest.main()
