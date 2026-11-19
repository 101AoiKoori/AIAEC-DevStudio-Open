"""Check real-parent compositor plan mapping with synthetic test doubles."""
import importlib
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
entry = importlib.import_module('generate_sticker_real_parent_v4')
NAMES = [f'item_{i}' for i in range(24)]


def plan(root, cls=3, count=2):
    name = NAMES[cls] if cls is not None else None
    bg = 'background.png'
    params = {'class_id': cls, 'gain': .72, 'saturation': .65}
    state = {'id': 'state', 'planned_count': count, 'parameters': params,
             'background': {'kind': 'real_frame', 'image': bg},
             'target': {'kind': 'sticker', 'class_id': cls, 'name': name} if name else {'kind': 'negative'},
             'labels': {'format': 'yolo_detection', 'class_id': cls},
             'augmentation': {'gain': .72, 'saturation': .65, 'variation': 'seeded_position_scale_and_paper'},
             'sources': [bg, 'scene.png', 'mask.png'] +
                        ([f'Archive/Assets/working_sources/digital/waste_sticker/toolkit/SourcePic/{name}.png'] if name else [])}
    return {'model_key': 'waste_sticker', 'target_split': 'train',
            'constraints': dict(entry.CONSTRAINTS), 'strata': [state]}


class ParentPlanTests(unittest.TestCase):
    def test_rejects_unsupported_state_or_constraint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(entry, 'ROOT', root), mock.patch.object(entry, 'BACKGROUND', root / 'background.png'):
                for change in (lambda p: p['constraints'].update(paper_size_rule='different'),
                               lambda p: p['strata'][0]['background'].update(image='other.png'),
                               lambda p: p['strata'][0]['parameters'].update(yaw=1),
                               lambda p: p['strata'][0]['augmentation'].update(saturation=1)):
                    candidate = plan(root)
                    change(candidate)
                    with self.assertRaises(ValueError):
                        entry.validate_plan(candidate, NAMES, root/'scene.png', root/'mask.png')

    def test_plan_drives_each_image_and_records_source_and_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src, mask, manifest = root/'scene.png', root/'mask.png', root/'scene.json'
            manifest.write_text(json.dumps({'samples': [{'image': src.name, 'mask_visible': mask.name}]}), encoding='utf8')
            source = np.zeros((480, 640, 3), dtype=np.uint8)
            raw = np.zeros((480, 640, 4), dtype=np.uint8)
            raw[50:300, 100:240, 1] = 255
            raw[50:300, 100:240, 3] = 255
            background = np.zeros((1080, 1920, 3), dtype=np.uint8)
            def read(path, *unused):
                return raw.copy() if path == mask else background.copy() if path.name == 'background.png' else source.copy()
            def write(path, image):
                path.write_bytes(b'fixture')
            tones = []
            def tone(image, value_gain, saturation_gain):
                tones.append((value_gain, saturation_gain))
                return image
            combined = plan(root)
            negative = plan(root, cls=None, count=1)['strata'][0]
            negative['id'] = 'negative'
            combined['strata'].append(negative)
            run = SimpleNamespace(plan=combined, source_refs={}, record_sample=mock.Mock())
            with mock.patch.object(entry, 'ROOT', root), mock.patch.object(entry, 'BACKGROUND', root/'background.png'), \
                 mock.patch.object(entry, 'BACKGROUND_MANIFEST', root/'background.json'), \
                 mock.patch.object(entry.base, 'load_icons', return_value=(NAMES, [None]*24, None, None)), \
                 mock.patch.object(entry.base, 'read_image', side_effect=read), \
                 mock.patch.object(entry.grouped, 'write_png', side_effect=write), \
                 mock.patch.object(entry.grouped, 'apply_tone', side_effect=tone), \
                 mock.patch.object(entry, 'place_paper_avoiding_bright_detail', return_value=([10, 30, 40, 60], [60, 60, 80, 80])):
                entry.produce(root/'out', src, mask, manifest, 11, run)
            result = json.loads((root/'out/manifest.json').read_text(encoding='utf8'))
            self.assertEqual(len(result['records']), 3)
            self.assertEqual(tones, [(.72, .65)] * 3)
            self.assertEqual(run.record_sample.call_count, 3)
            actual = [call.kwargs['actual'] for call in run.record_sample.call_args_list]
            self.assertNotEqual(actual[0]['seed'], actual[1]['seed'])
            self.assertEqual(actual[0]['source_manifest_record_index'], 0)
            self.assertEqual(result['records'][0]['source_mask_direct_reference']['field'], 'mask_visible')
            self.assertEqual([r['state_id'] for r in result['records']], ['state', 'state', 'negative'])
            self.assertEqual((root/'out/labels/train/train_00002.txt').read_text(encoding='utf8'), '')
            self.assertFalse((root/'out/images/val').exists())


if __name__ == '__main__':
    unittest.main()
