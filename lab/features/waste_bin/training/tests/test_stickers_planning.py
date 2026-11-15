"""Test planned sticker entrypoint with local fakes, without production data."""
import contextlib
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
entry = importlib.import_module('generate_stickers')
NAMES = [f'item_{i}' for i in range(24)]


def state(ids, layout, color, count=2):
    selected = [NAMES[i] for i in ids]
    sources = [f'toolkit/SourcePic/{name}.png' for name in selected] or ['toolkit/configs/gen_default.yaml']
    return {'id': 'one', 'planned_count': count,
            'parameters': {'class_ids': ids, 'layout': layout, 'panel_color': color},
            'background': {'kind': 'synthetic_panels', 'layout': layout,
                           'palette': color if layout == 'single' else 'shuffled'},
            'target': {'class_ids': ids, 'names': selected},
            'labels': {'format': 'yolo_detection', 'class_ids': ids},
            'augmentation': {'variation': 'seeded_toolkit_geometry_texture_photometric',
                             'config': 'toolkit/configs/gen_default.yaml'}, 'sources': sources}


def plan(states=None):
    return {'model_key': 'waste_sticker', 'target_split': 'train',
            'constraints': {'canvas': [640, 480], 'vehicle_scale': False,
                            'max_attempts': 20, 'one_sticker_per_panel': True},
            'strata': states or [state([2], 'single', 1)]}


class StickerPlanningTests(unittest.TestCase):
    def test_state_validation_and_seed_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(entry, 'WORKSPACE', root):
                candidate = plan([state([2, 5], 'row', None, 3)])
                entry.validate_plan(candidate, NAMES, root/'toolkit', 640, 480, False)
                jobs = entry.planned_jobs(candidate, NAMES, 7)
                self.assertEqual([job['items'] for job in jobs], [[NAMES[2], NAMES[5]]] * 3)
                seed_ranges = [set(range(job['seed'], job['seed'] + 20)) for job in jobs]
                self.assertTrue(all(not (a & b) for i, a in enumerate(seed_ranges) for b in seed_ranges[i+1:]))
                for bad in (state([2, 2], 'row', None), state([2, 5], 'single', 1),
                            state([2], 'row', 1)):
                    with self.assertRaises(ValueError):
                        entry.validate_plan(plan([bad]), NAMES, root/'toolkit', 640, 480, False)

    def test_each_job_checks_classes_and_writes_own_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for sub in ('images/train', 'labels/train', 'metadata/train', 'preview'):
                (root/sub).mkdir(parents=True)
            current = state([2], 'single', 1, 1)
            job = entry.planned_jobs(plan([current]), NAMES, 9)[0]
            calls = []
            def generate(request):
                calls.append(request)
                return {'n': 1, 'det': ['0 .5 .5 .3 .3'],
                        'inst': [{'name': NAMES[2], 'xyxy': [1, 2, 4, 5]}],
                        'img': np.zeros((8, 8, 3), dtype=np.uint8), 'panel': 'blue', 'distractor': None}
            def write(path, image, quality):
                Path(path).write_bytes(b'fixture')
                return True
            gen = SimpleNamespace(gen_one=generate, imwrite_u=write,
                                  cv2=SimpleNamespace(rectangle=lambda *args: None))
            cfg = {'scene': {'layout': {}, 'n_panels': {}}}
            with mock.patch.object(entry, 'GEN', gen), mock.patch.object(entry, 'CFG', cfg), \
                 mock.patch.object(entry, 'OUT', root), mock.patch.object(entry, 'NAMES', NAMES):
                record = entry._generate_job(job)
            self.assertEqual(calls[0]['items'], [NAMES[2]])
            self.assertEqual(record['state_id'], 'one')
            self.assertTrue((root/'labels/train/train_000000.txt').read_text().startswith('2 '))
            self.assertTrue((root/'preview/train_000000.jpg').exists())

    def test_entry_passes_plan_and_short_circuits_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            toolkit = root/'toolkit'
            (toolkit/'configs').mkdir(parents=True)
            (toolkit/'SourcePic').mkdir()
            (toolkit/'configs/gen_default.yaml').write_text('paths:\n  icons: SourcePic\n', encoding='utf8')
            (toolkit/'configs/classes.yaml').write_text(yaml.safe_dump({'items': [{'name': n} for n in NAMES]}), encoding='utf8')
            for name in NAMES:
                (toolkit/'SourcePic'/(name+'.png')).write_bytes(b'fixture')
            declaration = plan()
            request = root/'plan.json'
            request.write_text(json.dumps(declaration), encoding='utf8')
            output = root/'datasets/exports/waste_sticker_fake'
            captured = []
            @contextlib.contextmanager
            def guard(*args, **kwargs):
                captured.append(kwargs)
                yield SimpleNamespace(reused=True)
            argv = ['stickers', '--toolkit', str(toolkit), '--plan', str(request), '--out', str(output)]
            with mock.patch.object(entry, 'WORKSPACE', root), mock.patch.object(entry, 'generation_guard', guard), \
                 mock.patch.object(entry, '_generate') as generate, mock.patch.object(sys, 'argv', argv):
                entry.main()
            generate.assert_not_called()
            self.assertEqual(captured[0]['production_plan'], declaration)
            self.assertEqual(captured[0]['sample_output_paths'], [output/'images'])
            self.assertEqual(captured[0]['auxiliary_output_paths'], [output/'preview'])
            self.assertFalse(output.exists())

    def test_incorrect_generated_class_fails_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = entry.planned_jobs(plan([state([2], 'single', 1, 1)]), NAMES, 9)[0]
            generate = mock.Mock(return_value={'n': 1, 'det': ['0 .5 .5 .3 .3'],
                                              'inst': [{'name': NAMES[3]}]})
            gen = SimpleNamespace(gen_one=generate, imwrite_u=mock.Mock())
            cfg = {'scene': {'layout': {}, 'n_panels': {}}}
            with mock.patch.object(entry, 'GEN', gen), mock.patch.object(entry, 'CFG', cfg), \
                 mock.patch.object(entry, 'OUT', root), mock.patch.object(entry, 'NAMES', NAMES):
                with self.assertRaisesRegex(RuntimeError, 'could not render every planned class'):
                    entry._generate_job(job)
            self.assertEqual(generate.call_count, 20)
            gen.imwrite_u.assert_not_called()

    def test_parent_records_all_worker_results_against_planned_states(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            toolkit, out = root/'toolkit', root/'out'
            current = plan([state([2], 'single', 1, 2)])
            run = SimpleNamespace(plan=current, record_sample=mock.Mock())
            args = SimpleNamespace(width=640, height=480, vehicle_scale=False, seed=8, workers=2)
            def initialize(*unused):
                entry.NAMES = NAMES
            class Pool:
                def __enter__(self): return self
                def __exit__(self, *unused): return False
                def imap_unordered(self, worker, jobs, chunksize):
                    for job in reversed(jobs):
                        yield {'image': f"images/train/{job['idx']}.jpg", 'state_id': job['state_id'],
                               'seed': job['seed'], 'retries': 0, 'instances': [], 'distractor': None,
                               'reported_panel': 'blue'}
            ctx = SimpleNamespace(Pool=lambda *args, **kwargs: Pool())
            with mock.patch.object(entry, 'WORKSPACE', root), mock.patch.object(entry, 'init_worker', side_effect=initialize), \
                 mock.patch.object(entry.mp, 'get_context', return_value=ctx):
                entry._generate(args, toolkit, out/'.asset_cache', out, run, NAMES)
            self.assertEqual(run.record_sample.call_count, 2)
            calls = run.record_sample.call_args_list
            self.assertTrue(all(call.args[0] == 'one' for call in calls))
            self.assertTrue(all(call.kwargs['sources'] == current['strata'][0]['sources'] for call in calls))
            self.assertNotEqual(calls[0].kwargs['actual']['seed'], calls[1].kwargs['actual']['seed'])


if __name__ == '__main__':
    unittest.main()
