"""Exercise the flat sticker adapter without producing a dataset or using SQLite."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SCRIPT = ROOT / 'lab/features/waste_bin/training/scripts/generate_flat_controlled_v4.py'
spec = importlib.util.spec_from_file_location('flat_guard_test_target', SCRIPT)
flat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flat)

NAMES = [f'item_{i}' for i in range(24)]


def state(state_id, cls, count):
    params = {'class_id': cls, 'bin_color_index': 2, 'gain': .78, 'saturation': .72}
    name = NAMES[cls] if cls is not None else None
    return {'id': state_id, 'planned_count': count, 'parameters': params,
            'background': {'kind': 'synthetic_flat_panel', 'color_index': 2},
            'target': {'kind': 'sticker', 'class_id': cls, 'name': name} if name else {'kind': 'negative'},
            'labels': {'format': 'yolo_detection', 'class_id': cls},
            'augmentation': {'gain': .78, 'saturation': .72, 'gamma': 1.0, 'vignette': 0.0,
                             'color_temp_std': 0.0, 'blur_sigma': 0.0, 'noise_sigma': 0.0,
                             'variation': 'seeded_toolkit_geometry_and_texture'},
            'sources': [f'{flat.TOOLKIT}/SourcePic/{name}.png'] if name else
                       [f'{flat.TOOLKIT}/configs/gen_default.yaml']}


def plan():
    return {'model_key': 'waste_sticker', 'target_split': 'train',
            'constraints': dict(flat.CONSTRAINTS), 'strata': [state('positive', 3, 2), state('negative', None, 1)]}


class FakeGenerator:
    def __init__(self):
        self.jobs = []

    def gen_one(self, job):
        self.jobs.append(dict(job))
        positive = not job['negative']
        item = {'name': job['items'][0], 'xyxy': [10, 10, 150, 150],
                'quad': [[10, 10], [150, 10], [150, 150], [10, 150]]} if positive else None
        return {'img': np.zeros((480, 640, 3), dtype=np.uint8), 'inst': [item] if item else [],
                'det': ['3 .1 .1 .2 .2'] if item else [], 'distractor': None}


class FlatPlanTests(unittest.TestCase):
    def test_rejects_mismatched_state_and_unsupported_constraints(self):
        for mutate in (lambda p: p['strata'][0]['target'].update(name='wrong'),
                       lambda p: p['strata'][0]['augmentation'].update(blur_sigma=1),
                       lambda p: p['strata'][0]['parameters'].update(yaw=20),
                       lambda p: p['constraints'].update(other=True)):
            candidate = plan()
            mutate(candidate)
            with self.assertRaises(ValueError):
                flat.validate_plan(candidate, NAMES)

    def test_planned_counts_and_states_drive_jobs_and_record_actual(self):
        generator = FakeGenerator()
        config = {'scene': {'layout': {'mode': '', 'sticker': {}}, 'panel': {}, 'sticker': {}},
                  'photometric': {'color_temp': {}}}
        run = SimpleNamespace(plan=plan(), record_sample=mock.Mock())
        args = SimpleNamespace(seed=17)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / 'batch'
            def init(*unused):
                flat.existing.CFG, flat.existing.GEN, flat.existing.NAMES = config, generator, NAMES
            with mock.patch.object(flat, 'ROOT', root), mock.patch.object(flat.existing, 'init_worker', side_effect=init):
                flat._generate(args, out, run)
            report = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual([j['items'] for j in generator.jobs], [[NAMES[3]], [NAMES[3]], []])
            self.assertEqual(len({j['seed'] for j in generator.jobs}), 3)
            self.assertTrue(all(j['panel_color'] == 2 for j in generator.jobs))
            self.assertEqual([r['state_id'] for r in report['records']], ['positive', 'positive', 'negative'])
            self.assertEqual(run.record_sample.call_count, 3)
            self.assertEqual(run.record_sample.call_args_list[0].kwargs['actual']['paper_bbox_xyxy'], [10, 10, 150, 150])
            self.assertEqual((out / 'labels/train/train_000002.txt').read_text(encoding='utf-8'), '')

    def test_does_not_relabel_wrong_generated_class(self):
        class Wrong(FakeGenerator):
            def gen_one(self, job):
                result = super().gen_one(job)
                result['inst'][0]['name'] = NAMES[4]
                return result
        generator = Wrong()
        config = {'scene': {'layout': {'mode': '', 'sticker': {}}, 'panel': {}, 'sticker': {}},
                  'photometric': {'color_temp': {}}}
        run = SimpleNamespace(plan=copy.deepcopy(plan()), record_sample=mock.Mock())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def init(*unused):
                flat.existing.CFG, flat.existing.GEN, flat.existing.NAMES = config, generator, NAMES
            with mock.patch.object(flat, 'ROOT', root), mock.patch.object(flat.existing, 'init_worker', side_effect=init):
                with self.assertRaisesRegex(ValueError, 'differs from the planned class'):
                    flat._generate(SimpleNamespace(seed=17), root / 'batch', run)
            run.record_sample.assert_not_called()


if __name__ == '__main__':
    unittest.main()
