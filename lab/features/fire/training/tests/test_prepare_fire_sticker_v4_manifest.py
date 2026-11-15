"""Bounded tests for the guarded fire delivery-manifest adapter."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'lab/features/fire/training/scripts'))
import prepare_fire_sticker_v4_manifest as module


class FireDeliveryManifestTests(unittest.TestCase):
    def setUp(self):
        (ROOT / 'tmp').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old_root = patch.object(module, 'ROOT', self.root)
        self.old_root.start()
        self.addCleanup(self.old_root.stop)

    def file(self, relative, text='fixture'):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf8')
        return path

    def test_real_parents_and_exposure_are_preserved(self):
        for path in ('render/original.png', 'bg/frame.png', 'foreground/render.png'):
            self.file(path)
        arena = {'source_batch': 'render', 'source_image': 'original.png', 'image': 'selected/a.png',
                 'source_id': 'arena', 'source_kind': 'synthetic_scene', 'generation_method': 'synthetic_render',
                 'source_scene': 'scene/arena.blend', 'source_buildings': 'scene/buildings.blend',
                 'label_status': 'reviewed', 'review_state': 'complete_positive', 'label': 'labels/a.txt',
                 'source_review_evidence': 'proof/a.json', 'review_evidence': 'proof/a.json'}
        distractor = {'source_manifest': 'batch.json', 'source_image': 'render/original.png',
                      'image': 'selected/b.png', 'background_parent': 'bg/frame.png',
                      'source_foreground': 'foreground/render.png', 'source_id': 'site',
                      'source_kind': 'derived_image', 'generation_method': 'composite',
                      'label_status': 'reviewed', 'review_state': 'complete_negative', 'label': 'labels/b.txt',
                      'source_review_evidence': 'proof/b.json', 'review_evidence': 'proof/b.json',
                      'development_contact': 'development review; use does not prove independent session'}
        old = {'samples': [], 'near_duplicate_pairs': [], 'models': {'fire_sticker': {'data_yaml': 'data.yaml'}}}
        batches = {'batch.json': {'render/original.png': {'background': {}}}}
        refs = {'selected/a.png': [{'manifest': 'original.json', 'record_index': 1, 'relation': 'describes'}],
                'selected/b.png': [{'manifest': 'batch.json', 'record_index': 2, 'relation': 'describes'}]}
        result = module._candidate(old, [arena], [distractor], batches, {}, refs)
        self.assertEqual(old['samples'], [])
        rows = [row for row in result['samples'] if row['split'] == 'train']
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['parent_ids'][0], rows[1]['parent_ids'][0])
        self.assertEqual(len(rows[1]['parent_ids']), 3)
        self.assertEqual(rows[1]['development_contact'], distractor['development_contact'])
        self.assertFalse(rows[1]['tasks']['fire_sticker']['human_accepted'])
        self.assertEqual(rows[1]['indexed_source_references'], refs['selected/b.png'])

    def test_delivery_config_preserves_main_input(self):
        sources = [{'images': path, 'labels': path.replace('/images/', '/labels/')}
                   for path in module.TRAIN_DIRS]
        original = {'data_overrides': {'train': sources + [{'images': 'other', 'labels': 'other'}]}}
        result = module._delivery_config(original)
        self.assertEqual(len(result['data_overrides']['train']), 3)
        self.assertEqual(len(original['data_overrides']['train']), 4)

    def test_reuse_and_failed_gate_do_not_write(self):
        original = self.file('source.json', '{"samples": []}')
        descriptor = self.file('distribution.json', '{"scope": "original"}')
        config = self.file('config.yaml', 'data_overrides: {train: []}')
        delivery = self.root / 'delivery.yaml'
        before = (original.read_bytes(), descriptor.read_bytes())
        inputs = ({'samples': []}, [], [], {}, [], [], [])

        @contextmanager
        def reused(*args, **kwargs):
            yield SimpleNamespace(reused=True, source_refs={})

        @contextmanager
        def rejected(*args, **kwargs):
            raise ValueError('SQLite unavailable')
            yield

        @contextmanager
        def permitted(*args, **kwargs):
            yield SimpleNamespace(reused=False, source_refs={})

        patches = [patch.object(module, 'ORIGINAL', original), patch.object(module, 'DISTRIBUTION', descriptor),
                   patch.object(module, 'CONFIG', config), patch.object(module, 'DELIVERY', delivery),
                   patch.object(module, '_inputs', return_value=inputs),
                   patch.object(module, '_delivery_config', return_value={})]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        with patch.object(module, 'generation_guard', reused), patch.object(module, '_candidate') as body:
            module.main()
            body.assert_not_called()
        with patch.object(module, 'generation_guard', rejected), patch.object(module, '_candidate') as body:
            with self.assertRaisesRegex(ValueError, 'SQLite unavailable'):
                module.main()
            body.assert_not_called()
        with patch.object(module, 'generation_guard', permitted), \
             patch.object(module, '_candidate', return_value={'samples': []}), \
             patch.object(module.gate, 'site_sequence_times', return_value={}), \
             patch.object(module.gate, 'build_queue', return_value={}), \
             patch.object(module.gate, 'audit', return_value={'status': 'failed', 'errors': ['conflict']}):
            with self.assertRaisesRegex(ValueError, 'conflict'):
                module.main()
        self.assertEqual((original.read_bytes(), descriptor.read_bytes()), before)
        self.assertFalse(delivery.exists())

    def test_success_then_reuse_does_not_rewrite_outputs(self):
        original = self.file('source.json', '{"samples": []}')
        descriptor = self.file('distribution.json', '{"scope": "original"}')
        config = self.file('config.yaml', 'data_overrides: {train: []}')
        delivery = self.root / 'delivery.yaml'
        completed = []
        guard_calls = []

        @contextmanager
        def guard(operation, inputs, outputs, parameters, **kwargs):
            guard_calls.append((operation, outputs, kwargs['inplace_paths'], kwargs['adopt_outputs']))
            yield SimpleNamespace(reused=bool(completed), source_refs={})
            completed.append(True)

        inputs = ({'samples': []}, [], [], {}, [], [], [])
        patches = [patch.object(module, 'ORIGINAL', original), patch.object(module, 'DISTRIBUTION', descriptor),
                   patch.object(module, 'CONFIG', config), patch.object(module, 'DELIVERY', delivery),
                   patch.object(module, '_inputs', return_value=inputs),
                   patch.object(module, '_delivery_config', return_value={'task': 'detect'}),
                   patch.object(module, 'generation_guard', guard),
                   patch.object(module, '_candidate', return_value={'samples': []}),
                   patch.object(module.gate, 'site_sequence_times', return_value={}),
                   patch.object(module.gate, 'build_queue', return_value={}),
                   patch.object(module.gate, 'audit', return_value={'status': 'passed', 'assigned': 1110})]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        module.main()
        paths = (original, descriptor, delivery)
        before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]
        module.main()
        self.assertEqual(before, [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths])
        self.assertEqual(len(guard_calls), 2)
        self.assertEqual(guard_calls[0][2], [original, descriptor])
        self.assertEqual(guard_calls[0][3], [delivery])
        self.assertEqual(json.loads(original.read_text(encoding='utf8'))['delivery_machine_source_check']['assigned'], 1110)
        delivery.write_text('task: unexpected\n', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            module.main()
        self.assertEqual(len(guard_calls), 2)


if __name__ == '__main__':
    unittest.main()
