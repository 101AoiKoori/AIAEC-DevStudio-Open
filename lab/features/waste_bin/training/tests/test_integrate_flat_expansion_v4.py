"""Check manifest mapping and production guard on a temporary SQLite cache."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT / 'lab/scripts/media/tests')]
import test_media_index_sqlite as fixtures
from lab.training_common import production_guard

path = ROOT / 'lab/features/waste_bin/training/scripts/integrate_flat_expansion_v4.py'
spec = importlib.util.spec_from_file_location('flat_expansion_select', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class SelectionTest(unittest.TestCase):
    def test_source_reference_and_selection(self):
        image = mod.BATCH + '/images/train/c04_00.png'
        row = {'image': image, 'label': mod.BATCH + '/labels/train/c04_00.txt', 'class_id': 4,
               'source_name': '纸箱', 'source_artwork': 'datasets/raw/waste_bin/toolkit/SourcePic/纸箱.png',
               'normal_or_dark': 'dark'}
        reference = {'manifest': mod.MANIFEST, 'record_index': 0, 'relation': 'describes'}
        source = {'samples': []}
        config = {'data_overrides': {'train': []}, 'source_notes': []}
        distribution = {'additional_records': []}
        selected, updated, manifest = mod.integrate(source, config, distribution, [row], {image: [reference]})
        selected_row = selected['samples'][0]
        self.assertEqual(selected_row['indexed_source_references'], [reference])
        self.assertNotIn('generation_record', selected_row)
        self.assertEqual(selected_row['tasks']['waste_sticker']['state'], 'complete_positive')
        self.assertIn('low_saturation', selected_row['tasks']['waste_sticker']['strata']['conditions'])
        self.assertFalse(selected_row['selection_evidence']['per_image_human_review'])
        self.assertEqual(source['samples'], [])
        self.assertEqual(updated['data_overrides']['train'][0]['images'], mod.BATCH + '/images/train')
        self.assertEqual(manifest['additional_records'], [mod.MANIFEST])
        with self.assertRaisesRegex(ValueError, 'indexed original record missing'):
            mod.candidate_rows([row], {image: []})

    def test_preexisting_selection_skips_even_when_guard_does_not_reuse(self):
        from contextlib import contextmanager
        from types import SimpleNamespace
        fix = fixtures.CacheTests('runTest')
        fix.setUp()
        self.addCleanup(fix.doCleanups)
        root = fix.root
        source, config, distribution, manifest = (root / 'source.json', root / 'config.yaml',
                                                   root / 'distribution.json', root / 'batch.json')
        image = fix.image.relative_to(root).as_posix()
        record = {'image': image, 'label': 'label.txt', 'class_id': 4, 'source_name': '纸箱',
                  'source_artwork': 'art.png', 'normal_or_dark': 'dark'}
        reference = {'manifest': 'batch.json', 'record_index': 0, 'relation': 'describes'}
        selected_row = {'id': 'flat-expansion:' + image, 'image': image, 'split': 'train',
                        'selected_tasks': ['waste_sticker'], 'indexed_source_references': [reference],
                        'tasks': {'waste_sticker': {'state': 'complete_positive', 'label': 'label.txt'}}}
        source.write_text(json.dumps({'samples': [selected_row]}), encoding='utf8')
        config.write_text('data_overrides:\n  train:\n  - images: datasets/exports/waste_bin/waste_sticker_v4_flat_expansion_20261113/images/train\n    labels: datasets/exports/waste_bin/waste_sticker_v4_flat_expansion_20261113/labels/train\n', encoding='utf8')
        distribution.write_text(json.dumps({'source_manifest': 'source.json', 'additional_records': ['batch.json']}), encoding='utf8')
        manifest.write_text(json.dumps({'records': [record]}), encoding='utf8')
        (root / 'label.txt').write_text('', encoding='utf8')
        script = root / 'script.py'; script.write_text('# fixture', encoding='utf8')
        gen = root / 'gen.py'; gen.write_text('# fixture', encoding='utf8')
        gate_file = root / 'lab/scripts/data/v4_source_gate.py'; gate_file.parent.mkdir(exist_ok=True)
        gate_file.write_text('# fixture', encoding='utf8')
        @contextmanager
        def changed_guard(*args, **kwargs):
            yield SimpleNamespace(reused=False)
        patches = [mock.patch.object(mod, 'ROOT', root), mock.patch.object(mod, 'SOURCE', 'source.json'),
                   mock.patch.object(mod, 'CONFIG', 'config.yaml'), mock.patch.object(mod, 'DISTRIBUTION', 'distribution.json'),
                   mock.patch.object(mod, 'MANIFEST', 'batch.json'), mock.patch.object(mod, 'GENERATOR', 'gen.py'),
                   mock.patch.object(mod, '__file__', str(script)),
                   mock.patch.object(mod.gate, '_selection_files', return_value={fix.image}),
                   mock.patch.object(mod, 'generation_guard', changed_guard),
                   mock.patch.object(mod, 'validate_batch', side_effect=AssertionError('must not regenerate'))]
        for patch in patches:
            patch.start(); self.addCleanup(patch.stop)
        paths = (source, config, distribution, manifest)
        before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths]
        mod.main()
        self.assertEqual(before, [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths])

    def test_guard_refuses_missing_db_and_reuses_complete_result(self):
        fix = fixtures.CacheTests('runTest')
        fix.setUp()
        self.addCleanup(fix.doCleanups)
        root = fix.root
        shard = root / fix.parts['datasets/raw/camera']['shard']
        item = json.loads(shard.read_text(encoding='utf8'))
        item['modified_ns'] = fix.image.stat().st_mtime_ns
        shard.write_text(json.dumps(item, ensure_ascii=False) + '\n', encoding='utf8')
        fix.write_catalog()
        source, config, distribution, manifest = (root / 'source.json', root / 'config.yaml',
                                                   root / 'distribution.json', root / 'batch.json')
        label = root / 'label.txt'
        label.write_text('', encoding='utf8')
        source.write_text(json.dumps({'samples': []}), encoding='utf8')
        config.write_text('data_overrides:\n  train: []\nsource_notes: []\n', encoding='utf8')
        distribution.write_text(json.dumps({'additional_records': []}), encoding='utf8')
        record = {'image': fix.image.relative_to(root).as_posix(), 'label': 'label.txt'}
        manifest.write_text(json.dumps({'records': [record]}), encoding='utf8')
        script = root / 'script.py'; script.write_text('# fixture', encoding='utf8')
        gen = root / 'gen.py'; gen.write_text('# fixture', encoding='utf8')
        gate_file = root / 'lab/scripts/data/v4_source_gate.py'; gate_file.parent.mkdir(exist_ok=True)
        gate_file.write_text('# fixture', encoding='utf8')
        patches = [mock.patch.object(mod, 'ROOT', root), mock.patch.object(mod, 'SOURCE', 'source.json'),
                   mock.patch.object(mod, 'CONFIG', 'config.yaml'), mock.patch.object(mod, 'DISTRIBUTION', 'distribution.json'),
                   mock.patch.object(mod, 'MANIFEST', 'batch.json'), mock.patch.object(mod, 'GENERATOR', 'gen.py'),
                   mock.patch.object(mod, '__file__', str(script)),
                   mock.patch.object(mod.gate, '_selection_files', return_value=set()),
                   mock.patch.object(mod.gate, 'audit', return_value={'status': 'passed'}),
                   mock.patch.object(mod, 'validate_batch'),
                   mock.patch.object(mod, 'candidate_rows', return_value=[{'id': 'fixture'}]),
                   mock.patch.object(production_guard, '_index', return_value=fixtures.cache)]
        for patch in patches:
            patch.start(); self.addCleanup(patch.stop)
        before = source.read_text(encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'SQLite cache missing'):
            mod.main()
        self.assertEqual(source.read_text(encoding='utf8'), before)
        self.assertEqual(fix.call(['sync'])[0], 0)
        mod.main()
        stamp = source.stat().st_mtime_ns
        mod.main()
        self.assertEqual(source.stat().st_mtime_ns, stamp)
        self.assertEqual(len(json.loads(source.read_text(encoding='utf8'))['samples']), 1)


if __name__ == '__main__':
    unittest.main()
