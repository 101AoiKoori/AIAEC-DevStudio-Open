import json
from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import package_candidate_data as pack


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.spec = {'model_key': 'meter', 'dataset': {'names': ['meter'], 'kpt_shape': [8, 3]},
                     'batches': []}
        for name, method in [('real', 'camera'), ('render', 'render')]:
            base = self.root/'datasets'/name
            (base/'images').mkdir(parents=True)
            (base/'labels').mkdir()
            (base/'images/a.png').write_bytes(b'image-' + name.encode())
            (base/'labels/a.txt').write_text('0 0.5 0.5 0.2 0.2\n', encoding='utf8')
            (base/'manifest.json').write_text('{}', encoding='utf8')
            self.spec['batches'].append({'id': name, 'images': f'datasets/{name}/images',
                                         'labels': f'datasets/{name}/labels',
                                         'evidence': [f'datasets/{name}/manifest.json'],
                                         'source_kind': name, 'generation_method': method,
                                         'label_status': 'candidate', 'dataset': self.spec['dataset'].copy()})
        self.write_spec()
        self.guard_args = []

    def write_spec(self):
        (self.root/'spec.json').write_text(json.dumps(self.spec), encoding='utf8')

    @contextmanager
    def guard(self, *args, **kwargs):
        self.guard_args.append((args, kwargs))
        yield type('Run', (), {'reused': False, 'source_refs': {'fixture': 'declared'}})()

    def test_mixed_batches_keep_evidence_labels_pose_and_guard_inputs(self):
        with patch.object(pack, 'generation_guard', self.guard):
            result = pack.package('spec.json', 'datasets/exports/test.zip', self.root)
        self.assertEqual(result['images'], 2)
        self.assertEqual(result['nonempty_label_rows'], 2)
        args, kwargs = self.guard_args[0]
        self.assertEqual(len([x for x in args[1] if x.suffix == '.png']), 2)
        self.assertEqual(len([x for x in kwargs['procedural_inputs'] if x.suffix == '.txt']), 2)
        with zipfile.ZipFile(self.root/'datasets/exports/test.zip') as z:
            self.assertEqual(z.read('labels/train/real__00000000__a.txt'),
                             (self.root/'datasets/real/labels/a.txt').read_bytes())
            self.assertIn('images/train/render__00000000__a.png', z.namelist())
            self.assertIn('evidence/datasets/real/manifest.json', z.namelist())
            self.assertEqual(json.loads(z.read('data.yaml'))['kpt_shape'], [8, 3])
            self.assertEqual(json.loads(z.read('package_spec.json'))['batches'][0]['source_kind'], 'real')
            self.assertEqual(json.loads(z.read('package_manifest.json'))['source_refs'], {'fixture': 'declared'})

    def test_missing_label_or_evidence_fails_without_output(self):
        (self.root/'datasets/real/labels/a.txt').unlink()
        with self.assertRaisesRegex(ValueError, 'missing label'):
            pack.prepare(self.root, 'spec.json')
        (self.root/'datasets/real/labels/a.txt').write_text('', encoding='utf8')
        (self.root/'datasets/real/manifest.json').unlink()
        with self.assertRaisesRegex(ValueError, 'missing file'):
            pack.prepare(self.root, 'spec.json')

    def test_rejects_paths_outside_workspace_and_incompatible_contract(self):
        self.spec['batches'][0]['images'] = '../outside'
        self.write_spec()
        with self.assertRaisesRegex(ValueError, 'escapes workspace'):
            pack.prepare(self.root, 'spec.json')
        self.spec['batches'][0]['images'] = 'datasets/real/images'
        self.spec['batches'][1]['dataset'] = {'names': ['other'], 'kpt_shape': [8, 3]}
        self.write_spec()
        with self.assertRaisesRegex(ValueError, 'contract differs'):
            pack.prepare(self.root, 'spec.json')

    def test_reuse_and_empty_labels_are_preserved(self):
        self.spec['batches'][1]['images'] = 'datasets/real/images'
        (self.root/'datasets/render/labels/a.txt').write_text('', encoding='utf8')
        self.write_spec()
        with patch.object(pack, 'generation_guard', self.guard):
            pack.package('spec.json', 'datasets/exports/test.zip', self.root)
        with zipfile.ZipFile(self.root/'datasets/exports/test.zip') as z:
            self.assertEqual(z.read('labels/train/render__00000000__a.txt'), b'')
            manifest = json.loads(z.read('package_manifest.json'))
            self.assertEqual(manifest['reused_source_images']['datasets/real/images/a.png'], ['real', 'render'])

    def test_explicit_records_allow_same_stem_and_keep_exact_labels(self):
        batch = self.spec['batches'][0]
        batch.pop('images')
        batch.pop('labels')
        batch['records'] = [{'image': 'datasets/real/images/a.png', 'label': 'datasets/real/labels/a.txt'},
                            {'image': 'datasets/render/images/a.png', 'label': 'datasets/render/labels/a.txt',
                             'lineage': 'second source'}]
        self.write_spec()
        with patch.object(pack, 'generation_guard', self.guard):
            pack.package('spec.json', 'datasets/exports/test.zip', self.root)
        with zipfile.ZipFile(self.root/'datasets/exports/test.zip') as z:
            self.assertIn('images/train/real__00000000__a.png', z.namelist())
            self.assertIn('images/train/real__00000001__a.png', z.namelist())
            self.assertEqual(json.loads(z.read('package_spec.json'))['batches'][0]['records'][1]['lineage'], 'second source')

    def test_existing_output_not_replaced(self):
        path = self.root/'datasets/exports/test.zip'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'existing')
        with patch.object(pack, 'generation_guard', self.guard):
            with self.assertRaisesRegex(ValueError, 'already exists'):
                pack.package('spec.json', 'datasets/exports/test.zip', self.root)
        self.assertEqual(path.read_bytes(), b'existing')


if __name__ == '__main__':
    unittest.main()
