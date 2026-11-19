"""Manifest ownership checks for person sample replacement."""
import json
from pathlib import Path
import tempfile
import unittest

from datasets.scripts.person.sample_replace import owned_files


class PersonSampleOwnershipTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.output = self.root / 'datasets/samples/person'
        (self.output / 'images/synthetic').mkdir(parents=True)
        (self.output / 'labels/synthetic').mkdir(parents=True)
        (self.output / 'images/synthetic/person_000001.png').write_bytes(b'image')
        (self.output / 'labels/synthetic/person_000001.txt').write_text('label', encoding='utf8')
        self.record = {
            'image': 'datasets/samples/person/images/synthetic/person_000001.png',
            'label': 'datasets/samples/person/labels/synthetic/person_000001.txt',
        }
        self.write_manifest()
        self.inventory = [{'path': p.relative_to(self.root).as_posix(), 'type': 'file',
                           'bytes': p.stat().st_size, 'modified_ns': p.stat().st_mtime_ns}
                          for p in self.output.rglob('*') if p.is_file()]
        self.previous = {'outputs': self.inventory}

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self):
        (self.output / 'manifest.json').write_text(json.dumps({
            'status': 'complete_task_yolo', 'records': [self.record]}), encoding='utf8')

    def test_only_manifest_owned_files_are_selected(self):
        (self.output / 'images/synthetic/manual.png').write_bytes(b'manual')
        self.assertEqual(owned_files(self.output, self.previous, self.inventory, self.root), {
            'manifest.json', 'images/synthetic/person_000001.png', 'labels/synthetic/person_000001.txt'})

    def test_manifest_without_receipt_is_not_owned(self):
        with self.assertRaisesRegex(ValueError, 'no matching'):
            owned_files(self.output, None, self.inventory, self.root)

    def test_rejects_paths_outside_task_output(self):
        self.record['image'] = 'datasets/samples/person/images/../../manual.png'
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'canonical'):
            owned_files(self.output, self.previous, self.inventory, self.root)

    def test_rejects_files_changed_since_receipt(self):
        changed = [dict(row) for row in self.inventory]
        changed[0]['bytes'] += 1
        with self.assertRaisesRegex(ValueError, 'does not match'):
            owned_files(self.output, self.previous, changed, self.root)


if __name__ == '__main__':
    unittest.main()
