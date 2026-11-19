"""Test sample ownership and restart recovery on isolated temporary files."""
import json
from pathlib import Path
import tempfile
import unittest

from datasets.scripts.waste_bin_state import sample_publish as publish


class SamplePublicationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'datasets/samples/waste_bin_state'
        self.output.mkdir(parents=True)
        (self.output / 'images').mkdir()
        (self.output / 'labels').mkdir()
        (self.output / 'images/a.jpg').write_bytes(b'image')
        (self.output / 'labels/a.txt').write_text('label', encoding='utf-8')
        (self.output / '.gitkeep').touch()
        (self.output / 'notes.txt').write_text('human', encoding='utf-8')
        (self.output / 'manifest.json').write_text(json.dumps({
            'task': 'waste_bin_state', 'status': 'complete_representative_sample',
            'records': [{'image': 'images/a.jpg', 'label': 'labels/a.txt'}]}), encoding='utf-8')
        self.state = []
        for path in self.output.rglob('*'):
            stat = path.stat()
            self.state.append({'path': path.relative_to(self.root).as_posix(),
                               'type': 'directory' if path.is_dir() else 'file',
                               'bytes': stat.st_size, 'modified_ns': stat.st_mtime_ns})
        self.previous = {'outputs': self.state}
        self.token = 'a' * 32

    def make_stage(self):
        owned = publish.owned_files(self.output, self.previous, self.state, self.root)
        stage = publish.prepare(self.output, self.token)
        publish.copy_unowned(self.output, stage, owned)
        (stage / 'images/b.jpg').write_bytes(b'new')
        (stage / 'labels/b.txt').write_text('new', encoding='utf-8')
        (stage / 'manifest.json').write_text('{}', encoding='utf-8')
        return stage

    def test_manifest_verification_and_preserve_unowned(self):
        stage = self.make_stage()
        self.assertEqual((stage / 'notes.txt').read_text(), 'human')
        self.assertTrue((stage / '.gitkeep').exists())
        self.assertFalse((stage / 'images/a.jpg').exists())
        self.assertEqual((stage / 'images/b.jpg').read_bytes(), b'new')

    def test_manifest_cannot_claim_changed_or_external_files(self):
        (self.output / 'images/a.jpg').write_bytes(b'changed')
        stat = (self.output / 'images/a.jpg').stat()
        current = [dict(row, bytes=stat.st_size, modified_ns=stat.st_mtime_ns)
                   if row['path'].endswith('images/a.jpg') else row for row in self.state]
        with self.assertRaisesRegex(ValueError, 'previous receipt'):
            publish.owned_files(self.output, self.previous, current, self.root)
        with self.assertRaisesRegex(ValueError, 'outside'):
            publish._sample_path('images/../../user.jpg', 'images')

    def test_uncommitted_installed_sample_is_rolled_back(self):
        original = publish.signature(self.output)
        stage = self.make_stage()
        publish.publish(self.output, stage, self.token)
        self.assertTrue(publish.recover(self.output, lambda _token: False))
        self.assertEqual(publish.signature(self.output), original)
        self.assertFalse(publish.recover(self.output, lambda _token: False))

    def test_committed_installed_sample_keeps_new_files(self):
        stage = self.make_stage()
        new = publish.signature(stage)
        publish.publish(self.output, stage, self.token)
        self.assertTrue(publish.recover(self.output, lambda token: token == self.token))
        self.assertEqual(publish.signature(self.output), new)
        self.assertTrue((self.output / 'notes.txt').exists())
        self.assertEqual(list(self.output.parent.glob('.waste_bin_state.backup.*')), [])

    def test_changed_published_directory_is_never_removed(self):
        stage = self.make_stage()
        publish.publish(self.output, stage, self.token)
        (self.output / 'notes.txt').write_text('edited after interruption', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'changed'):
            publish.recover(self.output, lambda _token: False)
        self.assertEqual((self.output / 'notes.txt').read_text(), 'edited after interruption')
        self.assertTrue(list(self.output.parent.glob('.waste_bin_state.backup.*')))

    def test_interrupted_preparation_discards_stage_but_preserves_original(self):
        original = publish.signature(self.output)
        stage = self.make_stage()
        self.assertTrue(publish.recover(self.output, lambda _token: False))
        self.assertEqual(publish.signature(self.output), original)
        self.assertFalse(stage.exists())

    def test_interrupted_between_renames_restores_original(self):
        original = publish.signature(self.output)
        stage = self.make_stage()
        _, journal = publish._metadata(self.output)
        info = json.loads(journal.read_text(encoding='utf8'))
        info.update(new=publish.signature(stage), phase='prepared')
        publish._write_json(journal, info)
        backup = self.output.parent / info['backup']
        self.output.rename(backup)
        self.assertTrue(publish.recover(self.output, lambda _token: False))
        self.assertEqual(publish.signature(self.output), original)
        self.assertFalse(stage.exists())

    def test_missing_receipt_store_keeps_backup_and_journal(self):
        stage = self.make_stage()
        publish.publish(self.output, stage, self.token)
        def unavailable(_token):
            raise RuntimeError('receipt store unavailable')
        with self.assertRaisesRegex(RuntimeError, 'receipt store unavailable'):
            publish.recover(self.output, unavailable)
        self.assertTrue(list(self.output.parent.glob('.waste_bin_state.backup.*')))
        self.assertTrue(publish._metadata(self.output)[1].is_file())


if __name__ == '__main__':
    unittest.main()
