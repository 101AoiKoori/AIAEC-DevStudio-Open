"""Exercise publication while a Windows reader keeps the destination root open."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datasets.scripts.common import recoverable_publish as publisher


class ContentPublicationTests(unittest.TestCase):
    def setUp(self):
        workspace_tmp = Path(__file__).resolve().parents[4] / 'tmp'
        self.directory = tempfile.TemporaryDirectory(dir=workspace_tmp)
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / 'sample'
        self.output.mkdir()
        (self.output / 'manual.txt').write_text('user note', encoding='utf8')
        (self.output / 'old.txt').write_text('old generated image', encoding='utf8')
        self.before = publisher.signature(self.output)
        self.token = 'a' * 32
        self.stage = publisher.prepare(self.output, self.token)
        publisher.copy_unowned(self.output, self.stage, {'old.txt'})
        (self.stage / 'new.txt').write_text('new generated image', encoding='utf8')

    def test_committed_contents_keep_the_root_and_manual_file(self):
        with patch.object(Path, 'rename', side_effect=PermissionError('root is open')):
            publisher.publish(self.output, self.stage, self.token, keep_root=True)
            publisher.recover(self.output, lambda token: token == self.token)
        self.assertEqual((self.output / 'manual.txt').read_text(encoding='utf8'), 'user note')
        self.assertEqual((self.output / 'new.txt').read_text(encoding='utf8'), 'new generated image')
        self.assertFalse((self.output / 'old.txt').exists())
        self.assertFalse(publisher._metadata(self.output)[1].exists())

    def interrupt(self):
        def incomplete_copy(source, destination):
            (destination / 'new.txt').write_text('partial copy', encoding='utf8')
            raise OSError('interrupted copy')

        with patch.object(publisher, '_sync_contents', side_effect=incomplete_copy):
            with self.assertRaises(OSError):
                publisher.publish(self.output, self.stage, self.token, keep_root=True)

    def test_interrupted_file_copy_restores_the_complete_prior_sample(self):
        self.interrupt()
        publisher.recover(self.output, lambda _: False)
        self.assertEqual(publisher.signature(self.output), self.before)
        self.assertFalse(publisher._metadata(self.output)[1].exists())

    def test_recovery_preserves_a_user_edit_for_resolution(self):
        self.interrupt()
        (self.output / 'manual.txt').write_text('new user note', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'unowned sample content changed'):
            publisher.recover(self.output, lambda _: False)
        self.assertEqual((self.output / 'manual.txt').read_text(encoding='utf8'), 'new user note')
        self.assertTrue(publisher._metadata(self.output)[1].exists())
