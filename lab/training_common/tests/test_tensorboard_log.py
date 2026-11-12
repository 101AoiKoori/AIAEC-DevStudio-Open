import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

COMMON = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMMON))
import tensorboard_log  # noqa: E402


class TensorBoardPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'config').mkdir()

    def configure(self, config):
        (self.root / 'config' / 'train.yaml').write_text(
            yaml.safe_dump(config), encoding='utf-8')

    def test_default_and_custom_project_match_training(self):
        self.configure({})
        self.assertEqual(tensorboard_log.log_directory(self.root), self.root / 'runs')
        self.configure({'project': 'experiments/session'})
        self.assertEqual(tensorboard_log.log_directory(self.root), self.root / 'experiments/session')

    def test_project_cannot_escape_package(self):
        self.configure({'project': '../outside'})
        with self.assertRaisesRegex(ValueError, 'escapes package'):
            tensorboard_log.log_directory(self.root)
        with contextlib.redirect_stderr(io.StringIO()) as error:
            self.assertEqual(tensorboard_log.main([], self.root), 1)
        self.assertIn('escapes package', error.getvalue())

    def test_main_passes_custom_project_to_tensorboard(self):
        self.configure({'project': 'experiments/custom'})
        calls = []

        class FakeBoard:
            def configure(self, argv):
                calls.append(argv)

            def launch(self):
                return 'http://127.0.0.1:6006/'

        fake_module = types.ModuleType('tensorboard')
        fake_module.program = types.SimpleNamespace(TensorBoard=FakeBoard)
        with patch.dict(sys.modules, {'tensorboard': fake_module}), \
             patch('threading.Event.wait', side_effect=KeyboardInterrupt), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(tensorboard_log.main([], self.root), 0)
        self.assertEqual(calls[0][2], str(self.root / 'experiments/custom'))
        self.assertTrue((self.root / 'experiments/custom').is_dir())


if __name__ == '__main__':
    unittest.main()
