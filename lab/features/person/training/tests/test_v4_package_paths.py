"""Exercise task entries copied into the packager's actual directory layout."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[5]
CASES = (
    ('e_bike', 'e_bike'), ('fire', 'fire_sticker'),
    ('meter', 'meter'), ('plate', 'plate_detection'),
)


class PackagePathTests(unittest.TestCase):
    def test_copied_task_entries_use_package_config_and_engine(self):
        parent = ROOT / 'tmp'
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='v4 package with spaces ', dir=parent) as temp:
            package = Path(temp)
            (package / 'scripts/task').mkdir(parents=True)
            (package / 'config').mkdir()
            shutil.copytree(ROOT / 'lab/training_common/engine', package / 'scripts/common/engine',
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            shutil.copy2(ROOT / 'lab/training_common/train.py', package / 'scripts/common/train.py')
            launcher = package / 'scripts/train.py'
            for owner, model in CASES:
                with self.subTest(model=model):
                    source = ROOT / 'lab/features' / owner / 'training/scripts' / ('train_' + model + '_v4.py')
                    shutil.copy2(source, package / 'scripts/task' / source.name)
                    launcher.write_text("import runpy\nfrom pathlib import Path\n"
                                        "root = Path(__file__).resolve().parents[1]\n"
                                        "runpy.run_path(str(root / 'scripts/task' / " + repr(source.name) +
                                        "), run_name='__main__')\n", encoding='utf-8')
                    config_path = ROOT / 'lab/features' / owner / 'training/config' / (model + '_v4.yaml')
                    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
                    config['model'] = 'weights/missing-' + model + '.pt'
                    config['data'] = 'dataset/data.yaml'
                    config['output_root'] = 'runs/' + model
                    (package / 'config/train.yaml').write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
                    result = subprocess.run([sys.executable, '-I', str(launcher), 'check'], cwd=ROOT,
                                            capture_output=True, text=True, encoding='utf-8', errors='replace')
                    self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                    self.assertIn('missing-' + model + '.pt', result.stderr)
                    self.assertIn(str(package), result.stderr)
                    self.assertNotIn('config/train.yaml\'', result.stderr)


if __name__ == '__main__':
    unittest.main()
