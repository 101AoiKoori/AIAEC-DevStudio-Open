"""Check four entries through the copied package layout, without training."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[5]
CASES = (('fire', 'building_name_paper'), ('sign', 'sign'),
         ('waste_bin', 'waste_bin_state'), ('waste_bin', 'waste_sticker'))


class PackagePathTests(unittest.TestCase):
    def test_copied_entries_load_package_config(self):
        with tempfile.TemporaryDirectory(prefix='remaining v4 package ', dir=ROOT / 'tmp') as temp:
            package = Path(temp)
            (package / 'scripts/task').mkdir(parents=True)
            (package / 'config').mkdir()
            shutil.copytree(ROOT / 'lab/training_common/engine', package / 'scripts/common/engine',
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            shutil.copy2(ROOT / 'lab/training_common/train.py', package / 'scripts/common/train.py')
            for owner, model in CASES:
                with self.subTest(model=model):
                    source = ROOT / 'lab/features' / owner / 'training/scripts' / ('train_' + model + '_v4.py')
                    shutil.copy2(source, package / 'scripts/task' / source.name)
                    launcher = package / 'scripts/train.py'
                    launcher.write_text("import runpy\nfrom pathlib import Path\nroot=Path(__file__).resolve().parents[1]\n"
                                        "runpy.run_path(str(root / 'scripts/task' / " + repr(source.name) + "), run_name='__main__')\n", encoding='utf-8')
                    config = yaml.safe_load((ROOT / 'lab/features' / owner / 'training/config' / (model + '_v4.yaml')).read_text(encoding='utf-8'))
                    config.update(model='weights/missing-' + model + '.pt', data='dataset/data.yaml', output_root='runs/' + model)
                    (package / 'config/train.yaml').write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
                    result = subprocess.run([sys.executable, '-I', str(launcher), 'check'], cwd=ROOT,
                                            capture_output=True, text=True, encoding='utf-8', errors='replace')
                    self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                    self.assertIn('missing-' + model + '.pt', result.stderr)
                    self.assertIn(str(package), result.stderr)


if __name__ == '__main__':
    unittest.main()
