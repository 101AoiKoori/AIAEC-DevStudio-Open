import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import yaml

from datasets.scripts.common.design_context import design_context
from datasets.scripts.common.paths import TASKS, WORKSPACE


class DesignContextTest(unittest.TestCase):
    def test_all_tasks_reference_their_own_unverified_contrasts(self):
        for task in TASKS:
            with self.subTest(task=task):
                context = design_context(task)
                self.assertEqual(context['status'], 'referenced')
                self.assertEqual(context['task'], task)
                self.assertFalse(context['executable'])
                self.assertEqual(context['coverage'], 'unverified')
                self.assertTrue(context['crosses'])
                self.assertTrue(all(row['coverage'] == 'unverified' for row in context['crosses']))

    def test_fingerprint_changes_and_missing_inputs_do_not_claim_coverage(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'config').mkdir()
            for name in ('task_targets.yaml', 'combination_axes.yaml'):
                shutil.copyfile(WORKSPACE / 'config' / name, root / 'config' / name)
            first = design_context('sign', root)
            path = root / 'config/combination_axes.yaml'
            with path.open('a', encoding='utf8') as stream:
                stream.write('\n# Changed design snapshot\n')
            second = design_context('sign', root)
            self.assertNotEqual(first['sources']['axes']['sha256'], second['sources']['axes']['sha256'])
            axes = yaml.safe_load(path.read_text(encoding='utf8'))
            axes['tasks']['sign'].pop('crosses')
            path.write_text(yaml.safe_dump(axes, allow_unicode=True), encoding='utf8')
            self.assertEqual(design_context('sign', root)['status'], 'incomplete')
            path.unlink()
            missing = design_context('sign', root)
            self.assertEqual(missing['status'], 'unavailable')
            self.assertEqual(missing['coverage'], 'unverified')
            self.assertEqual(missing['crosses'], [])
            path.write_text('tasks: [', encoding='utf8')
            malformed = design_context('sign', root)
            self.assertEqual(malformed['status'], 'unavailable')
            self.assertEqual(malformed['coverage'], 'unverified')
            self.assertEqual(malformed['crosses'], [])


if __name__ == '__main__':
    unittest.main()
