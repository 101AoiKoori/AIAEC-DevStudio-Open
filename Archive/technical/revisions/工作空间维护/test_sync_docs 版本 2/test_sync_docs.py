import json
import tempfile
import unittest
from pathlib import Path

import sync_docs


class RenderTest(unittest.TestCase):
    def test_requirements_and_module_status_are_shared(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('config', 'lab/features', 'docs', 'ws/src/competition_perception/src/competition_perception'):
                (root / name).mkdir(parents=True, exist_ok=True)
            for name in ('lab/features/README.md', 'docs/TASK_MATRIX.md'):
                (root / name).write_text('before\n<!-- TASKS:START -->\nold\n<!-- TASKS:END -->\nafter\n', encoding='utf-8')
            task = {'id': 'person', 'summary': '统一摘要', 'score': '12',
                    'matrix_name': '人物', 'matrix_check': '检查', 'requirements_markdown': '# 详细需求',
                    'scheduling': {'mode': '后台处理候选', 'approach': '保留图像'}}
            config = root / 'config/perception_tasks.json'
            config.write_text(json.dumps({'tasks': [task]}, ensure_ascii=False), encoding='utf-8')
            original_root, original_config = sync_docs.ROOT, sync_docs.CONFIG
            try:
                sync_docs.ROOT, sync_docs.CONFIG = root, config
                outputs = sync_docs.render()
                self.assertIn('统一摘要', outputs[root / 'lab/features/person/README.md'])
                self.assertIn('统一摘要', outputs[root / 'docs/TASK_MATRIX.md'])
                self.assertIn('保留图像', outputs[root / 'lab/features/person/README.md'])
                self.assertIn('后台处理候选', outputs[root / 'lab/features/README.md'])
                self.assertIn('待开发', outputs[root / 'lab/features/README.md'])
                (root / 'ws/src/competition_perception/src/competition_perception/person.py').touch()
                self.assertIn('源码已建立', sync_docs.render()[root / 'lab/features/README.md'])
                self.assertIn('before\n', outputs[root / 'docs/TASK_MATRIX.md'])
                self.assertIn('\nafter\n', outputs[root / 'docs/TASK_MATRIX.md'])
            finally:
                sync_docs.ROOT, sync_docs.CONFIG = original_root, original_config


if __name__ == '__main__':
    unittest.main()
