import json
import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sync_docs


class RenderTest(unittest.TestCase):
    def test_stage_skips_index_write_when_generated_content_matches(self):
        path = sync_docs.ROOT / 'docs/fixture.md'
        with patch.object(sync_docs, 'check_staged_config'), \
             patch.object(sync_docs, 'render', return_value={path: 'current\n'}), \
             patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'read_text', return_value='current\n'), \
             patch.object(sys, 'argv', ['sync_docs.py', '--stage']), \
             patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, b'current\n', b'')), \
             patch.object(subprocess, 'check_call') as stage:
            self.assertEqual(sync_docs.main(), 0)
            stage.assert_not_called()

    def test_stage_updates_index_when_only_index_is_stale(self):
        path = sync_docs.ROOT / 'docs/fixture.md'
        with patch.object(sync_docs, 'check_staged_config'), \
             patch.object(sync_docs, 'render', return_value={path: 'current\n'}), \
             patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'read_text', return_value='current\n'), \
             patch.object(sys, 'argv', ['sync_docs.py', '--stage']), \
             patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, b'old\n', b'')), \
             patch.object(subprocess, 'check_call') as stage:
            self.assertEqual(sync_docs.main(), 0)
            stage.assert_called_once_with(['git', 'add', '--', 'docs/fixture.md'], cwd=str(sync_docs.ROOT))

    def test_waste_mapping_matches_generated_deployment_config(self):
        outputs = sync_docs.render()
        config = json.loads(sync_docs.CONFIG.read_text(encoding='utf-8'))
        task = next(t for t in config['tasks'] if t['id'] == 'waste_bin')
        generated = json.loads(outputs[sync_docs.ROOT / 'ws/src/aec_vision/config/waste_bin/items.json'])
        self.assertEqual(generated['items'], task['waste_items'])
        self.assertEqual(len(generated['items']), 24)

    def test_stage_refuses_unstaged_requirement_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'config/perception_tasks.json'
            config.parent.mkdir()
            config.write_text('{"tasks": []}\n', encoding='utf-8')
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            subprocess.run(['git', '-C', str(root), 'add', 'config/perception_tasks.json'], check=True)
            with patch.object(sync_docs, 'ROOT', root), patch.object(sync_docs, 'CONFIG', config):
                sync_docs.check_staged_config()
                config.write_text('{"tasks": [], "pending": true}\n', encoding='utf-8')
                error = io.StringIO()
                with patch.object(sys, 'argv', ['sync_docs.py', '--stage']), \
                        patch.object(sync_docs, 'render') as render, contextlib.redirect_stderr(error):
                    self.assertEqual(sync_docs.main(), 2)
                render.assert_not_called()
                self.assertIn('未暂存', error.getvalue())
                staged = subprocess.check_output(['git', '-C', str(root), 'show', ':config/perception_tasks.json'], text=True)
                self.assertEqual(json.loads(staged), {'tasks': []})

    def test_requirement_item_identity_and_states(self):
        task = {'id': 'person', 'requirements_markdown':
                '- `[待解决]` PERSON-R01：计数\n[已明确] PERSON-COUNT: 汇总口径'}
        sync_docs.validate_requirement_items([task])
        for body in ('- `[已完成]` PERSON-R01：计数',
                     '- `[待解决]` 无编号说明',
                     '- `[待解决]` PLATE-R01：计数',
                     '- `[待解决]` PERSON-R01：',
                     '- `[已明确]` PERSON-R01：设计\n- `[待解决]` PERSON-R01：实现'):
            with self.subTest(body=body), self.assertRaises(ValueError):
                sync_docs.validate_requirement_items([dict(task, requirements_markdown=body)])

    def test_requirement_prose_and_links_are_not_status_items(self):
        sync_docs.validate_requirement_items([{'id': 'person', 'requirements_markdown':
            '# 需求\n[文档](source.md)\n示例包含 `[已明确]` 状态标签。\n- 普通实现说明\n'
            '```markdown\n- `[未定]` 示例占位\n```\n'
            '~~~\n- `[未定]` EXAMPLE-1：示例\n~~~'}])
        with self.assertRaises(ValueError):
            sync_docs.validate_requirement_items([{'id': 'person', 'requirements_markdown':
                '```\n示例\n```\n- `[未定]` 缺少编号'}])

    def test_requirements_and_module_status_are_shared(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('config', 'lab/features', 'docs', 'ws/src/aec_vision/src/aec_vision'):
                (root / name).mkdir(parents=True, exist_ok=True)
            for name in ('lab/features/README.md', 'docs/TASK_MATRIX.md'):
                (root / name).write_text('before\n<!-- TASKS:START -->\nold\n<!-- TASKS:END -->\nafter\n', encoding='utf-8')
            task = {'id': 'person', 'summary': '统一摘要', 'score': '12',
                    'matrix_name': '人物', 'matrix_check': '检查', 'requirements_markdown': '# 详细需求',
                    'scheduling': {'mode': '后台处理候选', 'approach': '保留图像'},
                    'research_doc': '../../../docs/research/OPEN_SOURCE_METHODS.md#person'}
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
                (root / 'ws/src/aec_vision/src/aec_vision/person.py').touch()
                self.assertIn('源码已建立', sync_docs.render()[root / 'lab/features/README.md'])
                self.assertIn('before\n', outputs[root / 'docs/TASK_MATRIX.md'])
                self.assertIn('\nafter\n', outputs[root / 'docs/TASK_MATRIX.md'])
            finally:
                sync_docs.ROOT, sync_docs.CONFIG = original_root, original_config


if __name__ == '__main__':
    unittest.main()
