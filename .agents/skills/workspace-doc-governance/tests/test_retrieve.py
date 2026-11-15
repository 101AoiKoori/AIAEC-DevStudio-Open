import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/retrieve.py'
spec = importlib.util.spec_from_file_location('retrieve', SCRIPT)
retrieve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retrieve)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        return path

    def test_late_requirement_is_visible_without_generated_readme(self):
        body = '# 需求\n\n' + '背景说明' * 600 + '\n\n## 状态与提醒\n\n'
        body += '- `[待解决]` PLATE-2：引擎\n- `[待全局规划]` PLATE-3：扫描与算力'
        path = self.write('config/perception_tasks.json', json.dumps({'tasks': [
            {'id': 'plate', 'summary': '车牌', 'requirements_markdown': body}
        ]}, ensure_ascii=False, indent=2))
        results, errors = retrieve.search(self.root, 'PLATE-3', 1)
        self.assertEqual(errors, [])
        score, (source, line, heading, text) = results[0]
        self.assertEqual(source, 'config/perception_tasks.json')
        self.assertIn('PLATE-3', retrieve.excerpt(text, 'PLATE-3'))
        self.assertNotIn('PLATE-2', text)
        self.assertIn('状态与提醒', heading)
        self.assertIn('requirements_markdown', path.read_text(encoding='utf-8').splitlines()[line - 1])

    def test_long_paragraph_excerpt_keeps_matching_term(self):
        body = '背景' * 1200 + '可检索的针尖缺失' + '结束' * 100
        result = retrieve.excerpt(body, '针尖缺失')
        self.assertIn('针尖缺失', result)
        self.assertLessEqual(len(result), 1602)

    def test_archive_is_not_active_but_current_owner_is(self):
        self.write('refs/archive/old.md', '# 旧方案\n\n旧原型特有词')
        self.write('docs/current.md', '# 当前\n\n八点定位')
        self.assertEqual(retrieve.search(self.root, '旧原型特有词', 5)[0], [])
        results, errors = retrieve.search(self.root, '八点定位', 5)
        self.assertEqual(errors, [])
        self.assertEqual(results[0][1][0], 'docs/current.md')

    def test_script_owner_readmes_are_indexed(self):
        self.write('lab/scripts/README.md', '# 工具\n\n标注审计入口')
        self.write('datasets/scripts/README.md', '# 数据工具\n\n父子源检查入口')
        results, errors = retrieve.search(self.root, '标注审计', 5)
        self.assertEqual(errors, [])
        self.assertEqual(results[0][1][0], 'lab/scripts/README.md')
        results, errors = retrieve.search(self.root, '父子源检查', 5)
        self.assertEqual(errors, [])
        self.assertEqual(results[0][1][0], 'datasets/scripts/README.md')

    def test_model_and_runtime_docs_are_indexed_but_archive_is_not(self):
        self.write('models/README.md', '# 模型\n\n部署权重台账')
        self.write('ws/src/demo/README.md', '# 节点\n\n运行接口说明')
        self.write('refs/provincial/README.md', '# 省赛\n\n历史来源')
        self.write('refs/archive/old.md', '# 旧记录\n\nruntimehiddenword')
        for query, expected in (
            ('部署权重台账', 'models/README.md'),
            ('运行接口说明', 'ws/src/demo/README.md'),
            ('历史来源', 'refs/provincial/README.md'),
        ):
            results, errors = retrieve.search(self.root, query, 5)
            self.assertEqual(errors, [])
            self.assertEqual(results[0][1][0], expected)
        self.assertEqual(retrieve.search(self.root, 'runtimehiddenword', 5)[0], [])

    def test_bad_config_reports_error_and_retains_other_matches(self):
        self.write('config/perception_tasks.json', '{broken')
        self.write('docs/current.md', '# 当前\n\n八点定位')
        results, errors = retrieve.search(self.root, '八点定位', 5)
        self.assertTrue(results)
        self.assertEqual(len(errors), 1)
        self.assertIn('perception_tasks.json', errors[0])


if __name__ == '__main__':
    unittest.main()
