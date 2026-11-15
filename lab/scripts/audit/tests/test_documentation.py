import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "documentation.py"
SPEC = importlib.util.spec_from_file_location("documentation", SCRIPT)
documentation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = documentation
SPEC.loader.exec_module(documentation)


class DocumentationCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_missing_file_and_anchor_are_reported(self):
        source = self.write("docs/current.md", "[bad](missing.md#section)")
        result = documentation.check(self.root, [source])
        self.assertEqual([item["reason"] for item in result["findings"]], ["missing-file"])

    def test_existing_file_and_github_anchor_are_valid(self):
        target = self.write("docs/target.md", "# 中文 标题\n")
        source = self.write("docs/current.md", "[ok](target.md#中文-标题)")
        result = documentation.check(self.root, [source])
        self.assertEqual(result["findings"], [])

    def test_anchor_inside_target_document_is_checked(self):
        self.write("docs/target.md", "# 已有标题\n")
        source = self.write("docs/current.md", "[bad](target.md#missing)")
        result = documentation.check(self.root, [source])
        self.assertEqual(result["findings"][0]["reason"], "missing-anchor")

    def test_duplicate_headings_and_html_ids_are_supported(self):
        target = self.write(
            "docs/target.md",
            "# Same\n\n<a id='explicit'></a>\n\n# Same\n",
        )
        source = self.write(
            "docs/current.md",
            "[first](target.md#same) [second](target.md#same-1) [html](target.md#explicit)",
        )
        result = documentation.check(self.root, [source])
        self.assertEqual(result["findings"], [])

    def test_fenced_links_and_external_links_are_ignored(self):
        source = self.write(
            "docs/current.md",
            "```md\n[ignored](missing.md)\n```\n\n[web](https://example.com/a#b)\n",
        )
        result = documentation.check(self.root, [source])
        self.assertEqual(result["findings"], [])
        self.assertEqual(result["links"], 0)


if __name__ == "__main__":
    unittest.main()
