"""Review-sheet commands enforce production receipts before writing."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'lab/scripts/media/tests'))
sys.path.insert(0, str(ROOT / 'lab/features/waste_bin/training/scripts'))
import test_media_index_sqlite as fixtures
from lab.training_common import production_guard


def load(name):
    p = ROOT / 'lab/features/waste_bin/training/scripts' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name + '_gate_fixture', p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReviewGuardTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CacheTests('runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        (self.root / 'fixtures').mkdir(exist_ok=True)
        self.code = self.root / 'fixtures/sheet.py'
        self.code.write_text('# fixture', encoding='utf8')
        self.helper = self.root / 'fixtures/helper.py'
        self.helper.write_text('# fixture', encoding='utf8')
        patcher = mock.patch.object(production_guard, '_index', return_value=fixtures.cache)
        patcher.start(); self.addCleanup(patcher.stop)
        self.calls = []

    def prepare(self, kind):
        module = load(kind)
        image = self.fixture.image
        shard = self.root / self.fixture.parts['datasets/raw/camera']['shard']
        record = json.loads(shard.read_text(encoding='utf8'))
        if kind == 'review_sticker_sources':
            image = image.with_suffix('.npz')
            self.fixture.image.rename(image)
            for part in self.fixture.parts.values():
                for key in ('shard', 'evidence_shard'):
                    file = self.root / part[key]
                    file.write_text(file.read_text(encoding='utf8').replace('a.png', 'a.npz'), encoding='utf8')
            record['path'] = record['path'].replace('a.png', 'a.npz')
            record['id'] = record['id'].replace('a.png', 'a.npz')
            classes = self.root / 'classes.yaml'
            classes.write_text('items:\n- name: a\n', encoding='utf8')
            task = self.root / 'config/perception_tasks.json'
            task.parent.mkdir(); task.write_text('{}', encoding='utf8')
            out = self.root / 'runs/sheet.png'; out.parent.mkdir()
            patches = [mock.patch.object(module.base, 'ROOT', self.root), mock.patch.object(module.base, 'CLASSES', classes),
                       mock.patch.object(module.base, 'CACHE', image.parent), mock.patch.object(module.base, '__file__', str(self.helper)),
                       mock.patch.object(sys, 'argv', ['review', '--out', str(out)])]
        else:
            manifest = image.parent / 'manifest.json'
            manifest.write_text(json.dumps({'samples':[{'group':'a','states':['closed'], 'image':'a.png'}]}), encoding='utf8')
            out = self.root / 'runs/sheet.png'; out.parent.mkdir()
            patches = [mock.patch.object(module, 'ROOT', self.root),
                       mock.patch.object(sys, 'argv', ['sheet', str(image.parent), '--out', str(out)])]
        record['modified_ns'] = image.stat().st_mtime_ns
        shard.write_text(json.dumps(record) + '\n', encoding='utf8')
        self.fixture.write_catalog()
        patches += [mock.patch.object(module, '__file__', str(self.code))]
        for patcher in patches:
            patcher.start(); self.addCleanup(patcher.stop)
        def render(*args):
            self.calls.append(1)
            out.write_text('fixture', encoding='utf8')
        patcher = mock.patch.object(module, '_render', side_effect=render)
        patcher.start(); self.addCleanup(patcher.stop)
        return module, out

    def check_entry(self, kind):
        module, out = self.prepare(kind)
        with self.assertRaisesRegex(ValueError, 'SQLite cache missing'):
            module.main()
        self.assertFalse(out.exists())
        self.assertEqual(self.fixture.call(['sync'])[0], 0)
        module.main()
        before = out.stat().st_mtime_ns
        module.main()
        self.assertEqual(self.calls, [1])
        self.assertEqual(out.stat().st_mtime_ns, before)

    def test_bin_review_sheet(self):
        self.check_entry('make_bin_review_sheet')

    def test_sticker_source_sheet(self):
        self.check_entry('review_sticker_sources')


if __name__ == '__main__':
    unittest.main()
