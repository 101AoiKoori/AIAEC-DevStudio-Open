"""Check entrypoints reserve output only after source-index validation."""
import contextlib
import importlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
WORKSPACE = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(WORKSPACE / 'lab/scripts/media/tests'))
import test_media_index_sqlite as fixtures
from lab.training_common import production_guard


class CheckGuardTests(unittest.TestCase):
    def test_real_sqlite_missing_and_reuse(self):
        for name in ('check_sticker_closeups_v4', 'check_sticker_merge_v4'):
            with self.subTest(entry=name):
                fixture = fixtures.CacheTests('runTest')
                fixture.setUp()
                try:
                    root = fixture.root
                    shard = root / fixture.parts['datasets/raw/camera']['shard']
                    row = json.loads(shard.read_text(encoding='utf8'))
                    row['modified_ns'] = fixture.image.stat().st_mtime_ns
                    shard.write_text(json.dumps(row) + '\n', encoding='utf8')
                    module = importlib.import_module(name)
                    source = root / 'source'
                    dataset = root / 'dataset'
                    source.mkdir(); dataset.mkdir()
                    (source / 'manifest.json').write_text('{"records": []}', encoding='utf8')
                    (dataset / 'manifest.json').write_text('{"records": []}', encoding='utf8')
                    (dataset / 'foreground_isolation.json').write_text('[]', encoding='utf8')
                    script = root / 'script.py'; script.write_text('# fixture', encoding='utf8')
                    base = root / 'base.py'; base.write_text('# fixture', encoding='utf8')
                    calls = []
                    is_close = name == 'check_sticker_closeups_v4'
                    target = dataset / ('structural_check.json' if is_close else 'merge_isolation_proposal.json')
                    argv = [name, str(dataset)] if is_close else [name, '--base', str(source), '--supplement', str(dataset)]
                    def body(*args):
                        calls.append(1)
                        target.write_text('{"status": "structural_check_pass"}', encoding='utf8')
                    patches = [patch.object(production_guard, '_index', return_value=fixtures.cache),
                               patch.object(module, '__file__', str(script)),
                               patch.object(module.gen if is_close else module.base, 'ROOT', root),
                               patch.object(module.gen.base if is_close else module.base, '__file__', str(base)),
                               patch.object(module, '_check', side_effect=body), patch.object(sys, 'argv', argv)]
                    if is_close:
                        gen = root / 'gen.py'; gen.write_text('# fixture', encoding='utf8')
                        grouped = root / 'grouped.py'; grouped.write_text('# fixture', encoding='utf8')
                        patches += [patch.object(module.gen, '__file__', str(gen)),
                                    patch.object(module.grouped, '__file__', str(grouped)),
                                    patch.object(module, 'screen_sources', return_value=({fixture.image}, set()))]
                    with contextlib.ExitStack() as stack:
                        for item in patches:
                            stack.enter_context(item)
                        with self.assertRaisesRegex(ValueError, 'SQLite cache missing'):
                            module.main()
                        self.assertFalse(target.exists())
                        self.assertEqual(fixture.call(['sync'])[0], 0)
                        module.main()
                        timestamp = target.stat().st_mtime_ns
                        module.main()
                        self.assertEqual(calls, [1])
                        self.assertEqual(timestamp, target.stat().st_mtime_ns)
                finally:
                    fixture.doCleanups()


if __name__ == '__main__':
    unittest.main()
