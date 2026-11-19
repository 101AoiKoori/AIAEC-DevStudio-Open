"""Exercise sample replacement with the real guard and isolated SQLite state."""
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'lab/scripts/media/tests'))
import test_media_index_sqlite as fixtures

from datasets.scripts.waste_bin_state import build_current_sample as sample
from datasets.scripts.waste_bin_state import sample_publish


class SampleGuardIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CacheTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.image = self.fixture.root, self.fixture.image
        shard = self.root / self.fixture.parts['datasets/raw/camera']['shard']
        item = json.loads(shard.read_text(encoding='utf8'))
        stat = self.image.stat()
        item.update(bytes=stat.st_size, modified_ns=stat.st_mtime_ns)
        shard.write_text(json.dumps(item) + '\n', encoding='utf8')
        self.assertEqual(self.fixture.call(['sync'])[0], 0)
        for name in ('datasets/scripts/waste_bin_state/build_current_sample.py',
                     'datasets/scripts/waste_bin_state/sample_publish.py',
                     'datasets/scripts/waste_bin_state/config/dataset.yaml',
                     'datasets/scripts/waste_bin_state/config/source.yaml',
                     'datasets/scripts/common/recoverable_publish.py',
                     'datasets/scripts/common/package_direct_real.py',
                     'datasets/scripts/common/frame_environment.py',
                     'datasets/scripts/common/source_history.py',
                     'datasets/scripts/common/paths.py',
                     'datasets/annotated/frames.sqlite3',
                     'fixtures/composites.json', 'fixtures/plan.json', 'fixtures/allocation.json'):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}', encoding='utf8')
        self.output = self.root / 'datasets/samples/waste_bin_state'
        self.output.mkdir(parents=True)
        (self.output / '.gitkeep').touch()
        self.paths = [self.root / ('fixtures/' + name + '.json')
                      for name in ('composites', 'plan', 'allocation')]
        self.config = ({'sample_output': 'datasets/samples/waste_bin_state'},
                       {'databases': {'frame_environment': 'datasets/annotated/frames.sqlite3'}})
        self.record = {'source_kind': 'direct_empty_negative', 'split': 'train', 'bins': []}
        self.patches = [
            mock.patch.object(sample, 'ROOT', self.root),
            mock.patch.object(sample, '__file__', str(self.root / 'datasets/scripts/waste_bin_state/build_current_sample.py')),
            mock.patch.object(sample_publish, '__file__', str(self.root / 'datasets/scripts/waste_bin_state/sample_publish.py')),
            mock.patch.object(sample, 'sample_dir', return_value=self.output),
            mock.patch.object(sample, 'load_task_config', return_value=self.config),
            mock.patch.object(sample, 'direct_inputs', return_value=([(self.image, '', self.record)], {self.image}, set())),
            mock.patch.object(sample, 'composite_inputs', return_value=([], set(), set())),
            mock.patch.object(sample, 'verify_pair', return_value=[640, 480]),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def history(self):
        with sqlite3.connect(fixtures.core.INDEX / 'production_state.sqlite3') as db:
            return db.execute('SELECT run_token FROM production_receipt_history '
                              'WHERE operation=? ORDER BY id',
                              ('waste_bin_state.build_current_sample',)).fetchall()

    def test_real_guard_rerun_preserves_unowned_and_receipts(self):
        first = sample.package(*self.paths)
        self.assertTrue((self.output / '.gitkeep').exists())
        reused = sample.package(*self.paths)
        self.assertEqual(first, reused)
        self.assertEqual(len(self.history()), 1)
        (self.output / 'notes.txt').write_text('human', encoding='utf8')
        second = sample.package(*self.paths, replace=True)
        self.assertEqual(first['summary'], second['summary'])
        self.assertEqual((self.output / 'notes.txt').read_text(encoding='utf8'), 'human')
        self.assertEqual(len(self.history()), 2)
        self.assertEqual(list(self.output.parent.glob('.waste_bin_state.backup.*')), [])

    def test_real_guard_rejects_unowned_collision(self):
        (self.output / 'images').mkdir()
        collision = self.output / 'images/direct_empty_negative_0001.png'
        collision.write_bytes(b'human')
        with self.assertRaisesRegex(ValueError, 'unowned'):
            sample.package(*self.paths, replace=True)
        self.assertEqual(collision.read_bytes(), b'human')
        self.assertEqual(self.history(), [])

    def test_real_guard_failure_restores_previous_sample(self):
        sample.package(*self.paths)
        original = sample_publish.signature(self.output)
        with mock.patch.object(sample_publish, 'publish', side_effect=self.publish_then_fail):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                sample.package(*self.paths, replace=True)
        self.assertEqual(sample_publish.signature(self.output), original)
        self.assertEqual(len(self.history()), 1)

    def test_killed_publisher_recovers_using_real_receipt_store(self):
        sample.package(*self.paths)
        (self.output / 'notes.txt').write_text('human', encoding='utf8')
        script = r'''
import os, pathlib, sys
from unittest import mock
from datasets.scripts.waste_bin_state import build_current_sample as s, sample_publish as p
from lab.training_common import production_guard as g
r = pathlib.Path(sys.argv[1])
g._cache.core.ROOT = r
g._cache.core.INDEX = r / 'datasets/annotated/media_index'
g._cache.DATABASE = g._cache.core.INDEX / 'query_cache.sqlite3'
g._cache.core.load_lifecycle_rules()
out = r / 'datasets/samples/waste_bin_state'
image = r / 'datasets/raw/camera/a.png'
record = {'source_kind': 'direct_empty_negative', 'split': 'train', 'bins': []}
cfg = ({'sample_output': 'datasets/samples/waste_bin_state'},
       {'databases': {'frame_environment': 'datasets/annotated/frames.sqlite3'}})
paths = [r / ('fixtures/' + n + '.json') for n in ('composites', 'plan', 'allocation')]
old = p.publish
def crash(output, stage, token):
    old(output, stage, token)
    os._exit(43)
with mock.patch.object(s, 'ROOT', r), \
     mock.patch.object(s, '__file__', str(r / 'datasets/scripts/waste_bin_state/build_current_sample.py')), \
     mock.patch.object(p, '__file__', str(r / 'datasets/scripts/waste_bin_state/sample_publish.py')), \
     mock.patch.object(s, 'sample_dir', return_value=out), \
     mock.patch.object(s, 'load_task_config', return_value=cfg), \
     mock.patch.object(s, 'direct_inputs', return_value=([(image, '', record)], {image}, set())), \
     mock.patch.object(s, 'composite_inputs', return_value=([], set(), set())), \
     mock.patch.object(s, 'verify_pair', return_value=[640, 480]), \
     mock.patch.object(p, 'publish', side_effect=crash):
    s.package(*paths, replace=True)
'''
        child = subprocess.run([sys.executable, '-c', script, str(self.root)],
                               cwd=str(ROOT), capture_output=True, text=True, timeout=20)
        self.assertEqual(child.returncode, 43, child.stderr)
        self.assertTrue(sample_publish._metadata(self.output)[1].is_file())
        sample.package(*self.paths, replace=True)
        self.assertEqual((self.output / 'notes.txt').read_text(encoding='utf8'), 'human')
        self.assertEqual(len(self.history()), 2)
        self.assertFalse(sample_publish._metadata(self.output)[1].exists())

    @staticmethod
    def publish_then_fail(output, stage, token):
        # Call a saved reference so the patched function cannot recurse.
        ORIGINAL_PUBLISH(output, stage, token)
        raise RuntimeError('interrupted')


ORIGINAL_PUBLISH = sample_publish.publish


if __name__ == '__main__':
    unittest.main()
