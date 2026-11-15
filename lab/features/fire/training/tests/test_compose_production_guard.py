"""Exercise a fire generator through an isolated SQLite index fixture."""
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT / 'lab/scripts/media/tests'), str(ROOT / 'lab/features/fire/training/scripts')]
import generate_fire_dataset as generator
import test_media_index_sqlite as fixtures


class FireGeneratorGuardTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CacheTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        image = np.full((200, 300, 3), 180, dtype=np.uint8)
        ok, encoded = cv2.imencode('.png', image)
        self.assertTrue(ok)
        self.fixture.image.write_bytes(encoded.tobytes())
        shard = self.root / self.fixture.parts['datasets/raw/camera']['shard']
        item = json.loads(shard.read_text(encoding='utf8'))
        stat = self.fixture.image.stat()
        item.update(bytes=stat.st_size, modified_ns=stat.st_mtime_ns)
        shard.write_text(json.dumps(item) + '\n', encoding='utf8')
        self.assertEqual(self.fixture.call(['sync'])[0], 0)
        script_dir = self.root / 'lab/features/fire/training/scripts'
        script_dir.mkdir(parents=True)
        for name in ('generate_fire_dataset.py', 'prepare_fire_videos.py'):
            shutil.copyfile(ROOT / 'lab/features/fire/training/scripts' / name, script_dir / name)
        (self.root / 'fixtures').mkdir(exist_ok=True)
        self.regions = self.root / 'fixtures/regions.json'
        self.regions.write_text(json.dumps([{'split': 'train', 'verified_empty': True,
            'image': 'datasets/raw/camera/a.png', 'quad': [[40, 40], [100, 40], [100, 120], [40, 120]]}]), encoding='utf8')
        self.output = self.root / 'datasets/outputs/fire'
        self.patches = [mock.patch.object(generator, 'ROOT', self.root),
                        mock.patch.object(generator, '__file__', str(script_dir / 'generate_fire_dataset.py')),
                        mock.patch.object(generator, 'workspace_path', lambda value: (self.root / value).resolve())]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_identical_invocation_reuses_without_writing(self):
        self.assertEqual(generator.generate(self.regions, self.fixture.image, self.output, 4, 12), 4)
        target = self.output / 'manifest.json'
        source_refs = json.loads(target.read_text(encoding='utf8'))[0]['batch_source_refs']
        self.assertIn('datasets/raw/camera/a.png', source_refs)
        self.assertTrue(any(ref.get('manifest') and ref.get('relation') and ref.get('record_index') == 0
                            for ref in source_refs['datasets/raw/camera/a.png']))
        old = target.stat().st_mtime_ns
        with mock.patch.object(generator, '_generate', side_effect=AssertionError('body reran')):
            self.assertEqual(generator.generate(self.regions, self.fixture.image, self.output, 4, 12), 4)
        self.assertEqual(target.stat().st_mtime_ns, old)

    def test_missing_database_fails_before_output(self):
        database = self.root / 'datasets/annotated/media_index/query_cache.sqlite3'
        database.unlink()
        with self.assertRaisesRegex(ValueError, 'SQLite cache missing'):
            generator.generate(self.regions, self.fixture.image, self.output, 4, 12)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
