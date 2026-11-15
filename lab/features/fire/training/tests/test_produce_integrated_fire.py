"""Isolated contracts for source selection, framing and planned quotas."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SCRIPTS = ROOT / 'lab/features/fire/training/scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('produce_integrated_fire', SCRIPTS / 'produce_integrated_fire.py')
fire = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fire)


class IntegratedFireContracts(unittest.TestCase):
    def test_database_intersection_selection_and_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / 'datasets/annotated/fire/background_source_manifest.jsonl'
            manifest.parent.mkdir(parents=True)
            entries, reviewed = [], []
            for i in range(8):
                path = f'datasets/raw/fire/building_{i}.png'
                image = root / path
                image.parent.mkdir(parents=True, exist_ok=True)
                image.touch()
                entries.append({'path': path, 'observed_tasks': [], 'view': '正面'})
                if i < 7:
                    reviewed.append({'image': path})
            manifest.write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in entries), encoding='utf8')

            def connect(readonly):
                db = sqlite3.connect(':memory:')
                db.executescript('create table partitions(name text, descriptor text); '
                                 'create table media(path_key text, item_json text, partition_name text);')
                db.execute('insert into partitions values (?, ?)', ('raw', json.dumps({'name': 'raw', 'source_present': True})))
                db.executemany('insert into media values (?, ?, ?)',
                    [(e['path'], json.dumps({'media_type': 'image'}), 'raw') for e in entries if not e['path'].endswith('_6.png')])
                db.commit()
                return db

            with patch.object(fire, 'ROOT', root), patch.object(fire.media_index, 'connect', side_effect=connect), \
                 patch.object(fire.media_index, 'validated', return_value=({'snapshot_utc': 'fixture'}, [], True)), \
                 patch.object(fire.media_index, 'path_key', side_effect=lambda p: p), \
                 patch.object(fire.media_index.core, 'lifecycle_reason', return_value=None):
                one, evidence = fire.select_sources(reviewed, 42)
                two, _ = fire.select_sources(reviewed, 42)
                self.assertEqual(one, two)
                self.assertEqual(len(one), 6)
                self.assertNotIn('datasets/raw/fire/building_6.png', evidence['selected'])
                self.assertIn('datasets/raw/fire/building_7.png', evidence['excluded_from_reviewed_intersection'])
                with self.assertRaisesRegex(ValueError, 'no indexed and inspected'):
                    fire.select_sources([{'image': 'not_indexed.png'}], 42)

    def test_frame_transforms_labels_and_rejects_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / 'image.png'
            label = Path(tmp) / 'label.txt'
            cv2.imencode('.png', np.zeros((120, 160, 3), dtype=np.uint8))[1].tofile(str(image))
            label.write_text('0 0.250000 0.291667 0.125000 0.166667\n', encoding='utf8')
            fire.frame_and_labels(image, label, [20, 15, 100, 75])
            values = list(map(float, label.read_text().split()[1:]))
            for actual, expected in zip(values, (.25, 1/3, .25, 1/3)):
                self.assertAlmostEqual(actual, expected, places=5)
            self.assertEqual(cv2.imdecode(np.fromfile(str(image), np.uint8), 1).shape[:2], (480, 640))
            cv2.imencode('.png', np.zeros((120, 160, 3), dtype=np.uint8))[1].tofile(str(image))
            label.write_text('0 0.062500 0.291667 0.125000 0.166667\n', encoding='utf8')
            before = image.read_bytes(), label.read_bytes()
            with self.assertRaisesRegex(ValueError, 'truncate'):
                fire.frame_and_labels(image, label, [20, 15, 100, 75])
            self.assertEqual(before, (image.read_bytes(), label.read_bytes()))

    def test_bayesian_quotas_are_bounded_and_weighted(self):
        plan = fire.plan_production(fire.declaration(100, 42, ['photo']), ROOT)
        counts = [row['planned_count'] for row in plan['strata']]
        self.assertEqual(sum(counts), 100)
        self.assertTrue(all(1 <= n <= 97 for n in counts))
        self.assertGreater(len(set(counts)), 1)


if __name__ == '__main__':
    unittest.main()
