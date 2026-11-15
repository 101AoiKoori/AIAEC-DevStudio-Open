"""Contract checks for the meter source selector and bounded scene declarations."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'lab/features/meter/training/scripts'))
import plan_meter_sources_v4 as planner


class MeterSourcePlanTests(unittest.TestCase):
    def test_use_fields_must_match_the_same_relationship(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as tmp:
            root = Path(tmp)
            (root / 'fixture.dat').write_text('existing input', encoding='utf8')
            manifest = root / 'manifest.jsonl'
            records = [
                {'path': 'fixture.dat', 'task_use_status': [
                    {'task': 'meter', 'split': 'val', 'role': 'direct_sample', 'status': 'ready'},
                    {'task': 'person', 'split': 'train', 'role': 'direct_sample', 'status': 'ready'}]},
                {'path': 'fixture.dat', 'task_use_status': [
                    {'task': 'meter', 'split': 'train', 'role': 'direct_sample', 'status': 'ready'}]},
                {'path': 'fixture.dat', 'task_use_status': [
                    {'task': 'meter', 'split': 'train', 'role': 'direct_sample', 'status': 'blocked'}]},
            ]
            manifest.write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf8')
            with mock.patch.object(planner, 'ROOT', root):
                result = planner.image_uses(manifest, task='meter', split='train',
                                            role='direct_sample', status='ready')
            self.assertEqual([row['record_index'] for row in result], [1])

    def test_scene_states_are_unique_seeded_and_entry_compatible(self):
        for kind in ('arena', 'station'):
            plan = planner.scene_plan(kind, 369, 20261114, 'arena.blend', 'station.blend')
            repeated = planner.scene_plan(kind, 369, 20261114, 'arena.blend', 'station.blend')
            self.assertEqual(plan, repeated)
            self.assertEqual(len({json.dumps(row['parameters'], sort_keys=True) for row in plan['strata']}), 369)
            if kind == 'arena':
                planner.validate_arena_declaration(plan, count=369,
                    sources=['arena.blend', 'station.blend'], foreground=False)
            else:
                planner.validate_station_declaration(plan, count=369, source='station.blend', allow_occlusion=False)

    def test_bounded_scene_pool_rejects_excess(self):
        with self.assertRaisesRegex(ValueError, 'exceed'):
            planner.scene_plan('arena', 401, 1, 'arena.blend', 'station.blend')


if __name__ == '__main__':
    unittest.main()
