import copy
import json
import sys
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'src'))
from aec_vision.runtime_config import ConfigError, enabled_tasks, load_config, prepare_route, validate_config


class RuntimeConfigTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config(PACKAGE / 'config/competition.json')

    def usable_route(self):
        self.config['tasks']['traffic_light']['enabled'] = True
        self.config['tasks']['sign']['enabled'] = True
        self.config['tasks']['person']['enabled'] = True
        self.config['selected_route'] = 'route_a'
        self.config['routes']['route_a']['waypoints'] = [
            {'id': 'observation', 'pose': {'x': 1.0, 'y': 2.0, 'yaw': 0.0},
             'calibrated': True, 'tasks': [{'id': 'a_people', 'type': 'person', 'context': {'street': 'A'}},
                                           {'id': 'plate_1', 'type': 'plate'}]}]

    def test_default_configuration_is_safe_and_no_task_active(self):
        self.assertEqual(enabled_tasks(self.config), set())
        with self.assertRaises(ConfigError):
            prepare_route(self.config)

    def test_selection_filters_tasks_without_mutating_route(self):
        self.usable_route()
        original = copy.deepcopy(self.config)
        self.assertEqual([t['id'] for t in prepare_route(self.config)[0]['tasks']], ['a_people'])
        self.assertEqual(self.config, original)

    def test_route_cannot_run_with_missing_gates_or_calibration(self):
        self.usable_route()
        self.config['tasks']['sign']['enabled'] = False
        with self.assertRaises(ConfigError):
            prepare_route(self.config)
        self.config['tasks']['sign']['enabled'] = True
        self.config['routes']['route_a']['waypoints'][0]['calibrated'] = False
        with self.assertRaises(ConfigError):
            prepare_route(self.config)

    def test_rejects_string_boolean_and_invalid_waypoint(self):
        self.config['tasks']['person']['enabled'] = 'false'
        with self.assertRaises(ConfigError):
            validate_config(self.config)
        self.usable_route()
        self.config['routes']['route_a']['waypoints'][0]['pose']['x'] = float('nan')
        with self.assertRaises(ConfigError):
            validate_config(self.config)

    def test_route_selection_uses_selected_waypoints(self):
        self.usable_route()
        self.config['routes']['route_b']['waypoints'] = [
            {'id': 'other', 'pose': {'x': 3, 'y': 4, 'yaw': 1}, 'calibrated': True,
             'tasks': [{'id': 'b_people', 'type': 'person', 'context': {'street': 'B'}}]}]
        self.config['selected_route'] = 'route_b'
        self.assertEqual(prepare_route(self.config)[0]['id'], 'other')
        self.assertEqual(prepare_route(self.config, 'route_a')[0]['id'], 'observation')


if __name__ == '__main__':
    unittest.main()
