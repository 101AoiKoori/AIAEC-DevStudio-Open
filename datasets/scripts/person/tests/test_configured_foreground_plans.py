"""New person plans preserve the configured foreground source identities."""
import unittest
from unittest.mock import patch

from datasets.scripts.person import build_full_person_plan as planner
from datasets.scripts.person import build_evaluation_plan as evaluation
from datasets.scripts.person import planned_sources


class ConfiguredForegroundPlanTest(unittest.TestCase):
    def test_new_plan_actors_use_registered_manifests(self):
        roles = {'foreground_texture': 'new/front.json', 'vehicle_occluder': 'new/vehicle.json',
                 'white_back': ['new/observed.json', 'new/paired.json']}
        data = {
            'dispatch.json': {'records': []},
            'new/front.json': {'records': [{'image': 'front_alice.png', 'class': 'community'}]},
            'new/vehicle.json': {'records': [{'cropped_image': 'vehicle.png', 'class': 'fallen'}]},
            'new/observed.json': {'foreground': 'observed.png'},
            'new/paired.json': {'records': [{'canonical_parent_image': 'front_alice.png',
                                             'foreground': 'back_alice.png'}]},
        }
        with patch('datasets.scripts.common.config.load_task_config', return_value=({}, {'role_records': roles})), \
             patch.object(planner, 'read_json', side_effect=lambda path: data[path]), \
             patch.object(planner, 'png_size', return_value=(100, 200)):
            _, identities, vehicles = planner.build_sources('dispatch.json')
            person = planner.person_actor(identities[0], {'foot_xy': [300, 320], 'size_wh': [50, 100]}, 0)
            vehicle = planner.vehicle_actor(vehicles[0], 0, [[0, 200], [640, 200], [640, 359], [0, 359]],
                                            {'foot_xy': [300, 320], 'size_wh': [50, 100]}, 0)
            standalone = planner.standalone_vehicle(vehicles[0], 0,
                                                    [[0, 200], [640, 200], [640, 359], [0, 359]], 0)
            white = planner.white_back_textures()

        self.assertEqual(person['source_record']['manifest'], roles['foreground_texture'])
        self.assertIn('new/front.json:record_index=0', person['identity_evidence'])
        self.assertEqual(vehicle['source_record']['manifest'], roles['vehicle_occluder'])
        self.assertEqual(standalone['source_record']['manifest'], roles['vehicle_occluder'])
        self.assertEqual(white['front_alice.png'], (0, 'back_alice.png', .5, 'new/paired.json'))
        for role, actor in [('foreground_texture', person), ('vehicle_occluder', vehicle)]:
            self.assertIn(actor['source_record']['manifest'], planned_sources.role_paths({'role_records': roles}, role))

    def test_contrasts_and_evaluation_use_selected_white_back(self):
        white = {'front_b.png': (3, 'back_b.png', .5, 'new/paired.json')}
        actors = [
            {'id': 'a', 'category': 'community', 'source': 'front_a.png', 'foot_xy': [1, 2],
             'size_wh': [50, 100], 'angles_degrees': {'yaw': 0}},
            {'id': 'b', 'category': 'noncommunity', 'source': 'front_b.png', 'foot_xy': [2, 2],
             'size_wh': [50, 100], 'angles_degrees': {'yaw': 0}},
        ]
        state = {'id': 'sample', 'target': {'actors': actors},
                 'parameters': {'ground_placement': {'actors': [{'actor_id': 'a'}, {'actor_id': 'b'}]}},
                 'axes': {'person_count': 2, 'vehicle_occlusion': 'none'},
                 'background': {'record_index': 0}}
        contrasts = list(planner._contrast_states(state, 'community', white))
        self.assertEqual(contrasts[2]['target']['actors'][0]['source_record']['manifest'], 'new/paired.json')
        self.assertTrue(planner._has_white_contrast(state, 'community', white))
        self.assertFalse(planner._has_white_contrast(state, 'noncommunity', white))
        with patch.object(planner, 'white_back_textures', return_value=white) as loader, \
             patch.object(planner, 'attach_sources'), patch.object(planner, 'validate', return_value=[]):
            selected, _ = evaluation._contrasts({'strata': [state]}, [{}], per_class=0)
        loader.assert_called_once_with()
        self.assertEqual(selected, [])

    def test_multiple_registered_white_back_manifests_extend_coverage(self):
        roles = {'white_back': ['new/extra.json', planner.DIGITAL_WHITE_BACKS]}
        manifests = {
            planner.DIGITAL_WHITE_BACKS: {'records': [
                {'canonical_parent_image': 'a.png', 'foreground': 'old_a.png'}]},
            'new/extra.json': {'records': [
                {'canonical_parent_image': 'a.png', 'foreground': 'new_a.png'},
                {'canonical_parent_image': 'b.png', 'foreground': 'new_b.png'}]},
        }
        with patch('datasets.scripts.common.config.load_task_config', return_value=({}, {'role_records': roles})), \
             patch.object(planner, 'read_json', side_effect=lambda path: manifests[path]), \
             patch.object(planner, 'png_size', return_value=(100, 200)):
            white = planner.white_back_textures()
        self.assertEqual(white['a.png'][1:], ('old_a.png', .5, planner.DIGITAL_WHITE_BACKS))
        self.assertEqual(white['b.png'][1:], ('new_b.png', .5, 'new/extra.json'))


if __name__ == '__main__':
    unittest.main()
