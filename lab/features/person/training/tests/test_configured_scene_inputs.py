"""Read-only role resolution and multi-actor dispatch contracts."""
import json
from pathlib import Path
import unittest
from unittest import mock
import numpy as np
from datasets.scripts.person import planned_sources as sources
from datasets.scripts.person import compose_planned_ground_person as entry


class ConfiguredSceneInputsTest(unittest.TestCase):
    def test_selected_role_record_is_not_an_arbitrary_manifest(self):
        config = {'role_records': {'vehicle_occluder': 'datasets/annotated/person/vehicles.json'}}
        ref = {'manifest': 'datasets/annotated/person/vehicles.json', 'record_index': 1}
        with mock.patch.object(Path, 'is_file', return_value=True), \
             mock.patch.object(Path, 'read_text', return_value=json.dumps({'records':[{'class':'fallen'},{'class':'upright'}]})):
            _, _, row = sources.role_record(config, 'vehicle_occluder', ref)
        self.assertEqual(row['class'], 'upright')
        with self.assertRaisesRegex(ValueError, 'not configured'):
            sources.role_record(config, 'vehicle_occluder', dict(ref, manifest='unselected.json'))

    def test_white_back_role_accepts_only_registered_relocation(self):
        former = 'datasets/raw/person/targets/digital/white_back/manifest.json'
        current = 'datasets/annotated/person/white_back/manifest.json'
        config = {'role_records': {'white_back': current}}
        reference = {'manifest': former, 'record_index': 0}
        record = {'foreground': 'white_back.png'}
        with mock.patch.object(sources, 'read_source_history', return_value=((current, former),)), \
             mock.patch.object(Path, 'is_file', return_value=True), \
             mock.patch.object(Path, 'read_text', return_value=json.dumps({'records': [record]})):
            path, _, row = sources.role_record(config, 'white_back', reference)
            self.assertEqual(path, sources.ROOT / current)
            self.assertEqual(row, record)
            with self.assertRaisesRegex(ValueError, 'not configured'):
                sources.role_record(config, 'white_back', dict(reference, manifest='other.json'))

    def test_multi_actor_state_dispatch_and_optional_quota(self):
        states = [{'id':'a', 'target':{'actors':[]}, 'background':{'image':'a'}, 'parameters':{}},
                  {'id':'b', 'target':{'actors':[]}, 'background':{'image':'b'}, 'parameters':{}}]
        plan = {'budget':1, 'strata':states, 'constraints':{'foreground':'planned_actors'}}
        with mock.patch.object(entry, 'prepare_scene', side_effect=lambda d, **kwargs: {'state':d['strata'][0]}) as prepare:
            self.assertEqual(set(entry.prepare_batch(plan)), {'a','b'})
        self.assertEqual(prepare.call_count, 2)

    def test_plain_background_is_not_implicitly_negative(self):
        plan = {'model_key':'person', 'target_split':'train', 'constraints':{
            'foreground':'planned_actors', 'identity':'front_visible', 'background':'complete_existing_or_empty'},
            'strata':[{'min_count':0,'max_count':1,'background':{'role':'shared_background','image':'image.png',
                      'manifest':'shared.json','record_index':0},'parameters':{}}]}
        with mock.patch.object(sources, 'load_task_config', return_value=({}, {'role_records': {}})), \
             mock.patch.object(sources, 'role_record', return_value=(sources.ROOT/'shared.json',{}, {'image':'image.png'})), \
             mock.patch.object(sources, 'file', return_value=sources.ROOT/'image.png'):
            with self.assertRaisesRegex(ValueError, 'not an accepted candidate'):
                sources.prepare_scene(plan)


if __name__ == '__main__':
    unittest.main()
