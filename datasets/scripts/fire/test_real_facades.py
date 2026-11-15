"""Configuration and planning contracts without output images or receipts."""
import copy
import unittest
from unittest.mock import patch

from datasets.scripts.fire import compose_real_facades as module


class RealFacadeContracts(unittest.TestCase):
    def test_saved_plan_consumes_current_crops_and_single_artwork(self):
        config, _, rows, _ = module.configured_sources()
        request, seed = module.configured_plan(config, rows, module.source_path('fire', 'production_plan'))
        self.assertEqual(request['budget'], 18)
        self.assertEqual(seed, 431)
        self.assertEqual({s['target']['artwork'] for s in request['strata']},
                         {'datasets/raw/fire/team_assets/image1.png'})

    def test_configured_photos_are_distinct_and_planned_by_source_and_count(self):
        config, manifest, rows, media = module.configured_sources()
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({row['image'] for row in rows}), 3)
        request = module.build_plan(config, rows, 18, 431)
        self.assertEqual(len(request['strata']), 6)
        self.assertEqual({s['target']['count'] for s in request['strata']}, {0, 1})
        allocation = module.plan_production(request, module.ROOT)
        self.assertEqual(allocation['planned_total'], 18)
        self.assertTrue(all('saturation' in s['augmentation'] for s in allocation['strata']))
        self.assertTrue(all(not str(p).startswith(str(module.ROOT/'runs')) for p in media))

    def test_relation_fields_must_match_on_same_relation(self):
        config = {'patches': {'a': []}, 'task': 'fire_sticker', 'split': 'train',
                  'role': 'background', 'status': 'candidate'}
        records = {'records': [{'path': 'x', 'task_use_status': [
            {'source_id': 'a', 'task': 'fire_sticker', 'split': 'test', 'role': 'background', 'status': 'candidate'},
            {'source_id': 'a', 'task': 'meter', 'split': 'train', 'role': 'background', 'status': 'candidate'}]}]}
        with self.assertRaises(ValueError):
            module.selected_sources(config, records)

    def test_crop_candidate_output_cannot_be_training_directory(self):
        with patch('sys.argv', ['entry', '--output', 'datasets/exports/fire/train']), \
                patch.object(module, 'generation_guard') as guard:
            with self.assertRaises(ValueError):
                module.main()
            guard.assert_not_called()


if __name__ == '__main__':
    unittest.main()
