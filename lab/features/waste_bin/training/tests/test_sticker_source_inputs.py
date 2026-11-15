"""Read-only source binding and in-memory direct-real plan contracts."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.config import load_task_config
from datasets.scripts.waste_sticker.source_inputs import artwork_catalog, real_source_catalog, select_direct_source
from datasets.scripts.waste_sticker.package_real_stickers import prepare_inputs


class StickerSourceInputTests(unittest.TestCase):
    def test_designated_photo_and_numeric_frame_intervals_are_consumed(self):
        _, config = load_task_config('waste_sticker')
        catalog = real_source_catalog(config)
        self.assertEqual(len(catalog), 43)
        self.assertEqual(catalog[0]['image'], config['real_sticker_source'])
        self.assertEqual(sum(row['stationary'] for row in catalog), 29)
        self.assertTrue(all(row['exists'] for row in catalog))
        self.assertEqual(sum(row['annotation'] is not None for row in catalog), 3)
        changed = copy.deepcopy(config)
        changed['frame_indices']['last'] -= 1
        with self.assertRaisesRegex(ValueError, 'interval'):
            real_source_catalog(changed)

    def test_artwork_records_bind_all_24_class_ids(self):
        self.assertEqual(set(artwork_catalog()), set(range(24)))

    def test_p22_exclusion_partial_frame_and_formal_candidate_are_distinct(self):
        catalog = real_source_catalog()
        with self.assertRaisesRegex(ValueError, 'excluded'):
            select_direct_source(catalog, catalog[0]['image'], 'train')
        last = next(row for row in catalog if row['frame_number'] == 2044)
        with self.assertRaisesRegex(ValueError, 'partial'):
            select_direct_source(catalog, last['image'], 'train')
        first = next(row for row in catalog if row['frame_number'] == 2016)
        self.assertEqual(select_direct_source(catalog, first['image'], 'train')['record_index'], 1)
        self.assertEqual(select_direct_source(catalog, first['image'], 'train', formal=True)['annotation']['annotation_level'], 'candidate')

    def test_direct_plan_binds_annotation_and_parent_without_writing(self):
        catalog = real_source_catalog()
        row = next(row for row in catalog if row['frame_number'] == 2016)
        state = {'id': 'direct', 'conditions': [], 'max_count': 1,
                 'background': {'image': row['image']}, 'target': {'kind': 'existing_stickers'},
                 'labels': {'format': 'yolo_detection', 'annotation_manifest': row['annotation_manifest']},
                 'parameters': {'annotation_record_index': row['record_index']},
                 'augmentation': {'value_gain': 1.0, 'saturation_gain': .9},
                 'sources': [row['image'], row['parent']]}
        plan = {'model_key': 'waste_sticker', 'target_split': 'train',
                'constraints': {'labels': 'recorded_full_frame_stickers', 'resize': 'none'}, 'strata': [state]}
        self.assertEqual(set(prepare_inputs(plan, catalog)), {'direct'})
        state['sources'] = [row['image']]
        with self.assertRaisesRegex(ValueError, 'parent'):
            prepare_inputs(plan, catalog)


if __name__ == '__main__':
    unittest.main()
