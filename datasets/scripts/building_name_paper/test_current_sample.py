"""Verify the live recipe uses the existing task entry points and source decisions."""
import unittest

from datasets.scripts.building_name_paper import build_current_sample as sample
from datasets.scripts.common.config import load_task_config
from datasets.scripts.common import paths


class CurrentSampleTests(unittest.TestCase):
    def test_persistent_render_evidence_is_scoped_to_this_task(self):
        code = ['datasets/scripts/building_name_paper/build_current_sample.py']
        configs = ['datasets/scripts/building_name_paper/config/dataset.yaml',
                   'datasets/scripts/building_name_paper/config/source.yaml']
        output = paths.building_name_sample_work_dir().relative_to(paths.WORKSPACE).as_posix()
        self.assertEqual(paths.validate_guarded_outputs(code, [output + '/blender'], configs), sample.TASK)
        with self.assertRaises(ValueError):
            paths.validate_guarded_outputs(code, ['datasets/annotated/fire/other_task'], configs)

    def test_real_background_budget_expands_repeated_states_per_source(self):
        _, source = load_task_config(sample.TASK)
        request = sample.composite_plan(source['sample_recipe'])
        candidates = [state['background']['candidates'][0] for state in request['strata']]
        frames = {row['source_review']['frame_id'] for row in candidates}
        self.assertEqual(len(frames), 2)
        self.assertEqual(request['budget'], 2 * source['sample_recipe']['composites_per_background'])
        self.assertEqual(len({state['id'] for state in request['strata']}), request['budget'])
        self.assertEqual({state['augmentation']['mode'] for state in request['strata']},
                         {'none', 'scene_bounded_v1'})

    def test_direct_sources_keep_their_original_splits_and_complete_boxes(self):
        _, source = load_task_config(sample.TASK)
        records, _, _ = sample.direct_records(source['sample_recipe'])
        extras = {row[2]['frame_environment']['record_id'] for row in records
                  if row[2]['source_split'] == 'unassigned'}
        self.assertEqual(extras, set(source['sample_recipe']['additional_direct_records']))
        self.assertEqual({row[2]['source_split'] for row in records}, {'train', 'unassigned'})
        self.assertEqual(len({row[2]['frame_environment']['frame_id'] for row in records}), len(records))
