"""Check task-scoped selection; no image generation or data writes."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import plan_usage_batch as planner  # noqa: E402


class UsageSelectionTest(unittest.TestCase):
    def test_ready_relation_must_match_task_split_and_role(self):
        records = [
            {'image': 'bike.png', 'task_use_status': [
                {'task': 'e_bike', 'split': 'train', 'role': 'target_foreground', 'status': 'blocked', 'posture': 'fallen'},
                {'task': 'person', 'split': 'train', 'role': 'target_foreground', 'status': 'ready', 'posture': 'fallen'}]},
            {'image': 'upright.png', 'task_use_status': [
                {'task': 'e_bike', 'split': 'train', 'role': 'target_foreground', 'status': 'ready', 'posture': 'upright'}]},
        ]
        entries = [{'path': path, 'class': cls, 'source_frame': 'frame', 'source_video': 'video'}
                   for path, cls in [('bike.png', 'fallen'), ('upright.png', 'upright')]]
        with patch.object(Path, 'is_file', return_value=True), patch.object(
                planner.ground, 'source_role', return_value=('unknown', 'incomplete')):
            _, fg = planner.eligible_sources({'records': records}, [], entries, {})
        self.assertEqual([], fg['fallen'])
        self.assertEqual(['upright.png'], [row['path'] for row in fg['upright']])

    def test_background_candidate_needs_specific_negative_evidence(self):
        bg = {'id': 'a', 'image': 'bg.png', 'parent_image': 'frame', 'source_video': 'video',
              'task_state': 'candidate_complete_negative', 'v4_input_review_20261114': {'review': True}}
        rel = {'task': 'e_bike', 'split': 'train', 'role': 'background', 'status': 'candidate',
               'label_status': '整幅裁切图目视未见电动车'}
        usage = {'records': [{'image': 'bg.png', 'task_use_status': [rel]}]}
        with patch.object(Path, 'is_file', return_value=True), patch.object(
                planner.ground, 'source_role', return_value=('unknown', 'incomplete')):
            self.assertEqual([bg], planner.eligible_sources(usage, [bg], [], {})[0])
            rel['label_status'] = '未核验完整画面'
            self.assertEqual([], planner.eligible_sources(usage, [bg], [], {})[0])


if __name__ == '__main__':
    unittest.main()
