"""Legacy catalog candidates must not become complete labels by citation alone."""

import unittest

from datasets.scripts.common.backfill_frame_environment import _background_label


class LegacyBackgroundLabelTests(unittest.TestCase):
    def test_manifest_citation_is_not_a_negative_visual_verdict(self):
        entry = {'label_state': 'candidate_complete_negative',
                 'evidence': {'manifest': 'fixture_manifest.json', 'record_index': 0}}
        label = _background_label('e_bike', entry)
        self.assertEqual((label['presence'], label['completeness']), ('unknown', 'unknown'))
        entry['review_verdict'] = 'ready'
        label = _background_label('e_bike', entry)
        self.assertEqual((label['presence'], label['completeness']), ('absent', 'complete'))

    def test_crop_plan_is_not_a_complete_crop_label(self):
        entry = {'label_state': 'candidate_complete_negative_in_crop',
                 'geometry': {'crop_xyxy': [10, 20, 50, 60]},
                 'evidence': {'manifest': 'production_plan.json', 'record_index': 2}}
        label = _background_label('fire', entry)
        self.assertEqual(label['scope'], {'kind': 'crop', 'crop_xyxy': [10, 20, 50, 60]})
        self.assertEqual((label['presence'], label['completeness']), ('unknown', 'unknown'))
        entry['review_verdict'] = 'ready'
        label = _background_label('fire', entry)
        self.assertEqual((label['presence'], label['completeness']), ('absent', 'complete'))


if __name__ == '__main__':
    unittest.main()
