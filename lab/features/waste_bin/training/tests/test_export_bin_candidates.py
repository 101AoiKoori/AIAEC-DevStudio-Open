import importlib.util
import tempfile
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
SCRIPT = ROOT / 'datasets/scripts/waste_bin_state/export_bin_candidates.py'
spec = importlib.util.spec_from_file_location('export_bin_candidates', SCRIPT)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)
from datasets.scripts.waste_bin_state import audit_bin_renders as auditor
from datasets.scripts.waste_bin_state.bin_render_contract import STATE_CONTRACT, attach_state_evidence, decode_mask, measure_visibility


def sample():
    return {'image': 'images/a.png', 'candidate_label': 'labels_candidate/a.txt',
            'labels': [{'bin': name, 'label_eligible': True, 'in_frame': True,
                        'rejection_reasons': []} for name in sorted(exporter.BINS)]}


class ExportBinCandidatesTest(unittest.TestCase):
    def test_selected_single_bin_and_empty_scene_do_not_require_four_bins(self):
        single = sample()
        single['labels'] = single['labels'][:1]
        single['visible_bins'] = [single['labels'][0]['bin']]
        empty = sample()
        empty['labels'] = []
        empty['visible_bins'] = []
        included, excluded = exporter.plan_samples([single, empty])
        self.assertEqual(included, [single, empty])
        self.assertEqual(excluded, [])

    def test_only_four_known_eligible_lids_with_candidate_are_included(self):
        valid = sample()
        unknown = sample()
        del unknown['labels'][0]['label_eligible']
        clipped = sample()
        clipped['labels'][1]['in_frame'] = False
        clipped['labels'][1]['rejection_reasons'] = ['clipped_or_behind_camera']
        missing = sample()
        missing['candidate_label'] = None
        incomplete = sample()
        incomplete['labels'].pop()
        included, excluded = exporter.plan_samples([valid, unknown, clipped, missing, incomplete])
        self.assertEqual(included, [valid])
        self.assertEqual(len(excluded), 4)
        self.assertTrue(any('eligibility_unknown' in x for x in excluded[0]['reasons']))
        self.assertTrue(any('clipped_or_behind_camera' in x for x in excluded[1]['reasons']))
        self.assertIn('no_candidate_label', excluded[2]['reasons'])
        self.assertIn('missing_or_invalid_lid_records', excluded[3]['reasons'])

    def test_pairs_require_matching_names_four_finite_boxes_and_local_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / 'images').mkdir()
            (root / 'labels_candidate').mkdir()
            (root / 'images/a.png').write_bytes(b'image fixture')
            label = root / 'labels_candidate/a.txt'
            label.write_text('0 0.5 0.5 0.1 0.1\n' * 4, encoding='utf8')
            exporter.validate_pair(root, sample())
            label.write_text('0 nan 0.5 0.1 0.1\n' * 4, encoding='utf8')
            with self.assertRaises(ValueError):
                exporter.validate_pair(root, sample())
            label.write_text('0 0.5 0.5 0.1 0.1\n' * 3, encoding='utf8')
            with self.assertRaises(ValueError):
                exporter.validate_pair(root, sample())
            with self.assertRaises(ValueError):
                exporter.contained_file(root, '../outside.png')

    def test_auditor_rechecks_both_target_and_state_masks(self):
        rgba = np.zeros((8, 8, 4), dtype=float)
        rgba[2:6, 2:6] = [1, 0, 0, 1]
        mask = decode_mask(rgba, 'blue')
        target = measure_visibility(mask, mask, True, min_pixels=1)
        evidence = measure_visibility(mask, mask, True, min_pixels=1)
        evidence['isolated_mask'] = 'state_solo.png'
        item = attach_state_evidence(target, evidence, 'open')
        item.update({'bin': 'blue', 'isolated_mask': 'solo.png'})
        data = {'label_contract_id': STATE_CONTRACT, 'min_visibility': .7, 'min_visible_pixels': 1}
        entry = {'image': 'a.png', 'mask_visible': 'visible.png',
                 'mask_state_visible': 'state.png', 'labels': [item]}
        with mock.patch.object(auditor, 'read_mask', return_value=rgba):
            self.assertEqual(auditor.audit_masks(Path('.'), entry, data), [])
            item['state_evidence']['visible_pixels'] = 999
            self.assertTrue(auditor.audit_masks(Path('.'), entry, data))


if __name__ == '__main__':
    unittest.main()
