import copy
import subprocess
import tempfile
import unittest
from pathlib import Path

import sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import source_pairing

ROOT = Path(__file__).resolve().parents[4]


def pairing_data():
    return {
        'pairing': {
            'schema_version': 1,
            'status': 'draft',
            'composition_policy': {
                'target_source': 'video_frame_masks',
                'blender_priority': 'background_only',
                'target_components': ['fixture_video_frame_masks'],
                'background_components': ['real_backgrounds'],
            },
            'policy': {
                'parent_classes': ['task_real', 'background'],
                'split_unit': 'parent_group',
                'train_parent_cap': 0.8,
                'val_test_mode': 'reserve_only',
                'derived_inherits_parent_split': True,
                'cross_task_duplicates': 'allowed_with_separate_labels',
                'same_source_splits': 'record_and_report',
            },
            'task_real': [
                {
                    'id': 'tr_capture',
                    'class': 'task_real',
                    'kind': 'field_session',
                    'paths': ['datasets/raw/sign/capture_val'],
                    'evidence': [{'note': '待采'}],
                    'capture_session': 'planned',
                    'capture_id': 'cap_val',
                    'labels': {'status': 'unknown'},
                    'roles': ['direct_target'],
                    'split': {'primary': 'val', 'status': 'proposed', 'action': 'to_capture'},
                    'plan': None,
                }
            ],
            'backgrounds': [],
            'data_sources': [],
            'blender': [],
            'captures': [
                {'id': 'cap_val', 'target_split': 'val', 'status': 'proposed',
                 'save_to': 'datasets/raw/sign/capture_val'},
            ],
            'plan_bindings': {},
        }
    }


class PairingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = pairing_data()

    def tearDown(self):
        self.temporary.cleanup()

    def test_minimal_pairing_passes(self):
        self.assertEqual(
            source_pairing.validate_pairing(self.root, 'sign', self.data), [])

    def test_capture_session_must_be_planned(self):
        self.data['pairing']['task_real'][0]['capture_session'] = 'proven'
        errors = source_pairing.validate_pairing(self.root, 'sign', self.data)
        self.assertTrue(any('capture_session: planned' in error for error in errors))

    def test_capture_id_is_required(self):
        del self.data['pairing']['task_real'][0]['capture_id']
        errors = source_pairing.validate_pairing(self.root, 'sign', self.data)
        self.assertTrue(any('capture_id' in error for error in errors))

    def test_missing_non_capture_path_is_reported(self):
        self.data['pairing']['backgrounds'].append({
            'id': 'bg_missing',
            'class': 'background',
            'kind': 'parent_video',
            'paths': ['datasets/raw/missing'],
            'evidence': [{'note': 'fixture'}],
            'roles': ['composite_background'],
            'split': {'primary': 'train', 'status': 'proposed'},
        })
        errors = source_pairing.validate_pairing(self.root, 'sign', self.data, check_paths=True)
        self.assertTrue(any('路径不存在' in error for error in errors))
        self.assertEqual(
            source_pairing.validate_pairing(self.root, 'sign', self.data, check_paths=False), [])

    def test_composition_share_must_match_allocation(self):
        allocation = {'tasks': {'sign': {'source_mix': {'blender': 0.40}}}}
        errors = source_pairing.validate_pairing(
            self.root, 'sign', self.data, allocation=allocation)
        self.assertTrue(any('blender_share_target' in error for error in errors))

    def test_historical_mix_does_not_constrain_current_pairing(self):
        allocation = {'tasks': {'sign': {'source_mix': {'blender': 0.40},
                                          'source_mix_status': 'historical_reference'}}}
        errors = source_pairing.validate_pairing(
            self.root, 'sign', self.data, allocation=allocation)
        self.assertFalse(any('blender_share_target' in error for error in errors))

    def test_target_components_required(self):
        del self.data['pairing']['composition_policy']['target_components']
        errors = source_pairing.validate_pairing(self.root, 'sign', self.data)
        self.assertTrue(any('target_components' in error for error in errors))

    def test_derived_source_of_planned_capture_must_be_capture(self):
        self.data['pairing']['data_sources'].append({
            'id': 'ds_missing',
            'parent_group': 'tr_capture',
            'path': 'datasets/annotated/sign/capture_val',
            'roles': ['target_foreground'],
            'split': {'primary': 'inherit', 'status': 'confirmed'},
        })
        errors = source_pairing.validate_pairing(self.root, 'sign', self.data)
        self.assertTrue(any('派生源也必须标记 to_capture' in error for error in errors))

    def test_empty_negative_requires_an_empty_scene_declaration(self):
        background = {
            'id': 'bg_candidate',
            'class': 'background',
            'kind': 'parent_video',
            'paths': ['datasets/raw/missing'],
            'evidence': [{'note': 'fixture'}],
            'review': {'state': 'candidate_empty'},
            'roles': ['composite_background', 'empty_negative'],
            'split': {'primary': 'train', 'status': 'proposed'},
        }
        self.data['pairing']['backgrounds'].append(background)
        # This checks a parent-level declaration; actual frame eligibility and
        # placement are determined separately, without requiring human review.
        for state in ('confirmed_empty', 'candidate_empty', 'candidate_complete_negative'):
            with self.subTest(state=state):
                background['review']['state'] = state
                errors = source_pairing.validate_pairing(
                    self.root, 'sign', self.data, check_paths=False)
                self.assertFalse(any('empty_negative 需记录空场判定' in error for error in errors))
        for state in ('unknown', 'contains_target'):
            with self.subTest(state=state):
                background['review']['state'] = state
                errors = source_pairing.validate_pairing(
                    self.root, 'sign', self.data, check_paths=False)
                self.assertTrue(any('empty_negative 需记录空场判定' in error for error in errors))

    def test_unknown_role_is_rejected(self):
        self.data['pairing']['data_sources'].append({
            'id': 'ds_bad_role',
            'parent_group': None,
            'path': 'datasets/raw/x',
            'roles': ['bogus_role'],
            'split': {'primary': 'shared', 'status': 'confirmed'},
        })
        errors = source_pairing.validate_pairing(
            self.root, 'sign', self.data, check_paths=False)
        self.assertTrue(any('角色未登记' in error for error in errors))

    def test_stale_plan_alignment_requires_conflicts(self):
        self.data['pairing']['plan_alignment'] = {
            'status': 'stale', 'note': 'fixture', 'conflicts': []}
        errors = source_pairing.validate_pairing(
            self.root, 'sign', self.data, check_paths=False)
        self.assertTrue(any('stale plan_alignment' in error for error in errors))

    def test_plan_alignment_conflict_parent_must_exist(self):
        self.data['pairing']['plan_alignment'] = {
            'status': 'stale',
            'note': 'fixture',
            'conflicts': [{
                'parent': 'missing_parent',
                'plan': 'datasets/scripts/sign/config/production_plan.json',
                'pairing_split': 'test',
                'plan_use': 'train',
            }],
        }
        errors = source_pairing.validate_pairing(
            self.root, 'sign', self.data, check_paths=False)
        self.assertTrue(any('parent 不存在' in error for error in errors))

    def test_real_workspace_pairing_passes(self):
        errors = source_pairing.validate_all(ROOT)
        self.assertEqual(errors, [])

    def test_scoped_check_ignores_unrelated_task_errors(self):
        for task, data in (('sign', self.data), ('meter', {})):
            config = self.root / 'datasets/scripts' / task / 'config/source.yaml'
            config.parent.mkdir(parents=True)
            config.write_text(yaml.safe_dump(data), encoding='utf-8')
        self.assertEqual(source_pairing.validate_all(self.root, check_paths=False, task='sign'), [])
        self.assertTrue(source_pairing.validate_all(self.root, check_paths=False, task='meter'))
        self.assertTrue(source_pairing.validate_all(self.root, check_paths=False))

    def test_cli_scoped_task_and_unknown_task(self):
        config = self.root / 'datasets/scripts/sign/config/source.yaml'
        config.parent.mkdir(parents=True)
        config.write_text(yaml.safe_dump(self.data), encoding='utf-8')
        script = ROOT / 'lab/scripts/contracts/source_pairing.py'
        scoped = subprocess.run([sys.executable, str(script), '--root', str(self.root),
                                 '--check', '--task', 'sign'], capture_output=True, text=True,
                                encoding='utf-8')
        self.assertEqual(scoped.returncode, 0, scoped.stderr + scoped.stdout)
        invalid = subprocess.run([sys.executable, str(script), '--root', str(self.root),
                                  '--check', '--task', 'unknown'], capture_output=True, text=True,
                                 encoding='utf-8')
        self.assertEqual(invalid.returncode, 2)


if __name__ == '__main__':
    unittest.main()
