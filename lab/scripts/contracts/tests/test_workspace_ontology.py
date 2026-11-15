import copy
import tempfile
import unittest
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import workspace_ontology

ROOT = Path(__file__).resolve().parents[4]


class OntologyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ontology = workspace_ontology.load_ontology()
        cls.body = workspace_ontology.render_doc(cls.ontology)

    def test_ontology_loads_and_has_classes(self):
        self.assertIn('artifact', self.ontology['classes'])
        self.assertIn('plan_draft', self.ontology['classes'])
        self.assertIn('production_plan', self.ontology['classes'])

    def test_classify_key_paths(self):
        cases = {
            'config/workspace_ontology.yaml': 'ontology_contract',
            'config/workspace_layout.yaml': 'layout_contract',
            'config/task_targets.yaml': 'design_contract',
            'config/combination_axes.yaml': 'design_contract',
            'config/full_train_allocation.yaml': 'plan_draft',
            'docs/FULL_TRAIN_PLAN_DESIGN.md': 'generated_doc',
            'docs/DATASET_DESIGN.md': 'project_doc',
            'datasets/scripts/person/config/production_plan.json': 'production_plan',
            'datasets/scripts/person/config/blender_plan.json': 'generation_plan',
            'datasets/scripts/person/config/review_selection.json': 'review_decision',
            'datasets/scripts/person/config/source.yaml': 'task_config',
            'datasets/raw/site_capture/x.png': 'raw_media',
            'datasets/annotated/common/review/person/b/metrics.jsonl': 'review_observation',
            'datasets/exports/person/train/x.jpg': 'dataset_export',
            'datasets/samples/person/x.jpg': 'sample_set',
            'models/weights/person/x.pt': 'model_weight',
            'ws/src/aec_vision/models/plate.pt': 'deployed_model',
            'ws/src/aec_vision/package.xml': 'deploy_package',
            'lab/features/person/training/scripts/x.py': 'experiment_code',
            'lab/scripts/contracts/workspace_ontology.py': 'repo_tool',
            '.agents/skills/foo/SKILL.md': 'project_doc',
            '.agents/skills/foo/scripts/x.py': 'skill_script',
            'runs/person/x/metadata.json': 'training_run',
            'tmp/x': 'temp_artifact',
            'Archive/technical/x': 'technical_archive',
            'Archive/Assets/x': 'external_original',
        }
        for path, expected in cases.items():
            actual, reason = workspace_ontology.classify(path, self.ontology)
            self.assertEqual(actual, expected, '{} -> {} ({})'.format(path, actual, reason))

    def test_generated_doc_is_current(self):
        doc = ROOT / 'docs/WORKSPACE_ONTOLOGY.md'
        self.assertTrue(doc.is_file())
        self.assertEqual(doc.read_text(encoding='utf-8'), self.body)

    def test_workspace_check_has_no_errors(self):
        errors, warnings = workspace_ontology.check_workspace(ROOT, self.ontology)
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

    def test_missing_required_objects_are_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            errors, _ = workspace_ontology.check_workspace(Path(temporary), self.ontology)
            self.assertTrue(errors)

    def test_parent_cycle_is_rejected(self):
        data = copy.deepcopy(self.ontology)
        data['classes']['artifact']['parent'] = 'temp_artifact'
        with self.assertRaises(ValueError):
            workspace_ontology.validate_ontology(data)

    def test_duplicate_artifact_is_rejected(self):
        data = copy.deepcopy(self.ontology)
        data['artifacts'].append({'path': 'config/workspace_layout.yaml', 'class': 'project_doc'})
        with self.assertRaises(ValueError):
            workspace_ontology.validate_ontology(data)

    def test_unknown_relation_is_rejected(self):
        data = copy.deepcopy(self.ontology)
        data['classes']['plan_draft']['relations'].append('not_a_relation')
        with self.assertRaises(ValueError):
            workspace_ontology.validate_ontology(data)

    def test_unknown_path_is_unclassified(self):
        actual, _ = workspace_ontology.classify('config/not_registered.yaml', self.ontology)
        self.assertIsNone(actual)

    def test_doc_contains_required_sections(self):
        for text in ('## 类层次', '## 关系', '## 版本策略', '## 关键对象映射', '## 新增对象'):
            self.assertIn(text, self.body)


if __name__ == '__main__':
    unittest.main()
