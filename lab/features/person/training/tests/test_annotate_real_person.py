"""In-memory contracts for configured camera annotations; no media is written."""
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock
from datasets.scripts.person import annotate_real_person as subject


class RealAnnotationPlanTest(unittest.TestCase):
    def declaration(self, folder='社区人员'):
        source = 'Archive/Assets/captures/car_real_capture_20261112/frame.png'
        ref = f'Archive/Assets/working_sources/digital/person/team_assets/{folder}/1.png'
        state = {'id': 'one', 'failure_event': 'candidate', 'min_count': 1, 'max_count': 1,
                 'prior': {'alpha': 1, 'beta': 1}, 'exposure_weight': 1, 'failure_cost': 1,
                 'conditions': [], 'background': {'source_image': source},
                 'target': {'instances': [{'class_id': 0, 'xyxy': [10,20,50,100], 'reference_asset': ref}]},
                 'labels': {'person': 'yolo_visible_boxes', 'white_back': 'no_boxes'},
                 'augmentation': {'kind': 'none'},
                 'parameters': {'image_size': [640,480], 'image_operation': 'byte_copy',
                                'annotation': 'ai_visual_candidate'}, 'sources': sorted([source, ref])}
        return {'model_key': 'person', 'target_split': 'train', 'budget': 1,
                'constraints': {'scope': 'near_frontal', 'white_back': 'no_person_boxes',
                                'review': 'ai_candidate_not_human_accepted'}, 'strata': [state]}

    def test_recorded_image_label_and_references_follow_the_plan(self):
        plan = self.declaration()
        run = SimpleNamespace(reused=False, plan={'strata': [dict(plan['strata'][0], planned_count=1)]}, record_sample=mock.Mock())
        @contextmanager
        def fake_guard(*args, **kwargs):
            self.assertEqual(kwargs['production_plan'], plan)
            self.assertIn(subject.ROOT/'datasets/scripts/person/config/source.yaml', kwargs['procedural_inputs'])
            yield run
        image = mock.MagicMock()
        image.__enter__.return_value.size = (640,480)
        with mock.patch.object(Path, 'is_file', return_value=True), \
             mock.patch.object(Path, 'exists', return_value=False), \
             mock.patch.object(Path, 'read_text', return_value=json.dumps(plan)), \
             mock.patch.object(subject.Image, 'open', return_value=image), \
             mock.patch.object(subject, 'source_path', side_effect=lambda task,key: subject.ROOT/('Archive/Assets/working_sources/digital/person/team_assets' if key == 'person_artwork_root' else 'Archive/Assets/captures/car_real_capture_20261112')), \
             mock.patch.object(subject, 'generation_guard', fake_guard), \
             mock.patch.object(subject.shutil, 'copy2'), mock.patch.object(subject.shutil, 'copyfile'), \
             mock.patch.object(Path, 'mkdir'), mock.patch.object(Path, 'write_text') as write:
            result = subject.generate('fixture.json', 'datasets/samples/person')
        self.assertEqual(len(result['records']), 1)
        self.assertEqual(run.record_sample.call_args.args, ('one',))
        self.assertTrue(any('0 0.04687500 0.12500000 0.06250000 0.16666667' in str(c) for c in write.call_args_list))

    def test_wrong_class_artwork_is_rejected(self):
        with mock.patch.object(Path, 'is_file', return_value=True):
            with self.assertRaisesRegex(ValueError, 'outside its declared area'):
                subject.prepare(self.declaration('非社区人员'))


if __name__ == '__main__':
    unittest.main()
