"""Validate the explicit background policy without touching production data."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from PIL import Image

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT / 'lab/features/e_bike/training/scripts')]
import filter_v4_rejected_background as subject


def row(split, video, background, state='complete_positive'):
    return {'id': 'item', 'split': split, 'parent_video': video,
            'provenance': {'background_image': background}, 'tasks': {'e_bike': {'state': state}}}


class FilterTest(unittest.TestCase):
    def test_rejects_unwanted_families_and_unknowns(self):
        decide = lambda r: subject.classify(r, {subject.ALLOWED_TRAIN}, {subject.EVAL_VIDEO})
        self.assertEqual(decide(row('train', subject.REJECTED_VIDEO, 'anything')),
                         'rejected_user_background_family')
        self.assertEqual(decide(row('train', subject.FIELD_VIDEO, next(iter(subject.REJECTED_TRAIN)))),
                         'rejected_overhead_train')
        self.assertEqual(decide(row('val', subject.FIELD_VIDEO, next(iter(subject.REJECTED_VAL)))),
                         'rejected_nonvehicle_view_val')
        self.assertEqual(decide(row('train', subject.FIELD_VIDEO, subject.ALLOWED_TRAIN)),
                         'rejected_suspended_composite_family')
        self.assertEqual(decide(row('train', subject.FIELD_VIDEO, subject.ALLOWED_TRAIN, 'complete_negative')),
                         'retained_reviewed_negative')
        self.assertEqual(decide(row('test', subject.EVAL_VIDEO, subject.EVAL_PARENT + 'frame_001.png')),
                         'rejected_suspended_composite_family')
        with self.assertRaises(ValueError):
            decide(row('train', subject.FIELD_VIDEO, 'unknown.png'))

    def test_real_candidates_check_labels_and_diagnostic_sources(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            boxes = [{'class': 'fallen', 'xyxy': [0, 0, 4, 4], 'truncated': False}]
            frames, records = [], []
            for index in range(4):
                parent = f'parent_{index}.png'
                frames.append({'source_frame': parent, 'source_frame_index': index,
                               'output_image_size': [8, 8], 'boxes': boxes})
                for tone in ('original', 'mild', 'moderate'):
                    image, label = f'{index}_{tone}.png', f'{index}_{tone}.txt'
                    Image.new('RGB', (8, 8)).save(root / image)
                    (root / label).write_text('1 .25 .25 .5 .5\n', encoding='utf8')
                    records.append({'image': image, 'label': label, 'parent_image': parent,
                                    'source_video': subject.EVAL_VIDEO, 'source_frame_index': index,
                                    'split': 'train_candidate', 'boxes': boxes, 'tone': tone,
                                    'review': 'AI_visual_reviewed_not_human_verified'})
            review = {'status': 'candidate_review_passed', 'errors': [], 'records': records}
            with mock.patch.object(subject, 'ROOT', root), mock.patch.object(subject, 'source_role',
                                                                             return_value=('unassigned', 'test')):
                self.assertEqual(len(subject.checked_real_rows(review, {'rows': frames}, {})), 12)
                (root / records[0]['label']).write_text('0 .25 .25 .5 .5\n', encoding='utf8')
                with self.assertRaises(ValueError):
                    subject.checked_real_rows(review, {'rows': frames}, {})
            with mock.patch.object(subject, 'ROOT', root), mock.patch.object(subject, 'source_role',
                                                                             return_value=('regression', 'test')):
                with self.assertRaises(ValueError):
                    subject.checked_real_rows(review, {'rows': frames}, {})


if __name__ == '__main__':
    unittest.main()
