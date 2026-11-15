"""In-memory contract checks; no image files or production state are created."""
import copy
import contextlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'lab/features/building_name_paper/training/scripts'))
import compose_rooftop_papers as rooftop


def candidate(image='datasets/raw/fire/one.png', group='building-a'):
    return {'image': image, 'source_group': group, 'parent_source': 'datasets/raw/fire/parent.mp4',
            'generation_method': 'provided_video_frame', 'image_size': [100, 80],
            'coverage': 'unknown', 'coverage_evidence': 'source observation pending',
            'existing_papers': [{'building_id': 'B', 'box_xyxy': [5, 35, 25, 45],
                                 'ocr_status': 'unknown', 'text': None, 'evidence': 'candidate 1'}],
            'regions': [{'slot_id': 'roof-a', 'building_id': 'A', 'quad_xy':
                         [[30, 10], [60, 10], [60, 20], [30, 20]], 'evidence': 'candidate quad'}]}


def request():
    return {'schema_version': 1, 'model_key': 'building_name_paper', 'target_split': 'train',
            'purpose': 'candidate composition', 'budget': 2, 'uncertainty_weight': 0,
            'independence_basis': 'source group is a declared sampling unit',
            'constraints': copy.deepcopy(rooftop.CONSTRAINTS),
            'strata': [{'id': 'front', 'failure_event': 'paper_or_background_label_uncertainty',
                        'conditions': ['rooftop_paper', 'no_added_occlusion'],
                        'min_count': 0, 'max_count': 2, 'prior': {'alpha': 1, 'beta': 1},
                        'exposure_weight': 1, 'failure_cost': 1, 'observations': [],
                        'background': {'sampling': 'uniform_group_then_image', 'candidates': [candidate()]},
                        'target': {'kind': 'building_name_paper', 'papers': [{'slot_id': 'roof-a', 'text': '晨光大厦'}]},
                        'labels': copy.deepcopy(rooftop.LABELS), 'augmentation': {'mode': 'none'},
                        'sources': ['real_scene_background', 'name_font_asset'], 'parameters': {'seed': 7}}]}


class RooftopContractTest(unittest.TestCase):
    def test_allocation_accepts_explicit_candidate_plan(self):
        plan = request()
        rooftop.validate_request(plan)
        result = rooftop.plan_production(plan, ROOT)
        self.assertEqual(result['planned_total'], 2)
        self.assertEqual(result['strata'][0]['background']['candidates'][0]['source_group'], 'building-a')

    def test_existing_unknown_ocr_survives_and_generated_string_is_recorded(self):
        source = candidate()
        pixels = np.zeros((80, 100, 3), dtype=np.uint8)
        with mock.patch.object(rooftop, 'name_paper', return_value=object()), \
                mock.patch.object(rooftop, 'paste', return_value=[30, 10, 61, 21]):
            _, records = rooftop.compose_one(pixels, source, request()['strata'][0]['target']['papers'])
        self.assertEqual(records[0]['box_xyxy'], [5, 35, 25, 45])
        self.assertEqual(records[0]['ocr_status'], 'unknown')
        self.assertIsNone(records[0]['text'])
        self.assertEqual(records[1]['box_xyxy'], [30, 10, 61, 21])
        self.assertEqual(records[1]['generated_text'], '晨光大厦')
        self.assertEqual(records[1]['ocr_status'], 'generated_string')
        self.assertEqual(records[1]['text'], records[1]['generated_text'])

    def test_unknown_text_cannot_be_silently_asserted(self):
        source = candidate()
        source['existing_papers'][0]['text'] = '猜测'
        with self.assertRaisesRegex(ValueError, 'unknown OCR text'):
            rooftop.existing_annotation(source['existing_papers'][0], source['image_size'])

    def test_existing_known_text_stays_separate_from_generated_text(self):
        paper = candidate()['existing_papers'][0]
        paper.update(ocr_status='known', text='晨光大厦')
        result = rooftop.existing_annotation(paper, [100, 80])
        self.assertEqual(result['text'], '晨光大厦')
        self.assertEqual(result['ocr_status'], 'known')
        self.assertIsNone(result['generated_text'])

    def test_mirrored_invalid_or_overlapping_regions_are_rejected(self):
        for quad in ([[60, 10], [30, 10], [30, 20], [60, 20]],
                     [[30, 10], [60, 20], [60, 10], [30, 20]],
                     [[5, 35], [25, 35], [25, 45], [5, 45]]):
            plan = request()
            plan['strata'][0]['background']['candidates'][0]['regions'][0]['quad_xy'] = quad
            with self.subTest(quad=quad), self.assertRaises(ValueError):
                rooftop.validate_request(plan)

    def test_source_group_selection_is_seeded_and_order_independent(self):
        options = [candidate('datasets/raw/fire/b.png', 'b'), candidate(),
                   candidate('datasets/raw/fire/c.png', 'b')]
        first, second = random.Random(91), random.Random(91)
        self.assertEqual([rooftop.choose_background(options, first)['image'] for _ in range(20)],
                         [rooftop.choose_background(list(reversed(options)), second)['image'] for _ in range(20)])

    def test_wrong_dimensions_and_outside_workspace_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            rooftop.compose_one(np.zeros((20, 100, 3), dtype=np.uint8), candidate(), [])
        plan = request()
        plan['strata'][0]['background']['candidates'][0]['image'] = '../outside.png'
        with self.assertRaisesRegex(ValueError, 'workspace-relative'):
            rooftop.validate_request(plan)

    def test_background_parent_uses_task_source_configuration(self):
        source = {'background_selection': 'production_plan_candidates',
                  'background_parent_roots': ['datasets/raw/meter']}
        with mock.patch.object(rooftop, 'load_task_config', return_value=({}, source)):
            with self.assertRaisesRegex(ValueError, 'background_parent_roots'):
                rooftop.validate_request(request())
        source['background_parent_roots'] = ['datasets/raw']
        with mock.patch.object(rooftop, 'load_task_config', return_value=({}, source)):
            self.assertEqual(rooftop.validate_request(request()), request())

    def test_main_declares_background_and_parent_to_guard(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as tmp:
            root = Path(tmp).resolve()
            (root / 'plan.json').write_text(json.dumps(request(), ensure_ascii=False), encoding='utf8')
            plan = rooftop.plan_production(request(), root)
            run = mock.Mock(reused=True, plan=plan)
            guard = mock.MagicMock()
            guard.return_value.__enter__.return_value = run
            argv = ['compose_rooftop_papers.py', '--plan', 'plan.json', '--output',
                    'datasets/samples/building_name_paper']
            with mock.patch.object(rooftop, 'ROOT', root), mock.patch.object(rooftop, 'sample_dir', return_value=root/'datasets/samples/building_name_paper'), mock.patch.object(rooftop, 'generation_guard', guard), \
                    mock.patch.object(rooftop, '_check_font_text'), mock.patch.object(sys, 'argv', argv), \
                    contextlib.redirect_stdout(io.StringIO()):
                rooftop.main()
            inputs = guard.call_args.args[1]
            self.assertIn(root / 'datasets/raw/fire/one.png', inputs)
            self.assertIn(root / 'datasets/raw/fire/parent.mp4', inputs)
            self.assertEqual(guard.call_args.kwargs['production_plan'], request())
            self.assertEqual(guard.call_args.kwargs['sample_output_paths'],
                             [root / 'datasets/samples/building_name_paper/images'])


if __name__ == '__main__':
    unittest.main()
