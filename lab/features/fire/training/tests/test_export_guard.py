"""Check that export never writes after guard rejection or reuse."""
import json
import copy
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'lab/features/fire/training/scripts'))
import export_fire_labels as module


class ExportGuardTest(unittest.TestCase):
    def test_new_signal_candidates_require_inspected_source_and_plan(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            parent = root / 'datasets/exports/authoritative/fire_sticker'
            parent.mkdir(parents=True)
            source, output = parent / 'plan.json', parent / 'output'
            image_name = 'datasets/raw/car_real_capture_20261112/P01_TL1.png'
            image = root / image_name
            image.parent.mkdir(parents=True)
            image.write_bytes(b'original image fixture')
            state = {'id': 'signal_P01_TL1', 'failure_event': 'unrecognized_fire_or_context_uncertainty',
                     'min_count': 1, 'max_count': 1, 'prior': {'alpha': 1, 'beta': 1},
                     'exposure_weight': 1, 'failure_cost': 1,
                     'conditions': ['traffic_signal', 'no_visible_building', 'no_recognizable_fire'],
                     'background': {'image': image_name, 'route': 'signal_without_building'},
                     'target': {'class': 'fire_sticker', 'visible_count': 0},
                     'labels': {'format': 'yolo_v8', 'class_id': 0, 'status': 'ai_visual_candidate_not_human_accepted'},
                     'augmentation': {'method': 'none_original_capture'},
                     'sources': ['car_real_capture_20261112'], 'parameters': {'image': image_name}}
            plan = {'schema_version': 1, 'model_key': 'fire_sticker', 'target_split': 'train',
                    'purpose': 'fixture', 'budget': 1, 'uncertainty_weight': 0, 'independence_basis': 'fixture',
                    'constraints': {'route_context': 'signal_without_building',
                        'label_policy': 'no_recognizable_fire_in_visible_frame_candidate', 'human_acceptance': 'not_confirmed'},
                    'strata': [state]}
            source.write_text(json.dumps(plan), encoding='utf8')
            allocated = {'planned_total': 1, 'strata': [{**state, 'planned_count': 1}]}
            recorded = []
            @contextmanager
            def guard(*args, **kwargs):
                yield type('Run', (), {'reused': False, 'plan': allocated, 'source_refs': {image_name: []},
                                      'record_sample': lambda _, *a, **kw: recorded.append((a, kw))})()
            with patch.object(module, 'ROOT', root), patch.object(module, 'workspace_path', side_effect=lambda p: root / p), \
                    patch.object(module, 'plan_production', return_value=allocated), patch.object(module, 'generation_guard', guard):
                self.assertEqual(module.export_signal_negatives(source, output), 1)
                self.assertEqual((output / 'images/train/signal_P01_TL1.png').read_bytes(), image.read_bytes())
                self.assertEqual((output / 'labels/train/signal_P01_TL1.txt').read_text(), '')
                self.assertEqual(recorded[0][1]['actual']['annotation_state'], 'candidate_complete_negative_not_human_accepted')
                for change in ('augmentation', 'nonexhaustive', 'uninspected'):
                    bad = copy.deepcopy(plan)
                    if change == 'augmentation': bad['strata'][0]['augmentation'] = {'method': 'unsupported'}
                    if change == 'nonexhaustive': bad['strata'][0]['conditions'] = ['traffic_signal', 'fire_unknown']
                    if change == 'uninspected': bad['strata'][0]['parameters']['image'] = 'datasets/raw/fire/unknown.png'
                    source.write_text(json.dumps(bad), encoding='utf8')
                    with self.assertRaises(ValueError): module.export_signal_negatives(source, parent / 'bad')
                    self.assertFalse((parent / 'bad').exists())

    def test_existing_sources_records_contain_image_refs(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            source, media, output = root / 'input.json', root / 'input.png', root / 'result'
            record = {'image': 'input.png', 'reviewed': True, 'split': 'train', 'fires': [], 'names': []}
            source.write_text(json.dumps([record]), encoding='utf8')
            cv2.imencode('.png', np.zeros((32, 32, 3), dtype=np.uint8))[1].tofile(str(media))
            refs = [{'manifest': 'existing.json', 'record_index': 3, 'relation': 'image'}]

            @contextmanager
            def allowed(*args, **kwargs):
                yield type('Run', (), {'reused': False,
                                      'source_refs': {media.relative_to(ROOT).as_posix(): refs}})()

            with patch.object(module, 'workspace_path', return_value=media), \
                    patch.object(module, 'generation_guard', allowed):
                self.assertEqual(module.export(source, output), 1)
            data = json.loads((output / 'sources.json').read_text(encoding='utf8'))
            self.assertEqual(data, [{**record, 'source_refs': refs}])
            self.assertFalse((output / 'source_refs.json').exists())

    def test_reuse_and_rejection_skip_export_body(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            source, output = root / 'input.json', root / 'output'
            source.write_text(json.dumps([{'image': 'input.png'}]), encoding='utf8')
            media = root / 'input.png'
            media.write_bytes(b'fixture')
            @contextmanager
            def reused(*args, **kwargs):
                yield type('Run', (), {'reused': True})()
            with patch.object(module, 'workspace_path', return_value=media), \
                    patch.object(module, '_export_impl') as body, \
                    patch.object(module, 'generation_guard', reused):
                self.assertEqual(module.export(source, output), 1)
                body.assert_not_called()
                self.assertFalse(output.exists())

    def test_output_scoped_operation_and_real_refs(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            source, media = root / 'input.json', root / 'input.png'
            source.write_text(json.dumps([{'image': 'input.png'}]), encoding='utf8')
            media.write_bytes(b'fixture')
            operations = []
            ref_list = [{'manifest': 'input.json', 'record_index': 0, 'relation': 'image'}]
            refs = {media.relative_to(ROOT).as_posix(): ref_list}

            @contextmanager
            def allowed(operation, *args, **kwargs):
                operations.append(operation)
                yield type('Run', (), {'reused': False, 'source_refs': refs})()

            def produce(_, output, source_refs):
                output.mkdir()
                (output / 'sources.json').write_text(
                    json.dumps([{'image': 'input.png', 'source_refs': source_refs[media.relative_to(ROOT).as_posix()]}]),
                    encoding='utf8')
                return 1

            with patch.object(module, 'workspace_path', return_value=media), \
                    patch.object(module, '_export_impl', side_effect=produce), \
                    patch.object(module, 'generation_guard', allowed):
                for name in ('one', 'two'):
                    output = root / name
                    self.assertEqual(module.export(source, output), 1)
                    self.assertEqual(json.loads((output / 'sources.json').read_text(encoding='utf8'))[0]['source_refs'], ref_list)
                    self.assertFalse((output / 'source_refs.json').exists())
            self.assertNotEqual(*operations)
            self.assertEqual(operations[0], 'fire.export_reviewed_labels/' + (root / 'one').relative_to(ROOT).as_posix())
            output = root / 'rejected'
            @contextmanager
            def rejected(*args, **kwargs):
                raise RuntimeError('database unavailable')
                yield
            with patch.object(module, 'workspace_path', return_value=media), \
                    patch.object(module, '_export_impl') as body, \
                    patch.object(module, 'generation_guard', rejected):
                with self.assertRaisesRegex(RuntimeError, 'database unavailable'):
                    module.export(source, output)
                body.assert_not_called()
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
