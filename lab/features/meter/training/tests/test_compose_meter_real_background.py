"""State-plan and output/receipt checks with isolated synthetic fixtures."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from datasets.scripts.meter import compose_meter_real_background_v4 as module


class CompositePlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=module.ROOT / 'tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.patcher = mock.patch.object(module, 'ROOT', self.root)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.batch = self.root / 'datasets/exports/meter/batch'
        self.batch.mkdir(parents=True)
        self.bg = self.root / 'background.png'
        self.parent = self.root / 'parent.png'
        self.source_image = self.batch / 'source.png'
        self.foreground = self.batch / 'foreground.png'
        self.label = self.batch / 'label.txt'
        self.annotation = self.batch / 'annotation.json'
        self.manifest = self.batch / 'manifest.json'
        self.records = self.batch / 'samples.jsonl'
        self.evidence = self.root / 'background_manifest.json'
        self.review = self.root / 'meter_review.json'
        self.negative_label = self.root / 'negative.txt'
        Image.new('RGB', (640, 480), (10, 20, 30)).save(self.bg)
        Image.new('RGB', (640, 480), (1, 2, 3)).save(self.parent)
        Image.new('RGB', (640, 480), (4, 5, 6)).save(self.source_image)
        fg = Image.new('RGBA', (640, 480), (0, 0, 0, 0))
        fg.putpixel((320, 240), (200, 0, 0, 255))
        fg.save(self.foreground)
        for path in (self.label, self.annotation, self.manifest, self.records, self.evidence):
            path.write_text('{}', encoding='utf8')
        self.negative_label.write_text('', encoding='utf8')
        self.review.write_text(json.dumps({'images': [{'image': module.rel(self.bg),
            'parent_video': module.rel(self.parent), 'pose_state': 'candidate_complete_negative',
            'label': module.rel(self.negative_label), 'split': 'regression', 'human_accepted': False}]}), encoding='utf8')
        self.background = {'image': self.bg, 'parent': self.parent, 'manifest': self.evidence,
                           'record_index': 0, 'relation': 'describes', 'row': {'id': 'bg'}}
        self.row = {'id': 'arena_meter_00000'}
        self.selected = [(0, self.row, self.source_image, self.foreground, self.label, self.annotation,
                          [0] + [.5] * 28)]
        self.sources = [module.rel(p) for p in (self.bg, self.parent, self.source_image, self.foreground)]

    def plan(self):
        return {'model_key': 'meter', 'target_split': 'train', 'budget': 1,
                'constraints': {'resolution': [640, 480], 'geometry': 'identity', 'label_policy': 'candidate_pose'},
                'strata': [{'id': 'pair', 'min_count': 1, 'max_count': 1, 'conditions': [],
                            'background': {'kind': 'real_captured_scene'}, 'target': {'kind': 'rendered_station'},
                            'labels': {'kind': 'candidate_pose'},
                            'augmentation': {'kind': 'alpha_composite_identity'}, 'sources': self.sources,
                            'parameters': {'background_image': module.rel(self.bg), 'foreground_id': self.row['id']}}]}

    def test_rejects_duplicate_unavailable_and_unsupported_states(self):
        for issue in ('duplicate', 'missing', 'capacity', 'unknown'):
            plan = self.plan()
            state = plan['strata'][0]
            if issue == 'duplicate': plan['strata'].append(dict(state, id='second'))
            if issue == 'missing': state['parameters']['foreground_id'] = 'missing'
            if issue == 'capacity': state['max_count'] = 2
            if issue == 'unknown': state['augmentation']['kind'] = 'unsupported'
            with self.subTest(issue=issue), self.assertRaises(ValueError):
                module.validate_plan(plan, 1, [self.background], self.selected)

    def test_single_plan_controls_output_and_receipt(self):
        output = self.root / 'datasets/exports/meter/output'
        plan_path = self.root / 'plan.json'
        plan = self.plan()
        sampling_evidence = self.root / 'sampling.json'
        sampling_evidence.write_text(json.dumps({'sampling_seed': 42,
            'selected': [[module.rel(self.bg), self.row['id'], 'background_tone']]}), encoding='utf8')
        plan['sampling_evidence'] = module.rel(sampling_evidence)
        plan_path.write_text(json.dumps(plan), encoding='utf8')
        captured = []
        refs = {module.rel(self.bg): [{'manifest': module.rel(self.evidence), 'record_index': 0, 'relation': 'describes'}],
                module.rel(self.source_image): [{'manifest': module.rel(self.records), 'record_index': 0, 'relation': 'describes'}]}
        @contextmanager
        def guard(*args, **kwargs):
            self.assertEqual(kwargs['sample_output_paths'], [output / 'images'])
            self.assertIn(sampling_evidence, kwargs['procedural_inputs'])
            allocation = dict(plan, strata=[dict(plan['strata'][0], planned_count=1)])
            def record(state_id, **data):
                self.assertTrue(data['output'].is_file())
                captured.append((state_id, data))
            yield SimpleNamespace(reused=False, plan=allocation, source_refs=refs,
                                  record_sample=record, set_result=lambda value: None, value=None)
        argv = ['compose', '--batch', str(self.batch), '--output', str(output), '--count', '1',
                '--production-plan', str(plan_path), '--background-meter-review', str(self.review),
                '--background', 'x', 'x', 'x', '0', 'describes']
        with mock.patch.object(sys, 'argv', argv), mock.patch.object(module, 'background_spec', return_value=self.background), \
             mock.patch.object(module, 'validate_lineages'), mock.patch.object(module, 'record_at', return_value=self.background['row']), \
             mock.patch.object(module, 'source_rows', return_value=(self.manifest, self.records, {'status': 'fixture'}, self.selected)), \
             mock.patch.object(module, 'generation_guard', guard):
            module.main()
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0][0], 'pair')
        self.assertEqual(captured[0][1]['parameters'], plan['strata'][0]['parameters'])
        self.assertEqual(captured[0][1]['sources'], self.sources)
        self.assertEqual(captured[0][1]['actual']['background_meter_review']['prior_split'], 'regression')
        self.assertFalse(captured[0][1]['actual']['background_meter_review']['human_accepted'])
        self.assertEqual(captured[0][1]['actual']['sampling_seed'], 42)
        self.assertEqual(captured[0][1]['actual']['sampling_group'], 'background_tone')
        manifest = json.loads((output / 'manifest.json').read_text(encoding='utf8'))
        self.assertEqual(manifest['sampling_evidence'], module.rel(sampling_evidence))
        with Image.open(output / 'images/meter_real_composite_00000.png') as result:
            self.assertEqual(result.getpixel((0, 0)), (10, 20, 30))
            self.assertEqual(result.getpixel((320, 240)), (200, 0, 0))

    def test_missing_plan_is_rejected_before_output(self):
        output = self.root / 'datasets/exports/meter/output'
        with mock.patch.object(sys, 'argv', ['compose', '--batch', str(self.batch), '--output', str(output),
                                           '--count', '1', '--background', 'x', 'x', 'x', '0', 'describes']):
            with self.assertRaises(SystemExit):
                module.main()
        self.assertFalse(output.exists())

    def test_nonempty_negative_label_is_rejected(self):
        self.negative_label.write_text('0 .5 .5 .1 .1', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'not empty'):
            module.background_meter_review(self.review, [self.background])

    def test_placement_transforms_box_and_points_without_claiming_readability(self):
        fg = Image.new('RGBA', (640, 480), (0, 0, 0, 0))
        for x in range(316, 325):
            for y in range(236, 245):
                fg.putpixel((x, y), (200, 0, 0, 255))
        values = [0, .5, .5, .2, .2] + [.5, .5, 2] * 8
        values[5:8] = [0, 0, 0]
        placement = {'scale': .5, 'offset_xy': [100, 20], 'region_xyxy': [0, 0, 640, 480]}
        image, output, geometry = module.place_foreground(fg, values, placement)
        self.assertEqual(image.getpixel((260, 140)), (200, 0, 0, 255))
        self.assertAlmostEqual(output[1], 260/640)
        self.assertAlmostEqual(output[2], 140/480)
        self.assertAlmostEqual(output[3], .1)
        self.assertAlmostEqual(output[4], .1)
        self.assertEqual(output[5:8], [0, 0, 0])
        self.assertEqual(output[8:11], [260/640, 140/480, 1])
        self.assertIn('physical fit unconfirmed', geometry['region_semantics'])
        self.assertEqual(values[10], 2)

    def test_placement_rejects_invalid_region_clipping_and_invalid_points(self):
        fg = Image.new('RGBA', (640, 480), (0, 0, 0, 0))
        fg.putpixel((320, 240), (200, 0, 0, 255))
        values = [0, .5, .5, .2, .2] + [.5, .5, 2] * 8
        for placement in (
            {'scale': float('nan'), 'offset_xy': [0, 0], 'region_xyxy': [0, 0, 640, 480]},
            {'scale': 1, 'offset_xy': [0, 0], 'region_xyxy': [0, 0, 100, 100]},
            {'scale': 1, 'offset_xy': [500, 0], 'region_xyxy': [0, 0, 640, 480]},
        ):
            with self.subTest(placement=placement), self.assertRaises(ValueError):
                module.place_foreground(fg, values, placement)
        values[7] = 3
        with self.assertRaisesRegex(ValueError, 'invalid candidate point'):
            module.place_foreground(fg, values, {'scale': 1, 'offset_xy': [0, 0],
                                                 'region_xyxy': [0, 0, 640, 480]})
        values[7] = 2
        with self.assertRaisesRegex(ValueError, 'transformed foreground has no useful opaque station'):
            module.place_foreground(fg, values, {'scale': .5, 'offset_xy': [100, 20],
                                                 'region_xyxy': [0, 0, 640, 480]})

    def test_planned_transform_is_applied_and_recorded(self):
        output = self.root / 'datasets/exports/meter/output'
        fg = Image.new('RGBA', (640, 480), (0, 0, 0, 0))
        for x in range(316, 325):
            for y in range(236, 245):
                fg.putpixel((x, y), (200, 0, 0, 255))
        fg.save(self.foreground)
        plan = self.plan()
        plan['constraints']['geometry'] = 'uniform_scale_translate'
        state = plan['strata'][0]
        state['augmentation']['kind'] = 'alpha_composite_uniform_scale_translate'
        state['parameters']['placement'] = {'scale': .5, 'offset_xy': [100, 20],
                                            'region_xyxy': [0, 0, 640, 480]}
        self.selected[0] = (*self.selected[0][:-1], [0, .5, .5, .2, .2] + [.5, .5, 2] * 8)
        plan_path = self.root / 'plan.json'
        plan_path.write_text(json.dumps(plan), encoding='utf8')
        refs = {module.rel(self.bg): [{'manifest': module.rel(self.evidence), 'record_index': 0, 'relation': 'describes'}],
                module.rel(self.source_image): [{'manifest': module.rel(self.records), 'record_index': 0, 'relation': 'describes'}]}
        captured = []
        @contextmanager
        def guard(*args, **kwargs):
            yield SimpleNamespace(reused=False, plan=dict(plan, strata=[dict(state, planned_count=1)]),
                source_refs=refs, record_sample=lambda *args, **data: captured.append(data),
                set_result=lambda value: None, value=None)
        argv = ['compose', '--batch', str(self.batch), '--output', str(output), '--count', '1',
                '--production-plan', str(plan_path), '--background', 'x', 'x', 'x', '0', 'describes']
        with mock.patch.object(sys, 'argv', argv), mock.patch.object(module, 'background_spec', return_value=self.background), \
             mock.patch.object(module, 'validate_lineages'), mock.patch.object(module, 'record_at', return_value=self.background['row']), \
             mock.patch.object(module, 'source_rows', return_value=(self.manifest, self.records, {'status': 'fixture'}, self.selected)), \
             mock.patch.object(module, 'generation_guard', guard):
            module.main()
        record = json.loads((output / 'annotations/meter_real_composite_00000.json').read_text(encoding='utf8'))
        self.assertEqual(record['geometry_transform']['offset_xy'], [100, 20])
        self.assertEqual(set(record['keypoint_readability'].values()), {'unknown'})
        self.assertEqual(record['reflection_status'], 'unknown')
        self.assertEqual(record['physical_reading'], 'unknown')
        self.assertEqual(captured[0]['actual']['geometry_transform'], record['geometry_transform'])
        label = (output / 'labels_candidate/meter_real_composite_00000.txt').read_text().split()
        self.assertAlmostEqual(float(label[1]), 260/640)
        self.assertEqual(int(label[7]), 1)
        with Image.open(output / 'images/meter_real_composite_00000.png') as result:
            self.assertEqual(result.getpixel((260, 140)), (200, 0, 0))


if __name__ == '__main__':
    unittest.main()
