"""Exercise source allocation and generated label coordinates using test doubles."""
import copy
import contextlib
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import yaml
from lab.training_common.bayesian_allocation import plan_generation

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
entry = importlib.import_module('generate_sticker_real_photo_v4')
NAMES = [f'item_{i}' for i in range(24)]


def sample():
    regions = []
    for x in (0, 25, 50, 75):
        regions.append({'role':'bin_front_blank_candidate', 'status':'visually_blank_candidate_not_calibrated',
                        'polygon':[[x,50],[x+20,50],[x+20,70],[x,70]]})
    geometry = [{'path':'datasets/raw/photo.png', 'image_size':[100,100], 'regions':regions}]
    p = {'photo':'datasets/raw/photo.png','record_index':0,'class_pool':list(range(24)),
         'size_fraction':[.6,.9], 'gain_range':[.8,1.0], 'saturation_range':[.75,1.0]}
    state = {'id':'photo','parameters':p,'planned_count':3,'max_count':6,
             'background':{'kind':'camera_photo','image':p['photo'],'geometry_manifest':entry.GEOMETRY,'record_index':0},
             'target':{'kind':'four_distinct_stickers','class_pool':p['class_pool']},
             'labels':{'format':'yolo_detection','status':'inserted_only_candidate'},
             'augmentation':{'gain_range':p['gain_range'],'saturation_range':p['saturation_range'],
                             'placement':'seeded_uniform_within_candidate_regions'},
             'sources':[p['photo']]+[f'{entry.TOOLKIT}/SourcePic/{x}.png' for x in NAMES]}
    return {'model_key':'waste_sticker','target_split':'train','constraints':dict(entry.CONSTRAINTS),
            'strata':[state]}, geometry


class RealPhotoTests(unittest.TestCase):
    def test_rejects_unsupported_geometry_and_state(self):
        plan, geometry = sample()
        for edit in (lambda p,g:p['constraints'].update(max_long_edge=999),
                     lambda p,g:p['strata'][0]['parameters'].update(gain_range=[.8,.8]),
                     lambda p,g:p['strata'][0]['sources'].pop(),
                     lambda p,g:g[0]['regions'][0].update(status='unknown')):
            p,g = copy.deepcopy(plan),copy.deepcopy(geometry)
            edit(p,g)
            with self.assertRaises(ValueError): entry.validate_plan(p,NAMES,g)

    def test_seeded_samples_preserve_source_and_emit_exact_boxes(self):
        plan, geometry = sample()
        run = SimpleNamespace(plan=plan,record_sample=mock.Mock())
        background = np.zeros((100,100,3),dtype=np.uint8)
        icons = [(np.full((6,6,3),100,np.uint8),np.ones((6,6),np.float32)) for _ in NAMES]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with mock.patch.object(entry,'ROOT',root), mock.patch.object(entry.base,'read_image',return_value=background):
                entry._produce(root/'out',run,NAMES,geometry,icons,123)
            result=json.loads((root/'out/manifest.json').read_text(encoding='utf8'))['records']
            self.assertEqual(len(result),3)
            actual=[row['actual'] for row in result]
            self.assertEqual(len({row['seed'] for row in actual}),3)
            self.assertTrue(all(len(set(row['class_ids']))==4 for row in actual))
            self.assertTrue(all(row['source_photo']=='datasets/raw/photo.png' for row in actual))
            for row in result:
                labels=(root/row['label']).read_text(encoding='utf8').splitlines()
                self.assertEqual(len(labels),4)
                self.assertEqual([int(line.split()[0]) for line in labels],row['actual']['class_ids'])
            self.assertEqual(run.record_sample.call_count,3)
            self.assertEqual(len({tuple(map(tuple,x['raw_boxes_xyxy'])) for x in actual}),3)


def surface_sample():
    colors = ['blue', 'red', 'green', 'gray']
    bins, placements = [], []
    for i, color in enumerate(colors):
        x = 5 + 48*i
        bins.append({'bin_id': f'bin-{i}', 'bin_color': color,
                     'allowed_polygon': [[x, 20], [x+40, 20], [x+40, 100], [x, 100]],
                     'exclude_polygons': [[[x+2, 24], [x+12, 24], [x+12, 34], [x+2, 34]]],
                     'occluder_polygons': []})
        placements.append({'bin_id': f'bin-{i}', 'class_id': i,
                           'paper_quad': [[x+8, 55], [x+32, 56], [x+31, 80], [x+7, 79]],
                           'identifiability': 'identifiable'})
    geometry = [{'path': 'datasets/raw/photo.png', 'image_size': [200, 120], 'bins': bins}]
    p = {'photo': geometry[0]['path'], 'record_index': 0, 'placements': placements, 'gain': .9, 'saturation': .8}
    state = {'id': 'surface', 'parameters': p, 'planned_count': 1, 'max_count': 1,
             'background': {'kind': 'camera_photo', 'image': p['photo'], 'geometry_manifest': entry.GEOMETRY, 'record_index': 0},
             'target': {'kind': 'explicit_bin_stickers', 'placements': placements},
             'labels': {'format': 'yolo_detection', 'status': 'inserted_visible_candidates_background_unknown'},
             'augmentation': {'gain': .9, 'saturation': .8, 'placement': 'explicit_surface_quads'},
             'sources': [p['photo']] + [f'{entry.TOOLKIT}/SourcePic/{NAMES[i]}.png' for i in range(4)]}
    return {'model_key': 'waste_sticker', 'target_split': 'train', 'constraints': dict(entry.SURFACE_CONSTRAINTS),
            'strata': [state]}, geometry


class SurfaceContractTests(unittest.TestCase):
    def test_entry_admits_custom_geometry_and_passes_plan_to_guard(self):
        plan, geometry = surface_sample()
        geometry_ref = 'datasets/annotated/waste_sticker/surfaces.jsonl'
        plan['strata'][0]['background']['geometry_manifest'] = geometry_ref
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            geometry_path = root/geometry_ref
            geometry_path.parent.mkdir(parents=True)
            geometry_path.write_text(json.dumps(geometry[0])+'\n', encoding='utf8')
            config = root/entry.TOOLKIT/'configs/classes.yaml'
            config.parent.mkdir(parents=True)
            config.write_text(yaml.safe_dump({'items': [{'name': name} for name in NAMES]}), encoding='utf8')
            plan_path = root/'plan.json'
            plan_path.write_text(json.dumps(plan), encoding='utf8')
            captured = []
            @contextlib.contextmanager
            def guard(*args, **kwargs):
                captured.append(kwargs)
                yield SimpleNamespace(reused=True)
            args = ['entry', '--plan', 'plan.json', '--out', 'datasets/samples/waste_sticker']
            with mock.patch.object(entry, 'ROOT', root), mock.patch.object(entry, 'generation_guard', guard), \
                 mock.patch('datasets.scripts.common.paths.sample_dir', return_value=root/'datasets/samples/waste_sticker'), \
                 mock.patch('datasets.scripts.common.config.load_task_config', return_value=({}, {'real_photo_geometry_manifest': geometry_ref, 'real_annotations': 'annotation.json', 'real_video_frames_root': 'frames', 'records': []})), \
                 mock.patch.object(entry, 'source_path', return_value=root/entry.TOOLKIT), \
                 mock.patch.object(entry, 'real_source_catalog', return_value=[]), \
                 mock.patch.object(entry, 'artwork_catalog', return_value={i: f'{entry.TOOLKIT}/SourcePic/{name}.png' for i, name in enumerate(NAMES)}), \
                 mock.patch.object(sys, 'argv', args):
                entry.main()
            self.assertEqual(captured[0]['production_plan'], plan)
            self.assertIn(geometry_path, captured[0]['procedural_inputs'])
            self.assertFalse((root/'datasets/samples/waste_sticker').exists())

    def test_allocator_preserves_surface_contract(self):
        declaration, geometry = surface_sample()
        declaration.update(schema_version=1, purpose='fixture', budget=1, uncertainty_weight=0,
                           independence_basis='fixture')
        state = declaration['strata'][0]
        state.update(min_count=1, prior={'alpha': 1, 'beta': 1}, exposure_weight=1,
                     failure_cost=1, failure_event='fixture')
        state.pop('planned_count')
        planned = plan_generation(declaration)
        entry.validate_surface_plan(planned, NAMES, geometry)
        self.assertEqual(planned['strata'][0]['planned_count'], 1)
        self.assertEqual(planned['strata'][0]['parameters'], state['parameters'])

    def test_each_class_accepts_each_bin_color(self):
        for cls in range(24):
            for index in range(4):
                plan, geometry = surface_sample()
                state = plan['strata'][0]
                state['parameters']['placements'] = [state['parameters']['placements'][index]]
                state['parameters']['placements'][0]['class_id'] = cls
                state['target']['placements'] = state['parameters']['placements']
                state['sources'] = [state['parameters']['photo'], f'{entry.TOOLKIT}/SourcePic/{NAMES[cls]}.png']
                entry.validate_surface_plan(plan, NAMES, geometry)

    def test_rejects_wrong_identity_region_and_duplicate_outputs(self):
        def unknown_bin(p, g): p['strata'][0]['parameters']['placements'][0]['bin_id'] = 'other'
        def outside(p, g): p['strata'][0]['parameters']['placements'][0]['paper_quad'][0] = [1, 1]
        def excluded(p, g): g[0]['bins'][0]['exclude_polygons'] = [[[10, 50], [40, 50], [40, 90], [10, 90]]]
        def duplicate(p, g): p['strata'][0]['max_count'] = 2
        def hidden(p, g): g[0]['bins'][0]['occluder_polygons'] = [[[5, 45], [45, 45], [45, 95], [5, 95]]]
        for change in (unknown_bin, outside, excluded, duplicate, hidden):
            plan, geometry = surface_sample()
            change(plan, geometry)
            with self.assertRaises(ValueError):
                entry.validate_surface_plan(plan, NAMES, geometry)

    def test_outputs_visible_candidates_bin_ids_and_class_counts(self):
        plan, geometry = surface_sample()
        geometry[0]['bins'][0]['occluder_polygons'] = [[[10, 55], [22, 55], [22, 81], [10, 81]]]
        plan['strata'][0]['parameters']['placements'][1]['identifiability'] = 'unknown'
        run = SimpleNamespace(plan=plan, record_sample=mock.Mock())
        background = np.full((120, 200, 3), 30, dtype=np.uint8)
        icons = [(np.full((6, 6, 3), 100, np.uint8), np.ones((6, 6), np.float32)) for _ in NAMES]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(entry, 'ROOT', root), mock.patch.object(entry.base, 'read_image', return_value=background):
                entry._produce_surface(root/'out', run, NAMES, geometry, icons, 123)
            report = json.loads((root/'out/manifest.json').read_text(encoding='utf8'))
            row = report['records'][0]
            instances = row['actual']['instances']
            self.assertEqual([x['bin_id'] for x in instances], ['bin-0', 'bin-1', 'bin-2', 'bin-3'])
            self.assertEqual([x['bin_color'] for x in instances], ['blue', 'red', 'green', 'gray'])
            self.assertEqual(instances[1]['identifiability'], 'unknown')
            self.assertIsNone(instances[1]['class_id'])
            self.assertEqual(instances[1]['planned_class_id'], 1)
            self.assertEqual(len((root/row['label_candidate']).read_text(encoding='utf8').splitlines()), 3)
            self.assertEqual(report['counts_by_class']['item_1'], 0)
            self.assertEqual(sum(report['counts_by_class'].values()), 3)
            self.assertGreater(instances[0]['visible_bbox_xyxy'][0], 10)
            self.assertFalse((root/'out/labels/train').exists())
            run.record_sample.assert_called_once()


if __name__ == '__main__':
    unittest.main()
