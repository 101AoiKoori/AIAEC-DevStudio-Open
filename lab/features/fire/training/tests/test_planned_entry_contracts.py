"""Isolated plan admission checks; no renderer, production database or images."""
import ast
import argparse
import copy
import contextlib
import io
import json
import math
import random
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from lab.training_common.production_guard import plan_production

ROOT = Path(__file__).resolve().parents[5]
SCRIPTS = ROOT / 'lab/features/fire/training/scripts'


def functions(filename, names, bindings):
    path = (ROOT / 'datasets/scripts/fire' / filename
            if filename in {'render_blender_fire_dataset.py', 'fire_blender_labels.py'} else SCRIPTS / filename)
    tree = ast.parse(path.read_text(encoding='utf8'), filename=str(path))
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    env = {'Path': Path, 'json': json, **bindings}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), 'exec'), env)
    return env


class PlannedEntryContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / 'tmp')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / 'plan.json'
        self.common = {'schema_version': 1, 'target_split': 'train', 'purpose': 'contract fixture',
                       'budget': 1, 'uncertainty_weight': 0, 'independence_basis': 'fixture parents'}
        self.risk = {'min_count': 0, 'max_count': 1, 'prior': {'alpha': 1, 'beta': 1},
                     'exposure_weight': 1, 'failure_cost': 1}

    def write(self, value):
        self.path.write_text(json.dumps(value), encoding='utf8')

    def test_fire_scenario_validation_and_unsupported_conditions(self):
        labels = functions('fire_blender_labels.py', {'scenario'}, {})
        env = functions('render_blender_fire_dataset.py', {'read_production_declaration'},
                        {'ROOT': ROOT, 'scenario': labels['scenario']})
        args = SimpleNamespace(count=1, scenario_offset=0, vehicle_pilot=False, seed=42,
                               resolution=640, height=480, samples=4, device='CPU')
        state = dict(self.risk, id='blender_scenario_01_single',
            failure_event='renderer_state_or_label_uncertainty',
            conditions=['scenario:single', 'traffic_signal_injection:none'],
            background={'domain': 'synthetic_building_arena', 'traffic_signal_injection': 'none'},
            target={'scenario': 'single', 'scenario_index': 1, 'active_buildings': ['A'], 'fire_counts': {'A': 1}},
            labels={k: {'class_id': 0, 'format': 'yolo_v8', 'status': 'candidate_requires_visual_review'} for k in ('fire', 'name')},
            augmentation={'camera': 'seeded_random', 'lighting': 'seeded_random', 'sticker_placement': 'seeded_random', 'traffic_signal_injection': 'none'},
            sources=['blender_building_library', 'fire_sticker_artwork'],
            parameters={'scenario_index': 1, 'scenario_offset': 0, 'seed': 42, 'resolution': [640, 480], 'samples': 4, 'device': 'CPU', 'mode': 'standard'})
        plan = dict(self.common, model_key='fire_sticker', constraints={'route_context': 'building_facade',
            'traffic_signal_cooccurrence': 'forbidden_separate_route', 'traffic_signal_generation': 'no_traffic_signal_injection',
            'unknown_visual_targets': 'suppress_labels'}, strata=[state])
        self.write(plan)
        self.assertEqual(env['read_production_declaration'](self.path, args, None)[1]['strata'][0]['target']['fire_counts'], {'A': 1})
        for mutate in ('counts', 'duplicate', 'constraint'):
            bad = copy.deepcopy(plan)
            if mutate == 'counts': bad['strata'][0]['target']['fire_counts']['A'] = 2
            if mutate == 'duplicate': bad['strata'].append(copy.deepcopy(state))
            if mutate == 'constraint': bad['constraints']['traffic_signal_generation'] = 'add'
            self.write(bad)
            with self.assertRaises(ValueError): env['read_production_declaration'](self.path, args, None)
        mixed = copy.deepcopy(plan)
        mixed['strata'][0]['background'] = {'domain': 'real_scene_with_blender_building',
            'traffic_signal_injection': 'none', 'source_ids': ['a', 'b'], 'sampling': 'seeded_uniform'}
        mixed['strata'][0]['augmentation'].update(tone_sampling='seeded_uniform',
            tone_profiles=[{'exposure_ev': e, 'saturation': s} for e in (-.5, 0.) for s in (.65, 1.)])
        mixed['strata'][0]['sources'].append('reviewed_real_building_background')
        self.write(mixed)
        env['read_production_declaration'](self.path, args, [{'id': 'a'}, {'id': 'b'}])
        mixed['strata'][0]['background']['sampling'] = 'undeclared'
        self.write(mixed)
        with self.assertRaises(ValueError): env['read_production_declaration'](self.path, args, [{'id': 'a'}, {'id': 'b'}])

    def test_real_background_and_tone_sample_use_each_seed(self):
        choose = functions('render_blender_fire_dataset.py', {'sample_real_scene'}, {})['sample_real_scene']
        sources = ['a', 'b']
        choices = [choose(sources, random.Random(seed)) for seed in range(100)]
        self.assertEqual(choices, [choose(sources, random.Random(seed)) for seed in range(100)])
        self.assertEqual({x[0] for x in choices}, set(sources))
        self.assertEqual({(x[1], x[2]) for x in choices}, {(-.5, .65), (-.5, 1.), (0., .65), (0., 1.)})

    def test_explicit_states_are_admitted_without_weakening_legacy_contract(self):
        labels = functions('fire_blender_labels.py', {'scenario', 'explicit_scenario'}, {})
        env = functions('render_blender_fire_dataset.py', {'read_production_declaration'},
                        {'ROOT': ROOT, **labels})
        args = SimpleNamespace(count=9, scenario_offset=0, vehicle_pilot=False, seed=42,
                               resolution=640, height=480, samples=4, device='CPU')
        states = []
        for building in ('A', 'B', 'C'):
            for count in (0, 1, 5):
                states.append(dict(self.risk, id=f'blender_explicit_{building}_{count}',
                    failure_event='renderer_state_or_label_uncertainty',
                    conditions=['scenario:explicit', 'visibility:no_added_occluder', 'traffic_signal_injection:none'],
                    background={'domain': 'synthetic_building_arena', 'traffic_signal_injection': 'none'},
                    target={'scenario': 'explicit', 'active_buildings': [building], 'fire_counts': {building: count},
                            'view': 'oblique_left', 'placement': 'upper', 'visibility': 'no_added_occluder'},
                    labels={k: {'class_id': 0, 'format': 'yolo_v8', 'status': 'candidate_requires_visual_review'} for k in ('fire', 'name')},
                    augmentation={'camera': 'oblique_left', 'lighting': 'seeded_random',
                                  'sticker_placement': 'upper', 'traffic_signal_injection': 'none'},
                    sources=['blender_building_library', 'fire_sticker_artwork'],
                    parameters={'scenario_index': None, 'scenario_offset': 0, 'seed': 42,
                                'resolution': [640, 480], 'samples': 4, 'device': 'CPU', 'mode': 'standard'}))
        plan = dict(self.common, model_key='fire_sticker', budget=9, constraints={
            'route_context': 'building_facade', 'traffic_signal_cooccurrence': 'forbidden_separate_route',
            'traffic_signal_generation': 'no_traffic_signal_injection', 'unknown_visual_targets': 'suppress_labels'},
            strata=states)
        self.write(plan)
        declaration = env['read_production_declaration'](self.path, args, None)[1]
        self.assertEqual(plan_production(declaration, ROOT)['planned_total'], 9)
        supported = copy.deepcopy(plan)
        supported['strata'][0]['target'].update(active_buildings=['C'], fire_counts={'C': 1},
                                                 placement='c_supported_left')
        supported['strata'][0]['augmentation']['sticker_placement'] = 'c_supported_left'
        self.write(supported)
        env['read_production_declaration'](self.path, args, None)
        with self.assertRaisesRegex(ValueError, 'standalone building scene'):
            env['read_production_declaration'](self.path, args, [{'id': 'photo'}])
        for mutate in ('duplicate', 'view_mismatch', 'placement_mismatch', 'unsafe_visibility', 'unsupported_count'):
            bad = copy.deepcopy(plan)
            if mutate == 'duplicate':
                bad['strata'][1]['target'] = copy.deepcopy(bad['strata'][0]['target'])
            if mutate == 'view_mismatch': bad['strata'][0]['augmentation']['camera'] = 'near_frontal'
            if mutate == 'placement_mismatch': bad['strata'][0]['augmentation']['sticker_placement'] = 'lower'
            if mutate == 'unsafe_visibility': bad['strata'][0]['target']['visibility'] = 'occluder'
            if mutate == 'unsupported_count': bad['strata'][0]['target']['fire_counts']['A'] = 6
            self.write(bad)
            with self.assertRaises(ValueError):
                env['read_production_declaration'](self.path, args, None)

    def test_explicit_view_and_placement_reach_scene_operations(self):
        class VectorStub(list):
            def __sub__(self, other):
                return self

            def to_track_quat(self, *_args):
                return self

            def to_euler(self):
                return (0, 0, 0)

        camera = SimpleNamespace(data=SimpleNamespace(lens=0, clip_start=0))
        scene = SimpleNamespace(camera=camera)
        groups = {'A': {'x': 0}}
        env = functions('render_blender_fire_dataset.py', {'set_camera'},
                        {'math': math, 'Vector': VectorStub})
        for view, lower, upper in (('near_frontal', -8, 8), ('oblique_left', -25, -15),
                                   ('oblique_right', 15, 25)):
            record = env['set_camera'](scene, groups, ('A',), 'normal', random.Random(3), view=view)
            x, y, _ = camera.location
            self.assertLessEqual(lower, math.degrees(math.atan2(x, -y)))
            self.assertLessEqual(math.degrees(math.atan2(x, -y)), upper)
            self.assertEqual(record['view_profile'], view)

        class Sticker(dict):
            def __init__(self):
                super().__init__()
                self.location = SimpleNamespace(x=0, z=0)
                self.rotation_euler = SimpleNamespace(y=0)
                self.data = SimpleNamespace(vertices=[], update=lambda: None)

        slot_helper = functions('fire_blender_labels.py', {'front_facade_slots'}, {})['front_facade_slots']
        place = functions('render_blender_fire_dataset.py', {'place_stickers'},
                          {'math': math, 'front_facade_slots': slot_helper,
                           'FIRE_PLACEMENT_JITTER_M': .002, 'FIRE_ROTATION_MAX_RAD': .09,
                           'FIRE_PAPER_RIPPLE_MAX_M': .00003})['place_stickers']
        group = {'x': 0, 'pool': [Sticker() for _ in range(5)], 'paper_base_y': [[] for _ in range(5)]}
        place(group, 5, random.Random(4), placement='upper')
        self.assertTrue(all(sticker.location.z >= .263 for sticker in group['pool']))
        place(group, 1, random.Random(4), placement='left')
        self.assertLess(group['pool'][0].location.x, 0)
        self.assertTrue(all(sticker.hide_render for sticker in group['pool'][1:]))

    def test_name_text_and_ocr_policy_validation(self):
        image, mask, manifest = (self.root / name for name in ('image.png', 'mask.png', 'manifest.jsonl'))
        item = {'state_id': 'arena_name_00_000000_v00', 'image': image, 'mask': mask, 'manifest': manifest,
                'variant': 0, 'name': '青禾大厦', 'sign': {'building': 'A', 'pixel_bbox_xyxy': [1, 2, 50, 20], 'visible_pixels': 300}}
        rel = lambda p: p.relative_to(ROOT).as_posix()
        env = functions('compose_arena_name_v4.py', {'_production_declaration'}, {'ROOT': ROOT, 'rel': rel})
        state = dict(self.risk, id=item['state_id'], failure_event='name_paper_visibility_or_ink_projection_uncertainty',
            conditions=[f'source:{rel(image)}', 'variant:0', 'traffic_signal_injection:none'],
            background={'domain': 'synthetic_blender_building_arena', 'source_image': rel(image), 'traffic_signal_injection': 'none'},
            target={'kind': 'building_name_paper', 'building': 'A', 'text': item['name'], 'paper_bbox_xyxy': [1, 2, 50, 20], 'visible_pixels': 300},
            labels={'class_id': 0, 'format': 'yolo_v8', 'task': 'paper_detection', 'ocr_status': 'not_verified'},
            augmentation={'method': 'fitted_perspective_ink_projection', 'geometry': 'verified_instance_mask_quad', 'alpha': 0.88},
            sources=['blender_arena_name_paper', 'blender_instance_mask', 'name_font_asset'],
            parameters={'source_image': rel(image), 'source_mask': rel(mask), 'source_manifest': rel(manifest), 'variant_index': 0, 'variant_count': 0})
        plan = dict(self.common, model_key='building_name_paper', constraints={'route_context': 'building_facade',
            'traffic_signal_cooccurrence': 'forbidden_separate_route', 'traffic_signal_generation': 'no_traffic_signal_injection', 'ocr_truth': 'not_verified'}, strata=[state])
        self.write(plan)
        self.assertEqual(env['_production_declaration'](self.path, [item], 0)[1]['strata'][0]['target']['text'], item['name'])
        for field, value in (('text', '其他大厦'), ('ocr_status', 'readable')):
            bad = copy.deepcopy(plan)
            bad['strata'][0]['target' if field == 'text' else 'labels'][field] = value
            self.write(bad)
            with self.assertRaises(ValueError): env['_production_declaration'](self.path, [item], 0)
        bad = copy.deepcopy(plan); bad['strata'][0]['max_count'] = 2; self.write(bad)
        with self.assertRaises(ValueError): env['_production_declaration'](self.path, [item], 0)
        bad = copy.deepcopy(plan); bad['target_split'] = 'val'; self.write(bad)
        with self.assertRaises(ValueError): env['_production_declaration'](self.path, [item], 0)
        bad = copy.deepcopy(plan); bad['strata'].append(copy.deepcopy(state))
        with self.assertRaisesRegex(ValueError, 'Duplicate stratum'): plan_production(bad, ROOT)

    def test_missing_plan_and_source_fail_before_production(self):
        fire = functions('render_blender_fire_dataset.py', {'parse_args'}, {'argparse': argparse, 'sys': sys})['parse_args']
        name = functions('compose_arena_name_v4.py', {'main'}, {'argparse': argparse, 'ROOT': ROOT})['main']
        with contextlib.redirect_stderr(io.StringIO()):
            with patch.object(sys, 'argv', ['blender', '--', '--output', 'unused']):
                with self.assertRaises(SystemExit): fire()
            with patch.object(sys, 'argv', ['blender', '--', '--output', 'unused', '--production-plan', 'unused', '--vehicle-pilot']):
                with self.assertRaises(SystemExit): fire()
            with patch.object(sys, 'argv', ['name', '--output', 'unused', '--source', 'missing']):
                with self.assertRaises(SystemExit): name()
        with patch.object(sys, 'argv', ['name', '--output', 'unused', '--source', 'datasets/exports/fire/missing', '--production-plan', 'unused']):
            with self.assertRaisesRegex(ValueError, 'existing fire export batches'): name()

    def test_disabled_signal_composite(self):
        env = functions('tone_vehicle_fire_pilot.py', {'generate'}, {'DISABLED_REASON': 'disabled signal composite'})
        with self.assertRaisesRegex(ValueError, 'disabled signal composite'): env['generate']()

    def test_name_sources_without_assets_and_colliding_old_stems(self):
        sources = [self.root / 'batch_a_20261113', self.root / 'batch_b_20261113']
        row = {'image': 'fire_000001.png', 'instance_pass': 'mask.png', 'occluder': False, 'split': 'train',
               'instances': [{'kind': 'name_sign', 'state': 'visible', 'truncated': False, 'text': '青禾大厦', 'building': 'A'}]}
        for source in sources:
            source.mkdir()
            (source / 'run.json').write_text(json.dumps({'status': 'complete_candidates', 'source_scene': 'datasets/assets/scene.blend'}))
            (source / 'manifest.jsonl').write_text(json.dumps(row))
            (source / 'fire_000001.png').touch()
            (source / 'mask.png').touch()
        env = functions('compose_arena_name_v4.py', {'_candidate_specs'}, {'make_name': lambda _: 'unused', 'np': np})
        candidates = env['_candidate_specs'](sources, 0)
        self.assertEqual(len({item['stem'] for item in candidates}), 2)
        self.assertEqual(candidates[0]['source_assets'], ['datasets/assets/scene.blend'])
        self.assertEqual(candidates[0]['building'], 'A')
        self.assertIsNone(candidates[0]['tone'])
        (sources[1] / 'run.json').write_text(json.dumps({'status': 'complete_candidates'}))
        self.assertEqual(env['_candidate_specs'](sources, 0)[1]['source_assets'], [])
        invalid = copy.deepcopy(row); invalid['tone'] = {'exposure_ev': 'unknown'}
        (sources[1] / 'manifest.jsonl').write_text(json.dumps(invalid))
        with self.assertRaisesRegex(ValueError, 'finite exposure_ev'): env['_candidate_specs'](sources, 0)
        invalid = copy.deepcopy(row); del invalid['instances'][0]['building']
        (sources[1] / 'manifest.jsonl').write_text(json.dumps(invalid))
        with self.assertRaisesRegex(ValueError, 'building identifier'): env['_candidate_specs'](sources, 0)
        invalid = copy.deepcopy(row); invalid['split'] = 'unassigned'
        (sources[1] / 'manifest.jsonl').write_text(json.dumps(invalid))
        with self.assertRaisesRegex(ValueError, 'not assigned to train'): env['_candidate_specs'](sources, 0)
        (sources[1] / 'manifest.jsonl').write_text(json.dumps(row))
        (sources[1] / 'run.json').write_text(json.dumps({'status': 'complete_candidates', 'purpose': 'visual_review_pilot_not_training'}))
        with self.assertRaisesRegex(ValueError, 'not intended for training'): env['_candidate_specs'](sources, 0)


if __name__ == '__main__':
    unittest.main()
