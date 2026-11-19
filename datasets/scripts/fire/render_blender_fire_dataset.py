"""Render source-grouped synthetic fire/name candidates from the v13 Blender scene.

Run from the workspace root using Blender 5.2, loading the source scene first::

    blender -b datasets/assets/blender/aiaec_dataset.blend \
      --python-exit-code 1 --python datasets/scripts/fire/render_blender_fire_dataset.py -- \
      --output datasets/samples/fire --count 10 --device OPTIX

The first ten cases deliberately cover zero, one, many, multiple buildings,
occlusion and truncation. Source groups are synthetic; no real validation split
is inferred. A material-override instance pass measures visible pixels. Images
with unresolved visible targets are not exported as empty YOLO labels.
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector
import numpy as np

ROOT = Path.cwd().resolve()
SCRIPTS = ROOT / 'datasets/scripts/fire'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard
from datasets.scripts.common.bayesian import plan_production
from datasets.scripts.common.config import load_task_config, source_path
from datasets.scripts.common.paths import sample_dir, split_dir
from datasets.scripts.fire.fire_blender_labels import decode_ids, explicit_scenario, fire_label_suppression, fire_scale_suppression, front_facade_slots, label_state, paper_depth_supported, rectangle_uncovered_area, rotated_paper_envelope, ocr_status, placement_violations, scenario, srgb_to_linear, visible_bbox, yolo_row

COLLECTIONS = {'A': 'A - 开心大厦', 'B': 'B - 锦绣大厦', 'C': 'C - 晨光大厦'}
WORD_PARTS = ('青', '禾', '安', '和', '乐', '明', '新', '宏', '瑞', '锦', '泽', '云',
              '景', '晨', '华', '康', '福', '泰', '阳', '祥', '雅', '宁', '兴', '嘉')
NAME_SUFFIXES = ('大厦', '大楼', '中心', '公寓', '商厦')
FIRE_PLACEMENT_JITTER_M = .002
FIRE_ROTATION_MAX_RAD = .09
FIRE_PAPER_RIPPLE_MAX_M = .00003
SUPPORTED_C_PLACEMENTS = ('c_supported_left', 'c_supported_right')


def parse_args():
    parser = argparse.ArgumentParser(description='渲染楼名与火焰可见实例候选')
    parser.add_argument('--output', required=True)
    parser.add_argument('--production-plan', required=True,
                        help='workspace JSON declaration with reviewed sources, explicit priors, constraints and states')
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=431)
    parser.add_argument('--resolution', type=int, default=640)
    parser.add_argument('--height', type=int, help='image height; defaults to resolution for legacy square renders')
    parser.add_argument('--save-preview-scene', action='store_true',
                        help='save a copy of the first nonempty candidate scene for inspection')
    parser.add_argument('--vehicle-pilot', action='store_true',
                        help='render a fixed two-image exposed-background development comparison')
    parser.add_argument('--vehicle-oblique-pilot', action='store_true',
                        help='fit a wider building side for the bounded vehicle pilot')
    parser.add_argument('--vehicle-background-manifest',
                        help='workspace JSON of reviewed fire/name-negative real backgrounds')
    parser.add_argument('--current-geometry', help='current database-backed photo camera/ground review JSON')
    parser.add_argument('--review-only', action='store_true',
                        help='write a visual-review pilot to the unassigned split')
    parser.add_argument('--scenario-offset', type=int, default=0,
                        help='offset the ten-state schedule for a separately tracked batch')
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--device', choices=('CPU', 'OPTIX', 'CUDA'), default='OPTIX')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    if args.vehicle_pilot:
        parser.error('--vehicle-pilot is disabled: its P12_TL2 background contains traffic-signal targets and cannot be composited with buildings')
    args.output = (ROOT / args.output).resolve()
    source = Path(bpy.data.filepath).resolve()
    if source != source_path('fire', 'scene'):
        parser.error('loaded scene must match fire/config/source.yaml scene')
    if args.output not in {sample_dir('fire').resolve(), split_dir('fire', 'train').resolve()}:
        parser.error('output must be the fire sample or train directory')
    if args.save_preview_scene:
        parser.error('scene snapshots are not dataset outputs; use the source-asset maintenance workflow')
    args.height = args.resolution if args.height is None else args.height
    if (not 1 <= args.count <= 100000 or not 256 <= args.resolution <= 1280 or
            not 256 <= args.height <= 1280 or min(args.resolution, args.height) > 720 or
            not 1 <= args.samples <= 256):
        parser.error('count、resolution 或 samples 超出范围')
    if args.vehicle_pilot and (args.count != 2 or (args.resolution, args.height) != (640, 480)):
        parser.error('vehicle-pilot requires exactly 2 images at 640x480')
    if args.vehicle_oblique_pilot and not args.vehicle_pilot:
        parser.error('vehicle-oblique-pilot requires --vehicle-pilot')
    if args.vehicle_background_manifest and args.vehicle_pilot:
        parser.error('vehicle-background-manifest and vehicle-pilot are separate modes')
    if args.vehicle_background_manifest and (args.resolution, args.height) != (640, 480):
        parser.error('vehicle-background-manifest requires 640x480 output')
    if args.scenario_offset < 0:
        parser.error('scenario-offset must be non-negative')
    return args, source


def output_partition(args, backgrounds):
    return ('unassigned' if args.review_only else
            'regression' if args.vehicle_pilot or
            (backgrounds and any(item['usage'] == 'regression' for item in backgrounds)) else
            'unassigned' if backgrounds else 'train')


def sample_real_scene(backgrounds, rng):
    """Choose a source frame and a discrete simulated tone from declared uniform sets."""
    return rng.choice(backgrounds), rng.choice((-.5, 0.0)), rng.choice((.65, 1.0))


def read_production_declaration(path, args, backgrounds):
    """Validate the user declaration against states this renderer can execute."""
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    if path.is_symlink():
        raise ValueError('production plan symlinks are not supported')
    path = path.resolve()
    if not path.is_relative_to(ROOT) or not path.is_file() or path.suffix.lower() != '.json':
        raise ValueError('production plan must be an existing workspace JSON file')
    declaration = json.loads(path.read_text(encoding='utf-8-sig'))
    allowed_declaration_fields = {'schema_version', 'model_key', 'target_split', 'purpose', 'budget',
                                  'uncertainty_weight', 'independence_basis', 'constraints', 'strata'}
    if not isinstance(declaration, dict) or set(declaration) != allowed_declaration_fields:
        raise ValueError('production declaration has missing or unsupported top-level fields')
    required_constraints = {
        'route_context': 'building_facade',
        'traffic_signal_cooccurrence': 'forbidden_separate_route',
        'traffic_signal_generation': 'no_traffic_signal_injection',
        'unknown_visual_targets': 'suppress_labels',
    }
    if declaration.get('constraints') != required_constraints:
        raise ValueError('unsupported production constraints; traffic signals are a separate-route state and are never composited with buildings')
    if declaration.get('model_key') != 'fire_sticker' or declaration.get('target_split') != 'train':
        raise ValueError('this renderer accepts only fire_sticker train-candidate plans')
    if declaration.get('budget') != args.count:
        raise ValueError('production-plan budget must equal --count')
    if args.scenario_offset != 0:
        raise ValueError('planned production uses explicit scenario_index states; scenario-offset must be zero')
    expected_sources = ['blender_building_library', 'fire_sticker_artwork']
    if args.vehicle_pilot:
        expected_sources.append('real_car_capture_background')
    if backgrounds:
        expected_sources.append('reviewed_real_building_background')
    expected_sources = sorted(expected_sources)
    supported_indices = set(range(10))
    states = declaration.get('strata')
    if not isinstance(states, list) or not states:
        raise ValueError('production plan must declare executable scenario strata')
    seen = set()
    for state in states:
        if not isinstance(state, dict):
            raise ValueError('production state must be an object')
        allowed_state_fields = {'id', 'failure_event', 'min_count', 'max_count', 'prior',
                                'exposure_weight', 'failure_cost', 'conditions', 'background',
                                'target', 'labels', 'augmentation', 'sources', 'parameters'}
        if set(state) != allowed_state_fields:
            raise ValueError('production state has missing or unsupported fields')
        target = state.get('target')
        hybrid_id = state.get('background', {}).get('photo_geometry_id')
        hybrid = getattr(args, 'photo_geometries', {}).get(hybrid_id) if hybrid_id else None
        if hybrid_id and hybrid is None:
            raise ValueError('Production state refers to an unknown photo geometry fit')
        if not isinstance(target, dict):
            raise ValueError('unsupported Blender scenario state')
        is_explicit = target.get('scenario') == 'explicit'
        if is_explicit:
            if args.vehicle_pilot:
                raise ValueError('explicit scenarios do not support vehicle pilot')
            expected = explicit_scenario(target)
            if expected['placement'] in ('c_supported_left', 'c_supported_right') and backgrounds:
                raise ValueError('supported C wall states require the standalone building scene')
            scenario_index = None
            state_key = ('explicit', tuple(expected['active']), tuple(expected['counts']),
                         expected['view'], expected['placement'], expected['kind'])
        else:
            scenario_index = target.get('scenario_index')
            if type(scenario_index) is not int or scenario_index not in supported_indices:
                raise ValueError('unsupported Blender scenario state')
            expected = scenario(scenario_index)
            active = list(expected['active'])
            counts = dict(zip(active, expected['counts']))
            expected_target = {'scenario': expected['kind'], 'scenario_index': scenario_index,
                               'active_buildings': active, 'fire_counts': counts}
            if target != expected_target:
                raise ValueError('scenario target fields do not match the renderer schedule')
            state_key = ('legacy', scenario_index)
        state_key = (*state_key, hybrid_id)
        if state_key in seen:
            raise ValueError('each executable Blender scenario may appear only once in the plan')
        seen.add(state_key)
        active = list(expected['active'])
        counts = dict(zip(active, expected['counts']))
        mode = 'fixed' if args.vehicle_pilot else 'seeded_random'
        expected_background = {'domain': 'real_scene_with_blender_building' if backgrounds else 'synthetic_building_arena',
                               'traffic_signal_injection': 'none'}
        if backgrounds:
            expected_background.update(source_ids=[item['id'] for item in backgrounds], sampling='seeded_uniform')
        if hybrid:
            expected_background = {'domain': 'calibrated_real_scene', 'traffic_signal_injection': 'none',
                                   'photo_geometry_id': hybrid_id, 'parent_group': hybrid['parent_group']}
        if state.get('background') != expected_background:
            raise ValueError('unsupported background state; this renderer does not add traffic signals to building scenes')
        expected_labels = {
            'fire': {'class_id': 0, 'format': 'yolo_v8', 'status': 'candidate_requires_visual_review'},
            'name': {'class_id': 0, 'format': 'yolo_v8', 'status': 'candidate_requires_visual_review'},
        }
        if state.get('labels') != expected_labels:
            raise ValueError('unsupported label policy')
        expected_augmentation = {'camera': expected['view'] if is_explicit else mode, 'lighting': mode,
                                 'sticker_placement': expected['placement'] if is_explicit else 'seeded_random',
                                 'traffic_signal_injection': 'none'}
        if backgrounds:
            expected_augmentation.update(tone_sampling='seeded_uniform',
                tone_profiles=[{'exposure_ev': e, 'saturation': s} for e in (-0.5, 0.0) for s in (0.65, 1.0)])
        if hybrid:
            expected_augmentation['lighting'] = 'photo_estimated_physical_shadow'
        if state.get('augmentation') != expected_augmentation:
            raise ValueError('unsupported augmentation state')
        expected_parameters = {'scenario_index': scenario_index, 'scenario_offset': args.scenario_offset,
                               'seed': args.seed, 'resolution': [args.resolution, args.height],
                               'samples': args.samples, 'device': args.device,
                               'mode': 'vehicle_pilot' if args.vehicle_pilot else 'standard'}
        if state.get('parameters') != expected_parameters:
            raise ValueError('renderer parameters in production plan differ from CLI settings')
        state_sources = sorted([*expected_sources, 'reviewed_real_building_background']) if hybrid else expected_sources
        if state.get('sources') != state_sources:
            raise ValueError('production state sources differ from actual renderer source groups')
        expected_conditions = ([f"scenario:explicit", f"visibility:{target['visibility']}",
                                'traffic_signal_injection:none'] if is_explicit else
                               [f"scenario:{expected['kind']}", 'traffic_signal_injection:none'])
        if state.get('conditions') != expected_conditions:
            raise ValueError('unsupported or unenforced production state conditions')
        if (not is_explicit and state.get('id') != f"blender_scenario_{scenario_index:02d}_{expected['kind']}"):
            raise ValueError('production state id must identify its executable Blender scenario')
        if is_explicit and (not isinstance(state.get('id'), str) or not state['id'].startswith('blender_explicit_')):
            raise ValueError('explicit scenario id must start with blender_explicit_')
        if state.get('failure_event') != 'renderer_state_or_label_uncertainty':
            raise ValueError('failure_event must describe this renderer’s supported review risk')
        if args.vehicle_pilot and sum(counts.values()) == 0 and state.get('max_count', 0) > 1:
            raise ValueError('fixed empty vehicle-pilot state supports at most one image; repeated identical images are forbidden')
    return path, declaration


def configure_device(scene, requested):
    scene.render.engine = 'CYCLES'
    if requested == 'CPU':
        scene.cycles.device = 'CPU'
        return ['CPU']
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = requested
    prefs.get_devices()
    chosen = [d for d in prefs.devices if d.type == requested]
    if not chosen:
        raise RuntimeError(f'没有可用的 {requested} 设备')
    for device in prefs.devices:
        device.use = device.type == requested
    scene.cycles.device = 'GPU'
    return [d.name for d in chosen]


def get_source():
    groups = {}
    for key, collection_name in COLLECTIONS.items():
        collection = bpy.data.collections.get(collection_name)
        if collection is None:
            raise ValueError(f'场景缺少楼体集合: {collection_name}')
        signs = [o for o in collection.objects if o.name.startswith('Name paper - ')]
        text = [o for o in collection.objects if o.type == 'FONT' and o.name.startswith('Editable text - ')]
        originals = [o for o in collection.objects if o.name.startswith('Fire paper ')]
        if len(signs) != 1 or len(text) != 1 or len(originals) != 3:
            raise ValueError(f'{collection_name} 的纸条、文字或火焰模板数量不匹配')
        groups[key] = {'collection': collection, 'sign': signs[0], 'text': text[0],
                       'originals': originals, 'x': float(text[0].location.x)}
    return groups


def verify_c_supported_wall(groups):
    """Reject if the loaded C wall cannot support every bounded paper footprint."""
    artwork = bpy.data.images.get('image1.png')
    if (artwork is None or artwork.packed_file is None or
            bytes(artwork.packed_file.data) != (ROOT / 'Archive/Assets/working_sources/digital/fire/team_assets/image1.png').read_bytes()):
        raise ValueError('packed fire artwork does not match the official image')
    originals = {obj for group in groups.values() for obj in group['originals']}
    textured = set()
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        for material in obj.data.materials:
            if material and material.use_nodes and any(
                    node.type == 'TEX_IMAGE' and node.image == artwork for node in material.node_tree.nodes):
                textured.add(obj)
    if textured != originals:
        raise ValueError('scene contains missing or untracked official-fire textures')
    group = groups['C']
    wall = bpy.data.objects.get('C_front_small_holes')
    if wall is None or wall not in group['collection'].objects[:] or wall.type != 'MESH' or wall.modifiers:
        raise ValueError('C wall support mesh is missing or modified')
    vertices = [wall.matrix_world @ vertex.co for vertex in wall.data.vertices]
    if not vertices:
        raise ValueError('C wall support mesh is empty')
    front_y = min(v.y for v in vertices)
    rectangles = []
    for polygon in wall.data.polygons:
        points = [vertices[i] for i in polygon.vertices]
        if any(abs(v.y - front_y) > 1e-6 for v in points):
            continue
        if len(points) != 4:
            raise ValueError('C front support face is not an axis-aligned quad')
        x0, x1 = min(v.x for v in points), max(v.x for v in points)
        z0, z1 = min(v.z for v in points), max(v.z for v in points)
        if x1 - x0 <= 1e-8 or z1 - z0 <= 1e-8:
            continue
        corners = {(round(v.x, 6), round(v.z, 6)) for v in points}
        if corners != {(round(x, 6), round(z, 6)) for x in (x0, x1) for z in (z0, z1)}:
            raise ValueError('C front support face is not rectangular')
        rectangles.append((x0, z0, x1, z1))
    if not rectangles:
        raise ValueError('C front support faces are missing')
    footprints = []
    for template in group['originals']:
        if (template.parent is not None or template.type != 'MESH' or
                any(abs(v) > 1e-8 for v in (*template.location, *template.rotation_euler)) or
                any(abs(v - 1) > 1e-8 for v in template.scale) or not template.data.vertices or
                template.data.shape_keys is not None):
            raise ValueError('C paper template transform or mesh is unsupported')
        padding = 0.0
        for modifier in template.modifiers:
            if (modifier.type != 'SOLIDIFY' or modifier.use_even_offset or
                    getattr(modifier, 'solidify_mode', 'EXTRUDE') != 'EXTRUDE' or
                    not math.isfinite(modifier.thickness) or not math.isfinite(modifier.offset)):
                raise ValueError('C paper modifier cannot be bounded for support')
            # Normal displacement cannot exceed this conservative thickness bound.
            padding += abs(modifier.thickness) * (1 + abs(modifier.offset))
        if not paper_depth_supported(front_y, min(v.co.y for v in template.data.vertices),
                                     max(v.co.y for v in template.data.vertices),
                                     padding, FIRE_PAPER_RIPPLE_MAX_M):
            raise ValueError('C paper template is too far from or behind the supporting wall')
        center = sum((v.co for v in template.data.vertices), Vector()) / len(template.data.vertices)
        half_x = max(abs(v.co.x - center.x) for v in template.data.vertices) + padding
        half_z = max(abs(v.co.z - center.z) for v in template.data.vertices) + padding
        envelope_x, envelope_z = rotated_paper_envelope(
            half_x, half_z, FIRE_PLACEMENT_JITTER_M, FIRE_ROTATION_MAX_RAD)
        for placement in SUPPORTED_C_PLACEMENTS:
            for x, z in front_facade_slots(placement):
                box = (group['x'] + x - envelope_x, z - envelope_z,
                       group['x'] + x + envelope_x, z + envelope_z)
                if rectangle_uncovered_area(rectangles, box) > 1e-12:
                    raise ValueError('C paper footprint is not fully supported by the front wall')
                footprints.append({'template': template.name, 'placement': placement,
                                   'center_x_m': float(group['x'] + x), 'center_z_m': float(z),
                                   'envelope_xz_m': [float(value) for value in box]})
    return {'support_object': wall.name, 'front_plane_y_m': float(front_y),
            'positions': list(SUPPORTED_C_PLACEMENTS), 'jitter_m': FIRE_PLACEMENT_JITTER_M,
            'rotation_abs_rad': FIRE_ROTATION_MAX_RAD, 'support': 'bounded_full_paper_envelope',
            'footprints': footprints}


def make_name(rng):
    count = rng.choice((2, 2, 3))
    return ''.join(rng.choice(WORD_PARTS) for _ in range(count)) + rng.choice(NAME_SUFFIXES)


def reset_name(obj, value):
    obj.data.body = value
    obj.scale = (1, 1, 1)
    bpy.context.view_layer.update()
    if obj.dimensions.x > .115:
        obj.scale *= .115 / obj.dimensions.x


def create_stickers(groups):
    """Clone paper meshes, keeping their UVs, materials and shape unmodified."""
    for key, group in groups.items():
        pool = []
        for index in range(5):
            template = group['originals'][index % 3]
            copy = template.copy()
            copy.data = template.data.copy()
            copy.name = f'Candidate fire {key} {index + 1:02d}'
            center = sum((v.co for v in copy.data.vertices), Vector()) / len(copy.data.vertices)
            for vertex in copy.data.vertices:
                vertex.co.x -= center.x
                vertex.co.z -= center.z
            copy.location = (center.x, 0, center.z)
            group['collection'].objects.link(copy)
            pool.append(copy)
        for original in group['originals']:
            original.hide_render = True
        group['pool'] = pool
        group['paper_base_y'] = [[float(v.co.y) for v in obj.data.vertices] for obj in pool]
        front = bpy.data.objects[{'A': 'A_front_window_bands', 'B': 'B_front_glazed_openings',
                                  'C': 'C_front_small_holes'}[key]]
        heights = [(front.matrix_world @ vertex.co).z for vertex in front.data.vertices]
        group['facade_height_m'] = max(heights) - min(heights)
        group['wall_front_y_m'] = min((front.matrix_world @ vertex.co).y for vertex in front.data.vertices)


def place_stickers(group, count, rng, placement='all', size_policy=None, building=None, wall_offset=None):
    # Each footprint is separated from its neighbours by more than the paper
    # width or height, so anti-aliased instance masks do not blend targets.
    slots = front_facade_slots(placement)
    selected = [(dx, z, 1, None) for dx, z in rng.sample(slots, count)]
    if size_policy is not None:
        lower, upper = size_policy['height_fraction'][building]
        for attempt in range(size_policy['placement_attempts']):
            selected, boxes = [], []
            for index, (dx, z) in enumerate(rng.sample(slots, count)):
                template = group['originals'][index % len(group['originals'])]
                fraction = rng.uniform(lower, upper)
                multiplier = group['facade_height_m'] * fraction / template.dimensions.z
                half_x, half_z = rotated_paper_envelope(
                    template.dimensions.x * multiplier / 2, template.dimensions.z * multiplier / 2,
                    FIRE_PLACEMENT_JITTER_M, FIRE_ROTATION_MAX_RAD)
                box = (dx-half_x, z-half_z, dx+half_x, z+half_z)
                if any(box[0] < other[2] and other[0] < box[2] and
                       box[1] < other[3] and other[1] < box[3] for other in boxes):
                    break
                boxes.append(box)
                selected.append((dx, z, multiplier, fraction))
            if len(selected) == count:
                break
        if len(selected) != count:
            raise ValueError(f'No separated scaled paper layout for building {building}')
    for index, obj in enumerate(group['pool']):
        obj.hide_render = index >= count
        if index < count:
            dx, z, multiplier, fraction = selected[index]
            obj.scale = (multiplier, 1, multiplier)
            obj['scale_from_reference'] = multiplier
            if fraction is not None:
                obj['height_fraction'] = fraction
                obj['reference_facade_height_m'] = group['facade_height_m']
            obj.location.x = group['x'] + dx + rng.uniform(-FIRE_PLACEMENT_JITTER_M, FIRE_PLACEMENT_JITTER_M)
            obj.location.z = z + rng.uniform(-FIRE_PLACEMENT_JITTER_M, FIRE_PLACEMENT_JITTER_M)
            obj.rotation_euler.y = rng.uniform(-FIRE_ROTATION_MAX_RAD, FIRE_ROTATION_MAX_RAD)
            # Tiny independent ripple at the paper edge; preserve its centre,
            # width, height, UVs, measured template thickness and nominal shape.
            amplitude = (rng.uniform(-FIRE_PAPER_RIPPLE_MAX_M, FIRE_PAPER_RIPPLE_MAX_M)
                         if wall_offset is None else 0)
            baseline = group['paper_base_y'][index]
            for j, vertex in enumerate(obj.data.vertices):
                u, v = (j % 13) / 12.0, (j // 13) / 12.0
                edge = max(abs(2*u-1), abs(2*v-1))
                vertex.co.y = (baseline[j] + amplitude * math.sin(3*math.pi*u) * math.sin(2*math.pi*v) * edge**2
                               if wall_offset is None else group['wall_front_y_m'] - wall_offset - obj.location.y)
            obj.data.update()
            obj['paper_ripple_m'] = amplitude
            if wall_offset is not None:
                obj['wall_plane_y_m'] = group['wall_front_y_m']
                obj['wall_offset_m'] = wall_offset


def set_camera(scene, groups, active, kind, rng, view=None):
    camera = scene.camera
    if camera is None:
        raise ValueError('场景没有相机')
    if view and view.startswith('orange_peel_'):
        from datasets.scripts.fire.camera_trajectory import apply_pose, orange_peel_pose
        config = load_task_config('fire')[1]['camera_trajectory']
        centers = [groups[key]['x'] for key in active]
        pose = orange_peel_pose(config, int(view.rsplit('_', 1)[1]),
                                sum(centers) / len(centers), max(centers) - min(centers))
        camera.animation_data_clear()
        apply_pose(camera, pose)
        return pose
    single = len(active) == 1
    center = groups[active[0]]['x'] if single else sum(groups[k]['x'] for k in active) / len(active)
    distance = (.78 if kind == 'truncated' else
                rng.uniform(.75, 1.10) if single else rng.uniform(1.25, 1.70))
    if view is None:
        x_offset = rng.uniform(-.16, .16)
    elif view == 'near_frontal':
        x_offset = distance * math.tan(math.radians(rng.uniform(-8, 8)))
    elif view in ('oblique_left', 'oblique_right'):
        angle = rng.uniform(15, 25) * (-1 if view == 'oblique_left' else 1)
        x_offset = distance * math.tan(math.radians(angle))
    else:
        raise ValueError('unsupported camera view')
    camera.location = (center + x_offset, -distance, rng.uniform(.14, .34))
    look_x = center + (rng.choice((-.19, .19)) if kind == 'truncated' else rng.uniform(-.02, .02))
    look = Vector((look_x, 0, rng.uniform(.18, .23)))
    camera.rotation_euler = (look - camera.location).to_track_quat('-Z', 'Y').to_euler()
    camera.data.lens = 56 if kind == 'truncated' else rng.uniform(32, 44) if single else rng.uniform(28, 36)
    camera.data.clip_start = .001
    if hasattr(camera.data, 'dof'):
        camera.data.dof.use_dof = False
    return {'position_m': list(camera.location), 'target_m': list(look),
            'focal_length_mm': float(camera.data.lens), 'distortion': 'not modelled',
            'view_profile': view or 'legacy_seeded_random'}


def set_lighting(rng):
    values = {}
    for name, factor in (('Broad softbox', (0.75, 1.15)), ('Soft fill', (.7, 1.2))):
        light = bpy.data.objects.get(name)
        if light is None or light.type != 'LIGHT':
            raise ValueError(f'场景缺少灯光: {name}')
        if '_fire_baseline_energy' not in light:
            light['_fire_baseline_energy'] = light.data.energy
        light.data.energy = float(light['_fire_baseline_energy']) * rng.uniform(*factor)
        values[name] = float(light.data.energy)
    return values


def vehicle_background(scene, path=None):
    """Place a real frame on a camera-aligned plane behind the rendered scene."""
    path = path or (ROOT / 'Archive/Assets/captures/car_real_capture_20261112/P12_TL2.png')
    if not path.is_file():
        raise FileNotFoundError('P12 car background missing')
    camera = scene.camera
    frame = camera.data.view_frame(scene=scene)
    depth = 4.0
    corners = [(v.x * depth / -v.z, v.y * depth / -v.z, -depth) for v in frame]
    mesh = bpy.data.meshes.new('MESH-vehicle-background')
    mesh.from_pydata(corners, [], [(0, 1, 2, 3)])
    mesh.update()
    uv = mesh.uv_layers.new(name='Photo UV')
    xmin, xmax = min(v[0] for v in corners), max(v[0] for v in corners)
    ymin, ymax = min(v[1] for v in corners), max(v[1] for v in corners)
    for poly in mesh.polygons:
        for index in poly.loop_indices:
            vertex = mesh.vertices[mesh.loops[index].vertex_index].co
            uv.data[index].uv = ((vertex.x - xmin) / (xmax - xmin), (vertex.y - ymin) / (ymax - ymin))
    obj = bpy.data.objects.new('GEO-vehicle-background', mesh)
    scene.collection.objects.link(obj)
    obj.parent = camera
    obj.location = (0, 0, 0)
    mat = bpy.data.materials.new('MAT-vehicle-background-photo')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    texture = nodes.new('ShaderNodeTexImage')
    texture.name = 'AEC-vehicle-background-image'
    texture.image = bpy.data.images.load(str(path), check_existing=True)
    texture.image.pack()
    emission = nodes.new('ShaderNodeEmission')
    output = nodes.new('ShaderNodeOutputMaterial')
    mat.node_tree.links.new(texture.outputs['Color'], emission.inputs['Color'])
    mat.node_tree.links.new(emission.outputs[0], output.inputs['Surface'])
    obj.data.materials.append(mat)
    set_vehicle_background(scene, path)
    return path.relative_to(ROOT).as_posix()


def set_vehicle_background(scene, path):
    """Center-crop to the render aspect without stretching the source image."""
    obj = bpy.data.objects['GEO-vehicle-background']
    texture = obj.data.materials[0].node_tree.nodes.get('AEC-vehicle-background-image')
    if texture is None:
        raise ValueError('vehicle background image node missing')
    image = bpy.data.images.load(str(path), check_existing=True)
    texture.image = image
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError('invalid background image size')
    # The lens and aspect may have changed since the background was created.
    # Rebuild the camera-space plane from the current view, or gray borders
    # appear when sampling different focal lengths.
    depth = 4.0
    frame = scene.camera.data.view_frame(scene=scene)
    for vertex, corner in zip(obj.data.vertices, frame):
        vertex.co = (corner.x * depth / -corner.z, corner.y * depth / -corner.z, -depth)
    obj.data.update()
    source_aspect = width / height
    target_aspect = scene.render.resolution_x / scene.render.resolution_y
    u_span = min(1.0, target_aspect / source_aspect)
    v_span = min(1.0, source_aspect / target_aspect)
    xmin, xmax = min(v.co.x for v in obj.data.vertices), max(v.co.x for v in obj.data.vertices)
    ymin, ymax = min(v.co.y for v in obj.data.vertices), max(v.co.y for v in obj.data.vertices)
    for poly in obj.data.polygons:
        for loop_index in poly.loop_indices:
            vertex = obj.data.vertices[obj.data.loops[loop_index].vertex_index].co
            u = (vertex.x - xmin) / (xmax - xmin)
            v = (vertex.y - ymin) / (ymax - ymin)
            obj.data.uv_layers.active.data[loop_index].uv = ((1-u_span)/2 + u*u_span,
                                                             (1-v_span)/2 + v*v_span)
    return {'source_size': [width, height], 'center_crop_uv': [(1-u_span)/2, (1-v_span)/2,
                                                               (1+u_span)/2, (1+v_span)/2]}


def read_vehicle_backgrounds(path):
    manifest = (ROOT / path).resolve()
    if not manifest.is_relative_to(ROOT / 'datasets') or not manifest.is_file():
        raise ValueError('background manifest must be an existing file under datasets')
    data = json.loads(manifest.read_text(encoding='utf8'))
    backgrounds = data.get('backgrounds')
    if not isinstance(backgrounds, list) or not backgrounds:
        raise ValueError('background manifest has no reviewed images')
    allowed = {'unassigned', 'regression'}
    for item in backgrounds:
        if (not isinstance(item, dict) or not isinstance(item.get('id'), str) or not item['id'] or
                not isinstance(item.get('source_group'), str) or not item['source_group'] or
                item.get('usage') not in allowed or item.get('fire_state') != 'complete_negative' or
                item.get('name_state') != 'complete_negative' or not isinstance(item.get('distractors'), list)):
            raise ValueError('each background needs reviewed negative states, source and usage')
        if any('traffic' in str(d).lower() and ('light' in str(d).lower() or 'signal' in str(d).lower())
               for d in item['distractors']):
            raise ValueError('traffic-signal backgrounds cannot be combined with buildings')
        shift = item.get('target_shift_x', 0.0)
        if isinstance(shift, bool) or not isinstance(shift, (int, float)) or not math.isfinite(shift) or abs(shift) > .3:
            raise ValueError('background target_shift_x must be within -0.3..0.3')
        placement = item.get('placement')
        if not isinstance(placement, dict):
            raise ValueError('each background requires reviewed ground placement and foreground exclusions')
        anchor, heights = placement.get('ground_anchor_xyxy'), placement.get('building_height_px')
        forbidden, overlap = placement.get('forbidden_rects_xyxy'), placement.get('max_overlap_fraction')
        valid_rect = lambda r: (isinstance(r, list) and len(r) == 4 and
                                all(type(x) in (int, float) and math.isfinite(x) for x in r) and
                                0 <= r[0] < r[2] <= 640 and 0 <= r[1] < r[3] <= 480)
        if (not valid_rect(anchor) or not isinstance(heights, list) or len(heights) != 2 or
                any(type(x) not in (int, float) or not math.isfinite(x) for x in heights) or
                not 0 < heights[0] <= heights[1] <= 960 or not isinstance(forbidden, list) or
                any(not valid_rect(rect) for rect in forbidden) or type(overlap) not in (int, float) or
                not math.isfinite(overlap) or not 0 <= overlap <= .1 or
                not isinstance(placement.get('review_evidence'), str)):
            raise ValueError('invalid reviewed ground anchor, target scale or foreground exclusions')
        evidence = (ROOT / placement['review_evidence']).resolve()
        if not evidence.is_relative_to(ROOT) or evidence.relative_to(ROOT).as_posix() != placement['review_evidence'] or not evidence.is_file():
            raise ValueError('placement review evidence must be a canonical workspace file')
        for field in ('image', 'review_evidence'):
            value = item.get(field)
            if not isinstance(value, str):
                raise ValueError('background image/evidence path is missing')
            candidate = (ROOT / value).resolve()
            if not candidate.is_relative_to(ROOT) or candidate.relative_to(ROOT).as_posix() != value or not candidate.is_file():
                raise ValueError('background image/evidence must be a canonical workspace file')
    if len({item['id'] for item in backgrounds}) != len(backgrounds):
        raise ValueError('background IDs must be unique')
    return manifest, backgrounds


def vehicle_tone_compositor(scene):
    """Saturation is a documented simulation, not a calibrated camera value."""
    nodes = bpy.data.node_groups.new('V4 vehicle tone compositor', 'CompositorNodeTree')
    nodes.interface.new_socket(name='Image', in_out='OUTPUT', socket_type='NodeSocketColor')
    layers = nodes.nodes.new('CompositorNodeRLayers')
    tone = nodes.nodes.new('CompositorNodeHueSat')
    output = nodes.nodes.new('NodeGroupOutput')
    nodes.links.new(layers.outputs['Image'], tone.inputs['Image'])
    nodes.links.new(tone.outputs['Image'], output.inputs['Image'])
    scene.compositing_node_group = nodes
    scene.render.use_compositing = True
    return tone


def vehicle_camera(scene, group, oblique=False):
    """Approximate P10 perspective and target size; parameters are not measured."""
    camera = scene.camera
    center = group['x']
    camera.location = (center - (.80 if oblique else .34), -.85, .28)
    target = Vector((center, 0, .20))
    camera.rotation_euler = (target - camera.location).to_track_quat('-Z', 'Y').to_euler()
    camera.data.lens = 48 if oblique else 38
    camera.data.clip_start = .001
    camera.data.dof.use_dof = False
    return {'position_m': list(camera.location), 'target_m': list(target),
            'focal_length_mm': float(camera.data.lens), 'distortion': 'not modelled',
            'calibration_status': 'visual approximation, not measured'}


def vehicle_reference_details(scene, group, oblique=False):
    """Add a dark inner wall and a restrained approximate floor shadow."""
    x = group['x']
    # P10's name paper occupies a smaller fraction of the facade than the
    # source prototype. Match its apparent width while retaining paper height.
    name_width_factor = .80 if oblique else .65
    for vertex in group['sign'].data.vertices:
        vertex.co.x = x + (vertex.co.x - x) * name_width_factor
    group['sign'].data.update()
    mesh = bpy.data.meshes.new('MESH-vehicle-inner-wall')
    mesh.from_pydata([(x-.098, -.059, .003), (x+.098, -.059, .003),
                      (x+.098, -.059, .397), (x-.098, -.059, .397)], [], [(0, 1, 2, 3)])
    mesh.update()
    wall = bpy.data.objects.new('GEO-vehicle-inner-wall', mesh)
    group['collection'].objects.link(wall)
    material = bpy.data.materials.new('MAT-vehicle-inner-wall')
    material.diffuse_color = (.028, .03, .03, 1)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (.028, .03, .03, 1)
    bsdf.inputs['Roughness'].default_value = .9
    wall.data.materials.append(material)
    if oblique:
        side_mesh = bpy.data.meshes.new('MESH-vehicle-inner-side')
        side_mesh.from_pydata([(x-.098, -.060, .003), (x-.098, .060, .003),
                               (x-.098, .060, .397), (x-.098, -.060, .397)], [], [(0, 1, 2, 3)])
        side_mesh.update()
        side_wall = bpy.data.objects.new('GEO-vehicle-inner-side', side_mesh)
        group['collection'].objects.link(side_wall)
        side_wall.data.materials.append(material)

    vertices = [(x, 0, .00015)]
    vertices += [(x + .14 * math.cos(2*math.pi*i/48), .095 * math.sin(2*math.pi*i/48), .00015)
                 for i in range(48)]
    faces = [(0, i+1, (i+1)%48+1) for i in range(48)]
    mesh = bpy.data.meshes.new('MESH-vehicle-contact-shadow')
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    shadow = bpy.data.objects.new('GEO-vehicle-contact-shadow', mesh)
    scene.collection.objects.link(shadow)
    mat = bpy.data.materials.new('MAT-vehicle-contact-shadow')
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    coord = nodes.new('ShaderNodeTexCoord')
    distance = nodes.new('ShaderNodeVectorMath'); distance.operation = 'DISTANCE'
    distance.inputs[1].default_value = (.5, .5, .5)
    links.new(coord.outputs['Generated'], distance.inputs[0])
    fade = nodes.new('ShaderNodeMapRange')
    fade.inputs['From Min'].default_value = 0
    fade.inputs['From Max'].default_value = .5
    fade.inputs['To Min'].default_value = .35
    fade.inputs['To Max'].default_value = 0
    fade.clamp = True
    links.new(distance.outputs['Value'], fade.inputs['Value'])
    transparent = nodes.new('ShaderNodeBsdfTransparent')
    dark = nodes.new('ShaderNodeEmission'); dark.inputs['Color'].default_value = (0, 0, 0, 1)
    mix = nodes.new('ShaderNodeMixShader')
    links.new(fade.outputs['Result'], mix.inputs[0])
    links.new(transparent.outputs[0], mix.inputs[1]); links.new(dark.outputs[0], mix.inputs[2])
    output = nodes.new('ShaderNodeOutputMaterial'); links.new(mix.outputs[0], output.inputs['Surface'])
    shadow.data.materials.append(mat)


def set_blocker(scene, groups, active, kind):
    blocker = bpy.data.objects.get('Synthetic occlusion prop')
    if blocker is None:
        bpy.ops.mesh.primitive_cube_add(size=1)
        blocker = bpy.context.object
        blocker.name = 'Synthetic occlusion prop'
        mat = bpy.data.materials.new('Synthetic neutral occluder')
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        shader = nodes.get('Principled BSDF')
        if shader is None:
            shader = nodes.new('ShaderNodeBsdfPrincipled')
        output = nodes.get('Material Output')
        if output is None:
            output = nodes.new('ShaderNodeOutputMaterial')
        if not any(link.from_node == shader and link.to_node == output for link in mat.node_tree.links):
            mat.node_tree.links.new(shader.outputs['BSDF'], output.inputs['Surface'])
        shader.inputs['Base Color'].default_value = (.14, .15, .15, 1)
        blocker.data.materials.append(mat)
    blocker.hide_render = kind != 'occluder'
    if not blocker.hide_render:
        sticker = groups[active[0]]['pool'][0]
        center = sum((v.co for v in sticker.data.vertices), Vector()) / len(sticker.data.vertices)
        target = sticker.matrix_world @ center
        location = scene.camera.location.lerp(target, .65)
        location.x += .003
        blocker.location = location
        blocker.scale = (.014, .012, .09)
    return not blocker.hide_render


def projected_extent(scene, obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        if not mesh.vertices:
            return {'in_front': False, 'intersects': False, 'truncated': False, 'xyxy': None}
        points = [world_to_camera_view(scene, scene.camera, evaluated.matrix_world @ v.co) for v in mesh.vertices]
        in_front = all(p.z > 0 for p in points)
        xs = [float(p.x) for p in points]
        ys = [1.0 - float(p.y) for p in points]
        extent = [min(xs), min(ys), max(xs), max(ys)]
        intersects = extent[2] > 0 and extent[0] < 1 and extent[3] > 0 and extent[1] < 1
        truncated = min(extent[:2]) < 0 or max(extent[2:]) > 1
        return {'in_front': in_front, 'intersects': intersects, 'truncated': truncated, 'xyxy': extent}
    finally:
        evaluated.to_mesh_clear()


def building_pixel_boxes(scene, groups, active):
    """Project the actual facade and side bounding geometry in image pixels."""
    bpy.context.view_layer.update()
    boxes = []
    envelopes = {'A': ('A_front_window_bands', 'A_left', 'A_right', 'A_back'),
                 'B': ('B_front_glazed_openings', 'B_left', 'B_right', 'B_back'),
                 'C': ('C_front_small_holes', 'C_left_outer', 'C_right_outer', 'C_back_wing')}
    for key in active:
        wanted = set(envelopes[key])
        objects = [obj for obj in groups[key]['collection'].objects if obj.name in wanted]
        if len(objects) != 4:
            raise ValueError(f'{key}: building envelope objects missing')
        points = [world_to_camera_view(scene, scene.camera, obj.matrix_world @ Vector(corner))
                  for obj in objects for corner in obj.bound_box]
        if any(point.z <= 0 for point in points):
            return []
        boxes.append([min(p.x for p in points)*640, min(1-p.y for p in points)*480,
                      max(p.x for p in points)*640, max(1-p.y for p in points)*480])
    return boxes


def place_buildings_on_reviewed_ground(scene, groups, active, kind, placement, rng, max_attempts=24, view=None):
    """Fit projected scale and ground anchor; reject forbidden-object overlap."""
    anchor = placement['ground_anchor_xyxy']
    heights = placement['building_height_px']
    last_issues = ['no_projection']
    camera = scene.camera
    for attempt in range(1, max_attempts + 1):
        camera_record = set_camera(scene, groups, active, kind, rng, view=view)
        camera.data.shift_x = camera.data.shift_y = 0.0
        boxes = building_pixel_boxes(scene, groups, active)
        if not boxes:
            continue
        base_height = max(box[3]-box[1] for box in boxes)
        desired_height = rng.uniform(*heights)
        if base_height <= 0:
            continue
        camera.data.lens *= desired_height / base_height
        if not 18 <= camera.data.lens <= 70:
            last_issues = ['fitted_lens_outside_bounds']
            continue

        def projected_anchor():
            current = building_pixel_boxes(scene, groups, active)
            if not current:
                raise ValueError('building behind camera')
            return np.array([(min(b[0] for b in current)+max(b[2] for b in current))/2,
                             max(b[3] for b in current)], dtype=float)

        try:
            origin = projected_anchor()
            camera.data.shift_x = .01
            dx = (projected_anchor() - origin) / .01
            camera.data.shift_x = 0.0
            camera.data.shift_y = .01
            dy = (projected_anchor() - origin) / .01
            camera.data.shift_y = 0.0
            desired_anchor = np.array([rng.uniform(anchor[0], anchor[2]),
                                       rng.uniform(anchor[1], anchor[3])])
            shift = np.linalg.solve(np.column_stack((dx, dy)), desired_anchor - origin)
        except (ValueError, np.linalg.LinAlgError):
            last_issues = ['camera_anchor_fit_failed']
            continue
        if not np.isfinite(shift).all() or max(abs(shift)) > .6:
            last_issues = ['camera_shift_outside_bounds']
            continue
        camera.data.shift_x, camera.data.shift_y = map(float, shift)
        boxes = building_pixel_boxes(scene, groups, active)
        last_issues = placement_violations(boxes, placement)
        if not last_issues:
            camera_record.update({'focal_length_mm': float(camera.data.lens),
                'sensor_shift_xy': [float(camera.data.shift_x), float(camera.data.shift_y)],
                'calibration_status': 'image-space fit; camera parameters not measured'})
            return camera_record, {'status': 'accepted_geometric_candidate', 'attempts': attempt,
                'building_boxes_xyxy': boxes, 'target_anchor_xy': desired_anchor.tolist(),
                'target_height_px': desired_height, 'constraints': placement,
                'scope': '2D bounds only; object depth, contact shadow and physical fit require visual QA'}
    return None, {'status': 'rejected', 'attempts': max_attempts, 'violations': last_issues,
                  'constraints': placement}


def material_override():
    material = bpy.data.materials.new('Fire dataset instance-ID override')
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    info = nodes.new('ShaderNodeObjectInfo')
    emission = nodes.new('ShaderNodeEmission')
    output = nodes.new('ShaderNodeOutputMaterial')
    material.node_tree.links.new(info.outputs['Color'], emission.inputs['Color'])
    material.node_tree.links.new(emission.outputs[0], output.inputs['Surface'])
    return material


def render_id_pass(scene, targets, filename, mask_material):
    """Render an object-color pass with full scene depth and no illumination."""
    layer = bpy.context.view_layer
    old_override = layer.material_override
    objects = [o for o in scene.objects if o.type in {'MESH', 'FONT', 'CURVE', 'SURFACE', 'META'}]
    colors = [(o, tuple(o.color)) for o in objects]
    bg = next(node for node in scene.world.node_tree.nodes if node.type == 'BACKGROUND')
    old_world = tuple(bg.inputs['Color'].default_value)
    old_strength = float(bg.inputs['Strength'].default_value)
    old_view = scene.view_settings.view_transform
    old_exposure = scene.view_settings.exposure
    old_gamma = scene.view_settings.gamma
    old_samples = scene.cycles.samples
    old_denoise = scene.cycles.use_denoising
    old_compositing = scene.render.use_compositing
    try:
        for obj in objects:
            obj.color = (0, 0, 0, 1)
        for target in targets:
            target['object'].color = (1, target['id'] / 32.0, .25, 1)
        layer.material_override = mask_material
        bg.inputs['Color'].default_value = (0, 0, 0, 1)
        bg.inputs['Strength'].default_value = 0
        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.exposure = 0
        scene.view_settings.gamma = 1
        scene.cycles.samples = 1
        scene.cycles.use_denoising = False
        scene.render.use_compositing = False
        scene.render.filepath = str(filename)
        bpy.ops.render.render(write_still=True)
        image = bpy.data.images.load(str(filename), check_existing=False)
        try:
            width, height = image.size
            # Blender image.pixels exposes the stored PNG's sRGB-encoded
            # channel values, even though its color space is named sRGB. The
            # emitted object IDs were chosen in linear light and need decoding
            # before comparing ratios. Rows are bottom-up in Blender.
            pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape((height, width, 4))[::-1, :, :3]
            return decode_ids(srgb_to_linear(pixels), [t['id'] for t in targets])
        finally:
            bpy.data.images.remove(image)
    finally:
        for obj, color in colors:
            obj.color = color
        layer.material_override = old_override
        bg.inputs['Color'].default_value = old_world
        bg.inputs['Strength'].default_value = old_strength
        scene.view_settings.view_transform = old_view
        scene.view_settings.exposure = old_exposure
        scene.view_settings.gamma = old_gamma
        scene.cycles.samples = old_samples
        scene.cycles.use_denoising = old_denoise
        scene.render.use_compositing = old_compositing


def write_run(path, record):
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')


def scene_external_media():
    """List actual unpacked image/movie inputs of the loaded Blender scene."""
    paths = set()
    for image in bpy.data.images:
        if image.source not in {'FILE', 'MOVIE'}:
            continue
        if not image.filepath:
            if image.packed_file:
                continue
            raise ValueError('Unpacked Blender media has no file path: ' + image.name)
        path = Path(bpy.path.abspath(image.filepath, library=image.library)).resolve()
        if image.packed_file and (not path.is_relative_to(ROOT) or not path.is_file()):
            # Packed data are part of the tracked blend; do not fabricate an
            # external source when its recorded file is unavailable.
            continue
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError('Unpacked Blender media must exist within workspace: ' + image.name)
        paths.add(path)
    return paths


def scene_external_libraries():
    """Track linked Blender libraries as non-media procedural inputs."""
    paths = set()
    for library in bpy.data.libraries:
        path = Path(bpy.path.abspath(library.filepath, library=getattr(library, 'parent', None))).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError('Linked Blender library must exist within workspace: ' + library.name)
        paths.add(path)
    return paths


def configured_fire_artwork():
    """Require the configured artwork to be the actual loaded scene texture."""
    path = source_path('fire', 'target_artwork')
    permitted = (ROOT / 'Archive/Assets/working_sources/digital/fire/team_assets').resolve()
    if not path.is_relative_to(permitted):
        raise ValueError('fire artwork must come from the user-designated team_assets')
    expected = path.read_bytes()
    for image in bpy.data.images:
        if image.source != 'FILE':
            continue
        if image.packed_file:
            if bytes(image.packed_file.data) == expected:
                return path
        elif image.filepath and Path(bpy.path.abspath(image.filepath, library=image.library)).resolve() == path:
            return path
    raise ValueError('configured fire artwork is not present in the loaded scene')


def background_guard_inputs(background_manifest, backgrounds):
    """Resolve media ancestry and the reviewed configuration consumed by rendering."""
    media, procedural = set(), set()
    if background_manifest is None:
        return media, procedural
    procedural.add(background_manifest)
    suffixes = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp',
                '.mp4', '.mov', '.avi', '.mkv', '.m4v', '.webm'}

    def visit(value):
        if isinstance(value, dict):
            for v in value.values():
                visit(v)
        elif isinstance(value, list):
            for v in value:
                visit(v)
        elif isinstance(value, str) and Path(value).suffix.lower() in suffixes:
            candidate = (ROOT / value).resolve()
            if not candidate.is_relative_to(ROOT):
                raise ValueError('Background media escapes workspace: ' + value)
            media.add(candidate)
    for background in backgrounds:
        visit(background)
        for value in (background.get('review_evidence'), background['placement'].get('review_evidence')):
            path = (ROOT / value).resolve()
            if path.suffix.lower() not in suffixes:
                procedural.add(path)
    return media, procedural


def _produce(args, source, background_manifest, backgrounds, run):
    groups = get_source()
    scene = bpy.context.scene
    sticker_size_policy = load_task_config('fire')[1]['sample_recipe'].get('sticker_size')
    wall_projection = load_task_config('fire')[1]['sample_recipe'].get('wall_projection', {})
    support_geometry = None
    if any(state['target'].get('placement') in SUPPORTED_C_PLACEMENTS or
           state['target'].get('view', '').startswith('orange_peel_') for state in run.plan['strata']):
        if source != source_path('fire', 'scene'):
            raise ValueError('supported C wall states require the verified building scene')
        support_geometry = verify_c_supported_wall(groups)
    if (scene.world is None or not scene.world.use_nodes or
            not any(node.type == 'BACKGROUND' for node in scene.world.node_tree.nodes)):
        raise ValueError('场景没有可控的节点世界背景')
    devices = configure_device(scene, args.device)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('output is not empty')
    args.output.mkdir(parents=True, exist_ok=True)
    partition = output_partition(args, backgrounds)
    for folder in (f'images/{partition}', f'labels/fire/{partition}', 'instance_pass'):
        (args.output / folder).mkdir(parents=True)
    metadata = {'status': 'running', 'requested': args.count, 'completed': 0,
                'source_scene': source.relative_to(ROOT).as_posix(), 'seed': args.seed,
                'device': args.device, 'actual_devices': devices,
                'resolution': args.resolution, 'image_size': [args.resolution, args.height],
                'cycles_samples': args.samples,
                'purpose': 'visual_review_pilot_not_training' if args.review_only else 'dataset_candidate',
                'grouping': (
                    'visual-review only; exposed P12 photo and shared Blender scene; no independent split'
                    if args.review_only and args.vehicle_pilot else
                    'visual-review only; reviewed background groups with shared Blender scene; no independent split'
                    if args.review_only and backgrounds else
                    'single synthetic model family; visual-review only; no independent split'
                    if args.review_only else
                    'development regression: exposed P12 photo and shared Blender scene'
                    if args.vehicle_pilot else
                    'reviewed background groups with shared Blender scene; no independent split'
                    if backgrounds else
                    'single synthetic model family; train candidates only; no independent split'),
                'output_policy': 'unknown targets suppress class labels; occluded or truncated fire graphics suppress fire labels',
                'mask_encoding': 'RGB linear emission: R=1, G=instance_id/32, B=.25; black is background',
                'labels': 'candidate visible-pixel boxes; all output requires visual review',
                'input_source_refs': run.source_refs}
    if support_geometry is not None:
        metadata['verified_wall_support'] = support_geometry
        metadata['verified_wall_support_scope'] = 'unscaled_source_templates'
    if sticker_size_policy is not None:
        metadata['sticker_size_policy'] = sticker_size_policy
    if backgrounds:
        metadata.update(background_manifest=background_manifest.relative_to(ROOT).as_posix(),
                        background_count=len(backgrounds), scenario_offset=args.scenario_offset,
                        attempted=0, rejected_placement=0,
                        grouping='each background and its derivatives remain together; shared v13 Blender scene',
                        image_domain='real-scene composite with Blender render',
                        background_review='manifest declares complete-negative task review; audit source and split separately',
                        tone='2 exposure x 2 saturation simulated; not measured camera calibration')
    write_run(args.output / 'run.json', metadata)
    try:
        floor = bpy.data.objects.get('Presentation floor')
        if floor is None or floor.type != 'MESH':
            raise ValueError('场景缺少可调整的 Presentation floor')
        floor.scale.x *= 10
        floor.scale.y *= 10
        metadata['floor_horizontal_scale_multiplier'] = 10
        write_run(args.output / 'run.json', metadata)
        create_stickers(groups)
        baseline = None
        if args.current_geometry:
            from datasets.scripts.fire.hybrid_render import capture_scene
            baseline = capture_scene(scene)
        mask_material = material_override()
        rng = random.Random(args.seed)
        scene.render.resolution_x, scene.render.resolution_y = args.resolution, args.height
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGB'
        scene.render.image_settings.color_depth = '8'
        scene.render.dither_intensity = 0
        scene.render.film_transparent = False
        if args.vehicle_pilot:
            floor.hide_render = True
            vehicle_camera(scene, groups['A'], args.vehicle_oblique_pilot)
            metadata['background_photo'] = vehicle_background(scene)
            vehicle_reference_details(scene, groups['A'], args.vehicle_oblique_pilot)
            metadata['vehicle_oblique_pilot'] = args.vehicle_oblique_pilot
            metadata['background_review'] = 'P12 car diagnostic: no flame or name paper seen; already exposed'
            metadata['background_method'] = 'camera-aligned image plane behind Blender geometry'
            metadata['placement_note'] = ('photo floor with synthetic approximate contact shadow, dark inner wall, '
                                          f'and name paper at {0.80 if args.vehicle_oblique_pilot else 0.65:.2f} '
                                          'original width; physical fit not measured')
            write_run(args.output / 'run.json', metadata)
        elif backgrounds:
            floor.hide_render = True
            vehicle_background(scene, ROOT / backgrounds[0]['image'])
            for group in groups.values():
                vehicle_reference_details(scene, group, oblique=True)
            tone_compositor = vehicle_tone_compositor(scene)
            write_run(args.output / 'run.json', metadata)
    except Exception as exc:
        metadata['status'] = 'failed'
        metadata['error'] = f'{type(exc).__name__}: {exc}'
        write_run(args.output / 'run.json', metadata)
        raise
    try:
        with (args.output / 'manifest.jsonl').open('w', encoding='utf-8') as records:
            planned_states = run.plan['strata']
            scheduled = [(state, sample_index) for state in planned_states
                         for sample_index in range(state['planned_count'])]
            if len(scheduled) != args.count:
                raise ValueError('planned image total differs from --count')
            for index, (planned_state, state_sample_index) in enumerate(scheduled):
                sample_seed = args.seed + index
                rng = random.Random(sample_seed)
                target_state = planned_state['target']
                photo_geometry_id = planned_state.get('background', {}).get('photo_geometry_id')
                photo_review = args.photo_geometries.get(photo_geometry_id) if photo_geometry_id else None
                if baseline is not None:
                    from datasets.scripts.fire.hybrid_render import reset_scene
                    reset_scene(scene, baseline)
                state_index = target_state.get('scenario_index')
                plan = (explicit_scenario(target_state) if target_state.get('scenario') == 'explicit'
                        else scenario(state_index))
                active = plan['active']
                counts = dict(zip(active, plan['counts']))
                names = {}
                targets = []
                for key, group in groups.items():
                    group['collection'].hide_render = key not in active
                    if key not in active:
                        continue
                    names[key] = '开心大厦' if args.vehicle_pilot else make_name(rng)
                    reset_name(group['text'], names[key])
                    if args.vehicle_pilot:
                        group['text'].scale.x *= .80 if args.vehicle_oblique_pilot else .65
                    placement = plan.get('placement', 'all')
                    if key == 'C' and plan.get('view', '').startswith('orange_peel_'):
                        placement = rng.choice(SUPPORTED_C_PLACEMENTS)
                    size_policy = (sticker_size_policy
                                   if plan.get('view', '').startswith('orange_peel_') else None)
                    place_stickers(group, counts[key], rng, placement=placement,
                                   size_policy=size_policy, building=key,
                                   wall_offset=wall_projection.get('blender_offset_m') if size_policy else None)
                    targets.append({'id': len(targets) + 1, 'kind': 'name_sign', 'building': key,
                                    'text': names[key], 'object': group['sign']})
                    for sticker in group['pool'][:counts[key]]:
                        targets.append({'id': len(targets) + 1, 'kind': 'fire_sticker',
                                        'building': key, 'object': sticker})
                placement_result = None
                background_record = None
                if photo_review:
                    from datasets.scripts.fire.hybrid_render import prepare_photo
                    camera = prepare_photo(scene, groups, active, photo_review, rng, baseline)
                    placement_result = {'status': 'estimated_photo_fit', 'geometry_id': photo_geometry_id,
                                        'ground_anchor_px': camera['ground_anchor_px']}
                elif backgrounds:
                    background_record, exposure, saturation = sample_real_scene(backgrounds, rng)
                    camera, placement_result = place_buildings_on_reviewed_ground(
                        scene, groups, active, plan['kind'], background_record['placement'], rng,
                        view=plan.get('view'))
                    if camera is None:
                        records.write(json.dumps({'status': 'rejected_placement', 'state_index': state_index,
                            'scenario': plan['kind'], 'active_buildings': list(active), 'sample_seed': sample_seed,
                            'background': background_record, 'placement': placement_result}, ensure_ascii=False) + '\n')
                        records.flush()
                        metadata['attempted'] = index + 1
                        metadata['rejected_placement'] += 1
                        write_run(args.output / 'run.json', metadata)
                        continue
                else:
                    camera = (vehicle_camera(scene, groups['A'], args.vehicle_oblique_pilot) if args.vehicle_pilot
                              else set_camera(scene, groups, active, plan['kind'], rng, view=plan.get('view')))
                lights = ({name: float(bpy.data.objects[name].data.energy) for name in ('Broad softbox', 'Soft fill')}
                          if args.vehicle_pilot or photo_review else set_lighting(rng))
                background_crop, tone_record = None, None
                if backgrounds:
                    background_crop = set_vehicle_background(scene, ROOT / background_record['image'])
                    scene.view_settings.exposure = exposure
                    tone_compositor.inputs['Saturation'].default_value = saturation
                    tone_record = {'exposure_ev': exposure, 'saturation': saturation, 'sampling': 'seeded_uniform',
                                   'status': 'simulated_not_calibrated'}
                bpy.context.view_layer.update()
                occluder = set_blocker(scene, groups, active, plan['kind'])
                bpy.context.view_layer.update()
                for target in targets:
                    target['projection'] = projected_extent(scene, target['object'])
                stem = f'fire_{index + 1:06d}'
                scene.cycles.samples = args.samples
                scene.cycles.use_denoising = True
                scene.cycles.seed = args.seed + index
                image_relative = f'render_layers/{stem}.png' if photo_review else f'images/{partition}/{stem}.png'
                if photo_review:
                    (args.output/'render_layers').mkdir(exist_ok=True)
                scene.render.filepath = str(args.output / image_relative)
                bpy.ops.render.render(write_still=True)
                if args.save_preview_scene and (index == 1 or args.count == 1):
                    preview_scene = ROOT / 'Archive/technical/datasets/fire' / args.output.name / 'preview_scene.blend'
                    preview_scene.parent.mkdir(parents=True, exist_ok=True)
                    bpy.ops.wm.save_as_mainfile(filepath=str(preview_scene), copy=True)
                    metadata['editable_preview_scene'] = preview_scene.relative_to(ROOT).as_posix()
                id_path = args.output / 'instance_pass' / (stem + '.png')
                ids = render_id_pass(scene, targets, id_path, mask_material)
                obj_records, fire_rows, name_rows = [], [], []
                unknown_fires = False
                unknown_names = False
                for target in targets:
                    bbox, area = visible_bbox(ids == target['id'])
                    state = label_state(target['projection'], bbox, area)
                    data = {'id': target['id'], 'id_scope': 'sample', 'kind': target['kind'], 'building': target['building'],
                            'pixel_bbox_xyxy': bbox, 'visible_pixels': area, 'state': state,
                            'projected_extent_normalized': target['projection']['xyxy'],
                            'truncated': target['projection']['truncated'], 'occlusion': 'not quantified'}
                    data['world_translation_m'] = [float(value) for value in target['object'].matrix_world.translation]
                    data['world_rotation_euler_rad'] = [float(value) for value in target['object'].matrix_world.to_euler()]
                    data['dimensions_m'] = [float(value) for value in target['object'].dimensions]
                    if target['kind'] == 'name_sign':
                        data['text'] = target['text']
                        data['text_role'] = 'generated_full_text_ground_truth_not_ocr_label'
                        data['ocr_status'] = ocr_status(state, occluder)
                        if state in ('visible', 'visible_truncated'):
                            name_rows.append(yolo_row(bbox, args.resolution, args.height))
                        elif state != 'outside_frame':
                            unknown_names = True
                    elif state in ('visible', 'visible_truncated'):
                        fire_rows.append(yolo_row(bbox, args.resolution, args.height))
                    elif state != 'outside_frame':
                        unknown_fires = True
                    if target['kind'] == 'fire_sticker':
                        data['paper_ripple_m'] = float(target['object'].get('paper_ripple_m', 0))
                        data['paper_rotation_rad'] = float(target['object'].rotation_euler.y)
                        data['scale_from_reference'] = float(target['object'].get('scale_from_reference', 1))
                        if 'height_fraction' in target['object']:
                            data['height_fraction'] = float(target['object']['height_fraction'])
                            data['reference_facade_height_m'] = float(target['object']['reference_facade_height_m'])
                            if photo_review:
                                data['reference_facade_height_m'] *= camera['model_scale']
                                data['scale_from_reference'] *= camera['model_scale']
                            vertices = target['object'].data.vertices
                            data['paper_height_m'] = float((max(v.co.z for v in vertices) -
                                                            min(v.co.z for v in vertices)) * target['object'].scale.z)
                        if 'wall_plane_y_m' in target['object']:
                            obj = target['object']
                            plane, offset = float(obj['wall_plane_y_m']), float(obj['wall_offset_m'])
                            if photo_review:
                                transform = np.asarray(camera['world_transform'])
                                point = (transform @ [0,plane,0,1])[:3]
                                normal = transform[:3,:3] @ [0,1,0]
                                normal /= np.linalg.norm(normal)
                                offset *= camera['model_scale']
                                error = max(abs(np.dot(np.asarray(obj.matrix_world @ v.co)-point,normal)+offset)
                                            for v in obj.data.vertices)
                            else:
                                error = max(abs((obj.matrix_world @ vertex.co).y - (plane-offset))
                                            for vertex in obj.data.vertices)
                            if error > 1e-6:
                                raise ValueError('Fire paper vertices left the assigned facade plane')
                            data['wall_contact'] = {'plane_y_m': plane, 'offset_m': offset,
                                                    'max_plane_error_m': float(error)}
                            if photo_review:
                                data['wall_contact'].update(plane_point_m=point.tolist(),plane_normal_xyz=normal.tolist())
                    obj_records.append(data)
                name_reference_missing = (
                    not any(o['kind'] == 'name_sign' and o['visible_pixels'] >= 12 for o in obj_records) and
                    any(t['kind'] == 'name_sign' and t['projection']['intersects'] for t in targets))
                if name_reference_missing:
                    # This may be real occlusion or a failed ID pass.  Keep the
                    # image as unknown instead of stopping a whole batch or
                    # silently exporting a potentially false negative.
                    unknown_names = True
                    unknown_fires = True
                fire_label = None
                fire_suppression = fire_label_suppression(
                    [o['state'] for o in obj_records if o['kind'] == 'fire_sticker'], occluder)
                if backgrounds and fire_suppression is None:
                    fire_suppression = fire_scale_suppression(obj_records)
                if name_reference_missing:
                    fire_suppression = 'instance_reference_missing_or_hidden'
                if not unknown_fires and fire_suppression is None:
                    fire_label = (f'labels/{partition}/{stem}.txt' if args.current_geometry
                                  else f'labels/fire/{partition}/{stem}.txt')
                    (args.output/fire_label).parent.mkdir(parents=True, exist_ok=True)
                    (args.output / fire_label).write_text('\n'.join(fire_rows) + ('\n' if fire_rows else ''), encoding='utf-8')
                sample = {'image': image_relative, 'instance_pass': f'instance_pass/{stem}.png',
                          'fire_label': fire_label,
                          'fire_label_suppression_reason': fire_suppression,
                          'status': 'candidate_requires_visual_review', 'source_group': 'blender_v13_shared_model',
                          'split': partition, 'scenario': plan['kind'], 'active_buildings': list(active),
                          'requested_fire_counts': counts, 'names': names, 'camera': camera,
                          'lighting': lights, 'occluder': occluder, 'instances': obj_records,
                          'instance_reference_missing_or_hidden': name_reference_missing}
                if target_state.get('scenario') == 'explicit':
                    sample['explicit_state'] = target_state
                if target_state.get('placement') in SUPPORTED_C_PLACEMENTS:
                    sample['wall_support'] = {'support_object': support_geometry['support_object'],
                                              'support': support_geometry['support']}
                sample['production_state_id'] = planned_state['id']
                sample['production_state_sample_index'] = state_sample_index
                sample['sample_seed'] = sample_seed
                if photo_review:
                    sample.update(generation_method='blender_real_background_composite',
                                  photo_geometry_id=photo_geometry_id, photo_review=photo_review,
                                  placement=placement_result, source_assets=[source.relative_to(ROOT).as_posix(), photo_review['source_image']])
                if backgrounds:
                    sample.update(background=background_record, background_crop=background_crop,
                                  tone=tone_record, state_index=state_index, placement=placement_result,
                                  generation_method='real_scene_composite_blender',
                                  source_assets=[source.relative_to(ROOT).as_posix(), background_record['image']])
                records.write(json.dumps(sample, ensure_ascii=False) + '\n')
                records.flush()
                run.record_sample(
                    planned_state['id'], output=args.output / sample['image'],
                    sources=planned_state['sources'], parameters=planned_state['parameters'],
                    actual={
                        'actual_sources': [source.relative_to(ROOT).as_posix(),
                                           'Archive/Assets/working_sources/digital/fire/team_assets/image1.png',
                                           *(['Archive/Assets/captures/car_real_capture_20261112/P12_TL2.png']
                                             if args.vehicle_pilot else []),
                                           *([background_record['image']] if background_record else [])],
                        'scene_index': state_index, 'sample_index': index,
                        'python_seed': sample_seed, 'seed_base': args.seed, 'cycles_seed': sample_seed,
                        'camera': camera, 'lighting': lights, 'occluder': occluder,
                        'active_buildings': list(active), 'requested_fire_counts': counts,
                        'background': background_record, 'background_crop': background_crop,
                        'tone': tone_record,
                        'instances': obj_records,
                        'traffic_signal_added': False,
                    })
                metadata['completed'] += 1
                if backgrounds:
                    metadata['attempted'] = index + 1
                write_run(args.output / 'run.json', metadata)
        metadata['status'] = ('complete_candidates_with_rejections' if metadata.get('rejected_placement')
                              else 'complete_candidates')
    except Exception as exc:
        metadata['status'] = 'failed'
        metadata['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        write_run(args.output / 'run.json', metadata)
    print('FIRE_DATASET_CANDIDATES', args.output, metadata['completed'])


def main():
    args, source = parse_args()
    from datasets.scripts.common.blender_workspace import BUILDINGS, activate_scene
    activate_scene(load_task_config('fire')[1].get('blender_scene_name', BUILDINGS))
    args.photo_geometries = {}
    if args.current_geometry:
        geometry_path = (ROOT/args.current_geometry).resolve()
        if not geometry_path.is_relative_to(ROOT):
            raise ValueError('Photo geometry must stay in the workspace')
        geometry = json.loads(geometry_path.read_text(encoding='utf8'))
        args.photo_geometries = {r['id']: r for r in geometry['backgrounds']}
    background_manifest, backgrounds = (read_vehicle_backgrounds(args.vehicle_background_manifest)
                                        if args.vehicle_background_manifest else (None, None))
    plan_path, declaration = read_production_declaration(args.production_plan, args, backgrounds)
    supported = any(state['target'].get('placement') in SUPPORTED_C_PLACEMENTS
                    for state in declaration['strata'])
    if supported:
        expected_scene = source_path('fire', 'scene')
        if source != expected_scene:
            raise ValueError('supported C wall states require the verified building scene')
        verify_c_supported_wall(get_source())
    planned = plan_production(declaration, ROOT)
    if planned['planned_total'] != args.count:
        raise ValueError('planned total differs from --count')
    if args.vehicle_pilot:
        for state in planned['strata']:
            counts = state['target']['fire_counts']
            if sum(counts.values()) == 0 and state['planned_count'] > 1:
                raise ValueError('fixed empty vehicle-pilot state cannot produce repeated distinct samples')
    media, procedural = background_guard_inputs(background_manifest, backgrounds)
    if args.current_geometry:
        procedural.add(geometry_path)
        media.update(ROOT/r['source_image'] for r in args.photo_geometries.values())
    media.update(scene_external_media())
    # The building library was constructed from this workspace fire artwork.
    media.add(configured_fire_artwork())
    if args.vehicle_pilot:
        media.add(ROOT / 'Archive/Assets/captures/car_real_capture_20261112/P12_TL2.png')
    procedural.update(scene_external_libraries())
    procedural.add(source)
    procedural.add(plan_path)
    procedural.update(ROOT/'datasets/scripts/fire/config'/name for name in ('dataset.yaml', 'source.yaml'))
    parameters = {key: value for key, value in vars(args).items() if key != 'output'}
    parameters['source_scene'] = source.relative_to(ROOT).as_posix()
    outputs = [args.output]
    if args.save_preview_scene:
        outputs.append(ROOT / 'Archive/technical/datasets/fire' / args.output.name / 'preview_scene.blend')
    with generation_guard(
        'fire.render_blender_fire_dataset:' + args.output.relative_to(ROOT).as_posix(),
        sorted(media | procedural), outputs, parameters,
        code_paths=[Path(__file__), SCRIPTS / 'fire_blender_labels.py', ROOT / 'datasets/scripts/common/annotations.py',
                    ROOT / 'lab/training_common/bayesian_allocation.py', ROOT / 'datasets/scripts/common/bayesian.py',
                    ROOT / 'datasets/scripts/common/config.py', ROOT / 'datasets/scripts/common/paths.py',
                    ROOT / 'datasets/scripts/common/blender_workspace.py',
                    SCRIPTS / 'camera_trajectory.py', SCRIPTS / 'hybrid_render.py', SCRIPTS / 'photo_geometry.py'],
        procedural_inputs=sorted(procedural),
        production_plan=declaration,
        sample_output_paths=[args.output / 'images' / output_partition(args, backgrounds),
                             *([args.output / 'render_layers'] if args.current_geometry else [])],
        auxiliary_output_paths=[args.output / 'instance_pass'],
    ) as run:
        if run.plan != planned:
            raise ValueError('guard plan differs from the pre-generation feasibility plan')
        if not run.reused:
            _produce(args, source, background_manifest, backgrounds, run)


if __name__ == '__main__':
    main()
