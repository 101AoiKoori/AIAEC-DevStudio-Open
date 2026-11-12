"""Render grouped bin-lid candidates, with visible/isolated pixel evidence.

Run Blender from the workspace root, loading the existing .blend first::

  blender -b MODEL.blend --python-exit-code 1 --python \
    lab/features/waste_bin/training/scripts/render_blender_waste_bin.py -- \
    --out datasets/exports/waste_bin/NEW --coverage-plan --device CPU

The ten-frame coverage plan includes balanced mixed states, a foreground
occluder and clipping. All outputs are candidates; no source split is made.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Quaternion, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bin_render_contract import (BALANCED_ORDER, BIN_NAMES, MASK_COLORS, OPEN_POSES, STATE_CONTRACT,
                                 attach_state_evidence, capture_material_baseline, combinations,
                                 decode_mask, measure_visibility, projected_points_fit,
                                 required_hinge_shift, vary_material)


ROOT = Path.cwd().resolve()
GEOMETRY_TYPES = {'MESH', 'CURVE', 'SURFACE', 'FONT', 'META'}


def within_workspace(path):
    result = (ROOT / path).resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError('paths must be inside the workspace')
    return result


def configure_device(scene, device_type):
    if device_type == 'CPU':
        scene.cycles.device = 'CPU'
        return []
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = device_type
    prefs.get_devices()
    selected = []
    for device in prefs.devices:
        device.use = device.type == device_type
        if device.use:
            selected.append(device.name)
    if not selected:
        raise RuntimeError('no ' + device_type + ' render device available; use --device CPU')
    scene.cycles.device = 'GPU'
    return selected


def build_mask_material():
    material = bpy.data.materials.new('bin lid visibility measurement')
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    info = nodes.new('ShaderNodeObjectInfo')
    emission = nodes.new('ShaderNodeEmission')
    output = nodes.new('ShaderNodeOutputMaterial')
    material.node_tree.links.new(info.outputs['Color'], emission.inputs['Color'])
    material.node_tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])
    return material


def read_rgba(path):
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        image.colorspace_settings.name = 'Non-Color'
        width, height = image.size
        if width <= 0 or height <= 0 or image.channels != 4:
            raise ValueError('mask PNG must contain four channels')
        pixels = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        # Blender image pixels start at bottom-left, labels use image top-left.
        return pixels.reshape((height, width, 4))[::-1].copy()
    finally:
        bpy.data.images.remove(image)


def targets_inside_frame(scene, camera, targets, margin=0.0):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points = []
    for obj in targets:
        evaluated = obj.evaluated_get(depsgraph)
        for corner in evaluated.bound_box:
            point = world_to_camera_view(scene, camera, evaluated.matrix_world @ Vector(corner))
            points.append((point.x, point.y, point.z))
    return projected_points_fit(points, margin)


def correct_hanging_hinge_clearance(pivots):
    """Align each rear hinge outside the shell while preserving the closed lid."""
    shifts = {}
    for name, pivot in zip(BIN_NAMES, pivots):
        body = bpy.data.objects[name + ' body']
        panel = bpy.data.objects[name + ' raised panel']
        axle = bpy.data.objects[name + ' hinge axle']
        supports = [o for o in bpy.context.scene.objects if o.name.startswith(name + ' hinge support')]
        straps = [o for o in pivot.children if o.name.startswith(name + ' hinge strap')]
        if not supports or not straps or abs(pivot.parent.matrix_world.to_scale().y - 1) > 1e-6:
            raise ValueError(name + ': unsupported hinge geometry')
        original_angle = pivot.rotation_euler.x
        pivot.rotation_euler.x = math.radians(OPEN_POSES['hanging_back'])
        bpy.context.view_layer.update()
        body_back = max((body.matrix_world @ Vector(v)).y for v in body.bound_box)
        panel_front = min((panel.matrix_world @ Vector(v)).y for v in panel.bound_box)
        shift = required_hinge_shift(body_back, panel_front)
        pivot.rotation_euler.x = 0
        bpy.context.view_layer.update()
        closed_matrices = {o.name: o.matrix_world.copy() for o in pivot.children}
        pivot.location.y += shift
        bpy.context.view_layer.update()
        for obj in pivot.children:
            obj.matrix_world = closed_matrices[obj.name]
        bpy.context.view_layer.update()
        for obj in straps:
            obj.location.y += shift
        axle.location.y += shift
        for obj in supports:
            old_depth = obj.dimensions.y
            if old_depth <= 0:
                raise ValueError(name + ': invalid hinge support depth')
            obj.scale.y *= (old_depth + shift) / old_depth
            obj.location.y += shift / 2
        pivot.rotation_euler.x = math.radians(OPEN_POSES['hanging_back'])
        bpy.context.view_layer.update()
        corrected_front = min((panel.matrix_world @ Vector(v)).y for v in panel.bound_box)
        if corrected_front < body_back + .014:
            raise ValueError(name + ': hanging panel still intersects the bin envelope')
        shifts[name] = {'rearward_shift': shift, 'body_back_y': body_back,
                        'original_panel_front_y': panel_front, 'corrected_panel_front_y': corrected_front}
        pivot.rotation_euler.x = original_angle
    bpy.context.view_layer.update()
    return shifts


def fit_lids_in_camera(scene, camera, pivots, target_parts, margin=.025):
    """Fit whole bins across closed, hanging and upright lid poses."""
    rotations = [p.rotation_euler.x for p in pivots]
    try:
        for _ in range(20):
            fits = True
            for angle in (0, *OPEN_POSES.values()):
                for pivot in pivots:
                    pivot.rotation_euler.x = math.radians(angle)
                bpy.context.view_layer.update()
                if not targets_inside_frame(scene, camera,
                                            [obj for parts in target_parts.values() for obj in parts], margin):
                    fits = False
                    break
            if fits:
                return camera.data.lens
            camera.data.lens *= .90
            if camera.data.lens < 18:
                break
        raise RuntimeError('cannot fit both lid states in the camera without a lens below 18 mm')
    finally:
        for pivot, angle in zip(pivots, rotations):
            pivot.rotation_euler.x = angle
        bpy.context.view_layer.update()


def render_target_masks(scene, camera, target_parts, out, stem, min_visibility, min_pixels,
                        visible_dir, isolated_dir, force_visible=()):
    """Measure the visible and isolated pixels of four per-bin target sets."""
    layer = bpy.context.view_layer
    objects = [o for o in scene.objects if o.type in GEOMETRY_TYPES]
    original_hidden = {o.name: bool(o.hide_render) for o in objects}
    original_colors = {o.name: tuple(o.color) for o in objects}
    original = {
        'override': layer.material_override, 'samples': scene.cycles.samples,
        'denoise': scene.cycles.use_denoising, 'bounces': scene.cycles.max_bounces,
        'transparent': scene.render.film_transparent, 'file_format': scene.render.image_settings.file_format,
        'color_mode': scene.render.image_settings.color_mode, 'color_depth': scene.render.image_settings.color_depth,
        'transform': scene.view_settings.view_transform, 'look': scene.view_settings.look,
        'exposure': scene.view_settings.exposure, 'gamma': scene.view_settings.gamma,
        'filepath': scene.render.filepath,
    }
    mask_mat = build_mask_material()
    paths = {}
    try:
        layer.material_override = mask_mat
        for obj in force_visible:
            obj.hide_render = False
        scene.cycles.samples = 1
        scene.cycles.use_denoising = False
        scene.cycles.max_bounces = 0
        scene.render.film_transparent = True
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGBA'
        scene.render.image_settings.color_depth = '8'
        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.look = 'None'
        scene.view_settings.exposure = 0
        scene.view_settings.gamma = 1
        for obj in objects:
            obj.color = (0, 0, 0, 1)
        for name, parts in target_parts.items():
            for obj in parts:
                obj.color = (*MASK_COLORS[name], 1)
        bpy.context.view_layer.update()
        visible_path = out / visible_dir / (stem + '.png')
        scene.render.filepath = str(visible_path)
        bpy.ops.render.render(write_still=True)
        visible_rgba = read_rgba(visible_path)
        expected = (scene.render.resolution_y, scene.render.resolution_x)
        if visible_rgba.shape[:2] != expected:
            raise ValueError('visible mask dimensions differ from the render')
        paths['visible'] = visible_path.relative_to(out).as_posix()
        results = []
        for name in BIN_NAMES:
            selected = {o.name for o in target_parts[name]}
            for obj in objects:
                obj.hide_render = (False if obj in force_visible else original_hidden[obj.name]) if obj.name in selected else True
            bpy.context.view_layer.update()
            solo_path = out / isolated_dir / (stem + '_' + name + '.png')
            scene.render.filepath = str(solo_path)
            bpy.ops.render.render(write_still=True)
            solo_rgba = read_rgba(solo_path)
            if solo_rgba.shape != visible_rgba.shape:
                raise ValueError('isolated mask dimensions differ from visible mask')
            result = measure_visibility(decode_mask(visible_rgba, name),
                                        decode_mask(solo_rgba, name),
                                        targets_inside_frame(scene, camera, target_parts[name]),
                                        min_visibility, min_pixels)
            result['bin'] = name
            result['isolated_mask'] = solo_path.relative_to(out).as_posix()
            results.append(result)
        return paths, results
    finally:
        for obj in objects:
            obj.hide_render = original_hidden[obj.name]
            obj.color = original_colors[obj.name]
        layer.material_override = original['override']
        scene.cycles.samples = original['samples']
        scene.cycles.use_denoising = original['denoise']
        scene.cycles.max_bounces = original['bounces']
        scene.render.film_transparent = original['transparent']
        scene.render.image_settings.file_format = original['file_format']
        scene.render.image_settings.color_mode = original['color_mode']
        scene.render.image_settings.color_depth = original['color_depth']
        scene.view_settings.view_transform = original['transform']
        scene.view_settings.look = original['look']
        scene.view_settings.exposure = original['exposure']
        scene.view_settings.gamma = original['gamma']
        scene.render.filepath = original['filepath']
        bpy.context.view_layer.update()
        bpy.data.materials.remove(mask_mat)


def create_mouth_proxies(scene):
    """Mask-only inset planes derived from the existing flange's inner polygon."""
    proxies = {}
    for name in BIN_NAMES:
        flange = bpy.data.objects[name + ' flat rounded flange']
        vertices = flange.data.vertices
        if len(vertices) < 6 or len(vertices) % 2:
            raise ValueError(name + ': unsupported flange geometry')
        inner = [v.co.copy() for v in vertices[len(vertices) // 2:]]
        center = sum(inner, Vector()) / len(inner)
        # Slightly inset to avoid counting the rim as visible opening.
        inner = [center + (v - center) * .90 for v in inner]
        mesh = bpy.data.meshes.new(name + ' opening visibility proxy mesh')
        mesh.from_pydata(inner, [], [tuple(range(len(inner)))])
        mesh.update()
        obj = bpy.data.objects.new(name + ' opening visibility proxy', mesh)
        scene.collection.objects.link(obj)
        obj.matrix_world = flange.matrix_world.copy()
        obj.hide_render = True
        proxies[name] = obj
    bpy.context.view_layer.update()
    return proxies


def create_coverage_occluder(scene):
    bpy.ops.mesh.primitive_cube_add(size=1)
    obj = bpy.context.object
    obj.name = 'coverage foreground occluder'
    obj.dimensions = (.27, .16, .34)
    material = bpy.data.materials.new('coverage neutral obstacle')
    material.diffuse_color = (.32, .32, .32, 1)
    material.use_nodes = True
    material.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value = (.32, .32, .32, 1)
    obj.data.materials.append(material)
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--scenes', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--samples', type=int, default=12)
    parser.add_argument('--device', choices=('CPU', 'OPTIX', 'CUDA'), default='CPU')
    parser.add_argument('--combinations', type=int, default=16)
    parser.add_argument('--limit-images', type=int)
    parser.add_argument('--coverage-plan', action='store_true',
                        help='exactly ten balanced frames: normal, occluded, clipped')
    parser.add_argument('--min-visibility', type=float, default=.70)
    parser.add_argument('--min-visible-pixels', type=int, default=24)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    if args.coverage_plan:
        if args.limit_images is not None:
            parser.error('coverage-plan defines exactly ten frames and forbids limit-images')
        args.scenes, args.combinations = 5, 2
    if args.scenes < 1 or args.samples < 1 or not 1 <= args.combinations <= 16:
        parser.error('scenes and samples must be positive; combinations must be 1..16')
    if not 0 < args.min_visibility <= 1 or args.min_visible_pixels < 1:
        parser.error('invalid visibility thresholds')
    if args.limit_images is not None and not 1 <= args.limit_images <= args.scenes * args.combinations:
        parser.error('limit-images must be within the requested capacity')
    if args.limit_images is not None and args.combinations != 16:
        parser.error('limit-images requires all 16 combinations per complete group')
    out = within_workspace(args.out)
    source = within_workspace(bpy.data.filepath)
    allowed_roots = (ROOT / 'datasets' / 'exports' / 'waste_bin', ROOT / 'tmp')
    if not any(out.is_relative_to(p.resolve()) for p in allowed_roots):
        parser.error('output must be under datasets/exports/waste_bin or tmp')
    if not bpy.data.filepath or not source.is_file():
        parser.error('open an existing workspace Blender scene')
    scene = bpy.context.scene
    if scene.render.engine != 'CYCLES' or not scene.camera:
        parser.error('source scene must contain a camera and use Cycles')
    baselines = capture_material_baseline(bpy.data.materials)
    camera = scene.camera
    pivots = [bpy.data.objects[name + ' lid pivot - rotate X'] for name in BIN_NAMES]
    hinge_clearance = correct_hanging_hinge_clearance(pivots)
    lid_parts = {name: [obj for obj in scene.objects if obj.parent == pivot and obj.type in GEOMETRY_TYPES]
                 for name, pivot in zip(BIN_NAMES, pivots)}
    if any(not parts for parts in lid_parts.values()):
        raise ValueError('each lid must have visible geometric components')
    target_parts = {name: [obj for obj in scene.objects if obj.type in GEOMETRY_TYPES
                           and obj.name.startswith(name + ' ')] for name in BIN_NAMES}
    for name in BIN_NAMES:
        if not bpy.data.objects.get(name + ' body') or not bpy.data.objects.get(name + ' flat rounded flange'):
            raise ValueError(name + ': missing bin body or flange')
    proxies = create_mouth_proxies(scene)
    out.mkdir(parents=True, exist_ok=False)
    for folder in ('images', 'labels_candidate', 'mask_visible', 'mask_isolated',
                   'mask_state_visible', 'mask_state_isolated'):
        (out / folder).mkdir()
    requested = 10 if args.coverage_plan else (args.limit_images or args.scenes * args.combinations)
    report = {'status': 'running', 'seed': args.seed, 'requested_groups': args.scenes,
              'requested_images': requested, 'render_device': args.device, 'devices': [],
              'samples_per_pixel': args.samples, 'blender_version': bpy.app.version_string,
              'scene_file': source.relative_to(ROOT).as_posix(), 'scene_unit_cm': 10,
              'camera_distance_note': 'unmeasured scene distance; not a calibrated vehicle route',
              'decal_attribution': scene.get('decal_attribution', ''),
              'purpose': 'synthetic lid candidates; mask visibility is an estimate and requires review',
              'classes': {'0': 'open', '1': 'closed'},
              'split_policy': 'unsplit candidates; split by complete trajectory before training',
              'coverage_plan': bool(args.coverage_plan),
              'label_contract_id': STATE_CONTRACT, 'open_pose_degrees': OPEN_POSES,
              'hinge_clearance_adjustment': hinge_clearance,
              'label_contract': 'visible whole-bin bbox; open requires visible mouth proxy and closed requires visible lid; exclude whole image if any evidence fails',
              'min_visibility': args.min_visibility, 'min_visible_pixels': args.min_visible_pixels,
              'material_baselines_linear': baselines, 'samples': []}
    def save_report():
        (out / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    save_report()
    try:
        scene.render.resolution_x = scene.render.resolution_y = 1024
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = 'PNG'
        scene.cycles.samples = args.samples
        report['devices'] = configure_device(scene, args.device)
        camera.animation_data_clear()
        obstacle = create_coverage_occluder(scene) if args.coverage_plan else None
        rng = random.Random(args.seed)
        trajectory = None
        for group in range(args.scenes):
            if group % 8 == 0:
                trajectory = {'distance': rng.uniform(5.8, 6.2), 'height': rng.uniform(2.8, 3.2),
                              'lens_mm': rng.uniform(30, 38), 'yaw_phase': rng.uniform(0, 2 * math.pi)}
            t = ((group % 8) + .5) / 8
            x = -1.25 + 2.5 * t
            scenario = 'normal'
            if args.coverage_plan and group == 3:
                scenario = 'foreground_occlusion'
            elif args.coverage_plan and group == 4:
                scenario, x = 'clipping', 2.0
            distance, height = trajectory['distance'], trajectory['height']
            yaw = 2 * math.sin(trajectory['yaw_phase'] + 2 * math.pi * t)
            camera.location = (x, -distance, height)
            target = Vector((x if scenario == 'clipping' else 0, 0, 1.0))
            camera.rotation_euler = (target - camera.location).to_track_quat('-Z', 'Y').to_euler()
            camera.rotation_euler = (Quaternion((0, 0, 1), math.radians(yaw)) @
                                     camera.rotation_euler.to_quaternion()).to_euler()
            camera.data.lens = trajectory['lens_mm']
            bpy.context.view_layer.update()
            if scenario != 'clipping':
                fit_lids_in_camera(scene, camera, pivots, target_parts)
            if obstacle is not None:
                obstacle.hide_render = scenario != 'foreground_occlusion'
                # Position one compact obstacle on the camera-to-blue-lid line.
                target_position = Vector((-1.68, 0, 1.15))
                obstacle.location = camera.location.lerp(target_position, (distance - 1.0) / distance)
            key_energy, fill_energy = rng.uniform(700, 1100), rng.uniform(350, 650)
            bpy.data.objects['Key'].data.energy = key_energy
            bpy.data.objects['Fill'].data.energy = fill_energy
            background = {}
            for name in ('Floor', 'Backdrop'):
                shade = rng.uniform(.09, .23) if name == 'Floor' else rng.uniform(.24, .48)
                node = bpy.data.materials[name].node_tree.nodes.get('Principled BSDF')
                node.inputs['Base Color'].default_value = (shade, shade, shade, 1)
                background[name] = shade
            colors, roughness = {}, {}
            for name in BIN_NAMES:
                node = bpy.data.materials[name + ' adjustable body color'].node_tree.nodes.get('Principled BSDF')
                rgb, rough = vary_material(baselines[name], rng.uniform(.90, 1.10), rng.uniform(-.03, .03))
                node.inputs['Base Color'].default_value = (*rgb, 1)
                node.inputs['Roughness'].default_value = rough
                colors[name], roughness[name] = rgb, rough
            bpy.context.view_layer.update()
            if args.coverage_plan:
                states_to_render = BALANCED_ORDER[group * 2:group * 2 + 2]
            else:
                count = args.combinations
                if args.limit_images is not None:
                    count = min(count, args.limit_images - len(report['samples']))
                if count <= 0:
                    break
                states_to_render = combinations(count)
            pose_name = ('hanging_back' if group % 2 == 0 else 'upright')
            open_angle = OPEN_POSES[pose_name]
            for combination in states_to_render:
                states = []
                for i, pivot in enumerate(pivots):
                    state = 'open' if (combination >> i) & 1 else 'closed'
                    pivot.rotation_euler.x = math.radians(open_angle if state == 'open' else 0)
                    states.append(state)
                bpy.context.view_layer.update()
                stem = f'g{group:04d}_s{combination:02d}'
                filename = stem + '.png'
                scene.render.filepath = str(out / 'images' / filename)
                rgb_start = time.perf_counter()
                bpy.ops.render.render(write_still=True)
                rgb_seconds = time.perf_counter() - rgb_start
                mask_start = time.perf_counter()
                mask_paths, targets = render_target_masks(scene, camera, target_parts, out, stem,
                    args.min_visibility, args.min_visible_pixels, 'mask_visible', 'mask_isolated')
                state_parts = {name: ([proxies[name]] if states[i] == 'open' else lid_parts[name])
                               for i, name in enumerate(BIN_NAMES)}
                open_proxies = [proxies[name] for i, name in enumerate(BIN_NAMES) if states[i] == 'open']
                state_paths, evidence = render_target_masks(scene, camera, state_parts, out, stem,
                    args.min_visibility, args.min_visible_pixels, 'mask_state_visible', 'mask_state_isolated',
                    force_visible=open_proxies)
                labels = [attach_state_evidence(targets[i], evidence[i], states[i]) for i in range(4)]
                mask_seconds = time.perf_counter() - mask_start
                candidate = None
                if all(item['label_eligible'] for item in labels):
                    candidate = 'labels_candidate/' + stem + '.txt'
                    rows = []
                    for item in labels:
                        x1, y1, x2, y2 = item['candidate_xyxy_normalized']
                        rows.append(f"{0 if item['state'] == 'open' else 1} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}")
                    (out / candidate).write_text('\n'.join(rows) + '\n', encoding='utf-8')
                report['samples'].append({'image': 'images/' + filename, 'candidate_label': candidate,
                    'group': group, 'trajectory': group // 8, 'states': states, 'scenario': scenario,
                    'open_pose': pose_name, 'open_angle_degrees': open_angle,
                    'camera': {'x': x, 'distance': distance, 'height': height,
                               'yaw_degrees': yaw, 'lens_mm': camera.data.lens},
                    'timing_seconds': {'rgb': rgb_seconds, 'ten_masks': mask_seconds},
                    'lighting': {'key_energy': key_energy, 'fill_energy': fill_energy},
                    'background_gray_linear': background, 'body_colors_linear': colors,
                    'body_roughness': roughness, 'mask_visible': mask_paths['visible'],
                    'mask_state_visible': state_paths['visible'], 'labels': labels})
                save_report()
        if len(report['samples']) != requested:
            raise RuntimeError('incomplete image count')
        if args.coverage_plan:
            normal = [s for s in report['samples'] if s['scenario'] == 'normal']
            occluded = [s for s in report['samples'] if s['scenario'] == 'foreground_occlusion']
            clipped = [s for s in report['samples'] if s['scenario'] == 'clipping']
            checks = {
                'normal_lids_eligible': all(s['candidate_label'] for s in normal),
                'occlusion_detected': all(any(
                    x['bin'] == 'blue' and any(
                        fraction is not None and fraction < .98
                        for fraction in (x['visible_fraction'], x['state_evidence']['visible_fraction'])
                    ) for x in s['labels']) for s in occluded),
                'clipping_detected': all(any(not x['in_frame'] for x in s['labels']) for s in clipped),
            }
            report['coverage_checks'] = checks
            if not all(checks.values()):
                raise RuntimeError('coverage plan did not produce intended conditions: ' + str(checks))
        report['status'] = 'complete_candidates'
    except BaseException as exc:
        report['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        report['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        save_report()


if __name__ == '__main__':
    main()
