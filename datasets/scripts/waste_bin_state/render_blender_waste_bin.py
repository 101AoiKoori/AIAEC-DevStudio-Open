"""Render planned bin-lid candidates with visible/isolated mask evidence."""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Quaternion, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets.scripts.waste_bin_state.bin_render_contract import (BIN_NAMES, MASK_COLORS, MIN_MOUTH_THICKNESS_PX,
                                 OPEN_POSES, STATE_CONTRACT,
                                 attach_state_evidence, capture_material_baseline,
                                 decode_mask, measure_visibility, mouth_thickness, projected_points_fit, standalone_planned_jobs,
                                 required_hinge_shift, vary_material)


ROOT = Path(__file__).resolve().parents[3]
from datasets.scripts.common.config import source_path
from datasets.scripts.common.paths import BIN_STATE_RENDER_FOLDERS, bin_state_foreground_dir
GEOMETRY_TYPES = {'MESH', 'CURVE', 'SURFACE', 'FONT', 'META'}


def ground_bins(scene):
    """Move each bin hierarchy to the horizontal ground top in world coordinates."""
    ground = bpy.data.objects.get('Ground')
    if not ground or ground.type != 'MESH':
        raise ValueError('a mesh Ground is required')
    bpy.context.view_layer.update()
    points = [ground.matrix_world @ v.co for v in ground.data.vertices]
    top = max(p.z for p in points)
    faces = [[points[i] for i in poly.vertices] for poly in ground.data.polygons]
    top_faces = [face for face in faces if len(face) == 4 and all(abs(p.z - top) < 1e-6 for p in face)]
    if len(top_faces) != 1:
        raise ValueError('Ground must have exactly one horizontal quadrilateral top face')
    face = top_faces[0]
    xs, ys = sorted(set(round(p.x, 6) for p in face)), sorted(set(round(p.y, 6) for p in face))
    if len(xs) != 2 or len(ys) != 2 or {(round(p.x, 6), round(p.y, 6)) for p in face} != {(x, y) for x in xs for y in ys}:
        raise ValueError('Ground top must be an axis-aligned rectangle')
    result = {}
    for name in BIN_NAMES:
        body = bpy.data.objects[name + ' body']
        body_corners = [body.matrix_world @ Vector(v) for v in body.bound_box]
        if any(not (xs[0] <= p.x <= xs[1] and ys[0] <= p.y <= ys[1]) for p in body_corners):
            raise ValueError(name + ': body footprint is outside the support surface')
        bottom = min(p.z for p in body_corners)
        shift = top - bottom
        if abs(shift) > .2:
            raise ValueError(name + ': ground correction exceeds bounded tolerance')
        roots = [o for o in scene.objects if o.name.startswith(name + ' ') and
                 (not o.parent or not o.parent.name.startswith(name + ' '))]
        for obj in roots:
            matrix = obj.matrix_world.copy()
            matrix.translation.z += shift
            obj.matrix_world = matrix
        bpy.context.view_layer.update()
        actual_bottom = min((body.matrix_world @ Vector(v)).z for v in body.bound_box)
        if abs(actual_bottom - top) > 1e-5:
            raise ValueError(name + ': could not establish ground contact')
        result[name] = {'floor_top_z': top, 'original_body_bottom_z': bottom,
                        'world_z_shift': shift, 'body_bottom_z': actual_bottom}
    return result


def within_workspace(path):
    result = (ROOT / path).resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError('paths must be inside the workspace')
    return result


from datasets.scripts.common.blender_masks import configure_device


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


from datasets.scripts.common.blender_masks import read_rgba


from datasets.scripts.common.blender_masks import targets_inside_frame


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
            if name not in target_parts:
                continue
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


def _generate(run, args, jobs, media):
    out = within_workspace(args.out)
    source = within_workspace(bpy.data.filepath)
    destination = bin_state_foreground_dir(run.plan['target_split'])
    if out != destination.resolve():
        raise ValueError(f'render output must be {destination.relative_to(ROOT).as_posix()}')
    if (out / 'manifest.json').exists() or any(
            (out / name).exists() and any((out / name).iterdir()) for name in BIN_STATE_RENDER_FOLDERS):
        raise ValueError('managed bin-state render outputs must be empty')
    if not source.is_file():
        raise ValueError('loaded scene is missing')
    scene = bpy.context.scene
    if scene.render.engine != 'CYCLES' or not scene.camera:
        raise ValueError('source scene must contain a camera and use Cycles')
    body_materials = {name + ' adjustable body color': scene.objects[name + ' body'].active_material
                      for name in BIN_NAMES}
    baselines = capture_material_baseline(body_materials)
    camera = scene.camera
    ground_contact = ground_bins(scene)
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
    out.mkdir(parents=True, exist_ok=True)
    for folder in BIN_STATE_RENDER_FOLDERS:
        (out / folder).mkdir(exist_ok=True)
    report = {'status': 'running', 'seed': args.seed, 'requested_images': len(jobs),
              'render_device': args.device, 'devices': [], 'samples_per_pixel': args.samples,
              'blender_version': bpy.app.version_string, 'resolution': [args.width, args.height],
              'scene_file': source.relative_to(ROOT).as_posix(), 'scene_name': scene.name, 'scene_unit_cm': 10,
              'camera_distance_note': 'declared scene coordinates; not a calibrated vehicle route',
              'classes': {'0': 'open', '1': 'closed'}, 'split': run.plan['target_split'],
              'label_contract_id': STATE_CONTRACT, 'hinge_clearance_adjustment': hinge_clearance,
              'ground_contact_adjustment': ground_contact,
              'min_visibility': args.min_visibility, 'min_visible_pixels': args.min_visible_pixels,
              'material_baselines_linear': baselines, 'samples': []}

    def save_report():
        (out / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    save_report()
    try:
        scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = 'PNG'
        scene.cycles.samples = args.samples
        report['devices'] = configure_device(scene, args.device)
        camera.animation_data_clear()
        for index, job in enumerate(jobs):
            params, background, target = job['parameters'], job['background'], job['target']
            selected = set(target['bins'])
            states = {name: target['bins'].get(name, 'closed') for name in BIN_NAMES}
            camera.location = params['camera_location']
            aim = Vector(params['camera_aim'])
            camera.rotation_euler = (aim - camera.location).to_track_quat('-Z', 'Y').to_euler()
            camera.rotation_euler = (Quaternion((0, 0, 1), math.radians(params['yaw_degrees'])) @
                                     camera.rotation_euler.to_quaternion()).to_euler()
            camera.data.lens = params['lens_mm']
            bpy.data.objects['Key'].data.energy = params['key_energy']
            bpy.data.objects['Fill'].data.energy = params['fill_energy']
            for name, key in (('Ground', 'floor_gray'), ('Background wall', 'backdrop_gray')):
                shade = background[key]
                node = scene.objects[name].active_material.node_tree.nodes.get('Principled BSDF')
                node.inputs['Base Color'].default_value = (shade, shade, shade, 1)
            colors, roughness = {}, {}
            for name in BIN_NAMES:
                node = body_materials[name + ' adjustable body color'].node_tree.nodes.get('Principled BSDF')
                rgb, rough = vary_material(baselines[name], params['color_factor'], params['roughness_delta'])
                node.inputs['Base Color'].default_value = (*rgb, 1)
                node.inputs['Roughness'].default_value = rough
                colors[name], roughness[name] = rgb, rough
                for obj in target_parts[name]:
                    obj.hide_render = name not in selected
            for i, pivot in enumerate(pivots):
                pivot.rotation_euler.x = math.radians(OPEN_POSES[target['open_pose']] if states[BIN_NAMES[i]] == 'open' else 0)
            scene.cycles.seed = args.seed + index
            bpy.context.view_layer.update()
            stem = f'planned_{index:06d}'
            filename = stem + '.png'
            scene.render.filepath = str(out / 'images' / filename)
            bpy.ops.render.render(write_still=True)
            if selected:
                parts = {name: target_parts[name] for name in BIN_NAMES if name in selected}
                mask_paths, targets = render_target_masks(scene, camera, parts, out, stem,
                    args.min_visibility, args.min_visible_pixels, 'mask_visible', 'mask_isolated')
                open_bins = {name for name in selected if states[name] == 'open'}
                evidence_parts = {name: ([proxies[name]] if name in open_bins else lid_parts[name])
                                  for name in BIN_NAMES if name in selected}
                state_paths, evidence = render_target_masks(scene, camera, evidence_parts, out, stem,
                    args.min_visibility, args.min_visible_pixels, 'mask_state_visible', 'mask_state_isolated',
                    force_visible=[proxies[name] for name in open_bins])
                labels = [attach_state_evidence(item, evidence[i], states[item['bin']])
                          for i, item in enumerate(targets)]
                mouth_pixels = read_rgba(out / state_paths['visible'])
                for item in labels:
                    if item['state_evidence']['kind'] == 'mouth_proxy':
                        thickness = mouth_thickness(decode_mask(mouth_pixels, item['bin']))
                        item['state_evidence']['max_vertical_mouth_pixels'] = thickness
                        if thickness < MIN_MOUTH_THICKNESS_PX:
                            item['label_eligible'] = False
                            item['rejection_reasons'].append('state_evidence:mouth_too_thin')
            else:
                mask_paths, state_paths, labels = {'visible': None}, {'visible': None}, []
                if any(not obj.hide_render for parts in target_parts.values() for obj in parts):
                    raise RuntimeError('a bin object remains visible for the empty state')
            candidate = None
            if all(item['label_eligible'] for item in labels):
                candidate = 'labels_candidate/' + stem + '.txt'
                rows = []
                for item in labels:
                    x1, y1, x2, y2 = item['candidate_xyxy_normalized']
                    rows.append(f"{0 if item['state'] == 'open' else 1} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}")
                (out / candidate).write_text('\n'.join(rows) + ('\n' if rows else ''), encoding='utf-8')
            actual = {'scene': source.relative_to(ROOT).as_posix(), 'scene_name': scene.name, 'state_id': job['id'],
                      'visible_bins': [name for name in BIN_NAMES if name in selected],
                      'bin_states': {name: states[name] for name in BIN_NAMES if name in selected},
                      'open_pose': target['open_pose'], 'camera_location': list(camera.location),
                      'camera_rotation_euler': list(camera.rotation_euler), 'camera_aim': list(aim),
                      'lens_mm': camera.data.lens, 'key_energy': params['key_energy'],
                      'fill_energy': params['fill_energy'], 'floor_gray': background['floor_gray'],
                      'backdrop_gray': background['backdrop_gray'], 'body_colors_linear': colors,
                      'body_roughness': roughness, 'render_seed': args.seed + index,
                      'ground_contact_adjustment': ground_contact,
                      'executed_augmentation': job['augmentation'],
                      'planned_color_factor': params['color_factor'], 'planned_roughness_delta': params['roughness_delta'],
                      'candidate_label': candidate, 'media_inputs': [p.relative_to(ROOT).as_posix() for p in media]}
            report['samples'].append({'image': 'images/' + filename, 'candidate_label': candidate,
                                      'plan_state_id': job['id'], 'states': states,
                                      'visible_bins': actual['visible_bins'], 'bin_count': len(selected),
                                      'open_pose': target['open_pose'], 'mask_visible': mask_paths['visible'],
                                      'mask_state_visible': state_paths['visible'], 'labels': labels, 'actual': actual})
            run.record_sample(job['id'], output=out / 'images' / filename,
                              sources=job['sources'], parameters=params, actual=actual)
            save_report()
        report['status'] = 'complete_candidates'
    except BaseException as exc:
        report['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        report['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        save_report()


def _guard_inputs():
    # Blender 5.2 can mark a freshly loaded file dirty. Reload in this isolated
    # background process so the guarded source is always the on-disk scene.
    if not bpy.app.background:
        raise ValueError('render requires an isolated background Blender process')
    if not bpy.data.filepath:
        raise ValueError('an existing workspace scene is required')
    scene = Path(bpy.data.filepath).resolve()
    if not scene.is_relative_to(ROOT) or not scene.is_file():
        raise ValueError('scene must be an existing workspace file')
    if bpy.ops.wm.open_mainfile(filepath=str(scene), load_ui=False) != {'FINISHED'}:
        raise RuntimeError('could not reload the guarded scene')
    if Path(bpy.data.filepath).resolve() != scene:
        raise RuntimeError('loaded scene changed during source discovery')
    from datasets.scripts.common.blender_workspace import BIN_PHOTO, BINS_PHOTO_SCENE, activate_scene
    activate_scene(BINS_PHOTO_SCENE, (BIN_PHOTO,))
    procedural, media = {scene}, set()
    sidecar = scene.parent / 'manifest.json'
    if sidecar.is_file():
        procedural.add(sidecar)
        details = json.loads(sidecar.read_text(encoding='utf8'))
        if details.get('source_scene'):
            procedural.add(ROOT / details['source_scene'])
        if details.get('photo'):
            media.add(ROOT / details['photo'])
    for image in bpy.data.images:
        if image.source == 'FILE' and not image.packed_file:
            if not image.filepath:
                raise ValueError('scene image has no source path: ' + image.name)
            path = Path(bpy.path.abspath(image.filepath)).resolve()
            if path.is_file():
                media.add(path)
            else:
                raise ValueError('scene image source missing: ' + str(path))
    return sorted(media), sorted(procedural)


def main():
    raw = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--samples', type=int, default=12)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--device', choices=('CPU', 'OPTIX', 'CUDA'), default='OPTIX')
    parser.add_argument('--min-visibility', type=float, default=.70)
    parser.add_argument('--min-visible-pixels', type=int, default=24)
    args = parser.parse_args(raw)
    if (args.samples < 1 or min(args.width, args.height) < 1 or max(args.width, args.height) > 1280 or
            min(args.width, args.height) > 720 or not 0 < args.min_visibility <= 1 or args.min_visible_pixels < 1):
        parser.error('invalid render dimensions, samples or visibility thresholds')
    out, plan_path = within_workspace(args.out), within_workspace(args.plan)
    source = within_workspace(bpy.data.filepath)
    if not plan_path.is_file() or not bpy.data.filepath or not source.is_file():
        raise ValueError('load an existing scene and provide an existing plan inside the workspace')
    declaration = json.loads(plan_path.read_text(encoding='utf-8-sig'))
    destination = bin_state_foreground_dir(declaration['target_split'])
    if out != destination.resolve():
        parser.error(f'render output must be {destination.relative_to(ROOT).as_posix()}')
    if source != source_path('waste_bin_state', 'standalone_scene'):
        raise ValueError('loaded scene differs from waste_bin_state source.yaml')
    sys.path.insert(0, str(ROOT))
    from lab.training_common.production_guard import generation_guard
    media, procedural = _guard_inputs()
    with generation_guard('waste_bin.render_state:' + str(out.relative_to(ROOT)),
                          media, [out / name for name in (*BIN_STATE_RENDER_FOLDERS, 'manifest.json')],
                          {'argv': raw, 'blender_version': bpy.app.version_string},
                          code_paths=[Path(__file__).resolve(), ROOT / 'datasets/scripts/waste_bin_state/bin_render_contract.py',
                                      ROOT / 'datasets/scripts/common/blender_masks.py',
                                      ROOT / 'datasets/scripts/common/blender_workspace.py'],
                          procedural_inputs=[*procedural, plan_path, ROOT/'datasets/scripts/waste_bin_state/config/source.yaml', ROOT/'datasets/scripts/waste_bin_state/config/dataset.yaml'], root=ROOT, production_plan=declaration,
                          sample_output_paths=[out / 'images'],
                          auxiliary_output_paths=[out / name for name in ('mask_visible', 'mask_isolated',
                                                   'mask_state_visible', 'mask_state_isolated')]) as run:
        jobs = standalone_planned_jobs(run.plan, source.relative_to(ROOT).as_posix(), args.width, args.height,
                                       args.min_visibility, args.min_visible_pixels)
        if run.reused:
            return
        _generate(run, args, jobs, media)
        if json.loads((out / 'manifest.json').read_text(encoding='utf8'))['status'] != 'complete_candidates':
            raise RuntimeError('render did not finish; output retained without a completion receipt')


if __name__ == '__main__':
    main()
