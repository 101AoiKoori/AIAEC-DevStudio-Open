"""Render 36x24cm plate candidates over a task-owned Blender arena copy.

Use only in background Blender. The source scene is never saved. Rendered labels
are candidates until a representative image and possible background targets are
checked. All coordinates and support dimensions are scene-fitted, not measured.
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

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard
from lab.features.plate.training.scripts.plate_guard_inputs import blender_external_dependencies

SCENE = ROOT / 'datasets/assets/blender/plate/v4_arena_vehicle_20261113/arena_plate_working.blend'
FRAME = ROOT / 'datasets/annotated/site_capture_20261112/sequence_640_v1/full_scene/frame_003927.png'
VIDEO = ROOT / 'datasets/raw/site_capture_20261112/videos/full_scene.mp4'
ORIGIN = ROOT / 'datasets/assets/blender/plate/v4_arena_vehicle_20261113/source_assets.json'
REVIEW = ROOT / 'datasets/annotated/plate/v4_sequence_screen_20261113/candidate_manifest.json'
BASE = ROOT / 'datasets/exports/plate'


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def _gpu(scene):
    scene.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'
    prefs.get_devices()
    if not any(d.type == 'OPTIX' for d in prefs.devices):
        raise RuntimeError('No compatible OPTIX GPU available; CPU fallback disabled')
    for device in prefs.devices:
        device.use = device.type == 'OPTIX'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = 16
    scene.cycles.use_denoising = True


def _quad(name, material, with_uv=False):
    mesh = bpy.data.meshes.new(name + '_mesh')
    mesh.from_pydata([(0, 0, 0)] * 4, [], [(0, 1, 2, 3)])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(material)
    if with_uv:
        uv = mesh.uv_layers.new(name='UVMap')
        # Image-space crop [195,251,241,269] in a 640x360 source frame.
        for index, coord in zip(mesh.polygons[0].loop_indices,
                                ((195/640, 91/360), (241/640, 91/360),
                                 (241/640, 109/360), (195/640, 109/360))):
            uv.data[index].uv = coord
    return obj


def _place(obj, center, right, z0, z1, width):
    for vertex, xy in zip(obj.data.vertices,
                          ((-width/2, z0), (width/2, z0), (width/2, z1), (-width/2, z1))):
        point = center + right * xy[0]
        vertex.co = (point.x, point.y, xy[1])
    obj.data.update()


def _visible(scene, camera, plate, grid=7):
    """Reject target if projection is off-frame or any sampled face ray is occluded."""
    deps = bpy.context.evaluated_depsgraph_get()
    verts = [plate.matrix_world @ vertex.co for vertex in plate.data.vertices]
    projected = [world_to_camera_view(scene, camera, point) for point in verts]
    if any(v.z <= 0 for v in projected):
        return None, 'behind_camera'
    x0, x1 = min(v.x for v in projected), max(v.x for v in projected)
    y0, y1 = min(v.y for v in projected), max(v.y for v in projected)
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        return None, 'outside_frame'
    if (x1-x0) * 640 < 12 or (y1-y0) * 480 < 8:
        return None, 'plate_too_small'
    for iy in range(grid):
        for ix in range(grid):
            u, v = (ix + .5) / grid, (iy + .5) / grid
            target = verts[0].lerp(verts[1], u).lerp(verts[3].lerp(verts[2], u), v)
            ray = target - camera.matrix_world.translation
            hit, _, _, _, obj, _ = scene.ray_cast(deps, camera.matrix_world.translation,
                                                   ray.normalized(), distance=ray.length + .01)
            if not hit or obj.original != plate:
                return None, 'sampled_occlusion'
    return (x0, y0, x1, y1), None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=990)
    parser.add_argument('--train-count', type=int, default=780)
    parser.add_argument('--seed', type=int, default=20261113)
    argv = sys.argv[sys.argv.index('--') + 1:] if argv is None and '--' in sys.argv else (argv or [])
    args = parser.parse_args(argv)
    output = (ROOT / args.output).resolve()
    if output == BASE.resolve() or not output.is_relative_to(BASE.resolve()):
        parser.error('output must be a new directory inside datasets/exports/plate')
    if not 1 <= args.count <= 2000 or not 0 <= args.train_count <= args.count:
        parser.error('count must be 1..2000 and train-count must be within count')
    if Path(bpy.data.filepath).resolve() != SCENE.resolve():
        raise RuntimeError('Open the task-owned arena copy; never open or save the shared master')
    # Read-only enumeration before guard: exact external media need indexed provenance.
    external = blender_external_dependencies(bpy) - {SCENE, FRAME, VIDEO, ORIGIN, REVIEW}
    media = sorted({FRAME, VIDEO} | {p for p in external if p.suffix.lower() in
                  {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.mp4', '.mov', '.avi', '.mkv', '.exr', '.hdr'}})
    procedural = sorted({SCENE, ORIGIN, REVIEW} | (external - set(media)))
    params = {'count': args.count, 'train_count': args.train_count, 'seed': args.seed,
              'resolution': [640, 480], 'plate_size_m': [.36, .24],
              'camera_height_m': [.22, .42], 'target_height_m': .35,
              'profile_cycle': ['normal', 'dark', 'low_saturation', 'dark_low_saturation'],
              'source_crop_xyxy': [195, 251, 241, 269], 'visibility_ray_grid': 7,
              'max_attempts_per_sample': 40, 'samples': 16}
    with generation_guard('plate_arena_v4:' + rel(output), media, [output], params,
                          code_paths=[Path(__file__), ROOT / 'lab/features/plate/training/scripts/plate_guard_inputs.py'],
                          procedural_inputs=procedural, root=ROOT) as run:
        if run.reused:
            print('Reused indexed completed arena batch')
            return
        if output.exists():
            raise RuntimeError('Output already exists; preserve existing artifacts')
        _render(output, args, params, sorted(external), run.source_refs)


def _render(output, args, params, external, refs):
    scene = bpy.context.scene
    if scene.camera is None:
        raise RuntimeError('Active scene camera is missing')
    camera = scene.camera
    _gpu(scene)
    scene.render.resolution_x, scene.render.resolution_y = 640, 480
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    camera.data.type = 'PERSP'
    camera.data.lens, camera.data.sensor_width = 22, 36
    base_exposure = scene.view_settings.exposure

    board_mat = bpy.data.materials.new('plate_arena_support_material')
    board_mat.diffuse_color = (.25, .25, .25, 1)
    board_mat.use_nodes = True
    board_mat.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value = (.25, .25, .25, 1)
    face_mat = bpy.data.materials.new('plate_arena_source_frame')
    face_mat.use_nodes = True
    tex = face_mat.node_tree.nodes.new('ShaderNodeTexImage')
    tex.image = bpy.data.images.load(str(FRAME), check_existing=True)
    face_mat.node_tree.links.new(tex.outputs['Color'], face_mat.node_tree.nodes.get('Principled BSDF').inputs['Base Color'])
    board = _quad('plate_arena_support', board_mat)
    plate = _quad('plate_arena_face_36x24cm', face_mat, with_uv=True)

    # Apply saturation to the entire render rather than the plate texture alone.
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    nodes.clear()
    layers = nodes.new('CompositorNodeRLayers')
    saturation = nodes.new('CompositorNodeHueSat')
    composite = nodes.new('CompositorNodeComposite')
    scene.node_tree.links.new(layers.outputs['Image'], saturation.inputs['Image'])
    scene.node_tree.links.new(saturation.outputs['Image'], composite.inputs['Image'])
    output.mkdir(parents=True)
    (output / 'images').mkdir()
    (output / 'labels_candidate').mkdir()
    rng = random.Random(args.seed)
    records, skipped = [], []
    status = 'failed_or_interrupted'
    try:
        for index in range(args.count):
            split = 'train' if index < args.train_count else 'val'
            profile = params['profile_cycle'][index % len(params['profile_cycle'])]
            scene.view_settings.exposure = base_exposure - (.7 if 'dark' in profile else 0)
            saturation.inputs['Saturation'].default_value = .4 if 'low_saturation' in profile else 1.0
            accepted = None
            for attempt in range(params['max_attempts_per_sample']):
                cx, cy = rng.uniform(-1.10, 1.10), rng.uniform(-1.10, 1.10)
                distance, direction = rng.uniform(.85, 2.1), rng.uniform(-math.pi, math.pi)
                px, py = cx + math.cos(direction)*distance, cy + math.sin(direction)*distance
                if max(abs(px), abs(py)) > 1.85:
                    continue
                deps = bpy.context.evaluated_depsgraph_get()
                hit, point, normal, _, ground, _ = scene.ray_cast(deps, Vector((cx, cy, 2.0)),
                                                                   Vector((0, 0, -1)), distance=3.0)
                if not hit or normal.z < .9 or abs(point.z) > .1:
                    continue
                forward = Vector((cx-px, cy-py, 0)).normalized()
                angle = rng.uniform(-12, 12)
                a = math.radians(angle)
                right = Vector((forward.y*math.cos(a)+forward.x*math.sin(a),
                                -forward.x*math.cos(a)+forward.y*math.sin(a), 0))
                origin = Vector((cx, cy, point.z))
                _place(plate, origin, right, point.z+.23, point.z+.47, .36)
                _place(board, origin + forward*.012, right, point.z, point.z+.50, .44)
                camera.location = (px, py, point.z + rng.uniform(.22, .42))
                target = Vector((cx+rng.uniform(-.08,.08), cy+rng.uniform(-.08,.08),
                                 point.z+.35+rng.uniform(-.05,.05)))
                camera.rotation_euler = (target-camera.location).to_track_quat('-Z', 'Y').to_euler()
                bpy.context.view_layer.update()
                box, reason = _visible(scene, camera, plate)
                if box:
                    accepted = (box, point.copy(), ground.name, distance, angle, target, attempt+1)
                    break
            if accepted is None:
                skipped.append({'index': index, 'split': split, 'reason': 'no_valid_ground_and_visible_placement',
                                'attempt_limit': params['max_attempts_per_sample']})
                continue
            box, point, ground_name, distance, angle, target, attempts = accepted
            name = 'plate_arena_%04d' % index
            image, label = output / 'images' / (name + '.png'), output / 'labels_candidate' / (name + '.txt')
            scene.render.filepath = str(image)
            bpy.ops.render.render(write_still=True)
            x0, y0, x1, y1 = box
            label.write_text('0 %.8f %.8f %.8f %.8f\n' % ((x0+x1)/2, 1-(y0+y1)/2, x1-x0, y1-y0), encoding='utf-8')
            records.append({'image': rel(image), 'label': rel(label), 'split': split,
                'source_kind': 'blender_scene', 'generation_method': 'blender_render',
                'mother_scene': rel(SCENE), 'source_assets': [rel(FRAME)], 'parent_image': rel(FRAME),
                'source_video': rel(VIDEO), 'source_time_seconds': 131.000533,
                'source_crop_xyxy': [195,251,241,269], 'source_crop_pixels': [46,18],
                'plate_size_m': [.36,.24], 'texture_mapping': '46x18 source crop mapped to 36x24 cm; aspect ratio changes',
                'support_size_m': [.44,.50], 'support_size_status': 'illustrative',
                'ground_object': ground_name, 'ground_point': list(point), 'camera_position': list(camera.location),
                'camera_rotation_euler': list(camera.rotation_euler), 'camera_target': list(target),
                'camera_lens_mm': camera.data.lens, 'camera_distance_horizontal_m': distance,
                'face_yaw_offset_degrees': angle, 'placement_attempts': attempts,
                'illumination_profile': profile, 'simulated_exposure': scene.view_settings.exposure,
                'simulated_saturation': saturation.inputs['Saturation'].default_value,
                'visibility_test': '7x7 camera-to-face ray samples; thin occlusions may remain',
                'review_status': 'projected_box_and_sampled_visibility; background plate completeness needs review',
                'external_scene_assets': [rel(p) for p in external], 'image_size': [640,480]})
            (output / 'manifest.json').write_text(json.dumps({'status':'running', 'parameters':params,
                'source_refs':refs, 'records':records, 'skipped':skipped}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        status = 'rendered_review_pending'
    finally:
        (output / 'manifest.json').write_text(json.dumps({'status':status, 'parameters':params,
            'source_refs':refs, 'records':records, 'skipped':skipped}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status':status, 'rendered':len(records), 'skipped':len(skipped)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
