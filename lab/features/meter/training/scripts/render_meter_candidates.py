"""Render one-source meter images with eight *candidate* projected landmarks.

Run in Blender with --background SOURCE.blend --python-exit-code 1 --python THIS --
--output NEW_DIRECTORY --endpoints CONFIG [--count 10]. Nothing saves the blend.
All frames share a source group; this program does not create train/val/test splits.
Visibility, physical-meter agreement and scale readings are not established.
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bpy
from mathutils import Vector
from render_meter_batch import KEYPOINTS, look_at, project, texture_xy

ROOT = Path(__file__).resolve().parents[5]


def save_manifest(path, manifest):
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True)
    ap.add_argument('--endpoints', required=True)
    ap.add_argument('--count', type=int, default=10)
    ap.add_argument('--seed', type=int, default=9011)
    ap.add_argument('--resolution', type=int, default=1024)
    ap.add_argument('--samples', type=int, default=32)
    ap.add_argument('--device', choices=('CPU', 'OPTIX', 'CUDA'), default='CPU')
    ap.add_argument('--preflight', action='store_true', help='Check model and projection without writing or rendering')
    args = ap.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if not 1 <= args.count <= 10000 or not 256 <= args.resolution <= 2048 or not 1 <= args.samples <= 4096:
        ap.error('count 1..10000, resolution 256..2048, samples 1..4096 required')
    out = Path(args.output).resolve()
    export_root = (ROOT / 'datasets' / 'exports' / 'meter').resolve()
    if not out.is_relative_to(export_root) or out == export_root:
        ap.error('output must be a new directory under datasets/exports/meter')
    if out.exists():
        ap.error('output exists; refusing to overwrite')
    config = Path(args.endpoints).resolve()
    if not config.is_relative_to(ROOT):
        ap.error('endpoint configuration must be inside the workspace')
    endpoint_data = json.loads(config.read_text(encoding='utf-8-sig'))
    source = Path(bpy.data.filepath).resolve() if bpy.data.filepath else None
    expected_source = (ROOT / endpoint_data['source_blend']).resolve()
    if source != expected_source:
        ap.error('endpoint calibration applies only to the specified source blend')
    if endpoint_data['image_size'] != [1254, 1254]:
        ap.error('unexpected endpoint image dimensions')
    face = bpy.data.objects['PDF_face_inpainted']
    image_nodes = [node for material in face.data.materials if material and material.use_nodes
                   for node in material.node_tree.nodes if node.type == 'TEX_IMAGE' and node.image]
    reference_face = (ROOT / endpoint_data['source_face']).resolve()
    if (not reference_face.is_relative_to(ROOT) or len(image_nodes) != 1 or
            not image_nodes[0].image.packed_file or
            bytes(image_nodes[0].image.packed_file.data) != reference_face.read_bytes()):
        ap.error('packed face does not match the calibrated source image')
    if any(abs(face.matrix_world[i][j] - (1.0 if i == j else 0.0)) > 1e-6
           for i in range(4) for j in range(4)):
        ap.error('face transform differs from calibrated projection')
    uv_data = face.data.uv_layers.active.data
    for loop_index, loop in enumerate(face.data.loops):
        vertex = face.data.vertices[loop.vertex_index].co
        uv = uv_data[loop_index].uv
        if max(abs(vertex.x - (2 * uv.x - 1)), abs(vertex.y - (2 * uv.y - 1)),
               abs(vertex.z - .065)) > 1e-5:
            ap.error('face UV or geometry differs from calibrated projection')
    endpoints = endpoint_data['endpoints']
    if any(len(endpoints[k]) != 2 or not all(0 <= v < 1254 for v in endpoints[k]) for k in KEYPOINTS[4:]):
        ap.error('invalid endpoint coordinates')
    scene = bpy.context.scene
    camera = bpy.data.objects['training_camera']
    hands = [bpy.data.objects['temperature_needle_rotate_Z'], bpy.data.objects['humidity_needle_rotate_Z']]
    lights = [o for o in scene.objects if o.type == 'LIGHT']
    base_lights = {o.name: {'energy': o.data.energy, 'location': list(o.location)} for o in lights}
    # In-memory changes only. Blender source and its original actions are never saved.
    for obj in [camera] + hands:
        obj.animation_data_clear()
    scene.camera = camera
    scene.render.resolution_x = scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.cycles.samples = args.samples
    if scene.render.engine != 'CYCLES':
        ap.error('source must use the Cycles renderer')
    devices_used = ['CPU']
    if args.device == 'CPU':
        scene.cycles.device = 'CPU'
    else:
        preferences = bpy.context.preferences.addons['cycles'].preferences
        try:
            preferences.compute_device_type = args.device
            preferences.get_devices()
        except (TypeError, ValueError, RuntimeError) as exc:
            ap.error('requested device is unavailable: %s' % exc)
        selected = [d for d in preferences.devices if d.type == args.device]
        if not selected:
            ap.error('requested %s device is unavailable' % args.device)
        for device in preferences.devices:
            device.use = device in selected
        scene.cycles.device = 'GPU'
        devices_used = [d.name for d in selected]
    if args.preflight:
        camera.location = (0, 0, 4.2)
        look_at(camera, 0)
        bpy.context.view_layer.update()
        points3 = [hands[0].matrix_world @ Vector((0, 0, 0)), hands[0].matrix_world @ Vector((.76, 0, 0)),
                   hands[1].matrix_world @ Vector((0, 0, 0)), hands[1].matrix_world @ Vector((.31, 0, 0))]
        points3 += [texture_xy(endpoints[k]) for k in KEYPOINTS[4:]]
        points = [project(scene, camera, p) for p in points3]
        if any(p[2] <= 0 or not (0 <= p[0] <= 1 and 0 <= p[1] <= 1) for p in points):
            raise RuntimeError('Preflight projection falls outside the camera')
        print('PREFLIGHT_OK', args.device, devices_used, json.dumps(dict(zip(KEYPOINTS, points))), flush=True)
        return
    out.mkdir(parents=True, exist_ok=False)
    (out / 'images').mkdir()
    manifest = {
        'version': 1, 'status': 'running', 'requested_count': args.count,
        'source_blend': endpoint_data['source_blend'],
        'endpoint_config': config.relative_to(ROOT).as_posix(),
        'source_group': 'meter_9011_pbr_v2_generated_face',
        'split_status': 'unsplit; all draws share one model and face; not independent evaluation',
        'landmark_status': 'geometric projection candidates; four face ticks visually calibrated on generated texture; visibility unknown; not real meter truth',
        'face_calibration_status': endpoint_data['status'],
        'keypoint_order': KEYPOINTS, 'seed': args.seed, 'resolution': args.resolution,
        'cycles_samples': args.samples, 'requested_device': args.device,
        'cycles_device': scene.cycles.device, 'selected_devices': devices_used, 'original_lights': base_lights,
        'sample_records': 'samples.jsonl', 'completed_count': 0}
    manifest_path = out / 'manifest.json'
    save_manifest(manifest_path, manifest)
    rng = random.Random(args.seed)
    bins_t, bins_h = list(range(args.count)), list(range(args.count))
    rng.shuffle(bins_t)
    rng.shuffle(bins_h)
    try:
        for index in range(args.count):
            ft, fh = (bins_t[index] + rng.random()) / args.count, (bins_h[index] + rng.random()) / args.count
            hands[0].rotation_euler.z = math.radians(-147 - 241 * ft)
            hands[1].rotation_euler.z = math.radians(44 - 264 * fh)
            for camera_attempt in range(1, 65):
                camera.location = (rng.uniform(-.8, .8), rng.uniform(-.65, .65), rng.uniform(3.8, 4.7))
                roll = math.radians(rng.uniform(-8, 8))
                look_at(camera, roll)
                bpy.context.view_layer.update()
                outline = [project(scene, camera, (1.05 * math.cos(i * math.tau / 128),
                           1.05 * math.sin(i * math.tau / 128), z))
                           for z in (-.06, .15) for i in range(128)]
                x0, y0 = min(p[0] for p in outline), min(p[1] for p in outline)
                x1, y1 = max(p[0] for p in outline), max(p[1] for p in outline)
                if min(p[2] for p in outline) > 0 and min(x0, y0, 1-x1, 1-y1) >= .025:
                    break
            else:
                raise RuntimeError('Sample %d failed full-frame margin after 64 camera attempts' % index)
            lamp_values = {}
            for light in lights:
                base = base_lights[light.name]
                light.data.energy = base['energy'] * rng.uniform(.75, 1.25)
                light.location = Vector(base['location']) + Vector((rng.uniform(-.3, .3), rng.uniform(-.3, .3), 0))
                lamp_values[light.name] = {'energy': float(light.data.energy), 'location': list(light.location)}
            scene.cycles.seed = (args.seed + index) % 2147483647
            bpy.context.view_layer.update()
            points3 = [hands[0].matrix_world @ Vector((0, 0, 0)), hands[0].matrix_world @ Vector((.76, 0, 0)),
                       hands[1].matrix_world @ Vector((0, 0, 0)), hands[1].matrix_world @ Vector((.31, 0, 0))]
            points3 += [texture_xy(endpoints[k]) for k in KEYPOINTS[4:]]
            candidates = [project(scene, camera, p) for p in points3]
            filename = 'meter_%05d.png' % index
            scene.render.filepath = str(out / 'images' / filename)
            bpy.ops.render.render(write_still=True)
            sample = {
                'id': 'meter_%05d' % index, 'image': 'images/' + filename,
                'camera_location': list(camera.location), 'camera_rotation_euler': list(camera.rotation_euler),
                'camera_attempts': camera_attempt,
                'temperature_angle_rad': hands[0].rotation_euler.z, 'humidity_angle_rad': hands[1].rotation_euler.z,
                'lights': lamp_values, 'bbox_xyxy_candidate': [x0, y0, x1, y1],
                'keypoints_xy_depth_candidates': candidates, 'visibility': [None] * 8}
            with (out / 'samples.jsonl').open('a', encoding='utf8') as records:
                records.write(json.dumps(sample, ensure_ascii=False) + '\n')
            manifest['completed_count'] = index + 1
            save_manifest(manifest_path, manifest)
            print('CANDIDATE_PROGRESS', index + 1, '/', args.count, flush=True)
        manifest['status'] = 'complete_candidates'
    except BaseException as exc:
        manifest['status'] = 'failed_or_interrupted'
        manifest['error'] = '%s: %s' % (type(exc).__name__, exc)
        save_manifest(manifest_path, manifest)
        raise
    save_manifest(manifest_path, manifest)
    print('CANDIDATE_COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
