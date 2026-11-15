"""Render meter candidates in the user-provided arena using a background copy."""


import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.meter.render_meter_batch import KEYPOINTS, project
from datasets.scripts.meter.render_meter_candidates import (landmark_ray_evidence, station_dial_masks, station_tone_compositor,
                                     station_pose_rays, station_pose_labels)
from datasets.scripts.meter.meter_station_plan import validate_arena_declaration, allocated_arena_states

from datasets.scripts.common.config import source_path
from datasets.scripts.common.paths import sample_dir, split_dir
ARENA = source_path('meter', 'arena_source')
STATION = source_path('meter', 'station_source')
STATION_FACE = source_path('meter', 'station_face')
ENDPOINTS = source_path('meter', 'endpoint_config')


def render_station_foreground(scene, root, path):
    """Render the fitted station alone as RGBA for later source-traced compositing."""
    objects = list(scene.objects)
    hidden = {obj: obj.hide_render for obj in objects}
    old_transparent = scene.render.film_transparent
    old_mode = scene.render.image_settings.color_mode
    old_path = scene.render.filepath
    def belongs(obj):
        while obj:
            if obj == root:
                return True
            obj = obj.parent
        return False
    try:
        for obj in objects:
            if obj.type not in {'LIGHT', 'CAMERA'} and not belongs(obj):
                obj.hide_render = True
        scene.render.film_transparent = True
        scene.render.image_settings.color_mode = 'RGBA'
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
    finally:
        for obj, value in hidden.items():
            obj.hide_render = value
        scene.render.film_transparent = old_transparent
        scene.render.image_settings.color_mode = old_mode
        scene.render.filepath = old_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--count', type=int, default=4)
    ap.add_argument('--seed', type=int, default=20261113)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--pose', action='store_true', help='Export conservatively classified synthetic pose labels')
    ap.add_argument('--device', choices=('OPTIX', 'CUDA'), default='OPTIX')
    ap.add_argument('--samples', type=int, default=4)
    ap.add_argument('--preflight', action='store_true', help='Check source, framing and rays without rendering')
    ap.add_argument('--foreground', action='store_true', help='Also render station-only RGBA for traced real-background composites')
    ap.add_argument('--production-plan', help='Workspace JSON declaration for arena pose production')
    ap.add_argument('--parent-allocation', help='Prior-only scene quota record for this pilot')
    ap.add_argument('--face-override', help='Traced thick-tick face for the parametric station')
    args = ap.parse_args(sys.argv[sys.argv.index('--') + 1:])
    from datasets.scripts.meter.meter_production_guard import (guarded, blender_media, reload_unpacked_media,
                                        ensure_registered_blender_media, rebind_station_face)
    out = args.output.resolve()
    endpoints = json.loads(ENDPOINTS.read_text(encoding='utf8'))
    if (ROOT / endpoints['source_blend']).resolve() != STATION or (ROOT / endpoints['source_face']).resolve() != STATION_FACE:
        raise ValueError('configured endpoints must describe the selected station and face')
    declaration, plan_path, plan_bytes = None, None, None
    if not args.preflight:
        if not args.pose or args.offset:
            ap.error('planned arena production requires --pose and zero offset')
        if not args.production_plan:
            ap.error('--production-plan is required before sample generation')
        plan_path = Path(args.production_plan).resolve()
        if not plan_path.is_relative_to(ROOT) or not plan_path.is_file():
            ap.error('production plan must be an existing workspace JSON file')
        plan_bytes = plan_path.read_bytes()
        declaration = json.loads(plan_bytes.decode('utf-8-sig'))
        if args.parent_allocation:
            parent = Path(args.parent_allocation).resolve()
            if not parent.is_relative_to(ROOT) or not parent.is_file():
                ap.error('parent allocation must be an existing workspace JSON')
            parent_data = json.loads(parent.read_text(encoding='utf-8-sig'))
            quotas = {s['id']: s['planned_count'] for s in parent_data['allocation']['strata']}
            if quotas.get('arena') != args.count:
                ap.error('arena count differs from parent allocation')
        sources = [ARENA.relative_to(ROOT).as_posix(), STATION.relative_to(ROOT).as_posix()]
        validate_arena_declaration(declaration, count=args.count, sources=sources, foreground=args.foreground)
        if Path(bpy.data.filepath).resolve() != ARENA.resolve():
            ap.error('loaded arena must match the declared arena source')
    elif args.production_plan:
        ap.error('preflight does not generate samples and does not accept a production plan')
    auxiliary = [out / 'masks'] + ([out / 'foreground'] if args.foreground else [])
    with guarded('render_meter_arena_v4', args, __file__, bpy.data.filepath,
                 extra_inputs=[STATION, STATION_FACE, ENDPOINTS, ROOT/'datasets/scripts/meter/config/dataset.yaml', ROOT/'datasets/scripts/meter/config/source.yaml'] + ([args.face_override] if args.face_override else []) + ([args.parent_allocation] if args.parent_allocation else []) + ([plan_path] if plan_path else []),
                 code_dependencies=[Path(__file__).with_name('render_meter_batch.py'), Path(__file__).with_name('render_meter_candidates.py'), Path(__file__).with_name('power_meter_room.py'), Path(__file__).with_name('meter_station_plan.py')],
                 outputless=args.preflight, media_probe=lambda: blender_media(bpy), media_reload=lambda: reload_unpacked_media(bpy),
                 production_plan=declaration, sample_output_paths=[out / 'images'] if declaration else (),
                 auxiliary_output_paths=auxiliary if declaration else ()) as run:
        if plan_path is not None and plan_path.read_bytes() != plan_bytes:
            raise ValueError('production plan changed while entering the guard')
        if run.reused:
            if args.preflight:
                print('ARENA_METER_PREFLIGHT_OK', json.dumps(run.value), flush=True)
            return
        if out not in {sample_dir('meter'), split_dir('meter', 'train')} or (out.exists() and any(out.iterdir())):
            ap.error('candidate output must be the empty current meter sample or train directory')
        if (Path(bpy.data.filepath).resolve() != ARENA.resolve() or not 1 <= args.count <= 5000
                or args.offset < 0 or not 1 <= args.samples <= 32):
            ap.error('load arena source; count 1..5000, samples 1..32 and nonnegative offset required')
        if args.foreground and not args.pose:
            ap.error('--foreground requires --pose')
        scene = bpy.context.scene
        with bpy.data.libraries.load(str(STATION), link=False) as (src, dst):
            needed = ['Power meter station prototype', 'Station 9011 dial assembly']
            if not all(name in src.collections for name in needed):
                raise ValueError('station collections missing')
            dst.collections = needed
        rebind_station_face(bpy, STATION_FACE)
        ensure_registered_blender_media(bpy, run)
        if args.face_override:
            derived = Path(args.face_override).resolve()
            if derived != source_path('meter', 'face_override'):
                raise ValueError('face override must name the traced thick-tick derivative')
            face = bpy.data.objects['Station PDF_face_inpainted']
            nodes = [n for m in face.data.materials if m and m.use_nodes for n in m.node_tree.nodes
                     if n.type == 'TEX_IMAGE' and n.image]
            if len(nodes) != 1:
                raise ValueError('expected exactly one station face image')
            image = bpy.data.images.load(str(derived), check_existing=False)
            if tuple(image.size) != (1254, 1254):
                raise ValueError('derived face dimensions differ from calibration')
            nodes[0].image = image
        for collection in dst.collections:
            scene.collection.children.link(collection)
        root = bpy.data.objects.new('Meter arena fitted assembly', None)
        scene.collection.objects.link(root)
        root.scale = (.13, .13, .13)
        root.rotation_euler.x = math.pi / 2
        root.location = (0.0, 1.0, .76765)
        for name in ('Power_meter_station_root', 'Station dial tilt pivot'):
            bpy.data.objects[name].parent = root
        pivot = bpy.data.objects['Station dial tilt pivot']
        hands = [bpy.data.objects['Station temperature_needle_rotate_Z'], bpy.data.objects['Station humidity_needle_rotate_Z']]
        markers = [bpy.data.objects['KPT-meter-' + name] for name in KEYPOINTS]
        roi = [bpy.data.objects['ROI-power-meter-' + s] for s in ('TL', 'TR', 'BR', 'BL')]
        for obj in hands:
            obj.animation_data_clear()
        camera = scene.objects['Vehicle camera']
        camera.animation_data_clear()
        scene.camera = camera
        scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = 640, 480, 100
        scene.render.image_settings.file_format = 'PNG'
        scene.render.threads_mode, scene.render.threads = 'FIXED', 2
        if args.preflight:
            root.location.x, root.location.y = -.60, .15
            pivot.rotation_euler.x = math.radians(65 - 90)
            camera.location = Vector((-.6, -.75, .42))
            camera.rotation_euler = (Vector((root.location.x, root.location.y, .55)) - camera.location).to_track_quat('-Z', 'Y').to_euler()
            bpy.context.view_layer.update()
            points = [project(scene, camera, marker.matrix_world.translation) for marker in markers]
            screen = [project(scene, camera, marker.matrix_world.translation) for marker in roi]
            rays = station_pose_rays(scene, camera, markers)
            if any(p[2] <= 0 or not (0 <= p[0] < 1 and 0 <= p[1] < 1) for p in points + screen):
                raise ValueError('reference arena framing does not include the meter and display')
            result = {'resolution': [640, 480], 'keypoints': dict(zip(KEYPOINTS, points)),
                      'display_quad': screen, 'rays': rays}
            run.set_result(result)
            print('ARENA_METER_PREFLIGHT_OK', json.dumps(result), flush=True)
            return
        if args.pose:
            preferences = bpy.context.preferences.addons['cycles'].preferences
            try:
                preferences.compute_device_type = args.device
                preferences.get_devices()
            except (TypeError, ValueError, RuntimeError) as exc:
                raise ValueError('requested GPU unavailable: %s' % exc) from exc
            devices = [device for device in preferences.devices if device.type == args.device]
            if not devices:
                raise ValueError('requested GPU unavailable: ' + args.device)
            for device in preferences.devices:
                device.use = device in devices
            scene.render.engine = 'CYCLES'
            scene.cycles.device = 'GPU'
            scene.cycles.samples = args.samples
        tone = station_tone_compositor(scene)
        states = allocated_arena_states(run.plan, count=args.count)

        def apply_state(params):
            hands[0].rotation_euler.z = math.radians(-147 - 241 * params['temperature_fraction'])
            hands[1].rotation_euler.z = math.radians(44 - 264 * params['humidity_fraction'])
            pivot.rotation_euler.x = math.radians(params['ground_angle_deg'] - 90)
            root.location.x, root.location.y = params['station_x'], params['station_y']
            camera.location = Vector((params['camera_x'], params['camera_y'], params['camera_z']))
            camera.data.lens = params.get('camera_lens_mm', 22.)
            target = Vector((root.location.x, root.location.y, .55))
            camera.rotation_euler = (target - camera.location).to_track_quat('-Z', 'Y').to_euler()
            scene.view_settings.exposure = params['exposure_ev']
            tone.inputs['Saturation'].default_value = params['saturation']
            bpy.context.view_layer.update()

        # Check all planned projected points before creating outputs or rendering pixels.
        for state in states:
            apply_state(state['parameters'])
            projected = [project(scene, camera, obj.matrix_world.translation) for obj in markers + roi]
            if any(point[2] <= 0 or not (0 <= point[0] < 1 and 0 <= point[1] < 1) for point in projected):
                raise ValueError('planned arena geometry falls outside the frame')
        out.mkdir(parents=True, exist_ok=True)
        for sub in ('images', 'annotations', 'masks'):
            (out / sub).mkdir()
        if args.pose:
            (out / 'labels').mkdir()
        if args.foreground:
            (out / 'foreground').mkdir()
        manifest = {'status': 'running', 'requested_count': args.count, 'completed_count': 0,
                    'arena': ARENA.relative_to(ROOT).as_posix(), 'station': STATION.relative_to(ROOT).as_posix(),
                    'face_override': Path(args.face_override).resolve().relative_to(ROOT).as_posix() if args.face_override else None,
                    'parent_scene_relation': 'fitted synthetic scene; camera and metric dimensions not measured',
                    'camera': 'Vehicle camera', 'resolution': [640, 480], 'threads': 2,
                    'seed': args.seed, 'offset': args.offset, 'samples': 'samples.jsonl',
                    'point_policy': 'synthetic projected geometry; visibility unknown, ray evidence not pixel truth',
                    'roi_policy': 'independent display ROI; visibility unknown', 'split': 'unassigned'}
        if args.pose:
            manifest.update(device=args.device, selected_devices=[d.name for d in devices],
                cycles_samples=args.samples, pose_label_count=0, keypoint_order=list(KEYPOINTS),
                point_policy='known synthetic geometry; v0 outside, v1 weak or obstructed; '
                             'v2 only unobstructed pivot cap with local contrast')
        if args.foreground:
            manifest['foreground_policy'] = ('station-only RGBA from the same camera and state; '
                'transparent world with existing lighting; compositing and ground contact need separate checks')
        def save():
            (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf8')
        save()
        try:
            for i in range(args.count):
                idx = args.offset + i
                state = states[i]
                params = state['parameters']
                apply_state(params)
                temp, humid = params['temperature_fraction'], params['humidity_fraction']
                ground, exposure, saturation = params['ground_angle_deg'], params['exposure_ev'], params['saturation']
                if args.pose:
                    scene.cycles.seed = (args.seed + idx) % 2147483647
                bpy.context.view_layer.update()
                points = {name: project(scene, camera, marker.matrix_world.translation) for name, marker in zip(KEYPOINTS, markers)}
                screen = [project(scene, camera, obj.matrix_world.translation) for obj in roi]
                rays = station_pose_rays(scene, camera, markers) if args.pose else landmark_ray_evidence(scene, camera, markers)
                name = f'arena_meter_{idx:05d}'
                scene.render.filepath = str(out / 'images' / (name + '.png'))
                bpy.ops.render.render(write_still=True)
                if args.foreground:
                    render_station_foreground(scene, root, out / 'foreground' / (name + '.png'))
                mask = station_dial_masks(scene, pivot, out, name)
                record = {'id': name, 'image': 'images/' + name + '.png', 'image_size': [640, 480],
                          'split': 'unassigned', 'camera': 'Vehicle camera', 'camera_location': list(camera.location),
                          'camera_rotation_euler': list(camera.rotation_euler), 'assembly_location': list(root.location),
                          'assembly_scale': .13, 'ground_angle_deg': ground, 'temperature_fraction_candidate': temp,
                          'humidity_fraction_candidate': humid, 'exposure_ev': scene.view_settings.exposure,
                          'saturation': saturation, 'keypoints_xy_depth_normalized_candidate': points,
                          'keypoint_visibility': dict.fromkeys(KEYPOINTS, 'unknown'), 'keypoint_ray_evidence': rays,
                          'display_roi_quad_xy_depth_normalized_candidate': screen, 'display_roi_visibility': 'unknown',
                          'display_text': 'synthetic 1234 kWh', 'dial_raster_evidence': mask,
                          'review_status': 'synthetic geometry candidate; per-point visibility and physical correspondence unknown'}
                if args.foreground:
                    record['foreground_rgba'] = 'foreground/' + name + '.png'
                if args.pose:
                    ordered = [points[key] for key in KEYPOINTS]
                    pose = station_pose_labels(scene, camera, markers, ordered, rays, out / 'images' / (name + '.png'))
                    record.update(keypoint_visibility=pose, reflection_status='transparent cover traversed; reflections may weaken features',
                                  state={'exposure_ev': exposure, 'saturation': saturation},
                                  review_status='synthetic geometry with conservative visibility; physical correspondence unknown')
                    box = mask['bbox_xyxy_pixels']
                    if box:
                        x1, y1, x2, y2 = box
                        values = [0, (x1+x2)/1280, (y1+y2)/960, (x2-x1)/640, (y2-y1)/480]
                        for key, point in zip(KEYPOINTS, ordered):
                            visibility = pose[key]['visibility']
                            values.extend((point[0], point[1], visibility) if visibility else (0, 0, 0))
                        (out / 'labels' / (name + '.txt')).write_text(
                            ' '.join(str(v) if isinstance(v, int) else f'{v:.8f}' for v in values) + '\n', encoding='utf8')
                        record.update(annotation_status='synthetic_pose_geometric', pose_label='labels/' + name + '.txt')
                        manifest['pose_label_count'] += 1
                    else:
                        raise ValueError('planned candidate pose has no eligible bounding box')
                (out / 'annotations' / (name + '.json')).write_text(json.dumps(record, indent=2) + '\n', encoding='utf8')
                with (out / 'samples.jsonl').open('a', encoding='utf8') as stream:
                    stream.write(json.dumps(record) + '\n')
                run.record_sample(state['id'], output=out / 'images' / (name + '.png'),
                                  sources=state['sources'], parameters=params,
                                  actual={'arena': sources[0], 'station': sources[1], 'seed': scene.cycles.seed,
                                          'camera_location': list(camera.location),
                                          'camera_rotation_euler': list(camera.rotation_euler),
                                          'assembly_location': list(root.location),
                                          'keypoints': points, 'display_quad': screen,
                                          'dial_raster_evidence': mask,
                                          'foreground_rgba': record.get('foreground_rgba'),
                                          'pose_label': record['pose_label'], 'visibility': record['keypoint_visibility']})
                manifest['completed_count'] = i + 1
                if (i + 1) % 25 == 0:
                    save()
                print('ARENA_METER_PROGRESS', i + 1, '/', args.count, flush=True)
            manifest['status'] = 'complete_synthetic_pose_candidates' if args.pose else 'complete_candidates'
        except BaseException as exc:
            manifest.update(status='failed_or_interrupted', error=str(exc))
            raise
        finally:
            save()


if __name__ == '__main__':
    main()
