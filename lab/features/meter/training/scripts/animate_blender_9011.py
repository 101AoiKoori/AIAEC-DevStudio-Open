"""Animate the existing meter scene and render a sparse, inspectable sequence.

Run with Blender --background MODEL --python-exit-code 1 --python THIS --
--output NEW_DIRECTORY [--frames 721] [--step 45]. Existing lights are untouched.
Geometry and nominal values are illustrative; no training labels are asserted.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from meter_blender_action import find_object_curve, object_action_curves

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector


def motion(frame, frames):
    # A smooth, slow, bounded trajectory; coprime-ish periods vary combinations.
    t = (frame - 1) / 24.0
    return (0.65 * math.sin(2 * math.pi * t / 19),
            0.45 * math.sin(2 * math.pi * t / 26 + 0.5),
            5.8 + 0.20 * math.sin(2 * math.pi * t / 23),
            0.5 + 0.45 * math.sin(2 * math.pi * t / 29),
            0.5 + 0.45 * math.sin(2 * math.pi * t / 23 + 1.1))


def xy(p, z=.065):
    return Vector(((p[0] - 627) / 627, (627 - p[1]) / 627, z))


def projected(scene, camera, point):
    p = world_to_camera_view(scene, camera, Vector(point))
    return [float(p.x), float(1 - p.y), float(p.z)]


def scale_camera_animation(cam, scale):
    curves = object_action_curves(cam)
    loc = [find_object_curve(curves, 'location', i) for i in range(3)]
    rot = [find_object_curve(curves, 'rotation_euler', i) for i in range(3)]
    for kp in loc[2].keyframe_points:
        kp.co.y *= scale
        kp.handle_left.y *= scale
        kp.handle_right.y *= scale
    loc[2].update()
    frames_by_axis = [[round(k.co.x, 5) for k in c.keyframe_points] for c in rot]
    if not (frames_by_axis[0] == frames_by_axis[1] == frames_by_axis[2]):
        raise RuntimeError('Camera rotation curves have different keyframe times')
    previous = None
    for index, frame in enumerate(frames_by_axis[0]):
        position = Vector(tuple(c.evaluate(frame) for c in loc))
        direction = -position.normalized()
        right = direction.cross(Vector((0, 1, 0))).normalized()
        up = right.cross(direction).normalized()
        rotation = Matrix((right, up, -direction)).transposed().to_euler()
        if previous is not None:
            rotation.make_compatible(previous)
        previous = rotation.copy()
        for axis, curve in enumerate(rot):
            curve.keyframe_points[index].co.y = rotation[axis]
    for curve in rot:
        curve.update()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--geometry', required=True)
    parser.add_argument('--frames', type=int, default=721)
    parser.add_argument('--step', type=int, default=45)
    parser.add_argument('--resolution', type=int, default=None,
                        help='Square output resolution in pixels (default: scene settings)')
    parser.add_argument('--camera-distance-scale', type=float, default=1.0,
                        help='Scale animated camera Z for this render only')
    parser.add_argument('--reuse-animation', action='store_true',
                        help='Render existing animation without replacing keyframes')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if args.frames < 2 or args.step < 1:
        parser.error('frames must be at least 2 and step at least 1')
    if args.resolution is not None and not 128 <= args.resolution <= 4096:
        parser.error('resolution must be between 128 and 4096 pixels')
    if not .4 <= args.camera_distance_scale <= 1.5:
        parser.error('camera distance scale must be between 0.4 and 1.5')
    if args.camera_distance_scale != 1.0 and not args.reuse_animation:
        parser.error('camera distance scale requires --reuse-animation')
    out = Path(args.output).resolve()
    if out.exists():
        parser.error('output exists; refusing to overwrite')
    if not bpy.data.filepath:
        parser.error('open the user model first')
    geom = json.loads(Path(args.geometry).read_text(encoding='utf-8-sig'))
    scene = bpy.context.scene
    if args.resolution is not None:
        scene.render.resolution_x = scene.render.resolution_y = args.resolution
        scene.render.resolution_percentage = 100
    cam = bpy.data.objects['training_camera']
    temp = bpy.data.objects['temperature_needle_rotate_Z']
    humid = bpy.data.objects['humidity_needle_rotate_Z']
    # Do not silently discard user-created animation on these objects.
    for obj in (cam, temp, humid):
        animated = bool(obj.animation_data and obj.animation_data.action)
        if animated and not args.reuse_animation:
            raise RuntimeError('Existing animation on %s; refusing to overwrite' % obj.name)
        if args.reuse_animation and not animated:
            raise RuntimeError('Missing animation on %s' % obj.name)
    if scene.camera != cam:
        raise RuntimeError('Active camera is not training_camera; inspect scene first')
    if args.reuse_animation and (scene.frame_start != 1 or scene.frame_end != args.frames):
        raise RuntimeError('Frame range differs from saved animation')
    scene.frame_start, scene.frame_end, scene.render.fps = 1, args.frames, 24
    last_euler = None
    keyframes = list(range(1, args.frames + 1, 6))
    if keyframes[-1] != args.frames:
        keyframes.append(args.frames)
    for f in ([] if args.reuse_animation else keyframes):
        x, y, z, tf, hf = motion(f, args.frames)
        cam.location = (x, y, z)
        direction = (Vector((0, 0, 0)) - cam.location).normalized()
        right = direction.cross(Vector((0, 1, 0))).normalized()
        up = right.cross(direction).normalized()
        rotation = Matrix((right, up, -direction)).transposed().to_euler()
        if last_euler is not None:
            rotation.make_compatible(last_euler)
        cam.rotation_euler = rotation
        last_euler = rotation.copy()
        cam.keyframe_insert(data_path='location', frame=f, group='Slow camera orbit')
        cam.keyframe_insert(data_path='rotation_euler', frame=f, group='Slow camera orbit')
        temp.rotation_euler.z = math.radians(-147 - 241 * tf)
        humid.rotation_euler.z = math.radians(44 - 264 * hf)
        temp.keyframe_insert(data_path='rotation_euler', index=2, frame=f, group='Temperature')
        humid.keyframe_insert(data_path='rotation_euler', index=2, frame=f, group='Humidity')
    if not args.reuse_animation:
        for obj in (cam, temp, humid):
            for fc in object_action_curves(obj):
                for kp in fc.keyframe_points:
                    kp.interpolation = 'BEZIER'
                    kp.handle_left_type = kp.handle_right_type = 'AUTO_CLAMPED'
    if args.camera_distance_scale != 1.0:
        scale_camera_animation(cam, args.camera_distance_scale)
    def set_frame(frame):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
    # Check *all* animation frames, not just rendered samples. Project a conservative
    # outline at the front and rear of the housing, and check upright + margin.
    worst = 1.0
    for f in range(1, args.frames + 1):
        set_frame(f)
        points = [projected(scene, cam, (1.05 * math.cos(i * math.tau / 128),
                                       1.05 * math.sin(i * math.tau / 128), z))
                  for z in (-.06, .15) for i in range(128)]
        margin = min(min(x, y, 1-x, 1-y) for x, y, depth in points)
        top = projected(scene, cam, (0, .8, .1))
        bottom = projected(scene, cam, (0, -.8, .1))
        if min(p[2] for p in points) <= 0 or margin < .025 or top[1] >= bottom[1]:
            raise RuntimeError('Frame %d fails full-frame/upright check (margin %.4f)' % (f, margin))
        worst = min(worst, margin)
    scene.frame_set(1)
    if not args.reuse_animation:
        bpy.context.preferences.filepaths.save_version = 0
        bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'images').mkdir()
    manifest = {'purpose':'Sparse animation sequence for inspection, not validated training data',
                'source_blend':bpy.data.filepath, 'fps':24, 'frames':args.frames,
                'sample_step':args.step, 'minimum_margin_all_frames':worst,
                'render_resolution': [scene.render.resolution_x, scene.render.resolution_y],
                'render_resolution_percentage': scene.render.resolution_percentage,
                'camera_distance_scale': args.camera_distance_scale,
                'geometry_status':'AI-generated face and assumed equally spaced scales, not verified',
                'label_status':'Projected geometric candidates only; no visibility or actual scale verification',
                'samples':[]}
    frames = list(range(1, args.frames + 1, args.step))
    if frames[-1] != args.frames:
        frames.append(args.frames)
    for f in frames:
        set_frame(f)
        filename = 'frame_%04d.png' % f
        scene.render.filepath = str(out / 'images' / filename)
        scene.render.image_settings.file_format = 'PNG'
        bpy.ops.render.render(write_still=True)
        points3 = [temp.matrix_world @ Vector((0,0,0)), temp.matrix_world @ Vector((.76,0,0)),
                   humid.matrix_world @ Vector((0,0,0)), humid.matrix_world @ Vector((.31,0,0)),
                   xy(geom['temperature'][0]['end']), xy(geom['temperature'][-1]['end']),
                   xy(geom['humidity'][0]['end']), xy(geom['humidity'][-1]['end'])]
        manifest['samples'].append({'frame':f, 'file':'images/'+filename,
              'camera_location':list(cam.location), 'camera_rotation_euler':list(cam.rotation_euler),
              'temperature_angle_rad':temp.rotation_euler.z, 'humidity_angle_rad':humid.rotation_euler.z,
              'keypoint_candidates_xy_depth':[projected(scene,cam,p) for p in points3]})
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('SEQUENCE_COMPLETE', out, 'samples', len(frames), 'minimum_margin', worst, flush=True)


if __name__ == '__main__':
    main()
