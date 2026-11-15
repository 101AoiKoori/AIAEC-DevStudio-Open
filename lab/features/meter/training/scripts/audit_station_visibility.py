"""Render a glass-on/off meter pair and trace candidate landmark geometry.

This audit cannot infer keypoint visibility or real-meter truth. Run in a
separate background Blender process with the saved station preview scene.
"""
import argparse
import json
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_meter_batch import KEYPOINTS, project

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard
from meter_production_guard import blender_media
SOURCE = ROOT / 'datasets/assets/blender/batch_scenes/meter/station_v4_room_context_grounded_20261112/preview_scene.blend'
OWNER = ROOT / 'datasets/exports/meter'


def _execute(args):
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(OWNER.resolve()) or output == OWNER.resolve() or output.exists():
        parser.error('output must be a new directory below datasets/exports/meter')
    if Path(bpy.data.filepath).resolve() != SOURCE.resolve():
        raise ValueError('audit requires the reviewed room preview scene')
    scene = bpy.context.scene
    if scene.name != 'Station prototype previews' or scene.camera is None:
        raise ValueError('station scene or camera missing')
    glass = scene.objects['Station glass_reflection_surface']
    if glass.hide_render:
        raise ValueError('source scene does not have the expected visible glass')
    scene.render.resolution_x, scene.render.resolution_y = 640, 480
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.threads_mode, scene.render.threads = 'FIXED', 4
    devices = ['CPU']
    if args.device == 'OPTIX':
        preferences = bpy.context.preferences.addons['cycles'].preferences
        preferences.compute_device_type = 'OPTIX'
        preferences.get_devices()
        selected = [device for device in preferences.devices if device.type == 'OPTIX']
        if not selected:
            raise RuntimeError('No OptiX device available; use --device CPU explicitly')
        for device in preferences.devices:
            device.use = device in selected
        scene.cycles.device = 'GPU'
        devices = [device.name for device in selected]
    else:
        scene.cycles.device = 'CPU'
    scene.cycles.samples, scene.cycles.seed = 8, 9011
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    camera = scene.camera
    points = {}
    for name in KEYPOINTS:
        marker = scene.objects['KPT-meter-' + name]
        target = marker.matrix_world.translation
        xy = project(scene, camera, target)
        if xy[2] <= 0 or not (0 <= xy[0] <= 1 and 0 <= xy[1] <= 1):
            raise ValueError('landmark outside camera: ' + name)
        origin = camera.matrix_world.translation.copy()
        distance = (target - origin).length
        direction = (target - origin).normalized()
        hit, location, _, _, obj, _ = scene.ray_cast(deps, origin, direction, distance=distance)
        points[name] = {'projection': xy, 'parent': marker.parent.name if marker.parent else None,
                        'first_ray_hit': obj.name if hit else None,
                        'first_hit_distance': (location-origin).length if hit else None,
                        'visibility': 'unknown'}
    output.mkdir(parents=True)
    report = {'status': 'running', 'role': 'development_regression',
              'source_scene': SOURCE.relative_to(ROOT).as_posix(),
              'resolution': [640, 480], 'device': args.device, 'devices': devices, 'threads': 4, 'samples': 8,
              'points': points, 'images': {},
              'scope': 'Straight ray hits and glass-on/off pixels are evidence only, not visibility or real-meter labels.'}
    try:
        for name, hide in [('glass_on', False), ('glass_off', True)]:
            glass.hide_render = hide
            scene.render.filepath = str(output / (name + '.png'))
            bpy.ops.render.render(write_still=True)
            report['images'][name] = name + '.png'
        report['status'] = 'complete_candidates'
    except BaseException as exc:
        report.update(status='failed_or_interrupted', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (output / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print('STATION_VISIBILITY_AUDIT', output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('CPU', 'OPTIX'), default='OPTIX')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    output = (ROOT / args.output).resolve()
    with generation_guard('meter.audit_station_visibility:' + output.relative_to(ROOT).as_posix(), blender_media(bpy), [output], {'device': args.device},
                          code_paths=[Path(__file__), Path(__file__).with_name('render_meter_batch.py'), Path(__file__).with_name('meter_production_guard.py')],
                          procedural_inputs=[SOURCE], root=ROOT) as run:
        if run.reused:
            print('REUSED', output)
            return
        _execute(args)


if __name__ == '__main__':
    main()
