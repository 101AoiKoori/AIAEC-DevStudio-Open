"""Restore the colored 9011 texture on a new copy of the fitted station.

Run Blender in the background with the existing station loaded. Pass a new
--output directory under datasets/assets/blender/meter. --preview renders one
640x480 OptiX frame and records the eight projections in endpoints.json.
"""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/meter/. Production remains paused.")

import argparse
import json
import math
from pathlib import Path
import sys

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from meter_production_guard import blender_media, guarded
from render_meter_batch import KEYPOINTS, project, texture_xy
from render_meter_candidates import STATION_PIVOT


ROOT = Path(__file__).resolve().parents[5]
CALIBRATION = ROOT / 'lab/features/meter/training/config/station_colored_face_calibration.json'


def needle_sweep(hand, endpoints, family):
    angles = []
    for bound in ('min', 'max'):
        ray = texture_xy(endpoints[f'{family}_{bound}']) - hand.location
        angles.append(math.degrees(math.atan2(ray.y, ray.x)))
    start, end = angles
    return [start, -((start - end) % 360)]


def render_preview(scene, hands, sweeps, out):
    preferences = bpy.context.preferences.addons['cycles'].preferences
    preferences.compute_device_type = 'OPTIX'
    preferences.get_devices()
    devices = [device for device in preferences.devices if device.type == 'OPTIX']
    if not devices:
        raise ValueError('The optional station preview requires an OptiX GPU')
    for device in preferences.devices:
        device.use = device in devices
    scene.cycles.device = 'GPU'
    scene.cycles.samples = 12
    scene.render.resolution_x, scene.render.resolution_y = 640, 480
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    camera = scene.objects['Preview vehicle front']
    camera.animation_data_clear()
    camera.location.z = 9.65
    scene.camera = camera
    for family, fraction in (('temperature', .43), ('humidity', .56)):
        start, sweep = sweeps[family]
        hands[family].rotation_euler.z = math.radians(start + sweep * fraction)
    bpy.context.view_layer.update()
    points = {name: project(scene, camera, scene.objects[f'KPT-meter-{name}'].matrix_world.translation)
              for name in KEYPOINTS}
    display = [project(scene, camera, scene.objects[f'ROI-power-meter-{corner}'].matrix_world.translation)
               for corner in ('TL', 'TR', 'BR', 'BL')]
    if any(not (0 <= x < 1 and 0 <= y < 1 and z > 0) for x, y, z in (*points.values(), *display)):
        raise ValueError('Station preview does not contain the eight points and the lower display')
    scene.render.filepath = str(out / 'preview.png')
    bpy.ops.render.render(write_still=True)
    return {'image': (out / 'preview.png').relative_to(ROOT).as_posix(),
            'image_size': [640, 480], 'devices': [d.name for d in devices],
            'keypoints_xy_depth_normalized': points, 'display_quad_xy_depth_normalized': display}


def main():
    raise RuntimeError("Retired production entry; use datasets/scripts/meter/ and its source.yaml")


def _legacy_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    owner = ROOT / 'datasets/assets/blender/meter'
    out = args.output.resolve()
    source = Path(bpy.data.filepath).resolve()
    if not out.is_relative_to(owner) or out == owner or source.is_relative_to(out):
        parser.error('output must be a separate directory under datasets/assets/blender/meter')
    if not source.is_file() or not source.is_relative_to(owner):
        parser.error('load a Blender asset under datasets/assets/blender/meter before restoring its texture')
    with guarded('restore_station_colored_face', args, __file__, source, extra_inputs=[CALIBRATION],
                 code_dependencies=[Path(__file__).with_name(name) for name in (
                     'render_meter_batch.py', 'render_meter_candidates.py', 'power_meter_room.py')],
                 media_probe=lambda: blender_media(bpy)) as run:
        if run.reused:
            return
        if out.exists():
            parser.error('output already exists')
        record = json.loads(CALIBRATION.read_text(encoding='utf8'))
        scene = bpy.context.scene
        if scene.name != 'Station prototype previews' or scene.render.engine != 'CYCLES':
            raise ValueError('The fitted Cycles station scene must be active')
        pivot = scene.objects['Station dial tilt pivot']
        tilt = 90 + math.degrees(pivot.rotation_euler.x)
        if not 60 <= tilt <= 70:
            raise ValueError('The station dial ground angle must be 60 to 70 degrees')
        face = scene.objects['Station PDF_face_inpainted']
        nodes = [node for material in face.data.materials if material and material.use_nodes
                 for node in material.node_tree.nodes if node.type == 'TEX_IMAGE' and node.image]
        if len(nodes) != 1 or face.parent != pivot:
            raise ValueError('Expected one texture node on the fitted dial face')
        previous = nodes[0].image
        source_face = ROOT / record['source_face']
        image = bpy.data.images.load(str(source_face), check_existing=False)
        if list(image.size) != record['image_size']:
            raise ValueError('The colored texture does not match the endpoint reference dimensions')
        image.pack()
        image.filepath = '//../blender_9011_reference/face_generated.png'
        nodes[0].image = image
        if previous.users == 0:
            bpy.data.images.remove(previous)

        endpoints = record['endpoints']
        for name in KEYPOINTS[4:]:
            marker = scene.objects[f'KPT-meter-{name}']
            if marker.parent != pivot:
                raise ValueError(f'Printed scale point is not on the station dial: {name}')
            marker.location = texture_xy(endpoints[name]) - STATION_PIVOT
        hands = {family: scene.objects[f'Station {family}_needle_rotate_Z']
                 for family in ('temperature', 'humidity')}
        sweeps = {family: needle_sweep(hand, endpoints, family) for family, hand in hands.items()}
        if any(not -300 < sweep < -200 for _, sweep in sweeps.values()):
            raise ValueError('The fitted endpoints do not form the expected clockwise needle range')
        for family, hand in hands.items():
            start, sweep = sweeps[family]
            hand.animation_data_clear()
            hand.rotation_euler.z = math.radians(start + .5 * sweep)

        out.mkdir(parents=True)
        scene_file = out / 'station.blend'
        bpy.ops.wm.save_as_mainfile(filepath=str(scene_file), copy=True, relative_remap=False)
        record.update(source_blend=scene_file.relative_to(ROOT).as_posix(),
                      mother_blend=source.relative_to(ROOT).as_posix(),
                      calibration_config=CALIBRATION.relative_to(ROOT).as_posix(),
                      needle_sweep_deg=sweeps, dial_ground_angle_deg=tilt)
        if args.preview:
            record['preview'] = render_preview(scene, hands, sweeps, out)
        (out / 'endpoints.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        print('STATION_COLORED_FACE_READY', scene_file, flush=True)


if __name__ == '__main__':
    main()
