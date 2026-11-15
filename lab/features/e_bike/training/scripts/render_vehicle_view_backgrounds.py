"""Render low-mounted camera backgrounds from the user's arena master.

Run in a separate background Blender process. Camera height and road markings
are fitted scene parameters, not measurements from a deployed vehicle.
"""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/e_bike/. Production remains paused.")

import argparse
import json
import random
import struct
import sys
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[5]
MASTER = ROOT / 'datasets/assets/blender/shared/arena_4m2_pbr_20261112/arena_master_user_20261113.blend'
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard  # noqa: E402


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--view-start', type=int, default=0)
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--adopt-extra', action='store_true', help='validate and adopt the previously rendered 8–23 batch')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    out = (ROOT / args.output).resolve()
    owner = (ROOT / 'datasets/exports/e_bike').resolve()
    if (not out.is_relative_to(owner) or out == owner or not 1 <= args.samples <= 64
            or args.view_start < 0 or args.count < 1 or args.view_start + args.count > 48):
        raise ValueError('output and sample count outside bounded e-bike export scope')
    if Path(bpy.data.filepath).resolve() != MASTER.resolve():
        raise ValueError('load the user arena master in an independent background process')
    if args.adopt_extra:
        adopt_extra(out)
        return
    params = {'samples': args.samples, 'device': 'OPTIX', 'size': [640, 480],
              'camera_height_m': .18, 'lens_mm': 22, 'view_start': args.view_start, 'views': args.count,
              'seed': 20261113}
    with generation_guard('e_bike.vehicle_view_backgrounds:' + relative(out), [], [out], params,
                          procedural_inputs=[MASTER], code_paths=[Path(__file__)], root=ROOT) as run:
        if run.reused:
            return
        if out.exists():
            raise ValueError('output exists; choose a new directory')
        render(out, args.samples, args.view_start, args.count)


def adopt_extra(out):
    """Record adoption after checking existing outputs; this does not claim a render receipt."""
    expected = ROOT / 'datasets/exports/e_bike/v4_vehicle_view_backgrounds_extra_20261113'
    random_dir = ROOT / 'datasets/exports/e_bike/v4_vehicle_view_backgrounds_random_20261113'
    if out != expected.resolve():
        raise ValueError('adoption is restricted to the interrupted 8–23 batch')
    other = [random_dir / f'background_{i:03d}.png' for i in range(24, 48)]
    with generation_guard('e_bike.vehicle_view_backgrounds_adoption:' + relative(out), other, [out],
                          {'scope': 'validation and adoption of interrupted 8–23 outputs', 'count': 16},
                          procedural_inputs=[MASTER, random_dir / 'manifest.json'],
                          code_paths=[Path(__file__)], adopt_outputs=[out], root=ROOT) as run:
        if run.reused:
            return
        manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
        rows = manifest['records']
        if manifest['source_scene'] != relative(MASTER) or len(rows) != 16:
            raise ValueError('unexpected mother scene or record count')
        other_bytes = [path.read_bytes() for path in other]
        positions = []
        for index, row in zip(range(8, 24), rows):
            path = out / f'background_{index:03d}.png'
            data = path.read_bytes()
            if (row['image'] != relative(path) or row['source_scene'] != relative(MASTER)
                    or row['resolution'] != [640, 480] or data[:8] != b'\x89PNG\r\n\x1a\n'
                    or struct.unpack('>II', data[16:24]) != (640, 480)
                    or data in other_bytes):
                raise ValueError('invalid provenance, image structure or duplicate: ' + str(path))
            i = index - 8
            expected_x = (-.38, -.18, .12, .36)[i % 4]
            expected_y = (-1.62, -1.34, -1.06, -.82)[i // 4]
            if (abs(row['camera_location_m'][0] - expected_x) > 1e-5 or
                    abs(row['camera_location_m'][1] - expected_y) > 1e-5 or
                    row['camera_lens_mm'] != 22 or row['camera_height_above_floor_m'] != .18):
                raise ValueError('unexpected camera parameters: ' + str(path))
            positions.append(tuple(row['camera_location_m']))
        if len(set(positions)) != 16:
            raise ValueError('repeated camera positions')
        run.set_result({'adopted': 16, 'original_render_receipt': 'missing_code_changed_during_render',
                        'validation': 'mother scene and fitted camera claims, PNG dimensions and exact-byte cross-batch duplicates'})
    print(json.dumps({'adopted': 16, 'output': relative(out)}))


def render(out, samples, view_start, count):
    raise RuntimeError("Retired production API; use datasets/scripts/e_bike/render_e_bike_blender_batch.py")
    scene = bpy.context.scene
    floor = scene.objects.get('Arena floor 4.2 x 4.2 m')
    camera = scene.objects.get('Vehicle camera')
    if floor is None or camera is None:
        raise ValueError('arena floor or vehicle camera missing')
    preferences = bpy.context.preferences.addons['cycles'].preferences
    preferences.compute_device_type = 'OPTIX'
    preferences.get_devices()
    if not any(device.type == 'OPTIX' for device in preferences.devices):
        raise RuntimeError('no Cycles OPTIX device')
    for device in preferences.devices:
        device.use = device.type == 'OPTIX'
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = 640, 480
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.camera = camera
    camera.data.lens = 22
    # The user arena has boundary rails but no continuous road edges. Add
    # fitted white strips in the temporary in-memory scene, above the floor.
    top = max((floor.matrix_world @ Vector(corner)).z for corner in floor.bound_box)
    paint = bpy.data.materials.new('e_bike_temporary_lane_paint')
    paint.diffuse_color = (0.82, 0.82, 0.82, 1)
    paint.use_nodes = True
    shader = paint.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (0.82, 0.82, 0.82, 1)
    shader.inputs['Roughness'].default_value = 0.8
    for x in (-.55, .55):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(x, 0, top + .0015))
        stripe = bpy.context.object
        stripe.name = 'e_bike_temporary_lane_edge'
        stripe.dimensions = (.018, 3.8, .001)
        stripe.data.materials.append(paint)
    out.mkdir(parents=True)
    rows = []
    for index in range(view_start, view_start + count):
        if index < 8:
            x = (-.22, .0, .22, -.16, .16, -.25, .25, .0)[index]
            y = (-1.55, -1.45, -1.35, -1.20, -1.10, -.95, -.85, -.75)[index]
            target_x = x * .25
            composition = 'centered'
        elif index < 24:
            i = index - 8
            x = (-.38, -.18, .12, .36)[i % 4]
            y = (-1.62, -1.34, -1.06, -.82)[i // 4]
            offset = (-.36, 0.0, .36)[i % 3]
            target_x = x * .25 + offset
            composition = ('left', 'centered', 'right')[i % 3]
        else:
            rng = random.Random(20261113 + index)
            group = (index - 24) % 4
            x = rng.uniform(-.52, .52)
            y = rng.uniform(-1.65, -.72)
            composition = ('straight', 'turn', 'oblique', 'barrier_view')[group]
            offset = (rng.uniform(-.12, .12), rng.choice((-1, 1)) * rng.uniform(.42, .62),
                      rng.choice((-1, 1)) * rng.uniform(.23, .40),
                      rng.choice((-1, 1)) * rng.uniform(.58, .85))[group]
            target_x = x * .25 + offset
        camera.location = (x, y, top + .18)
        target = Vector((target_x, y + 1.65, top + .10))
        camera.rotation_euler = (target - camera.location).to_track_quat('-Z', 'Y').to_euler()
        scene.view_settings.exposure = 0.0
        name = f'background_{index:03d}.png'
        scene.render.filepath = str(out / name)
        bpy.ops.render.render(write_still=True)
        rows.append({'image': relative(out / name), 'source_scene': relative(MASTER),
                     'generation_method': 'blender_render', 'camera_location_m': list(camera.location),
                     'camera_target_m': list(target), 'camera_lens_mm': camera.data.lens,
                     'camera_height_above_floor_m': .18, 'floor_top_m': float(top),
                     'view_group': f'background_{index:03d}', 'road_composition': composition,
                     'random_seed': 20261113 if index >= 24 else None,
                     'road_edges': {'x_m': [-.55, .55], 'width_m': .018},
                     'resolution': [640, 480], 'samples': samples, 'device': 'OPTIX',
                     'review_status': 'generated_candidate_visual_review_pending'})
    value = {'status': 'candidate_not_measured', 'source_scene': relative(MASTER),
             'generation_method': 'blender_render', 'scene_changes': 'two fitted road-edge strips in memory',
             'scope': 'One user arena master with fitted cameras; not new physical capture.', 'records': rows}
    (out / 'manifest.json').write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': relative(out), 'rendered': len(rows)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
