"""Render flame-free C-building backgrounds and record visible placement quads."""

import argparse
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.blender_masks import configure_device
from datasets.scripts.common.blender_workspace import BUILDINGS, WORKSPACE_BLEND, activate_scene
from datasets.scripts.fire.fire_blender_labels import front_facade_slots
from datasets.scripts.fire.render_blender_fire_dataset import get_source, verify_c_supported_wall
from lab.training_common.production_guard import generation_guard

SIZE = (640, 480)
BATCH = 'aiaec_fire_c_facade_20261117'
ARTWORK = ROOT / 'Archive/Assets/working_sources/digital/fire/team_assets/image1.png'


def set_view(camera, center_x, yaw, distance):
    angle = math.radians(yaw)
    aim = Vector((center_x, 0, .216))
    camera.location = aim + Vector((distance * math.sin(angle), -distance * math.cos(angle), .10))
    camera.rotation_euler = (aim - camera.location).to_track_quat('-Z', 'Y').to_euler()
    bpy.context.view_layer.update()


def placement_quads(center_x, front_y, positions):
    for placement in positions:
        for index, (x, z) in enumerate(front_facade_slots(placement), 1):
            quad = [Vector((center_x + x + dx, front_y, z + dz)) for dx, dz in
                    ((-.0125, .0175), (.0125, .0175), (.0125, -.0175), (-.0125, -.0175))]
            yield f'{placement}_{index:02d}', quad


def verify_projection(scene, camera, wall, slots):
    evaluated = bpy.context.evaluated_depsgraph_get()
    origin = camera.matrix_world.translation
    regions = []
    for slot, quad in slots:
        points = [world_to_camera_view(scene, camera, point) for point in quad]
        if any(point.z <= 0 or not (.01 < point.x < .99 and .01 < point.y < .99) for point in points):
            raise ValueError(f'{slot} leaves the camera frame')
        for i, j in product(range(7), repeat=2):
            p = quad[0].lerp(quad[1], i / 6).lerp(quad[3].lerp(quad[2], i / 6), j / 6)
            ray = p - origin
            hit, where, _, _, obj, _ = scene.ray_cast(evaluated, origin, ray.normalized(), distance=ray.length + .004)
            if not hit or obj != wall or (where - p).length > .003:
                raise ValueError(f'{slot} does not project onto an unobstructed C facade')
        regions.append({'slot': slot, 'support_object': wall.name, 'quad_world_m': [list(p) for p in quad],
                        'quad_px': [[round(p.x * SIZE[0], 4), round((1 - p.y) * SIZE[1], 4)] for p in points],
                        'visibility': {'method': 'camera_to_facade_ray_grid', 'passed_rays': 49}})
    return regions


def prepare():
    scene = bpy.context.scene
    groups = get_source()
    support = verify_c_supported_wall(groups)
    hidden = []
    for key, group in groups.items():
        if key != 'C':
            group['collection'].hide_render = True
            group['collection'].hide_viewport = True
        for obj in (group['sign'], group['text'], *group['originals']):
            obj.hide_render = True
            if obj.name in bpy.context.view_layer.objects:
                obj.hide_set(True)
            hidden.append(obj.name)
    scene.render.resolution_x, scene.render.resolution_y = SIZE
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = False
    scene.camera.data.lens = 48
    return groups['C'], support, hidden


def render(output, manifest, samples):
    scene = bpy.context.scene
    camera = scene.camera
    group, support, hidden = prepare()
    wall = scene.objects[support['support_object']]
    slots = list(placement_quads(group['x'], support['front_plane_y_m'], support['positions']))
    if len(slots) != 10:
        raise ValueError('expected the ten bounded C facade slots')
    records = []
    digest = sha256(WORKSPACE_BLEND.read_bytes()).hexdigest()
    blend = WORKSPACE_BLEND.relative_to(ROOT).as_posix()
    views = list(product((-20, -12, -4, 4, 12, 20), (.86, .94, 1.02, 1.10)))
    for index, (yaw, distance) in enumerate(views, 1):
        set_view(camera, group['x'], yaw, distance)
        records.append({'id': f'c_facade_{index:02d}', 'image_size': list(SIZE),
                        'source_blend': blend, 'source_blend_sha256': digest, 'parent_group': blend,
                        'source_kind': 'blender_render', 'scene': scene.name, 'building_collection': group['collection'].name,
                        'camera': camera.name, 'camera_lens_mm': camera.data.lens,
                        'camera_matrix_world': [list(row) for row in camera.matrix_world],
                        'view': {'yaw_deg': yaw, 'distance_m': distance},
                        'fire_target_presence': 'absent', 'placement_surfaces': verify_projection(scene, camera, wall, slots)})

    scene.render.engine = 'CYCLES'
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    try:
        devices = configure_device(scene, 'OPTIX')
    except (RuntimeError, TypeError):
        devices = configure_device(scene, 'CPU')
    output.mkdir(parents=True, exist_ok=True)
    for record in records:
        set_view(camera, group['x'], record['view']['yaw_deg'], record['view']['distance_m'])
        image = output / f"{record['id']}.png"
        scene.render.filepath = str(image)
        bpy.ops.render.render(write_still=True)
        record['image'] = image.relative_to(ROOT).as_posix()
        record['content_sha256'] = sha256(image.read_bytes()).hexdigest()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({'schema_version': 1, 'task': 'fire', 'purpose': 'flame_free_facade_background',
                                    'status': 'complete_candidates', 'image_count': len(records),
                                    'split': 'train', 'coordinate_order': 'top_left_top_right_bottom_right_bottom_left',
                                    'blender_version': bpy.app.version_string, 'samples': samples, 'devices': devices,
                                    'temporarily_hidden_papers_and_text': hidden, 'wall_support': support,
                                    'records': records}, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print('FIRE_FACADE_BACKGROUNDS', len(records), manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, choices=range(1, 33), default=8)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    activate_scene(BUILDINGS)
    output = ROOT / 'datasets/raw/shared/backgrounds/rendered' / BATCH
    manifest = ROOT / 'datasets/annotated/common/sources' / BATCH / 'manifest.json'
    inputs = [WORKSPACE_BLEND, ROOT / 'datasets/scripts/fire/render_blender_fire_dataset.py',
              ROOT / 'datasets/scripts/fire/fire_blender_labels.py']
    code = [Path(__file__).resolve(), ROOT / 'datasets/scripts/common/blender_workspace.py',
            ROOT / 'datasets/scripts/common/blender_masks.py']
    with generation_guard('blender.fire_facade_backgrounds:' + BATCH, [ARTWORK], [output, manifest],
                          {'samples': args.samples, 'source_scene': BUILDINGS}, code_paths=code,
                          procedural_inputs=inputs, root=ROOT) as receipt:
        if not receipt.reused:
            render(output, manifest, args.samples)


if __name__ == '__main__':
    main()
