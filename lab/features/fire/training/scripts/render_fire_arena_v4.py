"""Render fire candidates in an isolated fitted arena process.

Open the user-confirmed arena with Blender --background and execute this file.
The source scene is never saved. Name-paper supervision remains unknown and no
name labels are exported. Fire labels remain candidates until reviewed.
"""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/fire/.")

import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_blender_fire_dataset as fire
from lab.training_common.production_guard import generation_guard
from fire_blender_labels import (fire_label_suppression, fire_scale_suppression,
                                 label_state, ocr_status, visible_bbox, yolo_row)

MASTER = ROOT / 'datasets/assets/blender/shared/arena_4m2_pbr_20261112/arena_master_user_20261113.blend'
BUILDINGS = ROOT / 'datasets/assets/blender/fire/blender_buildings_v13/three_buildings.blend'
FONT = ROOT / 'datasets/raw/building_name_paper/targets/digital/typeface/NotoSansSC.ttf'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--count', type=int, default=12)
    parser.add_argument('--seed', type=int, default=20261113)
    parser.add_argument('--samples', type=int, default=4)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to((ROOT / 'datasets/exports/fire').resolve()):
        raise ValueError('Use a new task-owned fire export directory')
    if Path(bpy.data.filepath).resolve() != MASTER.resolve() or not BUILDINGS.is_file() or not FONT.is_file():
        raise ValueError('Open the confirmed arena master in an isolated background process')
    if not 1 <= args.count <= 10000 or not 1 <= args.samples <= 256:
        raise ValueError('Invalid bounded count or render samples')
    return args, output


def _produce(args, output, source_refs):
    scene = bpy.context.scene
    camera = bpy.data.objects.get('Vehicle camera')
    if camera is None or camera.type != 'CAMERA' or camera.name not in scene.objects:
        raise ValueError('Vehicle camera missing from arena scene')
    if scene.world is None or not scene.world.use_nodes or scene.world.node_tree.nodes.get('Background') is None:
        raise ValueError('Arena requires a node-based world for the instance pass')
    scene.camera = camera
    # Probe below the fitted camera, not above the room's ceiling. The e-bike
    # arena pilot records the fitted ground mesh top near z=-0.021 m; reject
    # horizontal hits far from that level and retain the actual support object.
    bpy.context.view_layer.update()
    direction = camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
    direction.z = 0
    if direction.length < 1e-5:
        raise ValueError('Camera has no horizontal viewing direction')
    direction.normalize()
    right = Vector((direction.y, -direction.x, 0))
    camera_pos = camera.matrix_world.translation
    candidates = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for distance in (0.75, 0.90, 1.05):
        for lateral in (-0.18, 0.0, 0.18):
            xy = camera_pos + direction * distance + right * lateral
            hit, location, normal, _, obj, _ = scene.ray_cast(
                depsgraph, Vector((xy.x, xy.y, camera_pos.z)), Vector((0, 0, -1)), distance=3)
            if hit and normal.z >= 0.90 and abs(location.z - (-0.021)) <= 0.03:
                candidates.append({'location': list(location), 'distance_m': distance,
                                   'lateral_m': lateral, 'support_object': obj.name,
                                   'support_detection': 'downward ray from camera height, fitted ground z tolerance',
                                   'ground_reference_z_m': -0.021, 'ground_tolerance_m': 0.03})
    if not candidates:
        raise ValueError('No candidate horizontal support found in front of camera')
    if any(bpy.data.collections.get(name) for name in fire.COLLECTIONS.values()):
        raise ValueError('Arena already contains building collection names; refusing ambiguous import')
    with bpy.data.libraries.load(str(BUILDINGS), link=False) as (source, target):
        if any(name not in source.collections for name in fire.COLLECTIONS.values()):
            raise ValueError('Building library is missing expected collections')
        target.collections = list(fire.COLLECTIONS.values())
    for collection in target.collections:
        scene.collection.children.link(collection)
    groups = fire.get_source()
    fire.create_stickers(groups)
    original_font = bpy.data.fonts.load(str(FONT), check_existing=True)
    for key, group in groups.items():
        text = group['text']
        text.hide_render = False
        text.data.font = original_font
        # The source uses a six-micrometre shrinkwrap offset; preserve that
        # modifier and its target, but separate ink from the paper surface.
        wraps = [m for m in text.modifiers if m.type == 'SHRINKWRAP' and m.target == group['sign']]
        if len(wraps) != 1:
            raise ValueError('Expected the original name-paper shrinkwrap modifier')
        wraps[0].offset = 0.0003
        group['ink_wrap'] = wraps[0]
        parent = bpy.data.objects.new('Fire arena transform ' + key, None)
        scene.collection.objects.link(parent)
        for obj in list(group['collection'].objects):
            matrix = obj.matrix_world.copy()
            obj.parent = parent
            obj.matrix_world = matrix
        group['arena_parent'] = parent
    scene.render.resolution_x, scene.render.resolution_y = 640, 480
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '8'
    scene.render.dither_intensity = 0
    scene.render.film_transparent = False
    fire.configure_device(scene, 'CPU')
    tone = fire.vehicle_tone_compositor(scene)
    mask = fire.material_override()
    output.mkdir(parents=True)
    for folder in ('images/unassigned', 'labels/fire/unassigned', 'instance_pass'):
        (output / folder).mkdir(parents=True)
    metadata = {'status': 'running', 'requested': args.count, 'completed': 0, 'seed': args.seed,
                'scene': MASTER.relative_to(ROOT).as_posix(), 'building_asset': BUILDINGS.relative_to(ROOT).as_posix(),
                'camera': camera.name, 'image_size': [640, 480], 'cycles_samples': args.samples,
                'source_and_scale': 'Fitted scene and camera; physical scale and placement are not measured.',
                'support_candidates': candidates,
                'name_supervision': 'unknown: prior pilot showed blank text; no name labels exported',
                'distractors': 'red lights, red bins and vehicles not included in this generator',
                'scope': 'Unassigned synthetic candidates; no independent capture claim.',
                'input_source_refs': source_refs}
    fire.write_run(output / 'run.json', metadata)
    try:
        with (output / 'manifest.jsonl').open('w', encoding='utf-8') as manifest:
            for index in range(args.count):
                # Both tone variants use the same task geometry and name.
                rng = random.Random(args.seed + index // 2)
                key = ('A', 'B', 'C')[(index // 2) % 3]
                count = (index // 6) % 5
                group = groups[key]
                for k, g in groups.items():
                    g['collection'].hide_render = k != key
                name = fire.make_name(rng)
                fire.reset_name(group['text'], name)
                fire.place_stickers(group, count, rng)
                position = candidates[(index // 2) % len(candidates)]
                yaw = math.atan2(-direction.x, direction.y)
                group['arena_parent'].matrix_world = (Matrix.Translation(Vector(position['location'])) @
                    Matrix.Rotation(yaw, 4, 'Z') @ Matrix.Translation(Vector((-group['x'], 0, 0))))
                target_point = Vector(position['location']) + Vector((0, 0, 0.20))
                camera.rotation_euler = (target_point - camera.matrix_world.translation).to_track_quat('-Z', 'Y').to_euler()
                camera.data.lens = 35.0
                bpy.context.view_layer.update()
                occluded = count > 0 and (index // 2) % 7 == 6
                occluder = fire.set_blocker(scene, groups, [key], 'occluder' if occluded else 'normal')
                exposure, saturation = ((0.0, 1.0) if index % 2 == 0 else (-0.5, 0.65))
                scene.view_settings.exposure = exposure
                tone.inputs['Saturation'].default_value = saturation
                targets = [{'id': 1, 'kind': 'name_sign', 'building': key, 'text': name, 'object': group['sign']}]
                targets += [{'id': j + 2, 'kind': 'fire_sticker', 'building': key, 'object': obj}
                            for j, obj in enumerate(group['pool'][:count])]
                bpy.context.view_layer.update()
                for target in targets:
                    target['projection'] = fire.projected_extent(scene, target['object'])
                stem = f'fire_arena_{index:06d}'
                image = f'images/unassigned/{stem}.png'
                scene.cycles.samples = args.samples
                scene.cycles.use_denoising = True
                scene.cycles.seed = args.seed + index
                scene.render.filepath = str(output / image)
                bpy.ops.render.render(write_still=True)
                pass_path = f'instance_pass/{stem}.png'
                ids = fire.render_id_pass(scene, targets, output / pass_path, mask)
                instances, labels = [], {'fire': [], 'names': []}
                unknown = {'fire': False, 'names': False}
                for target in targets:
                    box, area = visible_bbox(ids == target['id'])
                    state = label_state(target['projection'], box, area)
                    item = {'id': target['id'], 'kind': target['kind'], 'building': key,
                            'pixel_bbox_xyxy': box, 'visible_pixels': area, 'state': state,
                            'projected_extent_normalized': target['projection']['xyxy'],
                            'truncated': target['projection']['truncated'], 'occlusion': 'not quantified'}
                    label_key = 'names' if target['kind'] == 'name_sign' else 'fire'
                    if label_key == 'names':
                        item.update(text=name, text_role='generated_text_not_ocr_acceptance',
                                    ocr_status=ocr_status(state, occluder))
                    if state in ('visible', 'visible_truncated'):
                        labels[label_key].append(yolo_row(box, 640, 480))
                    elif state != 'outside_frame':
                        unknown[label_key] = True
                    instances.append(item)
                suppression = fire_label_suppression([o['state'] for o in instances if o['kind'] == 'fire_sticker'], occluder)
                suppression = suppression or fire_scale_suppression(instances)
                if unknown['names']:
                    unknown['fire'] = True
                paths = {}
                for task in ('fire', 'names'):
                    path = None
                    if task == 'fire' and not unknown[task] and suppression is None:
                        path = f'labels/{task}/unassigned/{stem}.txt'
                        (output / path).write_text('\n'.join(labels[task]) + ('\n' if labels[task] else ''), encoding='utf-8')
                    paths[task] = path
                row = {'image': image, 'instance_pass': pass_path, 'fire_label': paths['fire'],
                       'name_label': paths['names'], 'fire_label_suppression_reason': suppression,
                       'split': 'unassigned', 'status': 'candidate_requires_visual_review',
                       'generation_method': 'fitted_real_scene_render', 'building': key,
                       'requested_fire_count': count, 'placement': position, 'yaw_rad': yaw,
                       'camera': {'location_m': list(camera.matrix_world.translation),
                                  'rotation_euler_rad': list(camera.rotation_euler),
                                  'lens_mm': float(camera.data.lens), 'target_m': list(target_point),
                                  'calibration': 'fitted; not measured'},
                       'name_text': {'font': FONT.relative_to(ROOT).as_posix(),
                                     'shrinkwrap_offset_m': float(group['ink_wrap'].offset),
                                     'status': 'unknown: prior pilot showed blank text; no name label exported'},
                       'tone': {'exposure_ev': exposure, 'saturation': saturation, 'calibrated': False},
                       'occluder': occluder, 'instances': instances,
                       'source_assets': [metadata['scene'], metadata['building_asset']]}
                manifest.write(json.dumps(row, ensure_ascii=False) + '\n')
                manifest.flush()
                metadata['completed'] += 1
                fire.write_run(output / 'run.json', metadata)
        metadata['status'] = 'complete_candidates'
    except Exception as exc:
        metadata['status'] = 'failed'
        metadata['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        fire.write_run(output / 'run.json', metadata)


def main():
    args, output = parse_args()
    procedural = {MASTER, BUILDINGS, FONT} | fire.scene_external_libraries()
    media = fire.scene_external_media()
    media.add(ROOT / 'Archive/Assets/working_sources/digital/fire/team_assets/image1.png')
    with generation_guard(
        'fire.render_fire_arena_v4:' + output.relative_to(ROOT).as_posix(),
        sorted(media | procedural), [output],
        {'count': args.count, 'seed': args.seed, 'samples': args.samples},
        code_paths=[Path(__file__), Path(fire.__file__),
                    ROOT / 'lab/features/fire/training/scripts/fire_blender_labels.py',
                    ROOT / 'datasets/scripts/common/annotations.py'],
        procedural_inputs=sorted(procedural),
    ) as run:
        if not run.reused:
            _produce(args, output, run.source_refs)


if __name__ == '__main__':
    main()
