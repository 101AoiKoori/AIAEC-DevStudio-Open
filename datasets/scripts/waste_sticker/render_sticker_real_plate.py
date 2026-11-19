"""Render detached, physically grounded bins and separate light transport passes."""
import argparse
import json
import math
from pathlib import Path
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.blender_photo import (add_receiver, aim_area, project, ground_ray, save_render, footprint_inside)
from datasets.scripts.common.blender_masks import GEOMETRY_TYPES, configure_device, masks, targets_inside_frame
from datasets.scripts.common.blender_workspace import STICKERS
from datasets.scripts.common.frame_environment import image_digest, query_frames
from datasets.scripts.waste_sticker.configure_blender_orbit import STICKER_SCENE, sticker_bin_objects, sticker_objects

PALETTE = {'blue': (1, 0, 0), 'red': (0, 1, 0), 'green': (0, 0, 1), 'gray': (1, 1, 0)}
ANIMATION = 'lab/features/waste_bin/training/config/waste_sticker_source_plan_20261114.json'
DEFAULT_JOBS = [
    # Both target footprints use visually unobstructed sections of the source floor.
    (2, 121, {'green': (468, 460)}, 147, 70, 750, -7),
    (0, 102, {'blue': (231, 330), 'red': (389, 331)}, 83, -105, 510, 5),
    (2, 67, {'gray': (270, 460), 'green': (474, 441), 'red': (678, 459)}, 141, 70, 750, -6),
    (0, 202, {'blue': (97, 333), 'red': (244, 332), 'green': (388, 331), 'gray': (536, 330)}, 83, -105, 510, 2),
]


def add_camera(scene):
    camera = bpy.data.objects.new('Plate fitted perspective camera', bpy.data.cameras.new('Plate fitted perspective'))
    scene.collection.objects.link(camera)
    camera.data.sensor_fit = 'HORIZONTAL'
    camera.data.sensor_width = 36
    camera.data.clip_start = .01
    scene.camera = camera
    return camera


def match_camera(scene, camera, resolution, center, target_height, horizon, focal):
    width, height = resolution
    scene.render.resolution_x, scene.render.resolution_y = resolution
    camera.data.lens = focal * camera.data.sensor_width / width
    depression = math.atan2(height / 2 - horizon, focal)
    camera.rotation_euler = (math.pi / 2 - depression, 0, 0)
    low, high = .9, 7.
    for _ in range(34):
        camera.location = (0, 0, (low + high) * .5)
        bpy.context.view_layer.update()
        ground = ground_ray(camera, center, resolution, focal)
        top = project(scene, camera, ground + Vector((0, 0, 1.19)), resolution)
        if center[1] - top[1] > target_height:
            low = camera.location.z
        else:
            high = camera.location.z
    return {'camera_location': list(camera.location), 'horizon_px': horizon, 'focal_px': focal,
            'downwards_angle_deg': math.degrees(depression), 'target_bin_image_height_px': target_height,
            'pose_basis': 'approximate source markings and planned apparent model height; not camera calibration'}


def move_bin(roots, body, target, yaw):
    evaluated = body.evaluated_get(bpy.context.evaluated_depsgraph_get())
    points = [evaluated.matrix_world @ Vector(p) for p in evaluated.bound_box]
    min_vals = [min(v[i] for v in points) for i in range(3)]
    max_vals = [max(v[i] for v in points) for i in range(3)]
    origin = Vector(((min_vals[0] + max_vals[0]) / 2, (min_vals[1] + max_vals[1]) / 2, min_vals[2]))
    matrix = Matrix.Translation(target) @ Matrix.Rotation(math.radians(yaw), 4, 'Z') @ Matrix.Translation(-origin)
    for obj in roots:
        obj.matrix_world = matrix @ obj.matrix_world
    return [matrix @ Vector((x, y, min_vals[2])) for x in (min_vals[0], max_vals[0]) for y in (min_vals[1], max_vals[1])]


def candidate_layouts(feet, height, polygon):
    yield feet, height
    if polygon and len(feet) == 1:
        color = next(iter(feet))
        xs, ys = zip(*polygon)
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        for factor in (1, .91, .82, .73, .64, .55, .50, .45):
            for dy in (0, -1.5, 1.5):
                yield {color: (cx, cy + dy)}, round(height * factor)


def main(args):
    out = args.out.resolve()
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        for label in ('foreground', 'masks', 'lighting'):
            (out / label).mkdir(exist_ok=True)
    scene = bpy.data.scenes[STICKER_SCENE]
    bpy.context.window.scene = scene
    bpy.context.window.view_layer = scene.view_layers[0]
    array = bpy.context.view_layer.layer_collection.children.get(STICKERS)
    if array is not None:
        array.exclude = array.collection.hide_render = False
        for layer in array.children:
            layer.exclude = layer.collection.hide_render = False
    groups = sticker_bin_objects(scene.objects)
    papers = sticker_objects(scene)
    roots = {key: [obj for obj in group if obj.parent not in group] for key, group in groups.items()}
    bins_base = {obj: obj.matrix_world.copy() for group in roots.values() for obj in group}
    original_hide = {obj: obj.hide_render for group in groups.values() for obj in group}

    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.use_motion_blur = False
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1
    scene.render.film_transparent = True
    scene.render.engine = 'CYCLES'
    try:
        print('DEVICE', configure_device(scene, 'OPTIX'), flush=True)
    except (RuntimeError, TypeError) as exc:
        print('OPTIX FALLBACK:', exc, flush=True)
        configure_device(scene, 'CPU')
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 12
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    scene.cycles.samples = args.samples
    scene.cycles.seed = 26
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 2
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    active = set(original_hide)
    for obj in scene.objects:
        if obj.type in GEOMETRY_TYPES and obj not in active:
            obj.hide_render = True
    for marker in scene.timeline_markers:
        marker.camera = None
    camera = add_camera(scene)
    receiver = add_receiver()
    background = scene.world.node_tree.nodes.get('Background')
    background.inputs['Color'].default_value = (.55, .56, .59, 1)
    background.inputs['Strength'].default_value = .56
    key = scene.objects['Waste Stickers | Key']
    fill = scene.objects['Waste Stickers | Fill']

    declaration = json.loads(args.plan.read_text(encoding='utf-8')) if args.plan else {}
    if args.plan:
        if (declaration.get('model_key') != 'waste_sticker' or declaration.get('target_split') not in ('train', 'test') or
                declaration.get('budget') != len(declaration.get('jobs', []))):
            raise ValueError('Invalid assigned whole-bin plan')
        plate_records = declaration['plates']
        jobs = declaration['jobs']
        assigned = {row['record_id']: row for row in query_frames(
            task='waste_sticker', role='composite_background', split=declaration['target_split'])}
        for plate in plate_records:
            stored = assigned.get(plate['record_id'])
            if (not stored or stored['label']['presence'] != 'absent' or
                    stored['label']['completeness'] != 'complete' or
                    stored['geometry']['ground_polygon'] != plate['geometry']['ground_polygon'] or
                    stored['parent_group'] != plate['parent'] or stored['image_size'] != plate['size'] or
                    stored['frame_id'] != plate['frame_id'] or plate['source_photo'] not in stored['paths'] or
                    image_digest(ROOT / plate['source_photo']) != stored['content_sha256']):
                raise ValueError(f'Plate is inconsistent with the current assigned proof: {plate["source_photo"]}')
    else:
        plate_records = json.loads(args.plates.read_text(encoding='utf-8'))
        jobs = json.loads(args.jobs.read_text(encoding='utf-8')) if args.jobs else DEFAULT_JOBS
    animation = json.loads((ROOT / ANIMATION).read_text(encoding='utf-8'))
    schedule = {r['frame']: {item['bin_color']: item for item in r['bins']} for r in animation['samples']}
    records = []
    for i, (plate, frame, feet, height, horizon, focal, yaw) in enumerate(jobs):
        if args.single is not None and args.single != i:
            continue
        source = plate_records[plate]
        w, h = source['size']
        resolution = (w, h) if w <= 960 else (960, round(h * 960 / w))
        job_details = declaration.get('job_details', [{} for _ in jobs])[i]
        stem = job_details.get('id', f'blender_plate_{i + 1:02d}')
        scene.frame_set(frame)
        overrides = job_details.get('class_ids', {})
        for color, class_id in overrides.items():
            if color not in feet or type(class_id) is not int or not 0 <= class_id < 24:
                raise ValueError('Invalid declared artwork override')
            # render.render updates the dependency graph again. Clear only the
            # mesh's material-index animation in this temporary render process
            # so that it cannot replace the declared class during render.
            papers[color].data.animation_data_clear()
            for face in papers[color].data.polygons:
                face.material_index = class_id
            papers[color].data.update()
        bpy.context.view_layer.update()
        scene.camera = camera
        main = [obj for color in feet for obj in groups[color] if obj.type in GEOMETRY_TYPES and not original_hide[obj]]
        for color, objects in groups.items():
            for obj in objects:
                obj.hide_render = color not in feet or original_hide[obj]
                if obj.type in GEOMETRY_TYPES:
                    obj.visible_camera = True
        polygon = ([[x * resolution[0] / w, y * resolution[1] / h]
                    for x, y in source['geometry']['ground_polygon']] if args.plan else None)
        for positions, apparent_height in candidate_layouts(feet, height, polygon):
            for obj, matrix in bins_base.items():
                obj.matrix_world = matrix
            bpy.context.view_layer.update()
            middle = tuple(sum(p[j] for p in positions.values()) / len(positions) for j in (0, 1))
            fit = match_camera(scene, camera, resolution, middle, apparent_height, horizon, focal)
            projected_feet = {}
            for color, foot in positions.items():
                body = papers[color].parent
                point = ground_ray(camera, foot, resolution, focal)
                corners = move_bin(roots[color], body, point, yaw)
                projected_feet[color] = {'center': list(foot), 'footprint': [project(scene, camera, c, resolution) for c in corners]}
            bpy.context.view_layer.update()
            if ((polygon is None or all(footprint_inside(info['footprint'], polygon) for info in projected_feet.values()))
                    and targets_inside_frame(scene, camera, main)):
                break
        else:
            raise ValueError(f'{stem}: projected footprint {projected_feet} falls outside ground {polygon}')
        if args.dry_run:
            print('PLACEMENT_OK', stem, source['frame_id'], list(feet), apparent_height, flush=True)
            continue
        aim = ground_ray(camera, middle, resolution, focal)
        aim_area(key, aim + Vector((-3.2, -2.8, 5.6)), aim, 840 if plate == 2 else 690, 4.0)
        aim_area(fill, aim + Vector((3.0, -.6, 4.4)), aim, 155, 5.0)
        bpy.context.view_layer.update()
        if not targets_inside_frame(scene, camera, main):
            bad = []
            for o in main:
                coords = [world_to_camera_view(scene, camera, o.matrix_world @ Vector(v)) for v in o.bound_box]
                if any(not (0 <= v.x <= 1 and 0 <= v.y <= 1 and v.z > 0) for v in coords):
                    bad.append((o.name, [(round(min(v[j] for v in coords),3),round(max(v[j] for v in coords),3)) for j in range(3)]))
            raise ValueError(f'{stem}: bin is outside the image {(fit, projected_feet, bad)}')
        ids = {}
        deps = bpy.context.evaluated_depsgraph_get()
        for color in feet:
            evaluated = papers[color].evaluated_get(deps)
            index = evaluated.data.polygons[0].material_index
            item = dict(schedule[frame][color])
            if color in overrides:
                class_id = overrides[color]
                artwork = next(row for sample in animation['samples'] for row in sample['bins']
                               if row['class_id'] == class_id)
                item.update(class_id=class_id, source_artwork=artwork['source_artwork'])
            if index != item['class_id'] or not evaluated.data.materials[index].name.startswith(f'waste_sticker_{index:02d}_'):
                raise ValueError(f'{stem} {color}: sticker does not match animation schedule')
            ids[color] = item

        receiver.visible_camera = False
        foreground = 'foreground/' + stem + '.png'
        save_render(scene, out / foreground)
        for color, expected in ids.items():
            actual_index = papers[color].evaluated_get(bpy.context.evaluated_depsgraph_get()).data.polygons[0].material_index
            if actual_index != expected['class_id']:
                raise ValueError(f'{stem}: render changed the declared artwork class for {color}')

        receiver.visible_camera = True
        for obj in main:
            obj.visible_camera = False
        shadowed = 'lighting/' + stem + '_shadowed.png'
        save_render(scene, out / shadowed)
        for obj in main:
            obj.hide_render = True
        lit = 'lighting/' + stem + '_lit.png'
        save_render(scene, out / lit)
        for obj in main:
            obj.hide_render = False
            obj.visible_camera = True
        receiver.hide_render = True
        mask, instances = masks(scene, camera, {c: [papers[c]] for c in feet}, out, stem, 'paper', palette=PALETTE)
        w, h = resolution
        for instance in instances:
            color = instance['bin']
            if not instance['label_eligible']:
                raise ValueError(f'{stem} {color}: insufficient mask visibility {instance}')
            instance['bbox_xyxy'] = [round(n * size) for n, size in zip(instance['candidate_xyxy_normalized'], (w, h, w, h))]
            instance.update(class_id=ids[color]['class_id'], source_artwork=ids[color]['source_artwork'])
        receiver.hide_render = False
        record = {
            'id': stem, 'source_photo': source['source_photo'], 'frame_id': source['frame_id'],
            'source_size': source['size'], 'image_size': list(resolution), 'geometry': source['geometry']['ground_polygon'],
            'parent_group': source['parent'], 'frame': frame, 'selected_bins': list(feet),
            'camera': fit, 'bin_yaw_deg': yaw, 'projected_feet': projected_feet, 'foreground': foreground,
            'visible_mask': mask, 'shadowed_receiver': shadowed, 'lit_receiver': lit, 'instances': instances,
        }
        if args.plan:
            record.update(source_decision=source['record_id'], source_evidence=source['evidence_ids'],
                          split=declaration['target_split'])
        records.append(record)
        manifest = {'scene': STICKER_SCENE, 'palette': PALETTE,
            'blend': Path(bpy.data.filepath).relative_to(ROOT).as_posix(), 'blender': bpy.app.version_string,
            'samples': args.samples, 'records': records}
        if args.plan:
            manifest['target_split'] = declaration['target_split']
        (out / 'render_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        print('RENDERED', stem, resolution, frame, [(p['bin'], p['bbox_xyxy'], p['class_id']) for p in instances], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--plates', type=Path)
    source.add_argument('--plan', type=Path, help='compiled Test whole-bin plan including source evidence')
    parser.add_argument('--jobs', type=Path, help='optional explicit plate index, frame, feet, height, horizon, focal and yaw declarations')
    parser.add_argument('--samples', type=int, default=12)
    parser.add_argument('--single', type=int)
    parser.add_argument('--dry-run', action='store_true', help='verify footprints without rendering')
    main(parser.parse_args(sys.argv[sys.argv.index('--') + 1:]))
