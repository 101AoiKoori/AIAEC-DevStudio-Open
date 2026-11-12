"""Render source-grouped synthetic fire/name candidates from the v13 Blender scene.

Run from the workspace root using Blender 5.2, loading the source scene first::

    blender -b datasets/annotated/fire/blender_buildings_v13/three_buildings.blend \
      --python-exit-code 1 --python lab/features/fire/training/scripts/render_blender_fire_dataset.py -- \
      --output datasets/exports/fire/blender_candidates_new --count 10 --device CPU

The first ten cases deliberately cover zero, one, many, multiple buildings,
occlusion and truncation. Source groups are synthetic; no real validation split
is inferred. A material-override instance pass measures visible pixels. Images
with unresolved visible targets are not exported as empty YOLO labels.
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
import numpy as np

ROOT = Path.cwd().resolve()
SCRIPTS = ROOT / 'lab/features/fire/training/scripts'
sys.path.insert(0, str(SCRIPTS))
from fire_blender_labels import (decode_ids, fire_label_suppression, label_state,
                                 ocr_status, scenario, srgb_to_linear, visible_bbox, yolo_row)

COLLECTIONS = {'A': 'A - 开心大厦', 'B': 'B - 锦绣大厦', 'C': 'C - 晨光大厦'}
WORD_PARTS = ('青', '禾', '安', '和', '乐', '明', '新', '宏', '瑞', '锦', '泽', '云',
              '景', '晨', '华', '康', '福', '泰', '阳', '祥', '雅', '宁', '兴', '嘉')
NAME_SUFFIXES = ('大厦', '大楼', '中心', '公寓', '商厦')


def parse_args():
    parser = argparse.ArgumentParser(description='渲染楼名与火焰可见实例候选')
    parser.add_argument('--output', required=True)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=431)
    parser.add_argument('--resolution', type=int, default=640)
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--device', choices=('CPU', 'OPTIX', 'CUDA'), default='CPU')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    args.output = (ROOT / args.output).resolve()
    source = Path(bpy.data.filepath).resolve()
    if not source.is_relative_to(ROOT / 'datasets' / 'annotated' / 'fire') or not source.is_file():
        parser.error('必须先打开工作空间 fire 素材目录内的原始场景')
    if not args.output.is_relative_to(ROOT / 'datasets' / 'exports' / 'fire'):
        parser.error('输出必须位于 datasets/exports/fire')
    if args.output.exists():
        parser.error('输出目录已存在')
    if not 1 <= args.count <= 100000 or not 256 <= args.resolution <= 2048 or not 1 <= args.samples <= 256:
        parser.error('count、resolution 或 samples 超出范围')
    return args, source


def configure_device(scene, requested):
    scene.render.engine = 'CYCLES'
    if requested == 'CPU':
        scene.cycles.device = 'CPU'
        return ['CPU']
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = requested
    prefs.get_devices()
    chosen = [d for d in prefs.devices if d.type == requested]
    if not chosen:
        raise RuntimeError(f'没有可用的 {requested} 设备')
    for device in prefs.devices:
        device.use = device.type == requested
    scene.cycles.device = 'GPU'
    return [d.name for d in chosen]


def get_source():
    groups = {}
    for key, collection_name in COLLECTIONS.items():
        collection = bpy.data.collections.get(collection_name)
        if collection is None:
            raise ValueError(f'场景缺少楼体集合: {collection_name}')
        signs = [o for o in collection.objects if o.name.startswith('Name paper - ')]
        text = [o for o in collection.objects if o.type == 'FONT' and o.name.startswith('Editable text - ')]
        originals = [o for o in collection.objects if o.name.startswith('Fire paper ')]
        if len(signs) != 1 or len(text) != 1 or len(originals) != 3:
            raise ValueError(f'{collection_name} 的纸条、文字或火焰模板数量不匹配')
        groups[key] = {'collection': collection, 'sign': signs[0], 'text': text[0],
                       'originals': originals, 'x': float(text[0].location.x)}
    return groups


def make_name(rng):
    count = rng.choice((2, 2, 3))
    return ''.join(rng.choice(WORD_PARTS) for _ in range(count)) + rng.choice(NAME_SUFFIXES)


def reset_name(obj, value):
    obj.data.body = value
    obj.scale = (1, 1, 1)
    bpy.context.view_layer.update()
    if obj.dimensions.x > .115:
        obj.scale *= .115 / obj.dimensions.x


def create_stickers(groups):
    """Clone paper meshes, keeping their UVs, materials and shape unmodified."""
    for key, group in groups.items():
        pool = []
        for index in range(5):
            template = group['originals'][index % 3]
            copy = template.copy()
            copy.data = template.data.copy()
            copy.name = f'Candidate fire {key} {index + 1:02d}'
            center = sum((v.co for v in copy.data.vertices), Vector()) / len(copy.data.vertices)
            for vertex in copy.data.vertices:
                vertex.co.x -= center.x
                vertex.co.z -= center.z
            copy.location = (center.x, 0, center.z)
            group['collection'].objects.link(copy)
            pool.append(copy)
        for original in group['originals']:
            original.hide_render = True
        group['pool'] = pool
        group['paper_base_y'] = [[float(v.co.y) for v in obj.data.vertices] for obj in pool]


def place_stickers(group, count, rng):
    # Each footprint is separated from its neighbours by more than the paper
    # width or height, so anti-aliased instance masks do not blend targets.
    slots = [(x, z) for x in (-.065, -.025, .022, .061)
             for z in (.115, .165, .215, .265, .315)]
    selected = rng.sample(slots, count)
    for index, obj in enumerate(group['pool']):
        obj.hide_render = index >= count
        if index < count:
            dx, z = selected[index]
            obj.location.x = group['x'] + dx + rng.uniform(-.002, .002)
            obj.location.z = z + rng.uniform(-.002, .002)
            obj.rotation_euler.y = rng.uniform(-.09, .09)
            # Tiny independent ripple at the paper edge; preserve its centre,
            # width, height, UVs, measured template thickness and nominal shape.
            amplitude = rng.uniform(-.00003, .00003)
            baseline = group['paper_base_y'][index]
            for j, vertex in enumerate(obj.data.vertices):
                u, v = (j % 13) / 12.0, (j // 13) / 12.0
                edge = max(abs(2*u-1), abs(2*v-1))
                vertex.co.y = baseline[j] + amplitude * math.sin(3*math.pi*u) * math.sin(2*math.pi*v) * edge**2
            obj.data.update()
            obj['paper_ripple_m'] = amplitude


def set_camera(scene, groups, active, kind, rng):
    camera = scene.camera
    if camera is None:
        raise ValueError('场景没有相机')
    single = len(active) == 1
    center = groups[active[0]]['x'] if single else sum(groups[k]['x'] for k in active) / len(active)
    distance = (.78 if kind == 'truncated' else
                rng.uniform(.75, 1.10) if single else rng.uniform(1.25, 1.70))
    camera.location = (center + rng.uniform(-.16, .16), -distance, rng.uniform(.14, .34))
    look_x = center + (rng.choice((-.19, .19)) if kind == 'truncated' else rng.uniform(-.02, .02))
    look = Vector((look_x, 0, rng.uniform(.18, .23)))
    camera.rotation_euler = (look - camera.location).to_track_quat('-Z', 'Y').to_euler()
    camera.data.lens = 56 if kind == 'truncated' else rng.uniform(32, 44) if single else rng.uniform(28, 36)
    camera.data.clip_start = .001
    if hasattr(camera.data, 'dof'):
        camera.data.dof.use_dof = False
    return {'position_m': list(camera.location), 'target_m': list(look),
            'focal_length_mm': float(camera.data.lens), 'distortion': 'not modelled'}


def set_lighting(rng):
    values = {}
    for name, factor in (('Broad softbox', (0.75, 1.15)), ('Soft fill', (.7, 1.2))):
        light = bpy.data.objects.get(name)
        if light is None or light.type != 'LIGHT':
            raise ValueError(f'场景缺少灯光: {name}')
        if '_fire_baseline_energy' not in light:
            light['_fire_baseline_energy'] = light.data.energy
        light.data.energy = float(light['_fire_baseline_energy']) * rng.uniform(*factor)
        values[name] = float(light.data.energy)
    return values


def set_blocker(scene, groups, active, kind):
    blocker = bpy.data.objects.get('Synthetic occlusion prop')
    if blocker is None:
        bpy.ops.mesh.primitive_cube_add(size=1)
        blocker = bpy.context.object
        blocker.name = 'Synthetic occlusion prop'
        mat = bpy.data.materials.new('Synthetic neutral occluder')
        mat.use_nodes = True
        mat.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value = (.14, .15, .15, 1)
        blocker.data.materials.append(mat)
    blocker.hide_render = kind != 'occluder'
    if not blocker.hide_render:
        sticker = groups[active[0]]['pool'][0]
        center = sum((v.co for v in sticker.data.vertices), Vector()) / len(sticker.data.vertices)
        target = sticker.matrix_world @ center
        location = scene.camera.location.lerp(target, .65)
        location.x += .003
        blocker.location = location
        blocker.scale = (.014, .012, .09)
    return not blocker.hide_render


def projected_extent(scene, obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        if not mesh.vertices:
            return {'in_front': False, 'intersects': False, 'truncated': False, 'xyxy': None}
        points = [world_to_camera_view(scene, scene.camera, evaluated.matrix_world @ v.co) for v in mesh.vertices]
        in_front = all(p.z > 0 for p in points)
        xs = [float(p.x) for p in points]
        ys = [1.0 - float(p.y) for p in points]
        extent = [min(xs), min(ys), max(xs), max(ys)]
        intersects = extent[2] > 0 and extent[0] < 1 and extent[3] > 0 and extent[1] < 1
        truncated = min(extent[:2]) < 0 or max(extent[2:]) > 1
        return {'in_front': in_front, 'intersects': intersects, 'truncated': truncated, 'xyxy': extent}
    finally:
        evaluated.to_mesh_clear()


def material_override():
    material = bpy.data.materials.new('Fire dataset instance-ID override')
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    info = nodes.new('ShaderNodeObjectInfo')
    emission = nodes.new('ShaderNodeEmission')
    output = nodes.new('ShaderNodeOutputMaterial')
    material.node_tree.links.new(info.outputs['Color'], emission.inputs['Color'])
    material.node_tree.links.new(emission.outputs[0], output.inputs['Surface'])
    return material


def render_id_pass(scene, targets, filename, mask_material):
    """Render an object-color pass with full scene depth and no illumination."""
    layer = bpy.context.view_layer
    old_override = layer.material_override
    objects = [o for o in scene.objects if o.type in {'MESH', 'FONT', 'CURVE', 'SURFACE', 'META'}]
    colors = [(o, tuple(o.color)) for o in objects]
    bg = scene.world.node_tree.nodes.get('Background')
    old_world = tuple(bg.inputs['Color'].default_value)
    old_strength = float(bg.inputs['Strength'].default_value)
    old_view = scene.view_settings.view_transform
    old_exposure = scene.view_settings.exposure
    old_gamma = scene.view_settings.gamma
    old_samples = scene.cycles.samples
    old_denoise = scene.cycles.use_denoising
    try:
        for obj in objects:
            obj.color = (0, 0, 0, 1)
        for target in targets:
            target['object'].color = (1, target['id'] / 32.0, .25, 1)
        layer.material_override = mask_material
        bg.inputs['Color'].default_value = (0, 0, 0, 1)
        bg.inputs['Strength'].default_value = 0
        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.exposure = 0
        scene.view_settings.gamma = 1
        scene.cycles.samples = 1
        scene.cycles.use_denoising = False
        scene.render.filepath = str(filename)
        bpy.ops.render.render(write_still=True)
        image = bpy.data.images.load(str(filename), check_existing=False)
        try:
            width, height = image.size
            # Blender image.pixels exposes the stored PNG's sRGB-encoded
            # channel values, even though its color space is named sRGB. The
            # emitted object IDs were chosen in linear light and need decoding
            # before comparing ratios. Rows are bottom-up in Blender.
            pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape((height, width, 4))[::-1, :, :3]
            return decode_ids(srgb_to_linear(pixels), [t['id'] for t in targets])
        finally:
            bpy.data.images.remove(image)
    finally:
        for obj, color in colors:
            obj.color = color
        layer.material_override = old_override
        bg.inputs['Color'].default_value = old_world
        bg.inputs['Strength'].default_value = old_strength
        scene.view_settings.view_transform = old_view
        scene.view_settings.exposure = old_exposure
        scene.view_settings.gamma = old_gamma
        scene.cycles.samples = old_samples
        scene.cycles.use_denoising = old_denoise


def write_run(path, record):
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    args, source = parse_args()
    groups = get_source()
    scene = bpy.context.scene
    if scene.world is None or not scene.world.use_nodes or scene.world.node_tree.nodes.get('Background') is None:
        raise ValueError('场景没有可控的节点世界背景')
    devices = configure_device(scene, args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    for folder in ('images/train', 'labels/fire/train', 'labels/names/train', 'instance_pass'):
        (args.output / folder).mkdir(parents=True)
    metadata = {'status': 'running', 'requested': args.count, 'completed': 0,
                'source_scene': source.relative_to(ROOT).as_posix(), 'seed': args.seed,
                'device': args.device, 'actual_devices': devices,
                'resolution': args.resolution, 'cycles_samples': args.samples,
                'grouping': 'single synthetic model family; train candidates only; no independent split',
                'output_policy': 'unknown targets suppress class labels; occluded or truncated fire graphics suppress fire labels',
                'mask_encoding': 'RGB linear emission: R=1, G=instance_id/32, B=.25; black is background',
                'labels': 'candidate visible-pixel boxes; all output requires visual review'}
    write_run(args.output / 'run.json', metadata)
    try:
        floor = bpy.data.objects.get('Presentation floor')
        if floor is None or floor.type != 'MESH':
            raise ValueError('场景缺少可调整的 Presentation floor')
        floor.scale.x *= 10
        floor.scale.y *= 10
        metadata['floor_horizontal_scale_multiplier'] = 10
        write_run(args.output / 'run.json', metadata)
        create_stickers(groups)
        mask_material = material_override()
        rng = random.Random(args.seed)
        scene.render.resolution_x = scene.render.resolution_y = args.resolution
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGB'
        scene.render.image_settings.color_depth = '8'
        scene.render.dither_intensity = 0
        scene.render.film_transparent = False
    except Exception as exc:
        metadata['status'] = 'failed'
        metadata['error'] = f'{type(exc).__name__}: {exc}'
        write_run(args.output / 'run.json', metadata)
        raise
    try:
        with (args.output / 'manifest.jsonl').open('w', encoding='utf-8') as records:
            for index in range(args.count):
                plan = scenario(index)
                active = plan['active']
                counts = dict(zip(active, plan['counts']))
                names = {}
                targets = []
                for key, group in groups.items():
                    group['collection'].hide_render = key not in active
                    if key not in active:
                        continue
                    names[key] = make_name(rng)
                    reset_name(group['text'], names[key])
                    place_stickers(group, counts[key], rng)
                    targets.append({'id': len(targets) + 1, 'kind': 'name_sign', 'building': key,
                                    'text': names[key], 'object': group['sign']})
                    for sticker in group['pool'][:counts[key]]:
                        targets.append({'id': len(targets) + 1, 'kind': 'fire_sticker',
                                        'building': key, 'object': sticker})
                camera = set_camera(scene, groups, active, plan['kind'], rng)
                lights = set_lighting(rng)
                bpy.context.view_layer.update()
                occluder = set_blocker(scene, groups, active, plan['kind'])
                bpy.context.view_layer.update()
                for target in targets:
                    target['projection'] = projected_extent(scene, target['object'])
                stem = f'fire_{index + 1:06d}'
                scene.cycles.samples = args.samples
                scene.cycles.use_denoising = True
                scene.cycles.seed = args.seed + index
                scene.render.filepath = str(args.output / 'images' / 'train' / (stem + '.png'))
                bpy.ops.render.render(write_still=True)
                id_path = args.output / 'instance_pass' / (stem + '.png')
                ids = render_id_pass(scene, targets, id_path, mask_material)
                obj_records, fire_rows, name_rows = [], [], []
                unknown_fires = False
                unknown_names = False
                for target in targets:
                    bbox, area = visible_bbox(ids == target['id'])
                    state = label_state(target['projection'], bbox, area)
                    data = {'id': target['id'], 'kind': target['kind'], 'building': target['building'],
                            'pixel_bbox_xyxy': bbox, 'visible_pixels': area, 'state': state,
                            'projected_extent_normalized': target['projection']['xyxy'],
                            'truncated': target['projection']['truncated'], 'occlusion': 'not quantified'}
                    if target['kind'] == 'name_sign':
                        data['text'] = target['text']
                        data['text_role'] = 'generated_full_text_ground_truth_not_ocr_label'
                        data['ocr_status'] = ocr_status(state, occluder)
                        if state in ('visible', 'visible_truncated'):
                            name_rows.append(yolo_row(bbox, args.resolution, args.resolution))
                        elif state != 'outside_frame':
                            unknown_names = True
                    elif state in ('visible', 'visible_truncated'):
                        fire_rows.append(yolo_row(bbox, args.resolution, args.resolution))
                    elif state != 'outside_frame':
                        unknown_fires = True
                    if target['kind'] == 'fire_sticker':
                        data['paper_ripple_m'] = float(target['object'].get('paper_ripple_m', 0))
                        data['paper_rotation_rad'] = float(target['object'].rotation_euler.y)
                    obj_records.append(data)
                if not any(o['kind'] == 'name_sign' and o['visible_pixels'] >= 12 for o in obj_records):
                    raise RuntimeError('实例通道中未检测到可见楼名纸条，不能接受可能无效的标注通道')
                fire_label = None
                fire_suppression = fire_label_suppression(
                    [o['state'] for o in obj_records if o['kind'] == 'fire_sticker'], occluder)
                if not unknown_fires and fire_suppression is None:
                    fire_label = f'labels/fire/train/{stem}.txt'
                    (args.output / fire_label).write_text('\n'.join(fire_rows) + ('\n' if fire_rows else ''), encoding='utf-8')
                name_label = None
                if not unknown_names:
                    name_label = f'labels/names/train/{stem}.txt'
                    (args.output / name_label).write_text('\n'.join(name_rows) + ('\n' if name_rows else ''), encoding='utf-8')
                sample = {'image': f'images/train/{stem}.png', 'instance_pass': f'instance_pass/{stem}.png',
                          'fire_label': fire_label, 'name_label': name_label,
                          'fire_label_suppression_reason': fire_suppression,
                          'status': 'candidate_requires_visual_review', 'source_group': 'blender_v13_shared_model',
                          'split': 'train', 'scenario': plan['kind'], 'active_buildings': list(active),
                          'requested_fire_counts': counts, 'names': names, 'camera': camera,
                          'lighting': lights, 'occluder': occluder, 'instances': obj_records}
                records.write(json.dumps(sample, ensure_ascii=False) + '\n')
                records.flush()
                metadata['completed'] = index + 1
                write_run(args.output / 'run.json', metadata)
        metadata['status'] = 'complete_candidates'
    except Exception as exc:
        metadata['status'] = 'failed'
        metadata['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        write_run(args.output / 'run.json', metadata)
    print('FIRE_DATASET_CANDIDATES', args.output, metadata['completed'])


if __name__ == '__main__':
    main()
