"""Render explicit e-bike Blender states with per-instance visible boxes and receipts."""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
from datasets.scripts.common.blender_standees import mask_material, standee  # noqa: E402
from datasets.scripts.common.paths import sample_dir, split_dir
from lab.training_common.production_guard import generation_guard  # noqa: E402
from datasets.scripts.e_bike import plan_blender_batch as planner  # noqa: E402


def vehicle_material(texture):
    """Use the same alpha blend as the ID mask, with artwork visible on both sides."""
    mat = bpy.data.materials.new('MAT-e-bike-artwork')
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    image = nodes.new('ShaderNodeTexImage')
    image.image = bpy.data.images.load(str(ROOT / texture), check_existing=True)
    transparent = nodes.new('ShaderNodeBsdfTransparent')
    emission = nodes.new('ShaderNodeEmission')
    mix = nodes.new('ShaderNodeMixShader')
    links.new(image.outputs['Color'], emission.inputs['Color'])
    links.new(image.outputs['Alpha'], mix.inputs[0])
    links.new(transparent.outputs[0], mix.inputs[1])
    links.new(emission.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], output.inputs['Surface'])
    return mat


def floor_height(scene, x, y):
    floor = scene.objects.get('Arena floor 4.2 x 4.2 m')
    if floor is None or floor.type != 'MESH' or floor.hide_render:
        raise ValueError('visible arena floor mesh is required')
    floor = floor.evaluated_get(bpy.context.evaluated_depsgraph_get())
    inverse = floor.matrix_world.inverted()
    hit, point, _, _ = floor.ray_cast(inverse @ Vector((x, y, 2)), inverse.to_3x3() @ Vector((0, 0, -1)))
    if not hit:
        raise ValueError('floor ray missed at declared vehicle foot')
    return (floor.matrix_world @ point).z


def visible_scene_masks(scene, actors, path, full_dir):
    """Render instance IDs with actual scene mesh occlusion and isolated full masks."""
    colors = [(1, 0, 0), (0, 1, 0)]
    boards = [bpy.data.objects[a['geometry']] for a in actors]
    meshes = [o for o in scene.objects if o.type == 'MESH']
    visibility = {o.name: o.hide_render for o in scene.objects}
    materials = {o.name: list(o.data.materials) for o in meshes}
    settings = (scene.render.film_transparent, scene.cycles.samples, scene.cycles.use_denoising,
                scene.view_settings.view_transform, scene.view_settings.exposure, scene.render.filepath)
    blocker = bpy.data.materials.new('MAT-e-bike-mask-occluder')
    blocker.use_nodes = True
    nodes, links = blocker.node_tree.nodes, blocker.node_tree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    black = nodes.new('ShaderNodeEmission')
    black.inputs['Color'].default_value = (0, 0, 0, 1)
    links.new(black.outputs[0], output.inputs['Surface'])
    try:
        for obj in meshes:
            obj.data.materials.clear()
            if obj in boards:
                i = boards.index(obj)
                obj.data.materials.append(mask_material(ROOT / actors[i]['texture'], colors[i]))
            else:
                obj.data.materials.append(blocker)
        scene.render.film_transparent = True
        scene.cycles.samples = 4
        scene.cycles.use_denoising = False
        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.exposure = 0
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        image = bpy.data.images.load(str(path), check_existing=False)
        pixels = np.flipud(np.asarray(image.pixels[:], dtype=np.float32).reshape((480, 640, 4)))
        bpy.data.images.remove(image)
        palette = np.asarray(colors[:len(actors)], dtype=np.float32)
        distance = ((pixels[:, :, :3, None] - palette.T[None, None, :, :]) ** 2).sum(axis=2)
        ids, confidence = distance.argmin(axis=2), distance.min(axis=2) < .32
        boxes, visible_counts, full_counts, full_paths = [], [], [], []
        for i, board in enumerate(boards):
            for obj in meshes:
                obj.hide_render = obj is not board
            full_path = full_dir / (path.stem + f'_{i:02d}.png')
            scene.render.filepath = str(full_path)
            bpy.ops.render.render(write_still=True)
            image = bpy.data.images.load(str(full_path), check_existing=False)
            full = np.flipud(np.asarray(image.pixels[:], dtype=np.float32).reshape((480, 640, 4)))
            bpy.data.images.remove(image)
            isolated = (((full[:, :, :3] - palette[i]) ** 2).sum(axis=2) < .32) & (full[:, :, 3] > .45)
            matched = (ids == i) & confidence & (pixels[:, :, 3] > .45) & isolated
            yy, xx = np.nonzero(matched)
            if not len(xx) or not isolated.sum():
                raise ValueError('instance is not visible')
            boxes.append([int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1])
            visible_counts.append(int(len(xx)))
            full_counts.append(int(isolated.sum()))
            full_paths.append(full_path.relative_to(ROOT).as_posix())
        return boxes, visible_counts, full_counts, full_paths
    finally:
        for obj in meshes:
            obj.data.materials.clear()
            for mat in materials[obj.name]:
                obj.data.materials.append(mat)
        for obj in scene.objects:
            if obj.name in visibility:
                obj.hide_render = visibility[obj.name]
        (scene.render.film_transparent, scene.cycles.samples, scene.cycles.use_denoising,
         scene.view_settings.view_transform, scene.view_settings.exposure, scene.render.filepath) = settings


def validate(plan):
    if plan.get('scene') not in planner.SCENES.values() or Path(bpy.data.filepath).resolve() != (ROOT / plan['scene']).resolve():
        raise ValueError('open the exact scene declared by the plan')
    if plan.get('model_key') != 'e_bike' or plan.get('target_split') not in ('train', 'val', 'test'):
        raise ValueError('unexpected task or split')
    if plan.get('constraints') != {'minimum_visible_fraction': .7, 'minimum_visible_pixels': 120, 'allow_truncation': False}:
        raise ValueError('unexpected constraints')
    ready = {}
    for record in planner.read(planner.USE)['records']:
        for rel in record['task_use_status']:
            if (rel.get('task'), rel.get('split'), rel.get('role'), rel.get('status')) == ('e_bike', 'train', 'target_foreground', 'ready'):
                ready[rel['posture']] = record['image']
    rows = planner.read(planner.FOREGROUNDS)['records']
    items = {cls: next(row for row in rows if row['path'] == ready[cls] and row['class'] == cls)
             for cls in ('upright', 'fallen')}
    if type(plan.get('budget')) is not int or not 1 <= plan['budget'] <= 1200 or plan['budget'] != len(plan['strata']):
        raise ValueError('invalid state budget')
    signatures = set()
    branch_classes = dict(zip(('EB-C0-empty', 'EB-C1-upright', 'EB-C2-fallen', 'EB-C3-two-upright',
                               'EB-C4-two-fallen', 'EB-C5-mixed-multiple'), planner.CLASSES))
    for state in plan['strata']:
        params = state['parameters']
        if (set(params) != {'camera_x_mm', 'exposure_hundredth_ev', 'actors'} or
                type(params['camera_x_mm']) is not int or type(params['exposure_hundredth_ev']) is not int or
                not -30 <= params['camera_x_mm'] <= 30 or not -30 <= params['exposure_hundredth_ev'] <= 20):
            raise ValueError('camera or exposure outside plan bounds')
        classes = state['target']['classes']
        if (classes != [a['class'] for a in params['actors']] or
                tuple(classes) != branch_classes.get(state['failure_event']) or
                state['labels'] != {'format': 'yolo_visible_box', 'classes': {'upright': 0, 'fallen': 1}} or
                state['augmentation'] != {'method': 'camera_and_exposure'} or
                state['min_count'] != 1 or state['max_count'] != 1):
            raise ValueError('actor class mismatch')
        sources = {plan['scene']}
        for actor in params['actors']:
            cls = actor['class']
            if (set(actor) != {'class', 'x_mm', 'y_mm', 'height_mm', 'yaw_degrees'} or cls not in items or
                    any(type(actor[k]) is not int for k in ('x_mm', 'y_mm', 'height_mm', 'yaw_degrees')) or
                    not -65 <= actor['x_mm'] <= 65 or not 10 <= actor['y_mm'] <= 35 or not 80 <= actor['height_mm'] <= 155 or
                    actor['yaw_degrees'] not in (-8, -4, 0, 4, 8)):
                raise ValueError('unsupported actor')
            sources.update(items[cls][key] for key in ('path', 'cropped_image', 'source_frame', 'source_video'))
        if state['sources'] != sorted(sources) or state['background'] != {'scene': plan['scene']}:
            raise ValueError('state sources or scene differ from actual assets')
        signature = json.dumps(params, sort_keys=True)
        if signature in signatures:
            raise ValueError('duplicate state')
        signatures.add(signature)
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    path, out = (ROOT / args.plan).resolve(), (ROOT / args.output).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('plan must be inside the workspace')
    plan = planner.read(path)
    if out not in {sample_dir('e_bike').resolve(), split_dir('e_bike', plan['target_split']).resolve()}:
        raise ValueError('output must be the task sample or matching split')
    items = validate(plan)
    scene = bpy.context.scene
    camera = scene.objects.get('Vehicle camera')
    if camera is None:
        raise ValueError('missing Vehicle camera')
    if any(obj.type == 'MESH' and not obj.hide_render and any(x in obj.name.lower() for x in ('bike', 'vehicle', 'motorcycle')) for obj in scene.objects):
        raise ValueError('scene may contain an unlabelled vehicle mesh')
    media = {ROOT / source for state in plan['strata'] for source in state['sources'] if source != plan['scene']}
    procedural = [path, ROOT / plan['scene'], planner.USE, planner.FOREGROUNDS,
                  ROOT/'datasets/scripts/e_bike/config/source.yaml', ROOT/'datasets/scripts/e_bike/config/dataset.yaml']
    code = [Path(__file__), Path(planner.__file__), ROOT / 'datasets/scripts/common/blender_standees.py',
            ROOT / 'lab/training_common/production_guard.py']
    relout = out.relative_to(ROOT).as_posix()
    with generation_guard('e_bike.blender_batch:' + relout, sorted(media), [out],
                          {'plan': path.relative_to(ROOT).as_posix()}, procedural_inputs=procedural,
                          code_paths=code, root=ROOT, production_plan=plan,
                          sample_output_paths=[out / 'images'], auxiliary_output_paths=[out / 'masks']) as run:
        if run.reused:
            return
        if out.exists() and any(out.iterdir()):
            raise ValueError('output is not empty')
        scene.camera = camera
        scene.render.engine = 'CYCLES'
        preferences = bpy.context.preferences.addons['cycles'].preferences
        preferences.compute_device_type = 'OPTIX'
        preferences.get_devices()
        if not any(device.type == 'OPTIX' for device in preferences.devices):
            raise RuntimeError('no OPTIX GPU available')
        for device in preferences.devices:
            device.use = device.type == 'OPTIX'
        scene.cycles.device = 'GPU'
        scene.cycles.samples = 12
        scene.cycles.use_denoising = True
        scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = 640, 480, 100
        scene.render.image_settings.file_format = 'PNG'
        for part in ('images', 'labels', 'masks/visible', 'masks/full'):
            (out / part).mkdir(parents=True, exist_ok=False)
        records = []
        for i, state in enumerate(run.plan['strata']):
            if state['planned_count'] != 1:
                raise ValueError('one output per state required')
            p = state['parameters']
            camera.location = (p['camera_x_mm']/1000, -.40, .16)
            camera.rotation_euler = (Vector((0, .03, .085)) - camera.location).to_track_quat('-Z', 'Y').to_euler()
            camera.data.lens = 22
            actors = []
            for spec in p['actors']:
                source = items[spec['class']]
                texture = bpy.data.images.load(str(ROOT / source['cropped_image']), check_existing=True)
                width = spec['height_mm'] / 1000 * texture.size[0] / texture.size[1]
                x, y = spec['x_mm']/1000, spec['y_mm']/1000
                yaw = math.radians(spec['yaw_degrees'])
                heights = [floor_height(scene, x + width*t*math.cos(yaw), y + width*t*math.sin(yaw))
                           for t in (-.5, -.25, 0, .25, .5)]
                board_bottom = max(heights) + .001
                actor = standee({'artwork': source['path'], 'image': source['cropped_image'], 'source': source},
                                (x, y, board_bottom - .004), yaw, False, spec['height_mm']/1000, True)
                actor['ground_surface_samples_m'] = heights
                actor['board_bottom_m'] = board_bottom
                board = bpy.data.objects[actor['geometry']]
                board.data.materials.clear()
                board.data.materials.append(vehicle_material(source['cropped_image']))
                actors.append(actor)
            stem = f'bike_{i+1:06d}'
            image, label = out / 'images' / (stem+'.png'), out / 'labels' / (stem+'.txt')
            boxes, visible, full, mask_paths = [], [], [], []
            if actors:
                boxes, visible, full, mask_paths = visible_scene_masks(scene, actors, out / 'masks/visible' / (stem+'.png'), out / 'masks/full')
            lines = []
            for spec, box, count, whole in zip(p['actors'], boxes, visible, full):
                x1,y1,x2,y2 = box
                if x1 <= 0 or y1 <= 0 or x2 >= 640 or y2 >= 480 or count < 120 or count/whole < .7:
                    raise ValueError('target is truncated or below visible threshold: ' + state['id'])
                class_id = {'upright': 0, 'fallen': 1}[spec['class']]
                lines.append(f"{class_id} {(x1+x2)/1280:.8f} {(y1+y2)/960:.8f} {(x2-x1)/640:.8f} {(y2-y1)/480:.8f}")
            label.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf8')
            scene.view_settings.exposure = p['exposure_hundredth_ev']/100
            scene.render.filepath = str(image)
            bpy.ops.render.render(write_still=True)
            actual = {'scene': plan['scene'], 'objects': p['actors'], 'ground_placement': [
                {'surface_samples_m': a['ground_surface_samples_m'], 'board_bottom_m': a['board_bottom_m']} for a in actors],
                      'boxes': boxes, 'visible_pixels': visible,
                      'isolated_pixels': full, 'isolated_masks': mask_paths, 'label': label.relative_to(ROOT).as_posix()}
            run.record_sample(state['id'], output=image, sources=state['sources'], parameters=p, actual=actual)
            records.append({'image': image.relative_to(ROOT).as_posix(), 'state_id': state['id'], **actual})
            for actor in actors:
                bpy.data.objects.remove(bpy.data.objects[actor['geometry']], do_unlink=True)
        (out / 'manifest.json').write_text(json.dumps({'status': 'training_candidates_not_deliverable', 'records': records}, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
        print(json.dumps({'output': relout, 'images': len(records)}))


if __name__ == '__main__':
    main()
