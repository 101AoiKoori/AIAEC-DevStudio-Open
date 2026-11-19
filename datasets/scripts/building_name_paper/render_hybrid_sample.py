"""Render low-view building foregrounds and physical floor light transport."""
import argparse
from hashlib import file_digest
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.building_name_paper import hybrid_sample as hybrid
from datasets.scripts.common.blender_masks import configure_device, GEOMETRY_TYPES, masks, targets_inside_frame
from datasets.scripts.common.blender_photo import (add_receiver, aim_area, footprint_inside,
                                                 ground_ray, project, save_render, set_plate_camera)
from datasets.scripts.common.blender_workspace import activate_scene, BUILDINGS
from datasets.scripts.common.config import load_task_config, source_path
from datasets.scripts.common.names import make_name
from lab.training_common.production_guard import generation_guard


def prepare_scene(device, config):
    activate_scene(BUILDINGS)
    scene = bpy.context.scene
    bpy.context.window.view_layer = scene.view_layers[0]
    bpy.context.view_layer.update()
    for layer in scene.view_layers:
        layer.use = layer == scene.view_layers[0]
    scene.render.engine = 'CYCLES'
    configure_device(scene, device)
    scene.render.use_compositing = scene.render.use_sequencer = scene.render.use_motion_blur = False
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    scene.cycles.samples = config['samples']
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 6
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 1
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 8
    for marker in scene.timeline_markers:
        marker.camera = None
    for obj in scene.objects:
        if obj.type in GEOMETRY_TYPES or obj.type == 'LIGHT':
            obj.hide_render = True
    camera = scene.camera
    camera.animation_data_clear()
    camera.data.animation_data_clear()
    _, source = load_task_config(hybrid.TASK)
    groups, baseline = {}, {}
    for key, name in source['blender_collections'].items():
        collection = bpy.data.collections[name]
        collection.hide_render = False
        members = [o for o in collection.all_objects if o.type in GEOMETRY_TYPES
                   and not o.name.startswith(('Fire paper ', 'Candidate fire '))]
        paper = next(o for o in members if o.name.startswith('Name paper - '))
        text = next(o for o in members if o.name.startswith('Editable text - '))
        roots = [obj for obj in members if obj.parent not in members]
        groups[key] = {'members': members, 'roots': roots, 'paper': paper, 'text': text}
        baseline.update({obj: obj.matrix_world.copy() for obj in roots})
    receiver = add_receiver()
    key = bpy.data.objects.new('Photo-ground broad area', bpy.data.lights.new('Photo-ground area', 'AREA'))
    scene.collection.objects.link(key)
    lighting = config['lighting']
    background = next(node for node in scene.world.node_tree.nodes if node.type == 'BACKGROUND')
    background.inputs['Color'].default_value = (*lighting['world_color'], 1)
    background.inputs['Strength'].default_value = lighting['world_strength']
    return scene, camera, groups, baseline, receiver, key


def layout(scene, camera, groups, baseline, plate, state):
    active, parameters = state['target']['buildings'], state['parameters']
    size = plate['image_size']
    fit = set_plate_camera(scene, camera, size, parameters['camera_height_m'],
                           plate['perspective']['horizon_y_px'], plate['focal_px_assumed'])
    for obj, matrix in baseline.items():
        obj.matrix_world = matrix
    for building, group in groups.items():
        for obj in group['members']:
            obj.hide_render = building not in active
            obj.visible_camera = True
        if building in active:
            group['text'].data.body = make_name(parameters['seed'] + ord(building))
            bpy.context.view_layer.update()
            if group['text'].dimensions.x > .115:
                group['text'].scale *= .115 / group['text'].dimensions.x
    main = [obj for building in active for obj in groups[building]['members']]
    # Envelope excludes the paper/font, whose bbox lies outside the facade plane.
    bounds = {}
    for building in active:
        points = [obj.matrix_world @ Vector(corner) for obj in groups[building]['members']
                  if obj.type == 'MESH' and obj != groups[building]['paper'] for corner in obj.bound_box]
        bounds[building] = ([min(v[i] for v in points) for i in range(3)],
                            [max(v[i] for v in points) for i in range(3)])
    center = Vector(((min(b[0][0] for b in bounds.values()) + max(b[1][0] for b in bounds.values()))/2,
                     (min(b[0][1] for b in bounds.values()) + max(b[1][1] for b in bounds.values()))/2,
                     min(b[0][2] for b in bounds.values())))
    # Rotating the cluster gives the same relative yaw as orbiting its camera.
    rotation = Matrix.Rotation(math.radians(-parameters['yaw_degrees']), 4, 'Z')
    attempts = []
    for anchor in plate['anchor_candidates_px']:
        point = ground_ray(camera, anchor, size, plate['focal_px_assumed'])
        transform = Matrix.Translation(point) @ rotation @ Matrix.Translation(-center)
        for building in active:
            for obj in groups[building]['roots']:
                obj.matrix_world = transform @ baseline[obj]
        bpy.context.view_layer.update()
        if any(groups[b]['text'].dimensions.y < .01 for b in active):
            raise ValueError('Building text did not evaluate in the facade plane')
        feet = {}
        for building, (lower, upper) in bounds.items():
            corners = [transform @ Vector((x, y, lower[2]))
                       for x, y in [(lower[0], lower[1]), (upper[0], lower[1]),
                                    (upper[0], upper[1]), (lower[0], upper[1])]]
            pixels = [project(scene, camera, p, size) for p in corners]
            feet[building] = {'footprint_px': pixels, 'footprint_world_m': [list(p) for p in corners]}
        ground_ok = all(footprint_inside(row['footprint_px'], plate['ground_polygon']) for row in feet.values())
        for info in feet.values():
            xs, ys = zip(*info['footprint_px'])
            if any(min(xs) < x1 and max(xs) > x0 and min(ys) < y1 and max(ys) > y0
                   for x0, y0, x1, y1 in plate['forbidden_footprint_xyxy']):
                ground_ok = False
        framed = targets_inside_frame(scene, camera, main, margin=.012)
        attempts.append({'anchor_px': anchor, 'ground_ok': ground_ok, 'framed': framed})
        if ground_ok and framed:
            fit.update(matrix_world=[list(row) for row in camera.matrix_world],
                       ground_anchor_px=anchor, ground_anchor_world_m=list(point),
                       projected_footprints=feet, attempts=attempts,
                       yaw_degrees=parameters['yaw_degrees'])
            return fit, point, main
    raise ValueError(f'{state["id"]}: no in-frame grounded layout: {attempts}; feet={feet}')


def render(scene, camera, groups, baseline, receiver, key, config, plates, run):
    out = hybrid.LAYERS
    for sub in ('foreground', 'lighting', 'masks'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rows = []
    by_id = {p['id']: p for p in plates}
    lighting = config['lighting']
    for state in run.plan['strata']:
        stem = state['id']
        plate = by_id[state['background']['plate_id']]
        fit, point, main = layout(scene, camera, groups, baseline, plate, state)
        scene.cycles.seed = state['parameters']['seed']
        aim_area(key, point + Vector(lighting['area_offset_m']), point,
                 lighting['area_energy_w'], lighting['area_size_m'])
        receiver.hide_render = False
        receiver.visible_camera = False
        foreground = f'foreground/{stem}.png'
        save_render(scene, out / foreground)
        receiver.visible_camera = True
        for obj in main:
            obj.visible_camera = False
        shadowed = f'lighting/{stem}_shadowed.png'
        save_render(scene, out / shadowed)
        for obj in main:
            obj.hide_render = True
        lit = f'lighting/{stem}_lit.png'
        save_render(scene, out / lit)
        for obj in main:
            obj.hide_render = False
            obj.visible_camera = True
        receiver.hide_render = True
        active = state['target']['buildings']
        visible, instances = masks(scene, camera, {b: [groups[b]['paper'], groups[b]['text']] for b in active},
                                   out, stem, 'paper', palette=hybrid.PALETTE)
        truth = []
        w, h = plate['image_size']
        for obj in instances:
            if not obj['label_eligible']:
                raise ValueError(f'{stem}: insufficient visible paper: {obj}')
            box = [round(v*dimension) for v, dimension in zip(obj['candidate_xyxy_normalized'], (w,h,w,h))]
            truth.append({'building_id': obj['bin'], 'xyxy': box, 'visibility': obj})
        actual = {'id': stem, 'state_id': stem, 'source_split': plate['source_split'],
            'source_photo': plate['image'], 'source_group': plate['parent_group'],
            'frame_environment': plate['source_review'], 'image_size': plate['image_size'],
            'known_evaluation_contact': plate['known_evaluation_contact'],
            'perspective_evidence': plate['perspective'],
            'camera': fit, 'parameters': state['parameters'], 'buildings': active,
            'foreground': foreground, 'visible_mask': visible, 'shadowed_receiver': shadowed,
            'lit_receiver': lit, 'instances': truth, 'appearance': lighting}
        rows.append(actual)
        run.record_sample(stem, output=out/foreground, sources=state['sources'],
                          parameters=state['parameters'], actual=actual)
        print('HYBRID_RENDERED', stem, fit['ground_anchor_px'], [r['xyxy'] for r in truth], flush=True)
    with source_path(hybrid.TASK, 'blender_scene').open('rb') as stream:
        blend_digest = file_digest(stream, 'sha256').hexdigest()
    (out / 'manifest.json').write_text(json.dumps({'task': hybrid.TASK, 'source_plan': config,
        'source_decisions': [p['source_review']['record_id'] for p in plates], 'records': rows,
        'source_blend_sha256': blend_digest, 'blender_version': bpy.app.version_string,
        'production_plan': run.plan}, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    run.set_result({'images': len(rows), 'paper_boxes': sum(len(r['instances']) for r in rows)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='OPTIX')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    config = hybrid.recipe()
    plates = hybrid.qualified_plates(config)
    declaration = hybrid.declaration(config, plates)
    scene, camera, groups, baseline, receiver, key = prepare_scene(args.device, config)
    if args.dry_run:
        for state in declaration['strata']:
            plate = next(p for p in plates if p['id'] == state['background']['plate_id'])
            fit, _, _ = layout(scene, camera, groups, baseline, plate, state)
            print('LAYOUT_OK', state['id'], fit['ground_anchor_px'], flush=True)
        return
    folder = Path(__file__).parent
    with generation_guard('building_name.hybrid_layers', [ROOT/p['image'] for p in plates], [hybrid.LAYERS],
        {'device': args.device, 'recipe': config}, root=ROOT, production_plan=declaration,
        code_paths=[Path(__file__), Path(hybrid.__file__), ROOT/'datasets/scripts/common/blender_photo.py',
                    ROOT/'datasets/scripts/common/blender_masks.py', ROOT/'datasets/scripts/common/blender_workspace.py',
                    ROOT/'datasets/scripts/common/config.py', ROOT/'datasets/scripts/common/frame_environment.py',
                    ROOT/'datasets/scripts/common/names.py'],
        procedural_inputs=[source_path(hybrid.TASK, 'hybrid_plan'), source_path(hybrid.TASK, 'blender_scene'),
                           source_path(hybrid.TASK, 'font'), folder/'config/source.yaml', folder/'config/dataset.yaml'],
        sample_output_paths=[hybrid.LAYERS/'foreground'],
        auxiliary_output_paths=[hybrid.LAYERS/'lighting',hybrid.LAYERS/'masks']) as run:
        if run.reused:
            return
        if hybrid.LAYERS.exists() and any(hybrid.LAYERS.iterdir()):
            raise ValueError('Hybrid layer output must be empty for rendering')
        render(scene, camera, groups, baseline, receiver, key, config, plates, run)


if __name__ == '__main__':
    main()
