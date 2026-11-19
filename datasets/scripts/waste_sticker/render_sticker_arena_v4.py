"""Render planned animated waste-sticker instances in a guarded arena batch.

Run in background Blender with the selected arena loaded. Every paper must
have an identifiable visible mask or the batch stops without a complete receipt.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'datasets/scripts/waste_sticker'))
from lab.training_common.production_guard import generation_guard
from datasets.scripts.common.blender_masks import masks as shared_masks, configure_device, targets_inside_frame
from datasets.scripts.common.config import source_path, load_task_config
from datasets.scripts.common.blender_workspace import ARENA, STICKERS, WORKSPACE_BLEND, activate_scene, source_ref
from datasets.scripts.waste_sticker.source_inputs import artwork_catalog, real_source_catalog
from datasets.scripts.waste_sticker.configure_blender_orbit import (
    ORBIT_VIEWS, STICKER_SCENE, camera_name, sticker_bin_objects, sticker_objects,
)
from datasets.scripts.common.paths import sample_dir, split_dir, sticker_orbit_sample_dir


def masks(scene, camera, parts, output, stem, tag, force_visible=()):
    return shared_masks(scene, camera, parts, output, stem, tag, force_visible,
                        palette={'blue': (1, 0, 0), 'red': (0, 1, 0),
                                 'green': (0, 0, 1), 'gray': (1, 1, 0)})

ASSET = source_ref(WORKSPACE_BLEND, STICKERS)
SOURCE_PLAN = 'lab/features/waste_bin/training/config/waste_sticker_source_plan_20261114.json'
COLORS = ('blue', 'red', 'green', 'gray')
CONSTRAINTS = {'resolution': [640, 480], 'labels': 'visible_paper_yolo_candidate',
               'min_visible_fraction': 0.70, 'min_visible_pixels': 24}


def validate(plan, scene):
    if plan.get('model_key') != 'waste_sticker' or plan.get('target_split') not in ('train', 'test', 'val') or plan.get('constraints') != CONSTRAINTS:
        raise ValueError('unsupported sticker arena plan')
    for state in plan['strata']:
        p = state['parameters']
        augmentation = ('exposure_ev', 'saturation', 'blur_pixels', 'occluder_color', 'paper_width_ratio', 'paper_aspect')
        required = {'frame', 'camera_location', 'camera_aim', *augmentation}
        if (set(p) - {'visible_bins'} != required or type(p['frame']) is not int
                or not 1 <= p['frame'] <= 240 or state['sources'] != [scene, ASSET]):
            raise ValueError('invalid frame or asset declaration')
        if 'visible_bins' in p:
            visible = p['visible_bins']
            if (not isinstance(visible, list) or not visible
                    or any(color not in COLORS for color in visible) or len(set(visible)) != len(visible)):
                raise ValueError('visible_bins must list one to four distinct bucket colors')
            if p['occluder_color'] is not None and p['occluder_color'] not in visible:
                raise ValueError('occluder_color must refer to a visible bin')
        for key in ('camera_location', 'camera_aim'):
            if not isinstance(p[key], list) or len(p[key]) != 3 or any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 100 for x in p[key]):
                raise ValueError('invalid camera vector')
        if (type(p['exposure_ev']) not in (int, float) or not math.isfinite(p['exposure_ev']) or not -1 <= p['exposure_ev'] <= 1
                or type(p['saturation']) not in (int, float) or not math.isfinite(p['saturation']) or not .5 <= p['saturation'] <= 1.5
                or type(p['blur_pixels']) is not int or not 0 <= p['blur_pixels'] <= 2
                or p['occluder_color'] not in (None, *COLORS)
                or type(p['paper_width_ratio']) not in (int, float) or not .39 <= p['paper_width_ratio'] <= .46
                or type(p['paper_aspect']) not in (int, float) or not .9 <= p['paper_aspect'] <= 1.04):
            raise ValueError('invalid bounded image augmentation')
        if state['augmentation'] != {k: p[k] for k in augmentation}:
            raise ValueError('augmentation declaration differs from execution parameters')
        if p['camera_location'] == p['camera_aim'] or state['max_count'] != 1:
            raise ValueError('invalid camera or duplicate state budget')
        if state['background'] != {'scene': scene} or state['target'] != {'asset': ASSET, 'frame': p['frame']}:
            raise ValueError('target does not match scene and animation')
        if state['labels'] != {'format': 'yolo_detection', 'visibility': 'visible_paper_candidate'}:
            raise ValueError('invalid label contract')
    if not sum(s['min_count'] for s in plan['strata']) <= plan['budget'] <= sum(s['max_count'] for s in plan['strata']):
        raise ValueError('budget is outside state bounds')


def set_bin_render_visibility(groups, selected, originally_hidden):
    for color, objects in groups.items():
        for obj in objects:
            obj.hide_render = color not in selected or obj in originally_hidden


def run_render(out, plan, objects, scene_path, source_plan, seed, samples, max_seconds):
    scene = bpy.context.scene
    for obj in objects:
        scene.collection.objects.link(obj)
    groups = sticker_bin_objects(objects)
    originally_hidden = {obj for obj in objects if obj.hide_render}
    bpy.context.view_layer.update()
    root = bpy.data.objects['Measured height scale']
    root.scale = (.1, .1, .1)
    root.location = (0, 1, 0)
    papers = {color: bpy.data.objects[color + ' item sticker paper'] for color in COLORS}
    for paper in papers.values():
        if paper.get('aec_sticker_placement'):
            paper.animation_data_clear()
            paper.location = (0, 0, 0)
            paper.rotation_quaternion = (1, 0, 0, 0)
            paper.scale = (1, 1, 1)
    base_vertices = {color: [v.co.copy() for v in paper.data.vertices] for color, paper in papers.items()}
    markings = {color: [bpy.data.objects[color + ' front marking'], bpy.data.objects[color + ' marking border']]
                for color in COLORS}
    for signs in markings.values():
        for sign in signs:
            sign.location -= Vector(sign.get('aec_sticker_orbit_sign_offset', (0, 0, 0)))
    marking_locations = {color: [obj.location.copy() for obj in objects] for color, objects in markings.items()}
    marking_bottom = {color: min(v.co.z for v in objects[0].data.vertices) for color, objects in markings.items()}
    occluders = {}
    for color, paper in papers.items():
        vertices = [v.co.copy() for v in paper.data.vertices]
        points = [vertices[0].lerp(vertices[1], .8), vertices[1].copy(), vertices[2].copy(), vertices[3].lerp(vertices[2], .8)]
        for point in points:
            point.y -= .03
        mesh = bpy.data.meshes.new('sticker occluder ' + color)
        mesh.from_pydata(points, [], [(0, 1, 2, 3)])
        obj = bpy.data.objects.new('sticker occluder ' + color, mesh)
        scene.collection.objects.link(obj)
        obj.parent = paper.parent
        obj.matrix_parent_inverse.identity()
        obj.hide_render = True
        occluders[color] = obj
    camera = scene.objects['Vehicle camera']
    scene.camera = camera
    scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = 640, 480, 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.engine, scene.cycles.samples = 'CYCLES', samples
    configure_device(scene, 'OPTIX')
    tree = bpy.data.node_groups.new('Waste sticker augmentation', 'CompositorNodeTree')
    tree.interface.new_socket(name='Image', in_out='OUTPUT', socket_type='NodeSocketColor')
    layer = tree.nodes.new('CompositorNodeRLayers')
    tone = tree.nodes.new('CompositorNodeHueSat')
    blur = tree.nodes.new('CompositorNodeBlur')
    result = tree.nodes.new('NodeGroupOutput')
    tree.links.new(layer.outputs['Image'], tone.inputs['Image'])
    tree.links.new(tone.outputs['Image'], blur.inputs['Image'])
    tree.links.new(blur.outputs['Image'], result.inputs['Image'])
    scene.compositing_node_group = tree
    (out / 'images').mkdir(parents=True)
    (out / 'masks').mkdir()
    (out / ('labels_candidate' if out == sample_dir('waste_sticker').resolve() else 'labels')).mkdir()
    records = []
    started = time.monotonic()
    schedule = {r['frame']: {b['bin_color']: b['class_id'] for b in r['bins']} for r in source_plan['samples']}
    for index, state in enumerate(plan.plan['strata']):
        if time.monotonic() - started >= max_seconds:
            raise TimeoutError('render batch time limit reached; partial outputs retained')
        if not state['planned_count']:
            continue
        p = state['parameters']
        scene.frame_set(p['frame'])
        selected = p.get('visible_bins', COLORS)
        set_bin_render_visibility(groups, selected, originally_hidden)
        for color in selected:
            paper = papers[color]
            base = base_vertices[color]
            body = paper.parent
            body_width = max(v.co.x for v in body.data.vertices) - min(v.co.x for v in body.data.vertices)
            width = body_width * p['paper_width_ratio']
            height = width / p['paper_aspect']
            center = (base[0].x + base[1].x) / 2
            low = base[0].z
            slope = (base[3].y - base[0].y) / (base[3].z - low)
            top_y = base[0].y + slope * height
            new = [(center-width/2,base[0].y,low), (center+width/2,base[0].y,low),
                   (center+width/2,top_y,low+height), (center-width/2,top_y,low+height)]
            for vertex, co in zip(paper.data.vertices, new):
                vertex.co = co
            paper.data.update()
            shift = max(0.0, low + height + .01 - marking_bottom[color])
            for obj, original in zip(markings[color], marking_locations[color]):
                obj.location = original + Vector((0, slope * shift, shift))
            verts = [v.co.copy() for v in paper.data.vertices]
            cover = [verts[0].lerp(verts[1],.8), verts[1].copy(), verts[2].copy(), verts[3].lerp(verts[2],.8)]
            for vertex, co in zip(occluders[color].data.vertices, cover):
                co.y -= .03
                vertex.co = co
            occluders[color].data.update()
        camera.location = p['camera_location']
        camera.rotation_euler = (Vector(p['camera_aim']) - camera.location).to_track_quat('-Z', 'Y').to_euler()
        scene.cycles.seed = seed + index
        for color, obj in occluders.items():
            obj.hide_render = color not in selected or p.get('occluder_color') != color
        scene.view_settings.exposure = p.get('exposure_ev', 0.0)
        tone.inputs['Saturation'].default_value = p.get('saturation', 1.0)
        blur.inputs['Size'].default_value = (p.get('blur_pixels', 0), p.get('blur_pixels', 0))
        scene.render.use_compositing = True
        bpy.context.view_layer.update()
        for color in selected:
            obj = papers[color]
            actual = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.polygons[0].material_index
            if actual != schedule[p['frame']][color]:
                raise ValueError('evaluated paper material differs from recorded class schedule: ' + color)
        stem = f'sticker_{index:05d}'
        image = out / 'images' / (stem + '.png')
        scene.render.filepath = str(image)
        bpy.ops.render.render(write_still=True)
        _, visibility = masks(scene, camera, {c: [papers[c]] for c in selected}, out, stem, 'paper')
        lines = []
        for info in visibility:
            color = info['bin']
            if not info['label_eligible'] or not targets_inside_frame(scene, camera, [papers[color]]):
                raise ValueError('sticker visibility does not support complete labels: ' + color)
            x1, y1, x2, y2 = info['candidate_xyxy_normalized']
            cls = schedule[p['frame']][color]
            lines.append(f'{cls} {(x1+x2)/2:.8f} {(y1+y2)/2:.8f} {x2-x1:.8f} {y2-y1:.8f}')
            info['class_id'] = cls
        label = out / ('labels_candidate' if out == sample_dir('waste_sticker').resolve() else 'labels') / (stem + '.txt')
        label.write_text('\n'.join(lines) + '\n', encoding='utf8')
        actual = {'scene': scene_path, 'asset': ASSET, 'frame': p['frame'], 'instances': visibility,
                  'visible_bins': list(selected),
                  'camera_location': p['camera_location'], 'camera_aim': p['camera_aim'],
                  'exposure_ev': p.get('exposure_ev', 0.0), 'saturation': p.get('saturation', 1.0),
                  'blur_pixels': p.get('blur_pixels', 0), 'occluder_color': p.get('occluder_color'),
                  'paper_width_ratio': p['paper_width_ratio'], 'paper_aspect': p['paper_aspect'],
                  'marking_shift_source_units': {color: markings[color][0].location.z - marking_locations[color][0].z
                                                 for color in selected}}
        plan.record_sample(state['id'], output=image, sources=state['sources'], parameters=p, actual=actual)
        records.append({'image': image.relative_to(ROOT).as_posix(), 'label_candidate': label.relative_to(ROOT).as_posix(),
                        'state_id': state['id'], 'actual': actual})
    (out / 'manifest.json').write_text(json.dumps({'status': 'candidate', 'seed': seed, 'records': records}, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def orbit_declaration(scene_ref, frame, selected):
    strata = [
        {'id': view, 'min_count': 1, 'max_count': 1, 'conditions': [],
         'prior': {'alpha': 1, 'beta': 1}, 'exposure_weight': 1, 'failure_cost': 1,
         'failure_event': 'sticker_visibility_incomplete',
         'background': {'scene': scene_ref, 'scene_name': STICKER_SCENE},
         'target': {'asset': ASSET, 'frame': frame},
         'parameters': {'frame': frame, 'view': view, 'visible_bins': list(selected)},
         'labels': {'format': 'yolo_detection', 'visibility': 'visible_paper_candidate'},
         'augmentation': {'mode': 'none'}, 'sources': [scene_ref, ASSET]}
        for view, _, _ in ORBIT_VIEWS
    ]
    return {'schema_version': 1, 'model_key': 'waste_sticker', 'target_split': 'test',
            'budget': len(strata), 'constraints': CONSTRAINTS, 'uncertainty_weight': 0,
            'purpose': 'Fixed sticker placement viewed from the nine stored Blender cameras',
            'independence_basis': 'one scene and the same sticker state at all nine views',
            'strata': strata}


def render_orbit(out, run, scene, schedule, samples, seed, max_seconds):
    papers = sticker_objects(scene)
    asset = bpy.context.view_layer.layer_collection.children.get(STICKERS)
    if asset is not None:
        # The batch selection takes precedence over the saved Outliner checkboxes.
        asset.exclude = asset.collection.hide_render = False
        for group in asset.children:
            group.exclude = group.collection.hide_render = False
    groups = sticker_bin_objects(scene.objects)
    originally_hidden = {obj for objects in groups.values() for obj in objects if obj.hide_render}
    if not all(paper.get('aec_sticker_placement') for paper in papers.values()):
        raise ValueError('Run configure_blender_orbit.py before rendering the stored views')
    # Timeline camera markers would override the chosen view at the fixed frame.
    for marker in scene.timeline_markers:
        marker.camera = None
    scene.render.image_settings.file_format = 'PNG'
    scene.render.use_motion_blur = False
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.seed = seed
    configure_device(scene, 'OPTIX')
    for directory in ('images', 'masks', 'labels_candidate'):
        (out / directory).mkdir(parents=True)
    started = time.monotonic()
    records = []
    for state in run.plan['strata']:
        if time.monotonic() - started >= max_seconds:
            raise TimeoutError('orbit review exceeded the render time limit')
        params = state['parameters']
        frame, view = params['frame'], params['view']
        scene.frame_set(frame)
        selected = params['visible_bins']
        set_bin_render_visibility(groups, selected, originally_hidden)
        camera = scene.objects[camera_name(view)]
        scene.camera = camera
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        placements = {}
        for color in selected:
            paper = papers[color]
            evaluated = paper.evaluated_get(depsgraph)
            if evaluated.data.polygons[0].material_index != schedule[frame][color]:
                raise ValueError(f'{color}: material differs from the class schedule at frame {frame}')
            center = sum((evaluated.matrix_world @ vertex.co for vertex in evaluated.data.vertices),
                         Vector()) / len(evaluated.data.vertices)
            placements[color] = {'center_world': list(center), 'scale': paper.scale[0],
                                 'rotation_quaternion': list(paper.rotation_quaternion)}
        image = out / 'images' / f'{view}.png'
        scene.render.filepath = str(image)
        bpy.ops.render.render(write_still=True)
        visible_path, instances = masks(scene, camera, {c: [papers[c]] for c in selected}, out, view, 'paper')
        if scene.camera != camera:
            raise RuntimeError(f'{view}: Blender switched the active render camera')
        lines = []
        for item in instances:
            color = item['bin']
            if not item['label_eligible']:
                raise ValueError(f'{view}: {color} paper lacks complete label evidence')
            x0, y0, x1, y1 = item['candidate_xyxy_normalized']
            cls = schedule[frame][color]
            lines.append(f'{cls} {(x0+x1)/2:.8f} {(y0+y1)/2:.8f} {x1-x0:.8f} {y1-y0:.8f}')
            item.update(class_id=cls, placement=placements[color])
        label = out / 'labels_candidate' / f'{view}.txt'
        label.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        actual = {'scene': scene.name, 'frame': frame, 'camera': camera.name, 'visible_bins': selected,
                  'azimuth_deg': camera['azimuth_deg'], 'elevation_deg': camera['elevation_deg'],
                  'pitch_deg': camera['pitch_deg'], 'distance': camera['distance'],
                  'camera_location': list(camera.location),
                  'camera_aim': list(scene['aec_sticker_orbit_target']),
                  'visible_mask': visible_path, 'instances': instances}
        run.record_sample(state['id'], output=image, sources=state['sources'], parameters=params, actual=actual)
        records.append({'image': image.relative_to(ROOT).as_posix(),
                        'label_candidate': label.relative_to(ROOT).as_posix(),
                        'state_id': state['id'], 'actual': actual})
    (out / 'manifest.json').write_text(json.dumps(
        {'status': 'candidate', 'seed': seed, 'records': records}, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')


def main():
    global ASSET, SOURCE_PLAN
    asset_path = source_path('waste_sticker', 'animated_asset')
    ASSET = source_ref(asset_path, STICKERS)
    SOURCE_PLAN = source_path('waste_sticker', 'animated_source_plan').relative_to(ROOT).as_posix()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scene', default=source_path('waste_sticker', 'arena_scene').relative_to(ROOT).as_posix())
    plan_input = ap.add_mutually_exclusive_group(required=True)
    plan_input.add_argument('--plan')
    plan_input.add_argument('--orbit-frame', type=int, help='render all saved orbit cameras at this sticker frame')
    ap.add_argument('--batch-index', type=int, help='select a batch from a multi-batch plan')
    ap.add_argument('--bins', help='comma-separated visible colors for --orbit-frame')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=20261114)
    ap.add_argument('--samples', type=int, default=4)
    ap.add_argument('--max-seconds', type=int, default=360)
    args = ap.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    scene, out = [(ROOT / x).resolve() for x in (args.scene, args.out)]
    plan_path = (ROOT / args.plan).resolve() if args.plan else None
    if (not scene.is_relative_to(ROOT) or not scene.is_file()
            or (plan_path and (not plan_path.is_relative_to(ROOT) or not plan_path.is_file()))
            or Path(bpy.data.filepath).resolve() != scene or not 1 <= args.samples <= 8 or not 20 <= args.max_seconds <= 360):
        raise ValueError('load the declared workspace scene and plan with samples in 1..8')
    scene_ref = scene.relative_to(ROOT).as_posix()
    if scene != source_path('waste_sticker', 'arena_scene'):
        raise ValueError('loaded scene differs from waste_sticker source.yaml')
    if out.exists() and any(out.iterdir()):
        raise ValueError('output must be empty')
    selected = COLORS
    if args.bins is not None:
        selected = tuple(color.strip() for color in args.bins.split(','))
        if args.orbit_frame is None or any(color not in COLORS for color in selected) or len(set(selected)) != len(selected):
            ap.error('--bins requires --orbit-frame and one to four distinct colors: blue,red,green,gray')
    if args.orbit_frame is not None:
        sticker_scene = bpy.data.scenes[STICKER_SCENE]
        if (args.batch_index is not None or not sticker_scene.frame_start <= args.orbit_frame <= sticker_scene.frame_end
                or out != sticker_orbit_sample_dir(args.orbit_frame).resolve()):
            raise ValueError('select an existing sticker frame and its orbit_frame_NNNN sample directory')
        declaration = orbit_declaration(scene_ref, args.orbit_frame, selected)
    else:
        source_declaration = json.loads(plan_path.read_text(encoding='utf8'))
        if 'batches' in source_declaration:
            if args.batch_index is None or not 0 <= args.batch_index < len(source_declaration['batches']):
                raise ValueError('a valid batch index is required for a multi-batch plan')
            batch = source_declaration['batches'][args.batch_index]
            if batch['index'] != args.batch_index or batch['scene'] != scene_ref:
                raise ValueError('selected batch does not match the loaded scene')
            declaration = batch['declaration']
        else:
            if args.batch_index is not None:
                raise ValueError('batch index is not supported for a single-batch plan')
            declaration = source_declaration
        validate(declaration, scene_ref)
        if out not in {sample_dir('waste_sticker').resolve(), split_dir('waste_sticker', declaration['target_split']).resolve()}:
            raise ValueError('output split differs from the plan')
    source_plan = json.loads((ROOT / SOURCE_PLAN).read_text(encoding='utf8'))
    _, source_config = load_task_config('waste_sticker')
    artworks = set(artwork_catalog(source_config).values())
    designated = {row['image'] for row in real_source_catalog(source_config)}
    if source_plan['reference_photo'] != source_config['real_sticker_source']:
        raise ValueError('animated asset reference photo differs from source.yaml')
    if any(frame not in designated for frame in source_plan['source_video_frames']):
        raise ValueError('animated asset frame reference differs from source.yaml interval')
    if any(b['source_artwork'] not in artworks for row in source_plan['samples'] for b in row['bins']):
        raise ValueError('animated material uses an artwork outside the 24 configured classes')
    media = sorted({ROOT / b['source_artwork'] for row in source_plan['samples'] for b in row['bins']})
    media.append(ROOT / source_plan['reference_photo'])
    procedural = [scene, ROOT / SOURCE_PLAN,
                  ROOT/source_config['real_annotations'], *[ROOT/p for p in source_config['records']],
                  ROOT/'datasets/scripts/waste_sticker/config/source.yaml',
                  ROOT/'datasets/scripts/waste_sticker/config/dataset.yaml']
    if asset_path != scene:
        procedural.append(asset_path)
    if plan_path:
        procedural.append(plan_path)
        asset_collection, = activate_scene(ARENA, (STICKERS,))
        objects = list(asset_collection.all_objects)
    else:
        activate_scene(STICKER_SCENE)
    for im in bpy.data.images:
        if im.source == 'FILE' and not im.packed_file:
            path = Path(bpy.path.abspath(im.filepath, library=im.library)).resolve()
            if not path.is_file() or not path.is_relative_to(ROOT):
                raise ValueError('unpacked scene image is missing or outside workspace')
            if path not in media:
                media.append(path)
    code = [Path(__file__), ROOT/'datasets/scripts/common/blender_masks.py', ROOT/'datasets/scripts/waste_sticker/source_inputs.py',
            ROOT/'datasets/scripts/common/config.py', ROOT/'datasets/scripts/common/paths.py',
            ROOT/'datasets/scripts/common/blender_workspace.py',
            ROOT/'datasets/scripts/waste_sticker/configure_blender_orbit.py']
    with generation_guard('waste_sticker.arena:' + out.relative_to(ROOT).as_posix(), media, [out],
          {'seed': args.seed, 'samples': args.samples, 'max_seconds': args.max_seconds, 'batch_index': args.batch_index,
           'blender_version': bpy.app.version_string}, code_paths=code,
          procedural_inputs=procedural, root=ROOT, production_plan=declaration,
          sample_output_paths=[out / 'images'], auxiliary_output_paths=[out / 'masks']) as run:
        if not run.reused:
            if args.orbit_frame is not None:
                schedule = {row['frame']: {item['bin_color']: item['class_id'] for item in row['bins']}
                            for row in source_plan['samples']}
                render_orbit(out, run, bpy.context.scene, schedule, args.samples, args.seed, args.max_seconds)
            else:
                run_render(out, run, objects, scene_ref, source_plan, args.seed, args.samples, args.max_seconds)


if __name__ == '__main__':
    main()
