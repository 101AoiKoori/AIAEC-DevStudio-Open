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
from datasets.scripts.waste_sticker.source_inputs import artwork_catalog, real_source_catalog
from datasets.scripts.common.paths import sample_dir, split_dir


def masks(scene, camera, parts, output, stem, tag, force_visible=()):
    return shared_masks(scene, camera, parts, output, stem, tag, force_visible,
                        palette={'blue': (1, 0, 0), 'red': (0, 1, 0),
                                 'green': (0, 0, 1), 'gray': (1, 1, 0)})

ASSET = 'datasets/assets/blender/waste_bin/waste_sticker_animated_v4.blend'
SOURCE_PLAN = 'lab/features/waste_bin/training/config/waste_sticker_source_plan_20261114.json'
COLORS = ('blue', 'red', 'green', 'gray')
CONSTRAINTS = {'resolution': [640, 480], 'labels': 'visible_paper_yolo_candidate',
               'min_visible_fraction': 0.70, 'min_visible_pixels': 24}


def validate(plan, scene):
    if plan.get('model_key') != 'waste_sticker' or plan.get('target_split') not in ('train', 'test', 'val') or plan.get('constraints') != CONSTRAINTS:
        raise ValueError('unsupported sticker arena plan')
    for state in plan['strata']:
        p = state['parameters']
        base_keys = {'frame', 'camera_location', 'camera_aim'}
        augmented_keys = base_keys | {'exposure_ev', 'saturation', 'blur_pixels', 'occluder_color', 'paper_width_ratio', 'paper_aspect'}
        if (set(p) != augmented_keys or type(p['frame']) is not int
                or not 1 <= p['frame'] <= 240 or state['sources'] != [scene, ASSET]):
            raise ValueError('invalid frame or asset declaration')
        for key in ('camera_location', 'camera_aim'):
            if not isinstance(p[key], list) or len(p[key]) != 3 or any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 100 for x in p[key]):
                raise ValueError('invalid camera vector')
        if set(p) == augmented_keys:
            if (type(p['exposure_ev']) not in (int, float) or not math.isfinite(p['exposure_ev']) or not -1 <= p['exposure_ev'] <= 1
                    or type(p['saturation']) not in (int, float) or not math.isfinite(p['saturation']) or not .5 <= p['saturation'] <= 1.5
                    or type(p['blur_pixels']) is not int or not 0 <= p['blur_pixels'] <= 2
                    or p['occluder_color'] not in (None, *COLORS)
                    or type(p['paper_width_ratio']) not in (int, float) or not .39 <= p['paper_width_ratio'] <= .46
                    or type(p['paper_aspect']) not in (int, float) or not .9 <= p['paper_aspect'] <= 1.04):
                raise ValueError('invalid bounded image augmentation')
            if state['augmentation'] != {k: p[k] for k in ('exposure_ev', 'saturation', 'blur_pixels', 'occluder_color', 'paper_width_ratio', 'paper_aspect')}:
                raise ValueError('augmentation declaration differs from execution parameters')
        if p['camera_location'] == p['camera_aim'] or state['max_count'] != 1:
            raise ValueError('invalid camera or duplicate state budget')
        if state['background'] != {'scene': scene} or state['target'] != {'asset': ASSET, 'frame': p['frame']}:
            raise ValueError('target does not match scene and animation')
        if state['labels'] != {'format': 'yolo_detection', 'visibility': 'visible_paper_candidate'}:
            raise ValueError('invalid label contract')
    if not sum(s['min_count'] for s in plan['strata']) <= plan['budget'] <= sum(s['max_count'] for s in plan['strata']):
        raise ValueError('budget is outside state bounds')


def run_render(out, plan, objects, scene_path, source_plan, seed, samples, max_seconds):
    scene = bpy.context.scene
    for obj in objects:
        scene.collection.objects.link(obj)
    bpy.context.view_layer.update()
    root = bpy.data.objects['Measured height scale']
    root.scale = (.1, .1, .1)
    root.location = (0, 1, 0)
    papers = {color: bpy.data.objects[color + ' item sticker paper'] for color in COLORS}
    base_vertices = {color: [v.co.copy() for v in paper.data.vertices] for color, paper in papers.items()}
    markings = {color: [bpy.data.objects[color + ' front marking'], bpy.data.objects[color + ' marking border']]
                for color in COLORS}
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
        for color, paper in papers.items():
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
            obj.hide_render = p.get('occluder_color') != color
        scene.view_settings.exposure = p.get('exposure_ev', 0.0)
        tone.inputs['Saturation'].default_value = p.get('saturation', 1.0)
        blur.inputs['Size'].default_value = (p.get('blur_pixels', 0), p.get('blur_pixels', 0))
        scene.render.use_compositing = True
        bpy.context.view_layer.update()
        for color, obj in papers.items():
            actual = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.polygons[0].material_index
            if actual != schedule[p['frame']][color]:
                raise ValueError('evaluated paper material differs from recorded class schedule: ' + color)
        stem = f'sticker_{index:05d}'
        image = out / 'images' / (stem + '.png')
        scene.render.filepath = str(image)
        bpy.ops.render.render(write_still=True)
        _, visibility = masks(scene, camera, {c: [papers[c]] for c in COLORS}, out, stem, 'paper')
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
                  'camera_location': p['camera_location'], 'camera_aim': p['camera_aim'],
                  'exposure_ev': p.get('exposure_ev', 0.0), 'saturation': p.get('saturation', 1.0),
                  'blur_pixels': p.get('blur_pixels', 0), 'occluder_color': p.get('occluder_color'),
                  'paper_width_ratio': p['paper_width_ratio'], 'paper_aspect': p['paper_aspect'],
                  'marking_shift_source_units': {color: markings[color][0].location.z - marking_locations[color][0].z
                                                 for color in COLORS}}
        plan.record_sample(state['id'], output=image, sources=state['sources'], parameters=p, actual=actual)
        records.append({'image': image.relative_to(ROOT).as_posix(), 'label_candidate': label.relative_to(ROOT).as_posix(),
                        'state_id': state['id'], 'actual': actual})
    (out / 'manifest.json').write_text(json.dumps({'status': 'candidate', 'seed': seed, 'records': records}, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def main():
    global ASSET, SOURCE_PLAN
    ASSET = source_path('waste_sticker', 'animated_asset').relative_to(ROOT).as_posix()
    SOURCE_PLAN = source_path('waste_sticker', 'animated_source_plan').relative_to(ROOT).as_posix()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scene', default=source_path('waste_sticker', 'arena_scene').relative_to(ROOT).as_posix())
    ap.add_argument('--plan', required=True)
    ap.add_argument('--batch-index', type=int, help='select a batch from a multi-batch plan')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=20261114)
    ap.add_argument('--samples', type=int, default=4)
    ap.add_argument('--max-seconds', type=int, default=360)
    args = ap.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    scene, plan_path, out = [(ROOT / x).resolve() for x in (args.scene, args.plan, args.out)]
    if (not scene.is_relative_to(ROOT) or not plan_path.is_relative_to(ROOT) or not scene.is_file() or not plan_path.is_file()
            or Path(bpy.data.filepath).resolve() != scene or not 1 <= args.samples <= 8 or not 20 <= args.max_seconds <= 360):
        raise ValueError('load the declared workspace scene and plan with samples in 1..8')
    if out not in {sample_dir('waste_sticker').resolve(), *[split_dir('waste_sticker', split).resolve() for split in ('train', 'test', 'val')]} or (out.exists() and any(out.iterdir())):
        raise ValueError('output must be the empty waste_sticker sample or split directory')
    scene_ref = scene.relative_to(ROOT).as_posix()
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
    if scene != source_path('waste_sticker', 'arena_scene'):
        raise ValueError('loaded scene differs from waste_sticker source.yaml')
    if out not in {sample_dir('waste_sticker').resolve(), split_dir('waste_sticker', declaration['target_split']).resolve()}:
        raise ValueError('output split differs from the plan')
    validate(declaration, scene_ref)
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
    procedural = [scene, ROOT / ASSET, ROOT / SOURCE_PLAN, plan_path, ROOT / source_plan['parent_blend'],
                  ROOT/source_config['real_annotations'], *[ROOT/p for p in source_config['records']],
                  ROOT/'datasets/scripts/waste_sticker/config/source.yaml',
                  ROOT/'datasets/scripts/waste_sticker/config/dataset.yaml']
    with bpy.data.libraries.load(str(ROOT / ASSET), link=False) as (src, dst):
        dst.objects = [n for n in src.objects if n == 'Measured height scale' or any(n.startswith(c + ' ') for c in COLORS)]
    objects = list(dst.objects)
    for im in bpy.data.images:
        if im.source == 'FILE' and not im.packed_file:
            path = Path(bpy.path.abspath(im.filepath, library=im.library)).resolve()
            if not path.is_file() or not path.is_relative_to(ROOT):
                raise ValueError('unpacked scene image is missing or outside workspace')
            if path not in media:
                media.append(path)
    code = [Path(__file__), ROOT/'datasets/scripts/common/blender_masks.py', ROOT/'datasets/scripts/waste_sticker/source_inputs.py',
            ROOT/'datasets/scripts/common/config.py', ROOT/'datasets/scripts/common/paths.py']
    with generation_guard('waste_sticker.arena:' + out.relative_to(ROOT).as_posix(), media, [out],
          {'seed': args.seed, 'samples': args.samples, 'max_seconds': args.max_seconds, 'batch_index': args.batch_index,
           'blender_version': bpy.app.version_string}, code_paths=code,
          procedural_inputs=procedural, root=ROOT, production_plan=declaration,
          sample_output_paths=[out / 'images'], auxiliary_output_paths=[out / 'masks']) as run:
        if not run.reused:
            run_render(out, run, objects, scene_ref, source_plan, args.seed, args.samples, args.max_seconds)


if __name__ == '__main__':
    main()
