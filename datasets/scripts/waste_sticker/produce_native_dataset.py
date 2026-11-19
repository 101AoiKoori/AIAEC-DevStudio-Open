"""Render the native Waste Stickers branch into the current full-batch staging area."""
import argparse
from collections import Counter
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import random
import sys
import time

import bpy

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.blender_masks import GEOMETRY_TYPES, configure_device, targets_inside_frame
from datasets.scripts.common.blender_workspace import STICKERS, ORBIT_VIEWS, WORKSPACE_BLEND
from datasets.scripts.waste_sticker.configure_blender_orbit import (
    STICKER_SCENE, camera_name, sticker_bin_objects, sticker_objects,
)
from datasets.scripts.waste_sticker.render_sticker_arena_v4 import masks, set_bin_render_visibility

COLORS = ('blue', 'red', 'green', 'gray')
QUOTAS = {
    'train': {0: 180, 1: 202, 2: 203, 3: 202, 4: 203},
    'val': {0: 24, 1: 34, 2: 34, 3: 34, 4: 34},
    'test': {0: 24, 1: 14, 2: 54, 3: 54, 4: 54},
}
ANIMATION = ROOT / 'lab/features/waste_bin/training/config/waste_sticker_source_plan_20261114.json'


def compile_plan():
    animation = json.loads(ANIMATION.read_text(encoding='utf8'))
    schedule = {row['frame']: {item['bin_color']: item for item in row['bins']} for row in animation['samples']}
    pools = {}
    for count in (1, 2, 3, 4):
        pools[count] = [(frame, list(colors)) for frame in schedule for colors in combinations(COLORS, count)]
    plan = {'model_key': 'waste_sticker', 'source_blend': WORKSPACE_BLEND.relative_to(ROOT).as_posix(),
            'source_animation': ANIMATION.relative_to(ROOT).as_posix(), 'splits': {}}
    used_layouts = set()
    for split_index, (split, quota) in enumerate(QUOTAS.items()):
        rng = random.Random(2026111800 + split_index)
        counts = Counter({c: 0 for c in range(24)})
        target_counts = [count for count, number in quota.items() for _ in range(number)]
        rng.shuffle(target_counts)
        states = []
        for index, count in enumerate(target_counts):
            if count:
                # Choose several underrepresented fixed artworks together, then
                # use the corresponding saved frame for its placement animation.
                candidates = rng.sample(pools[count], min(400, len(pools[count])))
                frame, colors = min(candidates, key=lambda choice: sum(
                    (counts[schedule[choice[0]][color]['class_id']] + 1) ** 2
                    for color in choice[1]) + rng.random())
                class_ids = [schedule[frame][color]['class_id'] for color in colors]
                counts.update(class_ids)
            else:
                frame = rng.randint(1, 240)
                visible_count = (1, 2, 3, 4)[index % 4]
                colors = rng.sample(COLORS, visible_count)
                class_ids = []
            view = ORBIT_VIEWS[rng.randrange(len(ORBIT_VIEWS))][0]
            if count:
                views = [view] + [pose[0] for pose in ORBIT_VIEWS if pose[0] != view]
                for alternate in views:
                    layout = (frame, alternate, tuple(sorted(colors)))
                    if layout not in used_layouts:
                        used_layouts.add(layout)
                        view = alternate
                        break
                else:
                    raise ValueError('All saved views of this sticker layout have been allocated')
            states.append(dict(id=f'native_{split}_{index:05d}', frame=frame,
                visible_bins=colors, class_ids=class_ids, negative=not count, view=view,
                exposure_ev=round(rng.uniform(-.23, .18), 4),
                key_gain=round(rng.uniform(.86, 1.12), 4), seed=2026111800 + split_index*2000 + index))
        plan['splits'][split] = states
    return plan, schedule


def render(args):
    if Path(bpy.data.filepath).resolve() != WORKSPACE_BLEND:
        raise ValueError('Load the unified Blender asset before running the native batch')
    plan, schedule = compile_plan()
    plan_path = ROOT / 'datasets/scripts/waste_sticker/config/native_full_plan.json'
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    scene = bpy.data.scenes[STICKER_SCENE]
    bpy.context.window.scene = scene
    bpy.context.window.view_layer = scene.view_layers[0]
    array = bpy.context.view_layer.layer_collection.children.get(STICKERS)
    if array:
        array.exclude = array.collection.hide_render = False
        for group in array.children:
            group.exclude = group.collection.hide_render = False
    groups, papers = sticker_bin_objects(scene.objects), sticker_objects(scene)
    for frame, expected in schedule.items():
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        for color, item in expected.items():
            actual = papers[color].evaluated_get(depsgraph).data.polygons[0].material_index
            if actual != item['class_id']:
                raise ValueError(f'{frame}: {color} material class differs from the source animation')
    originally_hidden = {obj for group in groups.values() for obj in group if obj.hide_render}
    for marker in scene.timeline_markers:
        marker.camera = None
    scene.render.engine = args.engine
    if args.engine == 'CYCLES':
        configure_device(scene, 'OPTIX')
    elif hasattr(scene.eevee, 'taa_render_samples'):
        scene.eevee.taa_render_samples = args.eevee_samples
    scene.render.resolution_x, scene.render.resolution_y = 640, 480
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '8'
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.use_motion_blur = False
    scene.render.use_persistent_data = True
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 3
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 1
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 4
    key = scene.objects.get('Waste Stickers | Key')
    base_energy = key.data.energy if key else 0
    names = [None] * 24
    for row in schedule.values():
        for item in row.values():
            names[item['class_id']] = Path(item['source_artwork']).stem
    out = (ROOT / args.out).resolve()
    if not out.is_relative_to((ROOT / 'datasets/annotated/waste_bin/waste_sticker_full/native').resolve()):
        raise ValueError('Native staging must use datasets/annotated/waste_bin/waste_sticker_full/native')
    started = time.monotonic()
    source_hash = sha256(WORKSPACE_BLEND.read_bytes()).hexdigest()
    total = 0
    for split, states in plan['splits'].items():
        folder = out / split
        for sub in ('images', 'labels', 'masks'):
            (folder / sub).mkdir(parents=True, exist_ok=True)
        manifest_path = folder / 'manifest.json'
        previous = json.loads(manifest_path.read_text(encoding='utf8')) if manifest_path.is_file() else {}
        complete = {row['id']: row for row in previous.get('records', [])}
        rgb_path = folder / 'rgb_progress.json'
        rgb_complete = json.loads(rgb_path.read_text(encoding='utf8')) if rgb_path.is_file() else {}
        engine_path = folder / 'rgb_engines.json'
        rgb_engines = json.loads(engine_path.read_text(encoding='utf8')) if engine_path.is_file() else {}
        for identity, item in complete.items():
            if item.get('state_sha256'):
                rgb_complete.setdefault(identity, item['state_sha256'])
        records = []
        for state in states:
            stem = state['id']
            image_path, label_path = folder / 'images' / f'{stem}.png', folder / 'labels' / f'{stem}.txt'
            signature = sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
            if (stem in complete and complete[stem].get('state_sha256') == signature
                    and rgb_complete.get(stem) == signature
                    and image_path.is_file() and label_path.is_file()):
                item = complete[stem]
                item['render_engine'] = rgb_engines.get(stem, item.get('render_engine', 'CYCLES'))
                item['render_samples'] = args.samples if item['render_engine'] == 'CYCLES' else args.eevee_samples
                item['negative_kind'] = 'unmarked_bins' if state['negative'] else None
                records.append(item)
                continue
            scene.frame_set(state['frame'])
            colors = state['visible_bins']
            set_bin_render_visibility(groups, colors, originally_hidden)
            if state['negative']:
                for color in colors:
                    papers[color].hide_render = True
            camera = scene.objects[camera_name(state['view'])]
            scene.camera = camera
            scene.view_settings.exposure = state['exposure_ev']
            if key:
                key.data.energy = base_energy * state['key_gain']
            scene.cycles.seed = state['seed']
            bpy.context.view_layer.update()
            for color in colors:
                if not targets_inside_frame(scene, camera, [obj for obj in groups[color] if obj.type in GEOMETRY_TYPES]):
                    raise ValueError(f'{stem}: {color} bin falls outside the saved camera')
            if rgb_complete.get(stem) != signature or not image_path.is_file():
                scene.render.filepath = str(image_path)
                bpy.ops.render.render(write_still=True)
                rgb_complete[stem] = signature
                rgb_engines[stem] = args.engine
                rgb_path.write_text(json.dumps(rgb_complete, indent=2)+'\n', encoding='utf8')
                engine_path.write_text(json.dumps(rgb_engines, indent=2)+'\n', encoding='utf8')
            if args.rgb_only:
                print('NATIVE_RGB_PROGRESS', split, len(rgb_complete), '/', len(states), 'elapsed', round(time.monotonic()-started, 1), flush=True)
                total += 1
                if args.limit and total >= args.limit:
                    return
                continue
            instances, lines = [], []
            if not state['negative']:
                mask_path, instances = masks(scene, camera, {color: [papers[color]] for color in colors},
                                               folder, stem, 'paper')
                for instance in instances:
                    color = instance['bin']
                    if not instance['label_eligible']:
                        raise ValueError(f'{stem}: invalid paper visibility {instance}')
                    item = schedule[state['frame']][color]
                    cls = item['class_id']
                    x0, y0, x1, y1 = instance['candidate_xyxy_normalized']
                    lines.append(f'{cls} {(x0+x1)/2:.8f} {(y0+y1)/2:.8f} {x1-x0:.8f} {y1-y0:.8f}')
                    instance.update(class_id=cls, source_artwork=item['source_artwork'],
                        bbox_xyxy=[round(x0*640), round(y0*480), round(x1*640), round(y1*480)],
                        placement=dict(location=list(papers[color].location), scale=list(papers[color].scale),
                                       rotation=list(papers[color].rotation_quaternion)))
            label_path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf8')
            record = dict(id=stem, branch='blender_native', split=split,
                image=image_path.relative_to(folder).as_posix(), label=label_path.relative_to(folder).as_posix(),
                class_ids=state['class_ids'], instances=instances, selected_bins=colors, visible_bins=colors,
                visible_sticker_count=len(instances), source_parent=plan['source_blend']+'#'+STICKER_SCENE,
                frame=state['frame'], seed=state['seed'], image_size=[640, 480], state_sha256=signature,
                camera=dict(name=camera.name, pitch_deg=camera.get('pitch_deg'),
                    azimuth_deg=camera.get('azimuth_deg'), distance=camera.get('distance'),
                    location=list(camera.location), lens_mm=camera.data.lens),
                exposure_ev=state['exposure_ev'], key_gain=state['key_gain'],
                render_engine=rgb_engines.get(stem, 'CYCLES'),
                render_samples=args.samples if rgb_engines.get(stem, 'CYCLES') == 'CYCLES' else args.eevee_samples,
                negative_kind='unmarked_bins' if state['negative'] else None)
            if not state['negative']:
                record['visible_mask'] = mask_path
            records.append(record)
            manifest = dict(task='waste_sticker', split=split, branch='blender_native',
                status='complete' if len(records) == len(states) else 'in_progress',
                image_count=len(records), target_image_count=len(states), classes=names,
                source_parent=plan['source_blend']+'#'+STICKER_SCENE,
                source_blend_sha256=source_hash,
                plan=plan_path.relative_to(ROOT).as_posix(),
                render_settings={'CYCLES': {'samples': args.samples, 'denoising': True},
                                 'BLENDER_EEVEE': {'samples': args.eevee_samples}}, records=records)
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
            total += 1
            print('NATIVE_PROGRESS', split, len(records), '/', len(states), 'elapsed', round(time.monotonic()-started, 1), flush=True)
            if args.limit and total >= args.limit:
                return
        (folder/'classes.json').write_text(json.dumps(names, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print('NATIVE_COMPLETE', sum(len(states) for states in plan['splits'].values()), round(time.monotonic()-started, 1), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='datasets/annotated/waste_bin/waste_sticker_full/native')
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0, help='stop after this many new renders for an entry check')
    parser.add_argument('--rgb-only', action='store_true', help='fill RGB cache before a separate label pass')
    parser.add_argument('--engine', choices=('CYCLES', 'BLENDER_EEVEE'), default='CYCLES')
    parser.add_argument('--eevee-samples', type=int, default=128)
    render(parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []))
