"""Independent building-name rendering; no other task generator is imported.

CLI requires a scene already loaded by Blender and an explicit Bayesian plan.
Import and plan construction are file-free; rendering only happens in main().
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path
from hashlib import file_digest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.config import load_task_config, source_path
from datasets.scripts.common.names import make_name
from datasets.scripts.common.paths import building_name_sample_work_dir, sample_dir, split_dir
from datasets.scripts.common.bayesian import plan_production
from datasets.scripts.building_name_paper.configure_blender_scene import set_camera_pose
from lab.training_common.production_guard import generation_guard

TASK = 'building_name_paper'
CONSTRAINTS = {'resolution': [640, 480], 'labels': 'visible_board_and_generated_text',
               'min_visible_fraction': .70, 'min_visible_pixels': 24}
LABELS = {'class_id': 0, 'detection': 'visible_paper_yolo', 'text': 'generated_string'}


def sources():
    return [source_path(TASK, key).relative_to(ROOT).as_posix() for key in ('blender_scene', 'font')]


def build_plan(quotas, seed=20261114):
    """Turn explicit single/multiple/negative budgets into independent name states."""
    if set(quotas) != {'single', 'multiple', 'negative'} or any(type(n) is not int or n < 0 for n in quotas.values()):
        raise ValueError('single/multiple/negative nonnegative integer quotas required')
    if not 1 <= sum(quotas.values()) <= 10000:
        raise ValueError('budget must be 1..10000')
    _, config = load_task_config(TASK)
    keys = list(config['blender_collections'])
    if set(keys) != {'A', 'B', 'C'}:
        raise ValueError('the configured scene requires the three A/B/C building collections')
    refs = sources()
    rng = random.Random(seed)
    states = []
    for branch, count in quotas.items():
        for index in range(count):
            active = [keys[index % len(keys)]] if branch != 'multiple' else [keys[index % len(keys)], keys[(index+1) % len(keys)]]
            texts = {} if branch == 'negative' else {key: make_name(rng.randrange(10000)) for key in active}
            states.append({'id': f'{branch}_{index:06d}', 'failure_event': 'name_board_missed',
                'min_count': 1, 'max_count': 1, 'prior': {'alpha': 1, 'beta': 1},
                'exposure_weight': 1, 'failure_cost': 1, 'conditions': [],
                'background': {'scene': refs[0], 'buildings': active},
                'target': {'texts': texts}, 'labels': dict(LABELS),
                'augmentation': {'exposure_ev': rng.uniform(-.7, .7)},
                'parameters': {'yaw_degrees': rng.uniform(-22, 22), 'pitch_degrees': rng.uniform(-8, 12),
                               'distance_m': rng.uniform(.85, 1.1) if len(active)==1 else rng.uniform(1.4, 1.7),
                               'lens_mm': 42, 'samples': 16}, 'sources': refs})
    rng.shuffle(states)
    result = {'schema_version': 1, 'model_key': TASK, 'target_split': 'train',
        'purpose': 'Independent name-paper scene budget; design priors, not measured frequencies.',
        'budget': len(states), 'uncertainty_weight': 0, 'independence_basis': 'shared_scene_different_designed_states',
        'constraints': dict(CONSTRAINTS), 'strata': states}
    validate_plan(result)
    return result


def build_detection_plan(repeats, seed):
    """Cover paper presence independently of the number and identity of buildings."""
    if not 1 <= repeats <= 20:
        raise ValueError('detection plan requires 1..20 repetitions')
    rng = random.Random(seed)
    layouts = ((1, 1), (2, 1), (2, 2), (3, 3), (1, 0), (3, 0))
    _, config = load_task_config(TASK)
    view = config['blender_view']
    result = build_plan({'single': 18 * repeats, 'multiple': 0, 'negative': 0}, seed)
    scene_source = result['strata'][0]['sources'][0]
    result['purpose'] = 'Paper detection coverage with physical rooftop offsets and planned lighting.'
    for index, state in enumerate(result['strata']):
        repeat, variant = divmod(index, 18)
        rotation, layout = divmod(variant, len(layouts))
        building_count, paper_count = layouts[layout]
        active = ['ABC'[(rotation + offset) % 3] for offset in range(building_count)]
        # Reuse names across adjacent buildings within a repeat and change each repeat.
        texts = {key: make_name(104 + (seed + 211 * repeat + 41 * (layout // 2)) % 7850)
                 for key in active[:paper_count]}
        lighting = ('normal', 'dim', 'normal')[(layout + rotation + repeat) % 3]
        spec = view['single' if building_count == 1 else 'global']
        band = (layout // 2 + rotation + repeat) % len(spec['radius_m'])
        step = (1, 47, 81)[(layout + rotation + repeat) % 3]
        fraction = (step - 1) / (view['frames_per_sweep'] - 1)
        fraction = fraction * fraction * (3 - 2 * fraction)
        if band % 2:
            fraction = 1 - fraction
        left, right = spec['azimuth_degrees']
        yaw = left + fraction * (right - left)
        frame = band * (view['frames_per_sweep'] + view['transition_frames']) + step
        state.update(
            id=f'detect_{repeat:02d}_{rotation}_{layout}',
            background={'scene': scene_source, 'buildings': active}, target={'texts': texts},
            augmentation={'exposure_ev': rng.uniform(-.95, -.72) if lighting == 'dim' else rng.uniform(-.16, .40)},
            parameters={'yaw_degrees': yaw, 'pitch_degrees': math.degrees(math.asin((spec['camera_height_m'][band] - view['target_height_m']) / spec['radius_m'][band])),
                        'distance_m': spec['radius_m'][band], 'lens_mm': spec['lens_mm'], 'samples': 12,
                        'paper_offset_m': (-.026, 0, .026)[(rotation + layout + repeat) % 3],
                        'camera_frame': frame})
    return validate_plan(result)


def validate_plan(plan):
    _, config = load_task_config(TASK)
    keys, refs = set(config['blender_collections']), sources()
    if plan.get('model_key') != TASK or plan.get('target_split') not in {'train','test','val'} or plan.get('constraints') != CONSTRAINTS:
        raise ValueError('unsupported building-name plan')
    states = plan.get('strata')
    if not isinstance(states, list) or not states or type(plan.get('budget')) is not int or not 1 <= plan['budget'] <= 10000:
        raise ValueError('explicit states and a budget of 1..10000 are required')
    ids = set()
    for state in states:
        bg, target, aug, params = [state[k] for k in ('background','target','augmentation','parameters')]
        if state['id'] in ids or state['sources'] != refs or state['conditions'] != []:
            raise ValueError('invalid identity, sources, conditions or quota')
        if (type(state.get('min_count')) is not int or type(state.get('max_count')) is not int or
                not 0 <= state['min_count'] <= state['max_count'] <= 10000):
            raise ValueError('state quotas must satisfy 0 <= min_count <= max_count <= 10000')
        ids.add(state['id'])
        if set(bg) != {'scene','buildings'} or bg['scene'] != refs[0] or not isinstance(bg['buildings'],list) or not 1 <= len(bg['buildings']) <= 3 or len(set(bg['buildings'])) != len(bg['buildings']) or not set(bg['buildings']) <= keys:
            raise ValueError('invalid scene/building selection')
        if set(target) != {'texts'} or not isinstance(target['texts'],dict) or not set(target['texts']) <= set(bg['buildings']):
            raise ValueError('text targets must belong to selected buildings')
        if any(not isinstance(t,str) or not 1 <= len(t) <= 12 or any(ch.isspace() for ch in t) for t in target['texts'].values()):
            raise ValueError('text must be 1..12 non-whitespace characters')
        camera_keys = {'yaw_degrees','pitch_degrees','distance_m','lens_mm','samples'}
        if (state['labels'] != LABELS or set(aug) - {'paper_glare'} != {'exposure_ev'} or
                set(params) - {'paper_offset_m', 'camera_frame'} != camera_keys):
            raise ValueError('unsupported label or rendering fields')
        if type(aug.get('paper_glare', False)) is not bool or (aug.get('paper_glare') and not target['texts']):
            raise ValueError('paper glare requires at least one paper target')
        view = config['blender_view']
        frame_count = 3 * view['frames_per_sweep'] + 2 * view['transition_frames']
        if 'camera_frame' in params and (type(params['camera_frame']) is not int or not 1 <= params['camera_frame'] <= frame_count):
            raise ValueError('camera frame outside the configured track')
        yaw_bounds = view['single' if len(bg['buildings']) == 1 else 'global']['azimuth_degrees']
        for value, lo, hi in [(aug['exposure_ev'],-1,1),(params['yaw_degrees'],*yaw_bounds),(params['pitch_degrees'],-15,35),(params['distance_m'],.5,2.5),(params['lens_mm'],25,65),
                              (params.get('paper_offset_m', 0), -.03, .03)]:
            if type(value) not in (int,float) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError('rendering parameter outside bounds')
        if type(params['samples']) is not int or not 1 <= params['samples'] <= 64:
            raise ValueError('samples must be 1..64')
    if not sum(s['min_count'] for s in states) <= plan['budget'] <= sum(s['max_count'] for s in states):
        raise ValueError('budget is outside feasible state quotas')
    return plan


def _render(out, images, labels_dir, run, device, font, scene_digest):
    import bpy
    from mathutils import Vector
    from datasets.scripts.common.blender_masks import configure_device, masks
    _, config = load_task_config(TASK)
    scene = bpy.context.scene
    if scene.camera is None:
        raise ValueError('source scene requires a camera')
    groups = {}
    for key, name in config['blender_collections'].items():
        collection = bpy.data.collections.get(name)
        if collection is None:
            raise ValueError('missing building collection: '+name)
        boards = [o for o in collection.all_objects if o.name.startswith('Name paper - ')]
        texts = [o for o in collection.all_objects if o.type=='FONT' and o.name.startswith('Editable text - ')]
        if len(boards)!=1 or len(texts)!=1:
            raise ValueError('each building requires one board and one editable text')
        board, text = boards[0], texts[0]
        groups[key] = (collection, board, text, board.location.x, text.location.x, text.scale.copy())
    # Fire artwork is not a name target; hide the original scene's fire papers.
    for obj in scene.objects:
        if obj.name.startswith(('Fire paper ', 'Candidate fire ')):
            obj.hide_render = True
    scene.render.engine = 'CYCLES'
    devices = configure_device(scene, device)
    scene.render.resolution_x, scene.render.resolution_y = CONSTRAINTS['resolution']
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = False
    for folder in (images, labels_dir, out / 'masks'):
        folder.mkdir(parents=True, exist_ok=True)
    records = []
    schedule = [state for state in run.plan['strata'] for _ in range(state['planned_count'])]
    cameras = {o['building_keys']: o for o in scene.objects if o.type == 'CAMERA' and 'building_keys' in o}
    actions = {key: obj.animation_data.action for key, obj in cameras.items()}
    for number, state in enumerate(schedule):
        active, texts = state['background']['buildings'], state['target']['texts']
        p = state['parameters']
        offset = p.get('paper_offset_m', 0)
        parts = {}
        for key,(collection,board,text,board_x,text_x,scale) in groups.items():
            collection.hide_render = key not in active
            for obj in list(collection.all_objects):
                obj.hide_render = key not in active or obj.name.startswith(('Fire paper ','Candidate fire '))
            board.hide_render = text.hide_render = key not in texts
            if key in texts:
                board.location.x = board_x + offset
                text.location.x = text_x + offset
                text.data.body, text.data.font = texts[key], font
                text.scale = scale
                bpy.context.view_layer.update()
                if text.dimensions.x > .115:
                    text.scale *= .115/text.dimensions.x
                parts[key] = [board,text]
        center = sum(groups[k][4] for k in active)/len(active)
        distance = p['distance_m']
        view = config['blender_view']
        target_height = view['target_height_m']
        aim = Vector((center,0,target_height))
        key = active[0] if len(active) == 1 else 'ABC'
        scene.camera = cameras[key]
        # Each sample must control the pose without a saved animation overriding it.
        if 'camera_frame' in p:
            scene.camera.animation_data.action = actions[key]
            scene.frame_set(p['camera_frame'])
            scene.camera.data.lens = p['lens_mm']
        else:
            scene.camera.animation_data.action = None
            spec = view['single' if len(active) == 1 else 'global']
            heights = spec['camera_height_m']
            height = min(max(heights), max(min(heights), target_height + distance * math.sin(math.radians(p['pitch_degrees']))))
            elevation = math.degrees(math.asin((height - target_height) / distance))
            yaw = min(spec['azimuth_degrees'][1], max(spec['azimuth_degrees'][0], p['yaw_degrees']))
            set_camera_pose(scene.camera, aim, yaw, elevation, distance, p['lens_mm'])
        scene.cycles.samples = p['samples']
        scene.view_settings.exposure = state['augmentation']['exposure_ev']
        bpy.context.view_layer.update()
        stem = f'name_{number:06d}'
        image = images/f'{stem}.png'
        scene.render.filepath = str(image)
        bpy.ops.render.render(write_still=True)
        labels, truth = [], []
        visible_mask = None
        if parts:
            visible_mask, rows = masks(scene,scene.camera,parts,out,stem,'names',palette={'A':(1,0,0),'B':(0,1,0),'C':(0,0,1)})
            for row in rows:
                if not row['label_eligible']:
                    raise ValueError('name board does not meet declared visibility: '+str(row))
                x0,y0,x1,y1 = row['candidate_xyxy_normalized']
                labels.append(f'0 {(x0+x1)/2:.6f} {(y0+y1)/2:.6f} {x1-x0:.6f} {y1-y0:.6f}')
                truth.append({'building':row['bin'],'text':texts[row['bin']], 'bbox_xyxy_normalized':[x0,y0,x1,y1], 'visibility':row})
        label=labels_dir/f'{stem}.txt'
        label.write_text('\n'.join(labels)+('\n' if labels else ''),encoding='utf-8')
        actual={'image':image.relative_to(ROOT).as_posix(),'label':label.relative_to(ROOT).as_posix(),
                'objects':truth,'source_scene':state['background']['scene'],'devices':devices,
                'state_id':state['id'], 'buildings':active, 'parameters':p,
                'appearance':{'exposure_ev':state['augmentation']['exposure_ev'], 'lighting':'diffuse_world'},
                'camera':scene.camera.name, 'camera_matrix_world':[list(row) for row in scene.camera.matrix_world],
                'camera_height_m':float(scene.camera.location.z),
                'visible_mask':visible_mask, 'generation_method':'blender_render'}
        run.record_sample(state['id'],output=image,sources=state['sources'],parameters=p,actual=actual)
        records.append(actual)
    manifest = {'task': TASK, 'split': run.plan['target_split'], 'label_scope': 'paper_detection',
                'source_scene': 'Buildings', 'source_blend_sha256': scene_digest,
                'records': records}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    planning = parser.add_mutually_exclusive_group()
    planning.add_argument('--plan', help='external Blender plan, defaults to source.yaml')
    planning.add_argument('--detection-repeats', type=int, help='repeat the 18-state detection design')
    parser.add_argument('--seed', type=int, default=20261117, help='seed for the detection design')
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',choices=('OPTIX','CUDA','CPU'),default='OPTIX')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    import bpy
    plan_inputs = []
    if args.detection_repeats is None:
        plan_path = (ROOT / args.plan).resolve() if args.plan else source_path(TASK, 'blender_plan')
        if not plan_path.is_relative_to(ROOT) or not plan_path.is_file():
            raise ValueError('plan must be a workspace file')
        plan_inputs.append(plan_path)
        declaration = validate_plan(json.loads(plan_path.read_text(encoding='utf-8-sig')))
    else:
        declaration = build_detection_plan(args.detection_repeats, args.seed)
    source=source_path(TASK,'blender_scene')
    if Path(bpy.data.filepath).resolve()!=source:
        raise ValueError('load the source.yaml scene before invoking this renderer')
    out=(ROOT/args.output).resolve()
    sample, split = sample_dir(TASK), split_dir(TASK,declaration['target_split'])
    permitted = {sample, sample / 'blender', split, building_name_sample_work_dir() / 'blender'}
    if declaration['target_split'] == 'train':
        permitted.add(split / 'blender')
    if out not in permitted:
        raise ValueError('output must be the task sample or declared split directory')
    from datasets.scripts.common.blender_workspace import BUILDINGS, activate_scene
    activate_scene(BUILDINGS)
    font_source = source_path(TASK, 'font')
    font = next((item for item in bpy.data.fonts
                 if item.packed_file and Path(item.filepath).name == font_source.name), None)
    if font is None or bytes(font.packed_file.data) != font_source.read_bytes():
        raise ValueError('the packed Buildings font differs from the configured typeface')
    media=[]
    for image in bpy.data.images:
        if image.source=='FILE' and not image.packed_file:
            path=Path(bpy.path.abspath(image.filepath,library=image.library)).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file():
                raise ValueError('unpacked image is missing or outside workspace')
            media.append(path)
    configs=[ROOT/f'datasets/scripts/{TASK}/config/{name}.yaml' for name in ('dataset','source')]
    planned=plan_production(declaration,ROOT)
    images, labels = out / 'images', out / 'labels'
    outputs = [out]
    if out == split_dir(TASK, 'train') / 'blender':
        images, labels = split / 'images/blender', split / 'labels/blender'
        outputs += [images, labels]
    with source.open('rb') as stream:
        scene_digest = file_digest(stream, 'sha256').hexdigest()
    with generation_guard('building_name.blender:'+out.relative_to(ROOT).as_posix(),media,outputs,vars(args),
            code_paths=[Path(__file__),ROOT/'datasets/scripts/common/blender_masks.py',ROOT/'datasets/scripts/common/bayesian.py',
                        ROOT/'datasets/scripts/common/config.py',ROOT/'datasets/scripts/common/paths.py',ROOT/'datasets/scripts/common/names.py',
                        Path(__file__).with_name('configure_blender_scene.py'),
                        ROOT/'datasets/scripts/common/blender_workspace.py'],
            procedural_inputs=[*plan_inputs,source,font_source,*configs],root=ROOT,production_plan=declaration,
            sample_output_paths=[images],auxiliary_output_paths=[out/'masks']) as run:
        if run.plan!=planned:
            raise ValueError('guarded plan differs from precomputed allocation')
        if not run.reused:
            if any(path.exists() and any(path.iterdir()) for path in outputs):
                raise ValueError('output must be empty')
            _render(out,images,labels,run,args.device,font,scene_digest)


if __name__=='__main__':
    main()
