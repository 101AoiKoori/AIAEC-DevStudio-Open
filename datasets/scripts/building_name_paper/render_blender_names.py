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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.config import load_task_config, source_path
from datasets.scripts.common.names import make_name
from datasets.scripts.common.paths import sample_dir, split_dir
from datasets.scripts.common.bayesian import plan_production
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
        if state['labels'] != LABELS or set(aug) != {'exposure_ev'} or set(params) != {'yaw_degrees','pitch_degrees','distance_m','lens_mm','samples'}:
            raise ValueError('unsupported label or rendering fields')
        for value, lo, hi in [(aug['exposure_ev'],-1,1),(params['yaw_degrees'],-25,25),(params['pitch_degrees'],-15,20),(params['distance_m'],.5,2.5),(params['lens_mm'],25,65)]:
            if type(value) not in (int,float) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError('rendering parameter outside bounds')
        if type(params['samples']) is not int or not 1 <= params['samples'] <= 64:
            raise ValueError('samples must be 1..64')
    if not sum(s['min_count'] for s in states) <= plan['budget'] <= sum(s['max_count'] for s in states):
        raise ValueError('budget is outside feasible state quotas')
    return plan


def _render(out, run, device):
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
        groups[key] = (collection, boards[0], texts[0], float(texts[0].matrix_world.translation.x), tuple(texts[0].scale))
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
    font = bpy.data.fonts.load(str(source_path(TASK,'font')), check_existing=True)
    for folder in ('images','labels','masks'):
        (out/folder).mkdir(parents=True,exist_ok=True)
    records = []
    schedule = [state for state in run.plan['strata'] for _ in range(state['planned_count'])]
    for number, state in enumerate(schedule):
        active, texts = state['background']['buildings'], state['target']['texts']
        parts = {}
        for key,(collection,board,text,x,scale) in groups.items():
            collection.hide_render = key not in active
            for obj in collection.all_objects:
                obj.hide_render = key not in active or obj.name.startswith(('Fire paper ','Candidate fire '))
            board.hide_render = text.hide_render = key not in texts
            if key in texts:
                text.data.body, text.data.font = texts[key], font
                text.scale = scale
                bpy.context.view_layer.update()
                if text.dimensions.x > .115:
                    text.scale *= .115/text.dimensions.x
                parts[key] = [board,text]
        p = state['parameters']
        center = sum(groups[k][3] for k in active)/len(active)
        distance = p['distance_m']
        aim = Vector((center,0,.23))
        scene.camera.location = (center+distance*math.tan(math.radians(p['yaw_degrees'])),-distance,.23+distance*math.tan(math.radians(p['pitch_degrees'])))
        scene.camera.rotation_euler = (aim-scene.camera.location).to_track_quat('-Z','Y').to_euler()
        scene.camera.data.lens = p['lens_mm']
        scene.camera.data.clip_start = .001
        scene.camera.data.dof.use_dof = False
        scene.cycles.samples = p['samples']
        scene.view_settings.exposure = state['augmentation']['exposure_ev']
        bpy.context.view_layer.update()
        stem = f'name_{number:06d}'
        image = out/'images'/f'{stem}.png'
        scene.render.filepath = str(image)
        bpy.ops.render.render(write_still=True)
        labels, truth = [], []
        if parts:
            _, rows = masks(scene,scene.camera,parts,out,stem,'names',palette={'A':(1,0,0),'B':(0,1,0),'C':(0,0,1)})
            for row in rows:
                if not row['label_eligible']:
                    raise ValueError('name board does not meet declared visibility: '+str(row))
                x0,y0,x1,y1 = row['candidate_xyxy_normalized']
                labels.append(f'0 {(x0+x1)/2:.6f} {(y0+y1)/2:.6f} {x1-x0:.6f} {y1-y0:.6f}')
                truth.append({'building':row['bin'],'text':texts[row['bin']], 'bbox_xyxy_normalized':[x0,y0,x1,y1], 'visibility':row})
        label=out/'labels'/f'{stem}.txt'
        label.write_text('\n'.join(labels)+('\n' if labels else ''),encoding='utf-8')
        actual={'image':image.relative_to(ROOT).as_posix(),'label':label.relative_to(ROOT).as_posix(),
                'objects':truth,'source_scene':state['background']['scene'],'devices':devices,
                'text_truth':'generated_string; pixel readability unmeasured'}
        run.record_sample(state['id'],output=image,sources=state['sources'],parameters=p,actual=actual)
        records.append(actual)
    (out/'manifest.json').write_text(json.dumps({'task':TASK,'records':records},ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',default=str(source_path(TASK,'blender_plan').relative_to(ROOT)))
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',choices=('OPTIX','CUDA','CPU'),default='OPTIX')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    import bpy
    plan_path=(ROOT/args.plan).resolve()
    if not plan_path.is_relative_to(ROOT) or not plan_path.is_file():
        raise ValueError('plan must be a workspace file')
    declaration=validate_plan(json.loads(plan_path.read_text(encoding='utf-8-sig')))
    source=source_path(TASK,'blender_scene')
    if Path(bpy.data.filepath).resolve()!=source:
        raise ValueError('load the source.yaml scene before invoking this renderer')
    out=(ROOT/args.output).resolve()
    if out not in {sample_dir(TASK),split_dir(TASK,declaration['target_split'])}:
        raise ValueError('output must be the task sample or declared split directory')
    media=[]
    for image in bpy.data.images:
        if image.source=='FILE' and not image.packed_file:
            path=Path(bpy.path.abspath(image.filepath,library=image.library)).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file():
                raise ValueError('unpacked image is missing or outside workspace')
            media.append(path)
    configs=[ROOT/f'datasets/scripts/{TASK}/config/{name}.yaml' for name in ('dataset','source')]
    planned=plan_production(declaration,ROOT)
    with generation_guard('building_name.blender:'+out.relative_to(ROOT).as_posix(),media,[out],vars(args),
            code_paths=[Path(__file__),ROOT/'datasets/scripts/common/blender_masks.py',ROOT/'datasets/scripts/common/bayesian.py',
                        ROOT/'datasets/scripts/common/config.py',ROOT/'datasets/scripts/common/paths.py',ROOT/'datasets/scripts/common/names.py'],
            procedural_inputs=[plan_path,source,source_path(TASK,'font'),*configs],root=ROOT,production_plan=declaration,
            sample_output_paths=[out/'images'],auxiliary_output_paths=[out/'masks']) as run:
        if run.plan!=planned:
            raise ValueError('guarded plan differs from precomputed allocation')
        if not run.reused:
            if out.exists() and any(out.iterdir()):
                raise ValueError('output must be empty')
            _render(out,run,args.device)


if __name__=='__main__':
    main()
