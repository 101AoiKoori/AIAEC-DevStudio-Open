"""Compile enlarged low-camera, near/far, occlusion and mirror sample states."""
import copy
import json
import random
import sys
import cv2
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.config import load_task_config
from datasets.scripts.common.frame_environment import query_frames
from datasets.scripts.common.design_context import design_context


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def occlusion_relations(objects, minimum=.05):
    return {front['class']+'_over_'+rear['class']
            for rear in objects for front in rear.get('occluded_by', [])
            if front['kind'] == 'vehicle' and front['fraction'] >= minimum}


def main():
    cv2.setNumThreads(1)
    _, config = load_task_config('e_bike')
    recipe = json.loads((ROOT/config['sample_recipe']).read_text(encoding='utf-8'))
    rng = random.Random(recipe['seed'])
    reviews = query_frames(task='e_bike', role='composite_background', split='train')
    backgrounds = []
    for requested in recipe['backgrounds']:
        match = requested.get('match',requested)
        matches = [r for r in reviews if all(r.get(k) == v for k,v in match.items())]
        if len(matches) != 1:
            raise ValueError('background selection must match exactly one current review: '+str(requested))
        r = matches[0]
        backgrounds.append({'id': f'low_{len(backgrounds):02d}', 'image': r['image'],
                            'parent_image': r['image'],
                            'source_video': r.get('parent_video') or r.get('source_blend') or r['parent_group'],
                            'parent_group': r['parent_group'],
                            'scene': r.get('scene'), 'route': r.get('route'),
                            'location': 'street_or_parking_floor_approximation',
                            'ground_zone': {'poly': r['geometry']['ground_polygon']},
                            'source_kind': r.get('source_kind'),
                            'layout': requested.get('layout',{}),
                            'scale_basis': recipe['scale_basis']})
    dump(ROOT/config['ground_background_manifest'], {'backgrounds': backgrounds})
    from datasets.scripts.e_bike import compose_v4_grounded_train as ground
    backgrounds = ground.bind_background_reviews(backgrounds, 'train', config)
    _, vehicles = ground.bike.sources(selection=ROOT/config['ground_vehicle_selection'], include_people=False)
    states, coverage, failures = [], [], Counter()

    def spec(x,y,size,mirror=False):
        return {'foot_x': x, 'foot_y': y, 'size_px': size, 'yaw':0, 'pitch':0,
                'roll':0, 'relative_exposure':1., 'mirrored':mirror}

    def state(bg, targets, actors, tone, index):
        used = {bg[k] for k in ('image','parent_image','source_video')}
        for target in targets:
            entry = next(e for e,_ in vehicles[target['class']] if e['path']==target['source'])
            used.update(entry[k] for k in ('path','source_frame','source_video','source_image') if k in entry)
        gain,sat = ground.TONES[tone]
        return {'id': f'low_{index:06d}', 'failure_event':'low_angle_count', 'min_count':1,'max_count':1,
                'prior':{'alpha':1,'beta':1}, 'exposure_weight':1, 'failure_cost':1, 'conditions':[],
                'background': {'id':bg['id'],'image':bg['image'],
                               **ground.background_review_reference(bg['_frame_reviews']['composite_background' if targets else 'empty_negative'])},
                'target':{'vehicles':targets}, 'labels':ground.GROUND_LABELS,
                'augmentation':{'tone':tone}, 'parameters':{'actors':actors,'value_gain':gain,'saturation_gain':sat},
                'sources':sorted(used)}

    # A paired mirrored layout uses the same source views/tone and reflected actor positions.
    for branch, pairs in recipe['paired_layouts'].items():
        for pair_index in range(pairs):
            result = None
            for attempt in range(recipe['maximum_attempts_per_pair']):
                if attempt and attempt % 100 == 0:
                    print(json.dumps({'branch':branch,'pair':pair_index,'attempt':attempt,'rejections':dict(failures)}),flush=True)
                variant=recipe.get('queue_variants',{}).get(branch)
                is_queue=branch=='mixed_queue' or branch.startswith('queue_all_') or variant is not None
                available = [b for b in backgrounds if (branch!='mixed_dense' or b['layout'].get('max_vehicles',6)>=6)
                             and (not is_queue or b['layout'].get('queue'))]
                if not available:
                    raise ValueError('No background layout for '+branch)
                bg = available[(pair_index+attempt//100) % len(available)]
                layout = bg['layout']
                if branch == 'mixed_dense' and layout.get('max_vehicles',6)<6:
                    continue
                tone = 'mild' if pair_index % 3 == 2 else 'original'
                if is_queue:
                    positions = ([layout['queue'][i] for i in variant['slots']] if variant else layout['queue'])
                    classes = (list(variant['classes']) if variant else
                               [branch.removeprefix('queue_all_')]*len(positions) if branch.startswith('queue_all_')
                               else ['upright','fallen']*(len(positions)//2))
                    if len(classes)!=len(positions) or set(classes)-{'upright','fallen'}:
                        raise ValueError('Invalid queue posture assignment: '+branch)
                    rng.shuffle(classes)
                elif branch == 'mixed_clear':
                    classes = ['upright','fallen']
                    rng.shuffle(classes)
                    positions = layout.get('clear',[(145,345,60),(475,450,155)])
                elif branch == 'mixed_dense':
                    classes = ['upright','fallen']*3
                    rng.shuffle(classes)
                    positions = [(150,330,55),(310,345,70),(500,370,90),
                                 (80,410,115),(280,450,155),(520,450,155)]
                elif branch.startswith('single_'):
                    classes = [branch.removeprefix('single_')]
                    positions = [layout.get('near',[rng.randint(260,380),445,rng.choice(recipe['near_sizes'])])]
                else:
                    front,rear = branch.split('_over_')
                    classes = [rear, front, 'fallen' if rear == 'upright' else 'upright',
                               'upright' if front == 'fallen' else 'fallen']
                    cx,cy,rear_size = layout.get('occlusion',[rng.randint(235,310),398,95])
                    positions = [(cx,cy,rear_size),(cx+rng.choice((-1,1))*rng.randint(25,95),cy+15,round(rear_size*1.18))]
                    if layout.get('max_vehicles',6)>2:
                        positions += [(90,335,60),(510,450,155)]
                    else:
                        classes = classes[:2]
                targets, actors = [], []
                for cls,(x,y,size) in zip(classes,positions):
                    entry,_ = rng.choice(vehicles[cls])
                    # Fallen widths and upright heights are different projection measures.
                    size = round(size*recipe['fallen_width_multiplier']) if cls=='fallen' else size
                    targets.append({'class':cls,'source':entry['path']})
                    actors.append(spec(x,y,size))
                original = state(bg,targets,actors,tone,len(states)+1)
                mirrored = copy.deepcopy(original)
                mirrored['id'] = f'low_{len(states)+2:06d}'
                for actor in mirrored['parameters']['actors']:
                    if layout.get('mirror_positions',True):
                        actor['foot_x'] = 640-actor['foot_x']
                    actor['mirrored'] = True
                try:
                    rendered = [ground.render_ground_state(s,[bg],vehicles,render_rgb=False) for s in (original,mirrored)]
                    relations = [occlusion_relations(r[1],recipe['minimum_effective_occlusion']) for r in rendered]
                    if '_over_' in branch and any(branch not in r for r in relations):
                        failures['requested_effective_occlusion_absent'] += 1
                        continue
                    if branch=='mixed_clear' and any(any(o['visible_fraction']<.999 for o in r[1]) for r in rendered):
                        failures['clear_layout_has_overlap'] += 1
                        continue
                except ValueError as exc:
                    failures[str(exc)] += 1
                    continue
                result = original,mirrored,rendered
                break
            if result is None:
                raise ValueError(f'No feasible {branch} pair {pair_index}: {dict(failures)}')
            for s,r,mirror in zip(result[:2],result[2],(False,True)):
                states.append(s)
                coverage.append({'state_id':s['id'],'branch':branch,'mirror_pair':f'{branch}_{pair_index}',
                                 'mirrored':mirror,'background_id':bg['id'],
                                 'actual_relations':sorted(occlusion_relations(r[1],recipe['minimum_effective_occlusion']))})
            print(json.dumps({'branch':branch,'pair':pair_index,'images':len(states)}),flush=True)
    # Empty controls are current full-frame absence decisions, not erased positive scenes.
    for bg in backgrounds[:recipe['empty_images']]:
        s=state(bg,[],[],'original',len(states)+1);states.append(s)
        coverage.append({'state_id':s['id'],'branch':'empty','background_id':bg['id'],'mirrored':False})
    plan = {'schema_version':1,'model_key':'e_bike','target_split':'train',
            'purpose':'Enlarged low-angle counting sample with real intermediate orbit views and paired mirrors.',
            'budget':len(states),'uncertainty_weight':0,
            'independence_basis':'Two foreground videos and one Blender project. Mirrors/views are derivatives.',
            'constraints':ground.GROUND_CONSTRAINTS,'strata':states,
            'sampling':{'root_seed':recipe['seed'],'recipe':config['sample_recipe'],
                        'coverage':coverage,'scale_basis':recipe['scale_basis']},
            'allocation_evidence':{'images_by_branch':dict(Counter(c['branch'] for c in coverage)),
                                   'placement_rejections':dict(failures)},
            'design_context':design_context('e_bike')}
    ground.validate_ground_plan(plan,backgrounds,vehicles,ground.build_queue(ROOT))
    dump(ROOT/config['production_plan'],plan)
    print(json.dumps({'images':len(states),'branches':plan['allocation_evidence']['images_by_branch'],
                      'rejections':dict(failures)},ensure_ascii=False))


if __name__ == '__main__':
    main()
