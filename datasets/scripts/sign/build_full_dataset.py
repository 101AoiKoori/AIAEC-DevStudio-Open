"""Compile exact sign train/val quotas and consume the existing compositor."""
import argparse
import copy
import json
import hashlib
import math
import random
from collections import Counter, defaultdict, deque
from pathlib import Path

from datasets.scripts.common.config import load_task_config
from datasets.scripts.common.frame_environment import query_frames, edit_database, image_digest
from datasets.scripts.common.review.placement_record import decode_placement_record, contact_status_in_view
from datasets.scripts.sign import compose_sign_v4_measured as measured
from datasets.scripts.sign import plan_sign_sources as planner

ROOT = measured.ROOT


def shares(total, weights):
    """Round this batch's image-domain budget while keeping its exact sum."""
    raw = [total*w/sum(weights) for w in weights]
    result = [math.floor(v) for v in raw]
    for index in sorted(range(len(raw)),key=lambda i: (-(raw[i]-result[i]),i))[:total-sum(result)]:
        result[index] += 1
    return result


def current_path(path):
    if path is None or (ROOT/path).is_file(): return path
    moves = json.loads((ROOT/'datasets/raw_source_relocations.json').read_text(encoding='utf8'))['moves']
    for move in sorted(moves,key=lambda x:-len(x['original_path'])):
        old = move['original_path'].rstrip('/')
        if path==old or path.startswith(old+'/'):
            candidate = move['current_path'].rstrip('/')+path[len(old):]
            if (ROOT/candidate).is_file(): return candidate
    raise FileNotFoundError(path)


def empty_state(row, identity, tone, context, crop=None):
    parameters = {'kind':'empty_background','negative_context':context,
                  'scene_evidence':row['sign_absence_evidence']}
    if crop is not None: parameters['crop_xyxy'] = crop
    refs = {'image':row['image'],**{key:row[key] for key in ('frame_id','record_id','evidence_ids','placement_algorithm')}}
    return {'id':identity,'conditions':['negative','none',tone], 'min_count':1,'max_count':1,
        'prior':{'alpha':1,'beta':1},'exposure_weight':1,'failure_cost':1,'failure_event':'牌具漏检或误报',
        'background':refs,'target':{'state':'negative','mask':'none'},'augmentation':{'tone':tone},
        'labels':{'class_id':0,'kind':'retained_background_boxes' if row.get('existing_sign_boxes') else 'empty_candidate'},
        'parameters':parameters,'sources':list(dict.fromkeys([row['image'],row['parent_image']]+([row['parent_video']] if row.get('parent_video') else [])))}


def direct_rows(source, split, frames):
    by_id = {row['frame_id']:row for row in frames}
    results = []
    for review in query_frames(task='sign',role='direct_target',split=split,database=ROOT/source['databases']['frame_environment']):
        if review['frame_id'] in by_id:
            row = by_id[review['frame_id']]
        else:
            image = next(path for path in [review.get('source_image'),*review['paths']] if path and (ROOT/path).is_file())
            row = {'image':image,'parent_image':image,'parent_video':current_path(review.get('parent_video')),
                'source_group':review['parent_group'],'pts_seconds':review.get('timestamp_seconds'),
                'workspace_image':True,'crop_xyxy':None,'use_role':'direct_target',
                'frame_id':review['frame_id'],'record_id':review['record_id'],
                'evidence_ids':sorted(item['evidence_id'] for item in review['evidence']),
                'placement_algorithm':'direct_frame_labels','sign_absence':'existing_labelled',
                'sign_absence_evidence':f"frame_environment:{review['record_id']}",
                'existing_sign_boxes':measured.existing_sign_boxes(review)}
            frames.append(row)
        boxes = row['existing_sign_boxes']
        content = measured.output_content_rectangle(review['image_size'],row.get('crop_xyxy'))
        partial = any(box['output_box_xyxy'][0]<=content[0]+1 or box['output_box_xyxy'][1]<=content[1]+1 or
            box['output_box_xyxy'][2]>=content[2]-1 or box['output_box_xyxy'][3]>=content[3]-1 for box in boxes)
        state = empty_state(row,f'direct_{len(results)}','normal','direct_real')
        state['quota_group'] = 'partial' if partial else 'complete'
        state['direct_decision_id'] = review['record_id']
        results.append(state)
    return results


def qualify_test_backgrounds(source):
    """Resolve the old generator-probe gap using saved labels and ground masks."""
    database = ROOT/source['databases']['frame_environment']
    results = []
    for row in query_frames(task='sign',role='composite_background',split='test',status=None,database=database):
        if row['status'] != 'unknown' or row['reason_codes'] != ['generator_scale_and_placement_not_validated_for_this_visual_roi']:
            continue
        assert row['label']['presence']=='absent' and row['label']['completeness']=='complete'
        assert image_digest(row['image'])==row['content_sha256']
        width,height=row['image_size'];crop_width=height*4//3
        crop=[(width-crop_width)//2,0,(width+crop_width)//2,height]
        placement=decode_placement_record(row['geometry']['placeable_ground'])
        probes=[]
        for mask,_ in measured.VIEWS:
            for apparent_height in (75,240,650,1300):
                contact_width=measured._sign_contact_width(mask,apparent_height)
                found=False
                for bottom in range(416,461,11):
                    for center in range(80,561,80):
                        status=contact_status_in_view(placement,(center,bottom),contact_width=contact_width,
                            crop_xyxy=crop,content_xyxy=[0,0,640,480])
                        if status=='candidate':
                            probes.append({'mask':mask,'height':apparent_height,'foot_xy':[center,bottom],
                                           'contact_width':contact_width,'status':status})
                            found=True;break
                    if found:break
        if not probes:continue
        with edit_database(database) as writer:
            previous=writer.conn.execute('SELECT details_json FROM decisions WHERE record_id=? AND current=1',(row['record_id'],)).fetchone()
            if previous is None:raise ValueError('Test source changed during qualification')
            details=json.loads(previous['details_json'])
            proof=writer.observe(row['frame_id'],task='sign',kind='generator_probe',scope='placement',
                evidence_key='sign_test_compositor_probe',reference={'script':'datasets/scripts/sign/build_full_dataset.py'},
                payload={'image_sha256':row['content_sha256'],'crop_xyxy':crop,'probes':probes},
                algorithm_version='sign_test_contact_probe_v1')
            details.update(reason_codes=['placement_and_labels_cover_use'],
                qualification_note='Saved complete absence labels and source-pixel ground mask; current compositor contacts checked.')
            decision=writer.decide(row['frame_id'],task='sign',role='composite_background',split='test',status='eligible',
                evidence_ids=[e['evidence_id'] for e in row['evidence']]+[proof],details=details)
        results.append({'frame_id':row['frame_id'],'previous_record_id':row['record_id'],'record_id':decision,'probe_count':len(probes)})
    if results:
        receipt=ROOT/'datasets/annotated/sign/test_background_qualification.json'
        prior=json.loads(receipt.read_text(encoding='utf8')) if receipt.is_file() else []
        receipt.write_text(json.dumps(prior+results,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return results


def compile_full(split):
    _,source = load_task_config('sign')
    cfg = source['full_dataset']
    if split=='test':qualify_test_backgrounds(source)
    sample = json.loads((ROOT/source['diverse_sample_plans'].get(split,source['diverse_sample_plans']['val'])).read_text(encoding='utf8'))
    pool = measured.pool_plan_inputs(source,split)
    if split=='val':
        train_parents = {row['source_group'] for row in measured.pool_plan_inputs(source,'train')}
        pool = [row for row in pool if row['source_group'] not in train_parents]
    rng = random.Random(cfg['seed']+('train','val','test').index(split))
    frames_by_image = {}
    candidates = defaultdict(list)
    fingerprints = set()
    masks = json.loads((measured.base.ANNOTATED/'manifest.json').read_text(encoding='utf8'))
    cache_path = ROOT/f'datasets/annotated/sign/full_candidates_{split}.json'
    signature = hashlib.sha256(Path(__file__).read_bytes()+Path(planner.__file__).read_bytes()+Path(measured.__file__).read_bytes()+
        json.dumps(source,sort_keys=True,ensure_ascii=False).encode()+json.dumps([(row['frame_id'],row['record_id']) for row in pool]).encode()).hexdigest()
    cached = json.loads(cache_path.read_text(encoding='utf8')) if cache_path.is_file() else {}
    if cached.get('signature')==signature:
        frames_by_image = cached['frames_by_image']
        candidates.update({tuple(item['key']):item['states'] for item in cached['candidates']})
        rng_state = cached['rng_state']
        rng.setstate((rng_state[0], tuple(rng_state[1]), rng_state[2]))
        print(f'{split}: reusing exact-source candidate inventory',flush=True)
    else:
        for round_index in range(cfg['candidate_rounds']):
            frames = copy.deepcopy(pool if round_index==0 else [row for row in pool if '/rendered/' not in row['image']])
            rng.shuffle(frames)
            seed_source = copy.deepcopy(source)
            seed_source['background_expansion']['seed'] = cfg['seed']+101*round_index+(split=='val')
            for case in seed_source['sample_refinement']['occlusion_cases']:
                if case['split']==split and not case.get('negative_only'):
                    case['height'] += rng.randint(-12,12)
                    case['center'] += rng.randint(-15,15)
                    case['bottom'] += rng.randint(0,4)
            plan = {**copy.deepcopy(sample),'target_split':split,'background_inputs':frames,'strata':[]}
            plan = planner.refine_paired_sample(plan,seed_source,masks,validate=False)
            frames_by_image.update({row['image']:row for row in plan['background_inputs']})
            for state in plan['strata']:
                if state['target']['state']!='placed': continue
                layout = state['parameters']['layout']
                group = 'complete' if layout in ('small','medium') else 'partial' if layout in ('partial','occluded') else 'pole_only'
                domain = 'blender' if '/rendered/' in state['background']['image'] else 'camera'
                # Exact geometry/tone repeats from fixed occlusion cases are one image.
                fingerprint = json.dumps([state['background']['image'],state['target'],state['parameters'],state['augmentation']],sort_keys=True)
                if fingerprint in fingerprints: continue
                fingerprints.add(fingerprint)
                state['id'] = f'r{round_index}_{state["id"]}'
                state['quota_group'] = group
                candidates[group,state['conditions'][1],domain,layout=='occluded'].append(state)
            print(f'{split}: source/layout round {round_index+1}/{cfg["candidate_rounds"]}',flush=True)
        cache_path.write_text(json.dumps({'signature':signature,'frames_by_image':frames_by_image,'rng_state':rng.getstate(),
            'candidates':[{'key':list(key),'states':states} for key,states in candidates.items()]},ensure_ascii=False),encoding='utf8')
    frames = list(frames_by_image.values())
    selected = direct_rows(source,split,frames)
    frames_by_image = {row['image']:row for row in frames}
    quota = cfg['quota_groups'][split]
    matrix = copy.deepcopy(cfg['view_quotas'][split])
    for state in selected:
        matrix[state['quota_group']]['front'] -= 1  # Direct views stay unknown, not relabelled front.
    def take(group,view,domain,count,occluded=False):
        pool_states = candidates[group,view,domain,occluded]
        # Reserve both distance layouts before a larger fair-background draw.
        # Otherwise a small validation quota can take only medium side views.
        reserved = []
        if group == 'complete' and count >= 8:
            for layout in ('small', 'medium'):
                block = [state for state in pool_states if state['parameters']['layout'] == layout][:4]
                if len(block) != 4:
                    raise ValueError(f'Missing complete layout: {split}/{view}/{domain}/{layout}')
                reserved.extend(block)
        reserved_ids = {state['id'] for state in reserved}
        by_frame = defaultdict(deque)
        # Preserve four-tone blocks while cycling background frames fairly.
        for state in pool_states:
            if state['id'] not in reserved_ids: by_frame[state['background']['image']].append(state)
        keys = list(by_frame);rng.shuffle(keys)
        output = reserved
        while len(output)<count:
            before = len(output)
            for key in keys:
                for _ in range(min(4,count-len(output),len(by_frame[key]))):
                    output.append(by_frame[key].popleft())
                if len(output)==count: break
            if before==len(output):
                raise ValueError(f'Full quota lacks feasible variants: {split}/{group}/{view}/{domain}/{occluded}, need {count}, got {len(output)}')
        return output
    n_occluded = cfg['occluded_images'][split]
    for view,count in zip(('front','oblique'),shares(n_occluded,[1,1])):
        selected.extend(take('partial',view,'camera',count,True))
        matrix['partial'][view] -= count
    cells = [(group,view,count) for group,views in matrix.items() for view,count in views.items()]
    blender_counts = shares(cfg['blender_positive_train'],[n for _,_,n in cells]) if split=='train' else [0]*len(cells)
    for (group,view,count),blender in zip(cells,blender_counts):
        selected.extend(take(group,view,'blender',blender))
        selected.extend(take(group,view,'camera',count-blender))
    tones = source['background_expansion']['tones']
    negative_source = next((row for row in frames if row['image'].endswith(cfg['hard_negative_sources'].get(split,'/missing'))),None)
    bound = cfg['hard_negative_bounds'].get(split,[0,0,640,480])
    crops = []
    for width in range(640,431,-16):
        height=width*3//4
        for top in range(0,481-height,8):
            for left in range(0,641-width,8):
                if left<=bound[0] and top<=bound[1] and left+width>=bound[2] and top+height>=bound[3]:
                    crops.append([left,top,left+width,top+height])
    rng.shuffle(crops)
    for index in range(quota['ordinary_pole']):
        crop = crops[index//len(tones)]
        state = empty_state(negative_source,f'hard_negative_{index}',tones[index%len(tones)],'ordinary_signal_supports',crop)
        state['quota_group']='ordinary_pole';selected.append(state)
    empty_domains = [('blender',cfg['blender_images_train']-cfg['blender_positive_train']),
                     ('camera',quota['empty']-(cfg['blender_images_train']-cfg['blender_positive_train']))] if split=='train' else [('camera',quota['empty'])]
    for domain,count in empty_domains:
        rows = [row for row in frames if row.get('sign_absence')=='confirmed' and
            ('blender' if '/rendered/' in row['image'] else 'camera')==domain and (negative_source is None or row['image']!=negative_source['image'])]
        rng.shuffle(rows)
        if count>len(rows)*len(tones):raise ValueError('Insufficient distinct absent frame/tone combinations')
        for index in range(count):
            state=empty_state(rows[index%len(rows)],f'empty_{domain}_{index}',tones[index//len(rows)],'empty_scene')
            state['quota_group']='empty';selected.append(state)
    assert len(selected)==cfg['totals'][split],(split,len(selected))
    used={state['background']['image'] for state in selected}
    plan={key:copy.deepcopy(sample[key]) for key in ('schema_version','model_key','constraints','foreground_inputs','background_pool','design_context')}
    plan.update(target_split=split,budget=len(selected),sampling_seed=cfg['seed'],uncertainty_weight=0,
        purpose='Full recognition dataset with exact image quotas, shared foreground and parent-held validation',
        independence_basis=cfg['scope'],
        background_inputs=[row for row in frames if row['image'] in used],strata=selected,
        quota_groups=quota,source_scope=cfg['scope'])
    plan['background_inputs']=measured.bind_background_reviews(plan['background_inputs'],source,split)
    measured._prepare(plan,{},plan['background_inputs'],masks,ROOT)
    path=ROOT/cfg['plans'][split]
    path.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'split':split,'images':len(selected),'groups':dict(Counter(s['quota_group'] for s in selected)),
        'direct_real':sum(s['parameters'].get('negative_context')=='direct_real' for s in selected),'plan':str(path)},ensure_ascii=False),flush=True)
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=['train','val','test'],required=True)
    parser.add_argument('--generate',action='store_true')
    args=parser.parse_args()
    plan_path=compile_full(args.split)
    if args.generate:
        plan=json.loads(plan_path.read_text(encoding='utf8'))
        measured.generate(ROOT/f'datasets/exports/sign/{args.split}',plan['sampling_seed'],False,plan,plan_path=plan_path)


if __name__=='__main__':main()
