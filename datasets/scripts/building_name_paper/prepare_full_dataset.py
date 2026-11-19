"""Register selected photo plates and compile split-specific full-data plans."""
import argparse
from copy import deepcopy
import itertools
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from datasets.scripts.building_name_paper import full_plan as full
from datasets.scripts.common.frame_environment import edit_database,find_frames,query_frames
from datasets.scripts.common.config import source_path
from datasets.scripts.common.paths import building_name_full_work_dir


def destination(key):
    from datasets.scripts.common.config import load_task_config
    _,source=load_task_config(full.TASK)
    path=(ROOT/source[key]).resolve()
    if path.parent != Path(__file__).parent/'config':raise ValueError('Plan output escapes task config')
    return path


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def register(plates):
    path=source_path(full.TASK,'full_plates')
    with edit_database() as writer:
        for p in plates:
            for prior in query_frames(task=full.TASK,role='composite_background',frame_id=p['frame_id']):
                if prior['split'] not in ('unassigned',p['split']) and prior.get('geometry',{}).get('full_plate_id')==p['id']:
                    evidence={r['kind']:r['evidence_id'] for r in prior['evidence'] if r['kind'] in ('label','geometry')}
                    writer.decide(p['frame_id'],task=full.TASK,role='composite_background',split=prior['split'],
                        status='blocked',evidence_ids=list(evidence.values()),details={
                            'image':p['image'],'parent_group':p['parent_group'],
                            'label_evidence_id':evidence['label'],'geometry_evidence_id':evidence['geometry'],
                            'reason_codes':['reallocated_before_full_export'],'assigned_split':p['split']})
            ref={'manifest':path.relative_to(ROOT).as_posix(),'plate_id':p['id'],
                 'method':'codex_visual_review','algorithm_version':'building_full_photo_review_v1',
                 'content_sha256':p['content_sha256'],'note':p['review_note']}
            label={'presence':'absent','completeness':'complete','scope':{'kind':'frame'},'instances':[],'evidence':ref}
            geom={k:deepcopy(p[k]) for k in ('ground_polygon','forbidden_footprint_xyxy','horizon_y_px','perspective_basis')}
            geom.update(status='candidate',full_plate_id=p['id'],coordinate_size=p['image_size'],
                        coordinate_space='source_image_pixels',evidence=ref)
            ids={}
            for kind,scope,payload in [('label','frame',label),('geometry','placement',geom)]:
                ids[kind]=writer.observe(p['frame_id'],task=full.TASK,kind=kind,scope=scope,
                    evidence_key='building_full_plate',reference=ref,payload=payload,
                    algorithm_version='building_full_photo_review_v1')
            p['record_id']=writer.decide(p['frame_id'],task=full.TASK,role='composite_background',split=p['split'],
                status='eligible',evidence_ids=list(ids.values()),details={
                    'image':p['image'],'parent_group':p['parent_group'],'source_kind':'camera',
                    'label_evidence_id':ids['label'],'geometry_evidence_id':ids['geometry'],
                    'role_basis':'background_review','reason_codes':['whole_frame_absence_and_reviewed_ground'],
                    'split_reuse_evidence':{'scope':'selected full-batch assignment; prior development exposure retained'}})
    write(path,{'task':full.TASK,'plates':plates})


def register_real_tests():
    review=full.read(source_path(full.TASK,'full_real_test_review'));ids=[]
    with edit_database() as writer:
        for item in review['records']:
            frame=find_frames(path=item['image'])[0]
            evidence={'manifest':source_path(full.TASK,'full_real_test_review').relative_to(ROOT).as_posix(),
                      'frame_id':frame['frame_id'],'method':'codex_visual_review',
                      'algorithm_version':'building_full_real_test_v1','content_sha256':frame['content_sha256']}
            label={'presence':'present','completeness':'complete','scope':{'kind':'frame'},
                   'instances':[{'class_id':0,'class_name':'paper','xyxy':item['xyxy'],'text':None}],
                   'evidence':evidence}
            eid=writer.observe(frame['frame_id'],task=full.TASK,kind='label',scope='frame',
                evidence_key='building_full_real_test',reference=evidence,payload=label,
                algorithm_version='building_full_real_test_v1')
            record=writer.decide(frame['frame_id'],task=full.TASK,role='direct_target',split='test',status='eligible',
                evidence_ids=[eid],details={'image':item['image'],'parent_group':frame['parent_group'],
                    'parent_video':frame['parent_group'],'source_kind':'camera','generation_method':'direct_camera',
                    'scope':{'kind':'frame'},'label_evidence_id':eid,'reason_codes':['complete_current_paper_detection'],
                    'split_reuse_evidence':{'scope':'prior field evaluation and development exposure retained'}})
            ids.append(record)
    import yaml
    path=source_path(full.TASK,'detection_evaluation');data=yaml.safe_load(path.read_text(encoding='utf8'))
    data['test']['direct_target']=[r for r in data['test']['direct_target'] if r not in review.get('registered_record_ids',[])]+ids
    path.write_text(yaml.safe_dump(data,allow_unicode=True,sort_keys=False),encoding='utf8')
    review['registered_record_ids']=ids;write(source_path(full.TASK,'full_real_test_review'),review)


def state(stem,split,branch,target,background,parameters,sources):
    return {'id':stem,'min_count':1,'max_count':1,'prior':{'alpha':1,'beta':1},
        'failure_event':'paper_detection_miss_or_false_positive','exposure_weight':1,'failure_cost':1,
        'conditions':[f'branch:{branch}',f'split:{split}'],'target':target,'background':background,
        'parameters':parameters,'sources':sources,'labels':{'class_id':0,'scope':'complete_paper_detection'},
        'augmentation':{'mode':'bounded_sensor_and_tone'},'branch':branch}


def declaration(split,states,purpose):
    return {'schema_version':1,'model_key':full.TASK,'target_split':split,'budget':len(states),
        'purpose':purpose,'uncertainty_weight':0.,'independence_basis':
        'Capture videos define background parent groups. Shared building meshes are reusable assets; synthetic evaluation is diagnostic.',
        'constraints':{'supervision':'complete_paper_detection'},'strata':states}


def compile_templates(plates,recipe):
    states=[]
    for p in plates:
        for active in recipe['building_combinations']:
            for index,view in enumerate(recipe['render_views']):
                yaw=view['single_yaw_degrees' if len(active)==1 else 'multi_yaw_degrees']
                stem=f"plate_{p['id']}_{active}_{view['id']}"
                pars={'yaw_degrees':yaw,'camera_height_m':view['camera_height_m'],
                      'seed':recipe['seed']+len(states)}
                states.append(state(stem,p['split'],'physical_template',{'buildings':active,'papers':active},
                    {'plate_id':p['id'],'frame_id':p['frame_id'],'record_id':p['record_id']},pars,
                    [p['image'],'datasets/assets/blender/aiaec_dataset.blend']))
    result=declaration('train',states,'Physical counterfactual layers for all split-specific building/photo layouts')
    write(destination('full_render_plan'),result)
    return states


def compile_split(split,plates,templates,recipe):
    from datasets.scripts.building_name_paper import build_current_sample as sample
    from datasets.scripts.building_name_paper import package_detection_dataset as package
    from datasets.scripts.common.config import load_task_config
    from datasets.scripts.common.package_direct_real import prepare
    from datasets.scripts.common.source_history import historical_source,read_source_history
    states=[]
    if split=='train':
        rows,_,_=sample.direct_records({'direct_split':'train','additional_direct_records':[]})
        direct=[{'source_image':str(image.relative_to(ROOT)).replace('\\','/'),'label_text':text,
                 'metadata':meta} for image,text,meta in rows]
    else:
        _,source=load_task_config(full.TASK)
        import yaml
        selected=yaml.safe_load(source_path(full.TASK,'detection_evaluation').read_text(encoding='utf8'))
        direct=[]
        for image,_,_,labels,meta in package.evaluation_entries(selected,ROOT/source['databases']['frame_environment']):
            if meta['split']!=split:continue
            direct.append({'source_image':str(image.relative_to(ROOT)).replace('\\','/'),
                           'label_text':'\n'.join(labels)+('\n' if labels else ''),'metadata':meta})
    for i,item in enumerate(direct):
        count=len(item['label_text'].splitlines())
        states.append(state(f'{split}_direct_{i:04d}',split,'direct_real',{'paper_count':count},
            {'source_image':item['source_image']},{'direct':item,'seed':recipe['seed']+i},[item['source_image']]))
    quotas=recipe['splits'][split]
    if split=='train':
        composite=sample.composite_plan({'seed':recipe['seed'],'composites_per_background':1})
        for i in range(quotas['real_background_composite']):
            base=deepcopy(composite['strata'][i%len(composite['strata'])])
            candidate=base['background']['candidates'][0]
            seed=recipe['seed']+i+30000
            for paper in base['target']['papers']:paper['text']=sample.rooftop.make_name(seed)
            states.append(state(f'{split}_replace_{i:04d}',split,'real_background_composite',base['target'],
                base['background'],{'replacement_state':base,'seed':seed},[candidate['image'],candidate['parent_source']]))
    available=[x for x in templates if next(p for p in plates if p['id']==x['background']['plate_id'])['split']==split]
    if not available:raise ValueError('No executable templates for '+split)
    # Each route receives zero-paper, partial-paper and all-paper controls.
    for branch in ('blender_photo_composite','blender_render'):
        count=quotas[branch]
        roster=[(active,''.join(items)) for active in recipe['building_combinations']
                for n in range(len(active)+1) for items in itertools.combinations(active,n)]
        roster += [(active,active) for active in recipe['building_combinations']]
        used={active:0 for active in recipe['building_combinations']}
        for i in range(count):
            active,papers=roster[i%len(roster)]
            options=[item for item in available if item['target']['buildings']==active]
            if not options:raise ValueError('No accepted '+active+' layout in '+split)
            template=options[used[active]%len(options)]
            used[active]+=1
            cycle=used[active]//len(options)
            pars={'template_id':template['id'],'papers':papers,'seed':recipe['seed']+100000*(1+('train','val','test').index(split))+
                  50000*(branch=='blender_render')+i,'variant':cycle,
                  'yaw_degrees':template['parameters']['yaw_degrees'],
                  'camera_height_m':template['parameters']['camera_height_m']}
            states.append(state(f'{split}_{branch}_{i:05d}',split,branch,{'buildings':active,'papers':papers},
                template['background'],pars,template['sources']))
    expected=quotas['budget']
    if len(states)!=expected:raise ValueError(f'{split}: compiled {len(states)} != {expected}')
    write(destination('full_'+split+'_plan'),declaration(split,states,'Current full paper detection '+split+' batch'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register',action='store_true')
    parser.add_argument('--accepted-templates',type=Path)
    args=parser.parse_args()
    recipe=full.config();plates=full.read(source_path(full.TASK,'full_plates'))['plates']
    if args.register:
        register(plates)
        register_real_tests()
    plates=list(full.plates())
    templates=compile_templates(plates,recipe)
    if args.accepted_templates:
        approved=full.read(args.accepted_templates)['accepted']
        templates=[s for s in templates if s['id'] in approved]
    for split in ('train','val','test'):compile_split(split,plates,templates,recipe)
    print(json.dumps({'plates':len(plates),'physical_templates':len(templates),
                      'budgets':{s:full.split_plan(s)['budget'] for s in ('train','val','test')}},ensure_ascii=False))


if __name__=='__main__':main()
