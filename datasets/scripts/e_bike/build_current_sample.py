"""Build one review sample from complete camera labels and grounded composites."""
import copy
import json
from collections import Counter
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from datasets.scripts.e_bike import compose_v4_grounded_train as ground
from datasets.scripts.common.config import load_task_config
from datasets.scripts.common.package_direct_real import prepare, check_yolo, TARGETS
from datasets.scripts.common import recoverable_publish as publisher
from datasets.scripts.common.images import read_image, write_image
from datasets.scripts.common.frame_environment import query_frames, image_digest
from lab.training_common.production_guard import generation_guard, receipt_completed


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def owned_files(output):
    names = {'manifest.json','review.html','contact_sheet.jpg','generation_plan.json'}
    if (output/'manifest.json').is_file():
        for row in read(output/'manifest.json')['records']:
            for name in [row.get('image'),row.get('label'),row.get('overlay'),
                         *(o.get('visible_mask') for o in row.get('objects',[]))]:
                if name:
                    path=(ROOT/name).resolve()
                    if not path.is_relative_to(output):
                        raise ValueError('Sample-owned path escapes output: '+name)
                    names.add(path.relative_to(output).as_posix())
    return names


def direct_objects(text,width,height,mirrored=False):
    check_yolo(text,2,0)
    objects=[]
    for index,line in enumerate(text.splitlines()):
        cls,x,y,w,h=map(float,line.split())
        x1,x2=(x-w/2)*width,(x+w/2)*width
        if mirrored:
            x1,x2=width-x2,width-x1
        objects.append({'object_id':f'real_{index:02d}', 'class':('upright','fallen')[int(cls)],
                        'bbox_xyxy':[round(x1),round((y-h/2)*height),round(x2),round((y+h/2)*height)],
                        'mirrored':mirrored,'visibility_evidence':'existing complete source visible-box annotation'})
    return objects


def selected_direct(recipe):
    """Keep incomplete long-queue reviews separate from complete direct supervision."""
    complete,_=prepare('e_bike',frame_record_ids=recipe['direct_frame_record_ids'])
    if len(complete)!=len(recipe['direct_frame_record_ids']):
        raise ValueError('Direct camera selection is incomplete')
    selected=list(complete)
    for record_id in recipe.get('partial_frame_record_ids',[]):
        rows=query_frames(task='e_bike',role='direct_target',record_id=record_id,status=None)
        if len(rows)!=1 or not rows[0]['current']:
            raise ValueError('Partial queue review is not current')
        record=rows[0];label=record['label']
        if label['completeness']!='partial' or not label.get('unresolved_regions'):
            raise ValueError('Queue partial record must declare unresolved evidence')
        image=ROOT/record['image']
        if image_digest(record['image'])!=record['content_sha256']:
            raise ValueError('Queue review pixels changed')
        w,h=record['image_size']
        text=''.join(ground.bike.scene.yolo_row(obj['class_id'],obj['xyxy'],w,h)+'\n' for obj in label['instances'])
        manifest=ROOT/record['provenance']['manifest']
        row={'id':record['frame_id'],'image':record['image'],'source_group':record['parent_group'],
             'parent_video':record.get('parent_video'),'frame_environment':{
                 'frame_id':record['frame_id'],'record_id':record['record_id'],'role':record['role'],
                 'split':record['split'],'status':record['status'],'content_sha256':record['content_sha256']}}
        selected.append((manifest,row,{**label,'split':record['split']},image,
                         ROOT/record['parent_video'] if record.get('parent_video') else None,None,text))
    return selected


def main():
    cv2.setNumThreads(1)
    _,config=load_task_config('e_bike')
    recipe=read(ROOT/config['sample_recipe'])
    composite=read(ROOT/config['production_plan'])
    backgrounds=ground.bind_background_reviews(read(ROOT/config['ground_background_manifest'])['backgrounds'],'train',config)
    _,vehicles=ground.bike.sources(selection=ROOT/config['ground_vehicle_selection'],include_people=False)
    ground.validate_ground_plan(composite,backgrounds,vehicles,ground.build_queue(ROOT))
    selected=selected_direct(recipe)
    plan=copy.deepcopy(composite)
    plan['target_split']='unassigned'
    plan['purpose']='Review-only camera and composite sample; preserve original source split, no training reassignment.'
    direct={}
    for item in selected:
        for mirrored in ([False,True] if recipe.get('direct_mirrors') else [False]):
            _,row,truth,image,parent,_,text=item
            sid=f'real_{len(direct)+1:06d}'
            sources=sorted({image.relative_to(ROOT).as_posix(),*([parent.relative_to(ROOT).as_posix()] if parent else [])})
            params={'source_record_id':row['frame_environment']['record_id'],'source_split':truth['split'],'mirrored':mirrored}
            state={'id':sid,'failure_event':'camera_count','min_count':1,'max_count':1,
                   'prior':{'alpha':1,'beta':1},'exposure_weight':1,'failure_cost':1,'conditions':[],
                   'background':{'image':row['image']},'target':{
                       'complete_camera_label':truth.get('completeness')!='partial',
                       'classes':[('upright','fallen')[int(line.split()[0])] for line in text.splitlines()]},
                   'labels':ground.GROUND_LABELS,'augmentation':{'mirror':mirrored},
                   'parameters':params,'sources':sources}
            plan['strata'].append(state);direct[sid]=item
            plan['sampling']['coverage'].append({'state_id':sid,'branch':'direct_real_review_partial' if truth.get('completeness')=='partial' else 'direct_real','mirrored':mirrored})
    plan['budget']=len(plan['strata'])
    plan['independence_basis']='Retain camera parent identity, original split and mirrors; no independent accuracy claim.'
    write(ROOT/config['sample_plan'],plan)
    media={ROOT/s for state in plan['strata'] for s in state['sources']}
    metadata={ROOT/config[k] for k in ('production_plan','sample_plan','sample_recipe','ground_background_manifest','ground_vehicle_selection')}
    metadata.update([TARGETS,ROOT/config['databases']['frame_environment'],
                     ROOT/'datasets/scripts/e_bike/config/source.yaml',ROOT/'datasets/scripts/e_bike/config/dataset.yaml'])
    for manifest,_,_,_,_,label,_ in selected:
        metadata.add(manifest)
        if label:metadata.add(label)
    code=[Path(__file__),Path(ground.__file__),Path(publisher.__file__),
          ROOT/'datasets/scripts/e_bike/plan_low_angle_batch.py',ROOT/'datasets/scripts/e_bike/synthesize_vehicle_scene.py',
          ROOT/'datasets/scripts/common/package_direct_real.py',ROOT/'datasets/scripts/common/frame_environment.py',
          ROOT/'datasets/scripts/common/images.py',ROOT/'datasets/scripts/common/composite.py',
          ROOT/'datasets/scripts/common/perspective.py',ROOT/'datasets/scripts/common/annotations.py',
          ROOT/'datasets/scripts/common/review/placement_record.py']
    output=ROOT/'datasets/samples/e_bike'
    operation='e_bike.build_current_sample'
    with publisher.sample_lock(output):
        committed=lambda token:receipt_completed(ROOT,operation,token)
        publisher.recover(output,committed)
        owned=owned_files(output)
        preserved=[p.relative_to(ROOT).as_posix() for p in output.rglob('*') if p.is_file() and p.relative_to(output).as_posix() not in owned]
        try:
            with generation_guard(operation,sorted(media),[output],{'recipe':config['sample_recipe']},root=ROOT,
                    procedural_inputs=sorted(metadata),code_paths=code,production_plan=plan,
                    sample_output_paths=[output/'images'],auxiliary_output_paths=[output/'masks',output/'overlays',output/'contact_sheet.jpg'],
                    replace_outputs=True,replace_verifier=lambda *_:{'preserved_paths':preserved}) as run:
                if run.reused:
                    print(json.dumps(read(output/'manifest.json')['summary'],ensure_ascii=False));return
                stage=publisher.prepare(output,run.token)
                publisher.copy_unowned(output,stage,owned)
                for folder in ('images','labels','masks'):(stage/folder).mkdir(exist_ok=True)
                records=[]
                for state in run.plan['strata']:
                    if state['planned_count']!=1:raise ValueError('Every review state must be produced exactly once')
                    sid=state['id'];stem=sid
                    row={'state_id':sid,'image':ground.rel(output/'images'/(stem+'.png')),
                         'label':ground.rel(output/'labels'/(stem+'.txt')),'parameters':state['parameters'],'sources':state['sources']}
                    if sid in direct:
                        manifest,source,truth,image,_,_,text=direct[sid]
                        frame=read_image(image);h,w=frame.shape[:2]
                        mirror=state['parameters']['mirrored']
                        objects=direct_objects(text,w,h,mirror)
                        if mirror:frame=np.ascontiguousarray(frame[:,::-1])
                        regions=copy.deepcopy(truth.get('unresolved_regions',[]))
                        if mirror:
                            for region in regions:
                                x1,y1,x2,y2=region['xyxy'];region['xyxy']=[w-x2,y1,w-x1,y2]
                        row.update(branch='direct_real_review_partial' if regions else 'direct_real',
                                   unresolved_regions=regions,label_scope='known_postures_only_not_training' if regions else 'complete_visible_instances',
                                   source_image=ground.rel(image),source_manifest=ground.rel(manifest),
                                   source_split=truth['split'],source_group=source.get('source_group') or source.get('parent_video'),
                                   frame_environment=source['frame_environment'],mirrored=mirror)
                    else:
                        frame,objects,masks,bg=ground.render_ground_state(state,backgrounds,vehicles)
                        h,w=frame.shape[:2]
                        branch='real_background_composite' if bg.get('source_kind')!='blender_render' else 'blender_background_composite'
                        row.update(branch=branch,source_split='train',background=bg,background_review=state['background'])
                        for number,(obj,mask) in enumerate(zip(objects,masks)):
                            if mask is not None:
                                name=f'{stem}_{number:02d}.png';write_image(stage/'masks'/name,mask.astype(np.uint8)*255)
                                obj['visible_mask']=ground.rel(output/'masks'/name)
                    row.update(objects=objects,image_size=[w,h])
                    labels=''.join(ground.bike.scene.yolo_row(ground.GROUND_LABELS['classes'][o['class']],o['bbox_xyxy'],w,h)+'\n' for o in objects)
                    check_yolo(labels,2,0)
                    image_path=stage/'images'/(stem+'.png');label_path=stage/'labels'/(stem+'.txt')
                    if image_path.exists() or label_path.exists():raise ValueError('New sample collides with unowned content')
                    write_image(image_path,frame);label_path.write_text(labels,encoding='utf-8');records.append(row)
                ground.write_previews(stage,records,plan,published_out=output)
                summary={'images':len(records),'by_branch':dict(Counter(r['branch'] for r in records)),
                         'instances':dict(Counter(o['class'] for r in records for o in r['objects'])),
                         'source_splits':dict(Counter(r['source_split'] for r in records)),
                         'mirrored_direct_images':sum(r.get('mirrored',False) for r in records if r['branch'].startswith('direct_real')),
                         'partial_review_images':sum(bool(r.get('unresolved_regions')) for r in records)}
                summary['instance_count_scope']='Confirmed posture labels; partial review frames are lower bounds, not complete vehicle counts.'
                write(stage/'generation_plan.json',plan)
                write(stage/'manifest.json',{'task':'e_bike','status':'mixed_source_sample_for_review','plan':config['sample_plan'],
                       'generation_plan':ground.rel(output/'generation_plan.json'),
                       'composite_plan':config['production_plan'],'sampling':plan['sampling'],'summary':summary,'records':records,
                       'scope':'Diagnostic sample only. Direct source split stays unchanged; composite sources are train. No formal release.'})
                if selected_direct(recipe)!=selected:
                    raise ValueError('Direct evidence changed during production')
                publisher.publish(output,stage,run.token,keep_root=True)
                by_id={s['id']:s for s in run.plan['strata']}
                for row in records:
                    s=by_id[row['state_id']]
                    run.record_sample(s['id'],output=ROOT/row['image'],sources=s['sources'],parameters=s['parameters'],actual=row)
                run.set_result(summary)
            publisher.recover(output,committed)
        except BaseException:
            publisher.recover(output,committed);raise
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
