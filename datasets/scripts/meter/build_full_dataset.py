"""Compile, render in resumable chunks, audit and export the full meter dataset."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from datasets.scripts.common.config import load_task_config, read_yaml
from datasets.scripts.common import recoverable_publish as publisher
from datasets.scripts.common.frame_environment import image_digest, query_frames
from datasets.scripts.common.ingest_frame_environment import ingest_frame
from datasets.scripts.meter.build_current_sample import read, write, label_values, POINTS
from datasets.scripts.meter.backgrounds import adjust, pixel_stats
from datasets.scripts.meter.compose_meter_real_background_v4 import place_foreground
from datasets.scripts.meter.full_variations import random_placement, degrade, local_occluder, draw_points
from datasets.scripts.meter.full_plan import compile_plans, plan as make_plan, state as make_state, digest, SCENES, SPLITS
from lab.training_common.production_guard import generation_guard, receipt_completed

HERE=Path(__file__).resolve()


def relative(path):return Path(path).resolve().relative_to(ROOT).as_posix()


def dependencies():
    return [HERE,*[HERE.with_name(name) for name in ('full_plan.py','full_variations.py','backgrounds.py',
            'render_open_sample.py','build_current_sample.py','compose_meter_real_background_v4.py',
            'meter_station_plan.py','render_meter_candidates.py','render_meter_batch.py','source_index.py')],
            ROOT/'datasets/scripts/common/blender_workspace.py',Path(publisher.__file__)]


def configs():
    return [HERE.parent/'config/dataset.yaml',HERE.parent/'config/source.yaml']


def valid_cache(folder,operation,expected):
    path=folder/'completed.json'
    if not path.is_file():return False
    record=read(path)
    return (record['plan_sha256']==digest(expected) and receipt_completed(ROOT,operation,record['token'])
            and all((ROOT/name).is_file() and image_digest(name)==sha for name,sha in record.get('inputs',{}).items())
            and all((folder/name).is_file() and image_digest(folder/name)==sha for name,sha in record['files'].items()))


def finish_chunk(folder,operation,declared,media,metadata,command,record_reader,*,sample_folder):
    if valid_cache(folder,operation,declared):
        return record_reader()
    folder.parent.mkdir(parents=True,exist_ok=True)
    with generation_guard(operation,media,[folder],{'plan_sha256':digest(declared)},root=ROOT,
            procedural_inputs=[*configs(),*metadata],code_paths=dependencies()+[ROOT/'datasets/scripts/common/render_shared_route_backgrounds.py'],
            production_plan=declared,sample_output_paths=[folder/sample_folder],
            auxiliary_output_paths=[folder/'masks'] if sample_folder=='foreground' else [],
            replace_outputs=True,replace_verifier=lambda *_:{'preserved_paths':[]}) as run:
        if run.reused:return record_reader()
        if folder.exists():
            # Only this incomplete generated chunk is replaced; source media and other chunks stay intact.
            for file in folder.rglob('*'):
                if file.is_file():file.unlink()
        folder.mkdir(parents=True,exist_ok=True)
        log=folder/'render.log'
        with log.open('w',encoding='utf8') as stream:
            if callable(command):command(folder)
            else:subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        records=record_reader()
        by_id={r['id']:r for r in records}
        for state in declared['strata']:
            row=by_id[state['id']]
            run.record_sample(state['id'],output=ROOT/row['rendered_image'],sources=state['sources'],parameters=state['parameters'],actual=row)
        files={p.relative_to(folder).as_posix():image_digest(p) for p in folder.rglob('*') if p.is_file()}
        write(folder/'completed.json',{'plan_sha256':digest(declared),'token':run.token,'files':files,
              'inputs':{relative(p):image_digest(p) for p in [*media,*metadata]}})
        run.set_result({'images':len(records),'folder':relative(folder)})
    return records


def render_foreground_chunk(split,index,states,render_plan,recipe,source):
    folder=ROOT/recipe['work_root']/split/'foreground'/f'chunk_{index:04d}'
    chunk={**render_plan,'budget':len(states),'strata':states}
    path=folder.parent/f'chunk_{index:04d}_plan.json'
    if not path.is_file() or read(path)!=chunk:write(path,chunk)
    operation=f'meter.full.foreground.{split}.{index:04d}'
    sample=read(ROOT/source['sample_recipe'])
    def records():
        rows=read(folder/'render_records.json')
        return [{**r,'rendered_image':relative(folder/r['foreground_rgba']),
                 'render_annotation':relative(folder/'render_records'/(r['id']+'.json')),
                 'render_chunk':relative(folder)} for r in rows]
    command=[sample['blender_executable'],'--background',str(ROOT/render_plan['source_blend']),
             '--python-exit-code','1','--python',str(HERE.with_name('render_open_sample.py')),
             '--','--plan',str(path),'--output',str(folder)]
    rows=finish_chunk(folder,operation,chunk,[ROOT/render_plan['source_blend'],*[ROOT/p for p in render_plan['face_variants'].values()]],
                      [path,ROOT/render_plan['endpoint_config']],command,records,sample_folder='foreground')
    print('FOREGROUND_CHUNK',split,index,len(rows),'complete',flush=True)
    return rows


def render_foregrounds(split,recipe,source,pilot_states=None):
    declared=read(ROOT/f'datasets/scripts/meter/config/full_{split}_render_plan.json')
    if pilot_states is not None:
        ids={s['parameters']['foreground_id'] for s in pilot_states}
        states=[s for s in declared['strata'] if s['id'] in ids]
        return render_foreground_chunk(split,9999,states,declared,recipe,source)
    rows=[];size=recipe['chunk_size']
    for index,start in enumerate(range(0,len(declared['strata']),size)):
        rows.extend(render_foreground_chunk(split,index,declared['strata'][start:start+size],declared,recipe,source))
    return rows


def render_backgrounds(split,recipe,source):
    path=ROOT/recipe['work_root']/split/'background_inventory.json'
    inventory=read(path)['backgrounds']
    if split=='train':return inventory
    source_blend=relative(ROOT/recipe['work_root']/'production_inputs/source.blend')
    executable=read(ROOT/source['sample_recipe'])['blender_executable']
    all_records=[]
    for index,scene in enumerate(SCENES):
        folder=ROOT/recipe['work_root']/split/'backgrounds'/f'chunk_{index:04d}'
        selected=[r for r in inventory if r['scene']==scene]
        states=[make_state(r['background_key'],{'branch':'empty_negative','scene':scene,'route':r['route'],
                'frame':r['render_frame'],'lens_scale':recipe['background_lens_scale'][split]},[source_blend],negative=True) for r in selected]
        declared=make_plan(split,states,'Dedicated split-specific empty room backgrounds')
        plan_path=folder.parent/f'chunk_{index:04d}_plan.json'
        if not plan_path.exists() or read(plan_path)!=declared:write(plan_path,declared)
        def records():
            rows=read(folder/'manifest.json')['records']
            return [{**r,'id':f"{split}:{r['scene']}:{r['route']}:{r['frame']}",'rendered_image':r['image']} for r in rows]
        command=[executable,'--background',str(ROOT/source_blend),'--python-exit-code','1','--python',
                 str(ROOT/'datasets/scripts/common/render_shared_route_backgrounds.py'),'--','--scene',scene,
                 '--output',str(folder),'--manifest',str(folder/'manifest.json'),
                 '--frames',','.join(map(str,recipe['background_frames'][split])),
                 '--target-split',split,'--lens-scale',str(recipe['background_lens_scale'][split]),
                 '--exposure-ev',str(recipe['background_exposure_ev'][split])]
        rows=finish_chunk(folder,f'meter.full.background.{split}.{index:04d}',declared,[ROOT/source_blend],
                          [plan_path],command,records,sample_folder='')
        all_records.extend(rows)
        print('BACKGROUND_CHUNK',split,scene,len(rows),'complete',flush=True)
    by_key={r['id']:r for r in all_records}
    existing={r['image']:r for r in query_frames(task='meter',role='composite_background',split=split)}
    for i,row in enumerate(inventory):
        rendered=by_key[row['background_key']]
        if row['image'] in existing:
            current=existing[row['image']]
        else:
            evidence={'manifest':str(Path(rendered['image']).parent/'manifest.json').replace('\\','/'),
                      'method':'rendered_empty_scene_object_inventory','source_blend_sha256':rendered['source_blend_sha256']}
            evidence['record_index']=next(i for i,r in enumerate(read(ROOT/evidence['manifest'])['records'])
                                          if r['image']==row['image'])
            label={'presence':'absent','completeness':'complete','scope':{'kind':'frame'},'instances':[],
                   'split':split,'evidence':evidence}
            geometry={'status':'candidate','regions':[{'kind':'full_frame_empty_scene','polygon':[[0,0],[640,0],[640,480],[0,480]]}],'evidence':evidence}
            result=ingest_frame(row['image'],objects=[],task_annotations={'meter':label},task_geometry={'meter':geometry},
                   environment={'status':'observed','evidence':evidence},
                   lineage={'source_kind':'blender_render','parent_group':row['parent_group'],'source_blend':source_blend,
                            'capture_identity':row['background_key'],'scene':row['scene'],'route':row['route'],
                            'render_frame':row['render_frame'],'provenance':evidence},expected_sha256=rendered['content_sha256'])
            current=next(r for r in query_frames(task='meter',role='composite_background',split=split)
                         if r['record_id']==result['records']['meter']['composite_background'])
        inventory[i]={**row,**{k:current[k] for k in ('record_id','frame_id','content_sha256','geometry','label','evidence')},
                      'qualification':'eligible'}
    write(path,{'task':'meter','split':split,'backgrounds':inventory})
    return inventory


def compose_record(state,raw,background,recipe,photometry):
    p=state['parameters'];image=ImageOps.fit(Image.open(ROOT/background['image']).convert('RGB'),(640,480),method=Image.Resampling.LANCZOS)
    if image_digest(background['image'])!=background['content_sha256']:raise ValueError('Background content mismatch')
    image=adjust(image,saturation=photometry['saturation'][p['saturation']],brightness=photometry['exposure'][p['exposure']])
    values=[];transform=None;occlusion=None
    if raw is not None:
        fg=Image.open(ROOT/raw['rendered_image']).convert('RGBA')
        proposed,support=random_placement(background,fg,p['placement_controls'])
        fg,values,transform=place_foreground(fg,label_values(raw),proposed)
        transform['contact_support_xyxy']=support
        fg=adjust(fg,saturation=photometry['saturation'][p['saturation']],
                  brightness=photometry['exposure'][p['exposure']]*photometry['light_relation'][p['light_relation']])
        x1,y1,x2,y2=fg.getchannel('A').getbbox();shadow=Image.new('RGBA',(640,480))
        ImageDraw.Draw(shadow).ellipse((x1-3,y2-5,x2+3,y2+4),fill=(0,0,0,55))
        image=Image.alpha_composite(image.convert('RGBA'),shadow.filter(ImageFilter.GaussianBlur(4)))
        image=Image.alpha_composite(image,fg).convert('RGB')
        image,values,occlusion=local_occluder(image,values,p['occlusion'])
    image=degrade(image,p['degradation'])
    row={'id':state['id'],'parameters':p,'sources':state['sources'],'image_size':[640,480],
         'image':'images/'+state['id']+'.png','pose_label':'labels/'+state['id']+'.txt',
         'annotation':'annotations/'+state['id']+'.json','label_values':values,'instance_count':int(bool(values)),
         'background':background,'transform':transform,'occlusion_evidence':occlusion,'pixel_stats':pixel_stats(image),
         'training_eligible':True,'keypoint_order':list(POINTS)}
    if raw:
        row.update(foreground_id=raw['id'],foreground_annotation=raw['render_annotation'],
                   foreground_geometry={'bbox_xyxy_pixels':raw['bbox_xyxy_pixels'],'display_roi_quad':raw['display_roi_quad']},
                   bbox_xyxy_pixels=[(values[1]-values[3]/2)*640,(values[2]-values[4]/2)*480,
                                     (values[1]+values[3]/2)*640,(values[2]+values[4]/2)*480],
                   keypoints_xy_depth_normalized_candidate={key:[values[5+3*i],values[6+3*i],raw['keypoints_xy_depth_normalized_candidate'][key][2]] for i,key in enumerate(POINTS)},
                   keypoint_visibility={key:{'visibility':int(values[7+3*i]),'reason':'known_geometry_after_placement_and_capture_variation'} for i,key in enumerate(POINTS)})
        scale=transform['scale'];dx,dy=transform['offset_xy']
        row['display_roi_quad']=[[x*scale+dx/640,y*scale+dy/480,z] for x,y,z in raw['display_roi_quad']]
    return image,row


def direct_record(state,split,source):
    lines=(ROOT/source['direct_review_manifest']).read_text(encoding='utf8').splitlines()
    original=json.loads(lines[state['parameters']['source_record_index']])
    image=Image.open(ROOT/original['path']).convert('RGB');w,h=image.size;x1,y1,x2,y2=original['dial_bbox']
    values=[0,(x1+x2)/(2*w),(y1+y2)/(2*h),(x2-x1)/w,(y2-y1)/h]
    for point in original['keypoints']:
        x,y=point['xy'];values.extend([x/w,y/h,point['visibility']])
    sid=state['id']
    return image,{'id':sid,'parameters':state['parameters'],'sources':state['sources'],'image_size':list(image.size),
        'image':f'images/{sid}.png','pose_label':f'labels/{sid}.txt','annotation':f'annotations/{sid}.json',
        'label_values':values,'instance_count':1,'observed_keypoints':original['keypoints'],'pose_annotation':original['pose_annotation'],
        'source_image':original['path'],'source_manifest':source['direct_review_manifest'],'source_record_index':state['parameters']['source_record_index'],
        'source_group':original.get('source_group'),'source_use_history':original.get('task_use_status',[]),
        'keypoint_order':list(POINTS),'bbox_xyxy_pixels':original['dial_bbox'],'training_eligible':True}


def composition_chunks(split,declared,rendered,inventory,recipe,source):
    """Persist individually audited composite blocks before installing a split."""
    foreground={r['id']:r for r in rendered};backgrounds={r['background_key']:r for r in inventory}
    photometry=read(ROOT/source['sample_recipe'])['photometry'];all_rows=[]
    size=recipe.get('composition_chunk_size',250)
    for index,start in enumerate(range(0,len(declared['strata']),size)):
        states=declared['strata'][start:start+size]
        chunk=make_plan(split,states,'Audited persistent meter composition chunk')
        folder=ROOT/recipe['work_root']/split/'composites'/f'chunk_{index:04d}'
        path=folder.parent/f'chunk_{index:04d}_plan.json'
        if not path.is_file() or read(path)!=chunk:write(path,chunk)
        fg_ids={s['parameters']['foreground_id'] for s in states if 'foreground_id' in s['parameters']}
        bg_ids={s['parameters']['background_key'] for s in states if 'background_key' in s['parameters']}
        media=[ROOT/foreground[key]['rendered_image'] for key in fg_ids]+[ROOT/backgrounds[key]['image'] for key in bg_ids]
        media.extend(ROOT/s['parameters']['source_image'] for s in states if s['parameters']['branch']=='direct_task_pose')
        metadata=[path,ROOT/source['full_recipe'],ROOT/source['sample_recipe'],
                  *[ROOT/foreground[key]['render_annotation'] for key in fg_ids]]
        if any(s['parameters']['branch']=='direct_task_pose' for s in states):metadata.append(ROOT/source['direct_review_manifest'])
        def produce(destination):
            for name in ('images','labels','annotations'):(destination/name).mkdir(exist_ok=True)
            rows=[]
            for state in states:
                p=state['parameters']
                if p['branch']=='direct_task_pose':image,row=direct_record(state,split,source)
                else:
                    raw=foreground[p['foreground_id']] if 'foreground_id' in p else None
                    image,row=compose_record(state,raw,backgrounds[p['background_key']],recipe,photometry)
                row['split']=split;row['composition_input']=relative(destination/row['image'])
                image.save(destination/row['image'],compress_level=1)
                values=row['label_values']
                (destination/row['pose_label']).write_text((' '.join(f'{v:.8f}' for v in values)+'\n') if values else '',encoding='utf8')
                write(destination/row['annotation'],row);rows.append(row)
            with (destination/'samples.jsonl').open('w',encoding='utf8') as stream:
                for row in rows:stream.write(json.dumps(row)+'\n')
            write(destination/'audit.json',audit_records(destination,rows))
        def records():
            return [{**json.loads(line),'rendered_image':relative(folder/json.loads(line)['image'])}
                    for line in (folder/'samples.jsonl').read_text(encoding='utf8').splitlines()]
        all_rows.extend(finish_chunk(folder,f'meter.full.compose.{split}.{index:04d}',chunk,media,metadata,produce,records,sample_folder='images'))
        print('COMPOSITION_CHUNK',split,index,len(all_rows),'/',len(declared['strata']),flush=True)
    return {row['id']:row for row in all_rows}


def export_split(split,declared,rendered,inventory,recipe,source,*,pilot=False):
    output=ROOT/(recipe['work_root']+'/production_inputs/pilot' if pilot else 'datasets/exports/meter/'+split)
    operation='meter.full.export.'+('pilot' if pilot else split)
    cached=None if pilot else composition_chunks(split,declared,rendered,inventory,recipe,source)
    foreground={r['id']:r for r in rendered};backgrounds={r['background_key']:r for r in inventory}
    needed_fg={s['parameters']['foreground_id'] for s in declared['strata'] if 'foreground_id' in s['parameters']}
    needed_bg={s['parameters']['background_key'] for s in declared['strata'] if 'background_key' in s['parameters']}
    media=[ROOT/foreground[key]['rendered_image'] for key in needed_fg]+[ROOT/backgrounds[key]['image'] for key in needed_bg]
    media.extend(ROOT/s['parameters']['source_image'] for s in declared['strata'] if s['parameters']['branch']=='direct_task_pose')
    metadata=[ROOT/source['full_recipe'],ROOT/source['direct_review_manifest'],
              *[ROOT/foreground[key]['render_annotation'] for key in needed_fg]]
    chunk_plan=ROOT/recipe['work_root']/'production_inputs'/('pilot_plan.json' if pilot else f'{split}_export_plan.json')
    if not chunk_plan.is_file() or read(chunk_plan)!=declared:write(chunk_plan,declared)
    metadata.append(chunk_plan)
    bg_snapshot=chunk_plan.with_name(chunk_plan.stem+'_backgrounds.json')
    if not bg_snapshot.is_file() or read(bg_snapshot)!=inventory:write(bg_snapshot,inventory)
    metadata.append(bg_snapshot)
    if cached is not None:
        media=[ROOT/row['rendered_image'] for row in cached.values()]
        metadata.extend(ROOT/Path(row['rendered_image']).parents[1]/row['pose_label'] for row in cached.values())
        metadata.extend(ROOT/Path(row['rendered_image']).parents[1]/row['annotation'] for row in cached.values())
    photometry=read(ROOT/source['sample_recipe'])['photometry'];metadata.append(ROOT/source['sample_recipe'])
    output.parent.mkdir(parents=True,exist_ok=True)
    with publisher.sample_lock(output):
        committed=lambda token:receipt_completed(ROOT,operation,token)
        publisher.recover(output,committed)
        previous=read(output/'manifest.json') if (output/'manifest.json').is_file() else {}
        owned=set(previous.get('owned_files',[]))
        preserved=[relative(p) for p in output.rglob('*') if p.is_file() and p.relative_to(output).as_posix() not in owned]
        try:
            with generation_guard(operation,media,[output],{'plan_sha256':digest(declared)},root=ROOT,
                    procedural_inputs=[*configs(),*metadata],code_paths=dependencies(),production_plan=declared,
                    sample_output_paths=[output/'images'],auxiliary_output_paths=[output/'contact_sheet.jpg'],
                    replace_outputs=True,replace_verifier=lambda *_:{'preserved_paths':preserved}) as run:
                stage=publisher.prepare(output,run.token);publisher.copy_unowned(output,stage,owned)
                for name in ('images','labels','annotations'):(stage/name).mkdir(exist_ok=True)
                records=[];previews=[]
                take={round(i*(len(declared['strata'])-1)/19) for i in range(20)}
                for index,state in enumerate(declared['strata']):
                    p=state['parameters']
                    if cached is not None:
                        row={k:v for k,v in cached[state['id']].items() if k!='rendered_image'}
                        origin=ROOT/cached[state['id']]['rendered_image']
                        import os
                        os.link(origin,stage/row['image'])
                        base=origin.parent.parent
                        shutil.copy2(base/row['pose_label'],stage/row['pose_label'])
                        shutil.copy2(base/row['annotation'],stage/row['annotation'])
                        image=Image.open(origin).convert('RGB') if index in take else None
                    elif p['branch']=='direct_task_pose':image,row=direct_record(state,split,source)
                    else:
                        raw=foreground[p['foreground_id']] if 'foreground_id' in p else None
                        image,row=compose_record(state,raw,backgrounds[p['background_key']],recipe,photometry)
                    row['split']=split
                    if cached is None:image.save(stage/row['image'],compress_level=1)
                    values=row['label_values']
                    if cached is None:
                        (stage/row['pose_label']).write_text((' '.join(f'{v:.8f}' for v in values)+'\n') if values else '',encoding='utf8')
                        write(stage/row['annotation'],row)
                    records.append(row)
                    if index in take:
                        marked=draw_points(image,values) if image.size==(640,480) else image
                        previews.append((row['id'],marked))
                    if (index+1)%250==0:print('COMPOSITE_PROGRESS',split,index+1,'/',len(declared['strata']),flush=True)
                with (stage/'samples.jsonl').open('w',encoding='utf8') as stream:
                    for row in records:stream.write(json.dumps(row)+'\n')
                sheet=Image.new('RGB',(1600,4*265),'white');draw=ImageDraw.Draw(sheet)
                for index,(sid,image) in enumerate(previews):
                    x=index%5*320;y=index//5*265;sheet.paste(ImageOps.pad(image,(320,240),color='white'),(x,y+25));draw.text((x+3,y+4),sid,fill='black')
                sheet.save(stage/'contact_sheet.jpg',quality=94)
                summary=audit_records(stage,records)
                summary['branches']=dict(Counter(r['parameters']['branch'] for r in records))
                summary['axes']={axis:dict(Counter(str(r['parameters'].get(axis,'not_applicable')) for r in records)) for axis in ('reading_sampling','saturation','exposure','light_relation')}
                files=[p.relative_to(stage).as_posix() for p in stage.rglob('*') if p.is_file() and relative(output/p.relative_to(stage)) not in preserved]
                write(stage/'audit.json',summary)
                write(stage/'manifest.json',{'task':'meter','split':split,'status':'completed_working_dataset' if not pilot else 'pilot',
                      'plan':f'datasets/scripts/meter/config/full_{split}_plan.json','generation_plan':declared,
                      'samples':'samples.jsonl','summary':summary,'owned_files':files+['audit.json','manifest.json'],
                      'production_token':run.token,'scope':'Current working dataset; shared Blender assets; no formal release.'})
                publisher.publish(output,stage,run.token)
                states={s['id']:s for s in run.plan['strata']}
                for row in records:
                    state=states[row['id']];run.record_sample(row['id'],output=output/row['image'],sources=state['sources'],parameters=state['parameters'],actual=row)
                run.set_result(summary)
            publisher.recover(output,committed)
        except BaseException:
            publisher.recover(output,committed);raise
    print('EXPORT_COMPLETE',split,summary,flush=True)
    return summary


def audit_records(folder,records):
    hashes=set();duplicates=0;points=0;empty=0;centers=[];widths=[]
    for row in records:
        image=Image.open(folder/row['image']);image.load()
        values=list(map(float,(folder/row['pose_label']).read_text().split()))
        if values:
            if len(values)!=29 or values[0]!=0:raise ValueError('Invalid Pose label '+row['id'])
            if any(not 0<=v<=1 for v in values[1:5]):raise ValueError('Invalid normalized box')
            for i in range(8):
                x,y,v=values[5+3*i:8+3*i]
                if v not in (1,2) or not (0<=x<1 and 0<=y<1):raise ValueError('Invalid keypoint')
                points+=1
            centers.append(values[1]);widths.append(values[3]*image.width)
        else:empty+=1
        sha=hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()
        duplicates+=sha in hashes;hashes.add(sha)
        if len(values)!=len(row['label_values']) or any(abs(a-b)>1e-8 for a,b in zip(values,row['label_values'])):
            raise ValueError('Published label differs from annotation')
    if duplicates:raise ValueError(f'Exact RGB duplicates in {folder}: {duplicates}')
    return {'status':'passed','images':len(records),'instances':len(records)-empty,'empty_negative':empty,
            'keypoints':points,'rgb_duplicates':duplicates,'dial_center_x_range':([min(centers),max(centers)] if centers else None),
            'dial_width_px_range':([min(widths),max(widths)] if widths else None)}


def finish_dataset(recipe):
    output=ROOT/'datasets/exports/meter'
    manifests={split:read(output/split/'manifest.json') for split in SPLITS}
    seen={};duplicate=0
    for split in SPLITS:
        for line in (output/split/'samples.jsonl').read_text().splitlines():
            row=json.loads(line);image=Image.open(output/split/row['image']).convert('RGB')
            sha=hashlib.sha256(image.tobytes()).hexdigest()
            if sha in seen and seen[sha]!=split:duplicate+=1
            seen[sha]=split
    if duplicate:raise ValueError('Cross-split identical RGB images detected')
    metadata=[p for split in SPLITS for p in (output/split/'manifest.json',output/split/'samples.jsonl')]
    from datasets.scripts.meter.source_index import build_index
    build_index()
    with generation_guard('meter.full.dataset_index',[],[output/name for name in ('data.yaml','manifest.json')],
            {'images':sum(m['summary']['images'] for m in manifests.values())},root=ROOT,
            procedural_inputs=[*configs(),*metadata],code_paths=dependencies(),
            replace_outputs=True,replace_verifier=lambda *_:True) as run:
        if run.reused:return
        write(output/'data.yaml',{'names':['meter_dial'],'kpt_shape':[8,3],'flip_idx':list(range(8)),
                             'train':'train/images','val':'val/images','test':'test/images'})
        write(output/'manifest.json',{'task':'meter','status':'complete_current_working_dataset','splits':{s:{'manifest':f'{s}/manifest.json',**manifests[s]['summary']} for s in SPLITS},
          'images':sum(m['summary']['images'] for m in manifests.values()),'cross_split_rgb_duplicates':duplicate,
          'scope':'Shared Blender assets and known native captures; no formal release tag.'})
        run.set_result({'images':sum(m['summary']['images'] for m in manifests.values()),'cross_split_rgb_duplicates':duplicate})


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--pilot',action='store_true');ap.add_argument('--split',choices=SPLITS)
    ap.add_argument('--finalize-only',action='store_true');args=ap.parse_args()
    _,source=load_task_config('meter');recipe=read(ROOT/source['full_recipe'])
    if args.prepare:print(json.dumps(compile_plans(),ensure_ascii=False));return
    if args.finalize_only:finish_dataset(recipe);return
    if args.pilot:
        full=read(ROOT/'datasets/scripts/meter/config/full_train_plan.json')
        candidates=[s for s in full['strata'] if s['parameters']['branch']=='blender_background_composite']
        indices=[round(i*((len(candidates)//2)-1)/5)*2 for i in range(6)]
        selected=[candidates[i+j] for i in indices for j in (0,1)]
        raw=render_foregrounds('train',recipe,source,selected)
        inventory=read(ROOT/full['background_inventory'])['backgrounds']
        export_split('train',make_plan('train',selected,'Full production pilot'),raw,inventory,recipe,source,pilot=True)
        return
    for split in ([args.split] if args.split else SPLITS):
        inventory=render_backgrounds(split,recipe,source)
        rendered=render_foregrounds(split,recipe,source)
        declared=read(ROOT/f'datasets/scripts/meter/config/full_{split}_plan.json')
        export_split(split,declared,rendered,inventory,recipe,source)
    if all((ROOT/f'datasets/exports/meter/{s}/manifest.json').is_file() for s in SPLITS):finish_dataset(recipe)


if __name__=='__main__':main()
