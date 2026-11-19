"""Build failure-directed research inputs without modifying current samples/exports."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

import cv2
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from datasets.scripts.e_bike import compose_v4_grounded_train as ground
from datasets.scripts.common.config import load_task_config
from datasets.scripts.common.images import read_image,write_image
from lab.features.e_bike.training.scripts.keypoint_experiment import read,write
from lab.features.e_bike.training.scripts.two_stage_experiment import truths
from lab.training_common.engine.runtime import setup_paths,prepare_ultralytics


def safe_crop(box,objects,width,height):
    """Expand a crop until every intersecting annotated vehicle is wholly included."""
    box=np.array([np.floor(box[0]),np.floor(box[1]),np.ceil(box[2]),np.ceil(box[3])],dtype=float)
    for _ in range(len(objects)+1):
        before=box.copy()
        for obj in objects:
            b=np.asarray(obj['xyxy'])
            if min(box[2],b[2])>max(box[0],b[0]) and min(box[3],b[3])>max(box[1],b[1]):
                box[:2]=np.minimum(box[:2],b[:2]-2);box[2:]=np.maximum(box[2:],b[2:]+2)
        box[:2]=np.maximum(box[:2],0);box[2:]=np.minimum(box[2:],[width,height])
        box=np.array([np.floor(box[0]),np.floor(box[1]),np.ceil(box[2]),np.ceil(box[3])])
        if np.array_equal(before,box):break
    return [int(np.floor(box[0])),int(np.floor(box[1])),int(np.ceil(box[2])),int(np.ceil(box[3]))]


def contains_no_vehicle(rect,objects):
    x1,y1,x2,y2=rect
    return all(min(x2,b['xyxy'][2]+2)<=max(x1,b['xyxy'][0]-2) or
               min(y2,b['xyxy'][3]+2)<=max(y1,b['xyxy'][1]-2) for b in objects)


def crop_resize(frame,objects,rect):
    x1,y1,x2,y2=rect;patch=frame[y1:y2,x1:x2];h,w=patch.shape[:2]
    scale=min(640/w,480/h);rw,rh=round(w*scale),round(h*scale);left,top=(640-rw)//2,(480-rh)//2
    canvas=np.full((480,640,3),114,np.uint8)
    canvas[top:top+rh,left:left+rw]=cv2.resize(patch,(rw,rh))
    mapped=[]
    for obj in objects:
        bx1,by1,bx2,by2=obj['xyxy']
        if bx2<=x1 or by2<=y1 or bx1>=x2 or by1>=y2:continue
        epsilon=max(frame.shape[:2])*1e-6  # normalized six-decimal source label roundoff
        if not (bx1>=x1-epsilon and by1>=y1-epsilon and bx2<=x2+epsilon and by2<=y2+epsilon):raise ValueError('Crop truncates annotated vehicle')
        bx1,by1,bx2,by2=max(bx1,x1),max(by1,y1),min(bx2,x2),min(by2,y2)
        mapped.append({'class_id':obj['class_id'],'xyxy':[(bx1-x1)*rw/w+left,(by1-y1)*rh/h+top,
                                                         (bx2-x1)*rw/w+left,(by2-y1)*rh/h+top]})
    return canvas,mapped


def yolo_text(objects,w,h,one_class=False):
    return ''.join(f'{0 if one_class else o["class_id"]} {(o["xyxy"][0]+o["xyxy"][2])/2/w:.10f} {(o["xyxy"][1]+o["xyxy"][3])/2/h:.10f} {(o["xyxy"][2]-o["xyxy"][0])/w:.10f} {(o["xyxy"][3]-o["xyxy"][1])/h:.10f}\n' for o in objects)


def preview(frame,objects,title):
    image=frame.copy()
    for obj in objects:
        x1,y1,x2,y2=map(round,obj['xyxy']);cv2.rectangle(image,(x1,y1),(x2,y2),(40,220,40) if obj['class_id']==0 else (0,160,255),1)
    image=cv2.copyMakeBorder(image,26,0,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
    cv2.putText(image,title,(5,18),0,.45,(255,255,255),1);return image


def audit_focused(exp):
    manifest=read(exp/'manifest.json');recipe=read(exp/'recipe_snapshot.json');old=read(ROOT/recipe['old_inputs'])
    old_train={r['image']:r for r in old['records'] if r['split']=='train'}
    old_val={r['image']:r for r in old['records'] if r['split']=='val'}
    family_splits={};visibility=[];relations=Counter();classes=Counter();hashes={};negative_scores=[]
    for row in old_train.values():
        key=hashlib.sha256(read_image(ROOT/row['image']).tobytes()).hexdigest()
        hashes.setdefault(key,[]).append(('train',row['image']))
    for g in manifest['generated']:
        frame=read_image(ROOT/g['image']);h,w=frame.shape[:2]
        objects=truths(next(r for r in manifest['records'] if r['image']==g['image']))
        assert len(objects)==len(g['objects'])
        for a,b in zip(objects,g['objects']):
            assert a['class_id']==b['class_id'] and np.max(np.abs(np.asarray(a['xyxy'])-b['xyxy']))<1e-6
            classes[(g['split'],a['class_id'])]+=1
        key=hashlib.sha256(frame.tobytes()).hexdigest();hashes.setdefault(key,[]).append((g['split'],g['image']))
        d=g['details']
        if g['kind']=='focused_composite':
            family_splits.setdefault(d['family_id'],set()).add(g['split'])
            masks=[]
            for i,obj in enumerate(d['mask_geometry']):
                mask=read_image(exp/'masks'/f'{Path(g["image"]).stem}_{i:02d}.png',cv2.IMREAD_GRAYSCALE)>0
                yy,xx=np.nonzero(mask);assert [int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1]==obj['bbox_xyxy']
                assert int(mask.sum())==obj['visible_pixels'];masks.append(mask)
                visibility.append(obj['visible_fraction'])
                for occluder in obj['occluded_by']:
                    if occluder['fraction']>=.05:relations[occluder['class']+'_over_'+obj['class']]+=1
            assert np.max(np.stack(masks).sum(0))<=1
        else:
            parent=d['parent_image'];assert parent in old_train and parent not in old_val
            if g['kind']=='mined_vehicle_free_background':
                assert contains_no_vehicle(d['crop_xyxy'],truths(old_train[parent]));assert not objects
                negative_scores.append(d['miner_score'])
    assert all(len(v)==1 for v in family_splits.values())
    cross=[v for v in hashes.values() if len({s for s,_ in v})>1];assert not cross,'Exact new train/holdout pixel duplicate'
    for r in old_val.values():
        copied=exp/'data/val/images'/Path(r['image']).name
        assert hashlib.sha256(copied.read_bytes()).digest()==hashlib.sha256((ROOT/r['image']).read_bytes()).digest()
    audit={'generated_images':len(manifest['generated']),'layout_families_by_split':dict(Counter(next(iter(v)) for v in family_splits.values())),
           'classes':{f'{s}:{c}':n for (s,c),n in classes.items()},'minimum_composite_visibility':min(visibility),
           'composite_objects_below_previous070':sum(v<.7 for v in visibility),'occlusion_relations_instances':dict(relations),
           'new_cross_split_exact_pixel_duplicates':len(cross),'pixel_duplicate_audit_includes_original_train':True,'old26_image_bytes_unchanged':True,
           'training_negative_parents_verified':len(negative_scores),'negative_miner_scores':{'minimum':min(negative_scores),'maximum':max(negative_scores)},
           'scope':'Machine geometry/source audit, not new human confirmation; new layouts share real foreground/background parents.'}
    write(exp/'audit.json',audit);return audit


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',default='runs/e_bike/focused_inputs_20261118')
    p.add_argument('--resume-partial',action='store_true',help='Rebuild an incomplete batch, never an existing complete manifest')
    args=p.parse_args();exp=(ROOT/args.experiment).resolve()
    if not exp.is_relative_to(ROOT/'runs/e_bike'):raise ValueError('Research output must be under runs/e_bike')
    _,source=load_task_config('e_bike');recipe=read(ROOT/source['focused_recipe']);rng=random.Random(recipe['seed'])
    if args.resume_partial:
        if (exp/'manifest.json').exists() or (exp/'data/data.yaml').exists():raise ValueError('Partial rebuild refuses a complete input batch')
        if read(exp/'recipe_snapshot.json')!=recipe:raise ValueError('Partial batch recipe differs')
    exp.mkdir(parents=True,exist_ok=args.resume_partial);cv2.setNumThreads(1);torch.set_num_threads(2)
    write(exp/'recipe_snapshot.json',recipe);(exp/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    original=read(ROOT/recipe['old_inputs']);records=copy.deepcopy(original['records']);detector=[];generated=[];rejections=Counter()
    for split in ('train','val','holdout'):
        for folder in ('images','labels'):(exp/'data'/split/folder).mkdir(parents=True,exist_ok=True)
        (exp/'posture_labels'/split).mkdir(parents=True,exist_ok=True)
    (exp/'previews').mkdir(exist_ok=True);(exp/'masks').mkdir(exist_ok=True)
    for r in original['records']:
        image=exp/'data'/r['split']/'images'/Path(r['image']).name;label=image.parent.parent/'labels'/(image.stem+'.txt')
        shutil.copyfile(ROOT/r['image'],image)
        text=(ROOT/r['label']).read_text(encoding='utf-8');label.write_text(''.join('0 '+' '.join(l.split()[1:])+'\n' for l in text.splitlines()),encoding='utf-8')
        detector.append({**r,'image':image.relative_to(ROOT).as_posix(),'label':label.relative_to(ROOT).as_posix()})

    def emit(frame,objects,split,kind,details,template=None):
        stem=f'focus_{len(generated):05d}';image=exp/'data'/split/'images'/(stem+'.png')
        label=exp/'posture_labels'/split/(stem+'.txt');single=exp/'data'/split/'labels'/(stem+'.txt');h,w=frame.shape[:2]
        write_image(image,frame);label.write_text(yolo_text(objects,w,h),encoding='utf-8');single.write_text(yolo_text(objects,w,h,True),encoding='utf-8')
        row=copy.deepcopy(template or original['records'][0]);row.update(image=image.relative_to(ROOT).as_posix(),label=label.relative_to(ROOT).as_posix(),split=split,
            workspace_image=image.relative_to(ROOT).as_posix(),workspace_label=label.relative_to(ROOT).as_posix(),label_status='reviewed',
            review_state='complete_positive' if objects else 'complete_negative',review_evidence=(exp/'manifest.json').relative_to(ROOT).as_posix(),
            source_review_evidence=(exp/'manifest.json').relative_to(ROOT).as_posix(),generation_method='composite' if kind=='focused_composite' else 'crop',
            source_kind='derived_image',condition=kind,source_record={'branch':kind,'image_size':[w,h],'objects':objects,'research_details':details},
            label_review='machine_geometry_checked_no_new_human_confirmation')
        if split=='holdout':row['evaluation_group']='new_layout_holdout'
        records.append(row);generated.append({'image':row['image'],'label':row['label'],'split':split,'kind':kind,'details':details,'objects':objects})
        if split!='holdout':detector.append({**row,'label':single.relative_to(ROOT).as_posix(),'workspace_label':single.relative_to(ROOT).as_posix()})
        write_image(exp/'previews'/(stem+'.jpg'),preview(frame,objects,f'{stem} {split} {kind} N={len(objects)}'))
        return stem

    backgrounds=ground.bind_background_reviews(read(ROOT/source['ground_background_manifest'])['backgrounds'],'train',source)
    _,vehicles=ground.bike.sources(selection=ROOT/source['ground_vehicle_selection'],include_people=False)
    total=recipe['train_families_per_background']+recipe['holdout_families_per_background']
    for bg_index,bg in enumerate(backgrounds):
        for family in range(total):
            split='train' if family<recipe['train_families_per_background'] else 'holdout';accepted=None
            for attempt in range(recipe['maximum_attempts_per_family']):
                queue=bg['layout'].get('queue',[(160,320,40),(205,342,52),(252,368,68),(302,400,88),(365,434,112),(438,458,135)])
                if bg['layout'].get('max_vehicles')==2 and not bg['layout'].get('queue'):queue=bg['layout']['clear']
                overlap=family%2==1
                if overlap:
                    cx,cy,size=bg['layout'].get('occlusion',queue[len(queue)//2])
                    slots=[(cx,cy,size),(cx+rng.choice((-1,1))*rng.randint(18,58),cy+rng.randint(10,22),round(size*1.15))]
                else:slots=rng.sample(queue,min(len(queue),rng.choice((3,4,5,6,7))))
                classes=[rng.choice(('upright','fallen')) for _ in slots]
                if overlap:classes[:2]=[('upright','fallen')[(family//2)%2],('upright','fallen')[(family//4)%2]]
                views=[rng.choice(vehicles[c])[0]['path'] for c in classes]
                base_actors=[{'foot_x':x+rng.randint(-10,10),'foot_y':y+rng.randint(-3,3),
                              'size_px':round(size*rng.uniform(.92,1.14)),
                              'yaw':rng.uniform(-4,4),'pitch':rng.uniform(-2,2),'roll':rng.uniform(-2,2),
                              'relative_exposure':rng.uniform(.94,1.04),'mirrored':False} for x,y,size in slots]
                states=[];rendered=[]
                try:
                    for swap in ((False,True) if recipe['posture_swap'] else (False,)):
                        targets=[];actors=[]
                        for k,(x,y,size) in enumerate(slots):
                            cls=('fallen' if classes[k]=='upright' else 'upright') if swap else classes[k]
                            path=rng.choice(vehicles[cls])[0]['path'] if swap else views[k]
                            targets.append({'class':cls,'source':path})
                            actor=copy.deepcopy(base_actors[k]);actor['size_px']=round(actor['size_px']*(1.16 if cls=='fallen' else 1));actors.append(actor)
                        state={'background':{'id':bg['id']},'target':{'vehicles':targets},'parameters':{'actors':actors,'value_gain':1.,'saturation_gain':1.},'augmentation':{'tone':'original'}}
                        for mirror in ((False,True) if recipe['source_mirror_pairs'] else (False,)):
                            copy_state=copy.deepcopy(state)
                            for a in copy_state['parameters']['actors']:a['mirrored']=mirror
                            ground.render_ground_state(copy_state,backgrounds,vehicles,render_rgb=False,minimum_visible_fraction=recipe['minimum_visible_fraction'])
                            states.append((copy_state,swap,mirror))
                    accepted=states;break
                except ValueError as error:rejections[str(error)]+=1
            if accepted is None:
                print(json.dumps({'background':bg['id'],'family':family,'status':'no_feasible_state'}),flush=True);continue
            for state,swap,mirror in accepted:
                frame,objects,masks,_=ground.render_ground_state(state,backgrounds,vehicles,minimum_visible_fraction=recipe['minimum_visible_fraction'])
                converted=[{'class_id':('upright','fallen').index(o['class']),'xyxy':o['bbox_xyxy']} for o in objects]
                details={'family_id':f'{bg["id"]}_{family:03d}','background':bg['image'],'parent_group':bg['parent_group'],
                         'posture_swap':swap,'source_mirrored':mirror,'generation_state':state,'mask_geometry':objects}
                template=next((r for r in original['records'] if r['split']=='train' and r['source_record'].get('background',{}).get('id')==bg['id']),original['records'][0])
                stem=emit(frame,converted,split,'focused_composite',details,template)
                for i,mask in enumerate(masks):
                    if mask is not None:write_image(exp/'masks'/f'{stem}_{i:02d}.png',mask.astype(np.uint8)*255)
            print(json.dumps({'background':bg['id'],'family':family,'split':split,'generated':len(generated)}),flush=True)

    direct=[r for r in original['records'] if r['split']=='train' and r['source_record']['branch']=='direct_real']
    negative_candidates=[]
    for r in direct:
        frame=read_image(ROOT/r['image']);h,w=frame.shape[:2];objects=truths(r)
        for number in range(recipe['real_crops_per_image']):
            target=objects[(number+len(objects)//2)%len(objects)]['xyxy'];x1,y1,x2,y2=target
            dx,dy=(x2-x1)*(.35+.25*number),(y2-y1)*(.35+.25*number)
            rect=safe_crop([x1-dx,y1-dy,x2+dx,y2+dy],objects,w,h)
            if (rect[2]-rect[0])*(rect[3]-rect[1])>=w*h*.85:continue
            patch,mapped=crop_resize(frame,objects,rect)
            emit(patch,mapped,'train','complete_real_crop',{'parent_image':r['image'],'crop_xyxy':rect},r)
        for bw,bh in ((240,160),(180,120),(320,200)):
            for x in (0,(w-bw)//2,w-bw):
                for y in (0,max(0,h//3-bh//2),max(0,h-bh)):
                    rect=[x,y,x+bw,y+bh]
                    if x>=0 and y>=0 and x+bw<=w and y+bh<=h and contains_no_vehicle(rect,objects):
                        negative_candidates.append((r,rect))
    rng.shuffle(negative_candidates);unique=[];seen=set()
    for r,rect in negative_candidates:
        patch,_=crop_resize(read_image(ROOT/r['image']),[],rect);key=hashlib.sha256(patch.tobytes()).hexdigest()
        if key not in seen:unique.append((r,rect));seen.add(key)
        if len(unique)>=recipe['negative_candidate_limit']:break
    negative_candidates=unique
    setup_paths(exp);YOLO=prepare_ultralytics(exp);miner=YOLO(str(ROOT/recipe['initial_weights']));ranked=[]
    for index,(r,rect) in enumerate(negative_candidates):
        patch,_=crop_resize(read_image(ROOT/r['image']),[],rect)
        pred=miner.predict(source=patch,imgsz=640,conf=.01,iou=.3,device='0',rect=True,verbose=False)[0]
        score=float(pred.boxes.conf.max()) if len(pred.boxes) else 0.
        ranked.append((score,index))
    for score,index in sorted(ranked,reverse=True)[:recipe['hard_negative_count']]:
        r,rect=negative_candidates[index];patch,_=crop_resize(read_image(ROOT/r['image']),[],rect)
        emit(patch,[],'train','mined_vehicle_free_background',{'parent_image':r['image'],'crop_xyxy':rect,'miner_score':score,'evidence':'No intersection with any complete GT bbox, expanded2px'},r)
    manifest={'model_key':'e_bike','records':records,'generated':generated,'recipe':source['focused_recipe'],
              'summary':{'records_by_split':dict(Counter(r['split'] for r in records)),
                         'generated_by_kind':dict(Counter(r['kind'] for r in generated)),'rejections':dict(rejections)},
              'scope':recipe['scope'],'annotation_method':'Existing labels inherited; synthetic visible masks and GT-free crops machine checked. No new human confirmation.'}
    write(exp/'manifest.json',manifest)
    write(exp/'data/workspace_sources.json',{'model_key':'e_bike','selection_mode':'manifest_only','records':detector,'source_reuse':[]})
    write(exp/'data/data.yaml',{'names':['e_bike'],'train':'train/images','val':'val/images'})
    evaluation=[{**r,'split':'val'} for r in records if r['split'] in ('val','holdout')]
    write(exp/'evaluation_snapshot.json',{'records':evaluation,'scope':recipe['scope']})
    config=yaml.safe_load((ROOT/'runs/e_bike/threshold_nms_probe_20261118/finetune.yaml').read_text(encoding='utf-8'))
    config.update(model=recipe['initial_weights'],data=(exp/'data/data.yaml').relative_to(ROOT).as_posix(),epochs=recipe['epochs'],
                  description='Focused new synthetic, complete real crops and mined vehicle-free negatives; original26 validation only')
    config['source_notes']=[recipe['scope'],
        'Actual added kinds/splits are recorded in the research input manifest; original train and val retained.',
        'New layout holdout excluded from training and epoch selection; evaluated after training.',
        f'Epoch cap{recipe["epochs"]}; external wall{recipe["wall_seconds"]}s; original26 validation used for early stopping only.']
    (exp/'train.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
    figures=''.join(f'<figure><figcaption>{g["split"]} {g["kind"]}</figcaption><a href="previews/{Path(g["image"]).stem}.jpg"><img loading="lazy" src="previews/{Path(g["image"]).stem}.jpg"></a></figure>' for g in generated)
    (exp/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>定向数据小样</title><style>body{font:16px sans-serif}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}figure{margin:0}img{width:100%}</style><h1>定向研究数据：可见框／实拍完整裁片／训练来源硬负例</h1><p>旧26图不参与新增训练数据；同一合成布局的姿态交换与镜像整组分集。新留出仍共享物料和底图，非独立实拍。</p><div class="grid">'+figures+'</div>',encoding='utf-8')
    selected=[]
    for kind in ('focused_composite','complete_real_crop','mined_vehicle_free_background'):
        values=[g for g in generated if g['kind']==kind];selected+=values[::max(1,len(values)//4)][:4]
    tiles=[cv2.resize(read_image(exp/'previews'/(Path(g['image']).stem+'.jpg')),(320,253)) for g in selected]
    while len(tiles)%4:tiles.append(np.zeros_like(tiles[0]))
    write_image(exp/'contact.jpg',np.concatenate([np.concatenate(tiles[i:i+4],axis=1) for i in range(0,len(tiles),4)],axis=0))
    audit_focused(exp)
    print(json.dumps(manifest['summary'],ensure_ascii=False),flush=True)


if __name__=='__main__':main()
