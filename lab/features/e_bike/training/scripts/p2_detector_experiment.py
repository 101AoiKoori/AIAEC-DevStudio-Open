"""Bounded P2/P3 detector comparison with identical backbone initialization."""
import argparse
from collections import Counter
import copy
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from datasets.scripts.common.config import load_task_config
from datasets.scripts.common.images import read_image,write_image
from lab.training_common.engine.runtime import setup_paths,prepare_ultralytics
from lab.training_common.engine.bounded import run_bounded
from lab.training_common.engine.reporting import box_iou,match_detections
from lab.features.e_bike.training.scripts.candidate_verifier_experiment import family_key,finish_pool
from lab.features.e_bike.training.scripts.two_stage_experiment import crop,truths
from lab.features.e_bike.training.scripts.input_scale_experiment import make_predictor
from lab.features.e_bike.training.scripts.evaluation_audit import image_metrics,optimal_pairs
from lab.features.e_bike.training.scripts.data_ablation_experiment import error_overlay
from lab.features.e_bike.training.scripts.part_hull_experiment import diagnostic


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def select_fit(records,real_count,synthetic_count):
    selected=[];families=set();backgrounds=set()
    for real,count in ((True,real_count),(False,synthetic_count)):
        candidates=sorted([r for r in records if bool(r['source_record'].get('source_image'))==real],key=lambda r:-len(truths(r)))
        chosen=[]
        for row in candidates:
            key=family_key(row);background=row['source_record'].get('background',{}).get('id')
            if key in families or (not real and background in backgrounds):continue
            chosen.append(row);families.add(key);backgrounds.add(background)
            if len(chosen)==count:break
        if len(chosen)<count:
            for row in candidates:
                key=family_key(row)
                if key in families:continue
                chosen.append(row);families.add(key)
                if len(chosen)==count:break
        if len(chosen)!=count:raise ValueError('Insufficient distinct training families for fit diagnostic')
        selected.extend(chosen)
    return selected


def prepare(exp):
    _,source=load_task_config('e_bike');recipe=read(ROOT/source['p2_recipe'])
    exp.mkdir(parents=True,exist_ok=False);write(exp/'recipe_snapshot.json',recipe)
    (exp/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    original=read(ROOT/recipe['inputs']);train=[r for r in original['records'] if r['split']=='train']
    fit=select_fit(train,recipe['fit_real_images'],recipe['fit_synthetic_images']);snapshots={}
    for stage,records in (('fit',fit),('full',original['records'])):
        root=exp/('data_'+stage);out=[]
        for r in records:
            row=copy.deepcopy(r);split=row['split'];image=root/'images'/split/Path(row['image']).name;label=root/'labels'/split/Path(row['label']).name
            image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/row['image'],image)
            lines=(ROOT/row['label']).read_text(encoding='utf-8').splitlines()
            label.write_text('\n'.join('0 '+' '.join(line.split()[1:]) for line in lines if line.strip())+'\n',encoding='utf-8')
            assert hashlib.sha256(image.read_bytes()).digest()==hashlib.sha256((ROOT/row['image']).read_bytes()).digest()
            row['image']=image.relative_to(ROOT).as_posix();row['detector_label']=label.relative_to(ROOT).as_posix()
            out.append(row)
        repeats=recipe.get('fit_repeats',1) if stage=='fit' else 1
        if stage=='fit' and repeats>1:
            for row in out:
                original_image=ROOT/row['image'];original_label=ROOT/row['detector_label']
                for i in range(1,repeats):
                    shutil.copyfile(original_image,original_image.with_stem(original_image.stem+f'__repeat_{i:02d}'))
                    shutil.copyfile(original_label,original_label.with_stem(original_label.stem+f'__repeat_{i:02d}'))
                for kind,path in (('images',original_image),('labels',original_label)):
                    target=root/kind/'val'/path.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        header={'path':str(root.resolve()),'train':'images/train','val':'images/train' if stage=='fit' and repeats==1 else 'images/val','names':{0:'e_bike'}}
        (exp/(stage+'.yaml')).write_text(yaml.safe_dump(header,allow_unicode=True),encoding='utf-8')
        write(exp/(stage+'_snapshot.json'),{'records':out,'scope':'Training fit; evaluation on these same8 images.' if stage=='fit' else recipe['scope']});snapshots[stage]=len(out)
    write(exp/'evaluation_snapshot.json',read(ROOT/recipe['evaluation_inputs']))
    write(exp/'preparation.json',{'snapshots':snapshots,'fit_images':[r['image'] for r in fit],
        'fit_instances':sum(len(truths(r)) for r in fit),'fit_families':len({family_key(r) for r in fit}),
        'fit_train_slots':len(fit)*recipe.get('fit_repeats',1),'fit_unique_images':len(fit),
        'original_train':len(train),'full_original_val':len(original['records'])-len(train),'scope':recipe['scope']})
    print(json.dumps(read(exp/'preparation.json'),ensure_ascii=False),flush=True)


def train_child(exp,stage,arm):
    folder=exp/stage/arm;folder.mkdir(parents=True,exist_ok=False);setup_paths(folder)
    YOLO=prepare_ultralytics(folder)
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    recipe=read(exp/'recipe_snapshot.json');source=YOLO(str(ROOT/recipe['common_backbone'])).model.cpu().state_dict()
    class CommonBackboneTrainer(DetectionTrainer):
        def optimizer_step(self):
            super().optimizer_step()
            if stage=='full':
                if not hasattr(self,'optimization_trace'):self.optimization_trace=[]
                self.optimization_trace.append({'epoch':self.epoch+1,'accumulate':self.accumulate,'lr':[float(g['lr']) for g in self.optimizer.param_groups]})

        def get_model(self,cfg=None,weights=None,verbose=True):
            model=super().get_model(cfg,weights=None,verbose=verbose);target=model.state_dict()
            shared={k:v for k,v in source.items() if k.startswith('model.') and k.split('.')[1].isdigit() and int(k.split('.')[1])<=9 and k in target and v.shape==target[k].shape}
            if not shared:raise ValueError('No compatible common backbone tensors')
            model.load_state_dict(shared,strict=False)
            digest=hashlib.sha256()
            for key in sorted(shared):digest.update(key.encode());digest.update(shared[key].numpy().tobytes())
            write(folder/'initialization.json',{'backbone_tensors':len(shared),'backbone_sha256':digest.hexdigest(),
                'neck_and_head_transferred':False,'parameters':sum(p.numel() for p in model.parameters()),
                'strides':model.stride.tolist(),'architecture':recipe['architectures'][arm]})
            return model
    torch.manual_seed(recipe['seed']);model=YOLO(recipe['architectures'][arm])
    options=dict(data=str(exp/(stage+'.yaml')),epochs=recipe[stage+'_epochs'],imgsz=recipe['imgsz'],batch=recipe['batch'],nbs=recipe['nominal_batch'],
        optimizer='AdamW',lr0=recipe[stage+'_lr'],lrf=.1,cos_lr=True,warmup_epochs=1,warmup_bias_lr=0.,
        seed=recipe['seed'],deterministic=True,amp=False,workers=0,device='0',patience=0,
        mosaic=0.,mixup=0.,copy_paste=0.,degrees=0.,translate=0.,scale=0.,shear=0.,perspective=0.,fliplr=0.,flipud=0.,hsv_h=0.,hsv_s=0.,hsv_v=0.,
        close_mosaic=0,plots=False,save=True,save_period=-1,project=str(folder),name='training',exist_ok=False)
    write(folder/'options.json',options);started=time.monotonic();model.train(trainer=CommonBackboneTrainer,**options)
    if stage=='full':write(folder/'optimization_trace.json',model.trainer.optimization_trace)
    write(folder/'status.json',{'status':'complete','epochs':options['epochs'],'seconds':time.monotonic()-started,
         'last_weights':(folder/'training/weights/last.pt').relative_to(ROOT).as_posix(),'scope':recipe['scope']})


def train_stage(exp,stage):
    recipe=read(exp/'recipe_snapshot.json')
    for arm in recipe['architectures']:
        folder=exp/stage/arm
        if (folder/'status.json').exists():
            status=read(folder/'status.json')
            if status['status']=='complete' and (ROOT/status['last_weights']).is_file():print(f'SKIP complete {stage} {arm}',flush=True);continue
        if folder.exists():raise RuntimeError(f'Partial existing training requires inspection: {folder}')
        print(f'START {stage} {arm}',flush=True)
        run_bounded([sys.executable,'-X','utf8','-B',str(Path(__file__)),'train-child','--experiment',exp.relative_to(ROOT).as_posix(),'--stage',stage,'--arm',arm],
                    time.monotonic()+recipe[stage+'_wall_seconds_per_arm'],exp/(stage+'_'+arm+'_train.log'),ROOT)
        print(f'COMPLETE {stage} {arm}',flush=True)


def evaluate(exp,stage):
    folder=exp/('evaluation_'+stage);folder.mkdir(exist_ok=False);setup_paths(folder);YOLO=prepare_ultralytics(folder)
    from ultralytics.utils import ops
    recipe=read(exp/'recipe_snapshot.json');classifier=YOLO(str(ROOT/recipe['semantic_classifier']))
    records=read(exp/'fit_snapshot.json')['records'] if stage=='fit' else read(exp/'evaluation_snapshot.json')['records']
    metrics={};allrows={};capacity={};initializations={};weights={};secondary_metrics={}
    write(folder/'postprocess_protocol.json',{'primary_iou':.3,'secondary_iou':.7,'confidence':.25,
        'reason':'Report both architectures at fixed strict and loose NMS; no full-validation threshold search. Capacity pass uses .7 too.'})
    for arm in recipe['architectures']:
        owner=exp/stage/arm;status=read(owner/'status.json');assert status['status']=='complete'
        detector=YOLO(str(ROOT/status['last_weights']));initializations[arm]=read(owner/'initialization.json');rows=[];secondary_rows=[];cap=Counter()
        weights[status['last_weights']]=hashlib.sha256((ROOT/status['last_weights']).read_bytes()).hexdigest()
        for record in records:
            frame=read_image(ROOT/record['image']);gt=truths(record);stem=Path(record['image']).stem
            simple=detector.predict(source=frame,imgsz=640,rect=True,conf=.25,iou=.7,max_det=100,device='0',verbose=False)[0]
            simple_pred=[{'xyxy':b} for b in simple.boxes.xyxy.cpu().tolist()];matches=optimal_pairs(gt,simple_pred,.5)
            small={i for i,g in enumerate(gt) if (g['xyxy'][2]-g['xyxy'][0])*(g['xyxy'][3]-g['xyxy'][1])<1024}
            cap.update(images=1,gt=len(gt),tp=len(matches),fp=len(simple_pred)-len(matches),fn=len(gt)-len(matches),
                       all_geometry_matched=int(len(matches)==len(gt)==len(simple_pred)),small_gt=len(small),small_tp=sum(i in small for i,_,_ in matches))
            detector.predict(source=frame,imgsz=640,rect=True,conf=.25,iou=.95,max_det=300,device='0',verbose=False,predictor=make_predictor())
            predictor=detector.predictor
            if not hasattr(predictor,'decoded_before_nms'):
                # A predictor already created by the direct capacity pass must be replaced explicitly.
                detector.predictor=None
                detector.predict(source=frame,imgsz=640,rect=True,conf=.25,iou=.95,max_det=300,device='0',verbose=False,predictor=make_predictor());predictor=detector.predictor
            input_boxes=ops.xywh2xyxy(predictor.decoded_before_nms[0,:4].T).numpy();ids=predictor.kept_raw_ids;pool=[]
            for start in range(0,len(ids),32):
                group=[i for i in ids[start:start+32] if predictor.raw_boxes[i][2]>predictor.raw_boxes[i][0] and predictor.raw_boxes[i][3]>predictor.raw_boxes[i][1]]
                if not group:continue
                results=classifier.predict(source=[crop(frame,predictor.raw_boxes[i])[0] for i in group],imgsz=160,device='0',verbose=False)
                for i,result in zip(group,results):pool.append({'raw_id':i,'xyxy':predictor.raw_boxes[i].tolist(),'input_xyxy':input_boxes[i].tolist(),
                    'raw_confidence':float(predictor.decoded_before_nms[0,4,i]),'verifier_probs':{classifier.names[k]:float(v) for k,v in enumerate(result.probs.data.cpu().tolist())}})
            found=finish_pool(pool,.25,.3,'joint');m,miss,extra=match_detections(gt,found,.5)
            group=record.get('evaluation_group') or ('synthetic_heldout' if record.get('condition') else 'original_real')
            if stage=='fit':group='fit_real' if record['source_record'].get('source_image') else 'fit_synthetic'
            row={'source_image':record['image'],'group':group,'ground_truth':gt,'predictions':found,'matches':m,'missed_gt_indices':miss,'false_prediction_indices':extra,**diagnostic(gt,found)}
            rows.append(row);target=folder/arm;target.mkdir(exist_ok=True);write_image(target/(stem+'.jpg'),error_overlay(row,stage+' '+arm))
            loose=finish_pool(pool,.25,.7,'joint');lm,lmiss,lextra=match_detections(gt,loose,.5)
            other={**row,'predictions':loose,'matches':lm,'missed_gt_indices':lmiss,'false_prediction_indices':lextra,**diagnostic(gt,loose)}
            secondary_rows.append(other);secondary_folder=target/'secondary070';secondary_folder.mkdir(exist_ok=True)
            write_image(secondary_folder/(stem+'.jpg'),error_overlay(other,stage+' '+arm+' NMS0.7'))
            pool_folder=target/'candidates';pool_folder.mkdir(exist_ok=True);write(pool_folder/(stem+'.json'),{'image':record['image'],'ground_truth':gt,'candidates':pool})
        (folder/arm/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        (folder/arm/'secondary070/predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in secondary_rows),encoding='utf-8')
        groups=('all_fit','fit_real','fit_synthetic') if stage=='fit' else ('old26','original_real','synthetic_heldout','new_layout_holdout')
        metrics[arm]={};secondary_metrics[arm]={}
        for group in groups:
            chosen=rows if group=='all_fit' else [r for r in rows if (r['group']!='new_layout_holdout' if group=='old26' else r['group']==group)]
            values=[image_metrics(r) for r in chosen];metrics[arm][group]={'images':len(values),**{k:sum(v[k] for v in values) for k in ('count_exact','strict_instance_exact','entity_tp','entity_fp','entity_fn','matched_pose_errors')},'duplicate_predictions':sum(r['duplicate_predictions'] for r in chosen)}
            secondary_chosen=secondary_rows if group=='all_fit' else [r for r in secondary_rows if (r['group']!='new_layout_holdout' if group=='old26' else r['group']==group)]
            sv=[image_metrics(r) for r in secondary_chosen];secondary_metrics[arm][group]={'images':len(sv),**{k:sum(v[k] for v in sv) for k in ('count_exact','strict_instance_exact','entity_tp','entity_fp','entity_fn','matched_pose_errors')},'duplicate_predictions':sum(r['duplicate_predictions'] for r in secondary_chosen)}
        allrows[arm]=rows;capacity[arm]=dict(cap)
    assert len({v['backbone_sha256'] for v in initializations.values()})==1
    summary={'stage':stage,'metrics':metrics,'secondary_metrics_nms070':secondary_metrics,'detector_capacity_conf025_nms070':capacity,'initialization':initializations,'weights':weights,
             'scope':'Training-fit same8 images only, not validation.' if stage=='fit' else recipe['scope']}
    write(folder/'summary.json',summary)
    cards=[]
    for row in allrows['P3']:
        stem=Path(row['source_image']).stem;cards.append('<h2>'+html.escape(stem)+'</h2><div class=grid>'+''.join(f'<figure><figcaption>{a}</figcaption><a href="{a}/{stem}.jpg"><img loading=lazy src="{a}/{stem}.jpg"></a></figure>' for a in recipe['architectures'])+'</div>')
    (folder/'review.html').write_text('<!doctype html><meta charset=utf-8><style>body{font:16px sans-serif;margin:24px}.grid{display:grid;grid-template-columns:repeat(2,1fr)}figure{margin:8px}img{width:100%}</style><h1>P2/P3 '+stage+'</h1><p>'+html.escape(summary['scope'])+'</p>'+''.join(cards),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('prepare','train-fit','evaluate-fit','train-full','evaluate-full','train-child'))
    parser.add_argument('--experiment',default='runs/e_bike/p2_detector_20261118');parser.add_argument('--stage',choices=('fit','full'));parser.add_argument('--arm',choices=('P3','P2'))
    args=parser.parse_args();exp=(ROOT/args.experiment).resolve()
    if not exp.is_relative_to(ROOT/'runs/e_bike'):raise ValueError('Expected research output')
    cv2.setNumThreads(1);torch.set_num_threads(2)
    if args.phase=='prepare':prepare(exp)
    elif args.phase=='train-child':train_child(exp,args.stage,args.arm)
    elif args.phase.startswith('train-'):train_stage(exp,args.phase.split('-')[1])
    else:evaluate(exp,args.phase.split('-')[1])


if __name__=='__main__':main()
