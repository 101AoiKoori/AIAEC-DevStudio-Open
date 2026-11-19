"""Frozen detector plus train-only background/upright/fallen candidate verifier."""
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

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from datasets.scripts.common.config import load_task_config
from datasets.scripts.common.images import read_image,write_image
from lab.features.e_bike.training.scripts.keypoint_experiment import read,write
from lab.features.e_bike.training.scripts.two_stage_experiment import crop,truths,compact
from lab.features.e_bike.training.scripts.input_scale_experiment import make_predictor,overlaps
from lab.features.e_bike.training.scripts.threshold_nms_experiment import select
from lab.features.e_bike.training.scripts.part_hull_experiment import diagnostic
from lab.features.e_bike.training.scripts.data_ablation_experiment import entity_pairs,error_overlay
from lab.training_common.engine.runtime import setup_paths,prepare_ultralytics
from lab.training_common.engine.reporting import match_detections
from lab.training_common.engine.bounded import run_bounded

CLASSES=('background','fallen','upright')
MODES=('raw','pool_old_pose','gate_old_pose','joint','joint_weighted')


def family_key(row):
    source=row['source_record']
    if source.get('source_image'):return 'real:'+source['source_image']
    state=source['state_id'];number=int(state.rsplit('_',1)[1])
    return 'composite:'+str(source.get('background',{}).get('id'))+':'+str((number-1)//2)


def family_split(key):
    return 'val' if int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%5==0 else 'train'


def candidate_label(box,padded_box,objects,positive_iou=.6,other_pose_iou=.3,reject_partial_below_iou=None):
    ious=np.array([float(overlaps([box],g['xyxy'])[0]) for g in objects])
    if len(ious) and ious.max()>=positive_iou:
        index=int(ious.argmax());cls=objects[index]['class_id']
        if any(i!=index and ious[i]>=other_pose_iou and g['class_id']!=cls for i,g in enumerate(objects)):
            return None,None
        return ('upright','fallen')[cls],index
    x1,y1,x2,y2=padded_box
    if all(min(x2,g['xyxy'][2])<=max(x1,g['xyxy'][0]) or min(y2,g['xyxy'][3])<=max(y1,g['xyxy'][1]) for g in objects):
        return 'background',None
    if reject_partial_below_iou is not None and ious.max()<reject_partial_below_iou:
        return 'background',None
    return None,None


def foreground(probs):
    return max(probs,key=probs.get)!='background' and 1-probs['background']>=.5


def prepare_control(exp,quality_exp):
    """Same slots/classes/order; replace partial rejects with repeated pure BG."""
    recipe=read(quality_exp/'recipe_snapshot.json');manifest=read(quality_exp/'crop_manifest.json')
    recipe['purpose']='matched_slot_pure_background_resampling_control'
    recipe['paired_quality_experiment']=quality_exp.relative_to(ROOT).as_posix()
    exp.mkdir(parents=True,exist_ok=False);write(exp/'recipe_snapshot.json',recipe)
    (exp/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    pure={split:[r for r in manifest['records'] if r['split']==split and r['label']=='background' and r['kind']!='low_IoU_vehicle_candidate_reject'] for split in ('train','val')}
    offset=Counter();records=[]
    for slot in manifest['records']:
        replacement=slot
        if slot['kind']=='low_IoU_vehicle_candidate_reject':
            choices=[r for r in pure[slot['split']] if r['family']==slot['family']] or pure[slot['split']]
            replacement=choices[offset[slot['split']]%len(choices)];offset[slot['split']]+=1
        target=exp/'classifier_data'/slot['split']/slot['label']/Path(slot['image']).name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/replacement['image'],target)
        records.append({**replacement,'image':target.relative_to(ROOT).as_posix(),
                        'matched_slot_source':slot['image'],'control_resampling':replacement is not slot})
    write(exp/'crop_manifest.json',{**manifest,'records':records,
        'background_semantics':'Control trains pure GT-free background repeats instead of low-IoU vehicle candidates.',
        'control_repeated_background_slots':dict(offset),
        'scope':'Matched crop-slot control; actual image parents inherited from replacement. Same warmstart and update budget; known-source development.'})
    for name in ('evaluation_snapshot.json',):shutil.copyfile(quality_exp/name,exp/name)
    print(json.dumps({'matched_control_replacements':dict(offset)}),flush=True)


def prepare(exp,recipe_key='verifier_recipe'):
    _,source=load_task_config('e_bike');recipe=read(ROOT/source[recipe_key])
    exp.mkdir(parents=True,exist_ok=False);setup_paths(exp);write(exp/'recipe_snapshot.json',recipe)
    (exp/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    old=read(ROOT/recipe['training_inputs']);records=[r for r in old['records'] if r['split']=='train']
    assert len(records)==156
    by_image={r['image']:r for r in records};pending={};counts=Counter();family_partitions={}
    YOLO=prepare_ultralytics(exp);detector=YOLO(str(ROOT/recipe['detector']))

    def add(patch,label,row,kind,box,extra=None):
        family=family_key(row);split=family_split(family);family_partitions[family]=split
        digest=hashlib.sha256(patch.tobytes()).hexdigest()
        if digest in pending:
            if pending[digest]['label']!=label:pending[digest]['label']=None;counts['conflicting_pixel_labels']+=1
            else:counts['deduplicated_pixels']+=1
            return
        pending[digest]={'patch':patch,'label':label,'split':split,'family':family,
                         'source_image':row['image'],'kind':kind,'padded_crop_xyxy':box,**(extra or {})}

    for index,row in enumerate(records):
        frame=read_image(ROOT/row['image']);gt=truths(row)
        for gt_index,obj in enumerate(gt):
            patch,box=crop(frame,obj['xyxy']);add(patch,('upright','fallen')[obj['class_id']],row,'GT_crop',box,{'gt_index':gt_index,'raw_box':obj['xyxy']})
        prediction=detector.predict(source=frame,imgsz=640,rect=True,conf=.01,iou=.7,max_det=100,device='0',verbose=False)[0]
        positive={};negative=[];partial=[]
        for box,score in zip(prediction.boxes.xyxy.cpu().tolist(),prediction.boxes.conf.cpu().tolist()):
            if box[2]<=box[0] or box[3]<=box[1]:counts['invalid_box']+=1;continue
            patch,padded=crop(frame,box)
            label,gt_index=candidate_label(box,padded,gt,recipe['positive_iou'],recipe['ambiguous_other_pose_iou'],recipe.get('reject_partial_below_iou'))
            if label is None:counts['ambiguous_intersection_ignored']+=1;continue
            item=(score,patch,label,padded,box,gt_index)
            if label=='background':
                pure,_=candidate_label(box,padded,gt,recipe['positive_iou'],recipe['ambiguous_other_pose_iou'])
                (negative if pure=='background' else partial).append(item)
            else:positive.setdefault(gt_index,[]).append(item)
        for items in positive.values():
            for score,patch,label,padded,box,gt_index in sorted(items,key=lambda x:x[0])[:recipe['predicted_positives_per_gt']]:add(patch,label,row,'predicted_positive',padded,{'gt_index':gt_index,'raw_box':box,'detector_score':score})
        for score,patch,label,padded,box,_ in sorted(negative,key=lambda x:-x[0])[:recipe['background_candidates_per_image']]:add(patch,label,row,'GT_free_predicted_background',padded,{'raw_box':box,'detector_score':score})
        for score,patch,label,padded,box,_ in sorted(partial,key=lambda x:-x[0])[:recipe.get('partial_candidates_per_image',0)]:add(patch,label,row,'low_IoU_vehicle_candidate_reject',padded,{'raw_box':box,'detector_score':score,'best_GT_iou':max(float(overlaps([box],g['xyxy'])[0]) for g in gt)})
        if index%25==0:print(json.dumps({'mined_training_images':index+1,'unique_crops':len(pending)}),flush=True)
    focused=read(ROOT/recipe['focused_inputs'])
    for row in focused['generated']:
        if row['kind']!='mined_vehicle_free_background':continue
        details=row['details'];parent=details['parent_image'];assert parent in by_image
        patch,padded=crop_resize_background(ROOT/row['image'],recipe['classifier_imgsz'])
        add(patch,'background',by_image[parent],'previously_mined_GT_free_background',details['crop_xyxy'],{'derived_source_image':row['image']})
    crops=[]
    for split in ('train','val'):
        for name in CLASSES:(exp/'classifier_data'/split/name).mkdir(parents=True)
    for digest,item in pending.items():
        if item['label'] is None:continue
        target=exp/'classifier_data'/item['split']/item['label']/(digest+'.png');write_image(target,item['patch'])
        crops.append({k:v for k,v in item.items() if k!='patch'}|{'image':target.relative_to(ROOT).as_posix(),'pixel_sha256':digest})
    distribution={split:dict(Counter(r['label'] for r in crops if r['split']==split)) for split in ('train','val')}
    assert all(distribution[s].get(c,0)>0 for s in distribution for c in CLASSES)
    write(exp/'crop_manifest.json',{'records':crops,'distribution':distribution,'counts':dict(counts),
        'family_splits':family_partitions,'old_val_and_layout_holdout_used_for_training':False,
        'initialization_contact_note':'Two-class initialization previously trained on original156 GT crops. Internal crop-family val is not independent of initializer.',
        'scope':recipe['scope'],'labels':'Machine GT IoU matching, ambiguous candidates ignored; no new human confirmation.',
        'background_semantics':'reject candidate, including low-IoU vehicle parts' if recipe.get('reject_partial_below_iou') is not None else 'GT-free background only'})
    write(exp/'evaluation_snapshot.json',read(ROOT/recipe['evaluation_inputs']))
    print(json.dumps(distribution),flush=True)


def crop_resize_background(path,size):
    frame=read_image(path);return crop(frame,[0,0,frame.shape[1],frame.shape[0]],size)


def train_child(exp):
    setup_paths(exp);YOLO=prepare_ultralytics(exp);recipe=read(exp/'recipe_snapshot.json')
    from ultralytics.models.yolo.classify.train import ClassificationTrainer
    from ultralytics.data.augment import classify_transforms
    class FullCandidateTrainer(ClassificationTrainer):
        def build_dataset(self,img_path,mode='train',batch=None):
            dataset=super().build_dataset(img_path,mode,batch)
            dataset.torch_transforms=classify_transforms(size=self.args.imgsz)
            return dataset
    torch.manual_seed(recipe['seed']);model=YOLO(str(ROOT/recipe['initial_classifier']))
    options=dict(data=str(exp/'classifier_data'),imgsz=recipe['classifier_imgsz'],epochs=recipe['epochs'],batch=recipe['batch'],
        device='0',workers=0,optimizer='AdamW',lr0=.0003,lrf=.1,cos_lr=True,warmup_epochs=1,warmup_bias_lr=0.,
        seed=recipe['seed'],deterministic=True,amp=False,patience=0,project=str(exp),name='classifier',exist_ok=False,
        fliplr=0.,flipud=0.,scale=0.,erasing=0.,auto_augment=None,save=True,save_period=10,plots=True)
    write(exp/'classifier_options.json',options);started=time.monotonic()
    model.train(trainer=FullCandidateTrainer,**options)
    write(exp/'classifier_status.json',{'status':'complete','epochs':recipe['epochs'],'seconds':time.monotonic()-started,
        'class_names':model.names,'primary_weights':(exp/'classifier/weights/last.pt').relative_to(ROOT).as_posix(),
        'transform':'Full160-square crop, deterministic resize/center identity and tensor normalization; no random subcrop.'})


def finish_pool(pool,threshold,iou,mode):
    from ultralytics.utils.nms import TorchNMS
    eligible=[p for p in pool if p['raw_confidence']>threshold and (mode=='pool_old_pose' or foreground(p['verifier_probs']))]
    if not eligible:return []
    scores=[p['raw_confidence']*(1-p['verifier_probs']['background']) if mode=='joint_weighted' else p['raw_confidence'] for p in eligible]
    ids=TorchNMS.nms(torch.tensor([p['input_xyxy'] for p in eligible],device='cuda'),torch.tensor(scores,device='cuda'),iou).cpu().tolist()[:100]
    result=[]
    for index in ids:
        p=copy.deepcopy(eligible[index]);p['confidence']=scores[index]
        p['class_id']=p['old_pose'] if mode in ('gate_old_pose','pool_old_pose') else int(p['verifier_probs']['fallen']>p['verifier_probs']['upright'])
        result.append(p)
    return result


def evaluate_probe(exp):
    setup_paths(exp);YOLO=prepare_ultralytics(exp);recipe=read(exp/'recipe_snapshot.json')
    from ultralytics.utils import ops
    detector=YOLO(str(ROOT/recipe['detector']));old=YOLO(str(ROOT/recipe['initial_classifier']));verifier=YOLO(str(exp/'classifier/weights/last.pt'))
    assert set(verifier.names.values())==set(CLASSES)
    records=read(exp/'evaluation_snapshot.json')['records'];rows={};capped=0;reproduction=[]
    write(exp/'evaluation_protocol.json',{'detector_settings':recipe['evaluation_detector_settings'],'postprocess_modes':list(MODES),
        'proposal_iou':recipe['proposal_iou'],'proposal_max_det':recipe['proposal_max_det'],'classifier_checkpoint':'last.pt',
        'scope':'Declared before evaluation. pool_old_pose isolates proposal-pool effects without verifier rejection.'})
    (exp/'evaluation_code_snapshot.py').write_bytes(Path(__file__).read_bytes())
    reference={r['source_image']:r for r in [json.loads(l) for l in (ROOT/'runs/e_bike/focused_comparison_20261118/before/evaluation/box_03/predictions.jsonl').read_text().splitlines()]}
    rawdir=exp/'evaluation_raw';rawdir.mkdir(exist_ok=False)
    for record in records:
        frame=read_image(ROOT/record['image']);gt=truths(record);stem=Path(record['image']).stem
        detector.predict(source=frame,imgsz=640,rect=True,conf=.01,iou=recipe['proposal_iou'],max_det=recipe['proposal_max_det'],device='0',half=False,verbose=False,predictor=make_predictor())
        pred=detector.predictor;decoded=pred.decoded_before_nms;boxes=pred.raw_boxes;pool_ids=pred.kept_raw_ids
        input_boxes=ops.xywh2xyxy(decoded[0,:4].T).numpy();capped+=int(len(pool_ids)==recipe['proposal_max_det'])
        legacy={f'c{round(c*100):03d}_i{round(i*100):03d}':select(decoded,{'method':'hard','confidence':c,'iou':i}) for c,i in recipe['evaluation_detector_settings']}
        unique=sorted(set(pool_ids)|{i for found in legacy.values() for i,_ in found});classified={}
        for start in range(0,len(unique),32):
            ids=[i for i in unique[start:start+32] if boxes[i][2]>boxes[i][0] and boxes[i][3]>boxes[i][1]]
            if not ids:continue
            patches=[crop(frame,boxes[i])[0] for i in ids]
            a=old.predict(source=patches,imgsz=160,device='0',verbose=False);b=verifier.predict(source=patches,imgsz=160,device='0',verbose=False)
            for index,first,second in zip(ids,a,b):
                classified[index]={'raw_id':index,'xyxy':boxes[index].tolist(),'input_xyxy':input_boxes[index].tolist(),
                    'raw_confidence':float(decoded[0,4,index]),'old_pose':('upright','fallen').index(old.names[first.probs.top1]),
                    'verifier_probs':{verifier.names[k]:float(v) for k,v in enumerate(second.probs.data.cpu().tolist())}}
        np.savez_compressed(rawdir/(stem+'.npz'),decoded=decoded.numpy(),original_boxes=boxes)
        write(rawdir/(stem+'_verifier.json'),{'proposal_ids':pool_ids,'classified_candidates':list(classified.values())})
        pool=[classified[i] for i in pool_ids if i in classified]
        for (threshold,iou),setting in zip(recipe['evaluation_detector_settings'],legacy):
            for mode in MODES:
                if mode=='raw':
                    found=[dict(classified[i],confidence=score,class_id=classified[i]['old_pose']) for i,score in legacy[setting]]
                else:found=finish_pool(pool,threshold,iou,mode)
                matches,miss,extra=match_detections(gt,found,.5)
                row={'source_image':record['image'],'group':record.get('evaluation_group') or ('synthetic_heldout' if record.get('condition') else 'original_real'),
                     'ground_truth':gt,'predictions':found,'matches':matches,'missed_gt_indices':miss,'false_prediction_indices':extra,**diagnostic(gt,found)}
                key=setting+'_'+mode;rows.setdefault(key,[]).append(row);folder=exp/'evaluation'/key;folder.mkdir(parents=True,exist_ok=True)
                write_image(folder/(stem+'.jpg'),error_overlay(row,key))
                if setting=='c025_i030' and mode=='raw':
                    before=reference[record['image']]
                    assert [(p['xyxy'],p['class_id']) for p in found]==[(p['xyxy'],p['class_id']) for p in before['predictions']]
                    reproduction.append(record['image'])
        print(json.dumps({'image':stem,'classified_proposals':len(unique)}),flush=True)
    reports={}
    for name,items in rows.items():
        (exp/'evaluation'/name/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items),encoding='utf-8')
        reports[name]={}
        for group in ('old26','original_real','synthetic_heldout','new_layout_holdout','all58'):
            chosen=items if group=='all58' else [r for r in items if (r['group']!='new_layout_holdout' if group=='old26' else r['group']==group)]
            reports[name][group]={**compact(chosen),'duplicate_predictions':sum(r['duplicate_predictions'] for r in chosen)}
    summary={'variants':reports,'recipe':recipe,'baseline_reproduced_images':len(reproduction),'proposal_cap_images':capped,
        'training':read(exp/'classifier_status.json'),'training_data':{k:v for k,v in read(exp/'crop_manifest.json').items() if k!='records'},
        'weights':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [recipe['detector'],recipe['initial_classifier'],(exp/'classifier/weights/last.pt').relative_to(ROOT).as_posix()]},
        'scope':recipe['scope'],'pipeline':'Loose.95 proposal pool -> three-class verification -> final NMS. gate_old_pose isolates rejection; joint also changes posture; weighted multiplies detector score by foreground probability.'}
    write(exp/'summary.json',summary);review(exp,rows,summary)
    print(json.dumps({k:{g:v[g]['exact_images'] for g in ('old26','original_real','new_layout_holdout')} for k,v in reports.items()}),flush=True)


def review(exp,rows,summary):
    table='<table border="1" cellpadding="6"><tr><th>设置</th><th>旧26</th><th>实拍10</th><th>布局32</th><th>旧26 FP/FN</th><th>重复</th></tr>'
    for name,v in summary['variants'].items():
        e=v['old26']['entity_counts_and_matching'];table+=f'<tr><td><a href="evaluation/{name}/review.html">{name}</a></td><td>{v["old26"]["exact_images"]}</td><td>{v["original_real"]["exact_images"]}</td><td>{v["new_layout_holdout"]["exact_images"]}</td><td>{e["fp"]}/{e["fn"]}</td><td>{v["old26"]["duplicate_predictions"]}</td></tr>'
    for name,items in rows.items():
        figures=''.join(f'<figure><figcaption>{Path(r["source_image"]).stem}</figcaption><a href="{Path(r["source_image"]).stem}.jpg"><img loading="lazy" src="{Path(r["source_image"]).stem}.jpg"></a></figure>' for r in items)
        (exp/'evaluation'/name/'review.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px sans-serif}.grid{display:grid;grid-template-columns:repeat(2,1fr)}img{width:100%}</style><a href="../../review.html">全部设置</a><h1>'+name+'</h1><div class="grid">'+figures+'</div>',encoding='utf-8')
    chosen=tuple(name for name in ('c025_i030_raw','c025_i030_pool_old_pose','c025_i030_gate_old_pose','c025_i030_joint','c010_i050_joint','c001_i070_joint_weighted') if name in rows)
    cards=[]
    for row in rows['c025_i030_raw']:
        stem=Path(row['source_image']).stem
        figures=''.join(f'<figure><figcaption>{name}</figcaption><a href="evaluation/{name}/{stem}.jpg"><img loading="lazy" src="evaluation/{name}/{stem}.jpg"></a></figure>' for name in chosen)
        cards.append(f'<h2>{html.escape(stem)}</h2><div class="grid">'+figures+'</div>')
    (exp/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>三类候选验证器</title><style>body{font:16px sans-serif;background:#eee;margin:24px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}figure{margin:0;background:white;padding:8px}img{width:100%}</style><h1>冻结检测器：非车／直立／倒伏验证</h1><p>raw旧两态分类器；gate_old_pose仅新增非车筛选；joint使用新姿态；joint_weighted另按检测分数×前景概率排序NMS。红漏GT、紫误检、黄姿态错误。GT只用于评估；已知同源开发集。</p>'+table+'</table>'+''.join(cards),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('prepare','prepare-control','train','train-child','evaluate'))
    parser.add_argument('--recipe-key',default='verifier_recipe',help='source.yaml recipe key, read only during prepare')
    parser.add_argument('--quality-experiment',help='Prepared quality arm used by prepare-control')
    parser.add_argument('--experiment',default='runs/e_bike/candidate_verifier_20261118');args=parser.parse_args();exp=(ROOT/args.experiment).resolve()
    if not exp.is_relative_to(ROOT/'runs/e_bike'):raise ValueError('Expected e_bike research output')
    cv2.setNumThreads(1);torch.set_num_threads(2)
    if args.phase=='prepare':prepare(exp,args.recipe_key)
    elif args.phase=='prepare-control':
        quality_exp=(ROOT/args.quality_experiment).resolve()
        if not quality_exp.is_relative_to(ROOT/'runs/e_bike'):raise ValueError('Expected e_bike control source')
        prepare_control(exp,quality_exp)
    elif args.phase=='train-child':train_child(exp)
    elif args.phase=='train':
        recipe=read(exp/'recipe_snapshot.json');run_bounded([sys.executable,'-X','utf8','-B',str(Path(__file__)),'train-child','--experiment',args.experiment],time.monotonic()+recipe['wall_seconds'],exp/'classifier_train.log',ROOT)
    else:evaluate_probe(exp)


if __name__=='__main__':main()
