"""Learn continuous candidate IoU and within-target ranking on frozen features."""
import argparse
from collections import Counter,defaultdict
import copy
import hashlib
import html
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from datasets.scripts.common.config import load_task_config
from datasets.scripts.common.images import read_image,write_image
from lab.training_common.engine.runtime import setup_paths,prepare_ultralytics
from lab.training_common.engine.reporting import box_iou,match_detections
from lab.features.e_bike.training.scripts.two_stage_experiment import crop,truths
from lab.features.e_bike.training.scripts.input_scale_experiment import make_predictor
from lab.features.e_bike.training.scripts.candidate_verifier_experiment import family_key,family_split,foreground
from lab.features.e_bike.training.scripts.evaluation_audit import image_metrics
from lab.features.e_bike.training.scripts.data_ablation_experiment import error_overlay,paired_errors


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def geometry(box,score,shape):
    h,w=shape[:2];x1,y1,x2,y2=box;bw,bh=max(x2-x1,1e-6),max(y2-y1,1e-6)
    clipped=max(0,min(x2,w)-max(x1,0))*max(0,min(y2,h)-max(y1,0))/(bw*bh)
    return [score,bw/w,bh/h,np.clip(np.log(bw/bh),-4,4),bw*bh/(w*h),clipped]


def association(ious,minimum=.3,margin=.1):
    if not len(ious):return None
    order=np.argsort(ious)[::-1]
    if ious[order[0]]<minimum or (len(order)>1 and ious[order[0]]-ious[order[1]]<margin):return None
    return int(order[0])


class Features:
    def __init__(self,model):
        self.model=model
        model.predict(source=np.full((160,160,3),114,np.uint8),imgsz=160,device='0',verbose=False)

    def __call__(self,patches):
        captured=[]
        handle=self.model.model.model[-1].linear.register_forward_pre_hook(lambda module,inputs:captured.append(inputs[0].detach().cpu()))
        try:results=self.model.predict(source=patches,imgsz=160,device='0',verbose=False)
        finally:handle.remove()
        features=torch.cat(captured).clone();assert len(features)==len(patches)
        probs=[{self.model.names[k]:float(v) for k,v in enumerate(r.probs.data.cpu().tolist())} for r in results]
        return features,probs


def prepare(exp):
    _,source=load_task_config('e_bike');recipe=read(ROOT/source['quality_rank_recipe'])
    exp.mkdir(parents=True,exist_ok=False);setup_paths(exp);write(exp/'recipe_snapshot.json',recipe)
    (exp/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());(exp/'features').mkdir()
    YOLO=prepare_ultralytics(exp);detector=YOLO(str(ROOT/recipe['detector']));extract=Features(YOLO(str(ROOT/recipe['feature_classifier'])))
    original=read(ROOT/recipe['training_inputs'])['records'];records=[r for r in original if r['split']=='train']
    assert len(records)==156
    metadata=[];vectors=[];seen={};duplicates=Counter()
    for index,row in enumerate(records):
        frame=read_image(ROOT/row['image']);gt=truths(row);family=family_key(row);split=family_split(family)
        prediction=detector.predict(source=frame,imgsz=640,rect=True,conf=recipe['mining_confidence'],iou=recipe['mining_iou'],max_det=300,device='0',verbose=False,predictor=make_predictor())
        predictor=detector.predictor;bybin=defaultdict(list)
        for raw_id in predictor.kept_raw_ids:
            box=predictor.raw_boxes[raw_id].tolist();score=float(predictor.decoded_before_nms[0,4,raw_id])
            if box[2]<=box[0] or box[3]<=box[1]:continue
            ious=[box_iou(g['xyxy'],box) for g in gt];target=max(ious,default=0.)
            bin_id=int(np.searchsorted(recipe['quality_bins'],target,side='right')-1)
            bybin[bin_id].append((score,raw_id,box,ious,target))
        selected=[item for group in bybin.values() for item in sorted(group,reverse=True)[:recipe['max_per_quality_bin']]]
        for start in range(0,len(selected),32):
            batch=selected[start:start+32];patches=[];details=[]
            for score,raw_id,box,ious,target in batch:
                patch,bounds=crop(frame,box);digest=hashlib.sha256(patch.tobytes()).hexdigest()
                if digest in seen:duplicates['same_split' if seen[digest]==split else 'cross_split_removed']+=1;continue
                seen[digest]=split;patches.append(patch)
                details.append({'source_image':row['image'],'family':family,'split':split,'raw_id':raw_id,'xyxy':box,'raw_confidence':score,
                                'quality_target':target,'GT_ious':ious,'associated_gt':association(ious,recipe['association_min_iou'],recipe['association_margin']),
                                'geometry':geometry(box,score,frame.shape),'crop_bounds':bounds,'pixel_sha256':digest})
            if not patches:continue
            values,_=extract(patches);vectors.append(values);metadata.extend(details)
        if index%25==0:print(json.dumps({'mined_frames':index+1,'quality_samples':len(metadata)}),flush=True)
    features=torch.cat(vectors);assert len(features)==len(metadata)
    torch.save(features,exp/'features/training.pt')
    write(exp/'training_manifest.json',{'records':metadata,'distribution':dict(Counter(r['split'] for r in metadata)),
        'duplicate_crops_removed':dict(duplicates),'feature_dimensions':features.shape[1],
        'scope':recipe['scope'],'features':'features/training.pt','labels':'Maximum IoU; relation labels only for unambiguous best GT. No new human confirmation.'})
    write(exp/'evaluation_snapshot.json',read(ROOT/recipe['reference_experiment']/'evaluation_snapshot.json'))
    print(json.dumps({'samples':len(metadata),'distribution':dict(Counter(r['split'] for r in metadata))}),flush=True)


def rank_pairs(records,ids,recipe):
    groups=defaultdict(list)
    for i in ids:
        r=records[i]
        if r['associated_gt'] is not None:groups[(r['source_image'],r['associated_gt'])].append(i)
    per_image=defaultdict(list)
    for (image,_),members in groups.items():
        for a in members:
            for b in members:
                gap=records[a]['quality_target']-records[b]['quality_target']
                if gap>=recipe['rank_pair_min_iou_gap']:per_image[image].append((gap,a,b))
    return [(a,b) for members in per_image.values() for _,a,b in sorted(members,reverse=True)[:recipe['max_pairs_per_image']]]


def model_for(dim):return nn.Sequential(nn.Linear(dim,128),nn.ReLU(),nn.Linear(128,1))


def train(exp):
    recipe=read(exp/'recipe_snapshot.json');manifest=read(exp/'training_manifest.json');records=manifest['records']
    features=torch.load(exp/'features/training.pt',weights_only=True);geom=torch.tensor([r['geometry'] for r in records],dtype=torch.float32)
    target=torch.tensor([r['quality_target'] for r in records],dtype=torch.float32)
    train_ids=[i for i,r in enumerate(records) if r['split']=='train'];val_ids=[i for i,r in enumerate(records) if r['split']=='val']
    pairs=rank_pairs(records,train_ids,recipe);val_pairs=rank_pairs(records,val_ids,recipe);report={};deadline=time.monotonic()+recipe['wall_seconds']
    for name,values in (('geometry',geom),('image_geometry',torch.cat([features,geom],dim=1))):
        torch.manual_seed(recipe['seed']);generator=torch.Generator().manual_seed(recipe['seed'])
        mean=values[train_ids].mean(0);std=values[train_ids].std(0).clamp_min(.05)
        normalized=((values-mean)/std).clamp(-10,10).cuda();targets=target.cuda()
        model=model_for(values.shape[1]).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=recipe['learning_rate'])
        pair_tensor=torch.tensor(pairs,dtype=torch.long,device='cuda');trace=[];started=time.monotonic()
        for epoch in range(recipe['epochs']):
            if time.monotonic()>deadline:raise TimeoutError('Quality head training exceeded declared wall budget')
            model.train();losses=[]
            order=torch.tensor(train_ids)[torch.randperm(len(train_ids),generator=generator)]
            for start in range(0,len(order),recipe['batch']):
                ids=order[start:start+recipe['batch']].cuda();logits=model(normalized[ids]).squeeze(1)
                loss=F.smooth_l1_loss(logits.sigmoid(),targets[ids],beta=.1)
                if len(pairs):
                    chosen=pair_tensor[torch.randint(len(pairs),(64,),generator=generator).cuda()]
                    rank=model(normalized[chosen[:,0]]).squeeze(1)-model(normalized[chosen[:,1]]).squeeze(1)
                    loss=loss+recipe['rank_loss_weight']*F.softplus(-rank).mean()
                optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
            trace.append({'epoch':epoch+1,'loss':float(np.mean(losses))})
        model.eval()
        with torch.no_grad():predicted=model(normalized).squeeze(1).sigmoid().cpu()
        metrics={split:{'MAE':float((predicted[ids]-target[ids]).abs().mean()),
                        'same_GT_pair_order_accuracy':float(np.mean([predicted[a]>predicted[b] for a,b in ps])) if ps else None,
                        'pairs':len(ps)} for split,ids,ps in (('train',train_ids,pairs),('internal_val',val_ids,val_pairs))}
        checkpoint={'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'mean':mean,'std':std,'input_dim':values.shape[1],
                    'kind':name,'recipe':recipe,'feature_classifier_sha256':hashlib.sha256((ROOT/recipe['feature_classifier']).read_bytes()).hexdigest()}
        torch.save(checkpoint,exp/(name+'.pt'));report[name]={'epochs':recipe['epochs'],'seconds':time.monotonic()-started,'metrics':metrics,'trace':trace}
        print(json.dumps({name:report[name]['metrics']}),flush=True)
    write(exp/'training_status.json',{'status':'complete','heads':report,'rank_pairs_use_train_only':True,'internal_val_scope':'Not independent of frozen feature initialization; head uses train only.'})


def evaluate(exp,diagnose=False):
    recipe=read(exp/'recipe_snapshot.json');setup_paths(exp);YOLO=prepare_ultralytics(exp)
    from ultralytics.utils.nms import TorchNMS
    extract=Features(YOLO(str(ROOT/recipe['feature_classifier'])));reference=ROOT/recipe['reference_experiment'];heads={}
    for kind in ('geometry','image_geometry'):
        state=torch.load(exp/(kind+'.pt'),weights_only=True);model=model_for(state['input_dim']).eval()
        model.load_state_dict(state['state_dict']);heads[kind]=(model,state)
    rows=[json.loads(line) for line in (reference/'evaluation/c025_i030_joint/predictions.jsonl').read_text().splitlines()]
    modes=('raw_control','geometry_quality','image_quality','GT_assisted_quality_DIAGNOSTIC');output={m:[] for m in modes};reproduced=[];probability_audit=[]
    destination=exp
    if diagnose:
        destination=exp/'ranking_diagnostic';destination.mkdir(exist_ok=False)
        modes=modes+('geometry_only','image_only','GT_only_DIAGNOSTIC');output={m:[] for m in modes}
        write(destination/'protocol.json',{'scope':'Post-hoc score-formula diagnostic after primary blend failed. No retraining or threshold search.',
                                         'modes':list(modes),'confidence':recipe['primary_confidence'],'nms_iou':recipe['primary_nms_iou']})
    (exp/'evaluation_code_snapshot.py').write_bytes(Path(__file__).read_bytes())
    for ref in rows:
        stem=Path(ref['source_image']).stem;frame=read_image(ROOT/ref['source_image']);cached=read(reference/'evaluation_raw'/(stem+'_verifier.json'))
        byid={p['raw_id']:p for p in cached['classified_candidates']}
        pool=[copy.deepcopy(byid[i]) for i in cached['proposal_ids'] if i in byid and byid[i]['raw_confidence']>recipe['primary_confidence'] and foreground(byid[i]['verifier_probs'])]
        if pool:
            feature_batches=[]
            for start in range(0,len(pool),32):
                group=pool[start:start+32];features,probs=extract([crop(frame,p['xyxy'])[0] for p in group]);feature_batches.append(features)
                assert torch.isfinite(features).all()
                for a,b in zip(probs,group):
                    probability_audit.append({'image':ref['source_image'],'raw_id':b['raw_id'],
                        'maximum_probability_difference':max(abs(a[k]-b['verifier_probs'][k]) for k in a),
                        'top1_changed':max(a,key=a.get)!=max(b['verifier_probs'],key=b['verifier_probs'].get)})
            image_features=torch.cat(feature_batches);geom=torch.tensor([geometry(p['xyxy'],p['raw_confidence'],frame.shape) for p in pool],dtype=torch.float32)
            estimates={}
            with torch.no_grad():
                for kind,values in (('geometry',geom),('image_geometry',torch.cat([image_features,geom],dim=1))):
                    model,state=heads[kind];estimates[kind]=model(((values-state['mean'])/state['std']).clamp(-10,10)).squeeze(1).sigmoid().tolist()
            for i,p in enumerate(pool):p['quality_estimates']={k:v[i] for k,v in estimates.items()}
        for mode in modes:
            scores=[]
            for p in pool:
                quality=(p['quality_estimates']['geometry'] if mode in ('geometry_quality','geometry_only') else p['quality_estimates']['image_geometry'] if mode in ('image_quality','image_only') else max((box_iou(g['xyxy'],p['xyxy']) for g in ref['ground_truth']),default=0.))
                scores.append(p['raw_confidence'] if mode=='raw_control' else quality if mode in ('geometry_only','image_only','GT_only_DIAGNOSTIC') else p['raw_confidence']*(1-recipe['quality_blend']+recipe['quality_blend']*quality))
            ids=TorchNMS.nms(torch.tensor([p['input_xyxy'] for p in pool],device='cuda'),torch.tensor(scores,device='cuda'),recipe['primary_nms_iou']).cpu().tolist()[:100] if pool else []
            found=[]
            for i in ids:
                p=copy.deepcopy(pool[i]);p['confidence']=scores[i];p['class_id']=int(p['verifier_probs']['fallen']>p['verifier_probs']['upright']);found.append(p)
            if mode=='raw_control':
                assert [(p['xyxy'],p['class_id']) for p in found]==[(p['xyxy'],p['class_id']) for p in ref['predictions']];reproduced.append(ref['source_image'])
            matches,miss,extra=match_detections(ref['ground_truth'],found,.5)
            row={**ref,'predictions':found,'matches':matches,'missed_gt_indices':miss,'false_prediction_indices':extra}
            output[mode].append(row);folder=destination/'evaluation'/mode;folder.mkdir(parents=True,exist_ok=True);write_image(folder/(stem+'.jpg'),error_overlay(row,mode))
        if diagnose:
            folder=destination/'candidates';folder.mkdir(exist_ok=True)
            write(folder/(stem+'.json'),{'image':ref['source_image'],'ground_truth':ref['ground_truth'],'candidates':pool})
    metrics={}
    for mode,items in output.items():
        (destination/'evaluation'/mode/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items),encoding='utf-8');metrics[mode]={}
        for group in ('old26','original_real','synthetic_heldout','new_layout_holdout'):
            selected=[r for r in items if (r['group']!='new_layout_holdout' if group=='old26' else r['group']==group)];values=[image_metrics(r) for r in selected]
            metrics[mode][group]={'images':len(values),**{k:sum(v[k] for v in values) for k in ('count_exact','strict_instance_exact','entity_fp','entity_fn','matched_pose_errors','strict_exact_with_assignment_ambiguity')}}
    changes={mode:[paired_errors(a,b) for a,b in zip(output['raw_control'],items)] for mode,items in output.items() if mode!='raw_control'}
    write(destination/'summary.json',{'metrics':metrics,'paired_changes':changes,'raw_control_reproduced_images':len(reproduced),'recipe':recipe,
                             'training':read(exp/'training_status.json'),'scope':recipe['scope'],
                             'feature_recompute_probability_audit':{'maximum_difference':max((p['maximum_probability_difference'] for p in probability_audit),default=0.),
                                'top1_changed':sum(p['top1_changed'] for p in probability_audit),'candidates':len(probability_audit),
                                'note':'Feature batches differ from original broad-pool batches. Original cached probabilities exclusively control gating and posture.'},
                             'weights':{(exp/(kind+'.pt')).relative_to(ROOT).as_posix():hashlib.sha256((exp/(kind+'.pt')).read_bytes()).hexdigest() for kind in heads}})
    cards=[]
    for ref in rows:
        stem=Path(ref['source_image']).stem;cards.append('<h2>'+html.escape(stem)+'</h2><div class=grid>'+''.join(f'<figure><figcaption>{m}</figcaption><a href="evaluation/{m}/{stem}.jpg"><img loading=lazy src="evaluation/{m}/{stem}.jpg"></a></figure>' for m in modes)+'</div>')
    (destination/'review.html').write_text('<!doctype html><meta charset=utf-8><style>body{font:16px sans-serif;margin:24px}.grid{display:grid;grid-template-columns:repeat(2,1fr)}figure{margin:8px}img{width:100%}</style><h1>连续IoU质量排序</h1><p>前景门槛、候选阈值与NMS IoU固定；只改变排序。GT-assisted明确使用GT，仅供诊断，不能交付。</p>'+metric_table(metrics)+''.join(cards),encoding='utf-8')
    print(json.dumps(metrics,ensure_ascii=False),flush=True)


def metric_table(metrics):
    table='<table border=1 cellpadding=8><tr><th>方案</th><th>旧26数量/框姿态匹配</th><th>实拍10</th><th>布局32</th><th>旧26FP/FN</th></tr>'
    for name,groups in metrics.items():
        a,b,c=(groups[g] for g in ('old26','original_real','new_layout_holdout'))
        table+=f'<tr><td>{name}</td><td>{a["count_exact"]}/{a["strict_instance_exact"]}</td><td>{b["count_exact"]}/{b["strict_instance_exact"]}</td><td>{c["count_exact"]}/{c["strict_instance_exact"]}</td><td>{a["entity_fp"]}/{a["entity_fn"]}</td></tr>'
    return table+'</table>'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('prepare','train','evaluate','diagnose-ranking'))
    parser.add_argument('--experiment',default='runs/e_bike/quality_rank_20261118');args=parser.parse_args();exp=(ROOT/args.experiment).resolve()
    if not exp.is_relative_to(ROOT/'runs/e_bike'):raise ValueError('Expected e_bike research output')
    cv2.setNumThreads(1);torch.set_num_threads(2)
    if args.phase=='diagnose-ranking':evaluate(exp,True)
    else:{'prepare':prepare,'train':train,'evaluate':evaluate}[args.phase](exp)


if __name__=='__main__':main()
