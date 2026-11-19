"""Fixed-budget, prediction-only part ownership and thick-hull suppression probe."""
import argparse
from collections import Counter
import copy
import hashlib
import html
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from lab.features.e_bike.training.scripts.keypoint_experiment import read,write
from lab.features.e_bike.training.scripts.two_stage_experiment import crop,truths,compact
from lab.training_common.engine.runtime import prepare_ultralytics,setup_paths
from lab.training_common.engine.reporting import box_iou,match_detections
from datasets.scripts.common.images import read_image,write_image

SETTINGS={'confidence':.25,'candidate_nms_iou':.95,'part_confidence':.35,
          'part_radius_diagonal_fraction':.06,'part_radius_bounds_px':[2,8],
          'hull_radius_diagonal_fraction':.025,'hull_radius_bounds_px':[1,6],
          'hull_minimum_overlap':.65,'hull_max_center_distance_fraction':.35}
WEIGHT='runs/e_bike/four_part_micro_20261118/training/weights/best.pt'
CLASSIFIER='runs/e_bike/two_stage_20261118/classifier/weights/best.pt'


def diagonal(p):
    x1,y1,x2,y2=p['xyxy'];return float(np.hypot(x2-x1,y2-y1))


def valid_parts(p):
    x1,y1,x2,y2=p['xyxy']
    return {k:np.asarray([x,y],np.float32) for k,(x,y,c) in enumerate(p['keypoints'])
            if c>=SETTINGS['part_confidence'] and np.isfinite([x,y]).all() and x1<=x<=x2 and y1<=y<=y2}


def thick_hull(points,radius):
    """Minkowski approximation gives collinear/two-point hulls nonzero thickness."""
    angles=np.arange(12)*2*np.pi/12
    circle=np.stack([np.cos(angles),np.sin(angles)],axis=1)*radius
    return cv2.convexHull(np.concatenate([p+circle for p in points]).astype(np.float32)).reshape(-1,2)


def associate(predictions):
    """Assign predicted parts to fixed confidence-ranked prototypes per semantic type.

    Radius uses the smaller vehicle scale; no chain-growing connected components.
    This is candidate association, not an independently trained part detector.
    """
    candidates=copy.deepcopy(predictions);prototypes=[]
    for index in sorted(range(len(candidates)),key=lambda i:-candidates[i]['confidence']):
        p=candidates[index];parts=valid_parts(p);owned={}
        for kind,xy in parts.items():
            choices=[]
            for n,proto in enumerate(prototypes):
                radius=np.clip(SETTINGS['part_radius_diagonal_fraction']*min(diagonal(p),proto['scale']),2,8)
                distance=float(np.linalg.norm(xy-proto['xy']))
                if kind==proto['kind'] and distance<=radius:choices.append((distance,n))
            if choices:owned[kind]=min(choices)[1]
            else:
                owned[kind]=len(prototypes)
                prototypes.append({'kind':kind,'xy':xy,'scale':diagonal(p)})
        p['part_ids']={str(k):v for k,v in owned.items()}
        radius=float(np.clip(SETTINGS['hull_radius_diagonal_fraction']*diagonal(p),1,6))
        p['hull']=thick_hull(list(parts.values()),radius).tolist() if len(parts)>=2 else []
    return candidates


def constrain(candidates,use_hull):
    kept=[];removed=[]
    for index in sorted(range(len(candidates)),key=lambda i:-candidates[i]['confidence']):
        p=candidates[index];reason=None
        for other_index in kept:
            q=candidates[other_index]
            shared=set(p['part_ids'].values()) & set(q['part_ids'].values())
            if len(shared)>=2:reason='two_shared_parts'
            elif use_hull and shared and p['hull'] and q['hull']:
                a,b=np.float32(p['hull']),np.float32(q['hull'])
                area=cv2.intersectConvexConvex(a,b)[0]
                overlap=area/max(1e-6,min(cv2.contourArea(a),cv2.contourArea(b)))
                distance=float(np.linalg.norm(a.mean(0)-b.mean(0)))
                if overlap>=SETTINGS['hull_minimum_overlap'] and distance<=SETTINGS['hull_max_center_distance_fraction']*min(diagonal(p),diagonal(q)):
                    reason='shared_part_and_hull'
            if reason:
                removed.append({'candidate':index,'kept_candidate':other_index,'reason':reason});break
        if not reason:kept.append(index)
    return [candidates[i] for i in kept],removed


def box_suppress(candidates):
    kept=[]
    for p in sorted(candidates,key=lambda p:-p['confidence']):
        if all(box_iou(p['xyxy'],q['xyxy'])<=.3 for q in kept):kept.append(p)
    return kept


def diagnostic(gt,pred):
    assigned=Counter()
    for p in pred:
        choices=[(box_iou(g['xyxy'],p['xyxy']),i) for i,g in enumerate(gt)]
        if choices and max(choices)[0]>=.5:assigned[max(choices)[1]]+=1
    return {'duplicate_predictions':sum(max(0,v-1) for v in assigned.values())}


def comparison_review(exp,summary,rows):
    """Audit lost GT coverage separately from removed duplicate candidates."""
    def coverage(row):
        return {i for i,g in enumerate(row['ground_truth'])
                if any(box_iou(g['xyxy'],p['xyxy'])>=.5 for p in row['predictions'])}
    original=rows['candidates'];audit={}
    for mode in ('box_03','ownership','ownership_hull'):
        audit[mode]={'gt_coverage_lost':sum(len(coverage(a)-coverage(b)) for a,b in zip(original,rows[mode])),
                     'removed_predictions':sum(len(a['predictions'])-len(b['predictions']) for a,b in zip(original,rows[mode]))}
    reference_path=ROOT/'runs/e_bike/keypoint_control_pipeline_20261118/evaluation/box_03/predictions.jsonl'
    reference=[json.loads(l) for l in reference_path.read_text(encoding='utf-8').splitlines()]
    assert {r['source_image']:r['ground_truth'] for r in reference}=={r['source_image']:r['ground_truth'] for r in original}
    summary['reference_yolo']={**compact(reference),'duplicate_predictions':sum(diagnostic(r['ground_truth'],r['predictions'])['duplicate_predictions'] for r in reference),
                               'source':reference_path.relative_to(ROOT).as_posix(),'caveat':'Different existing detector weights; same image/GT set. Same-model box_03 is the controlled comparison.'}
    summary['suppression_audit']=audit
    write(exp/'summary.json',summary)
    table='<table border="1" cellpadding="8"><tr><th>方式</th><th>两态数量全对</th><th>重复预测</th><th>车辆漏检</th></tr>'
    labels={'candidates':'宽松候选','box_03':'同模型框去重','ownership':'部件归属','ownership_hull':'部件归属＋凸包'}
    for mode,v in summary['variants'].items():
        m=v['overall'];table+=f'<tr><td>{labels[mode]}</td><td>{m["exact_images"]}/26</td><td>{m["duplicate_predictions"]}</td><td>{m["entity_counts_and_matching"]["fn"]}</td></tr>'
    table+='</table>'
    cards=[]
    for row in original:
        stem=Path(row['source_image']).stem
        figures=''.join(f'<figure><figcaption>{labels[m]}</figcaption><a href="{m}/{stem}.jpg"><img loading="lazy" src="{m}/{stem}.jpg"></a></figure>' for m in labels)
        cards.append(f'<h2>{html.escape(stem)}</h2><div class="grid">{figures}</div>')
    (exp/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>部件归属与凸包对照</title><style>body{font:16px sans-serif;margin:24px;background:#eee}.grid{display:grid;grid-template-columns:repeat(2,minmax(300px,1fr));gap:8px}figure{margin:0;background:white;padding:8px}img{width:100%}</style><h1>部件归属与凸包对照</h1><p>同一26张已知来源研究对照；本轮未训练。重复预测：按最大IoU≥0.5归到同一GT的多余预测，忽略姿态。</p>'+table+'<p>已有另一YOLO模型参照17/26。青：前轮；橙红：后轮；紫：座椅；黄：车篮。绿色框直立、橙色框倒伏，灰框为删除候选。凸包按尺度加厚；点不足时保留。所有部件来自候选框关键点预测，并非独立部件检测器。</p>'+''.join(cards),encoding='utf-8')
    ranked=sorted(range(len(original)),key=lambda i:original[i]['duplicate_predictions']-rows['ownership_hull'][i]['duplicate_predictions'],reverse=True)
    # Show one real and one synthetic frame; retain the full 26-frame review above.
    selected=[next(i for i in ranked if original[i]['group']==g) for g in ('original_real','synthetic_heldout')]
    strips=[]
    for i in selected:
        stem=Path(original[i]['source_image']).stem
        strips.append(np.concatenate([cv2.resize(read_image(exp/m/(stem+'.jpg')),(384,317)) for m in ('candidates','box_03','ownership_hull')],axis=1))
    write_image(exp/'comparison_contact.jpg',np.concatenate(strips,axis=0))


def render(frame,gt,pred,removed,candidates,label):
    frame=frame.copy()
    for removal in removed:
        p=candidates[removal['candidate']]
        x1,y1,x2,y2=map(round,p['xyxy']);cv2.rectangle(frame,(x1,y1),(x2,y2),(130,130,130),1)
        cv2.putText(frame,'drop',(x1,y1+10),0,.32,(180,180,180),1)
    colors=[(255,255,0),(0,80,255),(255,0,255),(0,255,255)]
    for i,p in enumerate(pred):
        color=(40,220,40) if p['class_id']==0 else (0,160,255)
        x1,y1,x2,y2=map(round,p['xyxy']);cv2.rectangle(frame,(x1,y1),(x2,y2),color,1)
        cv2.putText(frame,str(i+1),(x1,y1-3),0,.45,color,1)
        if p.get('hull'):cv2.polylines(frame,[np.int32(p['hull'])],True,color,1)
        for k,xy in valid_parts(p).items():cv2.circle(frame,tuple(np.int32(xy)),3,colors[k],-1)
    g,p=[Counter(o['class_id'] for o in objects) for objects in (gt,pred)]
    frame=cv2.copyMakeBorder(frame,48,0,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
    cv2.putText(frame,label,(6,18),0,.48,(255,255,255),1)
    cv2.putText(frame,f'GT U={g[0]} F={g[1]} | Pred U={p[0]} F={p[1]}',(6,38),0,.48,(255,255,255),1)
    return frame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',default='runs/e_bike/part_hull_probe_20261118')
    args=parser.parse_args();exp=(ROOT/args.experiment).resolve()
    if not exp.is_relative_to(ROOT/'runs/e_bike'):raise ValueError('Expected research output')
    exp.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);torch.set_num_threads(2)
    write(exp/'config.json',{'settings':SETTINGS,'weights':WEIGHT,'classifier':CLASSIFIER,
        'training':False,'policy':'Fixed before evaluation. No GT read by association or constraints. Unknown parts retained.'})
    setup_paths(exp)
    YOLO=prepare_ultralytics(exp);model=YOLO(str(ROOT/WEIGHT));classifier=YOLO(str(ROOT/CLASSIFIER))
    assert list(model.model.kpt_shape)==[4,3]
    snapshot=read(ROOT/'runs/e_bike/four_part_inputs_20261118/input_snapshot.json')
    records=[r for r in snapshot['records'] if r['split']=='val'];write(exp/'input_snapshot.json',snapshot)
    modes=('candidates','box_03','ownership','ownership_hull');rows={m:[] for m in modes}
    for record in records:
        frame=read_image(ROOT/record['image'])
        result=model.predict(source=frame,imgsz=640,rect=True,conf=.25,iou=.95,agnostic_nms=True,max_det=100,device='0',verbose=False)[0]
        raw=[{'xyxy':b,'confidence':float(c),'keypoints':k} for b,c,k in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),result.keypoints.data.cpu().tolist())]
        if raw:
            classes=classifier.predict(source=[crop(frame,p['xyxy'])[0] for p in raw],imgsz=160,device='0',verbose=False)
            for p,c in zip(raw,classes):p['class_id']=('upright','fallen').index(classifier.names[c.probs.top1])
        candidates=associate(raw);gt=truths(record)
        write(exp/(Path(record['image']).stem+'_candidates.json'),candidates)
        for mode in modes:
            pred,removed=(constrain(candidates,mode=='ownership_hull') if mode.startswith('ownership') else (box_suppress(candidates) if mode=='box_03' else candidates,[]))
            matches,missed,extra=match_detections(gt,pred,.5)
            row={'source_image':record['image'],'group':'synthetic_heldout' if record.get('condition') else 'original_real','ground_truth':gt,'predictions':pred,'matches':matches,'missed_gt_indices':missed,'false_prediction_indices':extra,'removed':removed,**diagnostic(gt,pred)}
            rows[mode].append(row);folder=exp/mode;folder.mkdir(exist_ok=True)
            name=Path(record['image']).stem+'.jpg';write_image(folder/name,render(frame,gt,pred,removed,candidates,mode))
    report={}
    for mode,items in rows.items():
        (exp/mode/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items),encoding='utf-8')
        report[mode]={'overall':{**compact(items),'duplicate_predictions':sum(r['duplicate_predictions'] for r in items)},'groups':{g:compact([r for r in items if r['group']==g]) for g in sorted({r['group'] for r in items})},'removed_reasons':dict(Counter(d['reason'] for r in items for d in r['removed']))}
    summary={'variants':report,'settings':SETTINGS,'weights':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (WEIGHT,CLASSIFIER)},'scope':'Same known-source 26-image diagnostic; not independent generalization. Predicted candidate parts, not independent part detections. No training or threshold search. Duplicate metric: surplus predictions assigned by max IoU>=0.5 to one GT, class agnostic.'}
    comparison_review(exp,summary,rows)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
