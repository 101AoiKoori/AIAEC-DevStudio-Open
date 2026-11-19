"""Fixed 640 raw-candidate replay: confidence, hard NMS and Gaussian Soft-NMS."""
import argparse
from collections import Counter
from datetime import datetime,timezone
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
from lab.features.e_bike.training.scripts.input_scale_experiment import WEIGHT,CLASSIFIER,INPUT,REFERENCE,make_predictor,overlaps
from lab.features.e_bike.training.scripts.keypoint_experiment import read,write
from lab.features.e_bike.training.scripts.two_stage_experiment import crop,truths,compact
from lab.features.e_bike.training.scripts.part_hull_experiment import diagnostic
from lab.training_common.engine.runtime import setup_paths,prepare_ultralytics
from lab.training_common.engine.reporting import match_detections
from datasets.scripts.common.images import read_image,write_image

CONFIDENCES=(.25,.20,.10,.03,.01)
NMS_IOUS=(.3,.4,.5,.6,.7)
BASE='hard_c025_i030'


def specifications():
    specs={f'hard_c{round(c*100):03d}_i{round(i*100):03d}':{'method':'hard','confidence':c,'iou':i} for c in CONFIDENCES for i in NMS_IOUS}
    specs.update({f'soft_c{round(c*100):03d}_s050':{'method':'gaussian_soft','confidence':c,'sigma':.5} for c in (.25,.20,.10)})
    return specs


def gaussian_soft_nms(boxes,scores,threshold,sigma=.5,max_det=100):
    """Prediction-only score decay. Input indices are retained for raw-ID audits."""
    boxes=np.asarray(boxes);scores=np.array(scores,dtype=float,copy=True)
    active=np.where(scores>threshold)[0];kept=[]
    while len(active) and len(kept)<max_det:
        chosen=int(active[np.argmax(scores[active])])
        if scores[chosen]<=threshold:break
        kept.append((chosen,float(scores[chosen])))
        active=active[active!=chosen]
        iou=overlaps(boxes[active],boxes[chosen])
        scores[active]*=np.exp(-(iou*iou)/sigma)
        active=active[scores[active]>threshold]
    return kept


def select(decoded,spec):
    from ultralytics.utils import nms,ops
    if spec['method']=='hard':
        found,ids=nms.non_max_suppression(decoded.to('cuda').clone(),spec['confidence'],spec['iou'],
                                         agnostic=True,max_det=100,return_idxs=True)
        return [(int(i),float(p[4])) for i,p in zip(ids[0].cpu().tolist(),found[0].cpu().tolist())]
    boxes=ops.xywh2xyxy(decoded[0,:4].T).numpy()
    return gaussian_soft_nms(boxes,decoded[0,4].numpy(),spec['confidence'],spec['sigma'])


def entity_match(row):
    return match_detections([{**v,'class_id':0} for v in row['ground_truth']],
                            [{**v,'class_id':0} for v in row['predictions']],.5)


def compare(base,row):
    old,_,old_extra=entity_match(base);new,miss,extra=entity_match(row)
    before={i for i,_,_ in old};after={i for i,_,_ in new}
    old_false={base['predictions'][i]['raw_id'] for i in old_extra}
    new_false={row['predictions'][i]['raw_id'] for i in extra}
    return {'recovered_gt_indices':sorted(after-before),'lost_gt_indices':sorted(before-after),
            'entity_missed_gt_indices':miss,'entity_false_prediction_indices':extra,
            'new_false_raw_ids':sorted(new_false-old_false),'removed_false_raw_ids':sorted(old_false-new_false)}


def render(frame,row,name):
    frame=frame.copy();audit=row['change_audit']
    for i in audit['entity_missed_gt_indices']:
        x1,y1,x2,y2=map(round,row['ground_truth'][i]['xyxy']);cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
    for i,p in enumerate(row['predictions']):
        x1,y1,x2,y2=map(round,p['xyxy']);color=(40,220,40) if p['class_id']==0 else (0,160,255)
        if i in audit['entity_false_prediction_indices']:color=(255,0,255)
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,1)
        cv2.putText(frame,f'{p["confidence"]:.2f}',(x1,max(10,y1-3)),0,.32,color,1)
    for i in audit['recovered_gt_indices']:
        x1,y1,x2,y2=map(round,row['ground_truth'][i]['xyxy']);cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),2)
    g,p=[Counter(o['class_id'] for o in objects) for objects in (row['ground_truth'],row['predictions'])]
    frame=cv2.copyMakeBorder(frame,48,0,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
    cv2.putText(frame,name,(6,18),0,.48,(255,255,255),1)
    cv2.putText(frame,f'GT U={g[0]} F={g[1]} | Pred U={p[0]} F={p[1]} | Dup={row["duplicate_predictions"]}',(6,38),0,.45,(255,255,255),1)
    return frame


def save_review(exp,rows,summary):
    table='<table border="1" cellpadding="7"><tr><th>参数组</th><th>两态全对</th><th>实拍/合成全对</th><th>车辆FP/FN</th><th>重复</th><th>找回/新漏</th><th>新增误检候选</th></tr>'
    for name,s in summary['variants'].items():
        m=s['overall'];e=m['entity_counts_and_matching'];a=s['change_totals']
        table+=f'<tr><td><a href="{name}/review.html">{name}</a></td><td>{m["exact_images"]}/26</td><td>{s["groups"]["original_real"]["exact_images"]}/10，{s["groups"]["synthetic_heldout"]["exact_images"]}/16</td><td>{e["fp"]}/{e["fn"]}</td><td>{s["duplicate_predictions"]}</td><td>{a["recovered_gt"]}/{a["lost_gt"]}</td><td>{a["new_false_predictions"]}</td></tr>'
    cards=[]
    for name,items in rows.items():
        figures=''.join(f'<figure><figcaption>{html.escape(Path(r["source_image"]).stem)}：找回{r["change_audit"]["recovered_gt_indices"]}，新漏{r["change_audit"]["lost_gt_indices"]}</figcaption><a href="{Path(r["source_image"]).stem}.jpg"><img loading="lazy" src="{Path(r["source_image"]).stem}.jpg"></a></figure>' for r in items)
        (exp/name/'review.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px sans-serif}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}img{width:100%}</style><a href="../review.html">全部参数表</a><h1>'+name+'</h1><p>红：漏检GT；紫：误检预测；青：相对原方案找回的GT。其余绿直立、橙倒伏。</p><div class="grid">'+figures+'</div>',encoding='utf-8')
    # Fixed interpretation panels, not a new held-out model selection.
    chosen=[BASE,'hard_c020_i030','hard_c025_i050','hard_c020_i050']
    targets=summary['original_seven_targets']
    for target in targets:
        stem=Path(target['source_image']).stem
        figures=''.join(f'<figure><figcaption>{name}</figcaption><a href="{name}/{stem}.jpg"><img loading="lazy" src="{name}/{stem}.jpg"></a></figure>' for name in chosen)
        cards.append(f'<h2>{stem} G{target["gt_index"]}：原因为{target["reason"]}</h2><div class="grid">{figures}</div>')
    (exp/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>640置信度与NMS对照</title><style>body{font:16px sans-serif;background:#eee;margin:24px}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}figure{margin:0;background:white;padding:8px}img{width:100%}</style><h1>固定640：置信度与NMS对照</h1><p>'+str(len(summary['variants']))+'组预先声明参数，复用同一未裁剪的NMS前解码输出；姿态分类器不变。表中“新增误检候选”按raw_id相对本模型conf0.25/IoU0.3误检集合计算，包含候选互换；不等同FP净增长。已反复探索的26图，非独立测试。</p>'+table+'</table><h2>原7例漏检追踪</h2><p>固定展示：原参数、仅降到0.20、仅放宽IoU到0.5、两者同时。点击参数名可查看该组全部26图。红漏检，紫误检，青找回。颜色中的“找回”相对本模型原参数，原17/26的7例逐车状态见summary。</p>'+''.join(cards),encoding='utf-8')
    strips=[]
    # One low-score example and one NMS example, fixed by prior diagnostic order.
    selected=[next(t for t in targets if t['reason']==reason) for reason in ('below_confidence_025','local_nms')]
    for t in selected:
        stem=Path(t['source_image']).stem
        strips.append(np.concatenate([cv2.resize(read_image(exp/name/(stem+'.jpg')),(384,317)) for name in chosen],axis=1))
    write_image(exp/'comparison_contact.jpg',np.concatenate(strips,axis=0))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',default='runs/e_bike/threshold_nms_probe_20261118')
    parser.add_argument('--weights',default=WEIGHT)
    parser.add_argument('--small-grid',action='store_true',help='Four predeclared mild settings for the trained candidate')
    args=parser.parse_args();exp=(ROOT/args.experiment).resolve()
    if not exp.is_relative_to(ROOT/'runs/e_bike'):raise ValueError('Expected research output')
    exp.mkdir(parents=True,exist_ok=False);setup_paths(exp);cv2.setNumThreads(1);torch.set_num_threads(2)
    specs=specifications()
    if args.small_grid:specs={k:specs[k] for k in (BASE,'hard_c020_i030','hard_c025_i050','hard_c020_i050')}
    snapshot=read(ROOT/INPUT);records=[r for r in snapshot['records'] if r['split']=='val']
    config={'training':False,'imgsz':640,'rect':True,'batch':1,'half':False,'max_det':100,
            'specifications':specs,'weights':args.weights,'classifier':CLASSIFIER,'input':INPUT,
            'policy':'Parameters fixed before evaluation; native hard NMS before clipping. No GT-assisted prediction.'}
    write(exp/'config.json',config);write(exp/'input_snapshot.json',snapshot)
    (exp/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    YOLO=prepare_ultralytics(exp);model=YOLO(str(ROOT/args.weights));classifier=YOLO(str(ROOT/CLASSIFIER))
    assert model.names=={0:'e_bike'}
    raw_dir=exp/'raw';raw_dir.mkdir();all_rows={name:[] for name in specs};reproduction=[]
    reference={r['source_image']:r for r in [json.loads(l) for l in (ROOT/REFERENCE).read_text(encoding='utf-8').splitlines()]}
    for record in records:
        frame=read_image(ROOT/record['image']);gt=truths(record);stem=Path(record['image']).stem
        native=model.predict(source=frame,imgsz=640,rect=True,conf=.25,iou=.3,agnostic_nms=True,max_det=100,half=False,device='0',verbose=False,predictor=make_predictor())[0]
        decoded=model.predictor.decoded_before_nms;boxes=model.predictor.raw_boxes.copy()
        np.savez_compressed(raw_dir/(stem+'.npz'),decoded=decoded.numpy(),original_boxes=boxes)
        selections={name:select(decoded,spec) for name,spec in specs.items()}
        assert [i for i,_ in selections[BASE]]==model.predictor.kept_raw_ids,'Replay differs from native retained IDs'
        unique=sorted({i for found in selections.values() for i,_ in found});pose={}
        for start in range(0,len(unique),32):
            ids=unique[start:start+32]
            outputs=classifier.predict(source=[crop(frame,boxes[i])[0] for i in ids],imgsz=160,device='0',verbose=False)
            for i,out in zip(ids,outputs):pose[i]=(('upright','fallen').index(classifier.names[out.probs.top1]),float(out.probs.top1conf))
        image_rows={}
        for name,selected in selections.items():
            pred=[{'raw_id':i,'xyxy':boxes[i].tolist(),'confidence':score,'raw_confidence':float(decoded[0,4,i]),
                   'class_id':pose[i][0],'pose_confidence':pose[i][1]} for i,score in selected]
            matches,miss,extra=match_detections(gt,pred,.5)
            image_rows[name]={'source_image':record['image'],'group':'synthetic_heldout' if record.get('condition') else 'original_real',
                              'ground_truth':gt,'predictions':pred,'matches':matches,'missed_gt_indices':miss,
                              'false_prediction_indices':extra,**diagnostic(gt,pred)}
        old=reference[record['image']];base=image_rows[BASE]
        same_count=len(base['predictions'])==len(old['predictions'])
        same_boxes=same_count and np.array_equal(np.array([p['xyxy'] for p in base['predictions']]),np.array([p['xyxy'] for p in old['predictions']]))
        same_pose=[p['class_id'] for p in base['predictions']]==[p['class_id'] for p in old['predictions']]
        if args.weights==WEIGHT:assert same_boxes and same_pose,'Original reference reproduction failed'
        reproduction.append({'source_image':record['image'],'native_ids_equal':True,'reference_boxes_and_poses_equal':same_boxes and same_pose})
        for name,row in image_rows.items():
            row['change_audit']=compare(base,row);all_rows[name].append(row)
            folder=exp/name;folder.mkdir(exist_ok=True);write_image(folder/(stem+'.jpg'),render(frame,row,name))
        print(json.dumps({'image':stem,'unique_classified_candidates':len(unique)}),flush=True)
    variants={}
    for name,rows in all_rows.items():
        (exp/name/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        variants[name]={'settings':specs[name],'overall':compact(rows),'groups':{g:compact([r for r in rows if r['group']==g]) for g in ('original_real','synthetic_heldout')},
                        'duplicate_predictions':sum(r['duplicate_predictions'] for r in rows),
                        'images_at_max_det':sum(len(r['predictions'])==100 for r in rows),
                        'change_totals':{'recovered_gt':sum(len(r['change_audit']['recovered_gt_indices']) for r in rows),
                                         'lost_gt':sum(len(r['change_audit']['lost_gt_indices']) for r in rows),
                                         'new_false_predictions':sum(len(r['change_audit']['new_false_raw_ids']) for r in rows),
                                         'removed_false_predictions':sum(len(r['change_audit']['removed_false_raw_ids']) for r in rows)}}
    prior=ROOT/'runs/e_bike/input_scale_probe_20261118/full_640/predictions.jsonl'
    original=[]
    for row in [json.loads(l) for l in prior.read_text(encoding='utf-8').splitlines()]:
        for e in row['candidate_audit']:
            if e['reason']!='detected':original.append({'source_image':row['source_image'],**{k:e[k] for k in ('gt_index','reason','best_raw_score_at_iou05')}})
    for target in original:
        target['recovered_by']=[name for name,rows in all_rows.items() if target['gt_index'] not in entity_match(next(r for r in rows if r['source_image']==target['source_image']))[1]]
    summary={'config':config,'variants':variants,'reference_reproduction':reproduction,'original_seven_targets':original,
             'weights':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (args.weights,CLASSIFIER)},
             'completed_at_utc':datetime.now(timezone.utc).isoformat(),
             'scope':'Same known-source26 images/102GT, parameter exploration not independent test. Native raw replay before clipping. Classifier predictions reused by raw_id across modes. Duplicate=surplus predictions mapped to same GT at maximum IoU>=.5, class agnostic.'}
    write(exp/'summary.json',summary);save_review(exp,all_rows,summary)
    print(json.dumps({k:{'exact':v['overall']['exact_images'],'entity':v['overall']['entity_counts_and_matching'],'duplicates':v['duplicate_predictions'],'changes':v['change_totals']} for k,v in variants.items()}),flush=True)


if __name__=='__main__':main()
