"""Small post-hoc confidence check for the completed P2/P3 head rebuild study."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import torch

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from datasets.scripts.common.images import read_image,write_image
from lab.training_common.engine.runtime import setup_paths,prepare_ultralytics
from lab.training_common.engine.reporting import match_detections
from lab.features.e_bike.training.scripts.p2_detector_experiment import read,write
from lab.features.e_bike.training.scripts.two_stage_experiment import crop,truths
from lab.features.e_bike.training.scripts.input_scale_experiment import make_predictor
from lab.features.e_bike.training.scripts.candidate_verifier_experiment import finish_pool
from lab.features.e_bike.training.scripts.part_hull_experiment import diagnostic
from lab.features.e_bike.training.scripts.evaluation_audit import image_metrics
from lab.features.e_bike.training.scripts.data_ablation_experiment import error_overlay


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--experiment',default='runs/e_bike/p2_detector_densefit_20261118')
    args=parser.parse_args();exp=(ROOT/args.experiment).resolve();out=exp/'confidence_probe'
    if not exp.is_relative_to(ROOT/'runs/e_bike'):raise ValueError('Expected research experiment')
    out.mkdir(exist_ok=False);setup_paths(out);cv2.setNumThreads(1);torch.set_num_threads(2)
    recipe=read(exp/'recipe_snapshot.json');settings=[(c,iou) for c in (.25,.10,.03) for iou in (.3,.7)]
    write(out/'protocol.json',{'settings':settings,'scope':'Post-hoc calibration check after fixed0.25 head-rebuild result. No training; no threshold tuning beyond this declared grid. Exposed development data.'})
    YOLO=prepare_ultralytics(out)
    from ultralytics.utils import ops
    classifier=YOLO(str(ROOT/recipe['semantic_classifier']));records=read(exp/'evaluation_snapshot.json')['records'];reports={};reproduced={};capped={}
    for arm in recipe['architectures']:
        detector=YOLO(str(ROOT/read(exp/'full'/arm/'status.json')['last_weights']));allrows={};original={r['source_image']:r for r in [json.loads(l) for l in (exp/'evaluation_full'/arm/'predictions.jsonl').read_text().splitlines()]};reproduced[arm]=0;capped[arm]=0
        for record in records:
            frame=read_image(ROOT/record['image']);gt=truths(record);stem=Path(record['image']).stem
            detector.predict(source=frame,imgsz=640,rect=True,conf=.03,iou=.95,max_det=300,device='0',half=False,verbose=False,predictor=make_predictor())
            pred=detector.predictor;ids=pred.kept_raw_ids;boxes=pred.raw_boxes;input_boxes=ops.xywh2xyxy(pred.decoded_before_nms[0,:4].T).numpy();pool=[];capped[arm]+=int(len(ids)==300)
            for start in range(0,len(ids),32):
                chunk=[i for i in ids[start:start+32] if boxes[i][2]>boxes[i][0] and boxes[i][3]>boxes[i][1]]
                if not chunk:continue
                results=classifier.predict(source=[crop(frame,boxes[i])[0] for i in chunk],imgsz=160,device='0',verbose=False)
                for i,result in zip(chunk,results):pool.append({'raw_id':i,'xyxy':boxes[i].tolist(),'input_xyxy':input_boxes[i].tolist(),'raw_confidence':float(pred.decoded_before_nms[0,4,i]),
                    'verifier_probs':{classifier.names[k]:float(v) for k,v in enumerate(result.probs.data.cpu().tolist())}})
            for confidence,iou in settings:
                found=finish_pool(pool,confidence,iou,'joint');name=f'c{round(confidence*100):03d}_i{round(iou*100):03d}'
                if confidence==.25 and iou==.3:
                    expected=original[record['image']]['predictions'];assert [(p['xyxy'],p['class_id']) for p in found]==[(p['xyxy'],p['class_id']) for p in expected];reproduced[arm]+=1
                m,miss,extra=match_detections(gt,found,.5);row={'source_image':record['image'],'group':original[record['image']]['group'],'ground_truth':gt,'predictions':found,
                    'matches':m,'missed_gt_indices':miss,'false_prediction_indices':extra,**diagnostic(gt,found)}
                allrows.setdefault(name,[]).append(row);folder=out/arm/name;folder.mkdir(parents=True,exist_ok=True);write_image(folder/(stem+'.jpg'),error_overlay(row,arm+' '+name))
        reports[arm]={}
        for name,rows in allrows.items():
            (out/arm/name/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8');groups={}
            for group in ('old26','original_real','new_layout_holdout'):
                selected=[r for r in rows if (r['group']!='new_layout_holdout' if group=='old26' else r['group']==group)];metrics=[image_metrics(r) for r in selected]
                groups[group]={'images':len(metrics),**{k:sum(m[k] for m in metrics) for k in ('count_exact','strict_instance_exact','entity_fp','entity_fn','matched_pose_errors')},'duplicate_predictions':sum(r['duplicate_predictions'] for r in selected)}
            reports[arm][name]=groups
    write(out/'summary.json',{'metrics':reports,'control_reproduced_images':reproduced,'proposal_cap_images':capped,'scope':'Post-hoc small grid only; original20 retained unless a net improvement is actually supported.'})
    print(json.dumps(reports,ensure_ascii=False))


if __name__=='__main__':main()
