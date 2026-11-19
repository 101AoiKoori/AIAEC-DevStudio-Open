"""Bounded research experiment: one-class bike detector plus crop posture classifier."""
import argparse
from collections import Counter
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from lab.training_common.engine.config import resolve
from lab.training_common.engine.runtime import setup_paths, prepare_ultralytics
from lab.training_common.engine.bounded import run_bounded
from lab.training_common.engine.reporting import match_detections
from lab.features.e_bike.training.scripts.report_pose_counts import pose_counts
from datasets.scripts.common.images import read_image, write_image

POSES = ('upright', 'fallen')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def crop(frame, box, size=160):
    """Preserve the full visible box plus 8% context with square padding, never center-crop it."""
    h,w = frame.shape[:2]
    x1,y1,x2,y2 = box
    dx,dy = max(2,(x2-x1)*.08),max(2,(y2-y1)*.08)
    left,top = max(0,math.floor(x1-dx)),max(0,math.floor(y1-dy))
    right,bottom = min(w,math.ceil(x2+dx)),min(h,math.ceil(y2+dy))
    if right<=left or bottom<=top:
        raise ValueError('Empty vehicle crop')
    patch = frame[top:bottom,left:right]
    scale = size/max(patch.shape[:2])
    resized = cv2.resize(patch,(max(1,round(patch.shape[1]*scale)),max(1,round(patch.shape[0]*scale))))
    result = np.full((size,size,3),114,np.uint8)
    py,px = (size-resized.shape[0])//2,(size-resized.shape[1])//2
    result[py:py+resized.shape[0],px:px+resized.shape[1]] = resized
    return result,[left,top,right,bottom]


def truths(record):
    w,h = record['source_record']['image_size']
    result = []
    for line in resolve(ROOT,record['label']).read_text(encoding='utf-8').splitlines():
        cls,x,y,bw,bh = map(float,line.split())
        result.append({'class_id':int(cls),'xyxy':[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]})
    return result


def prepare(exp, inputs):
    exp.mkdir(parents=True,exist_ok=False)
    original = read(inputs)
    write(exp/'input_snapshot.json',original)
    records,crops = [],[]
    for split in ('train','val'):
        for folder in ('images','labels'):
            (exp/'inputs/detect'/split/folder).mkdir(parents=True)
        for cls in POSES:
            (exp/'inputs/crops'/split/cls).mkdir(parents=True)
    for original_record in original['records']:
        row = dict(original_record)
        frame = read_image(resolve(ROOT,row['image']))
        image = exp/'inputs/detect'/row['split']/'images'/Path(row['image']).name
        label = exp/'inputs/detect'/row['split']/'labels'/Path(row['label']).name
        shutil.copyfile(resolve(ROOT,row['image']),image)
        text = resolve(ROOT,row['label']).read_text(encoding='utf-8')
        label.write_text(''.join('0 '+' '.join(line.split()[1:])+'\n' for line in text.splitlines()),encoding='utf-8')
        row.update(image=image.relative_to(ROOT).as_posix(),label=label.relative_to(ROOT).as_posix())
        records.append(row)
        for index,obj in enumerate(truths(original_record)):
            patch,box = crop(frame,obj['xyxy'])
            path = exp/'inputs/crops'/row['split']/POSES[obj['class_id']]/(image.stem+f'_{index:02d}.png')
            write_image(path,patch)
            crops.append({'image':path.relative_to(ROOT).as_posix(),'split':row['split'],
                          'source_image':original_record['image'],'gt_index':index,
                          'posture':POSES[obj['class_id']],'padded_crop_xyxy':box})
    data = exp/'inputs/detect'
    write(data/'data.yaml',{'names':['e_bike'],'train':'train/images','val':'val/images'})
    write(data/'workspace_sources.json',{'model_key':'e_bike','selection_mode':'manifest_only','records':records,'source_reuse':[]})
    write(exp/'crop_manifest.json',{'records':crops,'scope':'Ground-truth crops inherit image split. Shared original sources; same-source diagnostic only.'})
    config = yaml.safe_load((ROOT/'runs/e_bike/synthetic_short_inputs_20261118/train.yaml').read_text(encoding='utf-8'))
    config.update(data=(data/'data.yaml').relative_to(ROOT).as_posix(),description='One-class bike localization, ten-epoch research ablation',epochs=10,nms_iou=.5)
    (exp/'detector.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
    write(exp/'preparation.json',{'images':dict(Counter(r['split'] for r in records)),
          'crops':{split:dict(Counter(r['posture'] for r in crops if r['split']==split)) for split in ('train','val')},
          'input_manifest':inputs.relative_to(ROOT).as_posix(),'input_sha256':hashlib.sha256(inputs.read_bytes()).hexdigest(),
          'detector_supervision':'All boxes remapped to e_bike=0; original posture labels remain in input_snapshot.json',
          'scope':'Research snapshot only; current dataset config and exports unchanged.'})
    print(read(exp/'preparation.json'),flush=True)


def train_classifier(exp):
    setup_paths(exp)
    YOLO = prepare_ultralytics(exp)
    torch.set_num_threads(2)
    # Installed architecture and local detection weights, with common backbone weights transferred.
    model = YOLO('yolov8n-cls.yaml').load(str(ROOT/'runs/e_bike/v4_baseline_20261118/training/weights/best.pt'))
    options = dict(data=str(exp/'inputs/crops'),imgsz=160,epochs=10,batch=16,workers=0,
        device='0',optimizer='AdamW',lr0=.0003,lrf=.1,warmup_epochs=1,seed=42,
        deterministic=True,amp=False,patience=0,project=str(exp),name='classifier',
        pretrained=True,exist_ok=False,save=True,save_period=5,plots=True,
        fliplr=0.,flipud=0.,degrees=0.,translate=0.,scale=0.,hsv_h=.005,hsv_s=.1,hsv_v=.12)
    write(exp/'classifier_options.json',options)
    started = time.monotonic()
    model.train(**options)
    write(exp/'classifier_status.json',{'status':'complete','epochs':10,'seconds':time.monotonic()-started,
          'weights':(exp/'classifier/weights/best.pt').relative_to(ROOT).as_posix(),
          'initialization':'YOLOv8n classifier architecture, shape-compatible backbone weights from original bike detector; new classification head.'})


def compact(rows):
    entity = Counter()
    for row in rows:
        gt = [{**v,'class_id':0} for v in row['ground_truth']]
        pred = [{**v,'class_id':0} for v in row['predictions']]
        matched,missed,extra = match_detections(gt,pred,.5)
        entity.update(tp=len(matched),fp=len(extra),fn=len(missed),
                      count_exact=int(len(gt)==len(pred)),all_matched=int(not missed and not extra))
    return {**{k:v for k,v in pose_counts(rows).items() if k!='records'},
            'tp':sum(len(r['matches']) for r in rows),'fp':sum(len(r['false_prediction_indices']) for r in rows),
            'fn':sum(len(r['missed_gt_indices']) for r in rows),
            'entity_counts_and_matching':dict(entity)}


def evaluate(exp, detector_weight):
    setup_paths(exp)
    YOLO = prepare_ultralytics(exp)
    records = [r for r in read(exp/'input_snapshot.json')['records'] if r['split']=='val']
    classifier = YOLO(str(exp/'classifier/weights/best.pt'))
    if set(classifier.names.values())!=set(POSES):
        raise ValueError('Unexpected classifier names')

    def classify(patches):
        if not patches:
            return []
        results = classifier.predict(source=patches,imgsz=160,device='0',verbose=False)
        return [(POSES.index(classifier.names[r.probs.top1]),float(r.probs.top1conf)) for r in results]

    report,all_rows = {},{}
    def save(name, rows):
        folder = exp/'evaluation'/name
        folder.mkdir(parents=True,exist_ok=False)
        (folder/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        report[name] = {'overall':compact(rows),'groups':{g:compact([r for r in rows if r['group']==g])
                              for g in sorted({r['group'] for r in rows})}}
        all_rows[name] = rows
        return folder

    def row(record,gt,found):
        matched,missed,extra = match_detections(gt,found,.5)
        return {'source_image':record['image'],'group':'synthetic_heldout' if record.get('condition') else 'original_real',
                'ground_truth':gt,'predictions':found,'matches':matched,
                'missed_gt_indices':missed,'false_prediction_indices':extra}

    oracle = []
    for record in records:
        frame = read_image(resolve(ROOT,record['image']));gt = truths(record)
        classes = classify([crop(frame,g['xyxy'])[0] for g in gt])
        oracle.append(row(record,gt,[{'xyxy':g['xyxy'],'class_id':cls,'confidence':conf} for g,(cls,conf) in zip(gt,classes)]))
    save('oracle_gt_boxes',oracle)
    for name,weight in (('old_detector_reclassified',ROOT/'runs/e_bike/v4_baseline_20261118/training/weights/best.pt'),
                        ('one_class_pipeline',detector_weight)):
        detector = YOLO(str(weight))
        for iou in (.5,.3):
            rows,frames = [],[]
            for record in records:
                result = detector.predict(source=str(resolve(ROOT,record['image'])),imgsz=640,rect=True,
                    conf=.25,iou=iou,agnostic_nms=True,max_det=100,device='0',verbose=False)[0]
                frame = read_image(resolve(ROOT,record['image']))
                boxes = result.boxes.xyxy.cpu().tolist()
                assignments = classify([crop(frame,b)[0] for b in boxes])
                found = [{'xyxy':b,'class_id':cls,'confidence':float(conf),'posture_confidence':prob}
                         for b,conf,(cls,prob) in zip(boxes,result.boxes.conf.cpu().tolist(),assignments)]
                current = row(record,truths(record),found);rows.append(current)
                for prediction in found:
                    x1,y1,x2,y2 = [round(v) for v in prediction['xyxy']]
                    color = (40,220,40) if prediction['class_id']==0 else (0,160,255)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                    cv2.putText(frame,POSES[prediction['class_id']],(x1,max(15,y1-3)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
                gt_counts,pred_counts = [Counter(g['class_id'] for g in values) for values in (current['ground_truth'],found)]
                frame = cv2.copyMakeBorder(frame,30,0,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
                cv2.putText(frame,f'GT U={gt_counts[0]} F={gt_counts[1]} | Pred U={pred_counts[0]} F={pred_counts[1]}',
                            (8,21),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                frames.append(frame)
            variant = name+'_'+str(iou).replace('.','')
            folder = save(variant,rows)
            for record,frame in zip(records,frames):
                write_image(folder/(Path(record['image']).stem+'.jpg'),frame)
    reference = read(ROOT/'runs/e_bike/nms_comparison_single_20261118/summary.json')
    write(exp/'evaluation/summary.json',{'variants':report,
        'reference_variants':reference['variants'],
        'detector_weights':detector_weight.relative_to(ROOT).as_posix(),
        'classifier_weights':(exp/'classifier/weights/best.pt').relative_to(ROOT).as_posix(),
        'classifier_names':classifier.names,
        'settings':{'detector_imgsz':640,'classifier_imgsz':160,'confidence':.25,'nms_ious':[.5,.3],'crop_padding':.08},
        'scope':'Same 26 known-source validation images and prior 156 training images. Oracle uses GT boxes only to isolate classification errors. Reclassified old detector isolates classification; one-class pipeline is end-to-end. No deployment change or independent test.'})
    cards = []
    for record in records:
        stem = Path(record['image']).stem
        figures = ''.join(f'<figure><figcaption>{name}</figcaption><a href="{name}/{stem}.jpg"><img loading="lazy" src="{name}/{stem}.jpg"></a></figure>'
                          for name in report if name!='oracle_gt_boxes')
        cards.append(f'<h2>{html.escape(stem)}</h2><div class="grid">{figures}</div>')
    (exp/'evaluation/review.html').write_text('<!doctype html><meta charset="utf-8"><title>分车与姿态两阶段实验</title>'
        '<style>body{font:16px sans-serif;margin:24px;background:#eee}.grid{display:grid;grid-template-columns:repeat(2,minmax(300px,1fr));gap:12px}figure{margin:0;background:white;padding:8px}img{width:100%}</style>'
        '<h1>分车后判姿态：两种检测器与去重对照</h1><p>GT标注数量；Pred预测数量。研究实验，同源验证。</p>'+''.join(cards),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('prepare','classifier','classifier-child','evaluate'))
    p.add_argument('--experiment',required=True)
    p.add_argument('--inputs',default='runs/e_bike/synthetic_short_inputs_20261118/manifest.json')
    p.add_argument('--detector',default='runs/e_bike/two_stage_detector_20261118/training/weights/best.pt')
    args = p.parse_args()
    exp = resolve(ROOT,args.experiment)
    if not exp.is_relative_to(ROOT/'runs/e_bike'):
        raise ValueError('Experiment must be under runs/e_bike')
    cv2.setNumThreads(1);torch.set_num_threads(2)
    if args.phase=='prepare':
        prepare(exp,resolve(ROOT,args.inputs))
    elif args.phase=='classifier-child':
        train_classifier(exp)
    elif args.phase=='classifier':
        write(exp/'classifier_budget.json',{'epochs':10,'wall_seconds':600})
        run_bounded([sys.executable,'-X','utf8','-B',str(Path(__file__)),'classifier-child','--experiment',args.experiment],
                    time.monotonic()+600,exp/'classifier_train.log',ROOT)
    else:
        evaluate(exp,resolve(ROOT,args.detector))


if __name__=='__main__':
    main()
