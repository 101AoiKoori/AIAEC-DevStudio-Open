"""Bounded RT-DETR-L feasibility experiment using existing audited bike data and posture classifier."""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import sys
import time

import cv2
import torch
import yaml

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from lab.training_common.engine.config import resolve, load_config
from lab.training_common.engine.runtime import setup_paths, prepare_ultralytics, run_context, new_run, training_lock
from lab.training_common.engine.data import stage_dataset, preflight_dataset
from lab.training_common.engine.runner import _training_options
from lab.training_common.engine.bounded import run_bounded
from lab.training_common.engine.reporting import match_detections
from lab.features.e_bike.training.scripts.two_stage_experiment import crop, truths, compact
from datasets.scripts.common.images import read_image, write_image


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def prepare(exp):
    config = load_config(ROOT,'runs/e_bike/two_stage_20261118/detector.yaml')
    # Explicit AdamW bypasses the library's automatic optimizer path, which zeroes bias warmup.
    config.update(model='models/weights/rtdetr/rtdetr-l.pt',epochs=10,batch=2,lr0=.0001,warmup_bias_lr=0.,
                  weight_decay=.0001,deterministic=False,
                  description='RT-DETR-L ten-epoch COCO-to-miniature-bike feasibility experiment',
                  source_notes=['Same156 train and26 val images; source groups and derivative exposure retained.',
                                'RT-DETR uses generic COCO initialization; YOLO references have prior bike training. Not a matched-initialization architecture comparison.'])
    if exp.parent!=ROOT/'runs/e_bike':
        raise ValueError('Experiment must be a direct child of runs/e_bike')
    new_run(ROOT,config,exp.name)
    with run_context(ROOT,config,exp,'prepare'):
        _,_,report = preflight_dataset(ROOT,config)
        path,_ = stage_dataset(ROOT,config,exp,training=True)
        (exp/'config/experiment.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
        write(exp/'input_snapshot.json',read(ROOT/'runs/e_bike/synthetic_short_inputs_20261118/manifest.json'))
        write(exp/'metadata/data_check.json',report)
        write(exp/'metadata/pretrained_source.json',read(ROOT/'models/weights/rtdetr/rtdetr-l.source.json'))
        write(exp/'metadata/budget.json',{'epochs':10,'wall_seconds':900,'batch':2,'imgsz':640,
              'inference_confidences':[.1,.25,.5],'postprocessing':'Native RTDETRPredictor, no box NMS.'})
        print({'staged_data':str(path),'available_images':report['available_images']},flush=True)


def train(exp):
    config = load_config(ROOT,(exp/'config/experiment.yaml').relative_to(ROOT).as_posix())
    with training_lock(exp),run_context(ROOT,config,exp,'rtdetr_train'):
        prepare_ultralytics(exp)
        from ultralytics import RTDETR
        model = RTDETR(str(resolve(ROOT,config['model'])))
        options = _training_options(config,exp/'data/data.yaml',exp)
        write(exp/'config/native_train_options.json',options)
        def finite_batch(trainer):
            if trainer.loss is not None and not torch.isfinite(trainer.loss).all():
                raise RuntimeError('RT-DETR training produced non-finite loss')
        model.add_callback('on_train_batch_end',finite_batch)
        model.train(**options)


def evaluate(exp,weight):
    config = load_config(ROOT,(exp/'config/experiment.yaml').relative_to(ROOT).as_posix())
    with run_context(ROOT,config,exp,'rtdetr_evaluate'):
        YOLO = prepare_ultralytics(exp)
        from ultralytics import RTDETR
        detector = RTDETR(str(weight))
        if detector.names!={0:'e_bike'}:
            raise ValueError('Expected the trained one-class bike checkpoint')
        classifier = YOLO(str(ROOT/'runs/e_bike/two_stage_20261118/classifier/weights/best.pt'))
        records = [r for r in read(exp/'input_snapshot.json')['records'] if r['split']=='val']
        reports = {}
        for threshold in (.1,.25,.5):
            name = 'conf_'+str(threshold).replace('.','')
            folder = exp/'evaluation'/name;folder.mkdir(parents=True,exist_ok=False)
            rows = []
            for record in records:
                result = detector.predict(source=str(ROOT/record['image']),imgsz=640,conf=threshold,
                                          max_det=100,device='0',verbose=False)[0]
                frame = read_image(ROOT/record['image'])
                boxes = result.boxes.xyxy.cpu().tolist()
                found = []
                if boxes:
                    classes = classifier.predict(source=[crop(frame,b)[0] for b in boxes],imgsz=160,device='0',verbose=False)
                    for box,score,cls in zip(boxes,result.boxes.conf.cpu().tolist(),classes):
                        found.append({'xyxy':box,'confidence':float(score),
                                      'class_id':('upright','fallen').index(classifier.names[cls.probs.top1])})
                gt = truths(record);matched,missed,extra = match_detections(gt,found,.5)
                rows.append({'source_image':record['image'],'group':'synthetic_heldout' if record.get('condition') else 'original_real',
                    'ground_truth':gt,'predictions':found,'matches':matched,'missed_gt_indices':missed,'false_prediction_indices':extra})
                for p in found:
                    x1,y1,x2,y2 = [round(v) for v in p['xyxy']]
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(40,220,40) if p['class_id']==0 else (0,160,255),1)
                g,p = [Counter(v['class_id'] for v in values) for values in (gt,found)]
                frame = cv2.copyMakeBorder(frame,30,0,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
                cv2.putText(frame,f'GT U={g[0]} F={g[1]} | Pred U={p[0]} F={p[1]}',(6,21),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                write_image(folder/(Path(record['image']).stem+'.jpg'),frame)
            reports[name] = {'overall':compact(rows),'groups':{g:compact([r for r in rows if r['group']==g]) for g in sorted({r['group'] for r in rows})}}
            (folder/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
            print(json.dumps({name:reports[name]},ensure_ascii=False),flush=True)
        write(exp/'evaluation/summary.json',{'variants':reports,'weights':weight.relative_to(ROOT).as_posix(),
              'sha256':hashlib.sha256(weight.read_bytes()).hexdigest(),
              'scope':'Known-source26-image diagnostic after ten-epoch COCO-pretrained RT-DETR-L fine-tuning. Native NMS-free predictor and same prior crop posture classifier. Three predeclared confidence settings; not independent test or matched-initialization YOLO comparison.'})
        cards = []
        for record in records:
            stem = Path(record['image']).stem
            cards.append('<h2>'+html.escape(stem)+'</h2><div class="grid">'+''.join(
                f'<figure><figcaption>{name}</figcaption><a href="{name}/{stem}.jpg"><img loading="lazy" src="{name}/{stem}.jpg"></a></figure>' for name in reports)+'</div>')
        (exp/'evaluation/review.html').write_text('<!doctype html><meta charset="utf-8"><title>RT-DETR小实验</title><style>body{font:16px sans-serif;margin:24px;background:#eee}.grid{display:grid;grid-template-columns:repeat(3,minmax(260px,1fr));gap:8px}figure{margin:0;padding:8px;background:white}img{width:100%}</style><h1>RT-DETR原生无NMS推理＋姿态分类</h1><p>同图三档置信度；GT标注，Pred预测。仅作同源研究诊断。</p>'+''.join(cards),encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('prepare','train','train-child','evaluate'))
    p.add_argument('--experiment',required=True)
    p.add_argument('--weights')
    args = p.parse_args();exp = resolve(ROOT,args.experiment)
    if not exp.is_relative_to(ROOT/'runs/e_bike'):
        raise ValueError('Expected research output under runs/e_bike')
    cv2.setNumThreads(1);torch.set_num_threads(2)
    if args.phase=='prepare':
        prepare(exp)
    elif args.phase=='train-child':
        train(exp)
    elif args.phase=='train':
        run_bounded([sys.executable,'-X','utf8','-B',str(Path(__file__)),'train-child','--experiment',args.experiment],
                    time.monotonic()+900,exp/'logs/bounded_train.log',ROOT)
    else:
        evaluate(exp,resolve(ROOT,args.weights) if args.weights else exp/'training/weights/last.pt')


if __name__=='__main__':
    main()
