"""Re-run two checkpoints with fixed NMS variants on the same research validation snapshot."""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import sys

import cv2
import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from lab.training_common.engine.config import resolve
from lab.training_common.engine.reporting import match_detections
from lab.training_common.engine.runtime import setup_paths, prepare_ultralytics
from lab.features.e_bike.training.scripts.report_pose_counts import pose_counts
from datasets.scripts.common.images import write_image

VARIANTS = (('class_07', .7, False), ('class_05', .5, False),
            ('agnostic_05', .5, True), ('agnostic_03', .3, True))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def summarized(rows):
    return {**{k:v for k,v in pose_counts(rows).items() if k!='records'},
            'tp':sum(len(r['matches']) for r in rows),
            'fp':sum(len(r['false_prediction_indices']) for r in rows),
            'fn':sum(len(r['missed_gt_indices']) for r in rows)}


def geometry_coverage(row):
    # Reuse the matcher with a single class to distinguish missed entities from pose errors.
    truths = [{**v, 'class_id':0} for v in row['ground_truth']]
    predictions = [{**v, 'class_id':0} for v in row['predictions']]
    return {i for i,_,_ in match_detections(truths, predictions, .5)[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, help='Research input manifest, with train/val records')
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    inputs = resolve(ROOT, args.inputs)
    manifest = read(inputs)
    records = [r for r in manifest['records'] if r['split']=='val']
    if not records or len({r['image'] for r in records})!=len(records):
        raise ValueError('Expected unique validation images')
    output = resolve(ROOT, args.output)
    if not output.is_relative_to(ROOT/'runs/e_bike'):
        raise ValueError('Research output must be under runs/e_bike')
    output.mkdir(parents=True, exist_ok=False)
    setup_paths(output)
    YOLO = prepare_ultralytics(output)
    from ultralytics import __version__ as ultralytics_version
    (output/'experiment.py').write_bytes(Path(__file__).read_bytes())
    write(output/'arguments.json',vars(args))
    write(output/'input_snapshot.json', manifest)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    all_rows, summaries, models = {}, {}, {}
    for model_name, weight_name in (('baseline', args.baseline), ('candidate', args.candidate)):
        weight = resolve(ROOT, weight_name)
        models[model_name] = {'path':weight_name, 'sha256':hashlib.sha256(weight.read_bytes()).hexdigest()}
        model = YOLO(str(weight))
        if model.names != {0:'upright', 1:'fallen'}:
            raise ValueError('Wrong checkpoint class semantics')
        for variant, iou, agnostic in VARIANTS:
            key = model_name+'_'+variant
            folder = output/key
            folder.mkdir()
            rows = []
            for offset in range(len(records)):
                batch = records[offset:offset+1]
                predictions = model.predict(source=[str(resolve(ROOT,r['image'])) for r in batch],
                    imgsz=640, conf=.25, iou=iou, agnostic_nms=agnostic, max_det=100,
                    device='0', rect=True, verbose=False, save=False)
                for record, result in zip(batch, predictions):
                    w,h = record['source_record']['image_size']
                    truths = []
                    for line in resolve(ROOT,record['label']).read_text(encoding='utf-8').splitlines():
                        cls,x,y,bw,bh = map(float,line.split())
                        truths.append({'class_id':int(cls), 'xyxy':[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]})
                    boxes = result.boxes
                    found = [{'class_id':int(c),'confidence':float(s),'xyxy':b}
                             for c,s,b in zip(boxes.cls.cpu().tolist(),boxes.conf.cpu().tolist(),boxes.xyxy.cpu().tolist())]
                    matches, missed, extra = match_detections(truths, found, .5)
                    row = {'source_image':record['image'], 'ground_truth':truths, 'predictions':found,
                           'matches':matches,'missed_gt_indices':missed,'false_prediction_indices':extra,
                           'group':'synthetic_heldout' if record.get('condition') else 'original_real',
                           'domain':record['source_record']['branch'], 'condition':record.get('condition'),
                           'mirrored':record.get('mirrored',record['source_record'].get('mirrored',False))}
                    rows.append(row)
                    frame = result.plot()
                    gt, pred = [Counter(v['class_id'] for v in values) for values in (truths,found)]
                    frame = cv2.copyMakeBorder(frame,32,0,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
                    cv2.putText(frame,f"GT U={gt[0]} F={gt[1]} | Pred U={pred[0]} F={pred[1]}",
                                (8,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
                    write_image(folder/(Path(record['image']).stem+'.jpg'),frame)
            all_rows[key] = rows
            summaries[key] = {'overall':summarized(rows),
                'groups':{group:summarized([r for r in rows if r['group']==group])
                          for group in sorted({r['group'] for r in rows})},
                'settings':{'confidence':.25,'nms_iou':iou,'agnostic_nms':agnostic,'imgsz':640,'batch':1,'rect':True}}
            (folder/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
            print(json.dumps({'variant':key,**summaries[key]},ensure_ascii=False),flush=True)
    audits = {}
    for model_name in models:
        reference = all_rows[model_name+'_class_07']
        for variant,_,_ in VARIANTS[1:]:
            key = model_name+'_'+variant
            events = []
            for old,new in zip(reference,all_rows[key]):
                assert old['source_image']==new['source_image'] and old['ground_truth']==new['ground_truth']
                old_ids,new_ids = geometry_coverage(old),geometry_coverage(new)
                counts_old,counts_new = pose_counts([old])['records'][0],pose_counts([new])['records'][0]
                events.append({'source_image':old['source_image'],
                    'lost_geometry_gt_indices':sorted(old_ids-new_ids), 'recovered_geometry_gt_indices':sorted(new_ids-old_ids),
                    'before_counts':counts_old,'after_counts':counts_new})
            audits[key] = events
    report = {'models':models, 'input_manifest':args.inputs,
              'runtime':{'torch':torch.__version__,'ultralytics':ultralytics_version,
                         'device':torch.cuda.get_device_name(0),'batch':1,'rect':True},
              'input_manifest_sha256':hashlib.sha256(inputs.read_bytes()).hexdigest(),
              'variants':summaries,'suppression_audit':audits,
              'scope':'Fresh inference on the same 26-image known-source diagnostic val. Threshold exploration, not independent test. Geometry loss/recovery uses class-agnostic one-to-one IoU>=0.5 matching; it is an annotation-based audit, not proof of physical identity.'}
    write(output/'summary.json',report)
    headers = list(all_rows)
    cards = []
    for record in records:
        stem = Path(record['image']).stem
        figures = ''.join(f'<figure><figcaption>{html.escape(key)}</figcaption><a href="{key}/{stem}.jpg"><img loading="lazy" src="{key}/{stem}.jpg"></a></figure>' for key in headers)
        cards.append(f'<h2>{html.escape(stem)}</h2><div class="grid">{figures}</div>')
    (output/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>电动车去重对照</title>'
        '<style>body{font:16px sans-serif;margin:24px;background:#eee}.grid{display:grid;grid-template-columns:repeat(4,minmax(220px,1fr));gap:8px}figure{margin:0;background:white;padding:6px}img{width:100%}</style>'
        '<h1>两轮模型 × 四种去重设置</h1><p>GT 为标注数量，Pred 为预测数量。各图同源，仅作参数探索；逐车误删／恢复审计见 summary.json。</p>'
        +''.join(cards),encoding='utf-8')


if __name__=='__main__':
    main()
