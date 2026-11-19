"""Research pose training inputs and anchor-guided instance suppression for electric bikes."""
import argparse
from collections import Counter
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from datasets.scripts.common.images import read_image, load_rgba, resized, write_image
from datasets.scripts.common.perspective import project_corners, planar_warp
from datasets.scripts.common.composite import visible_masks
from datasets.scripts.e_bike.compose_v4_grounded_train import place_actor
from lab.training_common.engine.config import resolve
from lab.training_common.engine.runtime import setup_paths, prepare_ultralytics
from lab.training_common.engine.reporting import match_detections, box_iou
from lab.features.e_bike.training.scripts.two_stage_experiment import crop, truths, compact


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(p,value):
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def projected_points(source, points, spec, posture):
    """Replay the compositor's trim, homography, alpha crop, resize, mirror and placement."""
    raw = read_image(source,cv2.IMREAD_UNCHANGED)
    yy,xx = np.nonzero(raw[:,:,3]>127)
    sprite = load_rgba(source)
    height,width = sprite.shape[:2]
    destination = project_corners(width,height,spec['yaw'],spec['pitch'],spec['roll'])
    minimum,maximum = destination.min(axis=0),destination.max(axis=0)
    destination -= minimum
    matrix = cv2.getPerspectiveTransform(np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]]),destination)
    warped = cv2.warpPerspective(sprite,matrix,tuple(int(math.ceil(v))+2 for v in maximum-minimum))
    wy,wx = np.nonzero(warped[:,:,3]>8)
    projected = planar_warp(sprite,spec['yaw'],spec['pitch'],spec['roll'])
    final = resized(projected,**({'height':spec['size_px']} if posture=='upright' else {'width':spec['size_px']}))
    transformed = []
    for point in points:
        if point is None:
            transformed.append(None)
            continue
        local = np.float32([[[point[0]-xx.min(),point[1]-yy.min()]]])
        mapped = cv2.perspectiveTransform(local,matrix)[0,0]-[wx.min(),wy.min()]
        x = (mapped[0]+.5)*final.shape[1]/projected.shape[1]-.5
        y = (mapped[1]+.5)*final.shape[0]/projected.shape[0]-.5
        if spec.get('mirrored'):
            x = final.shape[1]-1-x
        transformed.append([float(round(spec['foot_x']-final.shape[1]/2)+x),
                            float(round(spec['foot_y']-final.shape[0])+y)])
    if spec.get('mirrored'):
        final = np.ascontiguousarray(final[:,::-1])
    return final,transformed


def projected_seat(source, point, spec, posture):
    sprite,points = projected_points(source,[point],spec,posture)
    return sprite,points[0]


def synthetic_points(record, anchors, point_names=None):
    source = record['source_record']
    objects = source['objects']
    layers = []
    for obj in objects:
        if not obj.get('foreground_source'):
            continue
        path = obj['foreground_source']['path']
        anchor = anchors[path]
        spec = source['parameters']['actors'][int(obj['object_id'].rsplit('_',1)[1])]
        points = [anchor['points'][name] for name in point_names] if point_names else [anchor['seat_xy']]
        sprite,projected = projected_points(ROOT/path,points,spec,obj['class'])
        full = place_actor(np.zeros((480,640,3),np.uint8),sprite,spec['foot_x'],spec['foot_y'],False,occupancy_only=True)
        layers.append((obj,full,projected))
    visible = visible_masks([r[1] for r in layers])
    result = {}
    for (obj,full,projected),mask in zip(layers,visible):
        y,x = np.nonzero(mask)
        assert [int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1]==obj['bbox_xyxy']
        assert int(mask.sum())==obj['visible_pixels']
        labels = []
        for point in projected:
            label = [0.,0.,0]
            if point is not None:
                px,py = [round(v) for v in point]
                if 0<=px<640 and 0<=py<480 and full[py,px]:
                    label = point+[2 if mask[py,px] else 1]
            labels.append(label)
        result[obj['object_id']] = labels if point_names else labels[0]
    return result


def prepare(exp, inputs, anchor_path):
    exp.mkdir(parents=True,exist_ok=False)
    setup_paths(exp)
    YOLO = prepare_ultralytics(exp)
    source = read(inputs)
    anchors_doc = read(anchor_path)
    semantic_names = anchors_doc.get('point_names')
    point_names = semantic_names or ['visible_box_center','seat_surface_center']
    if len(point_names) not in (2,4) or len(set(point_names))!=len(point_names):
        raise ValueError('Expected the two-point or four-part experiment schema')
    anchors = {r['source']:r for r in anchors_doc['records']}
    for row in anchors.values():
        assert hashlib.sha256((ROOT/row['source']).read_bytes()).hexdigest()==row['sha256']
    write(exp/'input_snapshot.json',source)
    write(exp/'seat_anchor_snapshot.json',anchors_doc)
    records,point_records = [],[]
    for split in ('train','val'):
        for folder in ('images','labels'):
            (exp/'inputs'/split/folder).mkdir(parents=True)
    for original in source['records']:
        row = dict(original)
        frame = read_image(ROOT/row['image']);h,w = frame.shape[:2]
        row_source = row['source_record']
        seat_points = synthetic_points(row,anchors,semantic_names) if row_source['branch']!='direct_real' else {}
        image = exp/'inputs'/row['split']/'images'/Path(row['image']).name
        label = exp/'inputs'/row['split']/'labels'/Path(row['label']).name
        shutil.copyfile(ROOT/row['image'],image)
        lines,instances = [],[]
        for obj in row_source['objects']:
            x1,y1,x2,y2 = obj['bbox_xyxy']
            center = [(x1+x2)/2,(y1+y2)/2,2]
            if semantic_names:
                points = seat_points.get(obj.get('object_id'),[[0.,0.,0] for _ in point_names])
            else:
                points = [center,seat_points.get(obj.get('object_id'),[0.,0.,0])]
            values = [0,(x1+x2)/(2*w),(y1+y2)/(2*h),(x2-x1)/w,(y2-y1)/h]
            for point in points:
                values += [point[0]/w,point[1]/h,point[2]]
            lines.append(' '.join(f'{v:.8f}' for v in values))
            instances.append({'object_id':obj.get('object_id'),'bbox':obj['bbox_xyxy'],**dict(zip(point_names,points))})
            for index,point in enumerate(points):
                color = [(255,255,0),(0,80,255),(255,0,255),(0,255,255)][index]
                if point[2]:
                    cv2.circle(frame,tuple(round(v) for v in point[:2]),3,color,-1)
        label.write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8')
        row.update(image=image.relative_to(ROOT).as_posix(),label=label.relative_to(ROOT).as_posix())
        records.append(row)
        point_records.append({'source_image':original['image'],'split':row['split'],'instances':instances})
        if row['split']=='val':
            (exp/'point_previews').mkdir(exist_ok=True)
            write_image(exp/'point_previews'/image.name,frame)
    write(exp/'point_manifest.json',{'records':point_records,
          'point_names':point_names,
          'center_definition':None if semantic_names else 'Visible-box geometric center, automatically derived; not additional semantic annotation.',
          'seat_definition':anchors_doc['definition'],
          'real_parts':'Not annotated in this experiment. v0 means unannotated, not confirmed absent/occluded.'})
    write(exp/'inputs/data.yaml',{'names':['e_bike'],'train':'train/images','val':'val/images','kpt_shape':[len(point_names),3],'flip_idx':list(range(len(point_names)))})
    write(exp/'inputs/workspace_sources.json',{'model_key':'e_bike','selection_mode':'manifest_only','records':records,'source_reuse':[]})
    torch.manual_seed(42)
    from ultralytics.nn.tasks import PoseModel
    model = YOLO('yolov8n-pose.yaml')
    config = dict(model.model.yaml)
    config.update(nc=1,kpt_shape=[len(point_names),3])
    model.model = PoseModel(config,ch=3,nc=1,data_kpt_shape=[len(point_names),3],verbose=False)
    model.model.load(YOLO(str(ROOT/'runs/e_bike/two_stage_detector_20261118/training/weights/best.pt')).model)
    model.save(exp/'initialized_pose.pt')
    config = yaml.safe_load((ROOT/'runs/e_bike/synthetic_short_inputs_20261118/train.yaml').read_text(encoding='utf-8'))
    config.update(task='pose',data=(exp/'inputs/data.yaml').relative_to(ROOT).as_posix(),
                  model=(exp/'initialized_pose.pt').relative_to(ROOT).as_posix(),lr0=.0003,nms_iou=.5,
                  keypoint_names=point_names,description='Ten-epoch visually annotated synthetic part-anchor experiment')
    (exp/'train.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
    counts = {split:{name:dict(Counter(p[name][2] for r in point_records if r['split']==split for p in r['instances'])) for name in point_names} for split in ('train','val')}
    write(exp/'preparation.json',{'images':dict(Counter(r['split'] for r in records)),'point_visibility_counts':counts,'point_names':point_names,
          'source_seat_points':sum(r.get('seat_xy',r.get('points',{}).get('seat_surface_center')) is not None for r in anchors.values()),'source_views':len(anchors),
          'scope':'Research pose snapshot; AI-reviewed approximate synthetic semantic points, unknowns unannotated. Same original splits, no new independent sources.'})
    print(read(exp/'preparation.json'),flush=True)


def anchor_suppress(predictions, mode, fraction):
    """Suppress by predicted anchor distance, without reading GT counts or positions."""
    def valid(prediction,index):
        x,y,confidence = prediction['keypoints'][index]
        x1,y1,x2,y2 = prediction['xyxy']
        dx,dy = .1*(x2-x1),.1*(y2-y1)
        return confidence>=.35 and np.isfinite([x,y]).all() and x1-dx<=x<=x2+dx and y1-dy<=y<=y2+dy

    def center(prediction):
        if valid(prediction,0):
            return prediction['keypoints'][0][:2]
        x1,y1,x2,y2 = prediction['xyxy']
        return [(x1+x2)/2,(y1+y2)/2]

    kept = []
    for candidate in sorted(predictions,key=lambda p:-p['confidence']):
        duplicate = False
        for other in kept:
            use_seats = mode=='seat' and valid(candidate,1) and valid(other,1)
            points = [p['keypoints'][1][:2] if use_seats else center(p) for p in (candidate,other)]
            sizes = [min(p['xyxy'][2]-p['xyxy'][0],p['xyxy'][3]-p['xyxy'][1]) for p in (candidate,other)]
            if np.linalg.norm(np.asarray(points[0][:2])-points[1][:2]) < max(2.,fraction*min(sizes)):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def parts_suppress(predictions, fraction):
    """At least two reliable corresponding parts can distinguish overlapping vehicles.

    If correspondence is unavailable, use the fixed box IoU=0.3 rule rather than
    treating unknown points as evidence that two boxes contain different vehicles.
    """
    def valid(p,k):
        x,y,c = p['keypoints'][k]
        x1,y1,x2,y2 = p['xyxy']
        return c>=.35 and np.isfinite([x,y]).all() and x1<=x<=x2 and y1<=y<=y2
    kept = []
    for candidate in sorted(predictions,key=lambda p:-p['confidence']):
        duplicate = False
        for other in kept:
            if box_iou(candidate['xyxy'],other['xyxy'])<=.3:
                continue
            shared = [k for k in range(4) if valid(candidate,k) and valid(other,k)]
            if len(shared)<2:
                duplicate = True
            else:
                size = max(2.,min(min(p['xyxy'][2]-p['xyxy'][0],p['xyxy'][3]-p['xyxy'][1]) for p in (candidate,other)))
                distances = [np.linalg.norm(np.array(candidate['keypoints'][k][:2])-other['keypoints'][k][:2])/size for k in shared]
                duplicate = np.median(distances)<fraction and max(distances)<2*fraction
            if duplicate:
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def evaluate(exp, weights):
    setup_paths(exp)
    YOLO = prepare_ultralytics(exp)
    detector = YOLO(str(weights))
    classifier = YOLO(str(ROOT/'runs/e_bike/two_stage_20261118/classifier/weights/best.pt'))
    records = [r for r in read(exp/'input_snapshot.json')['records'] if r['split']=='val']
    point_names = read(exp/'point_manifest.json').get('point_names',['visible_box_center','seat_surface_center']) if (exp/'point_manifest.json').is_file() else []
    if detector.task=='pose' and list(getattr(detector.model,'kpt_shape',[]))!=[len(point_names),3]:
        raise ValueError('Checkpoint keypoint shape does not match the input snapshot point order')
    variants = [('box_05',.5,None,0),('box_03',.3,None,0)]
    if detector.task=='pose':
        if len(point_names)==4:
            variants += [('parts_015',.95,'parts',.15),('parts_030',.95,'parts',.3)]
        else:
            variants += [('center_015',.95,'center',.15),('seat_015',.95,'seat',.15),('seat_030',.95,'seat',.3)]
    report = {}
    for name,iou,mode,fraction in variants:
        folder = exp/'evaluation'/name;folder.mkdir(parents=True,exist_ok=False)
        rows = []
        for record in records:
            frame = read_image(ROOT/record['image'])
            result = detector.predict(source=str(ROOT/record['image']),imgsz=640,rect=True,conf=.25,
                iou=iou,agnostic_nms=True,max_det=100,device='0',verbose=False)[0]
            keypoints = result.keypoints.data.cpu().tolist() if result.keypoints is not None else [[] for _ in result.boxes]
            found = [{'class_id':0,'xyxy':b,'confidence':float(c),'keypoints':k}
                     for b,c,k in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),keypoints)]
            if mode:
                found = parts_suppress(found,fraction) if mode=='parts' else anchor_suppress(found,mode,fraction)
            if found:
                classes = classifier.predict(source=[crop(frame,p['xyxy'])[0] for p in found],imgsz=160,device='0',verbose=False)
                for prediction,cls in zip(found,classes):
                    prediction['class_id'] = ('upright','fallen').index(classifier.names[cls.probs.top1])
            gt = truths(record);matches,missed,extra = match_detections(gt,found,.5)
            rows.append({'source_image':record['image'],'group':record.get('evaluation_group') or ('synthetic_heldout' if record.get('condition') else 'original_real'),
                         'ground_truth':gt,'predictions':found,'matches':matches,'missed_gt_indices':missed,'false_prediction_indices':extra})
            for p in found:
                x1,y1,x2,y2 = [round(v) for v in p['xyxy']]
                cv2.rectangle(frame,(x1,y1),(x2,y2),(40,220,40) if p['class_id']==0 else (0,160,255),1)
                for index,color in enumerate([(255,255,0),(0,80,255),(255,0,255),(0,255,255)][:len(point_names)]):
                    if len(p['keypoints'])>index and p['keypoints'][index][2]>=.35:
                        cv2.circle(frame,tuple(round(v) for v in p['keypoints'][index][:2]),3,color,-1)
            g,p = [Counter(v['class_id'] for v in values) for values in (gt,found)]
            frame = cv2.copyMakeBorder(frame,30,0,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
            cv2.putText(frame,f'GT U={g[0]} F={g[1]} | Pred U={p[0]} F={p[1]}',(6,21),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            write_image(folder/(Path(record['image']).stem+'.jpg'),frame)
        report[name] = {'overall':compact(rows),'groups':{g:compact([r for r in rows if r['group']==g]) for g in sorted({r['group'] for r in rows})}}
        (folder/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        print(json.dumps({name:report[name]},ensure_ascii=False),flush=True)
    write(exp/'evaluation/summary.json',{'variants':report,'weights':weights.relative_to(ROOT).as_posix(),'task':detector.task,'point_names':point_names,
        'classifier':'runs/e_bike/two_stage_20261118/classifier/weights/best.pt','settings':{'confidence':.25,'imgsz':640,'max_det':100,'anchor_seat_confidence':.35},
        'scope':f'Same-source {len(records)}-image diagnostic. Predicted points only used at inference. No independent test; real semantic points unannotated. Bbox-derived center only in the old two-point schema.'})
    cards = []
    for record in records:
        stem = Path(record['image']).stem
        figures = ''.join(f'<figure><figcaption>{html.escape(name)}</figcaption><a href="{name}/{stem}.jpg"><img loading="lazy" src="{name}/{stem}.jpg"></a></figure>' for name in report)
        cards.append(f'<h2>{html.escape(stem)}</h2><div class="grid">{figures}</div>')
    (exp/'evaluation/review.html').write_text('<!doctype html><meta charset="utf-8"><title>车辆关键点辅助分车</title><style>body{font:16px sans-serif;margin:24px;background:#eee}.grid{display:grid;grid-template-columns:repeat(3,minmax(260px,1fr));gap:8px}figure{margin:0;background:white;padding:8px}img{width:100%}</style><h1>车辆关键点辅助分车</h1><p>点序与颜色：'+html.escape(str(point_names))+'，依次青、橙红、紫、黄；置信度至少0.35。GT标注，Pred预测。研究对照。</p>'+''.join(cards),encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('prepare','evaluate'))
    p.add_argument('--experiment',required=True)
    p.add_argument('--inputs',default='runs/e_bike/synthetic_short_inputs_20261118/manifest.json')
    p.add_argument('--anchors',default='datasets/annotated/e_bike/seat_anchor_review.json')
    p.add_argument('--weights',default='runs/e_bike/keypoint_micro_20261118/training/weights/best.pt')
    args = p.parse_args();exp = resolve(ROOT,args.experiment)
    if not exp.is_relative_to(ROOT/'runs/e_bike'):
        raise ValueError('Research output must be under runs/e_bike')
    cv2.setNumThreads(1);torch.set_num_threads(2)
    if args.phase=='prepare':
        prepare(exp,resolve(ROOT,args.inputs),resolve(ROOT,args.anchors))
    else:
        evaluate(exp,resolve(ROOT,args.weights))


if __name__=='__main__':
    main()
