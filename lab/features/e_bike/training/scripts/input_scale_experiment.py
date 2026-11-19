"""Frozen detector/classifier input-scale and exact pre-NMS candidate diagnostic."""
import argparse
from collections import Counter
import copy
from datetime import datetime,timezone
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from lab.features.e_bike.training.scripts.keypoint_experiment import read,write
from lab.features.e_bike.training.scripts.two_stage_experiment import crop,truths,compact
from lab.training_common.engine.runtime import setup_paths,prepare_ultralytics
from lab.training_common.engine.reporting import box_iou,match_detections
from datasets.scripts.common.images import read_image,write_image

WEIGHT='runs/e_bike/keypoint_control_20261118/training/weights/best.pt'
CLASSIFIER='runs/e_bike/two_stage_20261118/classifier/weights/best.pt'
INPUT='runs/e_bike/synthetic_short_inputs_20261118/manifest.json'
REFERENCE='runs/e_bike/keypoint_control_pipeline_20261118/evaluation/box_03/predictions.jsonl'
FAR_CROPS=((0,.50,.65,.88),(.35,.50,1,.88))
MODES=('full_640','full_960','full_1280','full640_far640')
DIAGNOSTIC_LIMITS=[
    'Candidate match means decoded bbox IoU>=0.5, not semantic identification. A score below .01 is not an absence of all raw model outputs.',
    'GT only used for audit and marked previews. Combined-crop reason is the furthest surviving stage across passes; per-pass evidence retains multiple causes.',
    'Fixed far windows are image-coordinate heuristics for mixed camera views, not calibrated physical far zones.',
    'Native forward timings exclude classifier and crop fusion. Full640 contribution to combined-mode timing is reused; measurements are instrumented offline, not deployment profiling.']


def crop_rectangle(shape,rect):
    h,w=shape[:2];return [round(rect[0]*w),round(rect[1]*h),round(rect[2]*w),round(rect[3]*h)]


def translate_boxes(boxes,left,top):
    return np.asarray(boxes).reshape(-1,4)+np.array([left,top,left,top])


def overlaps(boxes,box):
    boxes=np.asarray(boxes).reshape(-1,4)
    lo=np.maximum(boxes[:,:2],box[:2]);hi=np.minimum(boxes[:,2:],box[2:])
    inter=np.maximum(hi-lo,0).prod(1)
    area=np.maximum(boxes[:,2:]-boxes[:,:2],0).prod(1)
    gt_area=max(0,box[2]-box[0])*max(0,box[3]-box[1])
    return inter/np.maximum(area+gt_area-inter,1e-9)


def diagnose_gt(gt,passes,final,entity_matches):
    """Only evaluation reads GT. Candidate generation and crop definitions never do."""
    matched={i:j for i,j,_ in entity_matches};events=[]
    for i,g in enumerate(gt):
        maximum=0.;strong=False;local=False;filtered=False;limited=False;evidence=[]
        for p in passes:
            ious=overlaps(p['boxes'],g['xyxy']);ids=np.where(ious>=.5)[0]
            scores=p['scores'][ids]
            if len(scores):maximum=max(maximum,float(scores.max()))
            strong=strong or bool(np.any(scores>.25))
            local=local or any(box_iou(v['xyxy'],g['xyxy'])>=.5 for v in p['after_local_nms'])
            filtered=filtered or any(box_iou(v['xyxy'],g['xyxy'])>=.5 for v in p['after_border_filter'])
            limited=limited or p['postprocess_count']>=100
            witnesses=[]
            for raw_id in sorted(ids,key=lambda n:-p['scores'][n])[:5]:
                if p['scores'][raw_id]<=.25:continue
                blockers=[q for q in p['after_local_nms'] if box_iou(q['xyxy'],p['boxes'][raw_id])>.3 and q['confidence']>=float(p['scores'][raw_id])-1e-6]
                if blockers:
                    witnesses.append({'raw_id':int(raw_id),'raw_box':p['boxes'][raw_id].tolist(),
                                      'raw_score':float(p['scores'][raw_id]),'gt_iou':float(ious[raw_id]),
                                      'retained_overlap_boxes':blockers})
            evidence.append({'pass':p.get('name','test'),'best_raw_score_at_iou05':float(scores.max()) if len(scores) else 0.,
                             'raw_matching_above025':int(np.sum(scores>.25)),
                             'post_nms_coverage':any(box_iou(v['xyxy'],g['xyxy'])>=.5 for v in p['after_local_nms']),
                             'post_border_coverage':any(box_iou(v['xyxy'],g['xyxy'])>=.5 for v in p['after_border_filter']),
                             'high_score_overlap_evidence':witnesses})
        final_cover=any(box_iou(v['xyxy'],g['xyxy'])>=.5 for v in final)
        if i in matched:reason='detected'
        elif final_cover:reason='one_to_one_competition'
        elif filtered:reason='fusion_nms'
        elif local:reason='crop_border_filter'
        elif strong:reason='local_nms_or_limit' if limited else 'local_nms'
        elif maximum>.01:reason='below_confidence_025'
        else:reason='no_candidate_above_001'
        events.append({'gt_index':i,'reason':reason,'best_raw_score_at_iou05':maximum,
                       'evidence_by_pass':evidence,
                       'pose_wrong':i in matched and final[matched[i]]['class_id']!=g['class_id']})
    return events


def make_predictor():
    # Local installed API supplies all decoding, coordinate mapping and NMS.
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from ultralytics.utils import ops,nms
    class CandidatePredictor(DetectionPredictor):
        def postprocess(self,preds,img,orig_imgs,**kwargs):
            decoded=preds[0] if isinstance(preds,(tuple,list)) else preds
            if decoded.ndim!=3 or decoded.shape[0]!=1 or decoded.shape[1]!=5:
                raise ValueError('Expected batch-one, single-class YOLO decoded BCH output')
            if getattr(self.model,'end2end',False):raise ValueError('Expected NMS-based detector')
            self.decoded_before_nms=decoded.detach().clone().cpu()
            tensor=decoded[0].detach().clone().transpose(0,1)
            boxes=ops.xywh2xyxy(tensor[:,:4])
            self.raw_boxes=ops.scale_boxes(img.shape[2:],boxes,orig_imgs[0].shape).cpu().numpy()
            self.raw_scores=tensor[:,4].cpu().numpy()
            self.actual_tensor_shape=list(img.shape)
            output,ids=nms.non_max_suppression(preds,self.args.conf,self.args.iou,
                classes=self.args.classes,agnostic=self.args.agnostic_nms,max_det=self.args.max_det,
                return_idxs=True)
            self.kept_raw_ids=ids[0].reshape(-1).cpu().tolist()
            return self.construct_results(output,img,orig_imgs,**kwargs)
    return CandidatePredictor


def run_pass(model,frame,size,name,offset=(0,0),border_filter=False):
    torch.cuda.synchronize();start=time.perf_counter()
    result=model.predict(source=frame,imgsz=size,rect=True,conf=.25,iou=.3,
                         agnostic_nms=True,max_det=100,device='0',half=False,verbose=False)[0]
    torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
    predictor=model.predictor
    found=[]
    for box,score,index in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),predictor.kept_raw_ids):
        found.append({'class_id':0,'xyxy':translate_boxes([box],*offset)[0].tolist(),
                      'confidence':float(score),'candidate_id':f'{name}:{index}'})
    kept=[];h,w=frame.shape[:2]
    for p in found:
        x1,y1,x2,y2=np.array(p['xyxy'])-np.array([*offset,*offset])
        if not border_filter or (x1>2 and y1>2 and x2<w-2 and y2<h-2):kept.append(p)
    return {'name':name,'boxes':translate_boxes(predictor.raw_boxes,*offset),'scores':predictor.raw_scores.copy(),
            'after_local_nms':found,'after_border_filter':kept,'postprocess_count':len(found),
            'actual_tensor_shape':predictor.actual_tensor_shape,'detector_wall_ms':elapsed,
            'native_speed_ms':result.speed,'offset':list(offset),'crop_image_size':[w,h]}


def fuse(passes):
    from ultralytics.utils.nms import TorchNMS
    candidates=[copy.deepcopy(p) for view in passes for p in view['after_border_filter']]
    if not candidates:return []
    ids=TorchNMS.nms(torch.tensor([p['xyxy'] for p in candidates],dtype=torch.float32),
                     torch.tensor([p['confidence'] for p in candidates]),.3).tolist()[:100]
    return [candidates[i] for i in ids]


def classify(classifier,frame,pred):
    if not pred:return
    result=classifier.predict(source=[crop(frame,p['xyxy'])[0] for p in pred],imgsz=160,device='0',verbose=False)
    for p,c in zip(pred,result):
        p['class_id']=('upright','fallen').index(classifier.names[c.probs.top1])
        p['pose_confidence']=float(c.probs.top1conf)


def render(frame,row,mode):
    image=frame.copy()
    if mode=='full640_far640':
        for rect in FAR_CROPS:
            x1,y1,x2,y2=crop_rectangle(frame.shape,rect);cv2.rectangle(image,(x1,y1),(x2,y2),(255,180,0),1)
    for p in row['predictions']:
        x1,y1,x2,y2=map(round,p['xyxy']);color=(40,220,40) if p['class_id']==0 else (0,160,255)
        cv2.rectangle(image,(x1,y1),(x2,y2),color,1)
        cv2.putText(image,f'{p["confidence"]:.2f}',(x1,y1-3),0,.35,color,1)
    colors={'detected':(180,180,180),'below_confidence_025':(0,255,255),'no_candidate_above_001':(0,0,255),
            'local_nms':(255,0,255),'fusion_nms':(255,0,255),'crop_border_filter':(255,180,0)}
    for event in row['candidate_audit']:
        if event['reason']=='detected':continue
        g=row['ground_truth'][event['gt_index']];x1,y1,x2,y2=map(round,g['xyxy'])
        color=colors.get(event['reason'],(255,255,255));cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
        cv2.putText(image,f'G{event["gt_index"]}:{event["reason"]}',(max(0,x1-10),min(image.shape[0]-4,y2+12)),0,.32,color,1)
    g,p=[Counter(o['class_id'] for o in objects) for objects in (row['ground_truth'],row['predictions'])]
    image=cv2.copyMakeBorder(image,48,0,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
    cv2.putText(image,mode,(6,18),0,.48,(255,255,255),1)
    cv2.putText(image,f'GT U={g[0]} F={g[1]} | Pred U={p[0]} F={p[1]}',(6,38),0,.48,(255,255,255),1)
    return image


def write_review(exp,all_rows,summary):
    labels={'full_640':'全图640','full_960':'全图960','full_1280':'全图1280','full640_far640':'640全图＋两块远端裁区640'}
    table='<table border="1" cellpadding="8"><tr><th>方案</th><th>两态数量全对</th><th>车辆FP/FN</th><th>检测前向中位ms</th></tr>'
    for m in MODES:
        s=summary['variants'][m];e=s['overall']['entity_counts_and_matching']
        table+=f'<tr><td>{labels[m]}</td><td>{s["overall"]["exact_images"]}/26</td><td>{e["fp"]}/{e["fn"]}</td><td>{s["timing"]["median_native_forward_ms"]:.1f}</td></tr>'
    cards=[];nms_cards=[]
    for row in all_rows['full_640']:
        for event in row['candidate_audit']:
            if event['reason']!='local_nms':continue
            image=read_image(ROOT/row['source_image']);g=row['ground_truth'][event['gt_index']]
            witness=next(v for p in event['evidence_by_pass'] for v in p['high_score_overlap_evidence'])
            blocker=max(witness['retained_overlap_boxes'],key=lambda p:box_iou(p['xyxy'],witness['raw_box']))
            for box,color,label in [(g['xyxy'],(255,255,255),'GT'),
                    (witness['raw_box'],(255,0,255),f'raw {witness["raw_score"]:.3f}'),
                    (blocker['xyxy'],(255,255,0),f'kept {blocker["confidence"]:.3f}')]:
                x1,y1,x2,y2=map(round,box);cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
                cv2.putText(image,label,(max(0,x1),max(14,y1-4)),0,.45,color,1)
            folder=exp/'nms_cases';folder.mkdir(exist_ok=True)
            name=Path(row['source_image']).stem+f'_G{event["gt_index"]}.jpg';write_image(folder/name,image)
            nms_cards.append(f'<figure><figcaption>{html.escape(name)}：白GT／紫删除候选／青保留框；候选与保留框IoU={box_iou(witness["raw_box"],blocker["xyxy"]):.3f}</figcaption><a href="nms_cases/{name}"><img src="nms_cases/{name}"></a></figure>')
    for i,row in enumerate(all_rows['full_640']):
        stem=Path(row['source_image']).stem;figures=[]
        for m in MODES:
            audit=all_rows[m][i]['candidate_audit']
            details='；'.join(f'G{e["gt_index"]}: {e["reason"]}, score={e["best_raw_score_at_iou05"]:.3f}'+('，姿态错' if e['pose_wrong'] else '') for e in audit)
            figures.append(f'<figure><figcaption>{labels[m]}</figcaption><a href="{m}/{stem}.jpg"><img loading="lazy" src="{m}/{stem}.jpg"></a><p>{html.escape(details)}</p><a href="{m}/{stem}_trace.json">候选追溯</a></figure>')
        cards.append(f'<h2>{html.escape(stem)}</h2><div class="grid">'+''.join(figures)+'</div>')
    (exp/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>输入尺度与候选诊断</title><style>body{font:16px sans-serif;margin:24px;background:#eee}.grid{display:grid;grid-template-columns:repeat(2,minmax(300px,1fr));gap:8px}figure{margin:0;background:white;padding:8px}img{width:100%}p{line-height:1.6}</style><h1>固定模型：输入尺度与NMS前候选诊断</h1>'+table+'</table><p>绿框直立、橙框倒伏。未匹配GT：红=没有分数&gt;0.01的IoU≥0.5候选，黄=低于0.25，紫=NMS删除，青=裁区边界过滤。GT只参与审计。分数0.01以下不等于模型完全没有输出。计时为Windows离线诊断，含额外候选捕获；前向时间不包含姿态分类。</p><h2>640的两例NMS漏检证据</h2><div class="grid">'+''.join(nms_cards)+'</div><h2>全部图的四方案对照</h2>'+''.join(cards),encoding='utf-8')
    old=all_rows['full_640'];choices=[]
    for group in ('original_real','synthetic_heldout'):
        candidates=[i for i,r in enumerate(old) if r['group']==group]
        choices.append(max(candidates,key=lambda i:sum(e['reason']!='detected' for e in old[i]['candidate_audit'])))
    strips=[]
    for i in choices:
        stem=Path(old[i]['source_image']).stem
        strips.append(np.concatenate([cv2.resize(read_image(exp/m/(stem+'.jpg')),(384,317)) for m in MODES],axis=1))
    write_image(exp/'comparison_contact.jpg',np.concatenate(strips,axis=0))


def paired_geometry_changes(all_rows):
    def coverage(row):
        matches,_,_=match_detections([{**g,'class_id':0} for g in row['ground_truth']],
                                     [{**p,'class_id':0} for p in row['predictions']],.5)
        return {i for i,_,_ in matches}
    result={}
    for mode,rows in all_rows.items():
        if mode=='full_640':continue
        changes=[{'source_image':a['source_image'],'recovered_gt_indices':sorted(coverage(b)-coverage(a)),
                  'lost_gt_indices':sorted(coverage(a)-coverage(b))} for a,b in zip(all_rows['full_640'],rows)]
        result[mode]={'recovered':sum(len(c['recovered_gt_indices']) for c in changes),
                      'lost':sum(len(c['lost_gt_indices']) for c in changes),'records':changes}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',default='runs/e_bike/input_scale_probe_20261118')
    args=parser.parse_args();exp=(ROOT/args.experiment).resolve()
    if not exp.is_relative_to(ROOT/'runs/e_bike'):raise ValueError('Expected e_bike research output')
    exp.mkdir(parents=True,exist_ok=False);setup_paths(exp)
    cv2.setNumThreads(1);torch.set_num_threads(2)
    snapshot=read(ROOT/INPUT);records=[r for r in snapshot['records'] if r['split']=='val']
    write(exp/'input_snapshot.json',snapshot)
    config={'training':False,'weights':WEIGHT,'classifier':CLASSIFIER,'input_manifest':INPUT,
            'modes':list(MODES),'far_crop_normalized_xyxy':FAR_CROPS,'far_crop_input':640,
            'confidence':.25,'nms_iou':.3,'max_det':100,'batch':1,'rect':True,'half':False,
            'candidate_floor_for_diagnosis':.01,'match_iou':.5,'classifier_imgsz':160,
            'crop_border_margin_px':2,'policy':'Fixed before inference; raw candidates saved before NMS. GT only read by evaluation.'}
    write(exp/'config.json',config);(exp/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    YOLO=prepare_ultralytics(exp);model=YOLO(str(ROOT/WEIGHT));classifier=YOLO(str(ROOT/CLASSIFIER))
    assert model.names=={0:'e_bike'} and set(classifier.names.values())=={'upright','fallen'}
    predictor_class=make_predictor()
    warm=read_image(ROOT/records[0]['image'])
    model.predict(source=warm,imgsz=640,conf=.25,iou=.3,agnostic_nms=True,max_det=100,rect=True,device='0',half=False,verbose=False,predictor=predictor_class)
    for size in (960,1280):run_pass(model,warm,size,'warm')
    classify(classifier,warm,[{'xyxy':[0,0,160,160]}])
    all_rows={m:[] for m in MODES};peak={m:0 for m in MODES}
    for record in records:
        frame=read_image(ROOT/record['image']);gt=truths(record);stem=Path(record['image']).stem
        passes={}
        for size in (640,960,1280):
            torch.cuda.reset_peak_memory_stats();passes[f'full_{size}']=run_pass(model,frame,size,f'full_{size}')
            peak[f'full_{size}']=max(peak[f'full_{size}'],torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats();far=[passes['full_640']]
        for i,rect in enumerate(FAR_CROPS):
            x1,y1,x2,y2=crop_rectangle(frame.shape,rect)
            far.append(run_pass(model,frame[y1:y2,x1:x2],640,f'far_{i}',(x1,y1),True))
        peak['full640_far640']=max(peak['full640_far640'],torch.cuda.max_memory_allocated())
        for mode in MODES:
            selected=far if mode=='full640_far640' else [passes[mode]]
            pred=fuse(selected) if mode=='full640_far640' else copy.deepcopy(selected[0]['after_border_filter'])
            torch.cuda.synchronize();start=time.perf_counter();classify(classifier,frame,pred);torch.cuda.synchronize()
            class_ms=(time.perf_counter()-start)*1000
            entity_matches,_,_=match_detections([{**g,'class_id':0} for g in gt],[{**p,'class_id':0} for p in pred],.5)
            matches,miss,extra=match_detections(gt,pred,.5)
            row={'source_image':record['image'],'group':'synthetic_heldout' if record.get('condition') else 'original_real',
                 'ground_truth':gt,'predictions':pred,'matches':matches,'missed_gt_indices':miss,'false_prediction_indices':extra,
                 'candidate_audit':diagnose_gt(gt,selected,pred,entity_matches),'classifier_wall_ms':class_ms,
                 'detector_wall_ms':sum(p['detector_wall_ms'] for p in selected),
                 'native_forward_ms':sum(p['native_speed_ms']['inference'] for p in selected)}
            all_rows[mode].append(row);folder=exp/mode;folder.mkdir(exist_ok=True)
            trace=[]
            for p in selected:
                np.savez_compressed(folder/(stem+'_'+p['name']+'_raw.npz'),boxes=p['boxes'],scores=p['scores'])
                trace.append({k:v for k,v in p.items() if k not in ('boxes','scores')})
            write(folder/(stem+'_trace.json'),{'passes':trace,'gt_audit':row['candidate_audit']})
            write_image(folder/(stem+'.jpg'),render(frame,row,mode))
        print(json.dumps({'image':stem,'counts':{m:len(all_rows[m][-1]['predictions']) for m in MODES}}),flush=True)
    variants={}
    for mode,rows in all_rows.items():
        (exp/mode/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        variants[mode]={'overall':compact(rows),'groups':{g:compact([r for r in rows if r['group']==g]) for g in sorted({r['group'] for r in rows})},
                        'candidate_reasons':dict(Counter(e['reason'] for r in rows for e in r['candidate_audit'])),
                        'matched_pose_errors':sum(e['pose_wrong'] for r in rows for e in r['candidate_audit']),
                        'timing':{'median_native_forward_ms':float(np.median([r['native_forward_ms'] for r in rows])),
                                  'median_detector_wall_ms':float(np.median([r['detector_wall_ms'] for r in rows])),
                                  'median_classifier_wall_ms':float(np.median([r['classifier_wall_ms'] for r in rows])),
                                  'peak_detector_allocated_bytes':peak[mode]}}
    reference=[json.loads(l) for l in (ROOT/REFERENCE).read_text(encoding='utf-8').splitlines()]
    reproduction=[]
    for old,new in zip(reference,all_rows['full_640']):
        assert old['source_image']==new['source_image'] and old['ground_truth']==new['ground_truth']
        reproduction.append({'source_image':old['source_image'],'same_count':len(old['predictions'])==len(new['predictions']),
                             'same_poses':Counter(p['class_id'] for p in old['predictions'])==Counter(p['class_id'] for p in new['predictions']),
                             'max_box_difference':float(np.max(np.abs(np.array([p['xyxy'] for p in old['predictions']])-np.array([p['xyxy'] for p in new['predictions']])))) if len(old['predictions'])==len(new['predictions']) and new['predictions'] else None})
    summary={'config':config,'variants':variants,'reference_reproduction':reproduction,
             'weights':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (WEIGHT,CLASSIFIER)},
             'runtime':{'torch':torch.__version__,'device':torch.cuda.get_device_name(0),
                        'ultralytics':importlib.metadata.version('ultralytics'),'numpy':np.__version__},
             'paired_geometry_changes':paired_geometry_changes(all_rows),
             'input_manifest_sha256':hashlib.sha256((ROOT/INPUT).read_bytes()).hexdigest(),
             'completed_at_utc':datetime.now(timezone.utc).isoformat(),'diagnostic_limits':DIAGNOSTIC_LIMITS,
             'scope':'Same known-source 26 images,102 GT. Input-scale diagnosis only, not independent test. No training. Raw candidates include scores below .01; diagnosis uses explicit floors. Timings instrumented Windows offline, not deployment latency.'}
    write(exp/'summary.json',summary);write_review(exp,all_rows,summary)
    print(json.dumps(variants,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
