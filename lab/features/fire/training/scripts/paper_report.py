"""Paper-detector diagnostics bound to final boxes and known generation controls."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from engine.reporting import match_detections


def load_metadata(path, root, split):
    data=json.loads(Path(path).read_text(encoding='utf8'))
    if data.get('model_key')!='building_name_paper':raise ValueError('Expected building_name_paper source index')
    result={}
    for row in data['records']:
        if row.get('split')!=split:continue
        for key in ('image','workspace_image'):
            alias=row.get(key)
            if not isinstance(alias,str) or not alias:continue
            aliases={alias.replace('\\','/')}
            resolved=(Path(path).parent/alias).resolve()
            if resolved.is_relative_to(root):aliases.add(resolved.relative_to(root).as_posix())
            for name in aliases:
                if name in result and result[name] is not row:raise ValueError('Ambiguous paper image identity: '+name)
                result[name]=row
    return result


def checked_metadata(row, record):
    meta=record.get('paper') or {};shape=row.get('shape_hw')
    if (meta.get('schema_version')!=1 or not isinstance(shape,list) or len(shape)!=2 or
        any(type(v) is not int or v<=0 for v in shape) or meta.get('image_size')!=[shape[1],shape[0]]):
        raise ValueError('Paper metadata image dimensions differ from evaluation')
    items=meta.get('instances')
    if not isinstance(items,list) or len(items)!=len(row['ground_truth']):raise ValueError('Paper metadata count differs from labels')
    for item,truth in zip(items,row['ground_truth']):
        box=item.get('xyxy')
        if (truth.get('class_id')!=0 or not isinstance(box,list) or len(box)!=4 or
            any(type(v) not in (int,float) or not math.isfinite(v) for v in box) or
            max(abs(a-b) for a,b in zip(box,truth['xyxy']))>.02):
            raise ValueError('Paper metadata box order or coordinates differ from labels')
    scene=meta.get('scene') or {};buildings=scene.get('buildings');papers=scene.get('papers')
    if buildings is not None:
        if (not isinstance(buildings,str) or not buildings or len(set(buildings))!=len(buildings) or
            set(buildings)-set('ABC') or not isinstance(papers,str) or len(set(papers))!=len(papers) or
            set(papers)-set(buildings) or len(papers)!=len(items) or
            {item.get('building_id') for item in items}!=set(papers)):
            raise ValueError('Building/paper controls contradict actual paper instances')
    for field in ('yaw_degrees','camera_height_m'):
        value=scene.get(field)
        if value is not None and (type(value) not in (int,float) or not math.isfinite(value)):
            raise ValueError('Invalid paper camera control')
    return meta


def image_counts():
    return dict(images=0,tp=0,fp=0,fn=0,exact_images=0,count_correct_images=0,
                absolute_count_error=0,negative_images=0,negative_false_positive_images=0)


def finish_images(counts):
    tp,fp,fn=counts['tp'],counts['fp'],counts['fn'];n=counts['images'];neg=counts['negative_images']
    return {**counts,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
        'exact_image_rate':counts['exact_images']/n if n else None,
        'count_correct_rate':counts['count_correct_images']/n if n else None,
        'mean_absolute_count_error':counts['absolute_count_error']/n if n else None,
        'negative_false_positive_rate':counts['negative_false_positive_images']/neg if neg else None}


def finish_instances(counts):
    return {**counts,'recall':counts['matched']/counts['instances'] if counts['instances'] else None}


def size_bucket(value):
    return '0_8' if value<=8 else '8_16' if value<=16 else '16_32' if value<=32 else 'over_32'


def scene_groups(record, meta):
    scene=meta['scene'];buildings=scene.get('buildings');papers=scene.get('papers');yaw=scene.get('yaw_degrees')
    state=('none' if not papers else 'all' if len(papers)==len(buildings) else 'partial') if buildings is not None else 'unknown'
    view=('left_large_yaw' if yaw<=-30 else 'frontal' if abs(yaw)<=15 else 'other_oblique') if yaw is not None else 'unknown'
    return {'branch':scene.get('branch','unknown'),'building_count':str(len(buildings)) if buildings else 'unknown',
        'paper_state':state,'yaw_control':view,'view_basis':scene.get('view_basis','unknown'),
        'camera_height_m':str(scene['camera_height_m']) if scene.get('camera_height_m') is not None else 'unknown',
        'source_group':record.get('source_group') or record.get('source_id') or 'unknown',
        'paper_count':str(len(meta['instances']))}


def build_paper_report(rows, metadata, imgsz, threshold):
    if type(imgsz) is not int or imgsz<=0 or not 0<threshold<=1:raise ValueError('Invalid paper evaluation settings')
    total=image_counts();images_by=defaultdict(lambda:defaultdict(image_counts))
    instances_by=defaultdict(lambda:defaultdict(lambda:dict(instances=0,matched=0,missed=0)))
    for bucket in ('0_8','8_16','16_32','over_32'):instances_by['input_short_side_px'][bucket]
    details=[];seen=set();seen_records=set()
    for row in rows:
        identity=row['source_image'].replace('\\','/')
        if identity in seen:raise ValueError('Repeated evaluated paper image')
        seen.add(identity)
        if identity not in metadata:raise ValueError('Missing paper metadata: '+identity)
        record=metadata[identity]
        if id(record) in seen_records:raise ValueError('Paper aliases repeat the same evaluated image')
        seen_records.add(id(record));meta=checked_metadata(row,record)
        truth,predicted=row['ground_truth'],row['predictions'];h,w=row['shape_hw']
        # Six-decimal YOLO center/size serialization can reconstruct an edge
        # up to 0.75e-6 image dimensions beyond the original clipped boundary.
        epsilon=max(w,h)*1e-6
        for item in truth+predicted:
            box=item.get('xyxy')
            if (type(item.get('class_id')) is not int or item['class_id']!=0 or not isinstance(box,list) or len(box)!=4 or
                any(type(v) not in (int,float) or not math.isfinite(v) for v in box) or
                not -epsilon<=box[0]<box[2]<=w+epsilon or not -epsilon<=box[1]<box[3]<=h+epsilon):
                raise ValueError('Invalid evaluated paper box/class')
        matched,missed,extra=match_detections(truth,predicted,threshold)
        located={p[0] for p in matched};groups=scene_groups(record,meta)
        for counts in [total,*[images_by[k][v] for k,v in groups.items()]]:
            counts['images']+=1;counts['tp']+=len(matched);counts['fp']+=len(extra);counts['fn']+=len(missed)
            counts['exact_images']+=not missed and not extra
            counts['count_correct_images']+=len(truth)==len(predicted)
            counts['absolute_count_error']+=abs(len(truth)-len(predicted))
            counts['negative_images']+=not truth
            counts['negative_false_positive_images']+=not truth and bool(extra)
        ratio=imgsz/max(w,h)
        for index,(obj,item) in enumerate(zip(truth,meta['instances'])):
            a,b,c,d=obj['xyxy'];size=min(c-a,d-b)*ratio
            building=item.get('building_id')
            axes={'input_short_side_px':size_bucket(size),'building_id':building if building in ('A','B','C') else 'unknown',
                  'yaw_control':groups['yaw_control'],'branch':groups['branch']}
            for axis,value in axes.items():
                counts=instances_by[axis][value];counts['instances']+=1
                counts['matched']+=index in located;counts['missed']+=index not in located
        details.append({'image':identity,'groups':groups,'missed_gt_indices':missed,'false_prediction_indices':extra,
                        'exact':not missed and not extra,'count_error':len(predicted)-len(truth)})
    if not seen:raise ValueError('No evaluated paper images')
    return {'images':finish_images(total),
        'instance_groups':{axis:{k:finish_instances(v) for k,v in groups.items()} for axis,groups in instances_by.items()},
        'image_groups':{axis:{k:finish_images(v) for k,v in groups.items()} for axis,groups in images_by.items()},
        'per_image':details,'input_size':imgsz,'match_iou':threshold,
        'scope':'Single-class paper detection only. Exact-image success requires no missed or extra boxes; equal counts alone '
                'do not prove correct detection. Camera axes are generation controls/estimates, real unknown views stay unknown. '
                'Shared building assets and prior development exposure limit this to controlled diagnostics; no OCR evaluation.'}


def write_paper_report(evaluation, manifest, *, root):
    root=Path(root).resolve();evaluation=Path(evaluation).resolve();manifest=Path(manifest).resolve()
    if not evaluation.is_relative_to(root) or 'runs' not in evaluation.relative_to(root).parts or not manifest.is_relative_to(root):
        raise ValueError('Paper evaluation and metadata must stay inside the workspace/package')
    before=manifest.read_bytes();summary=json.loads((evaluation/'summary.json').read_text(encoding='utf8'))
    if summary.get('model_key')!='building_name_paper' or summary.get('split') not in ('val','test'):
        raise ValueError('Expected paper Val or Test evaluation')
    rows=[json.loads(line) for line in (evaluation/'predictions/predictions.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
    report=build_paper_report(rows,load_metadata(manifest,root,summary['split']),
                              summary['evaluation_config']['imgsz'],float(summary['diagnostics']['match_iou']))
    expected=sum(r.get('split')==summary['split'] for r in json.loads(before)['records'])
    if report['images']['images']!=expected or summary.get('images')!=expected:
        raise ValueError('Paper report requires the complete declared evaluation split')
    if before!=manifest.read_bytes():raise ValueError('Paper source metadata changed during reporting')
    report.update(split=summary['split'],thresholds=summary['diagnostics'],dataset_manifest=manifest.relative_to(root).as_posix(),
                  dataset_manifest_sha256=hashlib.sha256(before).hexdigest())
    path=evaluation/'paper_report.json'
    with path.open('x',encoding='utf8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    return path
