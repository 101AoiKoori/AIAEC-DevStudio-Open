"""Fire-specific diagnostics using the shared detector's recorded predictions.

No inference or threshold search is performed here. False detections retain
image context; they are not assigned to a building without spatial evidence.
"""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from engine.reporting import match_detections


def load_metadata(path,root,split=None):
    data=json.loads(Path(path).read_text(encoding='utf8'))
    if data.get('model_key')!='fire_sticker':raise ValueError('Expected fire_sticker source index')
    result={}
    for row in data['records']:
        if split is not None and row.get('split')!=split:continue
        for key in ('image','workspace_image'):
            name=row.get(key)
            if not isinstance(name,str) or not name:continue
            aliases={name.replace('\\','/')}
            resolved=(Path(path).parent/name).resolve()
            if resolved.is_relative_to(root):aliases.add(resolved.relative_to(root).as_posix())
            for alias in aliases:
                if alias in result and result[alias] is not row:raise ValueError('Ambiguous fire source identity')
                result[alias]=row
    return result


def size_bucket(value):
    return '0_8' if value<=8 else '8_16' if value<=16 else '16_32' if value<=32 else 'over_32'


def image_counts():
    return dict(images=0,tp=0,fp=0,fn=0,exact_images=0,count_correct_images=0,absolute_count_error=0,
                negative_images=0,negative_false_positive_images=0)


def finish(counts):
    n=counts['images'];neg=counts['negative_images'];tp,fp,fn=counts['tp'],counts['fp'],counts['fn']
    return {**counts,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
        'exact_image_rate':counts['exact_images']/n if n else None,
        'count_correct_rate':counts['count_correct_images']/n if n else None,
        'mean_absolute_count_error':counts['absolute_count_error']/n if n else None,
        'negative_false_positive_rate':counts['negative_false_positive_images']/neg if neg else None}


def build_report(rows,metadata,imgsz,threshold):
    if type(imgsz) is not int or imgsz<=0 or not 0<threshold<=1:raise ValueError('Invalid fire report settings')
    total=image_counts();by_image=defaultdict(lambda:defaultdict(image_counts))
    by_instance=defaultdict(lambda:defaultdict(lambda:{'instances':0,'matched':0}))
    for key in ('0_8','8_16','16_32','over_32'):by_instance['input_short_side_px'][key]
    details=[];seen=set();seen_records=set()
    for row in rows:
        identity=row['source_image'].replace('\\','/')
        if identity in seen or identity not in metadata:raise ValueError('Duplicate or missing fire image metadata')
        seen.add(identity);record=metadata[identity]
        if id(record) in seen_records:raise ValueError('Fire aliases duplicate an evaluated image')
        seen_records.add(id(record))
        truths,predictions=row['ground_truth'],row['predictions'];h,w=row['shape_hw'];items=record.get('instances')
        if h<=0 or w<=0 or not isinstance(items,list) or len(items)!=len(truths):raise ValueError('Fire metadata shape/count differs')
        for item in truths+predictions:
            b=item.get('xyxy')
            if (item.get('class_id')!=0 or not isinstance(b,list) or len(b)!=4 or
                any(type(v) not in (int,float) or not math.isfinite(v) for v in b) or
                not 0<=b[0]<b[2]<=w or not 0<=b[1]<b[3]<=h):raise ValueError('Invalid fire class or box')
        for target,truth in zip(items,truths):
            box=target.get('xyxy')
            if not isinstance(box,list) or len(box)!=4 or max(abs(a-b) for a,b in zip(box,truth['xyxy']))>.02:
                raise ValueError('Fire source box order/coordinates differ from final labels')
        partition=record.get('building_counts') or {}
        if partition and sum(v['visible_count'] for v in partition.values())!=len(truths):
            raise ValueError('Fire building attribution does not partition targets')
        matched,missed,extra=match_detections(truths,predictions,threshold);located={i[0] for i in matched}
        distractor=(record.get('context') or {}).get('distractor') or (record.get('distractor') or {}).get('hard_negative_type') or 'unknown'
        groups={'branch':record.get('branch','unknown'),'source_group':record.get('source_group') or record.get('source_id') or 'unknown',
            'distractor':distractor,'layout':'zero' if not truths else 'single' if len(truths)==1 else 'multi',
            'negative_kind':'hard' if record.get('hard_negative') else 'ordinary_or_unclassified' if not truths else 'positive',
            'view_control':(record.get('camera') or {}).get('view_profile','unknown')}
        for counts in [total,*[by_image[k][v] for k,v in groups.items()]]:
            counts['images']+=1;counts['tp']+=len(matched);counts['fp']+=len(extra);counts['fn']+=len(missed)
            counts['exact_images']+=not missed and not extra;counts['count_correct_images']+=len(truths)==len(predictions)
            counts['absolute_count_error']+=abs(len(truths)-len(predictions));counts['negative_images']+=not truths
            counts['negative_false_positive_images']+=not truths and bool(extra)
        per_building=defaultdict(lambda:{'ground_truth':0,'matched':0,'missed':0})
        for index,(truth,item) in enumerate(zip(truths,items)):
            box=truth['xyxy'];size=min(box[2]-box[0],box[3]-box[1])*imgsz/max(w,h)
            building=item.get('building') if item.get('building') in ('A','B','C') else 'unknown'
            for axis,value in {'input_short_side_px':size_bucket(size),'building':building,'branch':groups['branch']}.items():
                by_instance[axis][value]['instances']+=1;by_instance[axis][value]['matched']+=index in located
            per_building[building]['ground_truth']+=1;per_building[building]['matched']+=index in located
            per_building[building]['missed']+=index not in located
        details.append({'image':identity,'groups':groups,'missed_gt_indices':missed,'false_prediction_indices':extra,
            'per_building_target_matches':dict(per_building),'unassigned_false_predictions':len(extra),
            'exact':not missed and not extra,'count_error':len(predictions)-len(truths)})
    if not seen:raise ValueError('No fire evaluation images')
    instances={axis:{key:{**c,'recall':c['matched']/c['instances'] if c['instances'] else None} for key,c in values.items()}
               for axis,values in by_instance.items()}
    return {'images':finish(total),'image_groups':{a:{k:finish(v) for k,v in g.items()} for a,g in by_image.items()},
        'instance_groups':instances,'per_image':details,'input_size':imgsz,'match_iou':threshold,
        'scope':'Designated fire-sticker detection only. Equal counts do not imply correct detection. '
            'False predictions have image context but no invented building assignment; input sizes are estimates. '
            'Shared assets, reused photo/video parents and development contact limit results to the declared diagnostic scope.'}


def write_report(evaluation,manifest,root):
    root=Path(root).resolve();evaluation=Path(evaluation).resolve();manifest=Path(manifest).resolve()
    if not evaluation.is_relative_to(root) or 'runs' not in evaluation.relative_to(root).parts or not manifest.is_relative_to(root):
        raise ValueError('Fire report inputs must remain in the workspace/package')
    before=manifest.read_bytes();summary=json.loads((evaluation/'summary.json').read_text(encoding='utf8'))
    if summary.get('model_key')!='fire_sticker' or summary.get('split') not in ('val','test'):raise ValueError('Expected Fire Val/Test evaluation')
    rows=[json.loads(line) for line in (evaluation/'predictions/predictions.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
    result=build_report(rows,load_metadata(manifest,root,summary['split']),summary['evaluation_config']['imgsz'],float(summary['diagnostics']['match_iou']))
    expected=sum(r.get('split')==summary['split'] for r in json.loads(before)['records'])
    if result['images']['images']!=expected or summary.get('images')!=expected:raise ValueError('Fire report requires the full declared evaluation split')
    if before!=manifest.read_bytes():raise ValueError('Fire metadata changed during reporting')
    result.update(split=summary['split'],thresholds=summary['diagnostics'],dataset_manifest=manifest.relative_to(root).as_posix(),
                  dataset_manifest_sha256=hashlib.sha256(before).hexdigest())
    path=evaluation/'fire_report.json'
    with path.open('x',encoding='utf8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    return path
