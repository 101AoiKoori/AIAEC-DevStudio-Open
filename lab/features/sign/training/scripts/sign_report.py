"""Report sign detection against per-instance metadata already bound to YOLO labels."""
from collections import defaultdict
import json
import math
import hashlib
from pathlib import Path

from engine.reporting import match_detections

VIEWS=('front','oblique','side','back','unknown')


def load_metadata(path,root,split):
    document=json.loads(path.read_text(encoding='utf-8-sig'))
    if document.get('model_key')!='sign':raise ValueError('Expected a sign source manifest')
    metadata=defaultdict(list)
    for row in document['records']:
        if row.get('split')!=split:continue
        aliases=set()
        for key in ('image','source_image','workspace_image'):
            value=row.get(key)
            if isinstance(value,str) and value:
                value=value.replace('\\','/');aliases.add(value)
                relative=(path.parent/value).resolve()
                if relative.is_relative_to(root):aliases.add(relative.relative_to(root).as_posix())
        for alias in aliases:metadata[alias].append(row)
    return metadata


def checked_instances(row,record):
    metadata=record.get('sign',{})
    shape=row.get('shape_hw')
    instances=metadata.get('instances')
    if (metadata.get('schema_version')!=1 or not isinstance(shape,list) or len(shape)!=2 or
        metadata.get('image_size')!=[shape[1],shape[0]] or not isinstance(instances,list) or
        len(instances)!=len(row['ground_truth'])):
        raise ValueError('Sign metadata size or instance count differs from evaluation truth')
    for truth,item in zip(row['ground_truth'],instances):
        box=item.get('xyxy')
        if (truth.get('class_id')!=0 or not isinstance(box,list) or len(box)!=4 or
            any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in box) or
            max(abs(a-b) for a,b in zip(box,truth['xyxy']))>.02 or item.get('view') not in VIEWS):
            raise ValueError('Sign metadata class, box order or view differs from evaluation truth')
    return instances


def instance_counts():return {'instances':0,'matched':0,'missed':0}


def image_counts():
    return {'images':0,'tp':0,'fp':0,'fn':0,'exact_images':0,'negative_images':0,'negative_false_positive_images':0}


def finish_instances(counts):
    return {**counts,'recall':counts['matched']/counts['instances'] if counts['instances'] else None}


def finish_images(counts):
    return {**counts,'precision':counts['tp']/(counts['tp']+counts['fp']) if counts['tp']+counts['fp'] else None,
        'recall':counts['tp']/(counts['tp']+counts['fn']) if counts['tp']+counts['fn'] else None,
        'exact_image_rate':counts['exact_images']/counts['images'] if counts['images'] else None,
        'negative_false_positive_rate':counts['negative_false_positive_images']/counts['negative_images'] if counts['negative_images'] else None}


def build_sign_report(rows,metadata,threshold):
    if not 0<threshold<=1:raise ValueError('Invalid match IoU')
    total=image_counts();instances_total=instance_counts()
    instances_by=defaultdict(lambda:defaultdict(instance_counts))
    images_by=defaultdict(lambda:defaultdict(image_counts))
    for view in VIEWS:instances_by['view'][view]
    seen=set();details=[]
    for row in rows:
        identity=row['source_image'].replace('\\','/')
        if identity in seen:raise ValueError('Duplicate evaluation image')
        seen.add(identity)
        choices=metadata.get(identity,[])
        if len(choices)!=1:raise ValueError('Missing or ambiguous sign metadata: '+identity)
        record=choices[0];items=checked_instances(row,record)
        truths,predictions=row['ground_truth'],row['predictions']
        if any(item.get('class_id')!=0 for item in predictions):raise ValueError('Unexpected sign prediction class')
        matched,missed,extra=match_detections(truths,predictions,threshold)
        located={pair[0] for pair in matched}
        for index,(truth,item) in enumerate(zip(truths,items)):
            box=truth['xyxy'];short_side=min(box[2]-box[0],box[3]-box[1])
            axes={'view':item['view'],'visibility':item.get('visibility','unknown'),
                  'layout':item.get('layout','unknown'),'origin':item.get('origin','unknown'),
                  'short_side_px':'under32' if short_side<32 else 'at_least32'}
            for counts in [instances_total,*[instances_by[axis][value] for axis,value in axes.items()]]:
                counts['instances']+=1;counts['matched']+=index in located;counts['missed']+=index not in located
        scene=record['sign'].get('scene',{})
        groups={key:scene.get(key,'unknown') for key in ('quota_group','tone','background_domain','source_kind','negative_context')}
        groups['source_group']=record.get('source_group') or record.get('source_id') or 'unknown'
        for counts in [total,*[images_by[axis][value] for axis,value in groups.items()]]:
            counts['images']+=1;counts['tp']+=len(matched);counts['fp']+=len(extra);counts['fn']+=len(missed)
            counts['exact_images']+=not missed and not extra
            counts['negative_images']+=not truths
            counts['negative_false_positive_images']+=not truths and bool(extra)
        details.append({'image':identity,'missed_gt_indices':missed,'false_prediction_indices':extra,
                        'exact':not missed and not extra})
    if not rows:raise ValueError('No evaluated sign images')
    return {'images':finish_images(total),'instances':finish_instances(instances_total),
        'instance_groups':{axis:{value:finish_instances(counts) for value,counts in groups.items()} for axis,groups in instances_by.items()},
        'image_groups':{axis:{value:finish_images(counts) for value,counts in groups.items()} for axis,groups in images_by.items()},
        'per_image':details,'match_iou':threshold,
        'scope':'Fixed-threshold sign detection. Instance groups use verified GT metadata; inherited views remain unknown. '
                'False positives are grouped by image context, never assigned an invented view. Shared-source diagnostic; '
                'does not validate route mapping, reliability gates or vehicle behavior.'}


def write_sign_report(evaluation,manifest,*,root):
    root=Path(root).resolve();evaluation=Path(evaluation).resolve();manifest=Path(manifest).resolve()
    if not evaluation.is_relative_to(root) or 'runs' not in evaluation.relative_to(root).parts or not manifest.is_relative_to(root):
        raise ValueError('Evaluation and source manifest must stay inside the workspace or package')
    summary=json.loads((evaluation/'summary.json').read_text(encoding='utf8'))
    if summary.get('model_key')!='sign' or summary.get('split') not in ('val','test'):
        raise ValueError('Expected sign validation or Test evaluation')
    rows=[json.loads(line) for line in (evaluation/'predictions/predictions.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
    report=build_sign_report(rows,load_metadata(manifest,root,summary['split']),float(summary['diagnostics']['match_iou']))
    report.update(split=summary['split'],dataset_manifest=manifest.relative_to(root).as_posix(),
                  thresholds=summary['diagnostics'],dataset_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    path=evaluation/'sign_report.json'
    with path.open('x',encoding='utf8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    return path
