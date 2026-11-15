"""Check e-bike candidate masks and draw a bounded, stratified contact sheet."""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    parser.add_argument('--record-visual-review', action='store_true')
    args = parser.parse_args()
    base=(ROOT/args.dataset).resolve()
    if not base.is_relative_to((ROOT/'datasets/exports/e_bike').resolve()):
        raise ValueError('Expected an e-bike export')
    data=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    if args.record_visual_review:
        record_visual_review(base, data)
        return
    media = [ROOT/record['image'] for record in data['records']]
    media += [ROOT/obj['visible_mask'] for record in data['records'] for obj in record['objects']]
    procedural = [base/'manifest.json'] + [ROOT/record['label'] for record in data['records']]
    target=ROOT/'runs/e_bike'/f'{base.name}_contact.png'
    sys.path.insert(0, str(ROOT))
    from lab.training_common.production_guard import generation_guard
    operation = 'e_bike.check_grounded:' + base.relative_to(ROOT).as_posix()
    with generation_guard(operation, list(dict.fromkeys(media)), [base/'qa.json', target], {},
                          code_paths=[Path(__file__)], procedural_inputs=list(dict.fromkeys(procedural)),
                          root=ROOT) as run:
        if run.reused:
            return
        _check(base, data)


def record_visual_review(base, data):
    """Record the completed representative AI visual review, with its limitations."""
    if base != (ROOT/'datasets/exports/e_bike/v4_vehicle_view_arena_20261113').resolve():
        raise ValueError('visual review applies only to the inspected batch')
    structural = base/'qa.json'
    evidence = ROOT/'runs/e_bike/v4_vehicle_view_arena_20261113_contact.png'
    out = base/'qa_review.json'
    sys.path.insert(0, str(ROOT))
    from lab.training_common.production_guard import generation_guard
    with generation_guard('e_bike.vehicle_view_representative_review', [evidence], [out], {},
                          procedural_inputs=[structural, base/'manifest.json'], code_paths=[Path(__file__)],
                          root=ROOT) as run:
        if run.reused:
            return
        qa = json.loads(structural.read_text(encoding='utf-8'))
        if (qa['status'] != 'structural_passed_visual_sample_pending' or qa['error_count'] != 0
                or qa['images'] != 240 or len(data['records']) != 240 or qa['visual_sample_count'] != 38):
            raise ValueError('structural check does not match the visually inspected sample')
        review = {'status': 'passed', 'structural_error_count': 0, 'images_checked': 240,
                  'representative_visual_review': 'passed', 'visual_sample_count': 38,
                  'reviewer': 'AI representative visual inspection; no per-image human review',
                  'manifest': (base/'manifest.json').relative_to(ROOT).as_posix(),
                  'visual_evidence': evidence.relative_to(ROOT).as_posix(),
                  'limitations': ['Fallen cutouts are small and show compositing edges',
                                  'Ground contacts are fitted, not measured',
                                  'Near-edge truncation is not covered',
                                  'Shared physical source images and Blender scene across splits'],
                  'scope': 'Structural checks for all records and representative contact-sheet review; no model evaluation.'}
        out.write_text(json.dumps(review, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(review, ensure_ascii=False))


def _check(base, data):
    errors,samples,counts,seen=[],{},Counter(),Counter()
    rng=random.Random(20261118)
    for record in data['records']:
        image,label=ROOT/record['image'],ROOT/record['label']
        frame=cv2.imdecode(np.fromfile(str(image),np.uint8),cv2.IMREAD_COLOR)
        if frame is None or frame.shape!=(480,640,3):
            errors.append({'image':record['image'],'error':'decode_or_dimensions'});continue
        lines=label.read_text(encoding='utf-8').splitlines()
        if len(lines)!=len(record['objects']):
            errors.append({'image':record['image'],'error':'label_count'});continue
        for obj,line in zip(record['objects'],lines):
            mask=cv2.imdecode(np.fromfile(str(ROOT/obj['visible_mask']),np.uint8),cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape!=(480,640) or not np.any(mask>127):
                errors.append({'image':record['image'],'error':'mask'});continue
            y,x=np.nonzero(mask>127)
            a,b,c,d=int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1
            values=line.split()
            if len(values)!=5:
                errors.append({'image':record['image'],'error':'label_fields'});continue
            cls,xc,yc,w,h=map(float,values)
            expected=((a+c)/1280,(b+d)/960,(c-a)/640,(d-b)/480)
            if (obj['bbox_xyxy']!=[a,b,c,d] or cls!={'upright':0,'fallen':1}[obj['class']]
                    or any(abs(i-j)>1e-6 for i,j in zip((xc,yc,w,h),expected))):
                errors.append({'image':record['image'],'error':'mask_bbox_mismatch'})
            cv2.rectangle(frame,(a,b),(c,d),(0,255,0) if cls==0 else (0,0,255),2)
            counts['instances_'+obj['class']]+=1
        scenario = record.get('scenario', record.get('plan'))
        if not isinstance(scenario, str) or not scenario:
            errors.append({'image': record['image'], 'error': 'missing_scenario'}); continue
        counts['images_'+scenario]+=1
        background = record.get('background', record.get('source_background'))
        if not isinstance(background, dict):
            errors.append({'image': record['image'], 'error': 'missing_background'}); continue
        background_id = str(background.get('id', background.get('view_group', background.get('frame', 'unknown'))))
        person_overlap = bool(record.get('person_bike_mask_overlap',
                                         record.get('person_vehicle_overlap',
                                                    background.get('person_vehicle_overlap', False))))
        key=(background_id,scenario,person_overlap)
        seen[key]+=1
        if key not in samples or rng.randrange(seen[key]) == 0:
            thumb=cv2.resize(frame,(320,240))
            cv2.putText(thumb,f"{background_id[-4:]} {scenario} "
                        +('person' if key[2] else ''),(2,16),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            samples[key]=thumb
    thumbs=list(samples.values())
    while len(thumbs)%4:
        thumbs.append(np.zeros((240,320,3),np.uint8))
    contact=np.vstack([np.hstack(thumbs[i:i+4]) for i in range(0,len(thumbs),4)])
    target=ROOT/'runs/e_bike'/f'{base.name}_contact.png'
    target.parent.mkdir(parents=True,exist_ok=True)
    cv2.imencode('.png',contact)[1].tofile(str(target))
    report={'status':'structural_passed_visual_sample_pending' if not errors else 'failed',
        'images':len(data['records']),'counts':dict(counts),'error_count':len(errors),'errors':errors[:40],
        'visual_sample_count':len(samples),'visual_evidence':target.relative_to(ROOT).as_posix(),
        'scope':'Masks and image sizes checked for each record; ground geometry and semantics require sample review.'}
    (base/'qa.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__=='__main__':
    main()
