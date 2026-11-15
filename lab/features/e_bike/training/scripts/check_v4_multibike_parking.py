"""Validate multi-bike scene labels, original boxes and sampled composition."""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[5]


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('dataset',type=Path);args=ap.parse_args()
    base=(ROOT/args.dataset).resolve()
    if not base.is_relative_to((ROOT/'datasets/exports/e_bike').resolve()):raise ValueError('Expected e-bike export')
    run_check(base)


def run_check(base, *, scale_check=False):
    base = Path(base).resolve()
    if not base.is_relative_to((ROOT/'datasets/exports/e_bike').resolve()):
        raise ValueError('Expected e-bike export')
    manifest_path = base/'manifest.json'
    review_path = ROOT/'datasets/annotated/e_bike/v4_low_angle_multibike_review_20261112/review_manifest.json'
    data = json.loads(manifest_path.read_text(encoding='utf-8'))
    review = json.loads(review_path.read_text(encoding='utf-8'))
    media = [ROOT/record['image'] for record in data['records']]
    media += [ROOT/obj['visible_mask'] for record in data['records'] for obj in record['objects']
              if obj.get('truth') == 'visible_mask_candidate']
    procedural = [manifest_path, review_path] + [ROOT/record['label'] for record in data['records']]
    code = [Path(__file__)]
    if scale_check:
        code += [Path(__file__).with_name('check_v4_scaled_parking.py'),
                 Path(__file__).with_name('compose_v4_scaled_parking.py')]
    target = ROOT/'runs/e_bike'/f'{base.name}_contact.png'
    sys.path.insert(0, str(ROOT))
    from lab.training_common.production_guard import generation_guard
    operation = 'e_bike.check_multibike:' + base.relative_to(ROOT).as_posix()
    with generation_guard(operation, list(dict.fromkeys(media)), [base/'qa.json', target],
                          {'scale_check': scale_check}, code_paths=code,
                          procedural_inputs=list(dict.fromkeys(procedural)), root=ROOT) as run:
        if run.reused:
            return
        _check_geometry(base, data, review)
        if scale_check:
            from check_v4_scaled_parking import _scale_check
            _scale_check(base, data, review)


def _check_geometry(base, data, review):
    originals={x['source_frame']:x['boxes'] for x in review['rows']}
    errors=[];samples={};counts=Counter();fg=set();seen=Counter();rng=random.Random(20261119)
    for record in data['records']:
        image=cv2.imdecode(np.fromfile(str(ROOT/record['image']),np.uint8),1)
        if image is None:
            errors.append([record['image'],'decode']);continue
        height,width=image.shape[:2]
        labels=(ROOT/record['label']).read_text(encoding='utf-8').splitlines()
        if [width,height]!=record['image_size'] or max(width,height)>640 or len(labels)!=len(record['objects']):
            errors.append([record['image'],'size_or_count']);continue
        real=[x for x in record['objects'] if x['truth']=='reviewed_real_parent']
        expected=originals.get(record['background_parent'],[])
        if [(x['class'],x['bbox_xyxy']) for x in real]!=[(x['class'],x['xyxy']) for x in expected]:
            errors.append([record['image'],'parent_annotation'])
        for obj,line in zip(record['objects'],labels):
            a,b,c,d=obj['bbox_xyxy'];cls,x,y,w,h=map(float,line.split())
            if cls!={'upright':0,'fallen':1}[obj['class']] or not (0<=a<c<=width and 0<=b<d<=height):
                errors.append([record['image'],'bounds_or_class']);continue
            calc=((a+c)/(2*width),(b+d)/(2*height),(c-a)/width,(d-b)/height)
            if any(abs(q-v)>1e-6 for q,v in zip((x,y,w,h),calc)):
                errors.append([record['image'],'normalized_coordinates'])
            if obj['truth']=='visible_mask_candidate':
                mask=cv2.imdecode(np.fromfile(str(ROOT/obj['visible_mask']),np.uint8),0)
                if mask is None or mask.shape!=(height,width) or not (mask>127).any():
                    errors.append([record['image'],'mask']);continue
                yy,xx=np.nonzero(mask>127);actual=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1]
                if actual!=obj['bbox_xyxy']:errors.append([record['image'],'mask_bbox'])
                if any(np.any(mask[bb['xyxy'][1]:bb['xyxy'][3],bb['xyxy'][0]:bb['xyxy'][2]]>127) for bb in expected):
                    errors.append([record['image'],'mask_overlaps_original'])
                fg.add(obj['foreground_source']['path'])
            cv2.rectangle(image,(a,b),(c,d),(0,255,0) if cls==0 else (0,0,255),1)
            counts['instances_'+obj['class']]+=1
        key=(record['plan'],record['tone'])
        seen[key]+=1
        if key not in samples or rng.randrange(seen[key])==0:
            canvas=np.zeros((200,320,3),np.uint8);scale=min(320/width,180/height)
            im=cv2.resize(image,(round(width*scale),round(height*scale)));canvas[:im.shape[0],:im.shape[1]]=im
            cv2.putText(canvas,record['plan']+' '+record['tone'],(2,195),cv2.FONT_HERSHEY_SIMPLEX,.34,(255,255,255),1)
            samples[key]=canvas
        counts['plan_'+record['plan']]+=1
    thumbs=list(samples.values())
    while len(thumbs)%3:thumbs.append(np.zeros((200,320,3),np.uint8))
    contact=np.vstack([np.hstack(thumbs[i:i+3]) for i in range(0,len(thumbs),3)])
    target=ROOT/'runs/e_bike'/f'{base.name}_contact.png';target.parent.mkdir(parents=True,exist_ok=True)
    cv2.imencode('.png',contact)[1].tofile(str(target))
    result={'status':'structural_passed_visual_sample_pending' if not errors else 'failed',
        'images':len(data['records']),'counts':dict(counts),'error_count':len(errors),'errors':errors[:50],
        'foreground_sources':sorted(fg),'visual_sample_count':len(samples),'visual_evidence':target.relative_to(ROOT).as_posix()}
    (base/'qa.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
