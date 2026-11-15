"""Audit generated facade paper labels and source groups; never run model tests."""
import json
import sys
from collections import Counter,defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from lab.training_common.production_guard import generation_guard
DATA=ROOT/'datasets/exports/fire/v4_tracked_facade_candidates_20261112'


def _run(source_refs):
    data=json.loads((DATA/'manifest.json').read_text(encoding='utf8'))
    review=json.loads((ROOT/data['records'][0]['existing_targets_review']).read_text(encoding='utf8'))
    backgrounds={row['image']:row for row in review['clips']};errors=[];parents=defaultdict(set);videos=defaultdict(set);counts=Counter();examples={}
    for row in data['records']:
        path=ROOT/row['image'];image=cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_COLOR)
        if image is None or image.shape!=(360,640,3):errors.append(row['image']+': invalid image');continue
        if row['background_parent'] not in backgrounds:errors.append(row['image']+': no background review')
        parents[row['split']].add(row['background_parent']);videos[row['split']].add(row['background_video'])
        if row['scenario']=='negative' and row['tone']=='normal':
            original=cv2.imdecode(np.fromfile(str(ROOT/row['background_parent']),np.uint8),cv2.IMREAD_COLOR)
            if not np.array_equal(image,original):errors.append(row['image']+': unchanged negative differs from source')
        polygon=np.asarray(row['facade_quad_xy'],np.float32)
        if polygon.shape!=(4,2) or not cv2.isContourConvex(polygon):errors.append(row['image']+': invalid facade')
        for task,kind in (('fire','fire_sticker'),('name','name_sign')):
            objects=[o for o in row['objects'] if o['kind']==kind];label=row['labels'][task]
            if task in row['partial_tasks']:
                if label is not None:errors.append(row['image']+': partial truth emitted')
                continue
            if label is None:errors.append(row['image']+': complete truth missing');continue
            lines=(ROOT/label).read_text(encoding='utf8').splitlines()
            if len(lines)!=len(objects):errors.append(row['image']+': label count mismatch');continue
            for line,obj in zip(lines,objects):
                a,b,c,d=obj['bbox_xyxy'];p=np.asarray(obj['paper_quad_xy'],np.float32)
                if p.shape!=(4,2) or any(cv2.pointPolygonTest(polygon,tuple(map(float,v)),False)<0 for v in p):errors.append(row['image']+': paper outside facade')
                mask=np.zeros((360,640),np.uint8);cv2.fillConvexPoly(mask,np.rint(p).astype(np.int32),255);ys,xs=np.nonzero(mask)
                raster=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1] if len(xs) else None
                if raster is None or max(abs(i-j) for i,j in zip((a,b,c,d),raster))>2:errors.append(row['image']+': paper raster box mismatch')
                cls,x,y,w,h=map(float,line.split());expected=((a+c)/1280,(b+d)/720,(c-a)/640,(d-b)/360)
                if cls!=0 or any(abs(i-j)>1e-5 for i,j in zip((x,y,w,h),expected)) or not (0<=a<c<=640 and 0<=b<d<=360):errors.append(row['image']+': invalid label')
                if task=='fire' and (c-a<14 or d-b<20):errors.append(row['image']+': tiny fire labeled')
                cv2.rectangle(image,(a,b),(c,d),(0,0,255) if task=='fire' else (0,255,0),1)
        counts[(row['split'],row['scenario'],row['tone'])]+=1
        key=(row['split'],row['scenario'])
        if row['tone']=='normal' and key not in examples:
            im=cv2.resize(image,(320,180));cv2.putText(im,' '.join(key),(3,15),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,0,255),1);examples[key]=im
    cross={'parent_frame_overlap':sorted(parents['val']&parents['test']),'source_video_overlap':sorted(videos['val']&videos['test'])}
    if cross['parent_frame_overlap'] or cross['source_video_overlap']:errors.append('source frame or video crosses development and synthetic stress sets')
    ims=list(examples.values())
    while len(ims)%2:ims.append(np.zeros_like(ims[0]))
    target=ROOT/'runs/fire_sticker/v4_tracked_facade_stratified.png';target.parent.mkdir(parents=True,exist_ok=True)
    cv2.imencode('.png',np.vstack([np.hstack(ims[i:i+2]) for i in range(0,len(ims),2)]))[1].tofile(str(target))
    out={'status':'structural_pass_visual_review_pending' if not errors else 'failed','images':len(data['records']),
         'errors':errors[:100],'total_errors':len(errors),'counts':{'|'.join(k):v for k,v in counts.items()},
         'parent_frames':{k:len(v) for k,v in parents.items()},'source_videos':{k:len(v) for k,v in videos.items()},
         'cross_split':cross,'reviewed_existing_targets':'AI candidate only, not human-accepted',
         'visual_sample_count':len(examples),'visual_evidence':target.relative_to(ROOT).as_posix(),
         'scope':'Raster/label/source checks; source videos share physical building props and site. Synthetic paper composites are not independent real-camera captures.',
         'indexed_source_references':source_refs}
    (DATA/'qa.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in out.items() if k!='counts'},ensure_ascii=False))
    if errors:raise SystemExit(1)


def main():
    manifest_path=DATA/'manifest.json'
    data=json.loads(manifest_path.read_text(encoding='utf8'))
    if not data['records']:
        raise ValueError('No candidate records')
    review=ROOT/data['records'][0]['existing_targets_review']
    media=sorted({ROOT/row['image'] for row in data['records']} |
                 {ROOT/row['background_parent'] for row in data['records']} |
                 {ROOT/row['background_video'] for row in data['records']} |
                 {ROOT/row['shared_artwork'] for row in data['records'] if row.get('shared_artwork')})
    labels=sorted({ROOT/path for row in data['records'] for path in row['labels'].values() if path})
    target=ROOT/'runs/fire_sticker/v4_tracked_facade_stratified.png'
    with generation_guard('fire.check_v4_tracked_facade_papers',media,[target,DATA/'qa.json'],{},
                          code_paths=[Path(__file__)],procedural_inputs=[manifest_path,review,*labels],root=ROOT) as run:
        if run.reused:
            print('已复用已有检查结果');return
        _run(run.source_refs)


if __name__=='__main__':main()
