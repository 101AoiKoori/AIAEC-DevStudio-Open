"""Audit grouped OpenCV meter augmentation and write sample overlays."""
import argparse
import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from audit_meter_source_quality import needle_contrast


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',type=Path)
    p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--previews',type=int,default=9)
    args=p.parse_args(); root=args.dataset
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf8'))
    sources=json.loads((args.sources/'manifest.json').read_text(encoding='utf8'))
    src={s['id']:s for s in sources['samples']}
    quality_path=(args.sources/'quality_report.json').resolve()
    quality=json.loads(quality_path.read_text(encoding='utf8'))
    assert manifest.get('source_quality_report')==str(quality_path)
    quality_by_id={s['id']:s for s in quality['samples']}
    assert set(quality_by_id)==set(src)
    animation_splits={s['split'] for s in sources['samples'] if s['origin']=='animation_frame'}
    assert len(animation_splits)<=1, 'One animation trajectory must not cross splits'
    samples=manifest['samples'];n=manifest['count'];size=manifest['resolution']
    assert len(samples)==n
    expected={'train':n*8//10,'val':n//10,'test':n//10}
    assert Counter(s['split'] for s in samples)==expected
    assert len({s['id'] for s in samples})==n
    held={split:set() for split in expected}
    source_cache={}; max_error=0.; max_saturation=0.; visibility=Counter()
    tick_contrasts=[];needle_contrasts=[]
    preview_count=Counter()
    for sample in samples:
        split=sample['split'];sid=sample['source_id'];name=sample['id']
        assert sid in src and src[sid]['split']==split
        assert quality_by_id[sid]['eligible'] and quality_by_id[sid]['split']==split
        held[split].add(sid)
        path=root/'images'/split/(name+'.png');label=root/'labels'/split/(name+'.txt')
        image=cv2.imread(str(path))
        assert image is not None and image.shape[:2]==(size,size),path
        row=[float(v) for v in label.read_text(encoding='utf8').split()]
        assert len(row)==29 and row[0]==0 and all(0<=v<=1 for v in row[1:5])
        assert row[3]>0 and row[4]>0
        if sid not in source_cache:
            source_cache[sid]=[float(x) for x in (args.sources/'labels'/(sid+'.txt')).read_text(encoding='utf8').split()]
        source=source_cache[sid];H=np.asarray(sample['homography'],dtype=np.float64)
        for k in range(8):
            x,y,v=row[5+3*k:8+3*k];visibility[int(v)]+=1
            assert v in (0,1,2) and int(v)==int(source[7+3*k])
            if not v:
                assert x==0 and y==0
                continue
            point=np.array([[[source[5+3*k]*size,source[6+3*k]*size]]],dtype=np.float32)
            expected_point=cv2.perspectiveTransform(point,H)[0,0]/size
            error=max(abs(x-expected_point[0]),abs(y-expected_point[1]))
            max_error=max(max_error,error)
            assert error<.000003 and 0<=x<1 and 0<=y<1
        x0,y0,x1,y1=sample['bbox_xyxy']
        assert abs(row[1]-(x0+x1)/2)<.000002 and abs(row[2]-(y0+y1)/2)<.000002
        mask=np.zeros((size,size),np.uint8)
        cv2.ellipse(mask,(round((x0+x1)*size/2),round((y0+y1)*size/2)),
                    (max(1,round((x1-x0)*size*.45)),max(1,round((y1-y0)*size*.45))),0,0,360,1,-1)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        tick_points=np.asarray(quality_by_id[sid]['projected_tick_samples'],dtype=np.float32).reshape(1,-1,2)
        tick_points=cv2.perspectiveTransform(tick_points,H)[0]
        tick_values=cv2.remap(gray,tick_points[:,0].reshape(-1,1),tick_points[:,1].reshape(-1,1),
                              cv2.INTER_LINEAR).reshape(3,-1).astype(float)
        tick_contrasts.append(float(np.median(np.maximum(tick_values[1],tick_values[2])-tick_values[0])))
        needle_contrasts.append(min(needle_contrast(gray,row,0),needle_contrast(gray,row,2)))
        assert tick_contrasts[-1]>=30 and needle_contrasts[-1]>=30, ('low local contrast',path)
        sat=float(np.count_nonzero((gray>=250)&(mask!=0)))/int(mask.sum())
        max_saturation=max(max_saturation,sat)
        assert sat<.05,('large saturated area',path,sat)
        if preview_count[split] < args.previews//3:
            (root/'review').mkdir(exist_ok=True)
            for k in range(8):
                x,y,v=row[5+3*k:8+3*k]
                if not v:
                    continue
                pos=(round(x*size),round(y*size))
                color=(0,0,255) if v==1 else (0,220,0) if k<4 else (0,180,255)
                cv2.circle(image,pos,5,color,-1)
                cv2.putText(image,str(k),(pos[0]+7,pos[1]-7),cv2.FONT_HERSHEY_SIMPLEX,.7,color,2)
            assert cv2.imwrite(str(root/'review'/(name+'.png')),image)
            preview_count[split]+=1
    assert not (held['train']&held['val'] or held['train']&held['test'] or held['val']&held['test'])
    for split in expected:
        ids={s['id'] for s in samples if s['split']==split}
        assert {p.stem for p in (root/'images'/split).glob('*.png')}==ids
        assert {p.stem for p in (root/'labels'/split).glob('*.txt')}==ids
    report={'splits':expected,'unique_source_images':{k:len(v) for k,v in held.items()},
            'keypoint_visibility':dict(visibility),'max_homography_error_normalized':float(max_error),
            'max_near_white_fraction_in_dial':max_saturation,
            'tick_median_contrast_min':min(tick_contrasts),
            'needle_median_contrast_min':min(needle_contrasts),
            'below_30_tick_contrast':sum(x<30 for x in tick_contrasts),
            'below_30_needle_contrast':sum(x<30 for x in needle_contrasts),
            'note':'source groups are disjoint; all share one meter; local contrast checks are heuristic, not per-tick readability proof'}
    (root/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('AUDIT_OK',json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    main()
