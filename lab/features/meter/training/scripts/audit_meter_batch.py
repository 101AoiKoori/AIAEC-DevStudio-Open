"""Check generated dual-needle candidate data and write a few review overlays."""
import argparse
import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np


def check(root, previews=6):
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf8'))
    samples=manifest['samples']
    count=manifest['count']
    assert len(samples)==count
    assert Counter(s['split'] for s in samples)=={'train':count*8//10,'val':count//10,'test':count//10}
    assert len({s['scene_id'] for s in samples})==count
    assert len({s['id'] for s in samples})==count
    assert len({(s['temperature_fraction'],s['humidity_fraction']) for s in samples})==count
    brightest_fraction=0.0
    for n,s in enumerate(samples):
        image_path=root/'images'/s['split']/(s['id']+'.png')
        label_path=root/'labels'/s['split']/(s['id']+'.txt')
        image=cv2.imread(str(image_path))
        assert image is not None and image.shape[:2] == (manifest['resolution'],manifest['resolution']), image_path
        row=label_path.read_text(encoding='utf8').split()
        assert len(row)==29 and row[0]=='0', label_path
        values=[float(x) for x in row]
        assert all(0 <= x <= 1 for x in values[1:5]) and values[3]>0 and values[4]>0
        x0,y0,x1,y1=s['bbox_xyxy']
        h,w=image.shape[:2]
        mask=np.zeros((h,w),dtype=np.uint8)
        center=(round((x0+x1)*w/2),round((y0+y1)*h/2))
        axes=(max(1,round((x1-x0)*w*.45)),max(1,round((y1-y0)*h*.45)))
        cv2.ellipse(mask,center,axes,0,0,360,1,-1)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        saturated=float(np.count_nonzero((gray>=250)&(mask!=0)))/int(mask.sum())
        brightest_fraction=max(brightest_fraction,saturated)
        assert saturated < .05, ('large saturated region requires review', image_path, saturated)
        assert abs(values[1]-(x0+x1)/2)<.000002 and abs(values[2]-(y0+y1)/2)<.000002
        assert abs(values[3]-(x1-x0))<.000002 and abs(values[4]-(y1-y0))<.000002
        assert 0 <= s['temperature_fraction'] <= 1 and 0 <= s['humidity_fraction'] <= 1
        for k in range(8):
            x,y,v=values[5+3*k:8+3*k]
            assert v in (0,1,2)
            if v:
                assert 0 <= x < 1 and 0 <= y < 1
                assert abs(x-s['projected_candidates'][k][0])<.000002
                assert abs(y-s['projected_candidates'][k][1])<.000002
            else:
                assert (x,y)==(0,0)
            assert int(v)==s['visibility'][k]
        if n < previews:
            review=root/'review'
            review.mkdir(exist_ok=True)
            for k,p in enumerate(s['projected_candidates']):
                xy=(round(p[0]*image.shape[1]),round(p[1]*image.shape[0]))
                color=(0,220,0) if k<4 else (0,180,255)
                cv2.circle(image,xy,5,color,-1)
                cv2.putText(image,str(k),(xy[0]+6,xy[1]-6),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
            assert cv2.imwrite(str(review/(s['id']+'.png')),image)
    for split in ('train','val','test'):
        expected={s['id'] for s in samples if s['split']==split}
        assert {p.stem for p in (root/'images'/split).glob('*.png')}==expected
        assert {p.stem for p in (root/'labels'/split).glob('*.txt')}==expected
    return {'splits':dict(Counter(s['split'] for s in samples)),
            'max_near_white_fraction_in_dial':round(brightest_fraction,6),
            'note':'pixel saturation check does not prove every tick is readable'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    parser.add_argument('--previews',type=int,default=6)
    args=parser.parse_args()
    print('AUDIT_OK',check(args.dataset,args.previews))
