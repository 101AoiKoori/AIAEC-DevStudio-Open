"""Check generated dual-needle candidate data and write a few review overlays."""
import argparse
import json
from collections import Counter
from pathlib import Path
import cv2


def check(root, previews=6):
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf8'))
    samples=manifest['samples']
    count=manifest['count']
    assert len(samples)==count
    assert Counter(s['split'] for s in samples)=={'train':count*8//10,'val':count//10,'test':count//10}
    assert len({s['scene_id'] for s in samples})==count
    assert len({s['id'] for s in samples})==count
    for n,s in enumerate(samples):
        image_path=root/'images'/s['split']/(s['id']+'.png')
        label_path=root/'labels'/s['split']/(s['id']+'.txt')
        image=cv2.imread(str(image_path))
        assert image is not None and image.shape[:2] == (manifest['resolution'],manifest['resolution']), image_path
        row=label_path.read_text(encoding='utf8').split()
        assert len(row)==29 and row[0]=='0', label_path
        values=[float(x) for x in row]
        assert all(0 <= x <= 1 for x in values[1:5]) and values[3]>0 and values[4]>0
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
    return dict(Counter(s['split'] for s in samples))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    parser.add_argument('--previews',type=int,default=6)
    args=parser.parse_args()
    print('AUDIT_OK',check(args.dataset,args.previews))
