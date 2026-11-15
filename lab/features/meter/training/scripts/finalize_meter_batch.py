"""Add approximate generated-texture endpoint labels to a completed candidate batch.

The four endpoints come from the image-specific geometry config. Occlusion from
needle overlap and near-white glare is conservatively estimated, not measured.
Run only on a complete batch produced by render_meter_batch.py.
"""
import argparse
import json
import math
import sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard


def distance_to_segment(point, a, b):
    dx,dy=b[0]-a[0],b[1]-a[1]
    length=dx*dx+dy*dy
    t=max(0.,min(1.,((point[0]-a[0])*dx+(point[1]-a[1])*dy)/length)) if length else 0.
    return math.hypot(point[0]-(a[0]+t*dx),point[1]-(a[1]+t*dy))


def _execute(args):
    root=args.dataset
    manifest_path=root/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    if len(manifest['samples']) != manifest['count']:
        raise RuntimeError('incomplete batch')
    for sample in manifest['samples']:
        label_path=root/'labels'/sample['split']/(sample['id']+'.txt')
        image=cv2.imread(str(root/'images'/sample['split']/(sample['id']+'.png')))
        if image is None:
            raise RuntimeError('missing image '+sample['id'])
        row=label_path.read_text(encoding='utf8').split()
        if len(row)!=29:
            raise RuntimeError('invalid label '+sample['id'])
        points=sample['projected_candidates']
        for k in range(4,8):
            x,y,depth=points[k]
            if not (0 <= x < 1 and 0 <= y < 1 and depth > 0):
                row[5+3*k:8+3*k]=['0','0','0']
                sample['visibility'][k]=0
                continue
            px=min(image.shape[1]-1,int(x*image.shape[1]))
            py=min(image.shape[0]-1,int(y*image.shape[0]))
            bright=int(image[py,px].min()) >= 250
            covered=any(distance_to_segment((x,y),points[start],points[start+1]) < .008
                        for start in (0,2))
            visibility=1 if bright or covered else 2
            row[5+3*k:8+3*k]=['%.6f'%x,'%.6f'%y,str(visibility)]
            sample['visibility'][k]=visibility
        label_path.write_text(' '.join(row)+'\n',encoding='utf8')
    manifest['endpoint_status']='image-specific estimated endpoints; coordinates are approximate, not measured tick marks'
    manifest['visibility_status']='approximate: near-white pixels and projected needle-segment overlap; review required'
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('ENDPOINTS_ADDED',len(manifest['samples']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    args=parser.parse_args()
    root=args.dataset.resolve()
    args.dataset=root
    manifest_path=root/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    images=[root/'images'/s['split']/(s['id']+'.png') for s in manifest['samples']]
    labels=[root/'labels'/s['split']/(s['id']+'.txt') for s in manifest['samples']]
    inplace=[manifest_path,*labels]
    with generation_guard('meter.finalize_batch:' + root.relative_to(ROOT).as_posix(), images, inplace, {},
                          code_paths=[Path(__file__)], procedural_inputs=inplace,
                          inplace_paths=inplace, root=ROOT) as run:
        if run.reused:
            print('REUSED', root)
            return
        if json.loads(manifest_path.read_text(encoding='utf8')) != manifest:
            raise RuntimeError('manifest changed before execution')
        _execute(args)


if __name__=='__main__':
    main()
