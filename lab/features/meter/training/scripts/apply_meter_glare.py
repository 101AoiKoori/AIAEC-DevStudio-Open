"""Apply only light, bounded synthetic highlights to a candidate meter batch.

At most 15% white is blended into any pixel. No highlight is deliberately
placed over a particular landmark. Run once after finalize_meter_batch.py.
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    parser.add_argument('--seed',type=int,default=20261107)
    args=parser.parse_args()
    args.dataset = args.dataset.resolve()
    root=args.dataset
    manifest_path=root/'manifest.json'
    manifest_bytes=manifest_path.read_bytes()
    manifest=json.loads(manifest_bytes.decode('utf8'))
    images=[root/'images'/sample['split']/(sample['id']+'.png') for sample in manifest['samples']]
    outputs=[manifest_path, *images]
    with generation_guard('meter.apply_glare:' + str(root.resolve()), images, outputs,
                          {'seed': args.seed}, procedural_inputs=[manifest_path],
                          inplace_paths=outputs, code_paths=[Path(__file__)]) as run:
        if manifest_path.read_bytes() != manifest_bytes:
            raise ValueError('dataset manifest changed during input enumeration')
        if run.reused:
            print('REUSED', root)
            return
        _produce(args, manifest)


def _produce(args, manifest):
    root=args.dataset
    manifest_path=root/'manifest.json'
    if manifest.get('glare_applied'):
        raise RuntimeError('highlights already applied; refusing to apply twice')
    if len(manifest['samples']) != manifest['count'] or not manifest['endpoint_status'].startswith('image-specific'):
        raise RuntimeError('complete and finalize the candidate batch first')
    rng=random.Random(args.seed)
    groups={}
    for split in ('train','val','test'):
        members=[s for s in manifest['samples'] if s['split']==split]
        mild_count=round(len(members)*.6)
        levels=['mild']*mild_count+['none']*(len(members)-mild_count)
        rng.shuffle(levels)
        groups[split]=dict(zip((s['id'] for s in members),levels))
    for sample in manifest['samples']:
        level=groups[sample['split']][sample['id']]
        sample['synthetic_glare']={'level':level}
        if level=='none':
            continue
        image_path=root/'images'/sample['split']/(sample['id']+'.png')
        image=cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError('image missing '+sample['id'])
        x0,y0,x1,y1=sample['bbox_xyxy']
        cx=rng.uniform(x0+.15*(x1-x0),x1-.15*(x1-x0))
        cy=rng.uniform(y0+.15*(y1-y0),y1-.15*(y1-y0))
        sx=rng.uniform(.035,.09);sy=rng.uniform(.012,.035)
        angle=rng.uniform(-math.pi,math.pi)
        strength=rng.uniform(.06,.15)
        height,width=image.shape[:2]
        yy,xx=np.ogrid[:height,:width]
        dx=xx/width-cx;dy=yy/height-cy
        u=(dx*math.cos(angle)+dy*math.sin(angle))/sx
        v=(-dx*math.sin(angle)+dy*math.cos(angle))/sy
        alpha=(strength*np.exp(-.5*(u*u+v*v))).astype(np.float32)
        image=np.clip(image.astype(np.float32)*(1-alpha[:,:,None])+255*alpha[:,:,None],0,255).astype(np.uint8)
        if not cv2.imwrite(str(image_path),image):
            raise RuntimeError('failed to write highlight '+str(image_path))
        sample['synthetic_glare'].update({'center':[cx,cy],'sigma':[sx,sy],
                                          'angle_rad':angle,'strength':strength})
    manifest['glare_applied']=True
    manifest['glare_seed']=args.seed
    manifest['glare_distribution']='per split: 40% none, 60% mild (rounded); alpha <= 0.15'
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('MILD_HIGHLIGHTS_APPLIED',len(manifest['samples']),manifest['glare_distribution'])


if __name__=='__main__':
    main()
