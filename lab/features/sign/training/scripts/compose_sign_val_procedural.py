"""Draw new complete sign-fixture candidates and mix them with truncated val cutouts.

All drawn pixels are procedural approximations. No train/test foreground pixels
are reused. They do not constitute an independent observed physical fixture.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import compose_sign_scene as base


def draw_fixture(rng):
    # Independently draw reference-like prop at 2x resolution, then downsample.
    w, h = 480, 1000
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    # Yellow cylindrical pole with directional gloss.
    cv2.rectangle(mask, (222, 390), (259, 909), 255, -1)
    for x in range(222, 260):
        f = .67 + .33 * max(0, 1 - abs(x - 235) / 28)
        rgb[390:910, x] = np.asarray([22, 194, 242]) * f
    # Green moulded base, three rounded legs and hub; shaded as a single surface.
    bm = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(bm, (240, 934), (195, 30), 0, 0, 360, 255, -1)
    cv2.ellipse(bm, (240, 951), (84, 31), 0, 0, 360, 255, -1)
    cv2.fillConvexPoly(bm, np.array([[216,875],[264,875],[275,905],[315,921],[165,921],[205,905]],np.int32), 255)
    y, x = np.indices((h,w)); shade = np.clip(.50 + .39*np.exp(-((x-210)/100)**2-((y-915)/34)**2) + .09*np.cos((x-240)/65), .35, 1.0)
    for channel, val in enumerate((20, 218, 40)):
        rgb[:,:,channel] = np.where(bm > 0, val*shade, rgb[:,:,channel]).astype(np.uint8)
    mask[bm>0]=255
    # Thin white rim, red ring, white face and reference-like prohibition mark.
    cx, cy, rad = 240, 214, 188
    cv2.circle(mask,(cx,cy),rad+6,255,-1,cv2.LINE_AA)
    cv2.circle(rgb,(cx,cy),rad+6,(205,205,205),-1,cv2.LINE_AA)
    cv2.circle(rgb,(cx,cy),rad,(37,37,190),-1,cv2.LINE_AA)
    cv2.circle(rgb,(cx,cy),151,(230,231,229),-1,cv2.LINE_AA)
    whiteface=np.zeros((h,w),np.uint8); cv2.circle(whiteface,(cx,cy),151,255,-1,cv2.LINE_AA)
    stripe=np.zeros_like(rgb); cv2.line(stripe,(113,85),(367,343),(30,31,192),35,cv2.LINE_AA)
    select=(stripe[:,:,2]>0)&(whiteface>0); rgb[select]=stripe[select]
    arrow=np.array([[240,91],[183,165],[210,165],[210,327],[269,327],[269,165],[297,165]],np.int32)
    cv2.fillPoly(rgb,[arrow],(17,17,17),lineType=cv2.LINE_AA)
    rgba=np.dstack((rgb,mask))
    # Approximate slight perspective narrowing without relabeling the mark.
    factor=rng.uniform(.74,1.0)
    return cv2.resize(rgba,(int(w*factor/2),h//2),interpolation=cv2.INTER_AREA)


def generate(output,count,seed):
    output=output.resolve()
    if output.exists() or not output.is_relative_to(base.EXPORTS.resolve()):
        raise ValueError('output must be new and under sign exports')
    if count<10 or count%10: raise ValueError('count must be a multiple of 10')
    rng=random.Random(seed)
    background=cv2.resize(base.val_frame(),(960,480),interpolation=cv2.INTER_AREA)
    truncated=base.cutout('val_frame15.png')
    (output/'images').mkdir(parents=True);(output/'labels').mkdir()
    scenarios=Counter()
    with (output/'manifest.jsonl').open('w',encoding='utf8') as f:
        for index in range(count):
            kind='negative' if index%5==0 else ('truncated_video' if index%5==1 else 'procedural_full')
            sprite=None if kind=='negative' else truncated if kind=='truncated_video' else draw_fixture(rng)
            result,box=base.make_image(background,sprite,rng,'val' if kind=='truncated_video' else 'train')
            light=('dim','normal','bright')[index%3]
            gain=min(1.14,max(.84,rng.gauss({'dim':.88,'normal':1.,'bright':1.10}[light],.025)))
            result=np.clip(result.astype(np.float32)*gain,0,255).astype(np.uint8)
            stem=f'sign_val_{index+1:06d}'
            ok,encoded=cv2.imencode('.jpg',result,[cv2.IMWRITE_JPEG_QUALITY,92])
            if not ok:raise OSError('image encoding failed')
            encoded.tofile(str(output/'images'/(stem+'.jpg')))
            (output/'labels'/(stem+'.txt')).write_text((base.yolo(box)+'\n') if box else '',encoding='utf8')
            f.write(json.dumps({'id':stem,'split_candidate':'val','kind':kind,'bbox_xyxy':box,
                'background_source':base.VAL_VIDEO.relative_to(base.ROOT).as_posix(),
                'background_seconds':4.5,'foreground_source':('procedural_drawing' if kind=='procedural_full' else 'video_012_at_15s' if kind=='truncated_video' else None),
                'light':light,'gain':gain,'status':'candidate_requires_review'},ensure_ascii=False)+'\n')
            scenarios[kind]+=1
    summary={'status':'complete_candidates','split':'val','count':count,'scenarios':dict(scenarios),
             'limitations':['procedurally drawn fixture is an approximation, not a separately photographed real object','truncated cutout and background come from one video','requires visual/label review'],'seed':seed}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--count',type=int,required=True);parser.add_argument('--seed',type=int,default=6941)
    args=parser.parse_args(); print(json.dumps(generate(base.ROOT/args.output,args.count,args.seed),ensure_ascii=False))
