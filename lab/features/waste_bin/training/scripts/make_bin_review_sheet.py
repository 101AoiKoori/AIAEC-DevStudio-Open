"""Make a contact sheet of open/closed bin render pairs and candidate boxes."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    manifest=json.loads((args.directory/'manifest.json').read_text(encoding='utf8'))
    groups=defaultdict(list)
    for item in manifest['samples']:
        groups[item['group']].append(item)
    items=[]
    for group in sorted(groups):
        for state in ('closed','open'):
            match=next((s for s in groups[group] if all(x==state for x in s['states'])),None)
            if match:
                items.append(match)
    if not items:
        raise ValueError('no all-open or all-closed frames')
    tile=512; columns=4
    sheet=np.full((math.ceil(len(items)/columns)*tile,columns*tile,3),35,np.uint8)
    for index,sample in enumerate(items):
        image=cv2.imdecode(np.fromfile(str(args.directory/sample['image']),np.uint8),cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('cannot read '+sample['image'])
        height,width=image.shape[:2]
        for item in sample['labels']:
            x1,y1,x2,y2=item['candidate_xyxy_normalized']
            a=(round(x1*width),round(y1*height)); b=(round(x2*width),round(y2*height))
            color=(0,220,255) if item['in_frame'] else (0,0,255)
            cv2.rectangle(image,a,b,color,2)
        image=cv2.resize(image,(tile,tile),interpolation=cv2.INTER_AREA)
        caption=Path(sample['image']).stem+'  '+sample['states'][0]
        cv2.putText(image,caption,(8,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),4,cv2.LINE_AA)
        cv2.putText(image,caption,(8,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2,cv2.LINE_AA)
        row,col=divmod(index,columns)
        sheet[row*tile:(row+1)*tile,col*tile:(col+1)*tile]=image
    ok,encoded=cv2.imencode('.png',sheet)
    if not ok:
        raise RuntimeError('cannot encode review sheet')
    encoded.tofile(str(args.out))
    print(f'Wrote {len(items)} sample views to {args.out}')


if __name__=='__main__':
    main()
