"""Project four known texture landmarks onto already-rendered source images.

SIFT is used only to prepare annotations from the known generated face. The
resulting endpoints inherit the approximate source geometry. Validates against
13 existing Blender-projected source examples before updating any labels.
"""
import argparse
import json
import math
from pathlib import Path
import cv2
import numpy as np


def distance(p,a,b):
    a,b,p=np.asarray(a),np.asarray(b),np.asarray(p)
    v=b-a;den=float(v@v)
    t=np.clip(float((p-a)@v)/den,0,1) if den else 0
    return float(np.linalg.norm(p-(a+t*v)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',required=True,type=Path)
    parser.add_argument('--face',required=True,type=Path)
    parser.add_argument('--geometry',required=True,type=Path)
    args=parser.parse_args()
    manifest_path=args.sources/'manifest.json'
    data=json.loads(manifest_path.read_text(encoding='utf8'))
    if data.get('endpoints_completed'):
        raise RuntimeError('endpoints already completed')
    config=json.loads(args.geometry.read_text(encoding='utf-8-sig'))
    face=cv2.imread(str(args.face),cv2.IMREAD_GRAYSCALE)
    if face is None or face.shape!=(1254,1254):
        raise RuntimeError('expected user-selected 1254px face')
    sift=cv2.SIFT_create(nfeatures=5000)
    reference,descriptors=sift.detectAndCompute(face,None)
    marks=np.float32([config['temperature'][0]['end'],config['temperature'][-1]['end'],
                      config['humidity'][0]['end'],config['humidity'][-1]['end']]).reshape(1,-1,2)
    prepared=[];errors=[];stats=[]
    for sample in data['samples']:
        name=sample['id'];label=args.sources/'labels'/(name+'.txt')
        image=cv2.imread(str(args.sources/'images'/(name+'.png')))
        row=label.read_text(encoding='utf8').split()
        if image is None or len(row)!=29:
            raise RuntimeError('invalid source '+name)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        keys,desc=sift.detectAndCompute(gray,None)
        if desc is None:
            raise RuntimeError('no features '+name)
        pairs=cv2.BFMatcher().knnMatch(descriptors,desc,k=2)
        matches=[m for m,n in pairs if m.distance < .75*n.distance]
        if len(matches)<40:
            raise RuntimeError('too few matches '+name)
        src=np.float32([reference[m.queryIdx].pt for m in matches])
        dst=np.float32([keys[m.trainIdx].pt for m in matches])
        H,inliers=cv2.findHomography(src,dst,cv2.RANSAC,4)
        if H is None or int(inliers.sum())<30:
            raise RuntimeError('unreliable homography '+name)
        fitted=cv2.perspectiveTransform(src.reshape(1,-1,2),H)[0]
        residual=np.linalg.norm(fitted-dst,axis=1)[inliers.ravel().astype(bool)]
        median=float(np.median(residual))
        if median>2:
            raise RuntimeError('large registration residual '+name)
        endpoints=cv2.perspectiveTransform(marks,H)[0]/1024
        if not np.all(np.isfinite(endpoints)) or np.any(endpoints<0) or np.any(endpoints>=1):
            raise RuntimeError('landmark outside image '+name)
        values=[float(v) for v in row]
        if sample['origin']=='animation_frame':
            expected=np.array([[values[5+3*k],values[6+3*k]] for k in range(4,8)])
            err=float(np.max(np.linalg.norm((endpoints-expected)*1024,axis=1)))
            errors.append(err)
            if err>2:
                raise RuntimeError('animation endpoint cross-check failed '+name)
        else:
            needle=[(values[5+3*k],values[6+3*k]) for k in range(4)]
            for j,point in enumerate(endpoints):
                px=min(1023,max(0,int(point[0]*1024)));py=min(1023,max(0,int(point[1]*1024)))
                obscured=bool(image[py,px].min()>=250 or any(distance(point,needle[k],needle[k+1])<.008 for k in (0,2)))
                k=j+4;row[5+3*k:8+3*k]=['%.6f'%point[0],'%.6f'%point[1],'1' if obscured else '2']
            prepared.append((label,row,sample))
        stats.append({'id':name,'matches':len(matches),'inliers':int(inliers.sum()),'median_error_px':median})
    if len(errors)<10:
        raise RuntimeError('not enough independent known-projection checks')
    for label,row,sample in prepared:
        label.write_text(' '.join(row)+'\n',encoding='utf8')
        sample['endpoint_status']='estimated texture endpoints registered by SIFT homography'
    data['endpoints_completed']=True
    data['endpoint_geometry']='four source texture coordinates are approximate; projected with SIFT/RANSAC'
    data['endpoint_validation']={'count':len(errors),'max_error_px':max(errors),'registration':stats}
    manifest_path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('SOURCE_ENDPOINTS_READY',len(prepared),'crosschecked',len(errors),'max_error_px',max(errors))


if __name__=='__main__':
    main()
