"""Measure rendered-source tick/needle contrast and flag washed-out images.

This is a heuristic screen for the selected synthetic dial, not a guarantee
that each printed tick is readable. Original images and labels are never changed.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from scipy.signal import find_peaks

TICK_MIN=35.0
NEEDLE_MIN=35.0
HUMIDITY_TICK_MIN=25.0


def needle_contrast(gray, row, index):
    height,width=gray.shape
    a=np.array(row[5+3*index:7+3*index])*[width,height]
    b=np.array(row[8+3*index:10+3*index])*[width,height]
    direction=b-a
    length=np.linalg.norm(direction)
    if length<1:
        raise ValueError('degenerate needle')
    normal=np.array([-direction[1],direction[0]])/length
    def pix(point):
        x,y=np.rint(point).astype(int)
        return float(gray[np.clip(y,0,height-1),np.clip(x,0,width-1)])
    differences=[]
    for t in (.55,.65,.75,.85,.93):
        point=a+t*direction
        side=max(pix(point+normal*12),pix(point-normal*12),
                 pix(point+normal*18),pix(point-normal*18))
        differences.append(side-pix(point))
    return float(np.median(differences))


def radial_points(angles, center=(623,646), radius=520):
    return np.column_stack((center[0]+radius*np.cos(np.deg2rad(angles)),
                            center[1]-radius*np.sin(np.deg2rad(angles))))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sources',required=True,type=Path)
    ap.add_argument('--face',required=True,type=Path)
    args=ap.parse_args()
    source_path=(args.sources/'manifest.json').resolve()
    manifest=json.loads(source_path.read_text(encoding='utf8'))
    face=cv2.imread(str(args.face),cv2.IMREAD_GRAYSCALE)
    if face is None or face.shape!=(1254,1254):
        raise ValueError('expected selected 1254px synthetic face')
    angles=np.linspace(-147,-388,963)
    candidates=radial_points(angles).astype(np.float32)
    intensity=cv2.remap(face,candidates[:,0].reshape(-1,1),candidates[:,1].reshape(-1,1),cv2.INTER_LINEAR).ravel()
    peaks,_=find_peaks(-intensity.astype(float),distance=6,prominence=25)
    if not 60<=len(peaks)<=100:
        raise ValueError('unexpected number of texture tick candidates')
    selected=angles[peaks]
    reference_points=np.concatenate([radial_points(selected),radial_points(selected-1.2),radial_points(selected+1.2)]).astype(np.float32).reshape(1,-1,2)
    humid_angles=np.linspace(44,-220,1057)
    humid_candidates=radial_points(humid_angles,(631,948),185).astype(np.float32)
    humid_intensity=cv2.remap(face,humid_candidates[:,0].reshape(-1,1),humid_candidates[:,1].reshape(-1,1),cv2.INTER_LINEAR).ravel()
    humid_peaks,_=find_peaks(-humid_intensity.astype(float),distance=8,prominence=20)
    if not 20<=len(humid_peaks)<=50:
        raise ValueError('unexpected number of humidity texture tick candidates')
    selected_humid=humid_angles[humid_peaks]
    humid_reference=np.concatenate([radial_points(selected_humid,(631,948),185),
                                     radial_points(selected_humid-2,(631,948),185),
                                     radial_points(selected_humid+2,(631,948),185)]).astype(np.float32).reshape(1,-1,2)
    cv2.setRNGSeed(20261106)
    sift=cv2.SIFT_create(nfeatures=5000)
    ref_keys,ref_desc=sift.detectAndCompute(face,None)
    records=[]
    for sample in manifest['samples']:
        name=sample['id']
        image=cv2.imread(str(args.sources/'images'/(name+'.png')),cv2.IMREAD_GRAYSCALE)
        row=[float(x) for x in (args.sources/'labels'/(name+'.txt')).read_text(encoding='utf8').split()]
        if image is None or image.shape!=(1024,1024) or len(row)!=29:
            raise ValueError('invalid source '+name)
        keys,desc=sift.detectAndCompute(image,None)
        if desc is None:
            raise ValueError('missing texture features '+name)
        matches=[a for a,b in cv2.BFMatcher().knnMatch(ref_desc,desc,k=2) if a.distance<.75*b.distance]
        if len(matches)<40:
            raise ValueError('too few texture matches '+name)
        src=np.float32([ref_keys[m.queryIdx].pt for m in matches])
        dst=np.float32([keys[m.trainIdx].pt for m in matches])
        H,inliers=cv2.findHomography(src,dst,cv2.RANSAC,4)
        if H is None or int(inliers.sum())<30:
            raise ValueError('unreliable texture alignment '+name)
        projected=cv2.perspectiveTransform(src.reshape(1,-1,2),H)[0]
        error=np.linalg.norm(projected-dst,axis=1)[inliers.ravel().astype(bool)]
        if float(np.median(error))>2:
            raise ValueError('excessive texture alignment residual '+name)
        points=cv2.perspectiveTransform(reference_points,H)[0]
        values=cv2.remap(image,points[:,0].reshape(-1,1),points[:,1].reshape(-1,1),cv2.INTER_LINEAR).reshape(3,-1).astype(float)
        ticks=np.maximum(values[1],values[2])-values[0]
        tick_median=float(np.median(ticks))
        humid_points=cv2.perspectiveTransform(humid_reference,H)[0]
        humid_values=cv2.remap(image,humid_points[:,0].reshape(-1,1),humid_points[:,1].reshape(-1,1),cv2.INTER_LINEAR).reshape(3,-1).astype(float)
        humid_tick_median=float(np.median(np.maximum(humid_values[1],humid_values[2])-humid_values[0]))
        temp=needle_contrast(image,row,0)
        humid=needle_contrast(image,row,2)
        eligible=(tick_median>=TICK_MIN and humid_tick_median>=HUMIDITY_TICK_MIN
                  and min(temp,humid)>=NEEDLE_MIN)
        records.append({'id':name,'split':sample['split'],'eligible':bool(eligible),
            'tick_median_contrast':tick_median,'humidity_tick_median_contrast':humid_tick_median,
            'temperature_needle_contrast':temp,
            'humidity_needle_contrast':humid,'texture_inliers':int(inliers.sum()),
            'projected_tick_samples':points.tolist(),'projected_humidity_tick_samples':humid_points.tolist()})
    eligible=Counter(x['split'] for x in records if x['eligible'])
    if any(eligible[k]<3 for k in ('train','val','test')):
        raise ValueError('too few eligible source images in a split')
    report={'source_manifest':str(source_path),'face':str(args.face.resolve()),
            'tick_candidates':len(peaks),'humidity_tick_candidates':len(humid_peaks),
            'minimum_tick_median_contrast':TICK_MIN,'minimum_humidity_tick_median_contrast':HUMIDITY_TICK_MIN,
            'minimum_needle_median_contrast':NEEDLE_MIN,'eligible_by_split':dict(eligible),
            'note':'Heuristic based on local grayscale contrast; may exclude readable sources or miss washed-out regions. Not per-tick truth.',
            'samples':records}
    (args.sources/'quality_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('SOURCE_QUALITY_READY',len(records),'eligible',dict(eligible),'rejected',sum(not r['eligible'] for r in records))


if __name__=='__main__':
    main()
