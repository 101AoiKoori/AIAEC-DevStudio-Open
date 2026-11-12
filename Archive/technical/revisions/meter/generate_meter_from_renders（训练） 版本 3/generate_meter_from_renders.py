"""Make a YOLO Pose candidate dataset from already-rendered 1024px source images.

Each base is assigned to one split before augmentation. Pitch/yaw are simulated
by a planar homography; this does not re-render 3-D glass, needles, or shadows.
"""
import argparse
import json
import math
from pathlib import Path
import cv2
import numpy as np
from audit_meter_source_quality import needle_contrast

SIZE=1024


def camera_transform(rng):
    yaw,pitch,roll=rng.uniform(-14,14),rng.uniform(-14,14),rng.uniform(-8,8)
    a,b,c=map(math.radians,(pitch,yaw,roll))
    rx=np.array([[1,0,0],[0,math.cos(a),-math.sin(a)],[0,math.sin(a),math.cos(a)]])
    ry=np.array([[math.cos(b),0,math.sin(b)],[0,1,0],[-math.sin(b),0,math.cos(b)]])
    rz=np.array([[math.cos(c),-math.sin(c),0],[math.sin(c),math.cos(c),0],[0,0,1]])
    R=rz@ry@rx
    distance=SIZE*1.6
    scale=float(rng.uniform(.90,1.05))
    focal=distance*scale
    dx,dy=float(rng.uniform(-.025,.025)*SIZE),float(rng.uniform(-.025,.025)*SIZE)
    corners=np.array([[0,0],[SIZE-1,0],[SIZE-1,SIZE-1],[0,SIZE-1]],np.float32)
    plane=np.column_stack((corners[:,0]-SIZE/2,corners[:,1]-SIZE/2,np.zeros(4)))
    camera=plane@R.T+np.array([dx,dy,distance])
    destination=np.column_stack((focal*camera[:,0]/camera[:,2]+SIZE/2,
                                 focal*camera[:,1]/camera[:,2]+SIZE/2)).astype(np.float32)
    H=cv2.getPerspectiveTransform(corners,destination)
    return H,{'yaw_deg':float(yaw),'pitch_deg':float(pitch),'roll_deg':float(roll),
              'scale':scale,'translation_px':[dx,dy]}


def transform(points,H):
    return cv2.perspectiveTransform(np.array(points,np.float32).reshape(1,-1,2),H)[0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--count',type=int,default=1000)
    p.add_argument('--seed',type=int,default=20261106)
    p.add_argument('--overwrite-existing',action='store_true',
                   help='Regenerate this exact existing dataset after verifying its manifest')
    a=p.parse_args()
    if a.count<10 or a.count%10:
        p.error('count must be >= 10 and divisible by 10')
    if a.output.exists():
        if not a.overwrite_existing:
            p.error('output exists; refusing to overwrite')
        try:
            previous=json.loads((a.output/'manifest.json').read_text(encoding='utf8'))
        except (OSError,ValueError) as exc:
            p.error('cannot verify existing dataset: '+str(exc))
        if (previous.get('count')!=a.count or
                previous.get('source_manifest')!=str((a.sources/'manifest.json').resolve()) or
                previous.get('purpose')!='Rendered image OpenCV augmentation candidate; not independent real-world validation'):
            p.error('existing dataset manifest does not match this operation')
        for split,number in (('train',a.count*8//10),('val',a.count//10),('test',a.count//10)):
            expected={'%s_%05d'%(split,i) for i in range(number)}
            for directory,extension in (('images','*.png'),('labels','*.txt')):
                existing={x.stem for x in (a.output/directory/split).glob(extension)}
                if existing!=expected:
                    p.error('existing dataset file names do not match expected split '+split)
    elif a.overwrite_existing:
        p.error('--overwrite-existing requires an existing generated dataset')
    source_manifest=json.loads((a.sources/'manifest.json').read_text(encoding='utf8'))
    quality_path=(a.sources/'quality_report.json').resolve()
    if not quality_path.is_file():
        p.error('source contrast quality report missing; run audit_meter_source_quality.py')
    quality=json.loads(quality_path.read_text(encoding='utf8'))
    if quality.get('source_manifest')!=str((a.sources/'manifest.json').resolve()):
        p.error('source quality report points to a different source manifest')
    quality_by_id={r['id']:r for r in quality['samples']}
    if len(quality_by_id)!=len(source_manifest['samples']) or set(quality_by_id)!={r['id'] for r in source_manifest['samples']}:
        p.error('source quality report does not cover the source manifest')
    if any(quality_by_id[r['id']]['split']!=r['split'] for r in source_manifest['samples']):
        p.error('source quality split does not match source manifest')
    grouped={split:[r for r in source_manifest['samples'] if r['split']==split and quality_by_id[r['id']]['eligible']]
             for split in ('train','val','test')}
    if any(not x for x in grouped.values()):
        p.error('each split needs source images')
    if len({x['id'] for x in source_manifest['samples']})!=len(source_manifest['samples']):
        p.error('source identifiers must be unique')
    a.output.mkdir(parents=True,exist_ok=a.overwrite_existing)
    if a.overwrite_existing:
        for name in ('audit.json','audit_ultralytics.json'):
            report=a.output/name
            if report.exists():
                report.write_text(json.dumps({'status':'stale','reason':'dataset regeneration in progress'})+'\n',encoding='utf8')
    for split in grouped:
        (a.output/'images'/split).mkdir(parents=True,exist_ok=a.overwrite_existing)
        (a.output/'labels'/split).mkdir(parents=True,exist_ok=a.overwrite_existing)
    rng=np.random.default_rng(a.seed)
    quota={'train':a.count*8//10,'val':a.count//10,'test':a.count//10}
    manifest={'purpose':'Rendered image OpenCV augmentation candidate; not independent real-world validation',
              'source_manifest':str((a.sources/'manifest.json').resolve()),'seed':a.seed,
              'source_quality_report':str(quality_path),
              'source_screening':'local tick and needle contrast heuristic; review required',
              'minimum_augmented_tick_and_needle_contrast':30,
              'keypoint_order':source_manifest['keypoint_order'],'count':a.count,'split_ratio':[8,1,1],
              'resolution':SIZE,'view_model':'planar homography; not a physical 3-D re-render',
              'light_policy':'existing rendered light; small gain variation; no synthetic strong glare',
              'split_policy':'split original rendered image before augmentation; one source never crosses splits',
              'same_model_warning':'all source images share one synthetic meter and texture', 'samples':[]}
    for split in ('train','val','test'):
        sources=grouped[split]
        order=[]
        while len(order)<quota[split]:
            order.extend(rng.permutation(len(sources)).tolist())
        for i,source_i in enumerate(order[:quota[split]]):
            source=sources[source_i];sid=source['id']
            img=cv2.imread(str(a.sources/'images'/(sid+'.png')))
            values=[float(x) for x in (a.sources/'labels'/(sid+'.txt')).read_text(encoding='utf8').split()]
            if img is None or img.shape[:2]!=(SIZE,SIZE) or len(values)!=29:
                raise RuntimeError('invalid source '+sid)
            cx,cy,w,h=values[1:5]
            bbox=np.array([[(cx-w/2)*SIZE,(cy-h/2)*SIZE],[(cx+w/2)*SIZE,(cy-h/2)*SIZE],
                           [(cx+w/2)*SIZE,(cy+h/2)*SIZE],[(cx-w/2)*SIZE,(cy+h/2)*SIZE]])
            source_tick_points=np.asarray(quality_by_id[sid]['projected_tick_samples'],dtype=np.float32).reshape(1,-1,2)
            for attempt in range(50):
                H,parameters=camera_transform(rng)
                quad=transform(bbox,H)
                x0,y0=quad.min(axis=0);x1,y1=quad.max(axis=0)
                if min(x0,y0,SIZE-1-x1,SIZE-1-y1) < SIZE*.015:
                    continue
                image=cv2.warpPerspective(img,H,(SIZE,SIZE),flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT,borderValue=(200,200,200))
                gain=float(rng.uniform(.94,1.02))
                image=cv2.convertScaleAbs(image,alpha=gain)
                output=[0,float((x0+x1)/(2*SIZE)),float((y0+y1)/(2*SIZE)),float((x1-x0)/SIZE),float((y1-y0)/SIZE)]
                projected=[];visibility=[]
                for k in range(8):
                    x,y,v=values[5+3*k:8+3*k];v=int(v)
                    if v:
                        pt=transform([[x*SIZE,y*SIZE]],H)[0]
                        if not (0<=pt[0]<SIZE and 0<=pt[1]<SIZE):
                            raise RuntimeError('source keypoint outside frame')
                        point=[float(pt[0]/SIZE),float(pt[1]/SIZE)]
                        output.extend(point+[v]);projected.append(point);visibility.append(v)
                    else:
                        output.extend([0.,0.,0]);projected.append([0.,0.]);visibility.append(0)
                gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                tick_points=cv2.perspectiveTransform(source_tick_points,H)[0]
                tick_values=cv2.remap(gray,tick_points[:,0].reshape(-1,1),tick_points[:,1].reshape(-1,1),
                                      cv2.INTER_LINEAR).reshape(3,-1).astype(float)
                tick_quality=float(np.median(np.maximum(tick_values[1],tick_values[2])-tick_values[0]))
                rounded_output=[v if isinstance(v,int) else float('%.6f'%v) for v in output]
                needle_quality=min(needle_contrast(gray,rounded_output,0),needle_contrast(gray,rounded_output,2))
                if tick_quality >= 30 and needle_quality >= 30:
                    break
            else:
                raise RuntimeError('could not keep source in frame with sufficient local contrast '+sid)
            name='%s_%05d'%(split,i)
            if not cv2.imwrite(str(a.output/'images'/split/(name+'.png')),image,[cv2.IMWRITE_PNG_COMPRESSION,1]):
                raise RuntimeError('failed image '+name)
            (a.output/'labels'/split/(name+'.txt')).write_text(' '.join(str(v) if isinstance(v,int) else '%.6f'%v for v in output)+'\n',encoding='utf8')
            manifest['samples'].append({'id':name,'split':split,'source_id':sid,'source_origin':source['origin'],
                'parameters':parameters,'homography':H.tolist(),'brightness_gain':gain,
                'bbox_xyxy':[float(x0/SIZE),float(y0/SIZE),float(x1/SIZE),float(y1/SIZE)],
                'keypoints':projected,'visibility':visibility,
                'tick_median_contrast':tick_quality,'needle_median_contrast':needle_quality})
            if (i+1)%100==0 or i+1==quota[split]:
                print('AUGMENT_PROGRESS',split,i+1,'/',quota[split],flush=True)
    (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (a.output/'data.yaml').write_text('path: '+json.dumps(a.output.resolve().as_posix())+
         '\ntrain: images/train\nval: images/val\ntest: images/test\nkpt_shape: [8, 3]\nnames:\n  0: meter\n',encoding='utf8')
    print('AUGMENT_COMPLETE',a.output,flush=True)


if __name__=='__main__':
    main()
