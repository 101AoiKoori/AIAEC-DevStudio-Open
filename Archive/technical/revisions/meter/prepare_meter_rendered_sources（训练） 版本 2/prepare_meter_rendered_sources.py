"""Curate already-rendered meter images as immutable image augmentation sources.

This runs only OpenCV/NumPy/SciPy; it never invokes Blender or saves the model.
"""
import argparse
import json
import math
import shutil
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def projected_outline(sample):
    rotation=Rotation.from_euler('xyz',sample['camera_rotation_euler']).as_matrix()
    location=np.array(sample['camera_location'])
    coords=[]
    for z in (-.06,.15):
        for i in range(128):
            angle=2*math.pi*i/128
            point=np.array([1.05*math.cos(angle),1.05*math.sin(angle),z])
            p=rotation.T@(point-location)
            coords.append((.5+(48/36)*p[0]/-p[2],.5-(48/36)*p[1]/-p[2]))
    coords=np.array(coords)
    return [float(coords[:,0].min()),float(coords[:,1].min()),float(coords[:,0].max()),float(coords[:,1].max())]


def line_distance(p,a,b):
    v=np.array(b[:2])-np.array(a[:2]);w=np.array(p[:2])-np.array(a[:2])
    denom=float(v@v);t=np.clip(float(w@v)/denom,0,1) if denom else 0
    return float(np.linalg.norm(w-t*v))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--animation',required=True,type=Path)
    parser.add_argument('--partial',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('output exists; refusing to overwrite')
    anim=json.loads((args.animation/'manifest.json').read_text(encoding='utf8'))
    if len(anim['samples']) != 17 or anim['render_resolution'] != [1024,1024]:
        raise ValueError('expected 17-frame 1024-square source animation')
    args.output.mkdir(parents=True)
    (args.output/'images').mkdir();(args.output/'labels').mkdir()
    records=[]
    # All frames share one animation trajectory, so none may cross a split.
    chosen={i:'train' for i in list(range(0,9))+[10,11,13,14]}
    for i,split in chosen.items():
        sample=anim['samples'][i]
        source=args.animation/sample['file']
        image=cv2.imread(str(source))
        if image is None or image.shape[:2]!=(1024,1024):
            raise ValueError('invalid animation image '+str(source))
        points=sample['keypoint_candidates_xy_depth']
        bbox=projected_outline(sample)
        row=[0,(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2,bbox[2]-bbox[0],bbox[3]-bbox[1]]
        for k,p in enumerate(points):
            bright=image[min(1023,max(0,int(p[1]*1024))),min(1023,max(0,int(p[0]*1024)))].min()>=250
            cover=any(line_distance(p,points[j],points[j+1])<.008 for j in ((2,) if k in (0,1) else (0,2) if k>=4 else ()))
            v=1 if bright or cover else 2
            row.extend([p[0],p[1],v])
        name='anim_%04d'%sample['frame']
        shutil.copyfile(source,args.output/'images'/(name+'.png'))
        (args.output/'labels'/(name+'.txt')).write_text(' '.join(str(v) if isinstance(v,int) else '%.6f'%v for v in row)+'\n',encoding='utf8')
        records.append({'id':name,'split':split,'origin':'animation_frame','frame':sample['frame'],
                        'source':str(source.resolve()),'endpoint_status':'estimated projected endpoints'})
    for split in ('train','val','test'):
        for label in sorted((args.partial/'labels'/split).glob('*.txt')):
            image=args.partial/'images'/split/(label.stem+'.png')
            tokens=label.read_text(encoding='utf8').split()
            if not image.is_file() or len(tokens)!=29 or any(float(tokens[7+3*k])!=0 for k in range(4,8)):
                raise ValueError('invalid interrupted batch source '+str(label))
            name='render_'+label.stem
            shutil.copyfile(image,args.output/'images'/(name+'.png'))
            shutil.copyfile(label,args.output/'labels'/(name+'.txt'))
            records.append({'id':name,'split':split,'origin':'interrupted_blender_batch',
                            'source':str(image.resolve()),'endpoint_status':'unlabelled'})
    if len(records)<30 or len({r['id'] for r in records})!=len(records):
        raise ValueError('not enough unique rendered sources')
    manifest={'purpose':'Rendered meter bases for OpenCV augmentation; no new Blender rendering',
              'keypoint_order':['temperature_pivot','temperature_tip','humidity_pivot','humidity_tip',
                                'temperature_min','temperature_max','humidity_min','humidity_max'],
              'split_policy':'prior random-render split; all frames from the single animation are train only',
              'same_model_warning':'all images share the same synthetic meter and texture',
              'animation_focal_length_mm':48,'animation_sensor_width_mm':36,
              'samples':records}
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('SOURCE_READY',len(records),{s:sum(r['split']==s for r in records) for s in ('train','val','test')})


if __name__=='__main__':
    main()
