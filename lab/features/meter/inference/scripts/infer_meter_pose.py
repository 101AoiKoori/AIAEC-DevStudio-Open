"""Infer two-needle meter readings and draw equally spaced inferred scale ticks."""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from functools import lru_cache

HERE=Path(__file__).resolve()
PACKAGED=(HERE.parent/'meter_geometry.py').is_file()
ROOT=HERE.parent if PACKAGED else HERE.parents[5]
if PACKAGED:
    from meter_geometry import read_pose_dials, POSE_POINT_NAMES
else:
    sys.path.insert(0,str(ROOT/'ws/src/aec_vision/src'))
    from aec_vision.meter import read_pose_dials, POSE_POINT_NAMES

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def load_config(path):
    cfg=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if cfg.get('schema_version')!=1 or cfg.get('geometry')!='image_plane_equal_angle':
        raise ValueError('Require schema_version 1 and image_plane_equal_angle geometry')
    for name in ('temperature','humidity'):
        scale=cfg['scales'][name]
        if scale['direction'] not in ('clockwise','counterclockwise'):
            raise ValueError('Scale direction must be explicit')
        lo,hi,step=float(scale['minimum']),float(scale['maximum']),float(scale['tick_step'])
        if not all(math.isfinite(v) for v in (lo,hi,step)) or hi<=lo or step<=0:
            raise ValueError('Invalid scale range or tick step')
        if not math.isclose((hi-lo)/step,round((hi-lo)/step),abs_tol=1e-7):
            raise ValueError('Tick step must divide the range')
    return cfg


@lru_cache(maxsize=16)
def font(size):
    spec=importlib.util.find_spec('matplotlib')
    if not spec or not spec.origin:raise RuntimeError('Installed matplotlib font is missing')
    path=Path(spec.origin).parent/'mpl-data/fonts/ttf/DejaVuSans.ttf'
    return ImageFont.truetype(str(path),max(9,int(size)))


def value_text(name,dial):
    label='Temperature' if name=='temperature' else 'Humidity'
    return '{}: {}'.format(label,('{:.1f} {}'.format(dial['value'],dial['unit'])
                                 if dial['value'] is not None else 'unknown'))


def draw_meter(image,reading,*,offset=(0,0),zoom=1,labels=False):
    """Draw into an RGB PIL image; all geometry is in original-image pixels."""
    draw=ImageDraw.Draw(image)
    def xy(point):return ((point[0]-offset[0])*zoom,(point[1]-offset[1])*zoom)
    def dot(point,color,r=3):
        x,y=xy(point);draw.ellipse((x-r,y-r,x+r,y+r),fill=color)
    def text_at(point,text,color):
        x,y=xy(point);draw.text((x+5,y-15),text,font=font(14),fill=color,stroke_width=1,stroke_fill='black')
    for name,dial in reading['dials'].items():
        scale=dial['scale'];color='#ffd52a' if name=='temperature' else '#18e5ed'
        if scale:
            ticks=scale['ticks'];arc=[xy(t['end']) for t in ticks]
            if len(arc)>1:draw.line(arc,fill='#f8da39',width=1)
            for tick in ticks:
                draw.line([xy(tick['start']),xy(tick['end'])],fill='#49ed30',width=2 if labels and tick['major'] else 1)
                if labels and tick['major']:
                    cx,cy=scale['center'];ex,ey=tick['end'];factor=1.13 if name=='temperature' else .76
                    p=(cx+(ex-cx)*factor,cy+(ey-cy)*factor);tx,ty=xy(p)
                    draw.text((tx,ty),'{:g}'.format(tick['value']),font=font(9 if name=='humidity' else 12),fill='white',anchor='mm',stroke_width=1,stroke_fill='black')
            dot(ticks[0]['end'],'#458bff');dot(ticks[-1]['end'],'#ff4242')
            if labels:
                text_at(ticks[0]['end'],'Min','#65a5ff');text_at(ticks[-1]['end'],'Max','#ff6767')
        if dial['center'] is not None:
            dot(dial['center'],color)
        if dial['center'] is not None and dial['needle_tip'] is not None:
            draw.line([xy(dial['center']),xy(dial['needle_tip'])],fill=color,width=2 if labels else 1)
            dot(dial['needle_tip'],color,2)
    return image


def annotate(rgb,readings):
    im=Image.fromarray(rgb).copy();draw=ImageDraw.Draw(im)
    for row in readings:
        draw.rectangle(row['bbox_xyxy'],outline='#ffd52a',width=2)
        draw_meter(im,row)
    text=[]
    if not readings:text=['No meter detected','Temperature: unknown | Humidity: unknown']
    for i,row in enumerate(readings):
        text.extend(['Meter {} | {}'.format(i+1,value_text(name,dial)) for name,dial in row['dials'].items()])
    panel=Image.new('RGBA',im.size,(0,0,0,0));pd=ImageDraw.Draw(panel)
    line_height=27
    pd.rectangle((0,0,im.width,min(im.height,len(text)*line_height+9)),fill=(12,18,24,220))
    for i,t in enumerate(text):pd.text((9,5+i*line_height),t,font=font(19),fill='#ffdc42')
    im=Image.alpha_composite(im.convert('RGBA'),panel).convert('RGB')
    ImageDraw.Draw(im).text((8,im.height-20),'Equal-angle estimate | green: inferred ticks',font=font(12),fill='white',stroke_width=1,stroke_fill='black')
    return im


def make_preview(rgb,readings):
    left=annotate(rgb,readings);left.thumbnail((800,650))
    out=Image.new('RGB',(left.width+570,max(left.height,570)+95),'#151a20')
    out.paste(left,(15,70));d=ImageDraw.Draw(out)
    d.text((15,12),'Meter reading | inferred equal-angle scale',font=font(23),fill='white')
    if readings:
        row=max(readings,key=lambda r:r['confidence']);x1,y1,x2,y2=row['bbox_xyxy']
        pad=max(x2-x1,y2-y1)*.40
        bounds=(max(0,int(x1-pad)),max(0,int(y1-pad)),min(rgb.shape[1],int(x2+pad+1)),min(rgb.shape[0],int(y2+pad+1)))
        crop=Image.fromarray(rgb).crop(bounds);zoom=min(520/crop.width,450/crop.height)
        crop=crop.resize((round(crop.width*zoom),round(crop.height*zoom)),Image.Resampling.NEAREST)
        draw_meter(crop,row,offset=bounds[:2],zoom=zoom,labels=True)
        px=left.width+35;out.paste(crop,(px,70))
        for i,(name,dial) in enumerate(row['dials'].items()):
            d.text((px,535+i*29),value_text(name,dial),font=font(20),fill='#ffdc42')
    else:d.text((left.width+35,100),'No detected dial',font=font(24),fill='#ff7777')
    return out


def infer_frame(model,bgr,cfg,imgsz,device,confidence):
    result=model.predict(bgr,imgsz=imgsz,conf=confidence,iou=cfg['nms_iou'],max_det=50,
                         device=device,verbose=False,save=False)[0]
    rows=[]
    if result.keypoints is None and len(result.boxes):raise ValueError('Checkpoint does not emit Pose keypoints')
    for i,box in enumerate(result.boxes):
        points=result.keypoints.data[i].cpu().tolist()
        if len(points)!=8:raise ValueError('Checkpoint requires kpt_shape [8,3]')
        dials=read_pose_dials(points,cfg['scales'],cfg['point_confidence'])
        rows.append(dict(confidence=float(box.conf.item()),bbox_xyxy=box.xyxy[0].cpu().tolist(),
                         keypoints=[dict(index=j,name=POSE_POINT_NAMES[j],xy=p[:2],confidence=p[2]) for j,p in enumerate(points)],dials=dials))
    return rows,result.speed


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source','--image',dest='source',required=True,type=Path,help='Image or video file')
    parser.add_argument('--config',type=Path,default=ROOT/'config/meter_readout.json' if PACKAGED else ROOT/'lab/features/meter/inference/config/pose_readout.json')
    parser.add_argument('--weights',type=Path)
    parser.add_argument('--output',type=Path,help='New output directory; existing directories are not replaced')
    parser.add_argument('--imgsz',type=int)
    parser.add_argument('--conf',type=float)
    parser.add_argument('--device',default='0',help='CUDA device number, or cpu')
    args=parser.parse_args(argv)
    try:
        source=args.source.resolve()
        if not source.is_file():raise FileNotFoundError(source)
        cfg=load_config(args.config);weights=(args.weights or ROOT/cfg['weights']).resolve()
        if not weights.is_file() or weights.suffix.lower()!='.pt':raise FileNotFoundError(weights)
        size=args.imgsz or cfg['imgsz'];confidence=args.conf if args.conf is not None else cfg['confidence']
        if size<32 or size%32 or not 0<confidence<1:raise ValueError('imgsz must be a positive multiple of 32; conf in (0,1)')
        output=(args.output or ROOT/'outputs'/(source.stem+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
        output.mkdir(parents=True,exist_ok=False)
        cache=output/'cache';cache.mkdir()
        os.environ.update(YOLO_CONFIG_DIR=str(cache),YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',MPLBACKEND='Agg')
        from ultralytics import YOLO
        model=YOLO(str(weights),task='pose')
        if getattr(model.model.model[-1],'kpt_shape',None)!=[8,3]:raise ValueError('Weights must be an eight-point Pose model')
        report=dict(source=str(source),weights=str(weights),weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                    config=cfg,inference=dict(imgsz=size,confidence=confidence,device=args.device),
                    geometry_note='Equal image-plane angles and configured linear ranges; no perspective rectification. Generated green ticks are estimates, not detected print.')
        video=source.suffix.lower() in ('.mp4','.avi','.mov','.mkv','.wmv')
        if not video:
            rgb=np.array(Image.open(source).convert('RGB'));rows,speed=infer_frame(model,rgb[:,:,::-1].copy(),cfg,size,args.device,confidence)
            annotate(rgb,rows).save(output/'annotated.png');make_preview(rgb,rows).save(output/'preview.jpg',quality=95)
            report.update(image_size=list(rgb.shape[1::-1]),detections=rows,speed_ms=speed)
            for i,row in enumerate(rows):
                print('Meter {}: {}'.format(i+1,' | '.join(value_text(n,d) for n,d in row['dials'].items())))
            if not rows:print('No meter detected; temperature and humidity are unknown.')
        else:
            cap=cv2.VideoCapture(str(source));fps=cap.get(cv2.CAP_PROP_FPS);w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if not cap.isOpened() or fps<=0:raise ValueError('Cannot decode video')
            writer=cv2.VideoWriter(str(output/'annotated.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
            if not writer.isOpened():cap.release();raise RuntimeError('Cannot create annotated MP4')
            count=detected=0
            try:
                with (output/'frames.jsonl').open('w',encoding='utf8') as stream:
                    while True:
                        ok,bgr=cap.read()
                        if not ok:break
                        rows,speed=infer_frame(model,bgr,cfg,size,args.device,confidence)
                        overlay=annotate(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),rows)
                        writer.write(np.array(overlay)[:,:,::-1].copy())
                        stream.write(json.dumps(dict(frame=count,timestamp_seconds=count/fps,detections=rows,speed_ms=speed),ensure_ascii=False,allow_nan=False)+'\n')
                        detected+=bool(rows);count+=1
                        if count%50==0:print('Processed {} frames'.format(count),flush=True)
            finally:cap.release();writer.release()
            report.update(frames=count,frames_with_detection=detected,fps=fps)
        (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
        print('OUTPUT:',output,flush=True)
        return 0
    except (OSError,ValueError,RuntimeError,KeyError) as error:
        print('ERROR: {}: {}'.format(type(error).__name__,error),file=sys.stderr)
        return 1


if __name__=='__main__':raise SystemExit(main())
