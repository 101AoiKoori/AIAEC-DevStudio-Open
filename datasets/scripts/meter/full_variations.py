"""Coordinate-preserving capture variations for the full meter batch."""
from io import BytesIO
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from datasets.scripts.meter.backgrounds import placement


def random_placement(background, foreground, controls):
    base, support = placement(background, foreground)
    x1,y1,x2,y2 = foreground.getchannel('A').getbbox()
    max_scale=min(2.,(support[2]-support[0]-12)/(x2-x1),(support[3]-16)/(y2-y1))
    scale=min(max_scale,base['scale']*controls['scale_factor'])
    if scale < .25:
        raise ValueError('Foreground cannot fit the reviewed support')
    half=(x2-x1)*scale/2
    left=max(support[0]+half+3,8+half)
    right=min(support[2]-half-3,632-half)
    top=max(support[1]+3,(y2-y1)*scale+8)
    bottom=min(support[3]-3,472)
    if left>right or top>bottom:
        raise ValueError('Placement controls cannot fit inside support')
    center=left+(right-left)*controls['x_fraction']
    foot=top+(bottom-top)*controls['y_fraction']
    return {'scale':scale,'offset_xy':[center-(x1+x2)*scale/2,foot-y2*scale],
            'region_xyxy':[0,0,640,480]},support


def continuous_reading(rng, index, levels, *, continuous_share=.7):
    if rng.random() < continuous_share:
        # Independent uniforms include the interior and near-endpoint regimes.
        return round(rng.uniform(.001,.999),6),round(rng.uniform(.001,.999),6),'continuous'
    pair=index%(len(levels)**2)
    return levels[pair//len(levels)],levels[pair%len(levels)],'grid_anchor'


def degradation_controls(rng, kind):
    return {'kind':kind,'blur_radius':round(rng.uniform(.25,.85),3) if 'blur' in kind else 0.,
            'jpeg_quality':rng.randint(58,88) if 'jpeg' in kind else None,
            'noise_sigma':round(rng.uniform(1.,3.5),3) if kind=='sensor_noise' else 0.,
            'seed':rng.randrange(2**31)}


def degrade(image, controls):
    image=image.convert('RGB')
    if controls['blur_radius']:
        image=image.filter(ImageFilter.GaussianBlur(controls['blur_radius']))
    if controls['noise_sigma']:
        array=np.asarray(image,dtype=float)
        noise=np.random.default_rng(controls['seed']).normal(0,controls['noise_sigma'],array.shape)
        image=Image.fromarray(np.clip(array+noise,0,255).astype('uint8'))
    if controls['jpeg_quality'] is not None:
        buffer=BytesIO()
        image.save(buffer,format='JPEG',quality=controls['jpeg_quality'],subsampling=2)
        buffer.seek(0)
        image=Image.open(buffer).convert('RGB');image.load()
    return image


def local_occluder(image, values, control):
    if control is None:
        return image,values,None
    cx,cy,w,h=values[1:5]
    # A narrow edge-entering obstruction, limited to a small part of the dial.
    x=(cx-w/2+control['x_fraction']*w)*640
    y=(cy-h/2+control['y_fraction']*h)*480
    bw=max(3,w*640*.10);bh=max(4,h*480*.22)
    box=[round(x-bw/2),round(y-bh/2),round(x+bw/2),round(y+bh/2)]
    result=image.copy();ImageDraw.Draw(result).rectangle(box,fill=tuple(control['rgb']))
    labels=list(values);affected=[]
    for i in range(8):
        px,py,v=labels[5+3*i:8+3*i]
        if v and box[0]<=px*640<=box[2] and box[1]<=py*480<=box[3]:
            labels[7+3*i]=1;affected.append(i)
    return result,labels,{'kind':'local_2d_occluder','box_xyxy':box,'affected_points':affected,
                          'scope':'known geometric positions retained; hidden points visibility=1'}


def draw_points(image, values):
    colors=('#ff3434','#ff9728','#1abb52','#24dcdf','#ee39cc','#b258ed','#447aff','#a3b927')
    marked=image.convert('RGB').copy();draw=ImageDraw.Draw(marked)
    if values:
        cx,cy,w,h=values[1:5]
        draw.rectangle(((cx-w/2)*640,(cy-h/2)*480,(cx+w/2)*640,(cy+h/2)*480),outline='orange',width=2)
        for i in range(8):
            x,y,v=values[5+i*3:8+i*3]
            if v:
                x*=640;y*=480;draw.ellipse((x-2,y-2,x+2,y+2),fill=colors[i])
                draw.text((x+4,y),str(i),fill=colors[i])
    return marked
