"""Composite physical Blender shadows and straight-alpha bins over native photographs."""
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.images import read_image, write_image

def linear(rgb):
    return np.where(rgb < .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)

def image8(rgb):
    rgb = np.clip(rgb, 0, 1)
    return np.rint(np.where(rgb < .0031308, rgb * 12.92, 1.055 * rgb ** (1/2.4) - .055) * 255).astype(np.uint8)

def composite(out, directory=None):
    out = Path(out)
    result_dir = Path(directory) if directory else out
    for sub in ('images','shadows'):
        (result_dir / sub).mkdir(parents=True, exist_ok=True)
    manifest = json.loads((out/'render_manifest.json').read_text(encoding='utf-8'))
    formal = manifest.get('target_split') is not None
    if formal:
        (result_dir / 'labels').mkdir(exist_ok=True)
    for record in manifest['records']:
        name = record['id']
        w,h = record['image_size']
        photo = read_image(ROOT/record['source_photo'])
        photo = cv2.resize(photo,(w,h),interpolation=cv2.INTER_AREA)
        foreground = read_image(out/record['foreground'],cv2.IMREAD_UNCHANGED)
        receiver = [read_image(out/record[p]) for p in ('shadowed_receiver','lit_receiver')]
        floor_luma = [linear(a.astype(np.float32)/255).mean(axis=2) for a in receiver]
        strength = 1-np.minimum(floor_luma[0]/np.maximum(floor_luma[1],.006),1)
        strength = np.maximum(0,strength-.022)
        # The ambient floor light is approximate. Moderate the physical attenuation
        # to match soft shadows and ground reflection in the reference capture.
        shadow = (1-strength*.77).astype(np.float32)
        assert foreground.shape == (h,w,4), name
        alpha = foreground[:,:,3:4].astype(np.float32)/255
        assert alpha.max() > .99 and alpha.min()==0, name
        assert strength.max() > .08, name
        distance = cv2.distanceTransform((alpha[:,:,0] <= .01).astype(np.uint8),cv2.DIST_L2,5)
        taper = np.clip((distance-28)/105,0,1)
        smooth_support = 1-taper*taper*(3-2*taper)
        shadow = 1-(1-shadow)*smooth_support
        subject = foreground[:,:,:3].astype(np.float32)/255
        luma = subject @ np.array([.0722,.7152,.2126],np.float32)
        subject = .82*subject + .18*luma[:,:,None]
        result = image8(linear(photo.astype(np.float32)/255)*shadow[:,:,None]*(1-alpha)+
                        .96*linear(subject)*alpha)
        mask = read_image(out/record['visible_mask'],cv2.IMREAD_UNCHANGED)
        mask = cv2.cvtColor(mask,cv2.COLOR_BGRA2RGBA)
        labels = []
        for instance in record['instances']:
            color = manifest['palette'][instance['bin']]
            visible = mask[:,:,3] >= 192
            for i,channel in enumerate(color):
                visible &= mask[:,:,i] >= 192 if channel else mask[:,:,i] <= 63
            y,x=np.nonzero(visible)
            assert len(x)==instance['visible_pixels'], (name,instance['bin'],len(x),instance['visible_pixels'])
            assert instance['bbox_xyxy'] == [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]
            assert np.count_nonzero(alpha[y,x,0] > .5) / len(x) > .98
            if formal:
                x0, y0, x1, y1 = instance['bbox_xyxy']
                class_id = instance['class_id']
                if type(class_id) is not int or not 0 <= class_id < 24:
                    raise ValueError(f'{name}: invalid sticker class')
                labels.append(f'{class_id} {(x0+x1)/(2*w):.8f} {(y0+y1)/(2*h):.8f} '
                              f'{(x1-x0)/w:.8f} {(y1-y0)/h:.8f}')
        image = 'images/' + name + '.png'
        shade = 'shadows/' + name + '.png'
        write_image(result_dir/image,result)
        write_image(result_dir/shade,np.rint(shadow*255).astype(np.uint8))
        record.update(composite=image,shadow_attenuation=shade,
                      shadow_basis='Cycles diffuse receiver with bin shadow rays divided by unoccluded receiver, linearly modulated',
                      shadow_stats={'min_attenuation':float(shadow.min()),'pixels_below_95pct':int((shadow<.95).sum())})
        if formal:
            label = f'labels/{name}.txt'
            (result_dir / label).write_text('\n'.join(labels) + '\n', encoding='utf-8')
            record.update(image=image, label=label)
        print('COMPOSITED',name,record['shadow_stats'],flush=True)
    (result_dir/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    composite(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else None)
