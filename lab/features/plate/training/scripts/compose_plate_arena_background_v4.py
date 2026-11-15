"""Composite one source-frame car standee onto indexed Blender-rendered backgrounds.

The backgrounds are four views of one procedural Blender scene. The physical
plate is 36 x 24 cm; car-standee dimensions, placement and shadows are fitted.
"""
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard

BG_DIR = ROOT / 'datasets/exports/sign/v4_arena_backgrounds_20261113'
BG_MANIFEST = BG_DIR / 'manifest.json'
BG_IMAGES = [BG_DIR / f'arena_{i:02d}.png' for i in range(4)]
FRAME = ROOT / 'datasets/annotated/site_capture_20261112/sequence_640_v1/full_scene/frame_003927.png'
VIDEO = ROOT / 'datasets/raw/site_capture_20261112/videos/full_scene.mp4'
OUT = ROOT / 'datasets/exports/plate/v4_arena_background_composites_20261113'
SCENE = 'datasets/assets/blender/shared/arena_4m2_pbr_20261112/arena_indoor_detailed.blend'
SEED = 20261113
SPLITS = {'train': 780, 'val': 210}
PROFILES = ('normal', 'dark', 'low_saturation')


def rel(path):
    return path.relative_to(ROOT).as_posix()


def read_image(path):
    result = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if result is None:
        raise ValueError('Cannot decode image: ' + str(path))
    return result


def write_image(path, image):
    ok, output = cv2.imencode('.png', image)
    if not ok:
        raise OSError('PNG encoding failed: ' + str(path))
    output.tofile(str(path))


def ref_for(source_refs, path, preferred):
    entries = source_refs.get(rel(path), [])
    matches = [r for r in entries if r.get('manifest') == preferred]
    if not matches:
        raise ValueError('Expected indexed source evidence missing: ' + rel(path))
    return matches[0]


def render_batch(source_refs):
    if OUT.exists():
        raise ValueError('Destination exists without a reusable production receipt')
    bg_meta = json.loads(BG_MANIFEST.read_text(encoding='utf8'))
    if bg_meta.get('source_scene') != SCENE or len(bg_meta.get('samples', [])) != 4:
        raise ValueError('Background scene metadata differs from expected source')
    backgrounds = [read_image(p) for p in BG_IMAGES]
    frame = read_image(FRAME)
    if frame.shape != (360, 640, 3) or any(im.shape != (480, 640, 3) for im in backgrounds):
        raise ValueError('Unexpected source dimensions')
    # Car silhouette from the identified 175x130 crop; outside pixels are masked.
    crop = frame[195:325, 125:305].copy()
    car_mask = np.zeros(crop.shape[:2], np.uint8)
    outline = np.array([(43,5),(125,5),(145,14),(166,23),(174,40),(174,107),
                        (162,121),(143,125),(135,120),(105,121),(96,127),(76,127),
                        (69,121),(40,121),(29,127),(12,122),(7,111),(8,42),
                        (15,30),(32,23)], np.int32)
    cv2.fillPoly(car_mask, [outline], 255)
    # Reproject the recorded 46x18 photo pixels to the user's physical 3:2 ratio.
    texture = frame[251:269,195:241]
    plate_x, plate_y, plate_w, plate_h = 70, 52, 46, 31
    crop[plate_y:plate_y+plate_h, plate_x:plate_x+plate_w] = cv2.resize(
        texture, (plate_w, plate_h), interpolation=cv2.INTER_LINEAR)
    cv2.rectangle(car_mask, (plate_x, plate_y), (plate_x+plate_w, plate_y+plate_h), 255, -1)
    rng = np.random.default_rng(SEED)
    expected = sum(SPLITS.values())
    records, counts, preview = [], {}, []
    for part in ('train', 'val'):
        for kind in ('images', 'labels'):
            (OUT / kind / part).mkdir(parents=True, exist_ok=False)
    try:
        k = 0
        for split, total in SPLITS.items():
            counts[split] = 0
            for i in range(total):
                bg_index = (i // 3) % 4
                profile = PROFILES[i % 3]
                height = float(rng.uniform(78, 155))
                width = height * 180 / 130
                center = float(rng.uniform(width / 2 + 12, 640 - width / 2 - 12))
                foot = float(rng.uniform(376, 450))
                yaw_proxy = float(rng.uniform(-0.12, 0.12))
                lean = float(rng.uniform(-0.04, 0.04) * width)
                left, right = center - width/2, center + width/2
                dst = np.float32([[left+lean,foot-height*(1+yaw_proxy)],
                                  [right+lean,foot-height*(1-yaw_proxy)],
                                  [right,foot],[left,foot]])
                src = np.float32([[0,0],[180,0],[180,130],[0,130]])
                homography = cv2.getPerspectiveTransform(src, dst)
                shadow = np.zeros((480,640),np.uint8)
                cv2.ellipse(shadow,(round(center),round(foot)),(max(5,round(width*.34)),max(2,round(height*.035))),0,0,360,105,-1)
                shadow = cv2.GaussianBlur(shadow,(0,0),max(1,height*.025))
                bg = backgrounds[bg_index].astype(np.float32) * (1 - shadow[:,:,None].astype(np.float32)/255)
                fg = cv2.warpPerspective(crop,homography,(640,480),flags=cv2.INTER_LINEAR)
                alpha = cv2.warpPerspective(car_mask,homography,(640,480),flags=cv2.INTER_LINEAR).astype(np.float32)/255
                merged = np.clip(fg.astype(np.float32)*alpha[:,:,None] + bg*(1-alpha[:,:,None]),0,255).astype(np.uint8)
                if profile == 'dark':
                    merged = np.clip(merged.astype(np.float32)*.62,0,255).astype(np.uint8)
                elif profile == 'low_saturation':
                    hsv = cv2.cvtColor(merged,cv2.COLOR_BGR2HSV)
                    hsv[:,:,1] = np.clip(hsv[:,:,1].astype(np.float32)*.45,0,255).astype(np.uint8)
                    merged = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
                corners = np.float32([[[plate_x,plate_y],[plate_x+plate_w,plate_y],
                                       [plate_x+plate_w,plate_y+plate_h],[plate_x,plate_y+plate_h]]])
                projected = cv2.perspectiveTransform(corners,homography)[0]
                x1,y1 = projected.min(axis=0); x2,y2 = projected.max(axis=0)
                if not (0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480):
                    raise ValueError('Projected plate box leaves the image')
                name=f'plate_arena_comp_{k:05d}'
                image = OUT/'images'/split/(name+'.png')
                label = OUT/'labels'/split/(name+'.txt')
                write_image(image,merged)
                label.write_text('0 %.8f %.8f %.8f %.8f\n'%((x1+x2)/1280,(y1+y2)/960,(x2-x1)/640,(y2-y1)/480),encoding='utf8')
                bg_path = BG_IMAGES[bg_index]
                rec = {'image':rel(image),'label':rel(label),'split':split,
                    'source_kind':'derived_image','generation_method':'composite',
                    'parent_image':rel(FRAME),'parent_video':rel(VIDEO),'source_time_seconds':131.000533,
                    'source_background':rel(bg_path),'mother_scene':SCENE,'blender_background':True,
                    'background_origin':'blender_render','background_camera':bg_meta['samples'][bg_index],
                    'source_assets':[rel(FRAME),rel(bg_path)],'source_plate_bbox_xyxy':[195,251,241,269],
                    'physical_plate_size_cm':[36,24],'photo_crop_pixels':[46,18],
                    'plate_texture_mapping':'source aspect ratio rescaled to 3:2',
                    'image_processing':{'resize':'perspective projection','compression':'PNG',
                        'blur':'soft simulated ground shadow','tone_profile':profile,
                        'brightness_gain':.62 if profile=='dark' else 1.0,
                        'saturation_gain':.45 if profile=='low_saturation' else 1.0,
                        'homography':homography.tolist(),'illustrative_foot_xy':[center,foot],
                        'perspective_yaw_proxy':yaw_proxy,'plate_quad_xy':projected.tolist()},
                    'scope':'2D projective composite; standee size, location and shadow fitted, not field measurements',
                    'source_evidence':{'frame':ref_for(source_refs,FRAME,'datasets/exports/plate/v4_empty_scene_gpu_20261113/manifest.json'),
                        'background':ref_for(source_refs,bg_path,rel(BG_MANIFEST))},
                    'review_status':'programmatic_projection_candidate'}
                records.append(rec);counts[split]+=1;k+=1
                if i in (0,1,2,total//2,total-1):
                    view=merged.copy();cv2.rectangle(view,(round(x1),round(y1)),(round(x2),round(y2)),(0,0,255),2)
                    preview.append((split,profile,view))
        if counts != SPLITS:
            raise ValueError('Unexpected generated counts')
        for rec in records:
            image=ROOT/rec['image'];label=ROOT/rec['label']
            im=read_image(image); row=label.read_text(encoding='utf8').split()
            if im.shape!=(480,640,3) or len(row)!=5 or row[0]!='0':raise ValueError('Output image or label invalid')
            x,y,w,h=map(float,row[1:])
            if not (0<w<=1 and 0<h<=1 and 0<=x-w/2<=x+w/2<=1 and 0<=y-h/2<=y+h/2<=1):
                raise ValueError('Output label is out of bounds')
        sheet=np.zeros((240*2,320*5,3),np.uint8)
        for j,(split,profile,view) in enumerate(preview):
            tile=cv2.resize(view,(320,240));cv2.putText(tile,split+' '+profile,(4,16),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,255,255),1)
            sheet[(j//5)*240:(j//5+1)*240,(j%5)*320:(j%5+1)*320]=tile
        write_image(OUT/'contact_review.png',sheet)
        manifest={'status':'generated_pending_representative_review','counts':counts,
            'source_scene':SCENE,'background_group_count':1,'background_views':4,
            'generation_method':'2d_composite_on_blender_render','seed':SEED,
            'source_scope':'All background views share one Blender mother scene; all foregrounds share one video frame.',
            'plate_size_cm':[36,24],'physical_standee_dimensions':'unknown; fitted in image plane',
            'programmatic_checks':'all output images decoded; one class 0 YOLO box inside each image',
            'representative_contact':'contact_review.png','source_refs':source_refs,'records':records}
        (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps({'status':manifest['status'],'counts':counts},ensure_ascii=False))
    except BaseException as error:
        (OUT/'FAILED.txt').write_text(type(error).__name__+': '+str(error)+'\n',encoding='utf8')
        raise


def main():
    inputs = [*BG_IMAGES,FRAME,VIDEO]
    params = {'seed':SEED,'counts':SPLITS,'resolution':[640,480],'profiles':PROFILES,
              'plate_cm':[36,24],'foreground_source_box':[125,195,305,325],
              'plate_source_box':[195,251,241,269],'source_scene':SCENE}
    with generation_guard('plate_arena_background_composites_20261113',inputs,[OUT],params,
                          code_paths=[Path(__file__)],procedural_inputs=[BG_MANIFEST],root=ROOT) as run:
        if run.reused:
            print('Reused indexed plate composites')
            return
        render_batch(run.source_refs)


if __name__ == '__main__':
    main()
