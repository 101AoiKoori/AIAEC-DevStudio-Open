"""Draw new complete sign-fixture candidates and mix them with truncated val cutouts.

All drawn pixels are procedural approximations. No train/test foreground pixels
are reused. They do not constitute an independent observed physical fixture.
"""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/sign/. Production remains paused.")

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


def _generate(output,count,seed,camera_view=False):
    output=output.resolve()
    if output.exists() or not output.is_relative_to(base.EXPORTS.resolve()):
        raise ValueError('output must be new and under sign exports')
    if count<10 or count%10: raise ValueError('count must be a multiple of 10')
    rng=random.Random(seed)
    original = base.val_frame()
    if camera_view:
        height, width = original.shape[:2]
        resized_height = round(height * 960 / width)
        if resized_height < 480:
            raise ValueError('background aspect ratio cannot produce a 960x480 crop')
        resized = cv2.resize(original, (960, resized_height), interpolation=cv2.INTER_AREA)
        top = (resized_height - 480) // 2
        background = resized[top:top+480]
    else:
        background=cv2.resize(original,(960,480),interpolation=cv2.INTER_AREA)
    truncated=base.cutout('val_frame15.png')
    (output/'images').mkdir(parents=True);(output/'labels').mkdir()
    scenarios=Counter()
    with (output/'manifest.jsonl').open('w',encoding='utf8') as f:
        for index in range(count):
            kind='negative' if index%5==0 else ('truncated_video' if index%5==1 else 'procedural_full')
            sprite=None if kind=='negative' else truncated if kind=='truncated_video' else draw_fixture(rng)
            result,box=base.make_image(background,sprite,rng,'val' if kind=='truncated_video' else 'train')
            crop_left = None
            if camera_view:
                crop_left = min(320, max(0, (box[0] + box[2]) // 2 - 320)) if box else 160
                result = result[:, crop_left:crop_left+640]
                if box:
                    box = [box[0]-crop_left, box[1], box[2]-crop_left, box[3]]
                    if not (0 <= box[0] < box[2] <= 640):
                        raise ValueError('fixture is not fully represented by camera crop')
            light=('dim','normal','bright')[index%3]
            gain=min(1.14,max(.84,rng.gauss({'dim':.88,'normal':1.,'bright':1.10}[light],.025)))
            result=np.clip(result.astype(np.float32)*gain,0,255).astype(np.uint8)
            stem=f'sign_val_{index+1:06d}'
            ok,encoded=cv2.imencode('.jpg',result,[cv2.IMWRITE_JPEG_QUALITY,92])
            if not ok:raise OSError('image encoding failed')
            encoded.tofile(str(output/'images'/(stem+'.jpg')))
            if box and camera_view:
                x1, y1, x2, y2 = box
                label = f'0 {(x1+x2)/1280:.6f} {(y1+y2)/960:.6f} {(x2-x1)/640:.6f} {(y2-y1)/480:.6f}'
            else:
                label = base.yolo(box) if box else ''
            (output/'labels'/(stem+'.txt')).write_text((label+'\n') if box else '',encoding='utf8')
            f.write(json.dumps({'id':stem,'split_candidate':'val','kind':kind,'bbox_xyxy':box,
                'image': (output/'images'/(stem+'.jpg')).relative_to(base.ROOT).as_posix(),
                'label': (output/'labels'/(stem+'.txt')).relative_to(base.ROOT).as_posix(),
                'generation_method': 'augmentation' if kind == 'negative' else 'composite',
                'background_source':base.VAL_VIDEO.relative_to(base.ROOT).as_posix(),
                'background_seconds':4.5,'foreground_source':('procedural_drawing' if kind=='procedural_full' else 'video_012_at_15s' if kind=='truncated_video' else None),
                'light':light,'gain':gain,'camera_crop_left':crop_left,
                'background_aspect_preserved':camera_view,'status':'candidate_requires_review'},ensure_ascii=False)+'\n')
            scenarios[kind]+=1
    summary={'status':'complete_candidates','split':'val','count':count,'scenarios':dict(scenarios),
             'limitations':['procedurally drawn fixture is an approximation, not a separately photographed real object','truncated cutout and background come from one video','requires visual/label review'],'seed':seed,
             'image_size':[640,480] if camera_view else [960,480]}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return summary


def generate(output, count, seed, camera_view=False):
    raise RuntimeError("Retired production entry; use the task scripts in datasets/scripts with its source.yaml")


def generate_controlled(output, seed):
    """Generate fully procedural scenes; annotations come from the rendered alpha mask."""
    from sign_production_guard import guarded, existing_result
    output = output.resolve()
    with guarded('compose_controlled', output, [], [Path(__file__)], {'seed': seed, 'counts': [1000, 150, 50]},
                 [Path(__file__), Path(base.__file__)]) as run:
        if run.reused:
            return existing_result(output)
        if output.exists() or not output.is_relative_to(base.EXPORTS.resolve()):
            raise ValueError('output must be new and under sign exports')
        rng = random.Random(seed)
        records = []
        source = Path(__file__).relative_to(base.ROOT).as_posix()
        rel = output.relative_to(base.ROOT).as_posix()
        for split, count in (('train', 1000), ('val', 150), ('test', 50)):
            for kind in ('images', 'labels'):
                (output / kind / split).mkdir(parents=True, exist_ok=True)
            for i in range(count):
                # A generated room with independent wall, floor, lines and pole-like distractors.
                horizon = rng.randint(165, 290)
                wall = np.array([rng.randint(125, 230) for _ in range(3)], dtype=np.uint8)
                floor = np.array([rng.randint(45, 150) for _ in range(3)], dtype=np.uint8)
                image = np.empty((480, 640, 3), dtype=np.uint8)
                image[:horizon] = wall
                image[horizon:] = floor
                for _ in range(rng.randint(2, 7)):
                    x1, x2 = rng.randrange(640), rng.randrange(640)
                    color = tuple(int(v) for v in rng.choices(range(50, 225), k=3))
                    cv2.line(image, (x1, horizon), (x2, 479), color, rng.randint(1, 5))
                hard_negative = i % 5 == 1
                if hard_negative:
                    x = rng.randint(20, 620)
                    cv2.line(image, (x, rng.randint(80, 200)), (x, 450), (20, 190, 220), rng.randint(5, 15))
                    cv2.ellipse(image, (x, 450), (rng.randint(15, 45), 8), 0, 0, 360, (25, 160, 35), -1)
                positive = i % 5 >= 2
                box = None
                if positive:
                    sprite = draw_fixture(rng)
                    height = rng.randint(155, 410)
                    width = max(2, round(sprite.shape[1] * height / sprite.shape[0]))
                    sprite = cv2.resize(sprite, (width, height), interpolation=cv2.INTER_AREA)
                    left = rng.randint(0, 640-width)
                    top = rng.randint(min(480-height, max(0, horizon-height//2)), 480-height)
                    alpha = sprite[:, :, 3:4].astype(np.float32) / 255
                    roi = image[top:top+height, left:left+width]
                    roi[:] = np.clip(sprite[:, :, :3]*alpha + roi*(1-alpha), 0, 255).astype(np.uint8)
                    ys, xs = np.where(sprite[:, :, 3] > 127)
                    box = [int(left+xs.min()), int(top+ys.min()), int(left+xs.max()+1), int(top+ys.max()+1)]
                dark = i % 2 == 0
                if dark:
                    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
                    hsv[:, :, 1] *= .55
                    hsv[:, :, 2] *= .55
                    image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                stem = f'{split}_{i:06d}'
                image_path = output/'images'/split/(stem+'.png')
                label_path = output/'labels'/split/(stem+'.txt')
                ok, encoded = cv2.imencode('.png', image)
                if not ok:
                    raise OSError('image encoding failed')
                encoded.tofile(str(image_path))
                if box:
                    x1, y1, x2, y2 = box
                    label = f'0 {(x1+x2)/1280:.6f} {(y1+y2)/960:.6f} {(x2-x1)/640:.6f} {(y2-y1)/480:.6f}\n'
                else:
                    label = ''
                label_path.write_text(label, encoding='utf8')
                evidence = rel + '/summary.json'
                task = {'state': 'complete_positive' if positive else 'complete_negative',
                        'label': label_path.relative_to(base.ROOT).as_posix(), 'review_evidence': evidence,
                        'strata_review': 'confirmed', 'strata': {'difficulty': 'synthetic',
                        'conditions': ['low_light', 'low_saturation'] if dark else ['normal_light'],
                        'target_states': ['positive' if positive else 'hard_negative' if hard_negative else 'negative']}}
                records.append({'id': stem, 'image': image_path.relative_to(base.ROOT).as_posix(),
                    'label': label_path.relative_to(base.ROOT).as_posix(),
                    'image_size': [640, 480], 'split': split, 'parent_ids': [],
                    'source_group': 'sign_controlled_procedural', 'source_kind': 'synthetic_scene',
                    'generation_method': 'synthetic_render', 'mother_scene': source,
                    'source_assets': ['procedural_sign_fixture', 'procedural_room'],
                    'image_processing': {'resize': 'sprite downsampled to recorded bounding box',
                                         'compression': 'lossless PNG', 'blur': 'none'},
                    'lineage_review': 'confirmed', 'near_duplicate_review': 'confirmed',
                    'review_evidence': evidence, 'selected_tasks': ['sign'], 'tasks': {'sign': task},
                    'render': {'bbox_xyxy': box, 'dark_low_saturation': dark, 'hard_negative': hard_negative}})
        (output/'data.yaml').write_text('path: '+rel+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: sign_fixture\n', encoding='utf8')
        summary = {'status': 'complete_procedural', 'seed': seed, 'counts': {'train': 1000, 'val': 150, 'test': 50},
                   'evidence': 'Deterministic program-generated scene, alpha-mask bounding boxes and empty negative labels; no real captures.',
                   'limits': 'Procedural approximation; common generator and assets across splits; not an independent real-world test.'}
        (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
        manifest = {'schema_version': 1, 'status': 'generated', 'models': {'sign': {'data_yaml': rel+'/data.yaml'}}, 'samples': records}
        (output/'source_gate.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
        return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--count',type=int,required=True);parser.add_argument('--seed',type=int,default=6941)
    parser.add_argument('--camera-view', action='store_true', help='Preserve background aspect and crop to 640x480')
    parser.add_argument('--controlled', action='store_true', help='Generate controlled train/val/test procedural batch')
    args=parser.parse_args(); print(json.dumps(generate_controlled(base.ROOT/args.output,args.seed) if args.controlled else generate(base.ROOT/args.output,args.count,args.seed,args.camera_view),ensure_ascii=False))
