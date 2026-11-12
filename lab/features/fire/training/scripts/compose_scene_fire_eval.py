"""Create two source-reserved real-scene/hand-drawn-paper evaluation candidates.

The source session and renderer-reference overlap are not established, so
outputs are intentionally not declared independent val/test sets.
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fire_blender_labels import yolo_row
from compose_scene_fire import ROOT, EXPORTS, read_image, distribute

BACKGROUND = ROOT / 'datasets/annotated/fire/real_review_candidates_20261110/images'
FONT = ROOT / 'datasets/annotated/fire/blender_buildings_v13/NotoSansSC.ttf'
SOURCES = {
    'scene_track': {'image': '0026.png', 'source_group': 'three_buildings_road_003',
                    'source_video': 'Archive/Assets/场景/三楼与场地视频/三栋楼_垃圾桶与路口_003.mp4',
                    'name_quad': [[336, 61], [492, 62], [493, 94], [337, 94]],
                    'fire_centers': [[340, 175], [435, 220], [353, 290], [475, 340]],
                    'paper_size': [36, 50]},
    'scene_tile': {'image': '0014.png', 'source_group': 'jinxiu_indoor_tile_001',
                   'source_video': 'Archive/Assets/楼宇/锦绣大厦/视频/锦绣楼在室内地砖上的多面绕拍_001.mp4',
                   'name_quad': [[373, 148], [524, 139], [527, 169], [377, 180]],
                   'fire_centers': [[382, 236], [513, 244], [391, 310], [514, 351]],
                   'paper_size': [34, 48]},
}
WORDS = tuple('东南西北春夏秋冬江河湖山新明')


def flame_paper(rng):
    image = np.full((140, 100, 3), rng.randrange(224, 246), np.uint8)
    red = np.array([[49, 11], [57, 38], [71, 28], [68, 55], [88, 49], [84, 92],
                    [75, 117], [54, 130], [31, 127], [14, 112], [10, 91], [20, 100],
                    [22, 70], [32, 51], [34, 36], [43, 48]], np.int32)
    orange = np.array([[50, 49], [59, 69], [72, 63], [74, 90], [70, 114], [53, 125],
                       [33, 122], [25, 105], [29, 86], [37, 77], [40, 66], [44, 76]], np.int32)
    yellow = np.array([[50, 86], [58, 98], [64, 94], [64, 111], [58, 121], [45, 123],
                       [36, 116], [37, 104], [44, 99], [45, 94]], np.int32)
    cv2.fillPoly(image, [red], (10, 20, 210)); cv2.fillPoly(image, [orange], (5, 130, 255)); cv2.fillPoly(image, [yellow], (30, 225, 255))
    return image


def name_paper(text):
    image = Image.new('RGB', (600, 160), (229, 228, 223))
    font = ImageFont.truetype(str(FONT), 98)
    draw = ImageDraw.Draw(image)
    bounds = draw.textbbox((0, 0), text, font=font)
    draw.text(((600 - (bounds[2]-bounds[0]))/2, (160-(bounds[3]-bounds[1]))/2-bounds[1]), text, font=font, fill=(20, 20, 20))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def paste(image, patch, quad):
    height, width = patch.shape[:2]
    source = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
    quad = np.asarray(quad, np.float32)
    matrix = cv2.getPerspectiveTransform(source, quad)
    warped = cv2.warpPerspective(patch, matrix, (image.shape[1], image.shape[0]))
    mask = cv2.warpPerspective(np.full((height, width), 255, np.uint8), matrix, (image.shape[1], image.shape[0]))
    image[mask > 0] = warped[mask > 0]
    ys, xs = np.where(mask > 0)
    return [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1]


def generate(output, count, seed):
    if output.exists() or not output.resolve().is_relative_to(EXPORTS.resolve()):
        raise ValueError('output exists or is outside fire exports')
    if count < 10 or count > 1000:
        raise ValueError('count out of bounds')
    rng = random.Random(seed)
    summary = {'status': 'running', 'count_per_group_and_task': count, 'source_lineage_status': 'unknown; not a final independent split'}
    output.mkdir(parents=True)
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    for group, cfg in SOURCES.items():
        base = cv2.resize(read_image(BACKGROUND/cfg['image']), (960, 540))
        schedule = [k for k, n in enumerate(distribute(count, [20,30,25,25])) for _ in range(n)]
        rng.shuffle(schedule)
        names = set(rng.sample(range(count),round(count*.8)))
        for task in ('fire','name'):
            for folder in ('images','labels'):(output/group/task/folder).mkdir(parents=True)
        with (output/group/'manifest.jsonl').open('w',encoding='utf-8') as f:
            for i, number in enumerate(schedule):
                img = base.copy(); boxes = []
                for x,y in rng.sample(cfg['fire_centers'],number):
                    w,h=cfg['paper_size'];x+=rng.randrange(-4,5);y+=rng.randrange(-4,5)
                    quad=[[x-w/2,y-h/2],[x+w/2,y-h/2],[x+w/2,y+h/2],[x-w/2,y+h/2]]
                    boxes.append(paste(img,flame_paper(rng),quad))
                # Separately derive the name image so negatives never contain paper.
                image_name=img.copy(); name_boxes=[]; text=None
                if i in names:
                    text=('星海' if group == 'scene_track' else '岳峰') + WORDS[(i//len(WORDS))%len(WORDS)] + WORDS[i%len(WORDS)] + '大厦'
                    name_boxes.append(paste(image_name,name_paper(text),cfg['name_quad']))
                gain=min(1.15,max(.8,rng.gauss(1,.07)))
                stem=f'{group}_{i+1:04d}'
                for task, result, annotations in (('fire',img,boxes),('name',image_name,name_boxes)):
                    result=np.clip(result.astype(np.float32)*gain,0,255).astype(np.uint8)
                    ok,data=cv2.imencode('.jpg',result,[cv2.IMWRITE_JPEG_QUALITY,92])
                    if not ok:raise ValueError('encode failed')
                    data.tofile(str(output/group/task/'images'/(stem+'.jpg')))
                    lines=[yolo_row(box,960,540) for box in annotations]
                    (output/group/task/'labels'/(stem+'.txt')).write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8')
                f.write(json.dumps({'id':stem,'source_image':(BACKGROUND / cfg['image']).relative_to(ROOT).as_posix(),
                    'source_video':cfg['source_video'],'source_group':cfg['source_group'],
                    'fire_count':number,'fire_boxes':boxes,'name_boxes':name_boxes,'generated_text':text,
                    'fire_source':'procedural_draw_not_training_icon','gain':gain,
                    'split_candidate':'val_candidate' if group == 'scene_track' else 'test_candidate',
                    'status':'AI-review candidate; source reference and capture independence unknown'},ensure_ascii=False)+'\n')
    summary.update(status='complete_candidates', note='Two scene frames from distinct videos and different physical layouts; synthetic overlays are not independent camera captures. Their overlap with v13 model reference sources and capture sessions is unconfirmed. Do not use as final val/test until audit.', source_groups=list(SOURCES), split_candidate={'scene_track':'val_candidate','scene_tile':'test_candidate'})
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--count',type=int,default=10);parser.add_argument('--seed',type=int,default=2919)
    args=parser.parse_args();print(json.dumps(generate((ROOT/args.output).resolve(),args.count,args.seed),ensure_ascii=False))
