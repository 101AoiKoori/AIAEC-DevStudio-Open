"""Create two source-reserved real-scene/hand-drawn-paper evaluation candidates.

The source session and renderer-reference overlap are not established, so
outputs are intentionally not declared independent val/test sets.
"""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/fire/. Production remains paused.")

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

from fire_blender_labels import yolo_row
from compose_scene_fire import ROOT, EXPORTS, read_image, distribute
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard
from datasets.scripts.common.paper import name_paper as _name_paper, paste

BACKGROUND = ROOT / 'datasets/annotated/fire/real_review_candidates_20261110/images'
FONT = ROOT / 'datasets/assets/blender/fire/blender_buildings_v13/NotoSansSC.ttf'
SOURCES = {
    'scene_track': {'image': '0026.png', 'source_group': 'three_buildings_road_003',
                    'source_video': 'datasets/raw/site_archive_legacy/场景/三楼与场地视频/三栋楼_垃圾桶与路口_003.mp4',
                    'name_quad': [[336, 61], [492, 62], [493, 94], [337, 94]],
                    'fire_centers': [[340, 175], [435, 220], [353, 290], [475, 340]],
                    'paper_size': [36, 50]},
    'scene_tile': {'image': '0014.png', 'source_group': 'jinxiu_indoor_tile_001',
                   'source_video': 'datasets/raw/fire/archive_assets_legacy/楼宇/锦绣大厦/视频/锦绣楼在室内地砖上的多面绕拍_001.mp4',
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
    return _name_paper(text, FONT)


def _generate(output, count, seed):
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


def generate(output, count, seed):
    backgrounds = [BACKGROUND / item['image'] for item in SOURCES.values()]
    videos = [ROOT / item['source_video'] for item in SOURCES.values()]
    with generation_guard('fire.compose_scene_eval:' + str(output.resolve().relative_to(ROOT)),
                          [FONT, *backgrounds, *videos], [output], {'count': count, 'seed': seed, 'sources': SOURCES},
                          code_paths=[Path(__file__), ROOT / 'lab/features/fire/training/scripts/compose_scene_fire.py',
                                      ROOT / 'lab/features/fire/training/scripts/fire_blender_labels.py',
                                      ROOT / 'datasets/scripts/common/images.py',
                                      ROOT / 'datasets/scripts/common/annotations.py',
                                      ROOT / 'datasets/scripts/common/paper.py',
                                      ROOT / 'datasets/scripts/common/perspective.py'],
                          procedural_inputs=[FONT], root=ROOT) as run:
        if run.reused:
            return json.loads((output / 'summary.json').read_text(encoding='utf-8'))
        result = _generate(output, count, seed)
        result['batch_source_refs'] = run.source_refs
        (output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--count',type=int,default=10);parser.add_argument('--seed',type=int,default=2919)
    args=parser.parse_args();print(json.dumps(generate((ROOT/args.output).resolve(),args.count,args.seed),ensure_ascii=False))
