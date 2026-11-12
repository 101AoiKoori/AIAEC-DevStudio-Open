"""Replace generated name text on existing training candidate paper boxes."""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from compose_scene_fire import ROOT, EXPORTS, read_image
from compose_scene_fire_eval import name_paper, paste

CHARS = tuple('清和安乐明新宏瑞锦泽云景晨华康福泰阳祥雅宁兴嘉文远广德元中天湖海东南西北春夏秋冬江河山')
CHARS = tuple(dict.fromkeys(CHARS))
SUFFIX = ('大厦', '大楼', '中心', '公寓', '商厦')


def make_name(index):
    n = len(CHARS)
    return CHARS[(index // n) % n] + CHARS[index % n] + SUFFIX[(index // (n*n)) % len(SUFFIX)]


def write_paper(image, box, text):
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 640):
        raise ValueError('invalid paper box')
    region = image[y1:y2, x1:x2]
    surface = np.percentile(region.reshape(-1, 3), 85, axis=0)
    surface = np.clip(surface, 100, 235)
    patch = name_paper(text).astype(np.float32)
    patch = np.clip(patch * (surface / 229.0), 0, 255).astype(np.uint8)
    paste(image, patch, [[x1, y1], [x2-1, y1], [x2-1, y2-1], [x1, y2-1]])


def generate(source, output, count):
    if output.exists() or not output.resolve().is_relative_to(EXPORTS.resolve()):
        raise ValueError('output exists or outside fire exports')
    info=json.loads((source/'summary.json').read_text(encoding='utf-8'))
    if info.get('status')!='complete_candidates': raise ValueError('source batch incomplete')
    records=[json.loads(line) for line in (source/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    if not 1 <= count <= len(records): raise ValueError('invalid count')
    for sub in ('images/train','labels/train'):(output/sub).mkdir(parents=True,exist_ok=True)
    names=set();name_idx=0
    with (output/'manifest.jsonl').open('w',encoding='utf-8') as manifest:
        for row in records[:count]:
            stem=row['id']; input_image=source/'name/images/train'/f'{stem}.jpg'; input_label=source/'name/labels/train'/f'{stem}.txt'
            img=read_image(input_image); generated=[]
            if row['name_positive']:
                for obj in row['instances']:
                    if obj['kind']!='name_sign':continue
                    text=make_name(name_idx); name_idx+=1
                    write_paper(img,obj['box'],text);names.add(text)
                    generated.append({'text':text,'box':obj['box'],'ocr_status':'candidate_requires_readability_review'})
            ok,data=cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,94])
            if not ok:raise ValueError('encode failed')
            data.tofile(str(output/'images/train'/f'{stem}.jpg'))
            shutil.copyfile(input_label,output/'labels/train'/f'{stem}.txt')
            manifest.write(json.dumps({'id':stem,'source_candidate':input_image.relative_to(ROOT).as_posix(),
                'source_groups': ['scene_video_49s_shared','blender_v13_shared_model'],
                'name_positive':row['name_positive'],'generated_papers':generated,
                'status':'candidate_requires_visual_review'},ensure_ascii=False)+'\n')
    summary={'status':'complete_candidates','count':count,'distinct_names':len(names),
        'positive':sum(row['name_positive'] for row in records[:count]),
        'purpose':'name paper detection train candidates; text records need OCR readability review',
        'source':source.relative_to(ROOT).as_posix(),
        'source_groups':['scene_video_49s_shared','blender_v13_shared_model']}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=10)
    args=p.parse_args();print(json.dumps(generate((ROOT/args.source).resolve(),(ROOT/args.output).resolve(),args.count),ensure_ascii=False))
