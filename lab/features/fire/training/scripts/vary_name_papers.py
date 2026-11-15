"""Replace generated name text on existing training candidate paper boxes."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from compose_scene_fire import ROOT, EXPORTS, read_image
from compose_scene_fire_eval import FONT, name_paper, paste
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard
from datasets.scripts.common.names import CHARS, SUFFIX, make_name

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


def _generate(source, output, count):
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


def generate(source, output, count):
    records = [json.loads(line) for line in (source / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    if not 1 <= count <= len(records):
        raise ValueError('invalid count')
    chosen = records[:count]
    images = [source / 'name/images/train' / (r['id'] + '.jpg') for r in chosen]
    labels = [source / 'name/labels/train' / (r['id'] + '.txt') for r in chosen]
    metadata = [source / 'summary.json', source / 'manifest.jsonl', FONT, *labels]
    with generation_guard('fire.vary_name_papers:' + str(output.resolve().relative_to(ROOT)),
                          [*metadata, *images], [output], {'count': count},
                          code_paths=[Path(__file__), ROOT / 'lab/features/fire/training/scripts/compose_scene_fire.py',
                                      ROOT / 'lab/features/fire/training/scripts/compose_scene_fire_eval.py',
                                      ROOT / 'datasets/scripts/common/images.py', ROOT / 'datasets/scripts/common/annotations.py',
                                      ROOT / 'datasets/scripts/common/names.py', ROOT / 'datasets/scripts/common/paper.py',
                                      ROOT / 'datasets/scripts/common/perspective.py'],
                          procedural_inputs=metadata, root=ROOT) as run:
        if run.reused:
            return json.loads((output / 'summary.json').read_text(encoding='utf-8'))
        result = _generate(source, output, count)
        result['batch_source_refs'] = run.source_refs
        (output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=10)
    args=p.parse_args();print(json.dumps(generate((ROOT/args.source).resolve(),(ROOT/args.output).resolve(),args.count),ensure_ascii=False))
