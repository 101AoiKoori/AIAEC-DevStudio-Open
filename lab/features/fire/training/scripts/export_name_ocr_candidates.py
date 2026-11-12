"""Export name-paper crops and generated text from existing detection candidates.

Split membership is inherited, including provisional evaluation status and
unknown source overlap. Readable-sized crops still require human OCR review.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from compose_scene_fire import ROOT, EXPORTS, read_image
from fire_blender_labels import yolo_row

TRAIN = EXPORTS / 'name_varied_candidates_4000_20261110'
EVAL = EXPORTS / 'scene_eval_candidates_120_per_group_20261110'


def text_crop_state(image, box, text):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    if not isinstance(text, str) or not text.strip():
        return 'missing_generated_text'
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        return 'invalid_box'
    if x1 == 0 or y1 == 0 or x2 == width or y2 == height:
        return 'touches_image_edge_possible_truncation'
    if y2-y1 < 16 or x2-x1 < 8*len(text):
        return 'too_small_for_text_review'
    gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    if float(np.percentile(gray, 95) - np.percentile(gray, 5)) < 15:
        return 'low_contrast_for_text_review'
    return 'candidate_requires_readability_review'


def expected_rows(objects, width, height):
    return [yolo_row(o['box'], width, height) for o in objects]


def check_rows(path, expected):
    rows = path.read_text(encoding='utf-8').splitlines()
    if len(rows) != len(expected):
        raise ValueError(f'paper label count mismatch: {path}')
    for actual, wanted in zip(rows, expected):
        a, b = actual.split(), wanted.split()
        if len(a) != 5 or a[0] != '0' or any(abs(float(x)-float(y)) > 0.000002 for x,y in zip(a[1:],b[1:])):
            raise ValueError(f'paper label coordinates mismatch: {path}')


def sample_records(split):
    if split == 'train':
        root = TRAIN
        rows = [json.loads(x) for x in (root/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
        for row in rows:
            yield row['id'], root/'images/train'/f"{row['id']}.jpg", root/'labels/train'/f"{row['id']}.txt", row['generated_papers'], row['source_groups'], {'source_candidate': row['source_candidate']}
    else:
        group = 'scene_track' if split == 'val' else 'scene_tile'
        root = EVAL/group
        rows = [json.loads(x) for x in (root/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
        for row in rows:
            objects = [{'box': b, 'text': row['generated_text']} for b in row['name_boxes']]
            yield row['id'], root/'name/images'/f"{row['id']}.jpg", root/'name/labels'/f"{row['id']}.txt", objects, [row['source_group']], {'source_image': row['source_image'], 'source_video': row['source_video']}


def export(output, limit):
    if output.exists() or not output.resolve().is_relative_to(EXPORTS.resolve()):
        raise ValueError('output exists or is outside fire exports')
    output.mkdir(parents=True)
    summary={'status':'running','split':{},'source_independence':'provisional; model reference overlap and sessions unknown'}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    for split in ('train','val','test'):
        (output/split/'images').mkdir(parents=True)
        counts=Counter(); unique=set(); seen=0
        with (output/split/'labels.tsv').open('w',encoding='utf-8',newline='\n') as labels, (output/split/'manifest.jsonl').open('w',encoding='utf-8',newline='\n') as manifest, (output/split/'excluded.jsonl').open('w',encoding='utf-8',newline='\n') as excluded:
            for stem,image_path,label_path,objects,groups,lineage in sample_records(split):
                if limit and seen >= limit:break
                if not objects: continue
                seen += 1
                image=read_image(image_path);height,width=image.shape[:2]
                check_rows(label_path,expected_rows(objects,width,height))
                for index,obj in enumerate(objects):
                    box=obj['box'];text=obj['text'];state=text_crop_state(image,box,text)
                    record={'source_detection_image':image_path.relative_to(ROOT).as_posix(),'source_id':stem,'source_groups':groups,
                            'lineage':lineage,'source_box_xyxy':box,'generated_text':text,'status':state,
                            'split_status':'train_candidate' if split=='train' else split+'_candidate'}
                    counts[state]+=1
                    if state != 'candidate_requires_readability_review':
                        excluded.write(json.dumps(record,ensure_ascii=False)+'\n');continue
                    x1,y1,x2,y2=box;crop=image[y1:y2,x1:x2]
                    name=f'{stem}_{index:02}.png';ok,data=cv2.imencode('.png',crop)
                    if not ok:raise ValueError('crop encoding failed')
                    data.tofile(str(output/split/'images'/name))
                    record['crop']='images/'+name;manifest.write(json.dumps(record,ensure_ascii=False)+'\n')
                    labels.write('images/'+name+'\t'+text+'\n');unique.add(text)
        summary['split'][split]={'included':counts['candidate_requires_readability_review'],
                                 'excluded':sum(counts.values())-counts['candidate_requires_readability_review'],
                                 'excluded_reasons':{k:v for k,v in counts.items() if k != 'candidate_requires_readability_review'},
                                 'unique_texts':len(unique)}
    summary['status']='complete_candidates';summary['review']='Generated text and boxes matched. Size, edge and contrast filters are automatic; readability and acceptance remain unverified. Val/test source and model-reference overlap unknown.'
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--limit-positive-images',type=int,default=0)
    args=parser.parse_args();print(json.dumps(export((ROOT/args.output).resolve(),args.limit_positive_images),ensure_ascii=False))
