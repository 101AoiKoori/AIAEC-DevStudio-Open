"""Complete the existing fire recipe without changing counts or parent splits."""
from collections import Counter,defaultdict
from copy import deepcopy
import json
from pathlib import Path
import random
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from datasets.scripts.common.config import load_task_config
from datasets.scripts.common.frame_environment import query_frames,image_digest
from datasets.scripts.fire import compose_real_facades as base
from datasets.scripts.fire.build_dataset import paper_artwork
from datasets.scripts.fire.replace_empty_negatives import read,apply_split,synchronize_root,EXPORT
from datasets.scripts.fire.fire_blender_labels import decode_ids,srgb_to_linear,visible_bbox

CONFIG=ROOT/'datasets/scripts/fire/config/training_preparation.json'
WORK=ROOT/'datasets/annotated/fire/training_preparation'


def train_pairs(manifest,additions):
    pools={n:[r for r in manifest['records'] if r['branch']=='blender_render' and not r.get('hard_negative')
              and not r.get('contrast_positive') and len(r['instances'])==n] for n in (1,3)}
    for n,rows in pools.items():
        random.Random(411+n).shuffle(rows)
    selected=[]
    for row in additions:
        count=len(row['instances'])
        if not pools.get(count):raise ValueError('Not enough same-count Blender positives for substitution')
        selected.append(pools[count].pop())
    return selected


def train_stage(cfg):
    folder=ROOT/cfg['render_output'];manifest=read(folder/'manifest.json');rows=manifest['records']
    if len(rows)!=cfg['replace_images']['train'] or Counter(r['context']['distractor'] for r in rows)!=cfg['train_types']:
        raise ValueError('Contrast-positive scene budget differs')
    if Counter(len(r['instances']) for r in rows)!={1:300,3:300}:
        raise ValueError('Contrast-positive target counts differ')
    for row in rows:
        rgb=np.asarray(Image.open(folder/row['instance_pass']).convert('RGB'),dtype=np.float32)/255
        ids=decode_ids(srgb_to_linear(rgb),[*[i['id'] for i in row['instances']],*row['distractor']['object_ids']])
        for item in row['instances']:
            bbox,area=visible_bbox(ids==item['id'])
            if bbox!=item['xyxy'] or area!=item['visible_pixels']:raise ValueError('Contrast target mask differs')
    base.OUTPUT=folder;base.validate_records(rows)
    return folder,rows


def select_small(rows,count,used):
    groups=defaultdict(list)
    for row in rows:
        if row['image'] not in used:groups[(row['frame_id'],row.get('tone'))].append(row)
    groups=[sorted(rs,key=lambda r:r['image']) for _,rs in sorted(groups.items())]
    chosen=[]
    if sum(len(g) for g in groups)<count:raise ValueError('Insufficient same-split wall positives')
    while len(chosen)<count:
        for group in groups:
            if group and len(chosen)<count:
                row=group.pop(0);chosen.append(row);used.add(row['image'])
    return chosen


def small_stage(split,cfg,source):
    manifest=read(EXPORT/split/'manifest.json')
    if manifest.get('training_preparation',{}).get('status')=='complete':
        rows=[r for r in manifest['records'] if r.get('patch_id')=='training_preparation']
        # Reuse the staged original paths; apply_split can finish the receipt.
        folder=WORK/'prepared'/split;stored=read(folder/'manifest.json')
        return folder,stored['records'],[]
    available=[r for r in manifest['records'] if r['branch']=='real_background_composite' and r['instances']]
    sources={r['frame_id']:r for r in query_frames(task='fire',role='composite_background',split=split)}
    recipe=source['sample_recipe'];paper=paper_artwork(source);folder=WORK/'prepared'/split
    additions=[];retired=[];used=set()
    for case in cfg['eval_small_cases']:
        old_rows=select_small([r for r in available if len(r['instances'])==case['count']],case['images'],used)
        for old in old_rows:
            original=sources[old['frame_id']]
            if image_digest(original['image'])!=original['content_sha256'] or original['label']['instances']:
                raise ValueError('Small-target wall no longer has complete target-absence evidence')
            row=deepcopy(original);row['geometry']['supervision_crop_xyxy']=old['transform']['crop_xyxy']
            attribution=base.identify_building(row,source);building=attribution['target_building']
            use={'row':row,'branch':'real_background_composite','count':case['count'],'tone':old['tone'],**attribution}
            old_size=np.median([min(i['xyxy'][2]-i['xyxy'][0],i['xyxy'][3]-i['xyxy'][1]) for i in old['instances']])
            old_size*=cfg['model_imgsz']/max(old['transform']['output_size'])
            fraction=sum(recipe['sticker_size']['height_fraction'][building])/2*case['target_short_side_px']/old_size
            seed=cfg['seed']+70000+len(additions)
            for attempt in range(8):
                local=deepcopy(recipe);local['sticker_size']['height_fraction'][building]=[fraction*.97,fraction*1.03]
                try:
                    image,instances,transform=base.compose_camera(use,paper,local,seed)
                except ValueError as error:
                    if 'below this batch discernibility size' not in str(error):raise
                    fraction*=1.25
                    continue
                scale=cfg['model_imgsz']/max(image.shape[:2])
                sizes=[min(i['xyxy'][2]-i['xyxy'][0],i['xyxy'][3]-i['xyxy'][1])*scale for i in instances]
                low,high=(8,16) if case['target_short_side_px']==12 else (16,32)
                if all(low<s<=high for s in sizes):break
                fraction*=case['target_short_side_px']/float(np.median(sizes))
            else:raise ValueError('Cannot realize the requested small-target input-pixel bin')
            stem=f'small_sticker_{len(additions)+1:06d}';image_path=folder/'images'/(stem+'.jpg');label=folder/'labels'/(stem+'.txt')
            image_path.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
            cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,95])[1].tofile(str(image_path))
            label.write_text(''.join(base.yolo_row(i['xyxy'],image.shape[1],image.shape[0])+'\n' for i in instances),encoding='utf8')
            result=deepcopy(old)
            for key in ('overlay','instance_pass','patch_id'):result.pop(key,None)
            result.update(image=f'images/{image_path.name}',label=f'labels/{label.name}',instances=instances,transform=transform,
                sample_seed=seed,production_state_id=stem,context=old.get('context',{}),
                building_counts=base.counts_for(attribution['visible_buildings'],instances,'visible_detectable_instances_in_output'),
                size_control={'target_input_short_side_px':case['target_short_side_px'],'actual_input_short_sides_px':sizes})
            additions.append(result);retired.append(old)
    base.OUTPUT=folder;base.validate_records(additions)
    base.write_json(folder/'manifest.json',{'records':additions,'images':len(additions)})
    return folder,additions,retired


def main():
    cfg=read(CONFIG);cfg.update(config_path=base.relative(CONFIG),receipt_key='training_preparation')
    source=load_task_config('fire')[1];root=read(EXPORT/'manifest.json')
    if root.get('training_preparation',{}).get('status')=='complete':
        print(json.dumps({'status':'already_complete','training_preparation':root['training_preparation']}));return
    if not (ROOT/cfg['render_output']/'manifest.json').is_file():
        subprocess.run([source['sample_recipe']['blender_executable'],'--background',str(ROOT/cfg['source_snapshot']),
            '--python-exit-code','1','--python',str(ROOT/'datasets/scripts/fire/render_contrast_positives.py')],cwd=ROOT,check=True)
    stage,train=train_stage(cfg);current=read(EXPORT/'train/manifest.json')
    old=[] if current.get('training_preparation',{}).get('status')=='complete' else train_pairs(current,train)
    prepared={'train':(stage,train,old),**{s:small_stage(s,cfg,source) for s in ('val','test')}}
    manifests={s:apply_split(s,p[0],p[1],cfg,source,retired_override=p[2]) for s,p in prepared.items()}
    synchronize_root(manifests,cfg,source)
    print(json.dumps({'status':'complete','replaced':cfg['replace_images'],'images':3600},ensure_ascii=False))


if __name__=='__main__':main()
