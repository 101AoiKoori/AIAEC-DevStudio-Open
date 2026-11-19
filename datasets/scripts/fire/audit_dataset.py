"""Independently re-read current fire pairs, instance passes and hybrid pixels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from datasets.scripts.fire import compose_real_facades as base
from datasets.scripts.fire.fire_blender_labels import decode_ids,srgb_to_linear,visible_bbox
from datasets.scripts.fire.photo_geometry import blend_render
from datasets.scripts.common.config import load_task_config
from datasets.scripts.common.frame_environment import image_digest


def inspect_folder(folder):
    base.OUTPUT=folder
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf8'))
    rows=manifest['records'];validation=base.validate_records(rows)
    by_pass={r['instance_pass']:r for r in rows if r.get('instance_pass')}
    raw_path=folder/'manifest.jsonl';masks=0;mask_boxes=0;retired_masks=0;retired_positive_masks=0
    by_image={r['image']:r for r in rows}
    if raw_path.exists():
        for line in raw_path.read_text(encoding='utf8').splitlines():
            if not line.strip():continue
            raw=json.loads(line)
            if raw['instance_pass'] not in by_pass:
                count=sum(i['kind']=='fire_sticker' for i in raw['instances'])
                replacement=manifest.get('retired_positive_replacements',{}).get(raw['instance_pass'])
                if raw['instance_pass'] not in manifest.get('retired_instance_passes',[]):
                    raise ValueError('Missing active renderer row or retired row contains a target')
                if count:
                    if not replacement or replacement['count']!=count or len(by_image[replacement['image']]['instances'])!=count:
                        raise ValueError('Retired positive has no same-count replacement')
                    retired_positive_masks+=1
                else:retired_masks+=1
                continue
            row=by_pass[raw['instance_pass']]
            rgb=np.asarray(Image.open(folder/raw['instance_pass']).convert('RGB'),dtype=np.float32)/255
            ids=decode_ids(srgb_to_linear(rgb),[i['id'] for i in raw['instances']])
            fires=[i for i in raw['instances'] if i['kind']=='fire_sticker']
            actual={i['id']:i for i in row['instances']}
            for item in fires:
                bbox,area=visible_bbox(ids==item['id'])
                if bbox!=item['pixel_bbox_xyxy'] or area!=item['visible_pixels'] or bbox!=actual[item['id']]['xyxy']:
                    raise ValueError(f'Instance pass differs from actual label: {row["image"]}/{item["id"]}')
                mask_boxes+=1
            masks+=1
    hybrid_metrics=[]
    for row in rows:
        if (row.get('hard_negative') or row.get('contrast_positive')) and row.get('instance_pass'):
            rgb=np.asarray(Image.open(folder/row['instance_pass']).convert('RGB'),dtype=np.float32)/255
            ids=decode_ids(srgb_to_linear(rgb),[1,*[i['id'] for i in row['instances']],*row['distractor']['object_ids']])
            if row.get('hard_negative') and (np.any(ids==1) or row['instances']):
                raise ValueError('Hard-negative replacement contains a target ID')
            for target in row['instances']:
                bbox,area=visible_bbox(ids==target['id'])
                if bbox!=target['xyxy'] or area!=target['visible_pixels']:
                    raise ValueError('Contrast positive target mask differs')
                mask_boxes+=1
            bbox,area=visible_bbox(np.isin(ids,row['distractor']['object_ids']))
            if bbox!=row['distractor']['xyxy'] or area!=row['distractor']['visible_pixels']:
                raise ValueError('Hard-negative distractor mask differs')
            masks+=1
        if row['branch']!='blender_real_background_composite':continue
        rgba=cv2.imdecode(np.fromfile(folder/row['render_layer'],np.uint8),cv2.IMREAD_UNCHANGED)
        geometries=json.loads((ROOT/row['hybrid_geometry']).read_text(encoding='utf8'))['backgrounds']
        review=next(r for r in geometries if r['id']==row['photo_geometry_id'])
        if image_digest(review['source_image'])!=row['photo_sha256']:
            raise ValueError('Hybrid background bytes changed')
        photo=cv2.imdecode(np.fromfile(ROOT/review['source_image'],np.uint8),cv2.IMREAD_COLOR)
        photo=cv2.resize(photo,(rgba.shape[1],rgba.shape[0]),interpolation=cv2.INTER_AREA)
        final=cv2.imdecode(np.fromfile(folder/row['image'],np.uint8),cv2.IMREAD_COLOR)
        if not np.array_equal(blend_render(photo,rgba),final):
            raise ValueError('Hybrid image differs from its physical render layer and photo')
        camera=row['camera'];scale=np.array([rgba.shape[1],rgba.shape[0]])/review['camera']['image_size']
        x1,y1,x2,y2=np.array(camera['body_bbox_photo_px'])*np.tile(scale,2)
        outside=np.ones(rgba.shape[:2],bool)
        outside[max(0,int(y1)-3):min(rgba.shape[0],int(y2)+4),max(0,int(x1)-3):min(rgba.shape[1],int(x2)+4)]=False
        shadow=outside & (rgba[:,:,3]>2) & (rgba[:,:,3]<245)
        darkness=np.mean(photo.astype(float)-final.astype(float),axis=2)
        hybrid_metrics.append({'image':row['image'],'photo_geometry_id':row['photo_geometry_id'],
            'photo_reprojection_rms_px':camera['photo_reprojection_rms_px'],
            'photo_calibration_size':camera['photo_calibration_size'],
            'model_scale':camera['model_scale'],'footprint_photo_px':camera['footprint_photo_px'],
            'external_shadow_pixels':int(shadow.sum()),
            'external_shadow_mean_darkening_8bit':float(darkness[shadow].mean()) if shadow.any() else None})
    hashes=Counter(image_digest(folder/r['image']) for r in rows)
    if any(n>1 for n in hashes.values()):
        raise ValueError('Exact duplicate final images in this fire output')
    validation.update(instance_pass_images_redecoded=masks,instance_pass_fire_boxes_rechecked=mask_boxes,
                      retired_empty_renderer_records=retired_masks,retired_positive_renderer_records=retired_positive_masks,
                      hard_negative_images=sum(bool(r.get('hard_negative')) for r in rows),
                      exact_duplicate_images=0,hybrid_metrics=hybrid_metrics)
    return manifest,validation


def contact(rows,folder,path,columns=3):
    sheet=Image.new('RGB',(columns*480,((len(rows)+columns-1)//columns)*440),'#eee')
    for index,row in enumerate(rows):
        im=Image.open(folder/row['image']).convert('RGB');d=ImageDraw.Draw(im)
        for item in row['instances']:
            d.rectangle(item['xyxy'],outline=(30,230,60),width=2)
            d.text((item['xyxy'][0],max(0,item['xyxy'][1]-12)),item['building'],fill='lime')
        im.thumbnail((480,402));x=(index%columns)*480;y=(index//columns)*440
        sheet.paste(im,(x+(480-im.width)//2,y));ImageDraw.Draw(sheet).text((x+4,y+406),
            f'{index+1} {row["split"]} '+str({k:v['visible_count'] for k,v in row['building_counts'].items()}),fill='black')
    path.parent.mkdir(parents=True,exist_ok=True);sheet.save(path,quality=92)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--full',action='store_true');args=parser.parse_args()
    output=ROOT/'datasets/annotated/fire'/('current_full_review' if args.full else 'current_sample_review')
    folders=[ROOT/f'datasets/exports/fire/{s}' for s in ('train','val','test')] if args.full else [ROOT/'datasets/samples/fire']
    report={};parents={};all_hashes={}
    for folder in folders:
        manifest,validation=inspect_folder(folder);report[folder.name]=validation
        for row in manifest['records']:
            for parent in row['parents']:parents.setdefault(parent,set()).add(row['split'])
            digest=image_digest(folder/row['image']);all_hashes.setdefault(digest,set()).add(row['split'])
        hybrid=[r for r in manifest['records'] if r['branch']=='blender_real_background_composite']
        if hybrid:contact(hybrid[:24],folder,output/'hybrid_contact_sheet.jpg')
        walls=[r for r in manifest['records'] if r['branch']=='real_background_composite' and len(r['instances'])==3]
        # Show both tone variants and every selected wall in the sample. In the
        # full export choose one actual layout per source/tone, rather than just
        # the first source's repeated variants.
        selected={}
        for row in walls:selected.setdefault((row['frame_id'],row['tone']),row)
        if selected:contact(list(selected.values()),folder,output/f'{folder.name}_walls.jpg')
    if any(len(v)>1 for v in parents.values()):raise ValueError('Parent appears in more than one split')
    if any(len(v)>1 for v in all_hashes.values()):raise ValueError('Identical image crosses splits')
    report.update(cross_split_parent_overlap=0,cross_split_exact_image_overlap=0,parent_groups=len(parents))
    base.write_json(output/'audit.json',report)
    print(json.dumps({'report':base.relative(output/'audit.json'),'images':sum(v['images'] for k,v in report.items() if isinstance(v,dict)),
        'parent_groups':len(parents),'cross_split_parent_overlap':0},ensure_ascii=False))


if __name__=='__main__':main()
