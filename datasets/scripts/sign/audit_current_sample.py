"""Audit current sign samples or full exports and rebuild their combined entry."""
import json
import argparse
import math
import hashlib
from itertools import combinations
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from datasets.scripts.sign import compose_sign_v4_measured as compositor

ROOT = compositor.ROOT


def enclosed_pixels(mask):
    """Diagnostic for isolated background windows; does not decide valid occlusion."""
    binary=np.asarray(mask,dtype=np.uint8)
    contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    filled=np.zeros_like(binary)
    cv2.drawContours(filled,contours,-1,1,-1)
    return int(np.count_nonzero((filled>0)&(binary==0)))


def audit(full=False):
    _,source = compositor.load_task_config('sign')
    root = ROOT / ('datasets/exports/sign' if full else 'datasets/samples/sign')
    evidence = ROOT / 'datasets/annotated/sign'
    errors, stats, parents, hashes, combined, previews = [], {}, {}, {}, [], []
    sprites = {}
    provenance = []
    full_plans = {}
    split_names=tuple(source['full_dataset']['plans']) if full else ('train','val')
    for split in split_names:
        folder = root / split
        manifest = json.loads((folder/'manifest.json').read_text(encoding='utf8'))
        plan_path = ROOT / (source['full_dataset']['plans'][split] if full else source['diverse_sample_plans'][split])
        plan = json.loads(plan_path.read_text(encoding='utf8'))
        full_plans[split] = plan_path.relative_to(ROOT).as_posix()
        frames = {row['image']:row for row in plan['background_inputs']}
        parents[split], hashes[split] = set(), set()
        pairs, layouts, footprints, photo_metrics = defaultdict(dict), Counter(), Counter(), {}
        boxes_total = empty = old_boxes = occluded = hard_negative = small = 0
        seen = set()
        image_groups = Counter()
        domain_counts = Counter()
        direct_count = 0
        enclosed_occlusions=[]
        for sample in manifest['samples']:
            actual, parameters = sample['actual'], sample['declared_parameters']
            image = Image.open(folder/sample['image']).convert('RGB')
            image.load()
            lines = (folder/sample['label']).read_text(encoding='utf8').splitlines()
            boxes = ([actual['bbox_xyxy']] if actual['bbox_xyxy'] else []) + [
                item['output_box_xyxy'] for item in actual.get('retained_background_signs', [])]
            if image.size != (640,480) or len(lines) != len(boxes):
                errors.append([split,sample['id'],'size_or_label_count'])
            for line, expected in zip(lines,boxes):
                cls,x,y,w,h = map(float,line.split())
                decoded = [(x-w/2)*640,(y-h/2)*480,(x+w/2)*640,(y+h/2)*480]
                if (cls != 0 or not all(math.isfinite(v) for v in decoded) or
                    not (0 <= decoded[0]+.001 < decoded[2] <= 640.001 and 0 <= decoded[1]+.001 < decoded[3] <= 480.001) or
                    max(abs(a-b) for a,b in zip(decoded,expected)) > .01):
                    errors.append([split,sample['id'],'label_box'])
            boxes_total += len(boxes)
            empty += not boxes
            old_boxes += len(actual.get('retained_background_signs', []))
            if actual['negative_context'] == 'ordinary_signal_supports' and not boxes:
                hard_negative += 1
            frame = frames[actual['background']]
            image_groups[actual.get('quota_group','sample')] += 1
            domain = 'blender' if '/rendered/' in frame['image'] else 'camera'
            domain_counts[domain] += 1
            direct = actual.get('source_kind')=='direct_real'
            direct_count += direct
            if full:
                evidence_path=(folder/'manifest.json').relative_to(ROOT).as_posix()
                instance_metadata=[]
                if actual['bbox_xyxy']:
                    instance_metadata.append({'xyxy':actual['bbox_xyxy'],'view':actual['view'],
                        'layout':actual['layout'],'origin':'composite',
                        'visibility':{'small':'complete','medium':'complete','partial':'partial',
                                      'occluded':'occluded','pole_only':'pole_only'}.get(actual['layout'],'unknown')})
                instance_metadata.extend({'xyxy':item['output_box_xyxy'],'view':'unknown','layout':'unknown',
                    'visibility':'unknown','origin':'retained_background'} for item in actual.get('retained_background_signs',[]))
                provenance.append({'image':(folder/sample['image']).relative_to(ROOT).as_posix(),
                    'label':(folder/sample['label']).relative_to(ROOT).as_posix(),'split':split,
                    'source_id':frame['source_group'],'source_group':frame['source_group'],
                    'parent_video':frame.get('parent_video'),
                    'physical_assets':[actual['foreground_video']] if actual.get('foreground_video') else [],
                    'source_assets':sample['sources'],'source_kind':'derived_image',
                    'generation_method':'augmentation' if direct or not actual['bbox_xyxy'] else 'composite',
                    'image_processing':{'origin_review':'confirmed','parent_origin':'synthetic' if domain=='blender' else 'direct_camera',
                                        'background_origin':'synthetic' if domain=='blender' else 'direct_camera'},
                    'label_status':'reviewed','review_owner':'machine',
                    'review_state':'complete_positive' if boxes else 'complete_negative',
                    'review_evidence':evidence_path,'source_review_evidence':evidence_path,
                    'development_contact':['shared sign source and existing indoor arena development']+
                        (['test parking video has earlier sign checks and person test exposure'] if split=='test' else []),
                    'evaluation_scope':'controlled_shared_foreground_diagnostic'})
                provenance[-1]['sign']={'schema_version':1,'image_size':list(image.size),'instances':instance_metadata,
                    'scene':{'quota_group':actual.get('quota_group','unknown'),'tone':actual['tone'],
                        'background_domain':domain,'source_kind':actual.get('source_kind','unknown'),
                        'negative_context':actual.get('negative_context') or 'none'}}
            parents[split].add(frame['source_group'])
            hashes[split].add(hashlib.sha256(image.tobytes()).hexdigest())
            if actual['bbox_xyxy']:
                name = actual['foreground_mask']
                if name not in sprites:
                    sprites[name] = (compositor.pilot.repaired_back(None,write_artifacts=False) if name=='train_0.png'
                                     else compositor.base.cutout(name))
                sprite = sprites[name]
                height = parameters['height']
                width = round(sprite.shape[1]*height/sprite.shape[0])
                alpha = cv2.resize(sprite[:,:,3],(width,height),interpolation=cv2.INTER_AREA)
                left,top = parameters['left'],parameters['top']
                x1,y1,x2,y2 = max(0,left),max(0,top),min(640,left+width),min(480,top+height)
                visible = alpha[y1-top:y2-top,x1-left:x2-left] >= 127.5
                before = int(visible.sum())
                original_holes=enclosed_pixels(visible) if parameters['occlusion_polygons'] else 0
                cover = np.zeros((480,640),dtype=np.uint8)
                for polygon in parameters['occlusion_polygons']:
                    cv2.fillPoly(cover,[np.asarray(polygon,dtype=np.int32)],1)
                visible &= cover[y1:y2,x1:x2] == 0
                ys,xs = np.nonzero(visible)
                expected = [int(xs.min()+x1),int(ys.min()+y1),int(xs.max()+x1+1),int(ys.max()+y1+1)]
                if expected != actual['bbox_xyxy']:
                    errors.append([split,sample['id'],'visible_mask_box'])
                if parameters['occlusion_polygons']:
                    occluded += 1
                    if enclosed_pixels(visible)>original_holes:
                        enclosed_occlusions.append(sample['id'])
                    if not 0 < int(visible.sum()) < before:
                        errors.append([split,sample['id'],'ineffective_occluder'])
                small += min(expected[2]-expected[0],expected[3]-expected[1]) < 32
                layout = parameters['layout']
                layouts[(layout,actual['view'])] += 1
                footprints[top+height] += 1
                key = json.dumps([actual['background'],name,parameters],sort_keys=True)
                pairs[key][actual['tone']] = sample['id']
                hsv = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2HSV)
                photo_metrics[sample['id']] = [float(hsv[:,:,1].mean()),float(hsv[:,:,2].mean())]
            preview_key = (actual.get('layout'),actual['view'])
            if actual['negative_context']=='ordinary_signal_supports':
                preview_key = ('ordinary_signal_supports',None)
            elif actual.get('retained_background_signs') and not actual['bbox_xyxy']:
                preview_key = ('retained',actual['background'])
            if ((actual['bbox_xyxy'] and actual['tone']=='normal' and preview_key not in seen) or
                actual.get('retained_background_signs') and actual['tone']=='normal' and preview_key not in seen or
                actual['negative_context']=='ordinary_signal_supports' and actual['tone']=='normal' and preview_key not in seen):
                previews.append((split,sample,boxes));seen.add(preview_key)
            combined.append({**sample,'id':split+'_'+sample['id'],'split':split,
                             'image':split+'/'+sample['image'],'label':split+'/'+sample['label']})
        for members in pairs.values():
            if not full and set(members) != set(compositor.TONE_GAINS):
                errors.append([split,'incomplete_tone_pair',members])
                continue
            if not {'normal','dark','low_saturation'} <= set(members):continue
            normal = photo_metrics[members['normal']]
            dark = photo_metrics[members['dark']]
            low_sat = photo_metrics[members['low_saturation']]
            if not (dark[1] < normal[1] and low_sat[0] < normal[0]):
                errors.append([split,'photometric_pair_not_measured',members])
        missing = [(layout,view) for layout in ('small','medium','partial','pole_only')
                   for view in ('front','oblique','side','back') if not layouts[(layout,view)]]
        if missing: errors.append([split,'missing_joint_cells',missing])
        if full and (image_groups!=Counter(source['full_dataset']['quota_groups'][split]) or len(manifest['samples'])!=source['full_dataset']['totals'][split]):
            errors.append([split,'full_image_quota_mismatch',dict(image_groups)])
        stats[split] = {'quota_groups':dict(image_groups),'background_domains':dict(domain_counts),'direct_real_images':direct_count,'images':len(manifest['samples']),'boxes':boxes_total,'empty_labels':empty,
            'within_split_identical_rgb_images':len(manifest['samples'])-len(hashes[split]),
            'background_frames':len({sample['actual']['background'] for sample in manifest['samples']}),
            'parent_groups':len(parents[split]),'retained_original_boxes':old_boxes,
            'four_tone_layout_groups':sum(set(v)==set(compositor.TONE_GAINS) for v in pairs.values()),'partial_tone_layout_groups':sum(set(v)!=set(compositor.TONE_GAINS) for v in pairs.values()),'occluded_images':occluded,
            'ordinary_signal_support_negatives':hard_negative,'small_box_short_side_under32':small,
            'enclosed_occlusion_window_images':enclosed_occlusions,
            'foot_y_counts':dict(footprints),'joint_cells':{f'{a}/{b}':n for (a,b),n in layouts.items()}}
    parent_pairs={f'{a}_{b}':sorted(parents[a]&parents[b]) for a,b in combinations(split_names,2)}
    duplicate_pairs={f'{a}_{b}':len(hashes[a]&hashes[b]) for a,b in combinations(split_names,2)}
    overlap=sorted({parent for values in parent_pairs.values() for parent in values})
    duplicates=sum(duplicate_pairs.values())
    report = {'status':'passed' if not errors and not overlap and not duplicates else 'failed',
        'splits':stats,'background_parent_overlap':overlap,'cross_split_identical_rgb_images':duplicates,
        'background_parent_overlap_by_pair':parent_pairs,'cross_split_identical_rgb_by_pair':duplicate_pairs,
        'scope':source['full_dataset']['scope'] if full else 'shared photographed sign foreground; controlled same-scene validation, not independent field evaluation',
        'errors':errors}
    (evidence/('full_dataset_audit.json' if full else 'current_sample_audit.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    if report['status'] != 'passed': raise ValueError(report)
    contact = Image.new('RGB',(1600,math.ceil(len(previews)/5)*265),'white')
    draw = ImageDraw.Draw(contact)
    for index,(split,sample,boxes) in enumerate(previews):
        image = Image.open(root/split/sample['image']).convert('RGB')
        for box in boxes: ImageDraw.Draw(image).rectangle(box,outline='lime',width=2)
        image.thumbnail((320,240));x=index%5*320;y=index//5*265;contact.paste(image,(x,y))
        draw.text((x+3,y+243),split+' '+sample['id']+' '+str(sample['actual']['layout']),fill='black')
    contact.save(evidence/('full_dataset_contact.jpg' if full else 'current_sample_contact.jpg'))
    (root/'manifest.json').write_text(json.dumps({'status':'complete_candidates',
        'split_manifests':{split:f'{split}/manifest.json' for split in stats},'counts':stats,'samples':combined,
        'audit':'datasets/annotated/sign/full_dataset_audit.json' if full else 'datasets/annotated/sign/current_sample_audit.json',
        'plans':full_plans,'generation_plans':full_plans},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    if full:
        by_split={split:{asset for row in provenance if row['split']==split for asset in row['source_assets']} for split in split_names}
        reused=[]
        for asset in sorted(set().union(*by_split.values())):
            used=[split for split in split_names if asset in by_split[split]]
            if len(used)>1:reused.append({'relation':'source_assets','source':asset,'splits':used})
        reused += [{'relation':'physical_assets','source':compositor.base.SIGN_VIDEO,'splits':list(split_names)}]
        (root/'workspace_sources.json').write_text(json.dumps({'model_key':'sign','selection_mode':'manifest_only',
            'records':provenance,'source_reuse':reused,'exposure_warnings':[
                source['full_dataset']['scope']]},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        (root/'data.yaml').write_text(json.dumps({'names':['sign_fixture'],**{split:f'{split}/images' for split in split_names}},indent=2)+'\n',encoding='utf8')
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full',action='store_true',help='audit current full exports and build their training input')
    args=parser.parse_args()
    print(json.dumps(audit(args.full),ensure_ascii=False))
