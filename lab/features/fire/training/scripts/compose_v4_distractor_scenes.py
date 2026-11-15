"""Compose source-grouped Blender building renders with reviewed road distractors."""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/fire/. Production remains paused.")

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT / 'lab/training_common'), str(Path(__file__).resolve().parent)]
from camera_tone import apply_tone, read_image, source_role, write_image  # noqa: E402
from lab.scripts.data.v4_source_gate import build_queue, site_sequence_times, SITE_VIDEO  # noqa: E402
from compose_scene_fire import DEFAULT_SOURCE, load_sources  # noqa: E402
from fire_blender_labels import yolo_row  # noqa: E402
from lab.training_common.production_guard import generation_guard  # noqa: E402
from lab.scripts.data import v4_source_gate as source_gate  # noqa: E402

SEED = 20261120
EXPORTS = ROOT / 'datasets/exports/fire'
REVIEW = ROOT / 'datasets/annotated/fire/v4_distractor_ground_review_20261112'
BIN_SOURCE = ROOT / 'datasets/exports/waste_bin/blender_quantity_candidates_160_20261112'
BACKGROUNDS = (
    # Rectangles are conservative 640x480 bounds for actual background objects.
    {'frame': 5000, 'split': 'train', 'distractors': ['red_bin', 'blue_bin', 'white_building_side'],
     'anchor': [410, 310, 465, 350], 'height': [185, 230], 'forbid': [[0, 140, 355, 405], [510, 0, 640, 315]]},
    {'frame': 6000, 'split': 'train', 'distractors': ['traffic_lights', 'people', 'e_bike'],
     'anchor': [260, 440, 400, 470], 'height': [210, 250], 'forbid': [[95, 60, 530, 205]]},
    {'frame': 4000, 'split': 'val', 'distractors': ['vehicle', 'red_vehicle_graphics'],
     'anchor': [490, 420, 575, 465], 'height': [190, 245], 'forbid': [[80, 0, 430, 225]]},
    {'frame': 8500, 'split': 'test', 'distractors': ['crosswalk', 'distant_vehicle'],
     'anchor': [270, 365, 420, 450], 'height': [190, 245], 'forbid': [[400, 0, 640, 40]]},
)


def rel(path):
    return path.relative_to(ROOT).as_posix()


def overlap(box, forbidden):
    a,b,c,d=box
    return max(0,min(c,forbidden[2])-max(a,forbidden[0])) * max(0,min(d,forbidden[3])-max(b,forbidden[1]))


def source_groups(rng):
    groups, skipped = load_sources(DEFAULT_SOURCE)
    result = {}
    for bucket in (0, 1, 3):
        valid = [(row,image,mask) for row,image,mask in groups[bucket]
                 if len(row['active_buildings']) == 1 and not (mask[0].any() or mask[-1].any() or
                                                               mask[:,0].any() or mask[:,-1].any())]
        valid.sort(key=lambda item: item[0]['image'])
        rng.shuffle(valid)
        n=len(valid)//4
        if n<4:raise ValueError('Insufficient unique source render frames')
        result[bucket]={'train':valid[:2*n], 'val':valid[2*n:3*n], 'test':valid[3*n:]}
    return result, dict(skipped)


def compose(background, row, foreground, mask, spec, rng):
    x,y,w,h=cv2.boundingRect(mask)
    if min(w,h)<=0:raise ValueError('Empty foreground silhouette')
    fire_objects=[o for o in row['instances'] if o['kind']=='fire_sticker' and o['state']=='visible']
    names=[o for o in row['instances'] if o['kind']=='name_sign' and o['state']=='visible']
    if len(names)!=1:raise ValueError('Expected exactly one visible building name')
    objects=names+fire_objects
    for _ in range(60):
        height=rng.uniform(*spec['height']);scale=height/h
        ax=rng.uniform(spec['anchor'][0],spec['anchor'][2]); ay=rng.uniform(spec['anchor'][1],spec['anchor'][3])
        tx=ax-scale*(x+w/2);ty=ay-scale*(y+h)
        box=[scale*x+tx,scale*y+ty,scale*(x+w)+tx,scale*(y+h)+ty]
        if not (0<=box[0]<box[2]<=640 and 0<=box[1]<box[3]<=480):continue
        if any(overlap(box,f)>0 for f in spec['forbid']):continue
        candidates=[]
        for obj in objects:
            a,b,c,d=obj['pixel_bbox_xyxy']; transformed=[scale*a+tx,scale*b+ty,scale*c+tx,scale*d+ty]
            if not (0<=transformed[0]<transformed[2]<=640 and 0<=transformed[1]<transformed[3]<=480):break
            if obj['kind']=='fire_sticker' and (transformed[2]-transformed[0]<14 or transformed[3]-transformed[1]<20):break
            candidates.append((obj,transformed))
        else:
            mat=np.float32([[scale,0,tx],[0,scale,ty]])
            warped=cv2.warpAffine(foreground,mat,(640,480),flags=cv2.INTER_LINEAR)
            silhouette=cv2.warpAffine(mask,mat,(640,480),flags=cv2.INTER_NEAREST)
            alpha=cv2.GaussianBlur(silhouette,(3,3),.45).astype(np.float32)[:,:,None]/255
            result=background.copy()
            shadow=np.zeros((480,640),np.uint8)
            cv2.ellipse(shadow,(round(ax),min(479,round(ay))),(max(5,round((box[2]-box[0])*.46)),5),0,0,360,55,-1)
            shadow=cv2.GaussianBlur(shadow,(0,0),4).astype(np.float32)[:,:,None]/255
            result=np.clip(result.astype(np.float32)*(1-shadow),0,255)
            gain=rng.uniform(.82,.98)
            result=np.clip(warped.astype(np.float32)*gain*alpha+result*(1-alpha),0,255).astype(np.uint8)
            return result,candidates,{'scale':scale,'tx':tx,'ty':ty,'target_bbox_xyxy':box,
                'ground_anchor_xy':[ax,ay],'foreground_gain':gain,'contact_shadow':'synthetic Gaussian, not measured'}
    return None


def add_red_bin(image, building_box, rng):
    """Place one isolated Blender bin on the road without covering the building."""
    image_path=BIN_SOURCE/'images/g0002_s09.png'
    mask_path=BIN_SOURCE/'mask_isolated/g0002_s09_red.png'
    source=cv2.imdecode(np.fromfile(str(image_path),np.uint8),cv2.IMREAD_COLOR)
    raw=cv2.imdecode(np.fromfile(str(mask_path),np.uint8),cv2.IMREAD_UNCHANGED)
    if source is None or raw is None or raw.shape!=(480,640,4):raise ValueError('Missing bin or isolated mask')
    alpha=(raw[:,:,3]>128).astype(np.uint8)*255;x,y,w,h=cv2.boundingRect(alpha)
    if not w or not h:raise ValueError('Empty isolated bin mask')
    for _ in range(50):
        height=rng.randint(48,72);width=max(1,round(w*height/h))
        center=rng.randint(470,565);bottom=rng.randint(410,460)
        left=center-width//2;top=bottom-height;box=[left,top,left+width,bottom]
        if left<0 or left+width>640 or (building_box is not None and overlap(box,building_box)>0):continue
        fg=cv2.resize(source[y:y+h,x:x+w],(width,height),interpolation=cv2.INTER_AREA)
        a=cv2.resize(alpha[y:y+h,x:x+w],(width,height),interpolation=cv2.INTER_AREA)
        a=cv2.GaussianBlur(a,(3,3),.4).astype(np.float32)[:,:,None]/255
        result=image.copy();shadow=np.zeros((480,640),np.uint8)
        cv2.ellipse(shadow,(center,bottom),(max(3,width//2),3),0,0,360,45,-1)
        shadow=cv2.GaussianBlur(shadow,(0,0),3).astype(np.float32)[:,:,None]/255
        result=np.clip(result.astype(np.float32)*(1-shadow),0,255)
        target=result[top:bottom,left:left+width];result[top:bottom,left:left+width]=target*(1-a)+fg.astype(np.float32)*a
        return np.clip(result,0,255).astype(np.uint8),{'bbox_xyxy':box,'source_render':rel(image_path),
            'source_instance_mask':rel(mask_path),'source_bin':'red','ground_anchor_xy':[center,bottom],
            'scope':'Blender bin used as fire/name distractor, not an additional task label'}
    return None


def _generate(args):
    out=(ROOT/args.output).resolve()
    if not out.is_relative_to(EXPORTS.resolve()) or out.exists():raise ValueError('Output must be a new fire export directory')
    rng=random.Random(SEED);groups,skipped_sources=source_groups(rng);known=build_queue(ROOT);times=site_sequence_times(ROOT)
    REVIEW.mkdir(parents=True,exist_ok=True)
    review={'scope':'AI visual review of four 640x360 site frames and conservative 640x480 placement regions; one video, no independent-capture claim',
            'evidence':'runs/fire_sticker/v4_candidate_ground_sources.png','human_accepted':False,'backgrounds':[]}
    for spec in BACKGROUNDS:
        source=ROOT/f"datasets/annotated/site_capture_20261112/sequence_640_v1/full_scene/frame_{spec['frame']:06d}.png"
        role,reason=source_role(ROOT,source,known)
        if role=='regression':raise ValueError(f'Background restricted to regression: {source} {reason}')
        review['backgrounds'].append({**spec,'parent_image':rel(source),'source_video':SITE_VIDEO,
            'source_time_seconds':times[spec['frame']],'review':'no discernible fire sticker or name paper in source; exclusion rectangles and floor anchors are fitted'})
    (REVIEW/'review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    records=[];counts=Counter();rejects=Counter()
    try:
        for spec in BACKGROUNDS:
            if args.pilot and spec['split']!='train':continue
            source=ROOT/f"datasets/annotated/site_capture_20261112/sequence_640_v1/full_scene/frame_{spec['frame']:06d}.png"
            original=read_image(source)
            if original.shape[:2]!=(360,640):raise ValueError('Expected 640x360 source frame')
            background=cv2.resize(original[:,80:560],(640,480),interpolation=cv2.INTER_LINEAR)
            split=spec['split'];per_state=(2 if args.pilot else (16 if split=='train' else 8))
            states=('hard_negative',) if spec['frame']==5000 else ('hard_negative','empty_building','single_fire','multi_fire')
            if spec['frame']==6000:
                states+=('red_bin_only','red_bin_empty_building','red_bin_single_fire','red_bin_multi_fire')
            for state in states:
                total=2 if state in ('hard_negative','red_bin_only') else per_state*2
                for index in range(total):
                    tone=('normal','dark_low_saturation')[index%2];value,sat=(1.,1.) if tone=='normal' else (.70,.65)
                    image=background.copy();objects=[];transform=None;source_record=None
                    base_state=state.removeprefix('red_bin_')
                    if base_state not in ('hard_negative','only'):
                        bucket={'empty_building':0,'single_fire':1,'multi_fire':3}[base_state]
                        pool=groups[bucket][split]
                        for _ in range(30):
                            source_record,foreground,mask=rng.choice(pool)
                            candidate=compose(background,source_record,foreground,mask,spec,rng)
                            if candidate is not None:
                                image,objects,transform=candidate;break
                        else:
                            rejects[split+'|'+state]+=1;continue
                    bin_object=None
                    if state.startswith('red_bin_'):
                        placed=add_red_bin(image,transform['target_bbox_xyxy'] if transform else None,rng)
                        if placed is None:
                            rejects[split+'|'+state+'_bin_placement']+=1;continue
                        image,bin_object=placed
                    if tone!='normal':image=apply_tone(image,value_gain=value,saturation_gain=sat)
                    blur=0.55 if (index//2)%3==1 else 0.
                    if blur:image=cv2.GaussianBlur(image,(0,0),blur)
                    physical_split=('held_'+split if split in ('val','test') and source_record else split)
                    stem=f"fns_{spec['frame']:06d}_{state}_{index:03d}";img=out/'images'/physical_split/(stem+'.png')
                    img.parent.mkdir(parents=True,exist_ok=True);write_image(img,image)
                    labels={};obj_info=[]
                    for task,kind in (('fire','fire_sticker'),('name','name_sign')):
                        label=out/'labels'/task/physical_split/(stem+'.txt');label.parent.mkdir(parents=True,exist_ok=True)
                        boxes=[(obj,box) for obj,box in objects if obj['kind']==kind]
                        label.write_text(''.join(yolo_row(box,640,480)+'\n' for obj,box in boxes),encoding='utf8')
                        labels[task]=rel(label)
                    for obj,box in objects:obj_info.append({'kind':obj['kind'],'bbox_xyxy':box,'source_bbox_xyxy':obj['pixel_bbox_xyxy'],'source_instance_id':obj['id']})
                    records.append({'image':rel(img),'labels':labels,'split':physical_split,'intended_split':split,'state':state,'image_size':[640,480],
                        'background_parent':rel(source),'background_video':SITE_VIDEO,'background_seconds':times[spec['frame']],
                        'background_crop_xyxy':[80,0,560,360],'background_resize':'480x360 to 640x480; no new source detail',
                        'background_review':rel(REVIEW/'review.json'),'distractors':spec['distractors'],
                        'source_render':rel(DEFAULT_SOURCE/source_record['image']) if source_record else None,
                        'source_render_manifest':rel(DEFAULT_SOURCE/'manifest.jsonl'),'mother_scene':'datasets/assets/blender/fire/blender_buildings_v13/three_buildings.blend',
                        'generation_method':'real_scene_composite_blender' if source_record or bin_object else 'direct_video_frame_or_tone_augmentation',
                        'transform':transform,'objects':obj_info,'bin_distractor':bin_object,'tone':tone,'value_gain':value,'saturation_gain':sat,'gaussian_sigma_px':blur,
                        'status':'held_scene_geometry' if physical_split!=split else 'AI_visual_candidate_not_frozen'})
                    counts[physical_split+'|'+state+'|'+tone]+=1
        manifest={'status':'draft_not_deliverable','seed':SEED,'pilot':args.pilot,'counts':dict(counts),'rejected':dict(rejects),
            'scope':'Four frames of one site video, one shared Blender mother scene; distinct foreground render frames by split, not independent captures. Test is synthetic stress candidate, not frozen.',
            'source_pool_counts':{str(k):{s:len(v) for s,v in by.items()} for k,by in groups.items()},
            'source_skips':skipped_sources,'records':records}
        (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps({'output':rel(out),'counts':dict(counts),'rejected':dict(rejects)},ensure_ascii=False))
    except BaseException:
        (out/'FAILED.txt').write_text('Generation failed; inspect before reuse.\n',encoding='utf8');raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args();out=(ROOT/args.output).resolve()
    source_manifest=DEFAULT_SOURCE/'manifest.jsonl'
    queue_media,queue_metadata,queue_code=source_gate.queue_inputs(ROOT)
    metadata=[DEFAULT_SOURCE/'run.json',source_manifest,ROOT/(source_gate.SITE_SEQUENCE+'/ffmpeg.log'),*queue_metadata]
    media=[ROOT/f"datasets/annotated/site_capture_20261112/sequence_640_v1/full_scene/frame_{s['frame']:06d}.png" for s in BACKGROUNDS]
    media += [BIN_SOURCE/'images/g0002_s09.png',BIN_SOURCE/'mask_isolated/g0002_s09_red.png',ROOT/SITE_VIDEO,*queue_media]
    for line in source_manifest.read_text(encoding='utf8').splitlines():
        row=json.loads(line)
        if not row.get('fire_label') or not row.get('name_label') or row.get('occluder'):
            continue
        if any(obj['state'] not in ('visible','outside_frame') for obj in row['instances']):
            continue
        media.extend([DEFAULT_SOURCE/row['image'],DEFAULT_SOURCE/row['instance_pass']])
    code=[Path(__file__),ROOT/'lab/features/fire/training/scripts/compose_scene_fire.py',
          ROOT/'lab/features/fire/training/scripts/fire_blender_labels.py',ROOT/'lab/training_common/camera_tone.py',
          ROOT/'datasets/scripts/common/images.py',ROOT/'datasets/scripts/common/annotations.py',
          *queue_code]
    with generation_guard('fire.compose_distractors:'+str(out.relative_to(ROOT)),
                          list(dict.fromkeys([*metadata,*media])),[out,REVIEW/'review.json'],
                          {'pilot':args.pilot,'seed':SEED,'backgrounds':BACKGROUNDS},code_paths=list(dict.fromkeys(code)),
                          procedural_inputs=list(dict.fromkeys(metadata)),root=ROOT) as run:
        if not run.reused:
            _generate(args)
            manifest=out/'manifest.json'
            result=json.loads(manifest.read_text(encoding='utf8'))
            result['batch_source_refs']=run.source_refs
            manifest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
