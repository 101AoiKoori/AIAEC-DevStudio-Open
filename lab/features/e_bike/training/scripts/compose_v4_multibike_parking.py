"""Combine reviewed multi-bike frames with varied fallen views on the ground."""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/e_bike/. Production remains paused.")

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[5]
sys.path[:0]=[str(ROOT),str(ROOT/'lab/training_common'),str(Path(__file__).resolve().parent)]
from camera_tone import apply_tone,read_image,source_role,write_image  # noqa: E402
from lab.scripts.data.v4_source_gate import build_queue  # noqa: E402
import synthesize_vehicle_scene as bike  # noqa: E402
from datasets.scripts.common.composite import visible_masks  # noqa: E402

REVIEW=ROOT/'datasets/annotated/e_bike/v4_low_angle_multibike_review_20261112/review_manifest.json'
BACKGROUNDS=ROOT/'datasets/annotated/e_bike/v4_complex_backgrounds_20261112/backgrounds.json'
FIELD=ROOT/'datasets/annotated/e_bike/field_eval_masks_20261110/selection_review_candidate.json'
TONES={'original':(1.,1.),'mild':(.9,.85),'moderate':(.7,.6)}
PLANS=('multiple_fallen','upright_multiple_fallen','edge_truncated','white_back_coexist',
       'person_occludes','overhead_multiple_fallen')
SEED=20261119


def rel(path):return path.relative_to(ROOT).as_posix()


def clean_rgba(image):
    """Remove isolated extraction debris; retain components near the bike."""
    alpha=(image[:,:,3]>127).astype(np.uint8)
    n,labels,stats,_=cv2.connectedComponentsWithStats(alpha,8)
    if n<2:raise ValueError('foreground has no alpha')
    biggest=int(np.argmax(stats[1:,4])+1);x,y,w,h,_=stats[biggest];keep=np.zeros_like(alpha)
    for i in range(1,n):
        a,b,c,d,area=stats[i]
        if area>=30 and a<=x+w+10 and a+c>=x-10 and b<=y+h+10 and b+d>=y-10:
            keep[labels==i]=1
    clean=image.copy();clean[:,:,3]=np.where(keep,clean[:,:,3],0)
    yy,xx=np.nonzero(clean[:,:,3]>127)
    if len(xx)<100:raise ValueError('foreground too small')
    return clean[yy.min():yy.max()+1,xx.min():xx.max()+1]


def place_clipped(canvas,rgba,x,foot):
    height,width=canvas.shape[:2];sh,sw=rgba.shape[:2];left=round(x-sw/2);top=round(foot-sh)
    l,t,r,b=max(left,0),max(top,0),min(left+sw,width),min(top+sh,height)
    if l>=r or t>=b:raise ValueError('sprite outside image')
    patch=rgba[t-top:b-top,l-left:r-left];alpha=patch[:,:,3:4].astype(np.float32)/255
    canvas[t:b,l:r]=np.clip(patch[:,:,:3]*alpha+canvas[t:b,l:r]*(1-alpha),0,255).astype(np.uint8)
    mask=np.zeros((height,width),bool);mask[t:b,l:r]=patch[:,:,3]>127
    if mask.sum()/max(1,(rgba[:,:,3]>127).sum())<.70:raise ValueError('excessive truncation')
    return mask


def _produce():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--count',type=int,default=30);args=parser.parse_args()
    out=(ROOT/args.output).resolve()
    if not out.is_relative_to((ROOT/'datasets/exports/e_bike').resolve()) or out.exists():
        raise ValueError('Output must be a new e-bike export')
    if not 1<=args.count<=5000:raise ValueError('count must be in 1..5000')
    review=json.loads(REVIEW.read_text(encoding='utf-8'))['rows']
    backgrounds=json.loads(BACKGROUNDS.read_text(encoding='utf-8'))['backgrounds']
    white=next(x for x in backgrounds if x['id']=='frame_002850')
    overhead=next(x for x in backgrounds if x['id']=='frame_007000')
    field=json.loads(FIELD.read_text(encoding='utf-8'))['records'];known=build_queue(ROOT)
    for row in review:
        for p in (row['source_frame'],row['source_video']):
            if source_role(ROOT,ROOT/p,known)[0]=='regression':raise ValueError('Reviewed source restricted to regression: '+p)
    if source_role(ROOT,ROOT/white['parent_image'],known)[0]=='regression':raise ValueError('White-back background is regression')
    if source_role(ROOT,ROOT/overhead['parent_image'],known)[0]=='regression':raise ValueError('Overhead background is regression')
    for entry in field:
        for p in (entry['path'],entry['source_frame'],entry['source_video']):
            if source_role(ROOT,ROOT/p,known)[0]=='regression':raise ValueError('Historical source is frozen regression: '+p)
    people,vehicles=bike.sources(include_people=True);scene=bike.scene
    historical=[]
    for entry in field:
        if entry['class']=='fallen':
            source={**entry,'historical_reuse':'used in field_validation_candidates_200_20261110; '
                    'now shared training source and not independent V4 development/test'}
            historical.append((source,clean_rgba(scene.assets.load_rgba(ROOT/entry['path']))))
    vehicles['fallen']=[(s,clean_rgba(im)) for s,im in vehicles['fallen']]
    vehicles['upright']=[(s,clean_rgba(im)) for s,im in vehicles['upright']]
    for folder in ('images/train','labels/train','masks'):(out/folder).mkdir(parents=True,exist_ok=True)
    records=[];counts=Counter();skipped=Counter()
    for index in range(args.count):
        rng=random.Random(SEED+index);plan=PLANS[index%len(PLANS)];tone=tuple(TONES)[(index//len(PLANS))%3]
        real=review[(index//len(PLANS))%len(review)] if plan not in ('white_back_coexist','overhead_multiple_fallen') else None
        nonreal=white if plan=='white_back_coexist' else overhead
        background=ROOT/(real['image'] if real else nonreal['image'])
        originals=real['boxes'] if real else []
        classes=['fallen','fallen']+(['fallen'] if plan!='overhead_multiple_fallen' and (index//18)%2 else [])
        if plan in ('upright_multiple_fallen','white_back_coexist'):classes.insert(0,'upright')
        result=None
        for attempt in range(60):
            frame=read_image(background);height,width=frame.shape[:2];actors=[]
            try:
                # Flat rows follow the observed low camera view; values are pixel fits.
                foot=rng.uniform(310,350) if real else rng.uniform(385,440)
                ref=rng.uniform(96,128) if real else rng.uniform(80,100) if plan=='overhead_multiple_fallen' else rng.uniform(95,125)
                spacing=ref*rng.uniform(.70,.92)
                center=rng.uniform(150,width-150) if real else rng.uniform(470,530) if plan=='overhead_multiple_fallen' else rng.uniform(175,370)
                used=set()
                for j,category in enumerate(classes):
                    pool=historical if plan=='overhead_multiple_fallen' else vehicles[category]
                    choices=[item for item in pool if item[0]['path'] not in used]
                    if not choices:raise ValueError('not enough distinct foreground views')
                    source,sprite=rng.choice(choices);used.add(source['path'])
                    x=center+(j-(len(classes)-1)/2)*spacing
                    if plan=='edge_truncated' and j==0:x=rng.uniform(30,40)
                    actors.append(scene.random_actor(rng,'vehicle',category,sprite,source,x,foot+j*2,ref))
                if plan=='person_occludes':
                    category=rng.choice(('community','noncommunity'));source,sprite=rng.choice(people[category])
                    actor=actors[-1]
                    actors.append(scene.random_actor(rng,'person',category,sprite,rel(source),
                        actor['foot_x']+rng.uniform(-.18,.18)*ref,foot+rng.choice((-5,5)),ref*.95))
                actors.sort(key=lambda a:a['foot_y'])
                for actor in actors:scene.draw_shadow(frame,actor)
                full=[place_clipped(frame,a['sprite'],a['foot_x'],a['foot_y']) for a in actors]
                # Do not hide any part of a source target whose segmentation is not known.
                if any(np.any(mask[b['xyxy'][1]:b['xyxy'][3],b['xyxy'][0]:b['xyxy'][2]]) for mask in full for b in originals):
                    raise ValueError('generated foreground overlaps an original target')
                visible=visible_masks(full)
                if any(a['kind']=='vehicle' and (v.sum()<120 or v.sum()/m.sum()<.70)
                       for a,m,v in zip(actors,full,visible)):raise ValueError('ambiguous vehicle occlusion')
                person_overlap=any(np.any(full[i]&full[j]) for i,a in enumerate(actors) for j,b in enumerate(actors)
                    if a['kind']=='person' and b['kind']=='vehicle')
                if plan=='person_occludes' and not person_overlap:raise ValueError('person overlap absent')
                whiteback_overlap=bool(plan=='white_back_coexist' and any(
                    np.any(mask[t:b,l:r]) for mask in full for l,t,r,b in ((14,278,88,426),(135,276,205,416))))
                if plan=='white_back_coexist' and not whiteback_overlap:raise ValueError('white-back overlap absent')
                edge=any(np.any(m[:,0]) or np.any(m[:,-1]) for a,m in zip(actors,full) if a['kind']=='vehicle')
                if plan=='edge_truncated' and not edge:raise ValueError('no visible truncated vehicle')
                result=frame,actors,full,visible,person_overlap,whiteback_overlap,edge,attempt+1;break
            except ValueError:continue
        if result is None:
            skipped[plan]+=1;continue
        frame,actors,full,visible,person_overlap,whiteback_overlap,edge,attempts=result
        value,sat=TONES[tone]
        if tone!='original':frame=apply_tone(frame,value_gain=value,saturation_gain=sat)
        height,width=frame.shape[:2];name=f'parking_{index:06d}'
        image,label=out/'images/train'/(name+'.png'),out/'labels/train'/(name+'.txt');write_image(image,frame)
        objects=[];lines=[]
        for obj in originals:
            box=obj['xyxy'];cls={'upright':0,'fallen':1}[obj['class']];lines.append(scene.yolo_row(cls,box,width,height))
            objects.append({'class':obj['class'],'bbox_xyxy':box,'truth':'reviewed_real_parent','truncated':obj['truncated']})
        for j,(actor,mask,shown) in enumerate(zip(actors,full,visible)):
            if actor['kind']!='vehicle':continue
            yy,xx=np.nonzero(shown);box=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1]
            cls={'upright':0,'fallen':1}[actor['category']];lines.append(scene.yolo_row(cls,box,width,height))
            mask_out=out/'masks'/f'{name}_{j:02d}.png';write_image(mask_out,shown.astype(np.uint8)*255)
            objects.append({'class':actor['category'],'bbox_xyxy':box,'truth':'visible_mask_candidate',
                'visible_mask':rel(mask_out),'foreground_source':actor['source'],
                'visible_fraction_in_image':float(shown.sum()/mask.sum()),'foot_xy':[actor['foot_x'],actor['foot_y']],
                'truncated':bool(mask[:,0].any() or mask[:,-1].any())})
        label.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        records.append({'image':rel(image),'label':rel(label),'split':'train_candidate','plan':plan,
            'background_image':rel(background),'background_parent':real['source_frame'] if real else nonreal['parent_image'],
            'background_source_video':real['source_video'] if real else nonreal['source_video'],
            'background_source_pts_seconds':real['source_pts_seconds'] if real else nonreal['source_time_seconds'],
            'original_review':rel(REVIEW) if real else rel(BACKGROUNDS),'image_size':[width,height],
            'objects':objects,'other_actors':[{'kind':a['kind'],'source':a['source']} for a in actors if a['kind']!='vehicle'],
            'person_bike_mask_overlap':person_overlap,'white_back_bbox_overlap_candidate':whiteback_overlap,
            'edge_truncation':edge,'tone':tone,
            'tone_parameters':{'value_gain':value,'saturation_gain':sat},'attempts':attempts,
            'generation_method':'real_scene_composite','label_review':'original_review_plus_visible_masks_candidate'})
        counts['plan_'+plan]+=1;counts['tone_'+tone]+=1;counts['vehicles_'+str(len(objects))]+=1
        for obj in objects:counts['instances_'+obj['class']]+=1
        if edge:counts['edge_truncation']+=1
        if person_overlap:counts['person_overlap']+=1
        if whiteback_overlap:counts['white_back_overlap_candidate']+=1
    manifest={'status':'held_scene_geometry','seed':SEED,'requested':args.count,'planned_proportions':{x:1/len(PLANS) for x in PLANS},
        'counts':dict(counts),'skipped':dict(skipped),'records':records,
        'scope':'Four real low-angle frames from one video, plus two vehicle-camera backgrounds from another video. '
                'Foreground sprites are reused; fallen_005/006/007 participated in historical field validation '
                'and are now shared training sources, not independent V4 development or test. '
                'No claim of independently collected samples or model results.'}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':rel(out),'requested':args.count,'counts':dict(counts),'skipped':dict(skipped)},ensure_ascii=False))


def main():
    from lab.training_common.production_guard import generation_guard
    from e_bike_queue_inputs import queue_inputs
    from lab.scripts.data import v4_source_gate as gate
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True); parser.add_argument("--count", type=int, default=30)
    opts = parser.parse_args()
    out = (ROOT / opts.output).resolve()
    manifests = [REVIEW, BACKGROUNDS, FIELD] + [bike.SELECTION]
    manifests.append(bike.scene.VEHICLE_SELECTION)
    people, vehicles = bike.sources(include_people=True)
    media = set()
    for group in people.values():
        media.update(Path(source).resolve() for source, _ in group)
    for group in vehicles.values():
        for source, _ in group:
            for key in ('path', 'source_frame', 'source_video'):
                media.add((ROOT / source[key]).resolve())
    suffixes = {'.png','.jpg','.jpeg','.bmp','.webp','.ppm','.mp4','.mov','.avi','.mkv'}
    def discover(value):
        if isinstance(value, dict):
            for child in value.values(): discover(child)
        elif isinstance(value, list):
            for child in value: discover(child)
        elif isinstance(value, str) and Path(value).suffix.lower() in suffixes:
            path = (ROOT / value).resolve()
            if path.is_relative_to(ROOT) and path.is_file(): media.add(path)
    for path in manifests:
        discover(json.loads(path.read_text(encoding='utf-8')))
    for row in json.loads(REVIEW.read_text(encoding='utf-8'))['rows']:
        media.update((ROOT / row[key]).resolve() for key in ('image', 'source_frame', 'source_video'))
    for row in json.loads(BACKGROUNDS.read_text(encoding='utf-8'))['backgrounds']:
        media.update((ROOT / row[key]).resolve() for key in ('image', 'parent_image'))
    for row in json.loads(FIELD.read_text(encoding='utf-8'))['records']:
        media.update((ROOT / row[key]).resolve() for key in ('path', 'source_frame', 'source_video'))
    queue = build_queue(ROOT)
    qm, qp = queue_inputs(ROOT, queue, media)
    media.update(qm)
    procedural = set(qp) | set(manifests)
    code = {Path(__file__).resolve(), Path(bike.__file__).resolve(), Path(bike.scene.__file__).resolve(), Path(bike.scene.assets.__file__).resolve(),
            Path(gate.__file__).resolve(), ROOT / 'lab/scripts/media/media_index.py', Path(__file__).resolve().parent / 'e_bike_queue_inputs.py',
            ROOT / 'lab/training_common/camera_tone.py', ROOT / 'datasets/scripts/common/perspective.py', ROOT / 'datasets/scripts/common/composite.py', ROOT / 'datasets/scripts/common/annotations.py',
            ROOT / 'lab/training_common/engine/data.py', ROOT / 'lab/training_common/engine/config.py',
            ROOT / 'lab/training_common/train.py', ROOT / 'lab/training_common/production_guard.py',
            Path(__file__).resolve().parent / 'preview_person_occlusions.py',
            Path(__file__).resolve().parent / 'compose_v4_multibike_parking.py'}
    params = {key: str(value) if isinstance(value, Path) else value for key, value in vars(opts).items()}
    with generation_guard('e_bike.compose_v4_multibike_parking:' + out.relative_to(ROOT).as_posix(), sorted(media), [out], params,
                          procedural_inputs=sorted(procedural), code_paths=sorted(code), root=ROOT) as run:
        if run.reused:
            return
        return _produce()


if __name__=='__main__':main()
