"""Compose multi-bike scenes using per-frame real bike size and ground anchors."""

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
from compose_v4_multibike_parking import clean_rgba,place_clipped  # noqa: E402
import synthesize_vehicle_scene as bike  # noqa: E402
from datasets.scripts.common.composite import visible_masks  # noqa: E402

REVIEW=ROOT/'datasets/annotated/e_bike/v4_low_angle_multibike_review_20261112/review_manifest.json'
PLANS=('multiple_fallen','upright_multiple_fallen','edge_truncated','person_occludes')
TONES={'original':(1.,1.),'mild':(.9,.85),'moderate':(.7,.6)}
SEED=20261120


def rel(path):return path.relative_to(ROOT).as_posix()


def references(row):
    result={}
    for category in ('upright','fallen'):
        boxes=[x for x in row['boxes'] if x['class']==category]
        if not boxes:raise ValueError('Frame lacks a real size reference for '+category)
        # A cropped reference supplies a lower bound, not an estimated full size.
        candidates=[x for x in boxes if not x['truncated']] or boxes
        sizes=[x['xyxy'][2]-x['xyxy'][0] if category=='fallen' else x['xyxy'][3]-x['xyxy'][1]
               for x in candidates]
        result[category]={'pixels':float(np.median(sizes)),
            'foot_y':float(np.median([x['xyxy'][3] for x in candidates])),
            'measure':'bbox_width' if category=='fallen' else 'bbox_height',
            'source_boxes':[x['xyxy'] for x in candidates],
            'reference_truncated':any(x['truncated'] for x in candidates)}
    return result


def resize_visible(actor,measure,target):
    sprite=actor['sprite'];alpha=sprite[:,:,3]>127;y,x=np.nonzero(alpha)
    if len(x)<100:raise ValueError('empty actor')
    sprite=sprite[y.min():y.max()+1,x.min():x.max()+1]
    height,width=sprite.shape[:2];current=width if measure=='bbox_width' else height
    scale=target/current
    sprite=cv2.resize(sprite,(max(1,round(width*scale)),max(1,round(height*scale))),
                      interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
    actor['sprite']=sprite
    return actor


def _produce():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--count',type=int,default=48);ap.add_argument('--start-index',type=int,default=0);args=ap.parse_args()
    out=(ROOT/args.output).resolve()
    if not out.is_relative_to((ROOT/'datasets/exports/e_bike').resolve()) or out.exists():
        raise ValueError('Output must be a new e-bike export')
    if not 1<=args.count<=5000:raise ValueError('count must be in 1..5000')
    if args.start_index<0:raise ValueError('start-index must be nonnegative')
    rows=json.loads(REVIEW.read_text(encoding='utf-8'))['rows'];known=build_queue(ROOT)
    if len(rows)!=4:raise ValueError('Expected four reviewed real frames')
    for row in rows:
        if row['split']!='train_candidate' or 'complete_candidate' not in row['label_status']:
            raise ValueError('Incomplete real-frame supervision')
        for path in (row['source_frame'],row['source_video']):
            if source_role(ROOT,ROOT/path,known)[0]=='regression':raise ValueError('Source is frozen regression: '+path)
    people,vehicles=bike.sources(include_people=True);scene=bike.scene
    for category in vehicles:
        vehicles[category]=[(source,clean_rgba(image)) for source,image in vehicles[category]]
    for folder in ('images/train','labels/train','masks'):(out/folder).mkdir(parents=True,exist_ok=True)
    records=[];counts=Counter();skipped=Counter()
    for index in range(args.start_index,args.start_index+args.count):
        rng=random.Random(SEED+index);plan=PLANS[index%4];row=rows[(index//4)%len(rows)]
        tone=tuple(TONES)[(index//16)%3];ref=references(row)
        classes=['fallen','fallen']+(['fallen'] if (index//48)%2 else [])
        if plan=='upright_multiple_fallen':classes.insert(0,'upright')
        result=None
        for attempt in range(100):
            frame=read_image(ROOT/row['image']);height,width=frame.shape[:2]
            try:
                foot=rng.uniform(max(v['foot_y'] for v in ref.values())+32,height-14)
                center=rng.uniform(170,width-170)
                # Space targets using the real lower bound at the proposed depth.
                fall_ref=ref['fallen'];min_width=fall_ref['pixels']*(1+.35*(foot-fall_ref['foot_y'])/height)
                spacing=min_width*rng.uniform(.86,1.08);actors=[];used=set()
                for j,category in enumerate(classes):
                    pool=[entry for entry in vehicles[category] if entry[0]['path'] not in used]
                    source,sprite=rng.choice(pool);used.add(source['path'])
                    x=center+(j-(len(classes)-1)/2)*spacing
                    reference=ref[category];actor_foot=foot+j*2
                    minimum=reference['pixels']*(1+.35*(actor_foot-reference['foot_y'])/height)
                    target=minimum*rng.uniform(1.04,1.18)
                    if plan=='edge_truncated' and j==0:x=target*rng.uniform(.34,.43)
                    actor=scene.random_actor(rng,'vehicle',category,sprite,source,x,actor_foot,120)
                    resize_visible(actor,reference['measure'],target)
                    # The unoccluded projected sprite must not shrink at closer depth.
                    alpha=actor['sprite'][:,:,3]>127;yy,xx=np.nonzero(alpha)
                    actual=(xx.max()-xx.min()+1) if category=='fallen' else (yy.max()-yy.min()+1)
                    if actual+1e-4<minimum:raise ValueError('projected target below reference')
                    actor['projected_dimension_px']=int(actual);actor['minimum_dimension_px']=float(minimum)
                    actor['reference']=reference;actors.append(actor)
                if plan=='person_occludes':
                    category=rng.choice(('community','noncommunity'));source,sprite=rng.choice(people[category]);near=actors[-1]
                    person_height=ref['upright']['pixels']*(1+.35*(foot-ref['upright']['foot_y'])/height)*1.1
                    actors.append(scene.random_actor(rng,'person',category,sprite,rel(source),
                        near['foot_x']+rng.uniform(-.12,.12)*min_width,foot+rng.choice((-5,5)),person_height))
                actors.sort(key=lambda a:a['foot_y'])
                for actor in actors:scene.draw_shadow(frame,actor)
                full=[place_clipped(frame,a['sprite'],a['foot_x'],a['foot_y']) for a in actors]
                if any(np.any(mask[b['xyxy'][1]:b['xyxy'][3],b['xyxy'][0]:b['xyxy'][2]]) for mask in full for b in row['boxes']):
                    raise ValueError('new actor overlaps original box')
                visible=visible_masks(full)
                if any(a['kind']=='vehicle' and (v.sum()<120 or v.sum()/m.sum()<.70)
                       for a,m,v in zip(actors,full,visible)):raise ValueError('ambiguous vehicle visibility')
                for actor,mask,shown in zip(actors,full,visible):
                    if actor['kind']!='vehicle' or mask[:,0].any() or mask[:,-1].any():
                        continue
                    y,x=np.nonzero(shown)
                    dimension=(x.max()-x.min()+1) if actor['category']=='fallen' else (y.max()-y.min()+1)
                    if dimension<actor['reference']['pixels']:
                        raise ValueError('visible nearer target smaller than real reference')
                person_overlap=any(np.any(full[i]&full[j]) for i,a in enumerate(actors) for j,b in enumerate(actors)
                    if a['kind']=='person' and b['kind']=='vehicle')
                if plan=='person_occludes' and not person_overlap:raise ValueError('no person/bike overlap')
                edge=any(np.any(m[:,0]) or np.any(m[:,-1]) for a,m in zip(actors,full) if a['kind']=='vehicle')
                if plan=='edge_truncated' and not edge:raise ValueError('no edge truncation')
                result=frame,actors,full,visible,person_overlap,edge,attempt+1;break
            except ValueError:continue
        if result is None:skipped[plan]+=1;continue
        frame,actors,full,visible,person_overlap,edge,attempts=result
        value,saturation=TONES[tone]
        if tone!='original':frame=apply_tone(frame,value_gain=value,saturation_gain=saturation)
        height,width=frame.shape[:2];name=f'parking_{index:06d}'
        image,label=out/'images/train'/(name+'.png'),out/'labels/train'/(name+'.txt');write_image(image,frame)
        objects=[];lines=[]
        for obj in row['boxes']:
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
                'angles_degrees':actor['angles_degrees'],'relative_exposure':actor['relative_exposure'],
                'projected_sprite_size':[actor['sprite'].shape[1],actor['sprite'].shape[0]],
                'projected_dimension_px':actor['projected_dimension_px'],'minimum_dimension_px':actor['minimum_dimension_px'],
                'scale_reference':actor['reference'],'truncated':bool(mask[:,0].any() or mask[:,-1].any())})
        label.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        records.append({'image':rel(image),'label':rel(label),'split':'train_candidate','plan':plan,
            'background_image':row['image'],'background_parent':row['source_frame'],
            'background_source_video':row['source_video'],'background_source_pts_seconds':row['source_pts_seconds'],
            'original_review':rel(REVIEW),'image_size':[width,height],'objects':objects,
            'other_actors':[{'kind':a['kind'],'source':a['source']} for a in actors if a['kind']!='vehicle'],
            'person_bike_mask_overlap':person_overlap,'edge_truncation':edge,'tone':tone,
            'tone_parameters':{'value_gain':value,'saturation_gain':saturation},'attempts':attempts,
            'scale_policy':'per-frame real bbox plus nondecreasing fitted size with lower-image foot position',
            'generation_method':'real_scene_composite','label_review':'original_review_plus_visible_masks_candidate'})
        counts['plan_'+plan]+=1;counts['tone_'+tone]+=1;counts['vehicles_'+str(len(objects))]+=1
        for obj in objects:counts['instances_'+obj['class']]+=1
        if edge:counts['edge_truncation']+=1
        if person_overlap:counts['person_overlap']+=1
    manifest={'status':'draft_QA_pending','seed':SEED,'requested':args.count,'start_index':args.start_index,
        'planned_proportions':{x:.25 for x in PLANS},'counts':dict(counts),'skipped':dict(skipped),'records':records,
        'scope':'Four low-camera frames share one source video and physical parking arrangement. '
                'Foreground side views are shared materials; per-frame real boxes and contact positions set '
                'nondecreasing projected size. The slope is fitted, not a measured camera calibration. '
                'No independent collection, validation/test or model result.'}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':rel(out),'requested':args.count,'counts':dict(counts),'skipped':dict(skipped)},ensure_ascii=False))


def main():
    from lab.training_common.production_guard import generation_guard
    from e_bike_queue_inputs import queue_inputs
    from lab.scripts.data import v4_source_gate as gate
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True); parser.add_argument("--count", type=int, default=48); parser.add_argument("--start-index", type=int, default=0)
    opts = parser.parse_args()
    out = (ROOT / opts.output).resolve()
    manifests = [REVIEW] + [bike.SELECTION]
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
    with generation_guard('e_bike.compose_v4_scaled_parking:' + out.relative_to(ROOT).as_posix(), sorted(media), [out], params,
                          procedural_inputs=sorted(procedural), code_paths=sorted(code), root=ROOT) as run:
        if run.reused:
            return
        return _produce()


if __name__=='__main__':main()
