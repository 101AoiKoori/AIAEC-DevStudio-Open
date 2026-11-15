"""Select bounded legacy fire/name scenes and crop them to camera aspect ratio.

Both tasks are training candidates only. Six reviewed background frames share
one video; foreground renders share one Blender scene and are not new captures.
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'lab/training_common'))
sys.path.insert(0, str(ROOT))
from camera_tone import apply_tone, read_image, write_image  # noqa: E402
from lab.training_common.production_guard import generation_guard

SOURCE = ROOT / 'datasets/exports/fire/scene_composite_candidates_4000_20261110'
OUT = ROOT / 'datasets/exports/fire/v4_legacy_supplement_20261112'
BACKGROUND = ROOT / 'datasets/annotated/fire/real_background_candidates_20261110/manifest.json'
REVIEW = ROOT / 'datasets/annotated/fire/v4_legacy_background_review_20261112/review.json'
SEED = 20261119


def rel(path):
    return path.relative_to(ROOT).as_posix()


def crop_labels(text, task, rng):
    """Return a 640x480 crop and boxes; reject partially clipped task targets."""
    boxes=[]
    for line in text.splitlines():
        fields=line.split()
        if len(fields)!=5 or int(fields[0])!=0:
            raise ValueError('Unexpected source label')
        x,y,w,h=map(float,fields[1:]);a,b,c,d=(x-w/2)*640,(y-h/2)*640,(x+w/2)*640,(y+h/2)*640
        if not (0<=a<c<=640.0001 and 0<=b<d<=640.0001):
            raise ValueError('Source box outside image')
        if task=='fire' and (c-a<14 or d-b<20):
            return None
        boxes.append((a,b,c,d))
    lower=max(0, max((int(np.ceil(d-480)) for _,_,_,d in boxes),default=0))
    upper=min(160, min((int(np.floor(b)) for _,b,_,_ in boxes),default=160))
    if lower>upper:
        return None
    top=rng.randint(lower,upper)
    output=[]
    for a,b,c,d in boxes:
        output.append(f'0 {(a+c)/1280:.8f} {(b+d-2*top)/960:.8f} {(c-a)/640:.8f} {(d-b)/480:.8f}')
    return top, output


def _run(source_refs):
    if OUT.exists():
        raise ValueError('Output exists; use a new version')
    review=json.loads(REVIEW.read_text(encoding='utf-8'))
    backgrounds=json.loads(BACKGROUND.read_text(encoding='utf-8'))['samples']
    if review['background_indices']!=list(range(1,7)) or len(backgrounds)!=6:
        raise ValueError('Six reviewed legacy backgrounds required')
    rows=[json.loads(line) for line in (SOURCE/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    rng=random.Random(SEED); records=[]; counts=Counter(); skipped=Counter()
    for task, quotas in (('fire',{0:80,1:80,2:80,3:80}),('name',{False:80,True:240})):
        (OUT/task/'images/train').mkdir(parents=True)
        (OUT/task/'labels/train').mkdir(parents=True)
        for state,quota in quotas.items():
            group=[r for r in rows if (r['fire_bucket'] if task=='fire' else r['name_positive'])==state]
            rng.shuffle(group);accepted=0
            for row in group:
                if accepted>=quota:break
                source_image=SOURCE/task/'images/train'/(row['id']+'.jpg')
                source_label=SOURCE/task/'labels/train'/(row['id']+'.txt')
                label=source_label.read_text(encoding='utf-8')
                crop=crop_labels(label,task,rng)
                if crop is None:
                    skipped[task+'_tiny_or_partial']+=1;continue
                top, lines=crop
                if task=='fire' and ((state < 3 and len(lines) != state) or (state == 3 and len(lines) < 3)):
                    raise ValueError('Fire source label count disagrees with scene bucket')
                if task=='name' and bool(lines)!=row['name_positive']:
                    raise ValueError('Name source label disagrees with scene state')
                image=read_image(source_image)
                if image.shape[:2]!=(640,640):raise ValueError('Unexpected source dimensions')
                image=image[top:top+480,:,:]
                profile=('original','mild','moderate')[accepted%3]
                value,saturation={'original':(1.,1.),'mild':(.9,.85),'moderate':(.7,.6)}[profile]
                if profile!='original' and float(image.mean())>=55:
                    image=apply_tone(image,value_gain=value,saturation_gain=saturation)
                elif profile!='original':
                    profile,value,saturation='already_dark',1.,1.
                stem=f'{task}_{len(records):06d}';img=OUT/task/'images/train'/(stem+'.png');txt=OUT/task/'labels/train'/(stem+'.txt')
                write_image(img,image);txt.write_text(('\n'.join(lines)+'\n') if lines else '',encoding='utf-8')
                bg=backgrounds[row['background_index']-1]
                records.append({'task':task,'image':rel(img),'label':rel(txt),'split':'train_candidate',
                    'source_image':rel(source_image),'source_label':rel(source_label),
                    'background':bg,'background_review':rel(REVIEW),'foreground_render':row['foreground_source'] if task=='fire' or lines else None,
                    'foreground_source_group':row['foreground_source_group'],'scenario':state,
                    'image_size':[640,480],'parent_image_size':[640,640],'crop_xyxy':[0,top,640,top+480],
                    'generation_method':'legacy_real_scene_composite_then_crop_and_tone',
                    'source_references':source_refs[rel(source_image)],
                    'tone_profile':profile,'value_gain':value,'saturation_gain':saturation,
                    'label_review':'AI reviewed background and prior generated labels; batch sample review pending'})
                counts[task+'|'+str(state)]+=1;counts[task+'|tone|'+profile]+=1;accepted+=1
            if accepted<quota:raise ValueError(f'{task} {state}: only {accepted}/{quota}')
    manifest={'status':'train_candidates_not_deliverable','seed':SEED,'counts':dict(counts),'skipped':dict(skipped),
              'scope':'Shared six-frame single-video background and one Blender mother scene. No independent capture, validation, test or model result.',
              'source_manifest':rel(SOURCE/'manifest.jsonl'),'records':records}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':rel(OUT),'counts':dict(counts),'skipped':dict(skipped)},ensure_ascii=False))


def main():
    manifest=SOURCE/'manifest.jsonl'
    rows=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
    backgrounds=json.loads(BACKGROUND.read_text(encoding='utf-8'))['samples']
    media={SOURCE/task/'images/train'/(row['id']+'.jpg') for row in rows for task in ('fire','name')}
    media.update(ROOT/bg['image'] for bg in backgrounds if bg.get('image'))
    media.update(ROOT/bg['source_video'] for bg in backgrounds if bg.get('source_video'))
    render_root=ROOT/'datasets/exports/fire/blender_candidates_160_20261110'
    media.update(render_root/row['foreground_source'] for row in rows if row.get('foreground_source'))
    labels={SOURCE/task/'labels/train'/(row['id']+'.txt') for row in rows for task in ('fire','name')}
    with generation_guard('fire.prepare_v4_legacy_supplement',sorted(media),[OUT],{'seed':SEED},
                          code_paths=[Path(__file__),ROOT/'lab/training_common/camera_tone.py'],
                          procedural_inputs=[manifest,BACKGROUND,REVIEW,*sorted(labels)],root=ROOT) as run:
        if run.reused:
            print('已复用既有候选');return
        _run(run.source_refs)


if __name__=='__main__':
    main()
