"""Validate paired fire/name candidate labels, transforms and source grouping."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from lab.training_common.production_guard import generation_guard
DATA=ROOT/'datasets/exports/fire/v4_distractor_composites_20261112'


def _run(source_refs):
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf8'));errors=[];counts=Counter();renders=defaultdict(set);parents=defaultdict(set)
    examples={}
    for row in manifest['records']:
        image=cv2.imdecode(np.fromfile(str(ROOT/row['image']),np.uint8),cv2.IMREAD_COLOR)
        if image is None or image.shape!=(480,640,3):errors.append(row['image']+': bad image');continue
        split=row['split'];parents[split].add(row['background_parent'])
        if row['source_render']:renders[split].add(row['source_render'])
        for task,kind in (('fire','fire_sticker'),('name','name_sign')):
            lines=(ROOT/row['labels'][task]).read_text(encoding='utf8').splitlines()
            objects=[o for o in row['objects'] if o['kind']==kind]
            if len(lines)!=len(objects):errors.append(row['image']+': label count '+task);continue
            for line,obj in zip(lines,objects):
                fields=line.split()
                if len(fields)!=5:errors.append(row['image']+': malformed '+task);continue
                c,x,y,w,h=map(float,fields);a,b,d,e=obj['bbox_xyxy'];expected=((a+d)/1280,(b+e)/960,(d-a)/640,(e-b)/480)
                if c!=0 or any(abs(i-j)>1e-5 for i,j in zip((x,y,w,h),expected)) or not (0<=a<d<=640 and 0<=b<e<=480):errors.append(row['image']+': bad box '+task)
                if kind=='fire_sticker' and (d-a<14 or e-b<20):errors.append(row['image']+': tiny fire labeled')
                transform=row['transform'];sa,sb,sd,se=obj['source_bbox_xyxy'];mapped=[transform['scale']*sa+transform['tx'],transform['scale']*sb+transform['ty'],transform['scale']*sd+transform['tx'],transform['scale']*se+transform['ty']]
                if any(abs(i-j)>1e-5 for i,j in zip((a,b,d,e),mapped)):errors.append(row['image']+': source projection mismatch')
                cv2.rectangle(image,(round(a),round(b)),(round(d),round(e)),(0,0,255) if task=='fire' else (0,255,0),2)
        if row['state'].startswith('red_bin_') != bool(row['bin_distractor']):errors.append(row['image']+': bin state mismatch')
        counts[(split,row['state'],row['tone'])]+=1
        key=(split,row['state'])
        if row['tone']=='normal' and key not in examples:
            im=cv2.resize(image,(320,240));cv2.putText(im,split+' '+row['state'],(3,16),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,0,255),1);examples[key]=im
    overlaps={}
    for a,b in (('train','val'),('train','test'),('val','test')):
        overlaps[a+'|'+b]={'background_parents':sorted(parents[a]&parents[b]),'foreground_renders':sorted(renders[a]&renders[b])}
        if parents[a]&parents[b] or renders[a]&renders[b]:errors.append('source crossed splits '+a+'/'+b)
    legacy=ROOT/'datasets/exports/fire/v4_legacy_supplement_20261112/manifest.json'
    old=json.loads(legacy.read_text(encoding='utf8'))['records']
    legacy_sources={'datasets/exports/fire/blender_candidates_160_20261110/'+r['foreground_render'] for r in old if r['foreground_render']}
    legacy_overlap={split:sorted(source&legacy_sources) for split,source in renders.items()}
    ordered=list(examples.values())
    while len(ordered)%4:ordered.append(np.zeros_like(ordered[0]))
    canvas=np.vstack([np.hstack(ordered[i:i+4]) for i in range(0,len(ordered),4)])
    contact=ROOT/'runs/fire_sticker/v4_distractor_composites_stratified.png';contact.parent.mkdir(parents=True,exist_ok=True);cv2.imencode('.png',canvas)[1].tofile(str(contact))
    held=sum(row['status']=='held_scene_geometry' for row in manifest['records'])
    result={'status':('structural_pass_with_geometry_holds' if held else 'structural_pass_visual_review_pending') if not errors else 'failed',
            'held_scene_geometry':held,'images':len(manifest['records']),
            'counts':{'|'.join(k):v for k,v in counts.items()},'errors':errors[:100],'total_errors':len(errors),
            'distinct_backgrounds':{k:len(v) for k,v in parents.items()},'distinct_foreground_renders':{k:len(v) for k,v in renders.items()},
            'cross_split_overlap':overlaps,'shared_with_legacy_train':{k:len(v) for k,v in legacy_overlap.items()},
            'legacy_shared_render_paths':legacy_overlap,'visual_sample_count':len(examples),'visual_evidence':contact.relative_to(ROOT).as_posix(),
            'scope':'Source tracking and label geometry only; visual plausibility, background semantic completeness and independence remain candidate assessments. Do not merge split sources shared with legacy train.',
            'indexed_source_references':source_refs}
    (DATA/'qa.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('legacy_shared_render_paths','counts')},ensure_ascii=False))
    if errors:raise SystemExit(1)


def main():
    manifest_path=DATA/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    legacy=ROOT/'datasets/exports/fire/v4_legacy_supplement_20261112/manifest.json'
    media={ROOT/row['image'] for row in manifest['records']}
    for row in manifest['records']:
        media.update(ROOT/row[key] for key in ('background_parent','background_video','source_render') if row.get(key))
        bin_source=row.get('bin_distractor') or {}
        media.update(ROOT/bin_source[key] for key in ('source_render','source_instance_mask') if bin_source.get(key))
    labels=sorted({ROOT/path for row in manifest['records'] for path in row['labels'].values() if path})
    procedural=[manifest_path,legacy,*labels]
    contact=ROOT/'runs/fire_sticker/v4_distractor_composites_stratified.png'
    with generation_guard('fire.check_v4_distractor_scenes',sorted(media),[contact,DATA/'qa.json'],{},
                          code_paths=[Path(__file__)],procedural_inputs=procedural,root=ROOT) as run:
        if run.reused:
            print('已复用已有检查结果');return
        _run(run.source_refs)


if __name__=='__main__':main()
