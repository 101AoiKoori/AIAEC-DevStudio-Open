"""Render reusable physical paper-on/paper-off counterfactual building layers."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from datasets.scripts.building_name_paper import full_plan as full
from datasets.scripts.building_name_paper import hybrid_sample as hybrid
from datasets.scripts.building_name_paper.render_hybrid_sample import prepare_scene,layout
from datasets.scripts.common.blender_masks import masks
from datasets.scripts.common.blender_photo import aim_area,save_render
from datasets.scripts.common.config import source_path
from lab.training_common.production_guard import generation_guard


def write(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def fit(scene,camera,groups,baseline,plate,state,recipe):
    errors=[]
    values=list(dict.fromkeys([state['parameters']['camera_height_m'],*recipe['camera_height_fallback_m']]))
    for height in values:
        candidate=deepcopy(state)
        candidate['parameters']['camera_height_m']=height
        try:
            fitted,point,main=layout(scene,camera,groups,baseline,plate,candidate)
            fitted['height_fallback_from_m']=state['parameters']['camera_height_m'] if height!=state['parameters']['camera_height_m'] else None
            return fitted,point,main
        except ValueError as exc:
            errors.append(str(exc))
    raise ValueError('; '.join(errors))


def capture(scene,camera,groups,baseline,receiver,key,plate,state,recipe):
    stem=state['id'];out=full.LAYERS
    fit_record,point,main=fit(scene,camera,groups,baseline,plate,state,recipe)
    light=recipe['lighting']
    scene.cycles.seed=state['parameters']['seed']
    factor=(.85,1.,1.15)[state['parameters']['seed']%3]
    aim_area(key,point+Vector(light['area_offset_m']),point,light['area_energy_w']*factor,light['area_size_m'])
    receiver.hide_render=False
    receiver.visible_camera=False
    foreground=f'foreground/{stem}_paper_on.png'
    save_render(scene,out/foreground)
    active=state['target']['buildings']
    for b in active:
        groups[b]['paper'].hide_render=groups[b]['text'].hide_render=True
    empty=f'paper_off/{stem}.png'
    save_render(scene,out/empty)
    receiver.visible_camera=True
    for obj in main:
        obj.visible_camera=False
    shadowed=f'lighting/{stem}_shadowed.png';save_render(scene,out/shadowed)
    for obj in main:obj.hide_render=True
    lit=f'lighting/{stem}_lit.png';save_render(scene,out/lit)
    for obj in main:
        obj.hide_render=False
        obj.visible_camera=True
    receiver.hide_render=True
    visible,instances=masks(scene,camera,{b:[groups[b]['paper'],groups[b]['text']] for b in active},
        out,stem,'paper',palette=hybrid.PALETTE)
    accepted=all(obj['in_frame'] and obj['visible_pixels']>=recipe['minimum_visible_paper_pixels']
                 and obj['visible_fraction'] is not None and obj['visible_fraction']>=.70 for obj in instances)
    w,h=plate['image_size']
    truth=[]
    for obj in instances:
        box=obj['candidate_xyxy_normalized']
        truth.append({'building_id':obj['bin'],'xyxy':[round(v*d) for v,d in zip(box,(w,h,w,h))] if box else None,
                      'visibility':obj,'text':groups[obj['bin']]['text'].data.body})
    return {'id':stem,'status':'accepted' if accepted else 'rejected_visibility','plate_id':plate['id'],
        'split':plate['split'],'source_photo':plate['image'],'source_group':plate['parent_group'],
        'image_size':plate['image_size'],'buildings':active,'camera':fit_record,
        'parameters':state['parameters'],'foreground':foreground,'paper_off_foreground':empty,
        'shadowed_receiver':shadowed,'lit_receiver':lit,'visible_mask':visible,'instances':truth,
        'lighting':{**light,'area_energy_w':light['area_energy_w']*factor},
        'scope':'Same camera and body geometry; paper-presence selection uses two Cycles foregrounds and visible paper masks. Receiver ratio is shared for the unchanged building footprint.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='OPTIX')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--supplement',action='store_true',help='Render only templates absent from the completed primary cache')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    recipe=full.config();decl=full.templates();plates={p['id']:full.render_plate(p) for p in full.plates()}
    layout_path=full.WORK/('layout_review_supplement.json' if args.supplement else 'layout_review.json')
    if args.supplement:
        prior=full.read(full.LAYERS/'manifest.json')
        done={row['id'] for row in prior['records']}
        decl['strata']=[s for s in decl['strata'] if s['id'] not in done]
        decl['budget']=len(decl['strata'])
        full.LAYERS=full.WORK/'supplement_layers'
    scene,camera,groups,baseline,receiver,key=prepare_scene(args.device,recipe)
    scene.render.use_persistent_data=True
    full.WORK.mkdir(parents=True,exist_ok=True)
    if args.dry_run:
        accepted=[];rejected=[]
        for state in decl['strata']:
            try:
                fit_record,_,_=fit(scene,camera,groups,baseline,plates[state['background']['plate_id']],state,recipe)
                accepted.append(state['id'])
            except ValueError as exc:rejected.append({'id':state['id'],'reason':str(exc)})
        write(layout_path,{'accepted':accepted,'rejected':rejected})
        print('FULL_LAYOUTS',len(accepted),'accepted',len(rejected),'rejected',flush=True)
        return
    review=full.read(layout_path)
    decl['strata']=[s for s in decl['strata'] if s['id'] in review['accepted']]
    decl['budget']=len(decl['strata'])
    code=[Path(__file__),Path(full.__file__),ROOT/'datasets/scripts/building_name_paper/render_hybrid_sample.py',
        ROOT/'datasets/scripts/building_name_paper/hybrid_sample.py',ROOT/'datasets/scripts/common/blender_masks.py',
        ROOT/'datasets/scripts/common/blender_photo.py',ROOT/'datasets/scripts/common/blender_workspace.py',
        ROOT/'datasets/scripts/common/frame_environment.py',ROOT/'datasets/scripts/common/config.py']
    configs=[source_path(full.TASK,k) for k in ('full_recipe','full_plates','full_render_plan','blender_scene','font')]
    configs += [Path(__file__).parent/'config/source.yaml',Path(__file__).parent/'config/dataset.yaml',layout_path]
    signature=hashlib.sha256()
    for path in code+configs:signature.update(path.read_bytes())
    fingerprint=signature.hexdigest()
    manifest=full.LAYERS/'manifest.json'
    previous=full.read(manifest) if manifest.exists() else {}
    if previous and (not args.resume or previous.get('input_fingerprint')!=fingerprint):
        raise ValueError('Existing layer cache requires matching --resume inputs')
    records={r['id']:r for r in previous.get('records',[])}
    with generation_guard('building_name.full_layers',[ROOT/p['image'] for p in plates.values()],[full.LAYERS],
        {'device':args.device,'fingerprint':fingerprint},root=ROOT,production_plan=decl,
        code_paths=code,procedural_inputs=configs,sample_output_paths=[full.LAYERS/'foreground'],
        auxiliary_output_paths=[full.LAYERS/'lighting',full.LAYERS/'masks',full.LAYERS/'paper_off'],
        replace_outputs=args.resume,replace_verifier=(lambda *_:{'preserved_paths':[]}) if args.resume else None) as run:
        if run.reused:return
        for sub in ('foreground','paper_off','lighting','masks'):(full.LAYERS/sub).mkdir(parents=True,exist_ok=True)
        for i,state in enumerate(run.plan['strata']):
            if state['id'] not in records:
                record=capture(scene,camera,groups,baseline,receiver,key,plates[state['background']['plate_id']],state,recipe)
                records[state['id']]=record
                write(manifest,{'task':full.TASK,'input_fingerprint':fingerprint,'source_blend':source_path(full.TASK,'blender_scene').relative_to(ROOT).as_posix(),
                    'records':list(records.values()),'accepted':[r['id'] for r in records.values() if r['status']=='accepted'],
                    'generation_plan':run.plan,'status':'in_progress'})
            record=records[state['id']]
            run.record_sample(state['id'],output=full.LAYERS/record['foreground'],sources=state['sources'],
                              parameters=state['parameters'],actual=record)
            print('FULL_RENDER',i+1,len(run.plan['strata']),state['id'],record['status'],flush=True)
        data=full.read(manifest);data['status']='complete';write(manifest,data)
        run.set_result({'templates':len(records),'accepted':len(data['accepted'])})


if __name__=='__main__':main()
