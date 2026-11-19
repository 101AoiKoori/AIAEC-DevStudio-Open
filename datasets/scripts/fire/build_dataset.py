"""Build current fire sample, then parent-isolated train/val/test exports."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
import subprocess
import sys

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.fire import compose_real_facades as base
from datasets.scripts.fire.photo_geometry import blend_render
from datasets.scripts.common.config import load_task_config, source_path
from datasets.scripts.common.frame_environment import query_frames, image_digest
from datasets.scripts.common.images import read_image
from datasets.scripts.common.paths import sample_dir, split_dir
from lab.training_common.production_guard import generation_guard

HERE = Path(__file__).resolve().parent


def category(count):
    return 'zero' if count == 0 else 'single' if count == 1 else 'multi'


def distribute(states, quotas, count_fn):
    for group, quota in quotas.items():
        chosen = [s for s in states if category(count_fn(s)) == group]
        if quota and not chosen:
            raise ValueError(f'No feasible source for {group}')
        for index, state in enumerate(chosen):
            state['samples'] = quota//len(chosen) + (index < quota % len(chosen))
    return [s for s in states if s['samples']]


def camera_uses(source, allocation, split=None, full=False):
    selected = {f['frame_id']: f for f in allocation['frames'] if split is None or f['split'] == split}
    rows = {r['frame_id']: r for role in ('empty_negative', 'direct_target')
            for r in query_frames(task='fire', role=role) if r['frame_id'] in selected}
    if set(rows) != set(selected):
        raise ValueError('The selected real frame inventory changed')
    uses = []
    for fid, row in sorted(rows.items()):
        if row['split'] != selected[fid]['split'] or row['parent_group'] != selected[fid]['parent_group']:
            raise ValueError(f'Selected parent split changed: {fid}')
        if row['scope'] != {'kind': 'frame'} or row['label'].get('completeness') != 'complete':
            raise ValueError(f'Complete frame labels required: {fid}')
        if image_digest(row['image']) != row['content_sha256']:
            raise ValueError(f'Source image bytes changed: {fid}')
        attribution = (base.identify_building(row, source) if row['label']['instances'] or
                       any(name in row['parent_group'] for name in source['building_aliases'].values())
                       else {'target_building': None, 'visible_buildings': []})
        count = len(row['label']['instances'])
        uses.append({'id': f'direct_real_{fid[:12]}', 'row': row, 'branch': 'direct_real',
                     'count': count, 'tone': 'as_captured', 'samples': 1, **attribution})
    walls = [r for r in query_frames(task='fire', role='composite_background')
             if r['frame_id'] in selected and selected[r['frame_id']]['wall_background']]
    for row in sorted(walls, key=lambda r: r['frame_id']):
        if row['split'] != selected[row['frame_id']]['split'] or not row.get('geometry', {}).get('wall_surfaces'):
            raise ValueError('Selected facade lacks current wall registration or split')
        for count in source['sample_recipe']['counts']:
            for tone in source['sample_recipe']['tones']:
                uses.append({'id': f'real_background_composite_{row["frame_id"][:12]}_{count}_{tone}',
                    'row': row, 'branch': 'real_background_composite', 'count': count,
                    'tone': tone, 'samples': 1, **base.identify_building(row, source)})
    if full:
        direct = [u for u in uses if u['branch'] == 'direct_real']
        expected = allocation['source_quotas']['direct_real'][split]
        if len(direct) != expected:
            raise ValueError(f'Direct source count changed: {len(direct)} != {expected}')
        composite = distribute([u for u in uses if u['branch'] == 'real_background_composite'],
                               allocation['composite_counts'][split], lambda u: u['count'])
        uses = direct + composite
    return {u.pop('id'): u for u in uses}


def render_plan(source, allocation, full=False):
    plan = base.renderer_plan(source)
    states = plan['strata']
    geometries = json.loads((ROOT/source['hybrid_geometry_review']).read_text(encoding='utf8'))['backgrounds']
    for review in geometries:
        if review['id'] not in allocation['hybrid_backgrounds']:
            continue
        for building in 'ABC':
            for count in source['sample_recipe']['counts']:
                state = deepcopy(states[0])
                state['id'] = f'blender_explicit_hybrid_{review["id"]}_{building}_{count}'
                state['target'].update(active_buildings=[building], fire_counts={building: count},
                                       view='orange_peel_04')
                state['background'] = {'domain': 'calibrated_real_scene', 'traffic_signal_injection': 'none',
                    'photo_geometry_id': review['id'], 'parent_group': review['parent_group']}
                state['sources'] = sorted([*state['sources'], 'reviewed_real_building_background'])
                state['augmentation'].update(camera='orange_peel_04', lighting='photo_estimated_physical_shadow')
                states.append(state)
    if full:
        output = []
        for branch, quotas in allocation['blender_counts'].items():
            chosen = [s for s in states if bool(s['background'].get('photo_geometry_id')) ==
                      (branch == 'blender_real_background_composite_trial')]
            chosen = distribute(chosen, quotas, lambda s: sum(s['target']['fire_counts'].values()))
            for state in chosen:
                state['min_count'] = state['max_count'] = state.pop('samples')
            output += chosen
        plan['strata'] = output
    plan['budget'] = sum(s['min_count'] for s in plan['strata'])
    plan['strata'].sort(key=lambda s: not bool(s['background'].get('photo_geometry_id')))
    plan['purpose'] = 'current database camera-matched hybrid and orange-peel fire ' + ('full train' if full else 'sample')
    return plan


def production_plan(source, uses, renderer, split):
    plan = base.production_plan(uses, renderer or {'strata': []}, source['target_artwork'], source)
    for state in plan['strata']:
        if state['id'] in uses:
            state['min_count'] = state['max_count'] = uses[state['id']]['samples']
    plan['budget'] = sum(s['min_count'] for s in plan['strata'])
    plan['target_split'] = split or 'unassigned'
    plan['purpose'] = f'parent-group fire {split or "development sample"}; four generation methods'
    return plan


def prepare_plans(source, allocation, full, split):
    uses = camera_uses(source, allocation, split, full)
    renderer = render_plan(source, allocation, full) if split in (None, 'train') else None
    plan = production_plan(source, uses, renderer, split)
    path = ROOT/(source['full_production_plans'][split] if full else source['production_plan'])
    base.write_json(path, plan)
    render_path = ROOT/'datasets/annotated/fire/production_inputs'/f'{split or "sample"}_renderer_plan.json'
    if renderer:
        base.write_json(render_path, renderer)
    return uses, renderer, plan, path, render_path


def paper_artwork(source):
    icon = read_image(ROOT/source['target_artwork'])
    paper = np.full((210,150,3),255,np.uint8)
    factor = min(140/icon.shape[1],200/icon.shape[0])
    patch = cv2.resize(icon,(round(icon.shape[1]*factor),round(icon.shape[0]*factor)),interpolation=cv2.INTER_AREA)
    y,x = (210-patch.shape[0])//2,(150-patch.shape[1])//2
    paper[y:y+patch.shape[0],x:x+patch.shape[1]] = patch
    return paper


def render_records(output, source, states, retained=None):
    raw = [json.loads(line) for line in (output/'manifest.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
    geometry_rows = {r['id']: r for r in json.loads((ROOT/source['hybrid_geometry_review']).read_text(encoding='utf8'))['backgrounds']}
    records = []
    scene_digest = retained['source_blend_sha256'] if retained else image_digest(source['scene'])
    for row in raw:
        if not row.get('fire_label'):
            raise ValueError(f'Unresolved rendered label: {row.get("fire_label_suppression_reason")}')
        instances = [{'building': i['building'], 'xyxy': i['pixel_bbox_xyxy'], 'origin': 'rendered',
            **{k: i.get(k) for k in ('id','height_fraction','scale_from_reference','paper_height_m','wall_contact','reference_facade_height_m')}}
            for i in row['instances'] if i['kind']=='fire_sticker' and i['pixel_bbox_xyxy']]
        if len(instances) != sum(row['requested_fire_counts'].values()):
            raise ValueError('Rendered requested count differs from visible fire instances')
        hybrid = row.get('photo_geometry_id')
        record = {'image': row['image'], 'label': row['fire_label'], 'instance_pass': row['instance_pass'],
            'branch': 'blender_real_background_composite' if hybrid else 'blender_render',
            'generation_method': 'blender_real_background_composite' if hybrid else 'blender_render',
            'split': 'train', 'source_split': 'train', 'instances': instances,
            'building_counts': base.counts_for(row['active_buildings'], instances, 'all_generated_front_facade_placements', True),
            'source_group': source['scene'], 'parents': [source['scene']], 'source_scene': source['scene']+'#'+source['blender_scene_name'],
            'source_blend_sha256': scene_digest, 'camera': row['camera'],
            'requested_fire_counts': row['requested_fire_counts'], 'supervision_scope': 'entire_render_frame',
            'production_state_id': row['production_state_id'], 'sample_seed': row['sample_seed']}
        if retained:
            record['scene_snapshot'] = retained['scene_snapshot']
            record['retained_render_verification'] = source['completed_renderer_verification']
        if hybrid:
            review = geometry_rows[hybrid]
            source_row = next(r for r in query_frames(task='fire',role='composite_background',frame_id=review['frame_id'])
                              if r['split']=='train')
            if source_row['label']['instances'] or source_row['content_sha256'] != review['content_sha256']:
                raise ValueError('Hybrid photo needs unchanged complete fire-negative evidence')
            bg = cv2.resize(read_image(ROOT/review['source_image']),tuple(row_size(output/row['image'])),interpolation=cv2.INTER_AREA)
            rgba = cv2.imdecode(np.fromfile(output/row['image'],dtype=np.uint8),cv2.IMREAD_UNCHANGED)
            if rgba is None or rgba.shape[2] != 4:
                raise ValueError('Hybrid render did not contain a shadow-catcher alpha channel')
            record['render_layer'] = row['image']
            record['image'] = f'images/train/{Path(row["image"]).name}'
            final = blend_render(bg, rgba)
            cv2.imencode('.png',final)[1].tofile(str(output/record['image']))
            record.update(photo_geometry_id=hybrid, photo_frame_id=review['frame_id'],
                photo_sha256=review['content_sha256'], photo_parent=review['parent_group'],
                photo_evidence=source_row['evidence'], hybrid_geometry=source['hybrid_geometry_review'])
            record['parents'].append(review['parent_group'])
        records.append(record)
    return records


def row_size(path):
    from PIL import Image
    with Image.open(path) as image:
        return image.size


def audit(records, allocation, split=None):
    validation = base.validate_records(records)
    parent_splits = {}
    for row in records:
        if split and row['split'] != split:
            raise ValueError('Export record split differs from its output')
        for parent in row['parents']:
            parent_splits.setdefault(parent,set()).add(row['split'])
    if any(len(s)>1 for s in parent_splits.values()):
        raise ValueError('Capture or Blender mother crosses train/val/test')
    validation.update(by_split=dict(Counter(r['split'] for r in records)), parent_groups=len(parent_splits),
                      cross_split_parent_overlap=0, parent_assignment=allocation['exposure_notes'])
    return validation


def retained_renderer(source, output, renderer):
    value=source.get('completed_renderer_verification')
    if not value or not (ROOT/value).is_file():
        return None
    proof=json.loads((ROOT/value).read_text(encoding='utf8'))
    if proof['render_output']!=base.relative(output) or proof['image_count']!=renderer['budget']:
        raise ValueError('Retained renderer proof differs from the selected output or budget')
    if image_digest(proof['scene_snapshot'])!=proof['source_blend_sha256']:
        raise ValueError('Retained mother snapshot bytes changed')
    if image_digest(output/'manifest.jsonl')!=proof['raw_manifest_sha256'] or image_digest(output/'run.json')!=proof['run_metadata_sha256']:
        raise ValueError('Retained renderer metadata changed after independent verification')
    for relative,expected in proof['verified_files'].items():
        path=(output/relative).resolve()
        if not path.is_relative_to(output) or image_digest(path)!=expected['sha256']:
            raise ValueError(f'Retained renderer file changed: {relative}')
    rows=[json.loads(line) for line in (output/'manifest.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
    if Counter(r['production_state_id'] for r in rows)!={s['id']:s['min_count'] for s in renderer['strata']}:
        raise ValueError('Retained renderer states differ from the selected execution plan')
    return proof


def run_batch(source, allocation, full, split, replace, resume=False):
    output = split_dir('fire',split) if full else sample_dir('fire')
    base.OUTPUT = output
    uses, renderer, plan, plan_path, render_path = prepare_plans(source, allocation, full, split)
    retained=retained_renderer(source,output,renderer) if resume and renderer else None
    media = {u['row']['image'] for u in uses.values()} | {source['target_artwork']}
    geometry_path = ROOT/source['hybrid_geometry_review']
    if renderer:
        media.update(r['source_image'] for r in json.loads(geometry_path.read_text(encoding='utf8'))['backgrounds'])
    procedural = {plan_path, HERE/'config/source.yaml', HERE/'config/dataset.yaml', ROOT/source['full_dataset_plan'],
                  ROOT/source['wall_plane_review']}
    if renderer:
        procedural.update({render_path,geometry_path,
            ROOT/retained['scene_snapshot'] if retained else ROOT/source['scene']})
    if retained:
        procedural.add(ROOT/source['completed_renderer_verification'])
    code = [Path(__file__), HERE/'compose_real_facades.py', HERE/'photo_geometry.py', HERE/'wall_projection.py',
            HERE/'hybrid_render.py', HERE/'render_blender_fire_dataset.py', HERE/'fire_blender_labels.py',
            HERE/'camera_trajectory.py', ROOT/'lab/training_common/bayesian_allocation.py']
    code += [ROOT/f'datasets/scripts/common/{name}.py' for name in
             ('config','paths','annotations','bayesian','blender_workspace','paper','perspective','images','frame_environment')]
    owned = base.generated_files()
    existing = {p for p in output.rglob('*') if p.is_file()} if output.exists() else set()
    preserved = existing-owned
    if preserved-{output/'.gitkeep'}:
        raise ValueError(f'Unowned files in fire output: {preserved}')
    operation = 'fire.render_blender_fire_dataset:'+base.relative(output)
    parameters = {'child_production_plans': {operation: renderer} if renderer and not retained else {},
                  'source_records': sorted({u['row']['record_id'] for u in uses.values()}),
                  'full': full, 'split': split}
    if retained:
        parameters['retained_renderer_verification']=source['completed_renderer_verification']
    with generation_guard('fire.current_dataset:'+base.relative(output), sorted(media), [output], parameters,
        code_paths=code, procedural_inputs=sorted(procedural), production_plan=plan,
        sample_output_paths=[output/'images'], auxiliary_output_paths=[output/'instance_pass', output/'render_layers',
            output/'overlays', output/'contact_sheet.jpg'], replace_outputs=replace,
        replace_verifier=(lambda workspace,state,receipt: {'preserved_paths':[base.relative(p) for p in preserved]}) if replace else None) as run:
        if run.reused:
            return json.loads((output/'manifest.json').read_text(encoding='utf8'))
        if existing and not replace:
            raise ValueError('Use --replace for current generated outputs')
        if not retained:
            for path in sorted(owned):
                path.unlink()
            for folder in sorted((p for p in output.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
                if not any(folder.iterdir()): folder.rmdir()
        scaffold = output/'.gitkeep'
        scaffold_bytes = scaffold.read_bytes() if scaffold.is_file() else None
        if scaffold_bytes is not None: scaffold.unlink()
        recipe=source['sample_recipe']
        if renderer and not retained:
            width,height=recipe['render_image_size']
            command=[recipe['blender_executable'],'--background',str(source_path('fire','scene')),
                '--python-exit-code','1','--python',str(HERE/'render_blender_fire_dataset.py'),'--',
                '--output',base.relative(output),'--production-plan',base.relative(render_path),
                '--count',str(renderer['budget']),'--seed',str(recipe['seed']),
                '--resolution',str(width),'--height',str(height),'--samples',str(recipe['blender_samples']),
                '--device',recipe['device'],'--current-geometry',source['hybrid_geometry_review']]
            subprocess.run(command,cwd=ROOT,env=run.child_env(),check=True)
        else:
            output.mkdir(parents=True,exist_ok=True)
        states={s['id']:s for s in run.plan['strata']}
        records=render_records(output,source,states,retained) if renderer else []
        for record in records:
            state=states[record['production_state_id']]
            run.record_sample(state['id'],output=output/record['image'],sources=state['sources'],parameters=state['parameters'],actual=record)
        paper=paper_artwork(source)
        image_index=0
        for key,use in uses.items():
            for repetition in range(use['samples']):
                seed=recipe['seed']+10000+image_index
                actual_use=deepcopy(use)
                if full and use['branch']=='real_background_composite':
                    # Expand the reviewed crop by bounded independent margins. The
                    # paper anchors stay inside the observed wall and output.
                    rng=random.Random(seed);crop=actual_use['row']['geometry']['supervision_crop_xyxy']
                    width,height=use['row']['image_size'];cw,ch=crop[2]-crop[0],crop[3]-crop[1]
                    actual_use['row']['geometry']['supervision_crop_xyxy']=[max(0,int(crop[0]-cw*rng.uniform(.02,.08))),
                        max(0,int(crop[1]-ch*rng.uniform(.02,.08))),min(width,int(crop[2]+cw*rng.uniform(.02,.08))),
                        min(height,int(crop[3]+ch*rng.uniform(.02,.08)))]
                image,instances,transform=base.compose_camera(actual_use,paper,recipe,seed)
                row=use['row'];partition=row['split'];stem=f'{use["branch"]}_{image_index:04d}'
                image_path=output/'images'/partition/(stem+'.jpg');label_path=output/'labels'/partition/(stem+'.txt')
                image_path.parent.mkdir(parents=True,exist_ok=True);label_path.parent.mkdir(parents=True,exist_ok=True)
                cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,95])[1].tofile(str(image_path))
                label_path.write_text(''.join(base.yolo_row(i['xyxy'],image.shape[1],image.shape[0])+'\n' for i in instances),encoding='utf8')
                record={'image':image_path.relative_to(output).as_posix(),'label':label_path.relative_to(output).as_posix(),
                    'branch':use['branch'],'generation_method':'camera_resampled' if use['branch']=='direct_real' else 'real_photo_plus_digital_artwork',
                    'split':partition,'source_split':partition,'source_image':row['image'],'source_group':row['parent_group'],
                    'parents':[row['parent_group']],'frame_id':row['frame_id'],'record_id':row['record_id'],
                    'source_sha256':row['content_sha256'],'evidence':row['evidence'],'instances':instances,
                    'transform':transform,'sample_seed':seed,'tone':use['tone'],'production_state_id':key,
                    'supervision_scope':'entire_output_crop' if use['branch']=='real_background_composite' else 'entire_resampled_frame',
                    'building_counts':base.counts_for(use['visible_buildings'],instances,'visible_detectable_instances_in_output'),
                    'development_contact':not full}
                run.record_sample(key,output=image_path,sources=states[key]['sources'],parameters=states[key]['parameters'],actual=record)
                records.append(record);image_index+=1
                base.write_json(output/'manifest.json',{'task':'fire','status':'running','records':records})
        validation=audit(records,allocation,split)
        if full:
            if len(records)!=allocation['totals'][split] or dict(Counter(category(len(r['instances'])) for r in records))!=allocation['counts'][split]:
                raise ValueError('Full output quota or zero/single/multi counts differ from allocation')
        base.previews(records)
        manifest={'task':'fire','model_key':'fire_sticker','status':'complete_four_method_'+('export' if full else 'sample'),
            'plan':base.relative(plan_path),'generation_plan':run.plan,'allocation':source['full_dataset_plan'],
            'summary':validation,'records':records}
        base.write_json(output/'manifest.json',manifest);base.write_json(output/'validation.json',validation)
        if scaffold_bytes is not None: scaffold.write_bytes(scaffold_bytes)
        elif full: scaffold.touch()
        run.set_result(validation)
        print(json.dumps({'output':base.relative(output),**validation},ensure_ascii=False),flush=True)
        return manifest


def finish_completed_batch(source, allocation, split):
    """Finish an intact generation after a metadata/preview failure, without rendering again."""
    output=split_dir('fire',split)
    path=output/'manifest.json'
    if not path.exists():
        return None
    saved=json.loads(path.read_text(encoding='utf8'))
    rows=saved.get('records',[])
    if len(rows)!=allocation['totals'][split]:
        raise ValueError(f'Cannot finish an incomplete {split} batch: {len(rows)} records')
    base.OUTPUT=output
    uses,renderer,plan,plan_path,render_path=prepare_plans(source,allocation,True,split)
    states={s['id']:s for s in plan['strata']}
    counts=Counter(r['production_state_id'] for r in rows)
    if counts!={s['id']:s['min_count'] for s in plan['strata']}:
        raise ValueError('Existing generation states differ from the selected plan')
    if renderer:
        run_metadata=json.loads((output/'run.json').read_text(encoding='utf8'))
        if run_metadata.get('completed')!=renderer['budget'] or run_metadata.get('status')!='complete_candidates':
            raise ValueError('The existing renderer did not complete its declared image budget')
        for row in rows:
            if row.get('source_blend_sha256') and image_digest(row.get('scene_snapshot',source['scene']))!=row['source_blend_sha256']:
                raise ValueError('The rendered mother snapshot differs from its recorded input')
    expected={state_id:u['row'] for state_id,u in uses.items()}
    for row in rows:
        if row.get('frame_id') and (row['record_id']!=expected[row['production_state_id']]['record_id'] or
                row['source_sha256']!=expected[row['production_state_id']]['content_sha256']):
            raise ValueError('A source observation changed after image generation')
    snapshot=ROOT/'datasets/annotated/fire/production_inputs'/f'{split}_completed_{image_digest(path)[:16]}.json'
    base.write_json(snapshot,saved)
    owned=base.generated_files()
    existing={p for p in output.rglob('*') if p.is_file()}
    preserved=existing-owned
    if preserved-{output/'.gitkeep'}:
        raise ValueError('Unowned files prevent finishing the selected generation')
    media={u['row']['image'] for u in uses.values()}|{source['target_artwork']}
    procedural={snapshot,plan_path,ROOT/source['full_dataset_plan'],HERE/'config/source.yaml',HERE/'config/dataset.yaml'}
    if renderer:
        procedural.update({render_path,ROOT/source['hybrid_geometry_review'],ROOT/source['wall_plane_review']})
        procedural.update(ROOT/r.get('scene_snapshot',source['scene']) for r in rows if r.get('source_blend_sha256'))
        if source.get('completed_renderer_verification'):
            procedural.add(ROOT/source['completed_renderer_verification'])
        media.update(r['source_image'] for r in json.loads((ROOT/source['hybrid_geometry_review']).read_text(encoding='utf8'))['backgrounds'])
    with generation_guard('fire.current_dataset:'+base.relative(output),sorted(media),[output],
        {'full':True,'split':split,'finish_existing_generation':base.relative(snapshot)},
        code_paths=[Path(__file__),HERE/'compose_real_facades.py',HERE/'wall_projection.py',HERE/'photo_geometry.py'],
        procedural_inputs=sorted(procedural),production_plan=plan,sample_output_paths=[output/'images'],
        auxiliary_output_paths=[output/'instance_pass',output/'render_layers',output/'overlays',output/'contact_sheet.jpg'],
        replace_outputs=True,replace_verifier=lambda workspace,state,receipt:{'preserved_paths':[base.relative(p) for p in preserved]}) as run:
        if run.reused:
            return json.loads(path.read_text(encoding='utf8'))
        validation=audit(rows,allocation,split)
        if dict(Counter(category(len(r['instances'])) for r in rows))!=allocation['counts'][split]:
            raise ValueError('Existing labels differ from the selected count allocation')
        for row in rows:
            state=states[row['production_state_id']]
            run.record_sample(state['id'],output=output/row['image'],sources=state['sources'],parameters=state['parameters'],
                              actual={**row,'reused_completed_generation':base.relative(snapshot)})
        base.previews(rows)
        manifest={'task':'fire','model_key':'fire_sticker','status':'complete_four_method_export',
            'plan':base.relative(plan_path),'generation_plan':run.plan,'allocation':source['full_dataset_plan'],
            'completed_generation_input':base.relative(snapshot),'summary':validation,'records':rows}
        base.write_json(path,manifest);base.write_json(output/'validation.json',validation)
        if not (output/'.gitkeep').exists():
            (output/'.gitkeep').touch()
        run.set_result(validation)
        print(json.dumps({'finished_existing':base.relative(output),**validation},ensure_ascii=False),flush=True)
        return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--replace',action='store_true')
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--resume',action='store_true',help='finish an intact full generation without rerendering its images')
    parser.add_argument('--replace-empty-negatives',action='store_true',help='apply the configured hard-negative substitution to existing exports')
    parser.add_argument('--foundation-only',action='store_true',help='explicitly rebuild the source/geometry foundation without hard-negative substitution')
    args=parser.parse_args(argv)
    if args.resume and not args.full:
        parser.error('--resume requires --full')
    source=load_task_config('fire')[1]
    current_path=ROOT/'datasets/exports/fire/manifest.json'
    current=json.loads(current_path.read_text(encoding='utf8')) if current_path.is_file() else {}
    substituted=current.get('negative_replacement',{}).get('status')=='complete'
    if args.plan_only and substituted and not args.foundation_only:
        print(json.dumps({'status':'current_substituted_plans','paths':source['full_production_plans']}));return
    if args.replace_empty_negatives or (args.full and substituted and not args.foundation_only):
        if args.plan_only:
            print(json.dumps({'status':'current_substituted_plans','paths':source['full_production_plans']}));return
        from datasets.scripts.fire.replace_empty_negatives import main as replace_negatives
        replace_negatives()
        if args.full and source.get('training_preparation_config'):
            from datasets.scripts.fire.prepare_training_data import main as prepare_training
            prepare_training()
        return
    allocation=json.loads((ROOT/source['full_dataset_plan']).read_text(encoding='utf8'))
    if args.plan_only:
        for full,split in [(False,None),*( (True,s) for s in ('train','val','test'))]:
            uses,renderer,plan,_,_=prepare_plans(source,allocation,full,split)
            print(json.dumps({'full':full,'split':split,'images':plan['budget'],'rendered':renderer['budget'] if renderer else 0},ensure_ascii=False))
        return
    for command in (['lab/scripts/media/media_index.py','build','--scan-root','datasets/scripts','--source','datasets/scripts/fire','--no-video-probe','--no-clean'],
                    ['lab/scripts/media/media_index_sqlite.py','sync']):
        subprocess.run([sys.executable,'-X','utf8','-B',*command],check=True,cwd=ROOT)
    manifests=[]
    for split in (('train','val','test') if args.full else (None,)):
        finished=finish_completed_batch(source,allocation,split) if args.resume else None
        manifests.append(finished if finished is not None else run_batch(source,allocation,args.full,split,args.replace,args.resume))
    if args.full:
        all_records=[r for m in manifests for r in m['records']]
        base.OUTPUT=split_dir('fire','train')
        # Cross-split lineage is checked on the union; geometry is already checked
        # against each split's actual files by its production run.
        parents={}
        for r in all_records:
            for p in r['parents']:parents.setdefault(p,set()).add(r['split'])
        if any(len(v)>1 for v in parents.values()): raise ValueError('Full union has parent leakage')
        export=split_dir('fire','train').parent
        (export/'data.yaml').write_text(yaml.safe_dump({'path':str(export),'train':'train/images/train','val':'val/images/val',
            'test':'test/images/test','names':{0:'fire_sticker'}},sort_keys=False),encoding='utf8')
        base.write_json(export/'manifest.json',{'task':'fire','status':'complete','allocation':source['full_dataset_plan'],
            'splits':{s:f'{s}/manifest.json' for s in ('train','val','test')},'images':len(all_records),'parent_groups':len(parents),'cross_split_parent_overlap':0})
        source_rows=[]
        for record in all_records:
            row=deepcopy(record);folder=split_dir('fire',row['split'])
            row.update(image=base.relative(folder/record['image']),label=base.relative(folder/record['label']),
                source_id=record['source_group'],label_status='reviewed',
                review_state='complete_positive' if record['instances'] else 'complete_negative',
                review_origin='machine',review_method='visible ID masks, wall projection, current frame labels and building partition',
                review_evidence=base.relative(folder/'validation.json'),
                source_review_evidence=base.relative(folder/'manifest.json'),
                parent_samples=record['parents'])
            branch=record['branch']
            if branch=='blender_render':
                row.update(source_kind='blender_scene',generation_method='blender_render',mother_scene=source['scene'])
            elif branch=='direct_real':
                row.update(source_kind='photo',generation_method='camera_capture',image_processing={
                    'resolution_review':'confirmed','source_size':record['transform']['crop_xyxy'][2:],
                    'output_size':record['transform']['output_size'],'resampling':'resampled','clarity':'modified'})
            else:
                row.update(source_kind='derived_image',generation_method='composite',image_processing={
                    'origin_review':'confirmed','background_origin':'direct_camera'})
                if branch=='blender_real_background_composite':row['mother_scene']=source['scene']
            row['development_contact']=['fire_sample_20261118','source_review']
            source_rows.append(row)
        base.write_json(export/'workspace_sources.json',{'task':'fire','model_key':'fire_sticker','selection_mode':'manifest_only',
            'allocation':source['full_dataset_plan'],'records':source_rows,'exposure_notes':allocation['exposure_notes']})
        if source.get('negative_replacement_config') and not args.foundation_only:
            from datasets.scripts.fire.replace_empty_negatives import main as replace_negatives
            replace_negatives()
            if source.get('training_preparation_config'):
                from datasets.scripts.fire.prepare_training_data import main as prepare_training
                prepare_training()


if __name__=='__main__':
    main()
