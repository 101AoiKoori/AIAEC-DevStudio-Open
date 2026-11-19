"""Incrementally replace reviewed empty negatives, retaining every positive byte.

The render stage is independent and reusable. New pairs are staged and checked
before any canonical pair is retired. Current plans describe the resulting
collection; original generation plans and production receipts remain historical.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.config import load_task_config
from datasets.scripts.common.frame_environment import query_frames, image_digest
from datasets.scripts.fire import compose_real_facades as base
from datasets.scripts.fire.fire_blender_labels import decode_ids, srgb_to_linear, visible_bbox
from lab.training_common.production_guard import generation_guard

CONFIG = ROOT/'datasets/scripts/fire/config/negative_replacement.json'
EXPORT = ROOT/'datasets/exports/fire'
WORK = ROOT/'datasets/annotated/fire/hard_negative_replacement'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def state(identifier, row):
    count=len(row.get('instances',[]))
    distractor=(row.get('context') or {}).get('distractor')
    return {'id': identifier, 'failure_event': 'color_shortcut', 'min_count': 1, 'max_count': 1,
        'prior': {'alpha': 1, 'beta': 1}, 'exposure_weight': 1, 'failure_cost': 1,
        'conditions': [distractor or 'unknown_context', f'designated_stickers:{count}'],
        'background': {'parent_group': row['source_group']},
        'target': {'count': count, 'building': next(iter(row.get('building_counts',{})),None),
                   'distractor': distractor},
        'labels': {'class_id': 0, 'scope': 'entire_output'},
        'augmentation': {'applied_stage': 'seeded_scene' if row['branch']=='blender_render' else 'source_gain',
                         'gain': row.get('transform',{}).get('gain',1)},
        'sources': [row.get('source_image') or row['scene_snapshot']],
        'parameters': {'sample_seed': row['sample_seed'], 'source_digest': row.get('source_sha256') or row['source_blend_sha256']}}


def patch_plan(split, rows, cfg):
    return {'schema_version': 1, 'model_key': 'fire_sticker', 'target_split': split,
        'purpose': cfg['purpose'], 'budget': len(rows), 'uncertainty_weight': .5,
        'independence_basis': cfg['eval_scope'] if split!='train' else 'Blender mother remains train-only',
        'constraints': {'only_replaces': cfg.get('retirement_kind','complete_empty_negatives')},
        'strata': [state(r['production_state_id'], r) for r in rows]}


def select_retired(rows, split, cfg):
    """Round-robin source groups; preserve direct real and all positive rows."""
    quotas = cfg['retire_train_branches'] if split=='train' else {cfg['retire_eval_branch']: cfg['replace_images'][split]}
    selected = []
    for branch, count in quotas.items():
        groups = defaultdict(list)
        for row in rows:
            if row['branch']==branch and not row['instances']:
                groups[(row.get('frame_id', ''), row.get('photo_geometry_id', ''), tuple(row['building_counts']))].append(row)
        groups = [sorted(v, key=lambda r:r['image']) for _,v in sorted(groups.items())]
        if sum(len(g) for g in groups) < count:
            raise ValueError(f'Not enough empty {branch} records in {split}')
        while count:
            for group in groups:
                if group and count:
                    selected.append(group.pop(0)); count -= 1
    if any(r['instances'] or r['branch']=='direct_real' for r in selected):
        raise ValueError('Replacement selection would touch a positive or scarce real frame')
    return selected


def verify_render_stage(cfg):
    folder = ROOT/cfg['render_output']; manifest = read(folder/'manifest.json')
    rows = manifest['records']
    if len(rows)!=cfg['replace_images']['train'] or Counter(r['context']['distractor'] for r in rows)!=cfg['train_types']:
        raise ValueError('Rendered hard-negative quota differs')
    for row in rows:
        if row['instances'] or (folder/row['label']).read_text(encoding='utf8'):
            raise ValueError('Render stage is not completely empty for the designated target')
        with Image.open(folder/row['image']) as image:
            image.load()
        rgb = np.asarray(Image.open(folder/row['instance_pass']).convert('RGB'), dtype=np.float32)/255
        ids = decode_ids(srgb_to_linear(rgb), [1, *row['distractor']['object_ids']])
        if np.any(ids==1):
            raise ValueError('A target ID exists in a hard negative')
        bbox, area = visible_bbox(np.isin(ids, row['distractor']['object_ids']))
        if bbox!=row['distractor']['xyxy'] or area!=row['distractor']['visible_pixels']:
            raise ValueError('Hard-negative visible mask differs')
        if row['actual']['ground_error_m']>.0001 or row['actual']['sticker_wall_plane_error_m']>.000001:
            raise ValueError('Hard-negative geometry evidence differs')
    return folder, rows


def real_rows(split, cfg):
    review = [json.loads(line) for line in (ROOT/cfg['real_review']).read_text(encoding='utf8').splitlines() if line.strip()]
    reviews = {r['content_sha256']: r for r in review if r['split']==split}
    sources = [r for r in query_frames(task='fire', role='hard_negative', split=split)
               if r['content_sha256'] in reviews]
    if len(sources)*len(cfg['real_gains'])!=cfg['replace_images'][split]:
        raise ValueError('Reviewed real source count differs from the replacement budget')
    output = WORK/'prepared'/split
    rows = []
    for source in sorted(sources, key=lambda r:r['frame_id']):
        observed = reviews[source['content_sha256']]
        if (source['label']['presence']!='absent' or source['label']['instances'] or
            source['label']['completeness']!='complete' or image_digest(source['image'])!=source['content_sha256']):
            raise ValueError('Real hard negative lost complete target-absence evidence')
        original = cv2.imdecode(np.fromfile(ROOT/source['image'], np.uint8), cv2.IMREAD_COLOR)
        for gain in cfg['real_gains']:
            stem = f'hard_negative_{len(rows)+1:06d}'
            image = output/'images'/(stem+'.jpg'); label = output/'labels'/(stem+'.txt')
            image.parent.mkdir(parents=True, exist_ok=True); label.parent.mkdir(parents=True, exist_ok=True)
            adjusted = np.clip(original.astype(np.float32)*gain, 0, 255).astype(np.uint8)
            cv2.imencode('.jpg', adjusted, [cv2.IMWRITE_JPEG_QUALITY,95])[1].tofile(str(image)); label.write_text('', encoding='utf8')
            rows.append({'image': f'images/{image.name}', 'label': f'labels/{label.name}', 'branch': 'direct_real',
                'generation_method': 'camera_gain_variant', 'split': split, 'source_split': split,
                'source_image': source['image'], 'source_group': source['parent_group'], 'parents': [source['parent_group']],
                'frame_id': source['frame_id'], 'record_id': source['record_id'], 'source_sha256': source['content_sha256'],
                'evidence': source['evidence'], 'instances': [], 'building_counts': {},
                'supervision_scope': 'entire_resampled_frame', 'production_state_id': stem,
                'sample_seed': cfg['seed']+len(rows), 'hard_negative': True,
                'context': {'distractor': 'red_orange_bin'}, 'distractor': observed['distractor'],
                'transform': {'crop_xyxy': [0,0,*source['image_size']], 'output_size': source['image_size'], 'gain': gain},
                'source_review': cfg['real_review']})
    return output, rows


def owned_paths(folder, retired):
    paths = set()
    for row in retired:
        for key in ('image', 'label', 'instance_pass', 'render_layer', 'overlay'):
            if row.get(key):
                path = (folder/row[key]).resolve()
                if not path.is_relative_to(folder.resolve()):
                    raise ValueError('Retired file points outside the declared split')
                paths.add(path)
    return paths


def current_plan(original, retained, additions):
    counts = Counter(r['production_state_id'] for r in retained)
    result = deepcopy(original); result['strata'] = []
    for item in original['strata']:
        if counts[item['id']]:
            item = deepcopy(item); item['min_count'] = item['max_count'] = counts.pop(item['id'])
            result['strata'].append(item)
    if counts:
        raise ValueError('Retained samples cannot be mapped to their current plan')
    result['strata'].extend(state(r['production_state_id'], r) for r in additions)
    result['budget'] = len(retained)+len(additions)
    result['purpose'] = 'Current fire collection after empty hard-negative substitution; positives retained'
    return result


def apply_split(split, stage, additions, cfg, source, retired_override=None):
    folder = EXPORT/split; manifest = read(folder/'manifest.json')
    section=cfg.get('receipt_key','negative_replacement')
    config_path=ROOT/cfg.get('config_path',base.relative(CONFIG))
    operation=('fire.replace_empty_negatives:' if section=='negative_replacement' else 'fire.'+section+':')+split
    def is_added(row):
        return row.get('patch_id')==section or (section=='negative_replacement' and row.get('hard_negative'))
    if manifest.get(section, {}).get('status')=='complete':
        retained = [r for r in manifest['records'] if not is_added(r)]
        path = ROOT/source['full_production_plans'][split]
        existing = [r for r in manifest['records'] if is_added(r)]
        # A final auxiliary-output check may have failed after writing the
        # canonical manifest. Reconcile an intact patch under a fresh receipt;
        # never delete/reselect ordinary negatives or rerender to finish it.
        plan = patch_plan(split, additions, cfg)
        generated = {folder/r['image'] for r in existing}
        preserved = [p for p in folder.rglob('*') if p.is_file() and p not in generated]
        inputs = [stage/r['image'] for r in additions]+[stage/r['label'] for r in additions]
        with generation_guard(operation, inputs, [folder],
                {'config':cfg,'split':split,'finish_existing_substitution':True},
                procedural_inputs=[config_path,CONFIG.with_name('source.yaml'),CONFIG.with_name('dataset.yaml')],
                code_paths=[Path(__file__),ROOT/'datasets/scripts/fire/compose_real_facades.py'], production_plan=plan,
                sample_output_paths=[folder/'images'],auxiliary_output_paths=[folder/'instance_pass',folder/'contact_sheet.jpg'],
                replace_outputs=True,replace_verifier=lambda workspace,state,receipt:{'preserved_paths':[base.relative(p) for p in preserved]}) as run:
            if not run.reused:
                base.OUTPUT=folder;base.validate_records(manifest['records'])
                states={s['id']:s for s in plan['strata']}
                for row in existing:
                    declaration=states[row['production_state_id']]
                    run.record_sample(row['production_state_id'],output=folder/row['image'],sources=declaration['sources'],
                        parameters=declaration['parameters'],actual=row)
                run.set_result({'images':len(existing),'retained_images':len(retained),'completion':'verified_existing_substitution'})
        base.write_json(path, current_plan(read(path), retained, additions))
        return manifest
    retired = retired_override if retired_override is not None else select_retired(manifest['records'], split, cfg)
    if len(retired)!=len(additions) or any(len(a['instances'])!=len(b['instances']) for a,b in zip(retired,additions)):
        raise ValueError('Substitution must preserve each selected image target count')
    if any(set(a['parents'])!=set(b['parents']) for a,b in zip(retired,additions)):
        if section!='negative_replacement':
            raise ValueError('Positive substitution must inherit the same parent')
    retired_images = {r['image'] for r in retired}
    retained = [r for r in manifest['records'] if r['image'] not in retired_images]
    remove = owned_paths(folder, retired)
    mutable = {folder/'manifest.json', folder/'validation.json', folder/'contact_sheet.jpg'}
    preserved = [p for p in folder.rglob('*') if p.is_file() and p.resolve() not in remove and p not in mutable]
    hashes = {p: image_digest(p) for p in preserved}
    plan_path = ROOT/source['full_production_plans'][split]
    new_plan = current_plan(read(plan_path), retained, additions)
    # Freeze retirement identities before changing any current output.
    basis = WORK/(f'{split}_retirement.json' if section=='negative_replacement' else f'{section}_{split}_retirement.json')
    base.write_json(basis, {'manifest_sha256': image_digest(folder/'manifest.json'), 'records': retired,
        'retired_file_hashes': {base.relative(p):image_digest(p) for p in remove if p.is_file()}})
    plan = patch_plan(split, additions, cfg)
    inputs = [stage/r['image'] for r in additions]+[stage/r['label'] for r in additions]
    inputs += [stage/r['instance_pass'] for r in additions if r.get('instance_pass')]
    inputs += [ROOT/r['source_image'] for r in additions if r.get('source_image')]
    inputs = list(dict.fromkeys(inputs))
    with generation_guard(operation, inputs, [folder], {'config': cfg, 'split': split},
            procedural_inputs=[config_path, CONFIG.with_name('source.yaml'), CONFIG.with_name('dataset.yaml'), basis, ROOT/cfg['real_review']],
            code_paths=[Path(__file__), ROOT/'datasets/scripts/fire/compose_real_facades.py'], production_plan=plan,
            sample_output_paths=[folder/'images'], auxiliary_output_paths=[folder/'instance_pass',folder/'contact_sheet.jpg'],
            replace_outputs=True, replace_verifier=lambda workspace, state, receipt: {'preserved_paths': [base.relative(p) for p in preserved]}) as run:
        if run.reused:
            return read(folder/'manifest.json')
        actual = []
        for addition in additions:
            row = deepcopy(addition); stem = Path(row['image']).name
            row['image'] = f'images/{split}/{stem}'; row['label'] = f'labels/{split}/{Path(stem).stem}.txt'
            for key in ('image', 'label', 'instance_pass'):
                if row.get(key):
                    destination = folder/row[key]
                    if destination.exists():
                        raise ValueError('New replacement filename collides with an existing file')
                    destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(stage/addition[key], destination)
            declaration = next(s for s in plan['strata'] if s['id']==row['production_state_id'])
            row['replacement_evidence'] = base.relative(basis)
            row['patch_id']=section
            actual.append(row)
            run.record_sample(row['production_state_id'], output=folder/row['image'], sources=declaration['sources'],
                parameters=declaration['parameters'], actual=row)
        records = retained+actual
        if len(records)!=len(manifest['records']) or Counter(len(r['instances']) for r in records)!=Counter(len(r['instances']) for r in manifest['records']):
            raise ValueError('Replacement changed the total or target-count distribution')
        base.OUTPUT = folder; validation = base.validate_records(records)
        # All replacement pairs exist and have been checked before old pairs leave.
        for path in remove:
            if path.is_file():
                path.unlink()
        for path, digest in hashes.items():
            if image_digest(path)!=digest:
                raise ValueError('A retained image, label, mask or auxiliary file changed')
        manifest.update(status='complete_with_reviewed_substitutions', records=records, summary=validation)
        manifest[section]={'status':'complete','config':base.relative(config_path),'images':len(actual),
            'retirement':base.relative(basis),'generation_plan':run.plan}
        manifest['retired_instance_passes']=sorted(set(manifest.get('retired_instance_passes',[]))|
            {r['instance_pass'] for r in retired if r.get('instance_pass')})
        replacements=manifest.setdefault('retired_positive_replacements',{})
        for old,new in zip(retired,actual):
            if old.get('instance_pass') and old['instances']:
                replacements[old['instance_pass']]={'image':new['image'],'count':len(new['instances'])}
        # Refresh only the overview. Retained per-image overlays stay byte-identical.
        examples = actual[::max(1,len(actual)//12)][:12]+[r for r in retained if r['instances']][:6]
        sheet = Image.new('RGB',(960,260*((len(examples)+2)//3)),'#eeeeee'); draw = ImageDraw.Draw(sheet)
        for index,row in enumerate(examples):
            image = Image.open(folder/row['image']).convert('RGB'); d = ImageDraw.Draw(image)
            for target in row['instances']:
                d.rectangle(target['xyxy'],outline='#00dd55',width=2)
            if row.get('distractor',{}).get('xyxy'):
                d.rectangle(row['distractor']['xyxy'],outline='#0099ff',width=2)
            image.thumbnail((320,240)); x,y=index%3*320,index//3*260
            sheet.paste(image,(x,y)); draw.text((x+4,y+243),Path(row['image']).stem,fill='black')
        sheet.save(folder/'contact_sheet.jpg',quality=90)
        base.write_json(folder/'manifest.json', manifest); base.write_json(folder/'validation.json', validation)
        run.set_result({'images':len(actual), 'retained_images':len(retained), 'retained_file_hashes_checked':len(hashes),
            'total_images':len(records), 'fire_instances':sum(len(r['instances']) for r in records)})
    base.write_json(plan_path, new_plan)
    return manifest


def synchronize_root(manifests, cfg, source):
    workspace = read(EXPORT/'workspace_sources.json')
    original = {r['image']: r for r in workspace['records']}
    records = []; parents = defaultdict(set)
    for split, manifest in manifests.items():
        for r in manifest['records']:
            full_image = base.relative(EXPORT/split/r['image'])
            for parent in r['parents']:
                parents[parent].add(split)
            if full_image in original and r.get('patch_id')!=cfg.get('receipt_key','negative_replacement'):
                records.append(original[full_image]); continue
            row = deepcopy(r)
            row.update(image=full_image, label=base.relative(EXPORT/split/r['label']), source_id=r['source_group'],
                label_status='reviewed', review_state='complete_positive' if r['instances'] else 'complete_negative', review_origin='machine',
                review_method='visible ID masks, calibrated wall projection or full-frame visual source review',
                review_evidence=base.relative(EXPORT/split/'validation.json'),
                source_review_evidence=base.relative(EXPORT/split/'manifest.json'), parent_samples=r['parents'],
                development_contact=['hard_negative_source_visual_review'])
            if r['branch']=='blender_render':
                row.update(source_kind='blender_scene', generation_method='blender_render', mother_scene=source['scene'])
            elif r['branch']=='direct_real':
                row.update(source_kind='photo', generation_method='camera_gain_variant', image_processing={
                    'resolution_review':'confirmed', 'source_size':r['transform']['output_size'],
                    'output_size':r['transform']['output_size'], 'resampling':'resampled', 'clarity':'modified'})
            else:
                row.update(source_kind='derived_image',generation_method='composite',image_processing={
                    'origin_review':'confirmed','background_origin':'direct_camera'})
            records.append(row)
    if any(len(v)>1 for v in parents.values()):
        raise ValueError('Replacement introduced a cross-split parent')
    config_path=cfg.get('config_path',base.relative(CONFIG))
    section=cfg.get('receipt_key','negative_replacement')
    workspace.update(records=records, allocation=config_path, foundation_allocation=source['full_dataset_plan'],
        coverage_scope='source_geometry_and_empty_hard_negative_substitution',
        hard_negative_materialization='empty_negative_replacement_complete_positive_pairs_pending',
        negative_replacement=base.relative(CONFIG), positive_distractor_materialization='pending')
    workspace.setdefault('exposure_notes', []).append(cfg['eval_scope'])
    base.write_json(EXPORT/'workspace_sources.json', workspace)
    root = read(EXPORT/'manifest.json')
    root.update(images=len(records), parent_groups=len(parents), cross_split_parent_overlap=0,
        allocation=config_path, foundation_allocation=source['full_dataset_plan'],
        coverage_scope=workspace['coverage_scope'], hard_negative_materialization=workspace['hard_negative_materialization'],
        positive_distractor_materialization='pending')
    if section=='negative_replacement':
        root['negative_replacement']={'status':'complete','config':config_path,'images':cfg['replace_images']}
    else:
        for obj in (root,workspace):
            obj.update(coverage_scope='first_round_color_contrasts_and_size_coverage',
                positive_distractor_materialization='complete',hard_negative_materialization='complete',
                training_preparation={'status':'complete','config':config_path,'images':cfg['replace_images']})
        base.write_json(EXPORT/'workspace_sources.json',workspace)
    base.write_json(EXPORT/'manifest.json', root)
    for path in EXPORT.rglob('*.cache'):
        path.unlink()


def main():
    cfg = read(CONFIG); source = load_task_config('fire')[1]
    root = read(EXPORT/'manifest.json')
    if root.get('negative_replacement', {}).get('status')=='complete':
        print(json.dumps({'status':'already_complete', 'images':root['negative_replacement']['images']})); return
    if not (ROOT/cfg['render_output']/'manifest.json').is_file():
        snapshot = ROOT/cfg['source_snapshot']
        if not snapshot.is_file():
            snapshot.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/cfg['source_scene'],snapshot)
        subprocess.run([source['sample_recipe']['blender_executable'], '--background', str(ROOT/cfg['source_snapshot']),
            '--python-exit-code', '1', '--python', str(ROOT/'datasets/scripts/fire/render_hard_negatives.py')], cwd=ROOT, check=True)
    train_stage, train = verify_render_stage(cfg)
    prepared = {'train': (train_stage, train), **{s:real_rows(s,cfg) for s in ('val','test')}}
    # Check the final union before entering any replacement transaction.
    parents = defaultdict(set)
    for split, (_, rows) in prepared.items():
        current = read(EXPORT/split/'manifest.json')
        complete = current.get('negative_replacement',{}).get('status')=='complete'
        remove = set() if complete else {r['image'] for r in select_retired(current['records'],split,cfg)}
        final = current['records'] if complete else [r for r in current['records'] if r['image'] not in remove]+rows
        for row in final:
            for parent in row['parents']:
                parents[parent].add(split)
    if any(len(v)>1 for v in parents.values()):
        raise ValueError('Planned substitution crosses parent splits')
    manifests = {s:apply_split(s,*prepared[s],cfg,source) for s in ('train','val','test')}
    synchronize_root(manifests,cfg,source)
    print(json.dumps({'status':'complete','replaced':cfg['replace_images'],'total':sum(len(m['records']) for m in manifests.values()),
        'parent_groups':len(parents)},ensure_ascii=False))


if __name__ == '__main__':
    main()
