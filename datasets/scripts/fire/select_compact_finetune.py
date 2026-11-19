"""Select compact fire replay without regenerating images, altering the pool, or training."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import random
import shutil
import sys

from PIL import Image
import yaml

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from datasets.scripts.fire.build_reshoot_finetune import Builder, digest, relative, write_json

PLAN=ROOT/'datasets/scripts/fire/config/reshoot_compact_finetune_plan.json'


def strata_key(row, package):
    with Image.open(package/row['image']) as image:w,h=image.size
    pixels=[min(i['xyxy'][2]-i['xyxy'][0],i['xyxy'][3]-i['xyxy'][1])*640/max(w,h) for i in row['instances']]
    size=min(pixels) if pixels else None
    bucket='negative' if size is None else '0_8' if size<=8 else '8_16' if size<=16 else '16_32' if size<=32 else 'over_32'
    return (tuple(sorted({i['building'] for i in row['instances']})),len(row['instances']),bucket,
            row['source_group'],row.get('context',{}).get('distractor','unknown'))


def balanced(rows, count, package, seed):
    buckets=defaultdict(list);rng=random.Random(seed)
    for row in rows:buckets[strata_key(row,package)].append(row)
    if len(rows)<count:raise ValueError('Insufficient distinct replay candidates')
    keys=sorted(buckets);rng.shuffle(keys)
    for group in buckets.values():rng.shuffle(group)
    selected=[]
    while len(selected)<count:
        for key in keys:
            if buckets[key]:selected.append(buckets[key].pop())
            if len(selected)==count:break
    return selected


def select(plan, records, package):
    replay=[r for r in records if r['split']=='train' and r['fine_tune_group']=='replay']
    selected=[]
    for kind,quotas in [('positive',plan['replay_selection']['positive']),('negative',plan['replay_selection']['negative'])]:
        for index,(branch,count) in enumerate(quotas.items()):
            pool=[r for r in replay if r['branch']==branch and bool(r['instances'])==(kind=='positive')]
            if branch=='blender_render' and kind=='negative':
                for offset,(context,number) in enumerate(plan['replay_selection']['negative_render_contexts'].items()):
                    sub=[r for r in pool if r.get('context',{}).get('distractor')==context]
                    selected.extend(balanced(sub,number,package,plan['seed']+100+offset))
            else:selected.extend(balanced(pool,count,package,plan['seed']+index+(20 if kind=='negative' else 0)))
    if len(selected)!=600 or len({r['image'] for r in selected})!=600:raise ValueError('Replay selection must contain 600 distinct pairs')
    all_direct={r['image'] for r in replay if r['branch']=='direct_real'}
    if not all_direct<={r['image'] for r in selected}:raise ValueError('Scarce direct-real inputs were lost')
    return selected+[r for r in records if r['split']!='train' or r['fine_tune_group']!='replay']


def configure(builder):
    p=builder.package;plan=builder.plan;budget=plan['training_proposal']
    config=yaml.safe_load((p/'config/finetune.yaml').read_text(encoding='utf8'))
    config.update(data=relative(builder.out/'data.yaml',p),epochs=budget['epochs'],lr0=budget['lr0'],
        batch=budget['batch'],imgsz=budget['imgsz'],warmup_epochs=budget['warmup_epochs'],
        warmup_bias_lr=budget['warmup_bias_lr'],patience=0,
        training_policy={'mode':'fixed_validation','save_period':5},
        description='Compact 1224-image fire fine-tuning; 10 epochs prepared, training not started.')
    (p/'config/finetune.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf8')
    launcher=yaml.safe_load((p/'config/finetune_launcher.yaml').read_text(encoding='utf8'))
    launcher.update(wall_seconds=budget['wall_seconds'])
    (p/'config/finetune_launcher.yaml').write_text(yaml.safe_dump(launcher,sort_keys=False),encoding='utf8')
    workspace=deepcopy(config)
    for key in ('data','model'):workspace[key]=relative(p/config[key])
    (ROOT/'lab/features/fire/training/config/fire_sticker_reshoot_finetune_v4.yaml').write_text(yaml.safe_dump(workspace,allow_unicode=True,sort_keys=False),encoding='utf8')
    return config


def link_review_evidence(row, package, pool):
    """Resolve inherited evidence through the base package map, without changing truth."""
    mapping=json.loads((package/'dataset/source_records/evidence_map.json').read_text(encoding='utf8'))
    original=row.get('source_review_evidence')
    if original:
        resolved=mapping.get(original,original)
    elif row['fine_tune_group']=='real_bin_negative':
        resolved=relative(pool/'source_records/reshoot_review.json',package)
    else:
        resolved=relative(pool/'source_records/wall_geometry.json',package)
    if not (package/resolved).is_file():raise FileNotFoundError('Missing packaged source evidence: '+resolved)
    row['source_review_evidence']=resolved
    row['source_evidence_resolution']={'inherited_reference':original,'package_reference':resolved,
                                      'sha256':digest(package/resolved)}
    if row['fine_tune_group']=='positive_with_real_bin':
        row['additional_source_review_evidence']=[relative(pool/'source_records/reshoot_review.json',package)]
    return row


def training_preflight(builder):
    """Exercise the real packaged consumer, including source evidence resolution."""
    sys.path.insert(0,str(builder.package/'scripts/common'))
    from engine.config import load_config
    from engine.data import preflight_dataset
    _,splits,report=preflight_dataset(builder.package,load_config(builder.package,'config/finetune.yaml'))
    result={'status':'passed','images':{k:len(v) for k,v in splits.items()},'data_report':report,
            'training_started':False,'scope':'Packaged engine training preflight; no model construction or optimization.'}
    write_json(builder.out/'training_preflight.json',result)
    write_json(builder.review/'training_preflight.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    plan=json.loads(PLAN.read_text(encoding='utf8'));pool=ROOT/plan['candidate_pool'];p=ROOT/plan['base_package'];out=ROOT/plan['output']
    source=json.loads((pool/'workspace_sources.json').read_text(encoding='utf8'))
    pool_sha={str(f.relative_to(p)):digest(f) for f in pool.rglob('*') if f.is_file()}
    b=Builder(plan,verify_only=args.verify)
    b.walls=json.loads((pool/'source_records/wall_geometry.json').read_text(encoding='utf8'))['records']
    if args.verify:
        b.rows=json.loads((out/'workspace_sources.json').read_text(encoding='utf8'))['records']
        preserved=json.loads((out/'candidate_pool_preservation.json').read_text(encoding='utf8'))
        if pool_sha!=preserved['files_sha256']:raise ValueError('Candidate pool differs from the preserved snapshot')
        report=b.verify()
        report.update(candidate_pool_files_byte_identical=len(pool_sha),
            replay_positive_images=sum(bool(r['instances']) for r in b.rows if r['split']=='train' and r['fine_tune_group']=='replay'),
            replay_negative_images=sum(not r['instances'] for r in b.rows if r['split']=='train' and r['fine_tune_group']=='replay'))
        write_json(out/'verification.json',report);write_json(b.review/'verification.json',report)
        training_preflight(b)
        print(json.dumps(report,ensure_ascii=False));return
    chosen=select(plan,source['records'],p)
    for original in chosen:
        row=deepcopy(original);row['candidate_origin']={'image':original['image'],'label':original['label']}
        for key in ('image','label'):
            old=p/original[key];new=out/old.relative_to(pool);new.parent.mkdir(parents=True,exist_ok=True)
            if digest(old)!=original[key+'_sha256']:raise ValueError('Candidate bytes changed: '+str(old))
            shutil.copy2(old,new);row[key]=relative(new,p)
        row['review_evidence']=relative(out/'verification.json',p)
        link_review_evidence(row,p,pool);b.rows.append(row)
    visual=json.loads((pool/'visual_review.json').read_text(encoding='utf8'))
    visual.update(method='Inherited visual review of byte-identical candidate images; no regeneration',
                  source_review=relative(pool/'visual_review.json',p),source_review_sha256=digest(pool/'visual_review.json'))
    visual['reviewed_images']=[v.replace(relative(pool,p)+'/',relative(out,p)+'/') for v in visual['reviewed_images']]
    write_json(out/'visual_review.json',visual)
    b.compile_plan(PLAN)
    report=b.verify()
    after={str(f.relative_to(p)):digest(f) for f in pool.rglob('*') if f.is_file()}
    if after!=pool_sha:raise ValueError('Candidate pool changed during subset selection')
    report['candidate_pool_files_byte_identical']=len(pool_sha)
    report['replay_positive_images']=sum(bool(r['instances']) for r in b.rows if r['split']=='train' and r['fine_tune_group']=='replay')
    report['replay_negative_images']=600-report['replay_positive_images']
    write_json(out/'verification.json',report);write_json(b.review/'verification.json',report)
    write_json(out/'candidate_pool_preservation.json',{'files_sha256':pool_sha,'status':'unchanged','candidate_pool':relative(pool,p)})
    write_json(out/'workspace_sources.json',{'task':'fire','model_key':'fire_sticker','records':b.rows,'plan':relative(PLAN),
        'generation_plan':relative(out/'generation_plan.json',p),'candidate_pool':relative(pool,p),
        'source_notes':[plan['independence_basis']],'training_started':False})
    write_json(out/'manifest.json',{'task':'fire','status':'ready_not_trained','plan':relative(PLAN),
        'generation_plan':relative(out/'generation_plan.json',p),'candidate_pool':relative(pool,p),
        'workspace_sources':relative(out/'workspace_sources.json',p),'verification':report})
    (out/'data.yaml').write_text(yaml.safe_dump({'names':['fire_sticker'],'train':'images/train','val':'images/val','test':'images/test'},sort_keys=False),encoding='utf8')
    configure(b)
    training_preflight(b)
    print(json.dumps({'status':'ready_not_trained','verification':report},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
