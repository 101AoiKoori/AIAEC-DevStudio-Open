"""Reproduce semantic candidate verification with fixed data and declared seeds."""
import argparse
import copy
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from datasets.scripts.common.config import load_task_config
from lab.features.e_bike.training.scripts.evaluation_audit import image_metrics, family


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def prepare(exp):
    _,source=load_task_config('e_bike');protocol=read(ROOT/source['seed_study_recipe'])
    exp.mkdir(parents=True,exist_ok=False);write(exp/'protocol.json',protocol)
    reference=ROOT/protocol['reference_experiment'];original=read(reference/'crop_manifest.json');checks=[]
    for seed in protocol['new_seeds']:
        target=exp/f'seed_{seed}';target.mkdir();shutil.copytree(reference/'classifier_data',target/'classifier_data')
        recipe=read(reference/'recipe_snapshot.json');recipe['seed']=seed;recipe['epochs']=protocol['epochs']
        recipe['evaluation_detector_settings']=[[protocol['primary_confidence'],protocol['primary_nms_iou']]]
        recipe['purpose']='semantic_verifier_seed_stability';recipe['scope']=protocol['scope'];write(target/'recipe_snapshot.json',recipe)
        manifest=copy.deepcopy(original)
        for row in manifest['records']:
            before=ROOT/row['image'];after=target/'classifier_data'/row['split']/row['label']/before.name
            assert hashlib.sha256(before.read_bytes()).digest()==hashlib.sha256(after.read_bytes()).digest()
            row['image']=after.relative_to(ROOT).as_posix()
        write(target/'crop_manifest.json',manifest);shutil.copyfile(reference/'evaluation_snapshot.json',target/'evaluation_snapshot.json')
        (target/'source_snapshot.py').write_bytes((ROOT/'lab/features/e_bike/training/scripts/candidate_verifier_experiment.py').read_bytes())
        checks.append({'seed':seed,'identical_copied_crops':len(manifest['records']),'split_unchanged':True,'crop_distribution':manifest['distribution']})
    write(exp/'preparation.json',{'checks':checks,'reference':protocol['reference_experiment'],'warmstart':read(reference/'recipe_snapshot.json')['initial_classifier']})
    print(json.dumps(checks),flush=True)


def execute(exp,phase):
    protocol=read(exp/'protocol.json');script=ROOT/'lab/features/e_bike/training/scripts/candidate_verifier_experiment.py'
    for seed in protocol['new_seeds']:
        target=exp/f'seed_{seed}';print(f'START {phase} seed={seed}',flush=True)
        if phase=='train' and (target/'classifier').exists():
            status=read(target/'classifier_status.json');options=read(target/'classifier_options.json')
            if status['status']!='complete' or options['seed']!=seed or options['epochs']!=protocol['epochs'] or not (ROOT/status['primary_weights']).is_file():
                raise RuntimeError(f'Existing training requires inspection: {target}')
            print(f'SKIP verified complete training seed={seed}',flush=True);continue
        if phase=='evaluate' and (target/'summary.json').exists():
            summary=read(target/'summary.json');weight=target/'classifier/weights/last.pt'
            if summary['baseline_reproduced_images']!=58 or summary['weights'][weight.relative_to(ROOT).as_posix()]!=hashlib.sha256(weight.read_bytes()).hexdigest():
                raise RuntimeError(f'Existing evaluation does not match current weights: {target}')
            print(f'SKIP verified complete evaluation seed={seed}',flush=True);continue
        with (target/f'{phase}_controller.log').open('w',encoding='utf-8') as stream:
            subprocess.run([sys.executable,'-X','utf8','-B',str(script),phase,'--experiment',target.relative_to(ROOT).as_posix()],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        print(f'COMPLETE {phase} seed={seed}',flush=True)


def summarize(exp):
    protocol=read(exp/'protocol.json');reference=ROOT/protocol['reference_experiment']
    owners={protocol['reference_seed']:reference,**{seed:exp/f'seed_{seed}' for seed in protocol['new_seeds']}}
    variants={};rows={};weights={}
    source={r['image']:r for r in read(reference/'evaluation_snapshot.json')['records']}
    for seed,owner in owners.items():
        rows[seed]=[json.loads(line) for line in (owner/'evaluation/c025_i030_joint/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        assert [(r['source_image'],r['ground_truth']) for r in rows[seed]]==[(r['source_image'],r['ground_truth']) for r in rows[protocol['reference_seed']]]
        report={}
        for group in ('old26','original_real','synthetic_heldout','new_layout_holdout'):
            selected=[r for r in rows[seed] if (r['group']!='new_layout_holdout' if group=='old26' else r['group']==group)]
            metrics=[image_metrics(r) for r in selected];families={}
            for r,m in zip(selected,metrics):families.setdefault(family(source[r['source_image']]),[]).append(m)
            report[group]={'images':len(metrics),**{key:sum(m[key] for m in metrics) for key in ('count_exact','strict_instance_exact','entity_fp','entity_fn','matched_pose_errors')},
                           'families':len(families),'all_variant_count_exact_families':sum(all(m['count_exact'] for m in ms) for ms in families.values()),
                           'all_variant_instance_exact_families':sum(all(m['strict_instance_exact'] for m in ms) for ms in families.values())}
        variants[str(seed)]={'metrics':report,'training':read(owner/'classifier_status.json')}
        path=owner/'classifier/weights/last.pt';weights[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    per_image=[]
    for index,base in enumerate(rows[protocol['reference_seed']]):
        outcomes={str(seed):image_metrics(items[index]) for seed,items in rows.items()}
        per_image.append({'image':base['source_image'],'group':base['group'],'family':family(source[base['source_image']]),
                          'all_seeds_count_exact':all(m['count_exact'] for m in outcomes.values()),
                          'all_seeds_instance_exact':all(m['strict_instance_exact'] for m in outcomes.values()),'outcomes':outcomes})
    summary={'protocol':protocol,'seeds':variants,'weights':weights,'per_image':per_image,
             'scope':'Training stochasticity only, same inherited backbone and exposed source groups; not independent-source replication.'}
    write(exp/'summary.json',summary)
    table='<table border=1 cellpadding=7><tr><th>seed</th><th>旧26数量/严格</th><th>实拍数量/严格</th><th>布局32数量/严格</th><th>旧26FP/FN</th></tr>'
    for seed,v in variants.items():
        a,b,c=(v['metrics'][g] for g in ('old26','original_real','new_layout_holdout'))
        table+=f'<tr><td>{seed}</td><td>{a["count_exact"]}/{a["strict_instance_exact"]}</td><td>{b["count_exact"]}/{b["strict_instance_exact"]}</td><td>{c["count_exact"]}/{c["strict_instance_exact"]}</td><td>{a["entity_fp"]}/{a["entity_fn"]}</td></tr>'
    cards=[]
    for base in rows[protocol['reference_seed']]:
        stem=Path(base['source_image']).stem;figures=[]
        for seed,owner in owners.items():
            path=owner/'evaluation/c025_i030_joint'/(stem+'.jpg');relative=__import__('os').path.relpath(path,exp).replace('\\','/')
            figures.append(f'<figure><figcaption>seed{seed}</figcaption><a href="{relative}"><img loading=lazy src="{relative}"></a></figure>')
        cards.append('<h2>'+html.escape(stem)+'</h2><div class=grid>'+''.join(figures)+'</div>')
    (exp/'review.html').write_text('<!doctype html><meta charset=utf-8><style>body{font:16px sans-serif;margin:24px}.grid{display:grid;grid-template-columns:repeat(3,1fr)}figure{margin:8px}img{width:100%}</style><h1>相同数据的训练种子稳定性</h1><p>数量/严格逐车全对并列；原图、镜像和布局共享来源。这不是独立场景验证。</p>'+table+'</table>'+''.join(cards),encoding='utf-8')
    print(json.dumps({seed:v['metrics'] for seed,v in variants.items()},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('prepare','train','evaluate','summarize'))
    parser.add_argument('--experiment',default='runs/e_bike/verifier_seed_study_20261118');args=parser.parse_args();exp=(ROOT/args.experiment).resolve()
    if not exp.is_relative_to(ROOT/'runs/e_bike'):raise ValueError('Expected research output')
    if args.phase=='prepare':prepare(exp)
    elif args.phase=='summarize':summarize(exp)
    else:execute(exp,args.phase)


if __name__=='__main__':main()
