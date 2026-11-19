"""Equal-slot controlled e_bike data ablation and detection/posture error audit."""
import argparse
from collections import Counter
import copy
import hashlib
import html
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

import cv2
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from datasets.scripts.common.config import load_task_config
from datasets.scripts.common.images import read_image,write_image
from lab.features.e_bike.training.scripts.keypoint_experiment import read,write,evaluate
from lab.features.e_bike.training.scripts.two_stage_experiment import compact,crop
from lab.features.e_bike.training.scripts.part_hull_experiment import diagnostic
from lab.training_common.engine.reporting import match_detections
from lab.training_common.engine.runtime import setup_paths,prepare_ultralytics

REFERENCE=ROOT/'runs/e_bike/focused_comparison_20261118/before/evaluation/box_03/predictions.jsonl'


def choose_additions(old,focused,quota,seed):
    rng=random.Random(seed)
    train=[r for r in old['records'] if r['split']=='train']
    selected={'control':rng.sample(train,quota)}
    for arm,kind in [('negatives','mined_vehicle_free_background'),('crops','complete_real_crop')]:
        pool=[r for r in focused['records'] if r['split']=='train' and r.get('condition')==kind]
        selected[arm]=rng.sample(pool,quota)
    groups={}
    for row in focused['records']:
        if row['split']=='train' and row.get('condition')=='focused_composite':
            family=row['source_record']['research_details']['family_id'];groups.setdefault(family,[]).append(row)
    backgrounds=sorted({g.rsplit('_',1)[0] for g in groups});per_background=quota//(4*len(backgrounds))
    if per_background*4*len(backgrounds)!=quota:raise ValueError('Synthetic quota must retain full four-image families balanced across backgrounds')
    selected['synthetic']=[]
    for bg in backgrounds:
        families=rng.sample(sorted(g for g in groups if g.rsplit('_',1)[0]==bg),per_background)
        for family in families:
            if len(groups[family])!=4:raise ValueError('Expected complete posture/mirror quartet')
            selected['synthetic']+=groups[family]
    return selected


def parent_frames(old,crops):
    by_image={r['image']:r for r in old['records'] if r['split']=='train'}
    return [by_image[r['source_record']['research_details']['parent_image']] for r in crops]


def prepare(exp,recipe_key='ablation_recipe'):
    _,source=load_task_config('e_bike');recipe=read(ROOT/source[recipe_key])
    old=read(ROOT/recipe['old_inputs']);focused=read(ROOT/recipe['focused_inputs']);evaluation=read(ROOT/recipe['evaluation_snapshot'])
    selections=choose_additions(old,focused,recipe['added_images_per_arm'],recipe['seed'])
    if 'real_parent' in recipe['arms']:selections['real_parent']=parent_frames(old,selections['crops'])
    exp.mkdir(parents=True,exist_ok=False);write(exp/'recipe_snapshot.json',recipe)
    (exp/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    write(exp/'evaluation_snapshot.json',evaluation)
    original_train=[r for r in old['records'] if r['split']=='train'];original_val=[r for r in old['records'] if r['split']=='val']
    holdout_images={r['image'] for r in evaluation['records']}
    prepared={}
    for arm in recipe['arms']:
        path=exp/'inputs'/arm;metadata=[];slots=original_train+selections[arm]
        assert len(slots)==len(original_train)+recipe['added_images_per_arm']
        for split,rows in [('train',slots),('val',original_val)]:
            for folder in ('images','labels'):(path/split/folder).mkdir(parents=True)
            for index,row in enumerate(rows):
                is_add=split=='train' and index>=len(original_train)
                if split=='train':assert row['image'] not in holdout_images
                name=f'{index:04d}_{Path(row["image"]).name}'
                image=path/split/'images'/name;label=path/split/'labels'/(image.stem+'.txt')
                shutil.copyfile(ROOT/row['image'],image)
                text=(ROOT/row['label']).read_text(encoding='utf-8')
                label.write_text(''.join('0 '+' '.join(line.split()[1:])+'\n' for line in text.splitlines()),encoding='utf-8')
                record=copy.deepcopy(row);record.update(image=image.relative_to(ROOT).as_posix(),label=label.relative_to(ROOT).as_posix(),
                    workspace_image=image.relative_to(ROOT).as_posix(),workspace_label=label.relative_to(ROOT).as_posix(),
                    ablation_source_image=row['image'],ablation_added_slot=is_add,ablation_control_resample=arm=='control' and is_add)
                metadata.append(record)
        write(path/'workspace_sources.json',{'model_key':'e_bike','selection_mode':'manifest_only','records':metadata,'source_reuse':[]})
        write(path/'data.yaml',{'names':['e_bike'],'train':'train/images','val':'val/images'})
        config=yaml.safe_load((ROOT/'runs/e_bike/focused_inputs_20261118/train.yaml').read_text())
        config.update(data=(path/'data.yaml').relative_to(ROOT).as_posix(),model=recipe['initial_weights'],
                      output_root=(exp/'models').relative_to(ROOT).as_posix(),epochs=recipe['epochs'],
                      seed=recipe['seed'],patience=0,training_policy={'mode':'fixed_validation','save_period':10},
                      description=f'Controlled{len(slots)}-slot data ablation arm {arm}; last checkpoint primary')
        config['source_notes']=[recipe['scope'],f'Arm{arm}; actual added source slots in selection.json. Primary last20; no early stopping.']
        (path/'train.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
        write(path/'selection.json',{'arm':arm,'base_train_images':len(original_train),'added_slots':[{ 'image':r['image'],'label':r['label'],
            'source_group':r.get('source_group'),'condition':r.get('condition'),'resample':arm in ('control','real_parent')} for r in selections[arm]],'training_slots':len(slots),'validation_images':26})
        eval_path=exp/'evaluation'/arm;eval_path.mkdir(parents=True);write(eval_path/'input_snapshot.json',evaluation)
        prepared[arm]={'train_slots':len(slots),'unique_parent_input_paths':len({r['image'] for r in slots}),
                       'val_images':26,'data':(path/'data.yaml').relative_to(ROOT).as_posix()}
    write(exp/'preparation.json',{'arms':prepared,'old_val_not_used_in_training':True,'new_layouts_not_used_in_training':True,
                                 'scope':recipe['scope']})
    print(json.dumps(prepared),flush=True)


def train(exp,arm):
    recipe=read(exp/'recipe_snapshot.json');arms=recipe['arms'] if arm=='all' else [arm]
    (exp/'controller_logs').mkdir(exist_ok=True)
    for name in arms:
        run=exp/'models'/name
        if run.exists():
            status=read(run/'metadata/train_status.json') if (run/'metadata/train_status.json').exists() else {}
            if status.get('status')=='complete':continue
            raise ValueError(f'Existing incomplete run: use common explicit resume for {run}')
        command=[sys.executable,str(ROOT/'lab/features/e_bike/training/scripts/train_e_bike_v4.py'),'train',
                 '--config',(exp/'inputs'/name/'train.yaml').relative_to(ROOT).as_posix(),
                 '--run-name',name,'--wall-seconds',str(recipe['wall_seconds_per_arm'])]
        print('START '+name,flush=True)
        with (exp/'controller_logs'/f'{name}.log').open('w',encoding='utf-8') as stream:
            subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        print('COMPLETE '+name,flush=True)


def entity_pairs(row):
    return match_detections([{**o,'class_id':0} for o in row['ground_truth']],
                            [{**o,'class_id':0} for o in row['predictions']],.5)


def error_summary(rows):
    pose_errors=Counter();pose_only=[];confusion=np.zeros((2,2),dtype=int)
    for row in rows:
        pairs,miss,extra=entity_pairs(row);wrong=0
        for i,j,_ in pairs:
            a,b=row['ground_truth'][i]['class_id'],row['predictions'][j]['class_id'];confusion[a,b]+=1;wrong+=int(a!=b)
        pose_errors[row['group']]+=wrong
        if not miss and not extra and wrong:pose_only.append(row['source_image'])
    return {'matched_box_posture_errors_by_group':dict(pose_errors),'matched_posture_confusion_gt_rows_pred_columns':confusion.tolist(),
            'fully_localized_images_with_posture_errors':pose_only}


def paired_errors(before,after):
    bp,_,_=entity_pairs(before);ap,_,_=entity_pairs(after)
    bm={i:j for i,j,_ in bp};am={i:j for i,j,_ in ap};common=set(bm)&set(am)
    wrong_before={i for i in common if before['ground_truth'][i]['class_id']!=before['predictions'][bm[i]]['class_id']}
    wrong_after={i for i in common if after['ground_truth'][i]['class_id']!=after['predictions'][am[i]]['class_id']}
    return {'source_image':before['source_image'],'group':before['group'],
            'recovered_gt_indices':sorted(set(am)-set(bm)),'lost_gt_indices':sorted(set(bm)-set(am)),
            'fixed_posture_gt_indices':sorted(wrong_before-wrong_after),'new_posture_errors_gt_indices':sorted(wrong_after-wrong_before)}


def oracle_posture(exp,rows):
    """GT-box classification is a diagnostic bound, never an inference proposal."""
    setup_paths(exp);YOLO=prepare_ultralytics(exp);recipe=read(exp/'recipe_snapshot.json')
    classifier=YOLO(str(ROOT/recipe['classifier']));output=[]
    for row in rows:
        frame=read_image(ROOT/row['source_image']);gt=row['ground_truth']
        result=classifier.predict(source=[crop(frame,g['xyxy'])[0] for g in gt],imgsz=160,device='0',verbose=False)
        pred=[{'xyxy':g['xyxy'],'class_id':('upright','fallen').index(classifier.names[r.probs.top1])} for g,r in zip(gt,result)]
        matches,miss,extra=match_detections(gt,pred,.5)
        output.append({**row,'predictions':pred,'matches':matches,'missed_gt_indices':miss,'false_prediction_indices':extra})
    report={}
    for group in ('old26','original_real','synthetic_heldout','new_layout_holdout','all58'):
        chosen=output if group=='all58' else [r for r in output if (r['group']!='new_layout_holdout' if group=='old26' else r['group']==group)]
        report[group]={'instances':sum(len(r['ground_truth']) for r in chosen),
                       'correct_postures':sum(sum(g['class_id']==p['class_id'] for g,p in zip(r['ground_truth'],r['predictions'])) for r in chosen),
                       'oracle_pose_counting':compact(chosen)}
    write(exp/'oracle_posture.json',{'groups':report,'scope':'GT boxes supplied; diagnosis only, no detector recall or deployment claim.'})
    (exp/'oracle_predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output),encoding='utf-8')
    return report


def error_overlay(row,name):
    frame=read_image(ROOT/row['source_image']);pairs,miss,extra=entity_pairs(row)
    for i,p in enumerate(row['predictions']):
        color=(255,0,255) if i in extra else ((40,220,40) if p['class_id']==0 else (0,160,255))
        x1,y1,x2,y2=map(round,p['xyxy']);cv2.rectangle(frame,(x1,y1),(x2,y2),color,1)
    wrong=0
    for i,j,_ in pairs:
        g,p=row['ground_truth'][i],row['predictions'][j]
        if g['class_id']!=p['class_id']:
            wrong+=1;x1,y1,x2,y2=map(round,g['xyxy']);cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
            cv2.putText(frame,f'G{i} pose',(max(0,x1),max(12,y1-4)),0,.38,(0,255,255),1)
    for i in miss:
        x1,y1,x2,y2=map(round,row['ground_truth'][i]['xyxy']);cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.putText(frame,f'G{i} miss',(max(0,x1),max(12,y1-4)),0,.38,(0,0,255),1)
    frame=cv2.copyMakeBorder(frame,28,0,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
    cv2.putText(frame,f'{name} FP={len(extra)} FN={len(miss)} PoseErr={wrong}',(5,19),0,.45,(255,255,255),1)
    return frame


def evaluate_arms(exp):
    recipe=read(exp/'recipe_snapshot.json')
    for name in recipe['arms']:
        run=exp/'models'/name
        status=read(run/'metadata/train_status.json');assert status['status']=='complete'
        folder=exp/'evaluation'/name
        if (folder/'evaluation/summary.json').exists():continue
        evaluate(folder,run/'training/weights/last.pt')


def summarize(exp,control_study=None):
    recipe=read(exp/'recipe_snapshot.json');all_rows={'reference':[json.loads(l) for l in REFERENCE.read_text().splitlines()]}
    reports={};steps={};logs={}
    locations={name:exp for name in recipe['arms']}
    if control_study:
        locations={**{name:control_study for name in read(control_study/'recipe_snapshot.json')['arms']},**locations}
    for name,owner in locations.items():
        folder=owner/'evaluation'/name/'evaluation';all_rows[name]=[json.loads(l) for l in (folder/'box_03/predictions.jsonl').read_text().splitlines()]
        logs[name]=read(owner/'models'/name/'metadata/optimization_summary.json');steps[name]=logs[name]['actual_optimizer_steps']
    if len(set(steps.values()))!=1:raise ValueError('Actual optimizer updates differ; cannot report matched-budget ablation')
    trajectories={name:[json.loads(l) for l in (owner/'models'/name/'metadata/optimizer_updates.jsonl').read_text().splitlines()] for name,owner in locations.items()}
    assert all(trajectory==trajectories['control'] for trajectory in trajectories.values()),'Optimizer LR/accumulation trajectories differ'
    for name,rows in all_rows.items():
        assert [(r['source_image'],r['ground_truth']) for r in rows]==[(r['source_image'],r['ground_truth']) for r in all_rows['reference']]
        reports[name]={}
        for group in ('old26','original_real','synthetic_heldout','new_layout_holdout','all58'):
            chosen=rows if group=='all58' else [r for r in rows if (r['group']!='new_layout_holdout' if group=='old26' else r['group']==group)]
            reports[name][group]={**compact(chosen),**error_summary(chosen),'duplicate_predictions':sum(diagnostic(r['ground_truth'],r['predictions'])['duplicate_predictions'] for r in chosen)}
    changes={name:[paired_errors(b,a) for b,a in zip(all_rows['control'],all_rows[name])] for name in locations if name!='control'}
    if control_study:
        old_summary=read(control_study/'summary.json')
        assert old_summary['weights'][recipe['classifier']]==hashlib.sha256((ROOT/recipe['classifier']).read_bytes()).hexdigest()
        oracle=read(control_study/'oracle_posture.json')['groups']
        shutil.copyfile(control_study/'oracle_predictions.jsonl',exp/'oracle_predictions.jsonl')
        shutil.copyfile(control_study/'oracle_posture.json',exp/'oracle_posture.json')
    else:oracle=oracle_posture(exp,all_rows['reference'])
    oracle_rows={r['source_image']:r for r in [json.loads(l) for l in (exp/'oracle_predictions.jsonl').read_text().splitlines()]}
    crop_pose_diagnosis={}
    for name,rows in all_rows.items():
        groups={}
        for row in rows:
            counter=groups.setdefault(row['group'],Counter());pairs,_,_=entity_pairs(row)
            for i,j,_ in pairs:
                truth=row['ground_truth'][i]['class_id'];pred=row['predictions'][j]['class_id'];oracle_pred=oracle_rows[row['source_image']]['predictions'][i]['class_id']
                if pred!=truth:counter['GT_crop_correct_predicted_crop_wrong' if oracle_pred==truth else 'both_GT_and_predicted_crop_wrong']+=1
        crop_pose_diagnosis[name]={g:dict(c) for g,c in groups.items()}
    train_labels={}
    for name,owner in locations.items():
        counts=[len(p.read_text().splitlines()) for p in (owner/'inputs'/name/'train/labels').glob('*.txt')]
        train_labels[name]={'images':len(counts),'positive_images':sum(n>0 for n in counts),
                            'negative_images':sum(n==0 for n in counts),'instances':sum(counts)}
    result={'status':'controlled_data_ablation_complete','recipe':recipe,'groups':reports,'optimizer_steps':steps,
            'actual_training_labels':train_labels,
            'optimizer_lr_accumulation_trajectories_identical':True,
            'paired_changes_vs_matched_control':changes,'oracle_posture':oracle,'crop_posture_diagnosis':crop_pose_diagnosis,
            'training_statuses':{name:read(owner/'models'/name/'metadata/train_status.json') for name,owner in locations.items()},
            'weights':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [recipe['initial_weights'],recipe['classifier'],*[(owner/'models'/name/'training/weights/last.pt').relative_to(ROOT).as_posix() for name,owner in locations.items()]]},
            'primary_control_study':control_study.relative_to(ROOT).as_posix() if control_study else None,
            'followup_designed_after_primary_results':bool(control_study),
            'review':(exp/'review.html').relative_to(ROOT).as_posix(),
            'scope':recipe['scope'],'limits':['Single seed and48 added slots per arm; synthetic is12 complete families, not whole128 batch.','Old26 and new32 are already examined shared-source diagnostics.','Data types alter class/instance/background frequencies; this tests each48-slot intervention, not a universal causal ranking.','Classifier unchanged. Primary result uses last20, not count-selected best.']}
    write(exp/'summary.json',result)
    table='<table border="1" cellpadding="8"><tr><th>组</th><th>旧26全对</th><th>实拍10全对</th><th>新32全对</th><th>旧26 FP/FN</th><th>旧26匹配框姿态错误</th><th>实际更新</th></tr>'
    for name,s in reports.items():
        e=s['old26']['entity_counts_and_matching'];wrong=sum(s['old26']['matched_box_posture_errors_by_group'].values())
        table+=f'<tr><td>{name}</td><td>{s["old26"]["exact_images"]}/26</td><td>{s["original_real"]["exact_images"]}/10</td><td>{s["new_layout_holdout"]["exact_images"]}/32</td><td>{e["fp"]}/{e["fn"]}</td><td>{wrong}</td><td>{steps.get(name,"未续训")}</td></tr>'
    cards=[]
    for i,row in enumerate(all_rows['reference']):
        stem=Path(row['source_image']).stem;figures=[]
        for name in all_rows:
            target=exp/'error_previews'/name/(stem+'.jpg');target.parent.mkdir(parents=True,exist_ok=True)
            write_image(target,error_overlay(all_rows[name][i],name))
            link=os.path.relpath(target,exp).replace('\\','/')
            figures.append(f'<figure><figcaption>{name}</figcaption><a href="{link}"><img loading="lazy" src="{link}"></a></figure>')
        cards.append(f'<h2>{html.escape(stem)} ({row["group"]})</h2><div class="grid">'+''.join(figures)+'</div>')
    (exp/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>等预算数据消融</title><style>body{font:16px sans-serif;background:#eee;margin:24px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}figure{margin:0;background:white;padding:8px}img{width:100%}</style><h1>204训练槽位／20轮／相同实际更新：数据消融</h1><p>reference原17/26；control补48旧图重采样；negatives补48硬负例；crops补48完整实拍裁片；synthetic补12族48张合成。GT只用于评估。均为单种子、同源开发对照；姿态分类器未训练。红=漏车GT，紫=误检，黄=匹配框内姿态错误；绿/橙为预测直立/倒伏。</p>'+table+'</table><p>GT框姿态分类诊断：旧26 '+str(oracle['old26']['correct_postures'])+'/'+str(oracle['old26']['instances'])+'，新32 '+str(oracle['new_layout_holdout']['correct_postures'])+'/'+str(oracle['new_layout_holdout']['instances'])+'；仅是提供真值框的诊断，不能当检测成绩。</p>'+''.join(cards),encoding='utf-8')
    print(json.dumps({n:{g:s[g]['exact_images'] for g in ('old26','original_real','new_layout_holdout')} for n,s in reports.items()}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('prepare','train','evaluate','summarize'))
    p.add_argument('--experiment',default='runs/e_bike/data_ablation_20261118');p.add_argument('--arm',choices=('all','control','negatives','crops','synthetic','real_parent'),default='all')
    p.add_argument('--recipe-key',default='ablation_recipe',choices=('ablation_recipe','sampling_recipe'))
    p.add_argument('--control-study',help='Existing four-arm study for paired parent-frame follow-up summary')
    args=p.parse_args();exp=(ROOT/args.experiment).resolve()
    if not exp.is_relative_to(ROOT/'runs/e_bike'):raise ValueError('Expected e_bike research output')
    cv2.setNumThreads(1);torch.set_num_threads(2)
    if args.phase=='prepare':prepare(exp,args.recipe_key)
    elif args.phase=='train':train(exp,args.arm)
    elif args.phase=='evaluate':evaluate_arms(exp)
    else:
        control=(ROOT/args.control_study).resolve() if args.control_study else None
        if control and not control.is_relative_to(ROOT/'runs/e_bike'):raise ValueError('Expected e_bike control study')
        summarize(exp,control)


if __name__=='__main__':main()
