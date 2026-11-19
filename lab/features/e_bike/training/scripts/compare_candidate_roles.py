"""Separate semantic gating from posture using frozen candidate probabilities."""
import argparse
import copy
import html
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from datasets.scripts.common.images import write_image
from lab.features.e_bike.training.scripts.candidate_verifier_experiment import finish_pool
from lab.features.e_bike.training.scripts.data_ablation_experiment import error_overlay
from lab.features.e_bike.training.scripts.evaluation_audit import image_metrics
from lab.training_common.engine.runtime import setup_paths
from lab.training_common.engine.reporting import match_detections


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='runs/e_bike/candidate_role_split_20261118')
    args=parser.parse_args();out=(ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT/'runs/e_bike'):raise ValueError('Expected research output')
    out.mkdir(parents=True,exist_ok=False);setup_paths(out)
    modes={'semantic_gate_semantic_pose':'candidate_verifier_20261118',
           'semantic_gate_control_pose':'candidate_completeness_control_20261118',
           'semantic_gate_partial_pose':'candidate_completeness_20261118'}
    protocol={'confidence':.25,'final_nms_iou':.3,'geometry_gate':'candidate_verifier_20261118 foreground unchanged',
              'posture':'argmax fallen/upright in selected classifier, without its reject output',
              'modes':modes,'scope':'Post-hoc branch diagnostic designed after seeing hard-rejection loss. All58 are exposed development data; not independent test.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    source=ROOT/'runs/e_bike/candidate_verifier_20261118'
    reference=[json.loads(line) for line in (source/'evaluation/c025_i030_joint/predictions.jsonl').read_text().splitlines()]
    reports={};allrows={}
    for name,experiment in modes.items():
        folder=out/name;folder.mkdir();rows=[];groups={}
        for ref in reference:
            stem=Path(ref['source_image']).stem
            gate=read(source/'evaluation_raw'/(stem+'_verifier.json'))
            posture=read(ROOT/'runs/e_bike'/experiment/'evaluation_raw'/(stem+'_verifier.json'))
            poses={p['raw_id']:p for p in posture['classified_candidates']}
            candidates={p['raw_id']:p for p in gate['classified_candidates']};pool=[]
            for index in gate['proposal_ids']:
                if index not in candidates:continue
                p=copy.deepcopy(candidates[index]);q=poses[index]
                assert p['xyxy']==q['xyxy'] and p['raw_confidence']==q['raw_confidence']
                p['old_pose']=int(q['verifier_probs']['fallen']>q['verifier_probs']['upright']);pool.append(p)
            prediction=finish_pool(pool,.25,.3,'gate_old_pose')
            assert [p['xyxy'] for p in prediction]==[p['xyxy'] for p in ref['predictions']]
            matches,miss,extra=match_detections(ref['ground_truth'],prediction,.5)
            row={**ref,'predictions':prediction,'matches':matches,
                 'missed_gt_indices':miss,'false_prediction_indices':extra};metrics=image_metrics(row)
            rows.append(row)
            for group in (row['group'], 'all58')+(() if row['group']=='new_layout_holdout' else ('old26',)):
                g=groups.setdefault(group,{'images':0});g['images']+=1
                for key,value in metrics.items():g[key]=g.get(key,0)+int(value)
            write_image(folder/(stem+'.jpg'),error_overlay(row,name))
        (folder/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        reports[name]=groups;allrows[name]=rows
    (out/'summary.json').write_text(json.dumps({'protocol':protocol,'metrics':reports,'geometry_identical_to_semantic58':True},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    cards=[]
    for ref in reference:
        stem=Path(ref['source_image']).stem
        cards.append('<h2>'+html.escape(stem)+'</h2><div class=grid>'+''.join(f'<figure><figcaption>{name}</figcaption><a href="{name}/{stem}.jpg"><img loading=lazy src="{name}/{stem}.jpg"></a></figure>' for name in modes)+'</div>')
    (out/'review.html').write_text('<!doctype html><meta charset=utf-8><style>body{font:16px sans-serif;margin:24px}.grid{display:grid;grid-template-columns:repeat(3,1fr)}figure{margin:8px}img{width:100%}</style><h1>筛选与姿态分工</h1><p>三个方案的框、数量及排序完全一致，只改变姿态。已知开发集；事后诊断，不是独立验证。</p>'+''.join(cards),encoding='utf-8')
    print(json.dumps({name:{group:reports[name][group] for group in ('old26','original_real','new_layout_holdout')} for name in modes},ensure_ascii=False))


if __name__=='__main__':main()
