"""GT-only diagnosis of attainable recall and NMS conflicts; never inference."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from lab.features.e_bike.training.scripts.evaluation_audit import optimal_pairs
from lab.training_common.engine.reporting import box_iou
from lab.features.e_bike.training.scripts.candidate_verifier_experiment import foreground


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='runs/e_bike/candidate_pool_audit_20261118')
    args=parser.parse_args();out=(ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT/'runs/e_bike'):raise ValueError('Expected research output')
    out.mkdir(parents=True,exist_ok=False)
    exp=ROOT/'runs/e_bike/candidate_verifier_20261118'
    rows=[json.loads(line) for line in (exp/'evaluation/c025_i030_joint/predictions.jsonl').read_text().splitlines()]
    records=[];totals={}
    for row in rows:
        cached=json.loads((exp/'evaluation_raw'/(Path(row['source_image']).stem+'_verifier.json')).read_text())
        byid={p['raw_id']:p for p in cached['classified_candidates']}
        pool=[byid[i] for i in cached['proposal_ids'] if i in byid];gt=row['ground_truth'];variants={}
        for confidence in (.25,.10,.03,.01):
            for gate in (False,True):
                selected=[p for p in pool if p['raw_confidence']>confidence and (not gate or foreground(p['verifier_probs']))]
                matches=optimal_pairs(gt,selected,.5);key=f'c{confidence}_gate{gate}'
                variants[key]={'candidates':len(selected),'GT_matched_before_final_NMS':len(matches),
                               'all_GT_coverable_before_final_NMS':len(matches)==len(gt),
                               'unmatched_candidate_count':len(selected)-len(matches)}
                for group in (row['group'],)+(() if row['group']=='new_layout_holdout' else ('old26',)):
                    count=totals.setdefault(group,{}).setdefault(key,Counter());count.update(variants[key]);count['images']+=1;count['GT']+=len(gt)
        eligible=[p for p in pool if p['raw_confidence']>.25 and foreground(p['verifier_probs'])]
        final=row['predictions'];before=optimal_pairs(gt,eligible,.5);after=optimal_pairs(gt,final,.5)
        lost=sorted({i for i,_,_ in before}-{i for i,_,_ in after});conflicts=[]
        final_ids={p['raw_id'] for p in final}
        for i in lost:
            candidates=[p for p in eligible if p['raw_id'] not in final_ids and box_iou(gt[i]['xyxy'],p['xyxy'])>=.5]
            for candidate in candidates:
                suppressors=[p for p in final if p['raw_confidence']>=candidate['raw_confidence'] and box_iou(p['input_xyxy'],candidate['input_xyxy'])>.3]
                if not suppressors:continue
                suppressor=suppressors[0];other=max(range(len(gt)),key=lambda j:box_iou(gt[j]['xyxy'],suppressor['xyxy']))
                candidate_best=max(range(len(gt)),key=lambda j:box_iou(gt[j]['xyxy'],candidate['xyxy']))
                kind=('same_target_box_quality_conflict' if candidate_best==i and other==i else
                      'ambiguous_candidate_covers_multiple_GT' if candidate_best!=i else 'distinct_best_GT_conflict')
                conflicts.append({'missed_gt':i,'candidate_raw_id':candidate['raw_id'],'candidate_confidence':candidate['raw_confidence'],
                                  'suppressor_raw_id':suppressor['raw_id'],'suppressor_best_gt':other,
                                  'candidate_best_gt':candidate_best,'diagnostic_type':kind,
                                  'candidate_GT_ious':[box_iou(g['xyxy'],candidate['xyxy']) for g in gt],
                                  'suppressor_GT_ious':[box_iou(g['xyxy'],suppressor['xyxy']) for g in gt],
                                  'suppressor_gt_iou':box_iou(gt[other]['xyxy'],suppressor['xyxy']),
                                  'pair_iou':box_iou(suppressor['xyxy'],candidate['xyxy'])})
        records.append({'image':row['source_image'],'group':row['group'],'variants':variants,
                        'GT_lost_at_final_NMS':lost,'matching_capacity_before_after':[len(before),len(after)],
                        'conflict_candidates':conflicts})
    summary={'scope':'GT diagnostic of frozen broad proposal pool only. Oracle coverability ignores background filtering quality and cannot be reported as achieved inference accuracy.',
             'totals':totals,'records':records}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'old26':totals['old26'],'capacity_loss_images':[r['image'] for r in records if r['matching_capacity_before_after'][0]>r['matching_capacity_before_after'][1]]},ensure_ascii=False))


if __name__=='__main__':main()
