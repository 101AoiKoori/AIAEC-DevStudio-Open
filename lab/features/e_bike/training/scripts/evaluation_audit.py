"""Replay frozen predictions under stricter counting and matching diagnostics.

Historical metrics stay intact. This does not change native AP matching or deploy
an alternate matcher; maximum-cardinality assignment is a diagnostic sensitivity.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from lab.training_common.engine.reporting import box_iou, match_detections


def optimal_pairs(gt, pred, threshold, class_aware=False):
    if not gt or not pred:
        return []
    ious = np.array([[box_iou(a['xyxy'], b['xyxy']) for b in pred] for a in gt])
    allowed = ious >= threshold
    if class_aware:
        allowed &= np.array([[a['class_id'] == b['class_id'] for b in pred] for a in gt])
    # One additional valid pair dominates every possible change in summed IoU.
    benefit = allowed * (min(len(gt), len(pred)) + 1 + ious)
    rows, cols = linear_sum_assignment(benefit, maximize=True)
    return [(int(i), int(j), float(ious[i, j])) for i, j in zip(rows, cols) if allowed[i, j]]


def image_metrics(row, threshold=.5):
    gt, pred = row['ground_truth'], row['predictions']
    pairs = optimal_pairs(gt, pred, threshold)
    aware = optimal_pairs(gt, pred, threshold, True)
    greedy = match_detections([{**o, 'class_id': 0} for o in gt],
                              [{**o, 'class_id': 0} for o in pred], threshold)[0]
    errors = sum(gt[i]['class_id'] != pred[j]['class_id'] for i, j, _ in pairs)
    counts_gt = Counter(o['class_id'] for o in gt)
    counts_pred = Counter(o['class_id'] for o in pred)
    count_exact = counts_gt == counts_pred
    localized = len(pairs) == len(gt) == len(pred)
    strict = len(aware) == len(gt) == len(pred)
    ambiguous_predictions = sum(sum(a['class_id']==b['class_id'] and box_iou(a['xyxy'],b['xyxy'])>=threshold for a in gt)>1 for b in pred)
    return {'count_exact': count_exact, 'strict_instance_exact': strict,
            'count_exact_but_instance_failed': count_exact and not strict,
            'geometry_exact': localized, 'entity_tp': len(pairs),
            'entity_fp': len(pred)-len(pairs), 'entity_fn': len(gt)-len(pairs),
            'matched_pose_errors': errors, 'pose_aware_tp': len(aware),
            'greedy_pairs': len(greedy), 'maximum_pairs': len(pairs),
            'matching_cardinality_changed': len(greedy) != len(pairs),
            'predictions_compatible_with_multiple_GT': ambiguous_predictions,
            'strict_exact_with_assignment_ambiguity': strict and ambiguous_predictions>0,
            'absolute_pose_count_error': sum(abs(counts_gt[c]-counts_pred[c]) for c in (0, 1))}


def family(row):
    source = row['source_record']
    if source.get('source_image'):
        return source['source_image']
    details = source.get('research_details', {})
    if details.get('family_id'):
        return details['family_id']
    state = source['state_id']
    return str(source.get('background', {}).get('id')) + ':' + str((int(state.rsplit('_', 1)[1])-1)//2)


def aggregate(records):
    result = dict(Counter({key: sum(int(r[key]) for r in records) for key in records[0]}))
    result['images'] = len(records)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', default='runs/e_bike/candidate_verifier_20261118')
    parser.add_argument('--output', default='runs/e_bike/evaluation_audit_20261118')
    args = parser.parse_args()
    exp, out = (ROOT/args.experiment).resolve(), (ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT/'runs/e_bike'):
        raise ValueError('Research output must stay in runs/e_bike')
    out.mkdir(parents=True, exist_ok=False)
    snapshot = json.loads((exp/'evaluation_snapshot.json').read_text(encoding='utf-8'))
    sources = {r['image']: r for r in snapshot['records'] if r['split'] == 'val'}
    results, detailed = {}, []
    for path in sorted((exp/'evaluation').glob('*/predictions.jsonl')):
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        variants = {}
        for threshold in (.3, .5, .7):
            grouped, families = defaultdict(list), defaultdict(list)
            for row in rows:
                metrics = image_metrics(row, threshold)
                source = sources[row['source_image']]
                grouped[row['group']].append(metrics)
                if row['group'] != 'new_layout_holdout':
                    grouped['old26'].append(metrics)
                families[(row['group'], family(source))].append(metrics)
                detailed.append({'variant':path.parent.name, 'iou':threshold,
                                 'image':row['source_image'], 'family':family(source), **metrics})
            variants[str(threshold)] = {group: aggregate(records) for group, records in grouped.items()}
            variants[str(threshold)]['family_summaries'] = [
                {'group': group, 'family': key, 'members': len(records),
                 'count_exact_fraction': sum(r['count_exact'] for r in records)/len(records),
                 'all_variants_count_exact': all(r['count_exact'] for r in records),
                 'all_variants_instance_exact': all(r['strict_instance_exact'] for r in records)}
                for (group,key), records in families.items()]
        results[path.parent.name] = variants
    provenance = defaultdict(lambda: {'images':0, 'families':set(), 'declared_parent_groups':set()})
    reference = [json.loads(line) for line in (exp/'evaluation/c025_i030_raw/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    for row in reference:
        s = sources[row['source_image']]; item = provenance[row['group']]
        item['images'] += 1; item['families'].add(family(s))
        item['declared_parent_groups'].add(s.get('source_group', 'unknown'))
    provenance = {k:{'images':v['images'], 'families':sorted(v['families']),
                     'declared_parent_groups':sorted(v['declared_parent_groups'])} for k,v in provenance.items()}
    report = {'source_experiment':args.experiment, 'variants':results, 'provenance':provenance,
              'scope':'Known-source development replay; grouping does not undo prior exposure or establish source isolation.',
              'definitions':{'count_exact':'Upright and fallen counts both equal GT, irrespective of localization.',
                             'strict_instance_exact':'Every GT and prediction paired at threshold with correct posture; no extras or misses.',
                             'optimal_matching':'Maximum valid pair cardinality, then maximum total IoU; diagnostic, not native AP.',
                             'family_all':'Every mirror/pose variant in the family succeeds; robustness check, not independent sample.'},
              'references':['https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html',
                            'https://scikit-learn.org/stable/modules/cross_validation.html']}
    (out/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (out/'per_image.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in detailed), encoding='utf-8')
    print(json.dumps({k:results[k]['0.5']['old26'] for k in ('c025_i030_raw','c025_i030_joint')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
