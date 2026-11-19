"""Compare upright/fallen counts per image on the same recorded evaluations."""
import argparse
from collections import Counter
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[5]


def pose_counts(rows):
    images=[]
    for row in rows:
        totals=[]
        for key in ('ground_truth','predictions'):
            counts=Counter(item['class_id'] for item in row[key])
            if set(counts)-{0,1}:raise ValueError('Only upright/fallen class IDs are supported')
            totals.append({'upright':counts[0],'fallen':counts[1]})
        truth,prediction=totals
        images.append({'source_image':row['source_image'],'truth':truth,'prediction':prediction,
                       'both_pose_counts_exact':truth==prediction,
                       'all_instances_matched':not row['missed_gt_indices'] and not row['false_prediction_indices'],
                       'absolute_count_error':sum(abs(truth[k]-prediction[k]) for k in truth)})
    if not images:raise ValueError('Evaluation is empty')
    if len({r['source_image'] for r in images})!=len(images):raise ValueError('Repeated evaluation source image')
    return {'images':len(images),'exact_images':sum(r['both_pose_counts_exact'] for r in images),
            'exact_image_rate':sum(r['both_pose_counts_exact'] for r in images)/len(images),
            'all_instances_matched_images':sum(r['all_instances_matched'] for r in images),
            'mean_absolute_pose_count_error':sum(r['absolute_count_error'] for r in images)/len(images),
            'records':images}


def read_evaluation(path):
    path=(ROOT/path).resolve()
    if not path.is_relative_to(ROOT/'runs/e_bike'):raise ValueError('Expected an e_bike evaluation directory')
    summary=json.loads((path/'summary.json').read_text(encoding='utf-8'))
    if summary['model_key']!='e_bike':raise ValueError('Wrong task')
    rows=[json.loads(s) for s in (path/'predictions/predictions.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    return summary,rows,path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--before',required=True);p.add_argument('--after',required=True);args=p.parse_args()
    before,before_rows,_=read_evaluation(args.before);after,after_rows,path=read_evaluation(args.after)
    settings=lambda s:tuple(s['diagnostics'][k] for k in ('confidence_threshold','match_iou','nms_iou'))
    if settings(before)!=settings(after):raise ValueError('Diagnostic thresholds differ')
    first,last=pose_counts(before_rows),pose_counts(after_rows)
    if {r['source_image']:r['truth'] for r in first['records']}!={r['source_image']:r['truth'] for r in last['records']}:
        raise ValueError('Evaluations have different images or ground truth')
    report={'confidence_threshold':settings(before)[0],'before':first,'after':last,
            'scope':'Per-image upright/fallen count comparison on the same known-scene diagnostic val; mirrors and neighbouring video frames are not independent captures.'}
    destination=path/'pose_count_comparison.json';destination.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:{field:value for field,value in report[k].items() if field!='records'} for k in ('before','after')},ensure_ascii=False))


if __name__=='__main__':main()
