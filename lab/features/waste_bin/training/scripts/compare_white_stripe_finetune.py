"""Compare existing baseline/fine-tune evaluations by the actual dataset groups."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[5]
DATA=ROOT/'datasets/annotated/waste_bin/waste_sticker_hard_negative'


def group_report(evaluation, records):
    summary=json.loads((evaluation/'summary.json').read_text(encoding='utf-8'))
    with (evaluation/'per_image.csv').open(encoding='utf-8-sig') as stream:
        rows=list(csv.DictReader(stream))
    expected={row['image']:row for row in records if row['split']==summary['split']}
    if len(rows)!=len(expected) or {r['source_image'] for r in rows}!=set(expected):
        raise ValueError('Evaluation images differ from the declared comparison dataset')
    groups=defaultdict(lambda:dict(images=0,tp=0,fp=0,fn=0,negative_images=0,
                                  negative_false_positive_images=0,exact_images=0))
    for row in rows:
        source=expected[row['source_image']]
        group=groups[source['group']]
        group['images']+=1
        for key,column in [('tp','true_positives'),('fp','false_positives'),('fn','false_negatives')]:
            group[key]+=int(row[column])
        negative=int(row['ground_truth'])==0
        group['negative_images']+=negative
        group['negative_false_positive_images']+=negative and int(row['false_positives'])>0
        group['exact_images']+=int(row['false_positives'])==0 and int(row['false_negatives'])==0
    for group in groups.values():
        group['false_boxes_per_100_images']=100*group['fp']/group['images']
        group['recall']=group['tp']/(group['tp']+group['fn']) if group['tp']+group['fn'] else None
        group['precision']=group['tp']/(group['tp']+group['fp']) if group['tp']+group['fp'] else None
        group['exact_image_rate']=group['exact_images']/group['images']
        group['negative_false_positive_rate']=(group['negative_false_positive_images']/group['negative_images']
                                                if group['negative_images'] else None)
    if 'nms_iou' not in summary['diagnostics']:
        raise ValueError('Comparison requires recorded diagnostic NMS IoU')
    return {'evaluation':evaluation.relative_to(ROOT).as_posix(),
            'evaluation_config':{**summary['evaluation_config'],'nms_iou':summary['diagnostics']['nms_iou']},'metrics':summary['metrics'],
            'diagnostics':{k:summary['diagnostics'][k] for k in ['true_positives','false_positives','false_negatives',
                                                                'negative_images','negative_images_with_false_positives']},
            'groups':dict(groups)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',required=True);parser.add_argument('--finetuned',required=True)
    parser.add_argument('--manifest',type=Path,default=DATA/'manifest.json')
    parser.add_argument('--baseline-manifest',type=Path,
                        help='Previous manifest for byte-equivalent evaluation reused at a new dataset path')
    parser.add_argument('--out',required=True);args=parser.parse_args()
    baseline=(ROOT/args.baseline).resolve();fine=(ROOT/args.finetuned).resolve();out=(ROOT/args.out).resolve()
    if not all(p.is_relative_to(ROOT) for p in (baseline,fine,out)):
        raise ValueError('Comparison paths must stay in this workspace')
    records=json.loads((ROOT/args.manifest).read_text(encoding='utf-8'))['records']
    baseline_records=records
    if args.baseline_manifest:
        baseline_records=json.loads((ROOT/args.baseline_manifest).read_text(encoding='utf-8'))['records']
        split=json.loads((baseline/'summary.json').read_text(encoding='utf-8'))['split']
        def identities(items):
            selected=[r for r in items if r['split']==split]
            if any(not r.get('decoded_rgb_sha256') for r in selected):
                raise ValueError('Path relocation comparison requires decoded RGB identities')
            return sorted((r['decoded_rgb_sha256'],r['group'],(ROOT/r['label']).read_text(encoding='utf-8'))
                          for r in selected)
        if identities(baseline_records)!=identities(records):
            raise ValueError('Relocated comparison differs in RGB, labels, or groups')
    before,after=group_report(baseline,baseline_records),group_report(fine,records)
    if before['evaluation_config']!=after['evaluation_config']:
        raise ValueError('Comparison thresholds, input size or class order differ')
    result={'baseline':before,'finetuned':after,
            'scope':'Same samples and fixed thresholds. Original Test and challenge parents have known development exposure; no independent real-scene claim.'}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k]['diagnostics'] for k in ['baseline','finetuned']},ensure_ascii=False))


if __name__=='__main__':main()
