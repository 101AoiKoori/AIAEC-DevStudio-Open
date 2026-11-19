"""Replay same-pose versus agnostic final NMS on three frozen seed models."""
import argparse
import copy
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from lab.training_common.engine.runtime import setup_paths
from lab.features.e_bike.training.scripts.candidate_verifier_experiment import foreground
from lab.features.e_bike.training.scripts.evaluation_audit import image_metrics
from lab.training_common.engine.reporting import match_detections
from lab.features.e_bike.training.scripts.part_hull_experiment import diagnostic


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='runs/e_bike/pose_aware_probe_20261118')
    args=parser.parse_args();out=(ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT/'runs/e_bike'):raise ValueError('Expected research output')
    out.mkdir(parents=True,exist_ok=False);setup_paths(out)
    from ultralytics.utils.nms import TorchNMS
    protocol={'confidence':.25,'final_nms_iou':.3,'seeds':[42,43,44],
              'scope':'Post-hoc diagnostic on exposed development images. Partition by predicted posture, never GT. No training or threshold search.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8');reports={}
    for seed in protocol['seeds']:
        owner=ROOT/('runs/e_bike/candidate_verifier_20261118' if seed==42 else f'runs/e_bike/verifier_seed_study_20261118/seed_{seed}')
        reference=[json.loads(s) for s in (owner/'evaluation/c025_i030_joint/predictions.jsonl').read_text().splitlines()]
        modes={'agnostic_control':[],'predicted_pose_aware':[]}
        for ref in reference:
            cached=json.loads((owner/'evaluation_raw'/(Path(ref['source_image']).stem+'_verifier.json')).read_text());by={p['raw_id']:p for p in cached['classified_candidates']}
            pool=[copy.deepcopy(by[i]) for i in cached['proposal_ids'] if i in by and by[i]['raw_confidence']>.25 and foreground(by[i]['verifier_probs'])]
            for p in pool:p['class_id']=int(p['verifier_probs']['fallen']>p['verifier_probs']['upright'])
            for name in modes:
                keep=[]
                partitions=[list(range(len(pool)))] if name=='agnostic_control' else [[i for i,p in enumerate(pool) if p['class_id']==c] for c in (0,1)]
                for ids in partitions:
                    if not ids:continue
                    retained=TorchNMS.nms(torch.tensor([pool[i]['input_xyxy'] for i in ids]),torch.tensor([pool[i]['raw_confidence'] for i in ids]),.3).tolist()
                    keep.extend(ids[i] for i in retained)
                keep=sorted(keep,key=lambda i:-pool[i]['raw_confidence'])[:100];found=[dict(pool[i],confidence=pool[i]['raw_confidence']) for i in keep]
                if name=='agnostic_control':assert [p['raw_id'] for p in found]==[p['raw_id'] for p in ref['predictions']]
                matches,miss,extra=match_detections(ref['ground_truth'],found,.5)
                modes[name].append({'source_image':ref['source_image'],'group':ref['group'],'ground_truth':ref['ground_truth'],
                                    'predictions':found,'matches':matches,'missed_gt_indices':miss,
                                    'false_prediction_indices':extra,**diagnostic(ref['ground_truth'],found)})
        report={}
        for name,rows in modes.items():
            folder=out/f'seed_{seed}'/name;folder.mkdir(parents=True)
            (folder/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8');report[name]={}
            for group in ('old26','original_real','new_layout_holdout'):
                selected=[r for r in rows if (r['group']!='new_layout_holdout' if group=='old26' else r['group']==group)];m=[image_metrics(r) for r in selected]
                report[name][group]={'images':len(m),**{k:sum(x[k] for x in m) for k in ('count_exact','strict_instance_exact','entity_fp','entity_fn','matched_pose_errors')},'duplicate_predictions':sum(r['duplicate_predictions'] for r in selected)}
        reports[str(seed)]=report
    summary={'protocol':protocol,'seeds':reports,'reference_reproduction_images':58*3}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
