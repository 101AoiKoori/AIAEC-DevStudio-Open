"""Compare a completed fire fine-tune with its original best at unchanged thresholds."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

from PIL import Image, ImageDraw
import yaml

ROOT=Path(__file__).resolve().parents[5]
PACKAGE=ROOT/'training_packages/fire_sticker'
sys.path.insert(0,str(PACKAGE/'scripts/common'))
sys.path.insert(0,str(PACKAGE/'scripts/training'))
from engine.config import load_config
from engine.freeze_gate import freeze_selection, require_frozen_selection
from engine.runner import evaluate_run
from engine.runtime import prepare_ultralytics, run_context, write_json
from engine.reporting import diagnostics
from fire_report import build_report, load_metadata, write_report


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    from hashlib import sha256
    return sha256(Path(path).read_bytes()).hexdigest()


def evaluate_pair(config, run, weights, reason):
    relative=run.relative_to(PACKAGE).as_posix()
    name='val_selection'
    if not (run/'evaluation'/name/'summary.json').exists():
        evaluate_run(PACKAGE,config,weights,'val',relative,name)
    if not (run/'evaluation'/name/'fire_report.json').exists():
        write_report(run/'evaluation'/name,PACKAGE/'dataset_finetune_compact/workspace_sources.json',PACKAGE)
    if not (run/'frozen_selection').exists():
        freeze_selection(PACKAGE,config,relative,weights,name,reason)
    frozen=(run/'frozen_selection/selected.pt').relative_to(PACKAGE).as_posix()
    require_frozen_selection(PACKAGE,config,relative,frozen)
    if not (run/'evaluation/test/summary.json').exists():
        evaluate_run(PACKAGE,config,frozen,'test',relative,'test')
    if not (run/'evaluation/test/fire_report.json').exists():
        write_report(run/'evaluation/test',PACKAGE/'dataset_finetune_compact/workspace_sources.json',PACKAGE)
    return {s:read(run/f'evaluation/{name if s=="val" else "test"}/fire_report.json') for s in ('val','test')}


def prepare_omitted_probe(destination):
    pool=read(PACKAGE/'dataset_finetune/workspace_sources.json')['records']
    compact=read(PACKAGE/'dataset_finetune_compact/workspace_sources.json')['records']
    replay={r['candidate_origin']['image'] for r in compact if r['split']=='train' and r['fine_tune_group']=='replay'}
    records=[r for r in pool if r['split']=='train' and r['fine_tune_group']=='replay' and r['image'] not in replay]
    if len(records)!=2400:raise ValueError('Expected 2400 original Train images omitted from fine-tuning')
    samples=[]
    for row in records:
        sample=deepcopy(row)
        for key,folder in [('image','images'),('label','labels')]:
            old=PACKAGE/row[key];new=destination/folder/'train'/old.name
            new.parent.mkdir(parents=True,exist_ok=True)
            if new.exists():
                if sha(old)!=sha(new):raise ValueError('Probe bytes differ')
            else:shutil.copy2(old,new)
            sample['run_'+key]=new.relative_to(destination).as_posix()
        samples.append(sample)
    write_json(destination/'samples.json',samples)
    (destination/'data.yaml').write_text(yaml.safe_dump({'names':['fire_sticker'],'train':'images/train'},sort_keys=False),encoding='utf8')
    return destination/'data.yaml'


def omitted_report(config, weights, data, destination, run):
    output=destination/'fire_report.json'
    if output.exists():return read(output)
    import torch
    torch.set_num_threads(2)
    with run_context(PACKAGE,config,run,'omitted_'+destination.name):
        model=prepare_ultralytics(run)(str(PACKAGE/weights),task='detect')
        counts=diagnostics(model,data,'train',config,destination)
        del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
    rows=[json.loads(s) for s in (destination/'predictions/predictions.jsonl').read_text(encoding='utf8').splitlines() if s]
    report=build_report(rows,load_metadata(PACKAGE/'dataset_finetune/workspace_sources.json',PACKAGE,'train'),config['imgsz'],config['match_iou'])
    report.update(split='train',thresholds=counts,checkpoint=weights,checkpoint_sha256=sha(PACKAGE/weights),
        scope='Old Train omitted from this fine-tune: known pretraining data, not independent evaluation. Covers three synthesis methods; all original direct-real Train images were replayed. Direct-real regression uses unchanged Val/Test.')
    write_json(output,report)
    return report


def simple(report):
    result=deepcopy(report['images'])
    result['target_instances']=result['tp']+result['fn']
    result['small_8_16']=report['instance_groups']['input_short_side_px'].get('8_16')
    result['buildings']=report['instance_groups'].get('building',{})
    result['branches']=report['image_groups'].get('branch',{})
    result['red_bins']=report['image_groups'].get('distractor',{}).get('red_orange_bin')
    return result


def red_bin_sheet(before_run, after_run, destination):
    source=before_run/'evaluation/test/predictions/predictions.jsonl'
    records=[json.loads(v) for v in source.read_text(encoding='utf8').splitlines() if v]
    failed=[r for r in records if not r['ground_truth'] and r['false_prediction_indices'] and 'hard_negative' in r['source_image']]
    chosen=failed[::2][:3]
    if not chosen:return
    sheet=Image.new('RGB',(1280,len(chosen)*390),'#eee');draw=ImageDraw.Draw(sheet)
    for index,row in enumerate(chosen):
        for col,run in enumerate([before_run,after_run]):
            filename=Path(row['image']).name
            path=run/'evaluation/test/predictions/previews'/filename
            if not path.exists():path=run/'evaluation/test/predictions/failures'/filename
            if not path.exists():continue
            im=Image.open(path).convert('RGB');im.thumbnail((640,360));sheet.paste(im,(col*640,index*390+25))
            draw.text((col*640+5,index*390+5),'Original best' if col==0 else 'Fine-tuned best',fill='black')
    sheet.save(destination,quality=94)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,help='Fine-tuned run relative to training package')
    args=parser.parse_args();run=(PACKAGE/args.run).resolve()
    if not run.is_relative_to(PACKAGE/'runs'):raise ValueError('Run must remain inside package runs')
    artifacts=read(run/'metadata/artifacts.json');status=read(run/'metadata/train_status.json')
    if status.get('status')!='complete' or artifacts.get('completed_epochs')!=10:raise ValueError('The requested ten training epochs are not complete')
    config=load_config(PACKAGE,'config/finetune.yaml')
    baseline='weights/finetune_initial.pt'
    if sha(PACKAGE/baseline)!=sha(PACKAGE/'fire_sticker_best_Val_Test/best.pt'):raise ValueError('Original best bytes changed')
    tuned=(run/artifacts['best']).relative_to(PACKAGE).as_posix()
    baseline_run=run.with_name(run.name+'_baseline_reference')
    print('EVALUATING original best on unchanged Val/Test',flush=True)
    before=evaluate_pair(config,baseline_run,baseline,'Fixed original best selected before this fine-tuning run; Test is regression only')
    print('EVALUATING validation-selected fine-tuned best on unchanged Val/Test',flush=True)
    after=evaluate_pair(config,run,tuned,'Best checkpoint selected only by Val during the fixed ten-epoch fine-tune')
    compare=run/'comparison';compare.mkdir(exist_ok=True)
    probe=prepare_omitted_probe(compare/'omitted_train_data')
    print('EVALUATING both models on 2400 old Train images omitted from fine-tuning',flush=True)
    before['omitted_old_train']=omitted_report(config,baseline,probe,compare/'baseline_omitted_old_train',baseline_run)
    after['omitted_old_train']=omitted_report(config,tuned,probe,compare/'finetuned_omitted_old_train',run)
    for split in before:
        if {r['image'] for r in before[split]['per_image']}!={r['image'] for r in after[split]['per_image']}:
            raise ValueError('Comparison image identities differ')
    result={'status':'complete','baseline_weights':baseline,'baseline_sha256':sha(PACKAGE/baseline),
        'finetuned_weights':tuned,'finetuned_sha256':sha(PACKAGE/tuned),
        'run':args.run,'epochs':10,'batch':4,'imgsz':640,'confidence':.25,'match_iou':.5,'nms_iou':.7,
        'scope':'Unchanged known-source Val/Test regression plus old Train omitted from this fine-tune; no independent field-generalization claim. Test did not select checkpoints or thresholds.',
        'before':{k:simple(v) for k,v in before.items()},'after':{k:simple(v) for k,v in after.items()},
        'baseline_run':baseline_run.relative_to(PACKAGE).as_posix()}
    write_json(compare/'comparison.json',result)
    lines=['# Fire fine-tuning comparison','',result['scope'],'',
        'Both models: input 640, confidence 0.25, NMS IoU 0.7, matching IoU 0.5. Ten training epochs, batch 4.','',
        '| Dataset | Original TP / GT | Fine-tuned TP / GT | Original FP | Fine-tuned FP |',
        '| --- | ---: | ---: | ---: | ---: |']
    for split in before:
        b=result['before'][split];a=result['after'][split]
        lines.append(f'| {split} | {b["tp"]}/{b["target_instances"]} | {a["tp"]}/{a["target_instances"]} | {b["fp"]} | {a["fp"]} |')
    lines+=['','## Red-bin negatives','', '| Dataset | Images | Original false boxes / affected images | Fine-tuned false boxes / affected images |','| --- | ---: | ---: | ---: |']
    for split in ['val','test']:
        b=result['before'][split]['red_bins'];a=result['after'][split]['red_bins']
        if b and a:lines.append(f'| {split} | {b["images"]} | {b["fp"]} / {b["negative_false_positive_images"]} | {a["fp"]} / {a["negative_false_positive_images"]} |')
    lines+=['','Per-building, source-method and small-target results are in comparison.json.','',f'Fine-tuned weights: `{tuned}`.']
    (compare/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    red_bin_sheet(baseline_run,run,compare/'red_bin_before_after.jpg')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
