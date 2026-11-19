"""Export and validate a verified user-stopped checkpoint without further training."""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'lab/training_common'))
from engine.runtime import run_context,prepare_ultralytics,training_lock,write_json
from engine.runner import _record_policy_artifacts,_finalize_policy_training


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);args=parser.parse_args()
    run=(ROOT/args.run).resolve()
    if not run.is_relative_to(ROOT) or 'runs' not in run.relative_to(ROOT).parts:
        raise ValueError('Expected an existing run inside the workspace')
    receipt_path=run/'metadata/user_epoch_stop.json'
    receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
    if receipt['status']!='stopped_after_checkpoint':raise ValueError('Requested checkpoint stop has not completed')
    config=json.loads((run/'config/effective.json').read_text(encoding='utf-8'))
    if config['model_key']!='waste_sticker':raise ValueError('This finalizer is only for waste stickers')
    last=run/'training/weights/last.pt'
    digest=sha256(last.read_bytes()).hexdigest()
    if digest!=receipt['checkpoint_sha256']:
        raise ValueError('Last checkpoint differs from the requested completed epoch')
    with training_lock(run),run_context(ROOT,config,run,'finalize_user_stop'):
        YOLO=prepare_ultralytics(run)
        import torch
        checkpoint=torch.load(last,map_location='cpu',weights_only=False)
        saved=checkpoint['policy_resume']
        epoch=receipt['requested_completed_epoch']
        if saved['trained_epochs']!=epoch or saved['epochs']!=receipt['original_epoch_budget']:
            raise ValueError('Checkpoint epoch provenance differs from the explicit user request')
        stopper=saved['stopper']
        trainer=SimpleNamespace(epoch=epoch-1,args=SimpleNamespace(task=config['task']),
                                best_epoch=stopper['best_epoch'],best_score=stopper['best_score'])
        artifacts=_record_policy_artifacts(run,trainer,config['training_policy'],'user_requested_epoch_limit')
        artifacts.update(original_epoch_budget=saved['epochs'],requested_epoch_limit=epoch,
                         user_stop_receipt='metadata/user_epoch_stop.json')
        write_json(run/'metadata/artifacts.json',artifacts)
        old_status=json.loads((run/'metadata/train_status.json').read_text(encoding='utf-8'))
        receipt.update(controller_observed_status=old_status,checkpoint_optimizer_steps=saved['optimizer_steps'],
                       best_epoch=stopper['best_epoch'],finalization_started_utc=datetime.now(timezone.utc).isoformat())
        write_json(receipt_path,receipt)
        write_json(run/'metadata/train_status.json',{**old_status,'status':'interrupted',
            'stop_reason':'user_requested_epoch_limit','completed_epochs':epoch,
            'original_epoch_budget':saved['epochs'],'error':None,
            'scope':'Explicit user stop after a complete checkpoint, not completion of the original epoch budget.'})
        _finalize_policy_training(YOLO,config,run,run/'data/data.yaml',artifacts)
        if sha256(last.read_bytes()).hexdigest()!=digest:raise RuntimeError('Full resume checkpoint changed during export')
        receipt.update(finalization_status='complete',last_checkpoint_preserved=True,
                       finalization_finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(receipt_path,receipt)
    print('User-requested epoch limit finalized without optimizer updates.',flush=True)


if __name__=='__main__':main()
