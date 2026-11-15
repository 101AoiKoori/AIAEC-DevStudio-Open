"""Record actual optimizer steps and learning rates for a bounded V4 run."""
import json
from pathlib import Path

from .runtime import write_json


class UpdateRecorder:
    """Observe optimizer.step after it executes, including AMP skipped-step behavior."""

    def __init__(self, run_dir, epoch_budget=None):
        self.directory = Path(run_dir) / 'metadata'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'optimizer_updates.jsonl'
        self.stream = self.path.open('x', encoding='utf-8')
        self.steps = 0
        self.registered = False
        self.handle = None
        self.epoch_budget = epoch_budget

    def attach(self, trainer):
        if self.registered:
            raise RuntimeError('Optimizer step recorder was registered more than once')
        optimizer = getattr(trainer, 'optimizer', None)
        if optimizer is None or not callable(getattr(optimizer, 'register_step_post_hook', None)):
            raise RuntimeError('Optimizer does not support actual-step recording')
        loader = getattr(trainer, 'train_loader', None)
        batches = len(loader) if loader is not None else None
        images = len(loader.dataset) if loader is not None and hasattr(loader, 'dataset') else None
        write_json(self.directory / 'optimization_budget.json',
                   {'epoch_budget': self.epoch_budget, 'training_images_per_epoch': images,
                    'microbatches_per_epoch': batches, 'batch_size': getattr(trainer, 'batch_size', None),
                    'optimizer_step_upper_bound': (self.epoch_budget * batches
                                                   if self.epoch_budget is not None and batches is not None else None),
                    'scope': 'At most one optimizer step per microbatch; actual accumulation may reduce updates.'})

        def record_step(current_optimizer, args, kwargs):
            self.steps += 1
            event = {'step': self.steps, 'epoch': int(trainer.epoch) + 1,
                     'learning_rates': [float(group['lr']) for group in current_optimizer.param_groups],
                     'gradient_accumulation': getattr(trainer, 'accumulate', None)}
            self.stream.write(json.dumps(event, allow_nan=False) + '\n')
            self.stream.flush()

        self.handle = optimizer.register_step_post_hook(record_step)
        self.registered = True

    def close(self, reason, trainer=None):
        if self.handle is not None:
            self.handle.remove()
        self.stream.close()
        epoch = getattr(trainer, 'epoch', None)
        write_json(self.directory / 'optimization_summary.json',
                   {'status': reason, 'recorded_optimizer_steps': self.steps,
                    'actual_optimizer_steps': self.steps if self.registered and reason in
                    ('completed_epoch_budget', 'framework_stopped_early_reason_unconfirmed',
                     'zero_optimizer_updates') else None,
                    'step_hook_registered': self.registered,
                    'last_epoch': int(epoch) + 1 if epoch is not None else None,
                    'step_log': self.path.name,
                    'scope': 'Post-optimizer.step events; resumed runs are not combined.'})


def attach_update_recorder(model, run_dir, epoch_budget=None):
    recorder = UpdateRecorder(run_dir, epoch_budget)
    model.add_callback('on_pretrain_routine_end', recorder.attach)
    return recorder
