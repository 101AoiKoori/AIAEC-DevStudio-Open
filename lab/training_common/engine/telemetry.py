"""Record actual optimizer steps and learning rates by training attempt."""
import json
from pathlib import Path

from .runtime import write_json


class UpdateRecorder:
    """Observe optimizer.step after it executes, including AMP skipped-step behavior."""

    def __init__(self, run_dir, epoch_budget=None, *, attempt=None, initial_steps=0, initial_epochs=0):
        self.directory = Path(run_dir) / 'metadata'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.suffix = '_' + attempt if attempt else ''
        self.path = self.directory / ('optimizer_updates' + self.suffix + '.jsonl')
        self.stream = self.path.open('x', encoding='utf-8')
        self.steps = 0
        self.initial_steps = initial_steps
        self.initial_epochs = initial_epochs
        self.registered = False
        self.handle = None
        self.epoch_budget = epoch_budget

    @property
    def total_steps(self):
        return self.initial_steps + self.steps

    def attach(self, trainer):
        if self.registered:
            raise RuntimeError('Optimizer step recorder was registered more than once')
        optimizer = getattr(trainer, 'optimizer', None)
        if optimizer is None or not callable(getattr(optimizer, 'register_step_post_hook', None)):
            raise RuntimeError('Optimizer does not support actual-step recording')
        loader = getattr(trainer, 'train_loader', None)
        batches = len(loader) if loader is not None else None
        images = len(loader.dataset) if loader is not None and hasattr(loader, 'dataset') else None
        write_json(self.directory / ('optimization_budget' + self.suffix + '.json'),
                   {'epoch_budget': self.epoch_budget, 'training_images_per_epoch': images,
                    'epochs_at_checkpoint': self.initial_epochs, 'steps_at_checkpoint': self.initial_steps,
                    'microbatches_per_epoch': batches, 'batch_size': getattr(trainer, 'batch_size', None),
                    'optimizer_step_upper_bound': (self.epoch_budget * batches
                                                   if self.epoch_budget is not None and batches is not None else None),
                    'scope': 'At most one optimizer step per microbatch; actual accumulation may reduce updates.'})

        def record_step(current_optimizer, args, kwargs):
            self.steps += 1
            event = {'step': self.total_steps, 'epoch': int(trainer.epoch) + 1,
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
        completed = reason in ('completed_epoch_budget', 'validation_early_stop',
                               'framework_stopped_early_reason_unconfirmed', 'zero_optimizer_updates')
        summary = {'status': reason, 'recorded_optimizer_steps': self.steps,
                   'actual_optimizer_steps': self.total_steps if self.registered and completed else None,
                   'epochs_at_checkpoint': self.initial_epochs, 'steps_at_checkpoint': self.initial_steps,
                   'step_hook_registered': self.registered,
                   'last_epoch': int(epoch) + 1 if epoch is not None else None,
                   'step_log': self.path.name,
                   'scope': 'Step counts resume at the checkpoint; partial attempts keep separate event logs.'}
        if self.suffix:
            write_json(self.directory / ('optimization_summary' + self.suffix + '.json'), summary)
        write_json(self.directory / 'optimization_summary.json', summary)


def attach_update_recorder(model, run_dir, epoch_budget=None, **attempt):
    recorder = UpdateRecorder(run_dir, epoch_budget, **attempt)
    model.add_callback('on_pretrain_routine_end', recorder.attach)
    return recorder
