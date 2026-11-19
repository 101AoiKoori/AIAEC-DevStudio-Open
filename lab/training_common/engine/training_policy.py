"""V4 training policies for the installed Ultralytics 8.3.203 trainer."""
from copy import deepcopy
from datetime import datetime
from math import isclose, isfinite
from numbers import Real
from pathlib import Path
import random
from shutil import copyfile
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from .config import VALIDATING_MODES, read_yaml


class NonFiniteTrainingError(RuntimeError):
    """Training produced a non-finite loss or checkpoint tensor."""


def validate_policy_checkpoint(checkpoint, task, policy, epochs, *, run_signature=None):
    """Return the number of completed epochs for a compatible resumable checkpoint."""
    state = checkpoint.get('policy_resume')
    if not isinstance(state, dict) or state.get('schema_version') != 1:
        raise ValueError('Checkpoint lacks full training state. Resume requires a checkpoint from the current engine.')
    if state.get('task') != task or state.get('policy') != policy or state.get('epochs') != epochs:
        raise ValueError('Checkpoint task, training policy or epoch budget differs from the original run')
    if run_signature is not None and state.get('run_signature') != run_signature:
        raise ValueError('Checkpoint configuration or frozen data signature differs from the original run')
    if checkpoint.get('train_args', {}).get('time'):
        raise ValueError('Checkpoints from internally timed Ultralytics runs cannot resume a fixed schedule')
    epoch = checkpoint.get('epoch')
    if type(epoch) is not int or not 0 <= epoch < epochs:
        raise ValueError('Checkpoint has an invalid completed epoch')
    if state.get('trained_epochs') != epoch + 1 or type(state.get('terminal')) is not bool:
        raise ValueError('Checkpoint completion state differs from the recorded epoch')
    if state['terminal'] or epoch + 1 >= epochs:
        raise ValueError('Checkpoint is from a finished training run')
    for key in ('model', 'ema', 'scheduler', 'rng'):
        if not isinstance(state.get(key), dict) or not state[key]:
            raise ValueError(f'Checkpoint lacks {key} training state')
    if state['scheduler'].get('last_epoch') != epoch:
        raise ValueError('Checkpoint scheduler position differs from the completed epoch')
    if set(state['rng']) != {'python', 'numpy', 'torch', 'cuda'}:
        raise ValueError('Checkpoint lacks random number generator state')
    optimizer = checkpoint.get('optimizer')
    if not isinstance(optimizer, dict) or not isinstance(optimizer.get('state'), dict) or not optimizer.get('param_groups'):
        raise ValueError('Checkpoint lacks full optimizer state')
    if not isinstance(checkpoint.get('scaler'), dict) or type(checkpoint.get('updates')) is not int:
        raise ValueError('Checkpoint lacks scaler or EMA update state')
    keys = _stopper_attributes(policy)
    stopper = state.get('stopper')
    if not isinstance(stopper, dict) or set(stopper) != set(keys):
        raise ValueError('Checkpoint early-stopping state does not match the training policy')
    for key, value in stopper.items():
        if key.endswith('_epoch') and (type(value) is not int or not 0 <= value <= epoch + 1):
            raise ValueError('Checkpoint has an invalid early-stopping epoch')
        if ('score' in key or 'fitness' in key) and (not isinstance(value, Real) or not isfinite(value)):
            raise ValueError('Checkpoint has an invalid early-stopping score')
    return epoch + 1


def _stopper_attributes(policy):
    if policy is None:
        return ('best_epoch', 'best_fitness', 'possible_stop')
    if policy['mode'] in VALIDATING_MODES:
        return ('best_epoch', 'best_score', 'progress_epoch', 'progress_score')
    return ()


def _capture_rng(device):
    import numpy as np
    import torch

    cuda = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
    algorithm, keys, position, has_gauss, gauss = np.random.get_state()
    numpy = (algorithm, keys.tolist(), position, has_gauss, gauss)
    return {'python': random.getstate(), 'numpy': numpy, 'torch': torch.get_rng_state(), 'cuda': cuda}


def _restore_rng(trainer):
    import numpy as np
    import torch

    state = trainer._resume_state['rng']
    random.setstate(state['python'])
    algorithm, keys, position, has_gauss, gauss = state['numpy']
    np.random.set_state((algorithm, np.asarray(keys, dtype=np.uint32), position, has_gauss, gauss))
    torch.set_rng_state(state['torch'])
    if trainer.device.type == 'cuda' and state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'], trainer.device)


def _require_finite(tensors, epoch, stage):
    import torch

    checks_by_device = {}
    for tensor in tensors:
        if torch.is_tensor(tensor) and tensor.is_floating_point():
            checks_by_device.setdefault(tensor.device, []).append(torch.isfinite(tensor).all())
    # Aggregate checks on each device before reading their results on the host.
    if not all(torch.stack(checks).all().item() for checks in checks_by_device.values()):
        raise NonFiniteTrainingError(f'Epoch {epoch}: non-finite {stage}')


def _check_epoch_loss(trainer):
    _require_finite((trainer.tloss,), trainer.epoch + 1, 'training loss')


class _ValidationStopper:
    """Keep the strict best score and track significant progress separately."""

    possible_stop = False  # Real validation already runs every epoch.

    def __init__(self, min_epochs, patience, min_delta):
        self.min_epochs = min_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.best_epoch = None
        self.best_score = None
        self.progress_epoch = None
        self.progress_score = None

    def __call__(self, epoch, score):
        if self.best_score is None or score > self.best_score:
            self.best_epoch, self.best_score = epoch, score
        if self.progress_score is None or self._advances_patience_baseline(score):
            self.progress_epoch, self.progress_score = epoch, score
        # The first min_epochs epochs never count against patience.
        anchor = max(self.min_epochs, self.progress_epoch)
        return self.patience is not None and epoch - anchor >= self.patience

    def _advances_patience_baseline(self, score):
        threshold = self.progress_score + self.min_delta
        return score > self.progress_score and (
            score >= threshold or isclose(score, threshold, rel_tol=0, abs_tol=1e-12)
        )


def _publish_checkpoints(last, copies, serialize):
    """Publish complete files by replacement and serialize the EMA only once."""
    with TemporaryDirectory(prefix='.checkpoint-', dir=last.parent) as temporary:
        staging = Path(temporary)
        pending = staging / last.name
        serialize(pending)
        pending.replace(last)
        for destination in copies:
            pending = staging / destination.name
            copyfile(last, pending)
            pending.replace(destination)


def build_policy_trainer(task, policy, *, run_signature=None, on_checkpoint=None, optimizer_steps=None):
    """Return a resumable Detect/PoseTrainer with the selected training policy."""
    from ultralytics import __version__

    if __version__ != '8.3.203':
        raise RuntimeError(f'Resumable training requires Ultralytics 8.3.203, found {__version__}')

    from ultralytics.engine.trainer import GIT, unwrap_model
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.models.yolo.pose import PoseTrainer

    base = {'detect': DetectionTrainer, 'pose': PoseTrainer}[task]
    managed = policy is not None
    fixed = managed and policy['mode'] == 'fixed'
    selection_metric = 'metrics/mAP50-95(B)' if task == 'detect' else 'metrics/mAP50-95(P)'

    class PolicyTrainer(base):
        def __init__(self, *args, overrides=None, **kwargs):
            self._resume_state = None
            self._validation_stop = _ValidationStopper(
                policy.get('min_epochs', 0), policy.get('patience'), policy.get('min_delta', 0)
            ) if managed and not fixed else None
            overrides = dict(overrides or {})
            if managed:
                overrides.update(val=not fixed, patience=0, save=True, save_period=policy['save_period'])
            super().__init__(*args, overrides=overrides, **kwargs)
            self.add_callback('on_train_epoch_end', _check_epoch_loss)
            if self.resume:
                self.add_callback('on_train_start', _restore_rng)

        @property
        def best_epoch(self):
            stopper = self._validation_stop if managed else self.stopper
            return stopper.best_epoch if stopper is not None else None

        @property
        def best_score(self):
            if not managed:
                return self.best_fitness
            return None if fixed else self._validation_stop.best_score

        @property
        def stopped_early(self):
            return (not managed or policy['mode'] == 'early_stopping') and self.stop and self.epoch + 1 < self.epochs

        def get_dataset(self):
            if not managed:
                return super().get_dataset()
            spec = read_yaml(self.args.data)
            if 'download' in spec or spec.get('test') or spec.get('minival') or spec.get('validation'):
                raise ValueError('Training policies only accept staged train/val datasets')
            if fixed and spec.get('val'):
                raise ValueError('Fixed training requires no validation split')
            if not fixed and not spec.get('val'):
                raise ValueError('Validation training requires a validation split')
            return super().get_dataset()

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
            # 8.3.203 requests a val loader even with args.val=False.
            if fixed and mode == 'val':
                return None
            return super().get_dataloader(dataset_path, batch_size, rank, mode)

        def get_validator(self):
            if not fixed:
                return super().get_validator()
            losses = ('box_loss', 'pose_loss', 'kobj_loss', 'cls_loss', 'dfl_loss')
            self.loss_names = losses if task == 'pose' else ('box_loss', 'cls_loss', 'dfl_loss')
            # The parent setup reads these metric keys before building EMA and the optimizer.
            return SimpleNamespace(metrics=SimpleNamespace(keys=[]))

        def label_loss_items(self, loss_items=None, prefix='train'):
            if fixed and prefix == 'val':
                return []
            return super().label_loss_items(loss_items, prefix)

        def _setup_train(self):
            super()._setup_train()
            if fixed:
                self.validator = None
            elif managed:
                self.stopper = self._validation_stop

        def check_resume(self, overrides):
            if not self.args.resume:
                return super().check_resume(overrides)
            # The engine supplies the original run's authoritative arguments and relocated data path.
            checkpoint_path = Path(self.args.resume).resolve()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f'Resume checkpoint missing: {checkpoint_path}')
            self.args.model = self.args.resume = str(checkpoint_path)
            self.args.exist_ok = True
            self.args.save_dir = str(checkpoint_path.parents[1])
            self.resume = True

        def setup_model(self):
            checkpoint = super().setup_model()
            if self.resume:
                self.start_epoch = validate_policy_checkpoint(checkpoint or {}, task, policy, self.epochs,
                                                              run_signature=run_signature)
                self._resume_state = checkpoint['policy_resume']
                # The native loader initializes from the inference EMA, then this restores the raw train model.
                self.model.load_state_dict(self._resume_state['model'], strict=True)
            return checkpoint

        def resume_training(self, checkpoint):
            if not self.resume:
                return
            state = self._resume_state
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scaler.load_state_dict(checkpoint['scaler'])
            self.ema.ema.load_state_dict(state['ema'], strict=True)
            self.ema.updates = checkpoint['updates']
            self.scheduler.load_state_dict(state['scheduler'])
            self.best_fitness = checkpoint['best_fitness']
            stopper = self._validation_stop if managed else self.stopper
            for key, value in state['stopper'].items():
                setattr(stopper, key, value)
            if self.args.close_mosaic and self.start_epoch > self.epochs - self.args.close_mosaic:
                self._close_dataloader_mosaic()
                self.train_loader.reset()

        def validate(self):
            if not managed:
                return super().validate()
            # The base loop calls this hook on the final epoch regardless of args.val.
            if fixed:
                return {}, None
            metrics = dict(self.validator(self))
            metrics.pop('fitness', None)
            score = metrics.get(selection_metric)
            counts = self.validator.metrics.nt_per_class
            if counts is None or counts.sum() <= 0:
                raise RuntimeError('Model selection requires validation ground-truth instances')
            if isinstance(score, bool) or not isinstance(score, Real) or not isfinite(score) or not 0 <= score <= 1:
                raise RuntimeError(f'Model selection requires a valid {selection_metric} score')
            return metrics, float(score)

        def save_model(self):
            import torch

            epoch = self.epoch + 1
            metrics = dict(self.metrics)
            copies = []
            if not fixed:
                metrics['fitness'] = self.fitness
                is_best = self.best_epoch == epoch if managed else self.best_fitness == self.fitness
                if is_best:
                    copies.append(self.best)
            saved_epoch = epoch if managed else self.epoch  # Keep native checkpoint names for legacy configurations.
            periodic = None
            if self.save_period > 0 and saved_epoch % self.save_period == 0:
                periodic = f'epoch{saved_epoch}.pt'
                copies.append(self.wdir / periodic)

            full_ema = unwrap_model(self.ema.ema)
            ema = deepcopy(full_ema).half()
            _require_finite(ema.state_dict().values(), epoch, 'EMA checkpoint tensors')
            raw_model = unwrap_model(self.model).state_dict()
            _require_finite(raw_model.values(), epoch, 'training model checkpoint tensors')
            optimizer = self.optimizer.state_dict()
            optimizer_tensors = (value for state in optimizer['state'].values() for value in state.values())
            _require_finite(optimizer_tensors, epoch, 'optimizer checkpoint tensors')

            stopper = self._validation_stop if managed else self.stopper
            resume_state = {
                'schema_version': 1, 'task': task, 'policy': policy, 'epochs': self.epochs,
                'run_signature': run_signature, 'trained_epochs': epoch,
                'terminal': bool(self.stop) or epoch >= self.epochs,
                'completed_at_budget': epoch == self.epochs, 'stopped_early': self.stopped_early,
                'optimizer_steps': optimizer_steps() if optimizer_steps is not None else None,
                'best_checkpoint': self.best in copies, 'periodic_checkpoint': periodic,
                'model': raw_model, 'ema': full_ema.state_dict(),
                'scheduler': self.scheduler.state_dict(), 'rng': _capture_rng(self.device),
                'stopper': {key: getattr(stopper, key) for key in _stopper_attributes(policy)},
            }
            # Keep the standard inference EMA and the exact training tensors in separate entries.
            checkpoint = {
                'epoch': self.epoch,
                'best_fitness': self.best_score,
                'model': None,
                'ema': ema,
                'updates': self.ema.updates,
                'optimizer': optimizer,
                'scaler': self.scaler.state_dict(),
                'train_args': vars(self.args),
                'train_metrics': metrics,
                'train_results': self.read_results_csv(),
                'policy_resume': resume_state,
                'date': datetime.now().isoformat(),
                'version': __version__,
                'git': {'root': str(GIT.root), 'branch': GIT.branch, 'commit': GIT.commit, 'origin': GIT.origin},
                'license': 'AGPL-3.0 (https://ultralytics.com/license)',
                'docs': 'https://docs.ultralytics.com',
            }
            _publish_checkpoints(self.last, copies, lambda path: torch.save(checkpoint, path))
            if on_checkpoint is not None:
                on_checkpoint(self, self.last)

        def final_eval(self):
            # Per-epoch validation supplies the early-stopping scores. Keep all saved training state.
            if not managed and self.best.is_file():
                self.validator.args.plots = self.args.plots
                self.validator.args.compile = False
                self.metrics = self.validator(model=self.best)
                self.metrics.pop('fitness', None)
                self.run_callbacks('on_fit_epoch_end')

        def plot_metrics(self):
            # The stock plot expects paired train and val losses. CSV and TensorBoard still record fixed runs.
            if not fixed:
                super().plot_metrics()

    return PolicyTrainer
