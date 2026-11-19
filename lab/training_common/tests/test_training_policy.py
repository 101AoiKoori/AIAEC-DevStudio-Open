"""Exercise V4 policy hooks and checkpoint publication without running a model."""
import json
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.training_policy import (NonFiniteTrainingError, _ValidationStopper, _publish_checkpoints,
                                    build_policy_trainer, validate_policy_checkpoint)


class FakeEMA:
    def __init__(self):
        self.weight = torch.tensor([1.0])

    def half(self):
        self.weight = self.weight.half()
        return self

    def state_dict(self):
        return {'weight': self.weight}

    def load_state_dict(self, state, *, strict):
        self.weight = state['weight'].clone()


class FakeTrainer:
    def __init__(self, *, overrides):
        self.args = SimpleNamespace(**overrides)
        self.args.close_mosaic = 0
        self.resume = bool(getattr(self.args, 'resume', False))
        self.wdir = Path(self.args.data).parent
        self.last, self.best = self.wdir / 'last.pt', self.wdir / 'best.pt'
        self.save_period = self.args.save_period
        self.model = FakeEMA()
        self.device = torch.device('cpu')
        self.ema = SimpleNamespace(ema=FakeEMA(), updates=5)
        self.optimizer = SimpleNamespace(state_dict=lambda: {'state': {0: {'step': 5}}, 'param_groups': [{'lr': 0.1}]},
                                         load_state_dict=Mock())
        self.scaler = SimpleNamespace(state_dict=lambda: {'scale': 1024}, load_state_dict=Mock())
        self.scheduler = SimpleNamespace(state_dict=lambda: {'last_epoch': self.epoch}, load_state_dict=Mock())
        self.stopper = SimpleNamespace(best_epoch=0, best_fitness=0.0, possible_stop=False)
        self.best_fitness = None
        self.callbacks = {}
        self.epoch = 0
        self.stop = False
        self.epochs = 12
        self.loaded_splits = []

    def add_callback(self, event, callback):
        self.callbacks.setdefault(event, []).append(callback)

    def run_callbacks(self, event):
        for callback in self.callbacks[event]:
            callback(self)

    def get_dataset(self):
        return 'parent parsed dataset'

    def setup_model(self):
        return self.resume_checkpoint

    def get_dataloader(self, dataset_path, batch_size, rank, mode):
        self.loaded_splits.append(mode)
        return dataset_path

    def get_validator(self):
        return self.real_validator

    def label_loss_items(self, loss_items, prefix):
        return [f'{prefix}/{name}' for name in self.loss_names]

    def _setup_train(self):
        self.train_loader = self.get_dataloader('train_dir', mode='train')
        self.test_loader = self.get_dataloader(None, mode='val')
        self.validator = self.get_validator()
        self.metrics = dict.fromkeys(self.validator.metrics.keys + self.label_loss_items(prefix='val'), 0.0)

    def read_results_csv(self):
        return {'train/test_loss': [0.2]}


class TrainingPolicyTests(unittest.TestCase):
    def test_fixed_validation_tracks_best_without_stopping_on_plateau(self):
        from engine.training_policy import _ValidationStopper

        selector = _ValidationStopper(min_epochs=0, patience=None, min_delta=0)
        for epoch, score in enumerate([0.4] + [0.2] * 35 + [0.6, 0.6], start=1):
            self.assertFalse(selector(epoch, score))
        self.assertEqual(selector.best_epoch, 37)
        self.assertEqual(selector.best_score, 0.6)

    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[3] / 'tmp'
        temporary_root.mkdir(exist_ok=True)
        temporary = TemporaryDirectory(dir=temporary_root)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.spec = self.root / 'data.yaml'
        self.spec.write_text('train: train/images\nval: null\nnames: [item]\n', encoding='utf-8')
        self.serialized = Mock(side_effect=lambda payload, path: path.write_text(
            json.dumps(payload, default=lambda value: value.tolist() if torch.is_tensor(value) else 'EMA'),
            encoding='utf-8'
        ))
        save_patch = patch.object(torch, 'save', self.serialized)
        save_patch.start()
        self.addCleanup(save_patch.stop)
        patches = patch.dict(sys.modules, {
            'ultralytics': SimpleNamespace(__version__='8.3.203'),
            'ultralytics.engine.trainer': SimpleNamespace(
                GIT=SimpleNamespace(root='', branch='', commit='', origin=''),
                unwrap_model=lambda model: model,
            ),
            'ultralytics.models.yolo.detect': SimpleNamespace(DetectionTrainer=FakeTrainer),
            'ultralytics.models.yolo.pose': SimpleNamespace(PoseTrainer=FakeTrainer),
        })
        patches.start()
        self.addCleanup(patches.stop)

    def make_trainer(self, task, mode='fixed', *, resume=None, **kw):
        policy = {'mode': mode, 'save_period': 2} if mode is not None else None
        if mode == 'early_stopping':
            policy.update(min_epochs=3, patience=2, min_delta=0.05)
        return build_policy_trainer(task, policy, **kw)(overrides={'data': str(self.spec), 'resume': resume,
                                                                  'save_period': -1, 'val': True})

    def validator(self, metrics, target_count=1):
        validator = Mock(return_value=metrics)
        validator.metrics = SimpleNamespace(keys=list(metrics), nt_per_class=SimpleNamespace(sum=lambda: target_count))
        return validator

    def test_fixed_keeps_only_training_loader_and_periodic_ema_checkpoints(self):
        trainer = self.make_trainer('pose')
        self.assertEqual(trainer.get_dataset(), 'parent parsed dataset')
        self.assertFalse(trainer.args.val)
        trainer._setup_train()
        self.assertEqual(trainer.loaded_splits, ['train'])
        self.assertEqual(trainer.loss_names, ('box_loss', 'pose_loss', 'kobj_loss', 'cls_loss', 'dfl_loss'))
        self.assertIsNone(trainer.validator)
        self.assertEqual(trainer.validate(), ({}, None))
        self.assertEqual(trainer.metrics, {})

        trainer.save_model()
        self.assertFalse((self.root / 'epoch1.pt').exists())
        trainer.epoch = 1
        trainer.save_model()
        trainer.final_eval()
        periodic = self.root / 'epoch2.pt'
        self.assertEqual(trainer.last.read_bytes(), periodic.read_bytes())
        self.assertFalse(trainer.best.exists())
        self.assertEqual(self.serialized.call_count, 2)
        payload = json.loads(periodic.read_text(encoding='utf-8'))
        self.assertEqual(payload['ema'], 'EMA')
        self.assertEqual(payload['optimizer'], {'state': {'0': {'step': 5}}, 'param_groups': [{'lr': 0.1}]})
        self.assertEqual(payload['scaler'], {'scale': 1024})
        self.assertEqual(payload['train_metrics'], {})
        self.assertFalse(payload['policy_resume']['best_checkpoint'])
        self.assertEqual(payload['policy_resume']['periodic_checkpoint'], 'epoch2.pt')
        self.assertEqual(validate_policy_checkpoint(self.serialized.call_args.args[0], 'pose',
                                                     {'mode': 'fixed', 'save_period': 2}, 12), 2)
        self.assertIsNone(trainer.best_epoch)
        self.assertFalse(trainer.stopped_early)

    def test_distinct_pose_selection_and_strict_best_checkpoint(self):
        trainer = self.make_trainer('pose', 'early_stopping')
        self.assertTrue(trainer.args.val)
        for epoch, (box, pose) in enumerate([(0.2, 0.6), (0.8, 0.5), (0.9, 0.6), (0.1, 0.65)], start=1):
            trainer.epoch = epoch - 1
            trainer.validator = self.validator({'metrics/mAP50-95(B)': box, 'metrics/mAP50-95(P)': pose,
                                                'fitness': box + pose})
            trainer.metrics, trainer.fitness = trainer.validate()
            trainer.stop = trainer._validation_stop(epoch, trainer.fitness)
            trainer.save_model()
            expected_best_epoch = 1 if epoch < 4 else 4
            self.assertEqual(trainer.best_epoch, expected_best_epoch)
            self.assertEqual(json.loads(trainer.best.read_text(encoding='utf-8'))['epoch'], expected_best_epoch - 1)
        self.assertEqual(trainer.best_score, 0.65)
        self.assertFalse(trainer.stopped_early)

    def test_early_stopping_uses_patience_after_minimum_and_separate_delta(self):
        stopper = _ValidationStopper(min_epochs=3, patience=2, min_delta=0.05)
        for epoch, score in enumerate([0.3, 0.31, 0.32, 0.35, 0.37, 0.38], start=1):
            self.assertEqual(stopper(epoch, score), epoch == 6)
        self.assertEqual(stopper.progress_epoch, 4)
        self.assertEqual(stopper.best_epoch, 6)
        self.assertEqual(stopper.best_score, 0.38)
        plateau = _ValidationStopper(min_epochs=3, patience=2, min_delta=0)
        self.assertEqual([plateau(epoch, 0.5) for epoch in range(1, 6)], [False, False, False, False, True])
        self.assertEqual(plateau.best_epoch, 1)

    def test_requires_real_validation_data_and_valid_box_score(self):
        trainer = self.make_trainer('detect', 'early_stopping')
        with self.assertRaisesRegex(ValueError, 'requires a validation split'):
            trainer.get_dataset()
        self.spec.write_text('train: train/images\nval: val/images\nnames: [item]\n', encoding='utf-8')
        self.assertEqual(trainer.get_dataset(), 'parent parsed dataset')
        with self.assertRaisesRegex(ValueError, 'requires no validation split'):
            self.make_trainer('detect').get_dataset()
        for score in (None, float('nan'), float('inf'), -0.1, 1.1):
            with self.subTest(score=score), self.assertRaisesRegex(RuntimeError, 'mAP50-95'):
                trainer.validator = self.validator({'metrics/mAP50-95(B)': score})
                trainer.validate()
        trainer.validator = self.validator({'metrics/mAP50-95(B)': 0.0}, target_count=0)
        with self.assertRaisesRegex(RuntimeError, 'ground-truth'):
            trainer.validate()
        trainer.validator = self.validator({'metrics/mAP50-95(B)': 0.0})
        self.assertEqual(trainer.validate(), ({'metrics/mAP50-95(B)': 0.0}, 0.0))

    def test_checkpoint_writer_leaves_complete_files_when_writes_fail(self):
        last = self.root / 'last.pt'
        last.write_bytes(b'committed last')
        best = self.root / 'best.pt'
        best.write_bytes(b'committed best')

        def fail(path):
            path.write_bytes(b'partial')
            raise OSError('interrupted checkpoint')

        with self.assertRaisesRegex(OSError, 'interrupted checkpoint'):
            _publish_checkpoints(last, [best], fail)
        self.assertEqual(last.read_bytes(), b'committed last')
        self.assertEqual(best.read_bytes(), b'committed best')
        self.assertFalse(list(self.root.glob('.checkpoint-*')))

    def test_epoch_loss_callback_rejects_nonfinite_components_in_both_modes(self):
        for mode in ('fixed', 'early_stopping'):
            trainer = self.make_trainer('detect', mode)
            trainer.epoch = 2
            trainer.tloss = torch.tensor([0.1, 0.0, 0.2])
            trainer.run_callbacks('on_train_epoch_end')
            for bad_loss in (float('nan'), float('inf')):
                with self.subTest(mode=mode, loss=bad_loss):
                    trainer.tloss[1] = bad_loss
                    with self.assertRaisesRegex(NonFiniteTrainingError, 'Epoch 3: non-finite training loss'):
                        trainer.run_callbacks('on_train_epoch_end')

    def test_checkpoint_guards_reject_ema_half_overflow_and_invalid_training_state(self):
        trainer = self.make_trainer('detect')
        trainer._setup_train()
        trainer.save_model()
        previous = trainer.last.read_bytes()
        trainer.epoch = 1

        trainer.ema.ema.weight.fill_(1e5)  # Finite in FP32 and outside the FP16 range.
        with self.assertRaisesRegex(NonFiniteTrainingError, 'Epoch 2: non-finite EMA checkpoint tensors'):
            trainer.save_model()
        trainer.ema.ema.weight.fill_(1.0)

        trainer.optimizer.state_dict = lambda: {'state': {0: {'exp_avg_sq': torch.tensor([float('inf')])}}}
        with self.assertRaisesRegex(NonFiniteTrainingError, 'Epoch 2: non-finite optimizer checkpoint tensors'):
            trainer.save_model()

        self.assertEqual(trainer.last.read_bytes(), previous)
        self.assertFalse((self.root / 'epoch2.pt').exists())
        self.assertEqual(self.serialized.call_count, 1)

    def test_managed_resume_restores_raw_weights_full_ema_and_patience(self):
        policy = {'mode': 'early_stopping', 'save_period': 2, 'min_epochs': 3, 'patience': 2, 'min_delta': 0.05}
        on_checkpoint = Mock()
        trainer = self.make_trainer('detect', 'early_stopping', run_signature='frozen', on_checkpoint=on_checkpoint,
                                    optimizer_steps=lambda: 23)
        trainer.epoch = 3
        trainer.metrics, trainer.fitness = {}, 0.64
        trainer._validation_stop(1, 0.60)
        trainer._validation_stop(4, 0.64)
        trainer.model.weight.fill_(0.847321)
        trainer.ema.ema.weight.fill_(0.712345)
        moments = torch.tensor([1e5])  # Valid in the original optimizer precision.
        trainer.optimizer.state_dict = lambda: {'state': {0: {'exp_avg_sq': moments}}, 'param_groups': [{'lr': 0.01}]}
        trainer.save_model()
        on_checkpoint.assert_called_once_with(trainer, trainer.last)
        checkpoint = self.serialized.call_args.args[0]
        state = checkpoint['policy_resume']
        next_rng_values = (random.random(), np.random.random(), torch.rand(()).item())
        self.assertEqual(state['optimizer_steps'], 23)
        self.assertEqual(checkpoint['ema'].weight.dtype, torch.float16)
        self.assertEqual(state['model']['weight'].dtype, torch.float32)
        self.assertEqual(checkpoint['optimizer']['state'][0]['exp_avg_sq'].item(), 1e5)
        self.assertEqual(validate_policy_checkpoint(checkpoint, 'detect', policy, 12, run_signature='frozen'), 4)
        with self.assertRaisesRegex(ValueError, 'signature'):
            validate_policy_checkpoint(checkpoint, 'detect', policy, 12, run_signature='other')

        resumed = self.make_trainer('detect', 'early_stopping', resume=str(trainer.last), run_signature='frozen')
        resumed.resume_checkpoint = checkpoint
        resumed.setup_model()
        resumed.resume_training(checkpoint)
        resumed.run_callbacks('on_train_start')
        self.assertEqual((random.random(), np.random.random(), torch.rand(()).item()), next_rng_values)
        self.assertEqual(resumed.start_epoch, 4)
        torch.testing.assert_close(resumed.model.weight, trainer.model.weight, atol=0, rtol=0)
        torch.testing.assert_close(resumed.ema.ema.weight, trainer.ema.ema.weight, atol=0, rtol=0)
        self.assertEqual(resumed.ema.updates, trainer.ema.updates)
        resumed.optimizer.load_state_dict.assert_called_once_with(checkpoint['optimizer'])
        resumed.scaler.load_state_dict.assert_called_once_with(checkpoint['scaler'])
        resumed.scheduler.load_state_dict.assert_called_once_with(state['scheduler'])
        self.assertEqual(resumed.best_score, 0.64)
        self.assertEqual(resumed.best_epoch, 4)
        self.assertEqual(resumed._validation_stop.progress_epoch, 1)
        self.assertTrue(resumed._validation_stop(5, 0.64))

    def test_native_stop_state_preserved_and_terminal_or_old_checkpoint_rejected(self):
        trainer = self.make_trainer('detect', None)
        trainer.epoch = 4
        trainer.metrics = {}
        trainer.fitness = 0.55
        trainer.best_fitness = 0.6
        trainer.stopper.best_epoch = 2
        trainer.stopper.best_fitness = 0.6
        trainer.stopper.possible_stop = True
        trainer.save_model()
        checkpoint = self.serialized.call_args.args[0]
        self.assertFalse(trainer.best.exists())
        self.assertEqual(validate_policy_checkpoint(checkpoint, 'detect', None, 12), 5)
        resumed = self.make_trainer('detect', None, resume=str(trainer.last))
        resumed.resume_checkpoint = checkpoint
        resumed.setup_model()
        resumed.resume_training(checkpoint)
        self.assertEqual(resumed.best_fitness, 0.6)
        self.assertEqual(resumed.stopper.best_epoch, 2)
        self.assertTrue(resumed.stopper.possible_stop)
        with self.assertRaisesRegex(ValueError, 'lacks full training state'):
            validate_policy_checkpoint({'ema': checkpoint['ema'], 'epoch': 4}, 'detect', None, 12)
        trainer.stop = True
        trainer.save_model()
        checkpoint = self.serialized.call_args.args[0]
        self.assertTrue(checkpoint['policy_resume']['stopped_early'])
        with self.assertRaisesRegex(ValueError, 'finished training run'):
            validate_policy_checkpoint(checkpoint, 'detect', None, 12)


if __name__ == '__main__':
    unittest.main()
