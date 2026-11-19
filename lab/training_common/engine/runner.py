"""Bounded training, explicit evaluation and validation-only experiment helpers."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone
import csv
import hashlib
import itertools
import json
import math
import shutil
from tempfile import TemporaryDirectory
import time
import uuid
import yaml
from .config import TRAIN_KEYS, VALIDATING_MODES, load_config, resolve, validate
from .config import read_yaml
from .data import (_file_sha256, preflight_dataset, relocate_staged_data, stage_dataset,
                   staged_dataset_signature, validate_staged_resume_data)
from .runtime import environment, new_run, prepare_ultralytics, run_context, training_lock, write_json
from .reporting import diagnostics, metric_report, tensorboard_report, write_csv, match_detections, artifact_manifest
from .bounded import BudgetExhausted, command as child_command, run_bounded
from .training_policy import NonFiniteTrainingError, build_policy_trainer, validate_policy_checkpoint


def preflight(root, config):
    root = Path(root).resolve(); validate(config)
    model = resolve(root, config['model'])
    if model.suffix.lower() != '.pt' or not model.is_file():
        raise ValueError('Local pretrained/checkpoint .pt missing: ' + str(model))
    header, _, report = preflight_dataset(root, config)
    return {**report, 'class_names': header['names'], 'task': config['task'],
            'model': config['model'], 'environment': environment(root),
            **({'training_policy': config['training_policy'], 'epochs': config.get('epochs')}
               if 'training_policy' in config else {})}


def _training_options(config, data_path, run_dir):
    options = {'data': str(data_path), 'imgsz': config['imgsz'], 'epochs': config['epochs'],
               'batch': config['batch'], 'workers': config.get('workers', 0),
               'patience': config.get('patience', 30), 'seed': config.get('seed', 42),
               'device': config.get('device', '0'), 'project': str(run_dir), 'name': 'training',
               'exist_ok': False, 'plots': True, 'save': True, 'cache': False,
               'amp': config.get('amp', False), 'max_det': config.get('max_det', 300),
               'val': True, 'verbose': True}
    options.update({key: config[key] for key in TRAIN_KEYS if key in config})
    options.update(config.get('augment', {}))
    if policy := config.get('training_policy'):
        options.update(val=policy['mode'] in VALIDATING_MODES, patience=0,
                       save_period=policy['save_period'])
    if config['task'] == 'pose': options.update(fliplr=0.0, flipud=0.0)
    if config.get('time_hours'): options['time'] = config['time_hours']
    return options


def _phase_name(kind):
    return kind + datetime.now(timezone.utc).strftime('_%Y%m%dT%H%M%S_%fZ')


def _attach_tensorboard(model, run_dir, attempt=None):
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(Path(run_dir) / 'tensorboard' / (attempt or _phase_name('train'))))
    def epoch_metrics(trainer):
        epoch = trainer.epoch + 1
        metrics = dict(getattr(trainer, 'metrics', {}) or {})
        metrics.update(getattr(trainer, 'lr', {}) or {})
        loss = getattr(trainer, 'tloss', None)
        if loss is not None:
            metrics.update(trainer.label_loss_items(loss, prefix='train'))
        for key, value in metrics.items():
            try: value = float(value)
            except (TypeError, ValueError): continue
            if math.isfinite(value): writer.add_scalar(key, value, epoch)
        writer.flush()
    model.add_callback('on_fit_epoch_end', epoch_metrics)
    return writer


def assert_model_contract(model, data_path, config):
    """Evaluation requires a task-trained model with matching label semantics."""
    expected = read_yaml(data_path)
    expected_names = expected['names']
    if isinstance(expected_names, dict):
        expected_names = [expected_names[index] for index in range(len(expected_names))]
    actual_names = model.names
    if isinstance(actual_names, dict):
        if set(actual_names) != set(range(len(actual_names))):
            raise ValueError('Evaluation checkpoint class IDs are not consecutive')
        actual_names = [actual_names[index] for index in range(len(actual_names))]
    if list(map(str, actual_names)) != list(map(str, expected_names)):
        raise ValueError('Evaluation checkpoint class count/order/names differ from staged dataset')
    if getattr(model, 'task', None) != config['task']:
        raise ValueError('Evaluation checkpoint task differs from configured detect/pose task')
    if config['task'] == 'pose':
        shape = getattr(model.model, 'kpt_shape', None)
        if shape is None:
            try: shape = model.model.model[-1].kpt_shape
            except (AttributeError, IndexError, TypeError): shape = None
        if shape is None or list(shape) != expected.get('kpt_shape'):
            raise ValueError('Evaluation checkpoint keypoint shape differs from staged dataset')


def _evaluate_loaded(model, config, run_dir, data_path, split, name):
    assert_model_contract(model, data_path, config)
    if Path(name).name != name or name in {'.', '..'}: raise ValueError('Invalid evaluation name')
    destination = Path(run_dir) / 'evaluation' / name
    if destination.exists(): raise FileExistsError('Evaluation already exists: ' + str(destination))
    options = {'data': str(data_path), 'split': split, 'imgsz': config['imgsz'],
               'batch': config['batch'], 'workers': config.get('workers', 0),
               'device': config.get('device', '0'), 'conf': .001, 'max_det': config.get('max_det', 300),
               'iou': config.get('nms_iou', .7),
               'plots': True, 'save_json': True, 'save_txt': True, 'save_conf': True,
               'project': str(Path(run_dir) / 'evaluation'), 'name': name, 'exist_ok': False}
    metrics = model.val(**options)
    actual = Path(metrics.save_dir).resolve()
    if actual != destination.resolve():
        raise RuntimeError('Validator output differs from requested path: ' + str(actual))
    summary, class_rows = metric_report(metrics, data_path, split, config, actual)
    diagnostic = diagnostics(model, data_path, split, config, actual)
    summary['diagnostics'] = diagnostic
    summary['data_selection'] = json.loads((Path(data_path).parent / 'selection.json').read_text(encoding='utf-8'))
    selected_records = json.loads((Path(data_path).parent / 'samples.json').read_text(encoding='utf-8'))
    summary['evaluation_sources'] = sorted(row.get('workspace_image', row['image']) for row in selected_records if row['split'] == split)
    summary['evaluation_config'] = {'imgsz': config['imgsz'], 'class_names': config.get('class_names'),
                                    'diagnostic_conf': config.get('diagnostic_conf', .25), 'match_iou': config.get('match_iou', .5)}
    write_json(actual / 'summary.json', summary)
    if config['model_key'] == 'meter':
        from .meter_report import write_meter_report
        report = write_meter_report(actual)
        summary['task_report'] = report.relative_to(actual).as_posix()
    if config['model_key'] == 'waste_sticker':
        from .sticker_report import write_sticker_report
        report = write_sticker_report(actual, summary['evaluation_config']['class_names'])
        summary['task_report'] = report.relative_to(actual).as_posix()
    tensorboard_report(run_dir, name, summary, class_rows, diagnostic)
    manifest = artifact_manifest(actual, config['task'])
    summary['artifact_manifest'] = manifest.relative_to(actual).as_posix()
    write_json(actual / 'summary.json', summary)
    return summary


def _resume_checkpoint(root, value):
    checkpoint = resolve(root, value, allow_absolute=True)
    if checkpoint.is_dir():
        checkpoint = resolve(root, checkpoint / 'training/weights/last.pt', allow_absolute=True)
    if tuple(checkpoint.parts[-3:]) != ('training', 'weights', 'last.pt'):
        raise ValueError('Resume expects an engine run or its training/weights/last.pt')
    if not checkpoint.is_file() or 'runs' not in checkpoint.relative_to(root).parts[:-3]:
        raise ValueError('Resume checkpoint is missing or outside runs: ' + str(checkpoint))
    return checkpoint


def recorded_resume_config(root, resume):
    checkpoint = _resume_checkpoint(root, resume)
    return validate(json.loads((checkpoint.parents[2] / 'config/effective.json').read_text(encoding='utf-8')),
                    require_epochs=True)


def resume_configuration(root, requested, run_dir, checkpoint):
    """Use the recorded run and staged inputs; source-package paths are not reused."""
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    if not run_dir.is_relative_to(root) or 'runs' not in run_dir.relative_to(root).parts:
        raise ValueError('Resume directory must be inside runs')
    previous = recorded_resume_config(root, checkpoint)
    requested = requested if requested is not None else previous
    allowed = {'batch', 'workers', 'device'}
    unused_inputs = {'data', 'model', 'data_overrides', 'output_root', 'amp_check_model', 'source_notes', 'description'}
    semantic = lambda value: {key:item for key,item in value.items() if key not in allowed | unused_inputs}
    if semantic(previous) != semantic(requested):
        raise ValueError('Resume config differs from recorded training semantics; only batch/workers/device may change')
    stage = run_dir / 'data'
    data_signature = validate_staged_resume_data(run_dir, previous)
    effective = deepcopy(previous)
    for key in allowed:
        if key in requested: effective[key] = requested[key]
    effective.update(data=(stage / 'data.yaml').relative_to(root).as_posix(),
                     model=Path(checkpoint).resolve().relative_to(root).as_posix(),
                     output_root=run_dir.parent.relative_to(root).as_posix())
    effective.pop('data_overrides', None)
    if requested.get('amp_check_model'): effective['amp_check_model'] = requested['amp_check_model']
    return effective, {'input_authority': 'Existing run/data and config/effective.json',
                       'ignored_source_fields': {key:requested[key] for key in sorted(unused_inputs) if key in requested},
                       'allowed_overrides': {key:effective.get(key) for key in sorted(allowed)},
                       'staged_data': effective['data'], 'checkpoint': effective['model'],
                       'dataset_signature': data_signature}


def _run_signature(run_dir, data_signature):
    recorded = json.loads((run_dir / 'config/effective.json').read_text(encoding='utf-8'))
    payload = json.dumps([recorded, data_signature], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _restore_checkpoint_outputs(run_dir, checkpoint_path, checkpoint, completed):
    """Rollback the CSV to the checkpoint and finish any same-epoch weight copies."""
    results = checkpoint.get('train_results')
    if (not isinstance(results, dict) or results.get('epoch') != list(range(1, completed + 1))
            or any(not isinstance(values, list) or len(values) != completed for values in results.values())):
        raise ValueError('Checkpoint does not contain a complete training metrics history')
    training = run_dir / 'training'
    state = checkpoint['policy_resume']
    copies = ['best.pt'] if state['best_checkpoint'] else []
    if state['periodic_checkpoint']:
        copies.append(state['periodic_checkpoint'])
    if state['periodic_checkpoint'] not in (None, f'epoch{completed}.pt', f'epoch{completed - 1}.pt'):
        raise ValueError('Checkpoint has an invalid periodic weights name')
    if (state['stopper'].get('best_epoch') and 'best.pt' not in copies
            and not (checkpoint_path.parent / 'best.pt').is_file()):
        raise ValueError('The run is missing its selected best.pt; copy the full training run')
    with TemporaryDirectory(prefix='.resume-', dir=training) as temporary:
        pending = Path(temporary) / 'results.csv'
        with pending.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(results.keys())
            writer.writerows(zip(*results.values(), strict=True))
        pending.replace(training / 'results.csv')
        for name in copies:
            pending = Path(temporary) / name
            shutil.copyfile(checkpoint_path, pending)
            pending.replace(checkpoint_path.parent / name)


def train_run(root, config, run_name=None, resume=None, *, attempt=None, controller_wall_seconds=None):
    root = Path(root).resolve()
    if controller_wall_seconds is not None and (not math.isfinite(controller_wall_seconds)
                                                 or controller_wall_seconds <= 0):
        raise ValueError('controller_wall_seconds must be finite and positive')
    if attempt is not None and (not resume or not attempt.startswith('resume_') or not attempt.isidentifier()):
        raise ValueError('Invalid resume attempt identifier')
    if resume:
        if run_name:
            raise ValueError('Resume uses the original run directory')
        model_path = _resume_checkpoint(root, resume)
        run_dir = model_path.parents[2]
        attempt = attempt or _phase_name('resume')
    else:
        config = validate(deepcopy(config), require_epochs=True)
        model_path = resolve(root, config['model'])
        run_dir = None
    if not model_path.is_file() or model_path.suffix.lower() != '.pt':
        raise ValueError('Local model/checkpoint missing: ' + str(model_path))
    if resume:
        config, resume_mapping = resume_configuration(root, config, run_dir, model_path)
    else:
        run_dir = new_run(root, config, run_name)
    print('RUN_DIR:', run_dir, flush=True)
    with training_lock(run_dir), run_context(root, config, run_dir, attempt or 'train'):
        if controller_wall_seconds is not None:
            write_json(run_dir / 'metadata/wall_budget.json',
                       {'wall_seconds': controller_wall_seconds, 'attempt': attempt or 'train'})
        if config['version'].lower() == 'v4' and not resume:
            write_json(run_dir / 'metadata/final_test_status.json', {'status': 'not_executed'})
        if resume:
            data_path = run_dir / 'data/data.yaml'
            data_signature = resume_mapping['dataset_signature']
        else:
            data_path, selection = stage_dataset(root, config, run_dir)
            print('Data selection:', selection, flush=True)
            data_signature = staged_dataset_signature(run_dir)
        run_signature = _run_signature(run_dir, data_signature)
        suffix = '_' + attempt if attempt else ''
        config_yaml = run_dir / 'config' / ('effective' + suffix + '.yaml')
        config_yaml.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
        if config.get('amp', False) and not (resume and (run_dir / 'yolov8n.pt').is_file()):
            if not config.get('amp_check_model'):
                raise ValueError('amp=True requires a local amp_check_model for offline AMP checks')
            shutil.copy2(resolve(root, config['amp_check_model']), run_dir / 'yolov8n.pt')
        YOLO = prepare_ultralytics(run_dir)
        import torch
        write_json(run_dir / 'metadata' / ('hardware' + suffix + '.json'), {'cuda_available': torch.cuda.is_available(),
                    'torch_cuda': torch.version.cuda,
                    'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
        model = YOLO(str(model_path), task=config['task'])
        policy = config.get('training_policy')
        legacy_resume = False
        initial_steps = initial_epochs = 0
        if resume:
            assert_model_contract(model, data_path, config)
            checkpoint = model.ckpt
            if 'policy_resume' not in checkpoint and config['version'].lower() == 'v3':
                # Preserve the original V3 compatibility path for stock Ultralytics checkpoints.
                initial_epochs = checkpoint.get('epoch', -1) + 1
                if not 0 < initial_epochs < config['epochs'] or not checkpoint.get('optimizer'):
                    raise ValueError('V3 checkpoint has no unfinished optimizer state')
                legacy_resume = True
                print('Legacy V3 checkpoint: using native Ultralytics resume with limited early-stopping history.',
                      flush=True)
            else:
                initial_epochs = validate_policy_checkpoint(checkpoint, config['task'], policy,
                                                            config['epochs'], run_signature=run_signature)
                initial_steps = checkpoint['policy_resume']['optimizer_steps']
                previous = recorded_resume_config(root, model_path)
                if config['batch'] != previous['batch']:
                    raise ValueError('Resume must preserve the recorded batch size and gradient accumulation')
                _restore_checkpoint_outputs(run_dir, model_path, checkpoint, initial_epochs)
            resume_mapping.update(checkpoint_epochs=initial_epochs, checkpoint_optimizer_steps=initial_steps,
                                  run_signature=run_signature, legacy_checkpoint=legacy_resume)
            data_path = relocate_staged_data(run_dir)
            write_json(run_dir / 'metadata' / ('resume_mapping' + suffix + '.json'), resume_mapping)
        writer = _attach_tensorboard(model, run_dir, attempt)
        options = _training_options(config, data_path, run_dir)
        if resume:
            options['exist_ok'] = True
        write_json(run_dir / 'config' / ('framework_options' + suffix + '.json'), options)
        recorder = None
        if not legacy_resume:
            from .telemetry import attach_update_recorder
            recorder = attach_update_recorder(model, run_dir, config['epochs'], attempt=attempt,
                                              initial_steps=initial_steps, initial_epochs=initial_epochs)
        reason = 'failed'
        try:
            if legacy_resume:
                if config['task'] == 'pose':
                    from ultralytics.models.yolo.pose import PoseTrainer as BaseTrainer
                else:
                    from ultralytics.models.yolo.detect import DetectionTrainer as BaseTrainer
                class RelocatedResumeTrainer(BaseTrainer):
                    def check_resume(self, overrides):
                        super().check_resume(overrides)
                        if self.resume:
                            for key in ('data', 'project', 'name', 'workers'):
                                setattr(self.args, key, options[key])
                            self.args.exist_ok = True
                            self.args.save_dir = str(run_dir / 'training')
                model.train(resume=True, trainer=RelocatedResumeTrainer, **options)
            else:
                trainer_type = build_policy_trainer(config['task'], policy, run_signature=run_signature,
                                                    optimizer_steps=lambda: recorder.total_steps)
                model.train(resume=bool(resume), trainer=trainer_type, **options)
            if recorder is not None:
                if not recorder.registered:
                    raise RuntimeError('Training returned without an optimizer-step recorder')
                if recorder.steps == 0:
                    reason = 'zero_optimizer_updates'
                    raise RuntimeError('Training completed without any optimizer updates')
                trainer = getattr(model, 'trainer', None)
                completed_epoch = getattr(trainer, 'epoch', -1) + 1
                if completed_epoch >= config['epochs']:
                    reason = 'completed_epoch_budget'
                elif trainer.stopped_early:
                    reason = 'validation_early_stop'
                elif policy:
                    raise RuntimeError('Training returned before its budget without a validation early stop')
                else:
                    reason = 'framework_stopped_early_reason_unconfirmed'
        except NonFiniteTrainingError:
            reason = 'non_finite_training'
            raise
        except KeyboardInterrupt:
            reason = 'interrupted'
            raise
        finally:
            if recorder is not None:
                recorder.close(reason, getattr(model, 'trainer', None))
            writer.close()
        best = run_dir / 'training/weights/best.pt'
        if policy:
            artifacts = _record_policy_artifacts(run_dir, model.trainer, policy, reason)
            del model
            _finalize_policy_training(YOLO, config, run_dir, data_path, artifacts)
            return run_dir
        if not best.is_file(): raise RuntimeError('Training completed without best.pt')
        write_json(run_dir / 'metadata/artifacts.json', {'best': 'training/weights/best.pt',
                    'last': 'training/weights/last.pt', 'training_metrics': 'training/results.csv',
                    'tensorboard': 'tensorboard', 'dataset': 'data/data.yaml'})
        if config.get('evaluate_after_training', True):
            name = 'val' if not resume else 'val_resume_' + str(int(time.time()))
            _evaluate_loaded(YOLO(str(best), task=config['task']), config, run_dir, data_path, 'val', name)
    return run_dir


def _record_policy_artifacts(run_dir, trainer, policy, reason):
    weights = run_dir / 'training/weights'
    last = weights / 'last.pt'
    best = weights / 'best.pt'
    completed_epochs = trainer.epoch + 1
    period = policy['save_period']
    checkpoints = [weights / f'epoch{epoch}.pt' for epoch in range(period, completed_epochs + 1, period)]
    if any(not path.is_file() for path in [last, *checkpoints]):
        raise RuntimeError('Training completed without its last or scheduled checkpoint')
    selection = None
    if policy['mode'] in VALIDATING_MODES:
        if not best.is_file() or trainer.best_epoch is None or trainer.best_score is None:
            raise RuntimeError('Validation training completed without a selected checkpoint')
        selection = {'metric': 'Box mAP50-95' if trainer.args.task == 'detect' else 'Pose mAP50-95',
                     'split': 'val', 'epoch': trainer.best_epoch, 'score': trainer.best_score}
    elif best.exists():
        raise RuntimeError('Fixed-budget training unexpectedly produced a selected checkpoint')
    artifacts = {
        'training_policy': policy, 'stop_reason': reason, 'completed_epochs': completed_epochs,
        'last': last.relative_to(run_dir).as_posix(),
        'best': best.relative_to(run_dir).as_posix() if selection else None,
        'selection': selection,
        'periodic_checkpoints': [path.relative_to(run_dir).as_posix() for path in checkpoints],
        'training_metrics': 'training/results.csv', 'tensorboard': 'tensorboard',
        'dataset': 'data/data.yaml', 'post_training': 'metadata/post_training.json',
        'best_format': 'training_checkpoint' if selection else None,
    }
    write_json(run_dir / 'metadata/artifacts.json', artifacts)
    return artifacts


def _export_inference_checkpoint(source, destination):
    """Apply YOLO FP16 inference finalization without changing the resume checkpoint."""
    from ultralytics.utils.torch_utils import strip_optimizer

    with TemporaryDirectory(prefix='.inference-', dir=destination.parent) as temporary:
        pending = Path(temporary) / destination.name
        # Ultralytics strips its own fields. Drop the additional V4 training tensors as well.
        stripped = strip_optimizer(source, s=str(pending), updates={'policy_resume': None, 'scaler': None})
        if not stripped or not pending.is_file():
            raise RuntimeError('Could not export an inference checkpoint from ' + str(source))
        pending.replace(destination)


def _finalize_policy_training(YOLO, config, run_dir, data_path, artifacts):
    status_path = run_dir / artifacts['post_training']
    artifact_path = run_dir / 'metadata/artifacts.json'
    state = {'status': 'running', 'phase': 'export', 'evaluation': None, 'inference_weights': {}}
    write_json(status_path, state)
    try:
        last = run_dir / artifacts['last']
        last_inference = last.with_name('last_inference.pt')
        _export_inference_checkpoint(last, last_inference)
        artifacts['last_inference'] = last_inference.relative_to(run_dir).as_posix()
        state['inference_weights']['last'] = {'path': artifacts['last_inference'], 'bytes': last_inference.stat().st_size}
        write_json(artifact_path, artifacts)
        write_json(status_path, state)
        if artifacts['best']:
            best = run_dir / artifacts['best']
            _export_inference_checkpoint(best, best)
            artifacts['best_format'] = 'inference_fp16'
            state['inference_weights']['best'] = {'path': artifacts['best'], 'bytes': best.stat().st_size}
            write_json(artifact_path, artifacts)
            write_json(status_path, state)
        if config.get('evaluate_after_training', False):
            name = 'val_final'
            state.update(phase='validation', evaluation=f'evaluation/{name}/summary.json')
            write_json(status_path, state)
            best = run_dir / artifacts['best']
            summary = _evaluate_loaded(YOLO(str(best), task=config['task']), config, run_dir, data_path, 'val', name)
            summary.update(checkpoint=artifacts['best'], checkpoint_sha256=_file_sha256(best),
                           model_selection=artifacts['selection'])
            write_json(run_dir / state['evaluation'], summary)
        state['status'] = 'complete'
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                     error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(status_path, state)


def evaluate_run(root, config, weights, split='val', run_dir=None, name=None):
    if split not in ('val', 'test'): raise ValueError('Evaluation split must be val or test')
    root = Path(root).resolve(); config = validate(deepcopy(config))
    if config['version'].lower() == 'v4' and split == 'test':
        return evaluate_frozen_test(root, config, weights, run_dir, name)
    weight_path = resolve(root, weights, allow_absolute=True)
    if not weight_path.is_file(): raise ValueError('Evaluation checkpoint missing: ' + str(weight_path))
    if run_dir is None:
        run_dir = new_run(root, config)
    else:
        run_dir = resolve(root, run_dir, allow_absolute=True)
        if 'runs' not in run_dir.relative_to(root).parts:
            raise ValueError('Evaluation run must be inside runs')
    name = name or split
    if Path(name).name != name or name in {'.', '..'}:
        raise ValueError('Invalid evaluation name')
    if config['version'].lower() == 'v4' and (
            (run_dir / 'evaluation' / name).exists() or
            (run_dir / 'metadata/evaluation_inputs' / name).exists()):
        raise FileExistsError('V4 evaluation name already used: ' + name)
    with run_context(root, config, run_dir, 'evaluate_' + name):
        data_path = run_dir / 'data/data.yaml'
        if not data_path.exists():
            data_path, _ = stage_dataset(root, config, run_dir, training=False,
                                         validation_only=config['version'].lower() == 'v4' and split == 'val')
        else:
            data_path = relocate_staged_data(run_dir)
        if config.get('training_policy') and not read_yaml(data_path).get(split):
            raise ValueError('The training data snapshot has no ' + split + ' split for evaluation')
        YOLO = prepare_ultralytics(run_dir)
        snapshot = None
        dataset_snapshot = None
        if config['version'].lower() == 'v4' and split == 'val':
            snapshot_dir = run_dir / 'metadata/evaluation_inputs' / name
            snapshot_dir.mkdir(parents=True, exist_ok=False)
            snapshot = snapshot_dir / 'checkpoint.pt'
            with weight_path.open('rb') as source, snapshot.open('xb') as target:
                shutil.copyfileobj(source, target)
            shutil.copytree(run_dir / 'data', snapshot_dir / 'data', ignore=shutil.ignore_patterns('*.cache'))
            dataset_snapshot = relocate_staged_data(snapshot_dir)
            data_path = dataset_snapshot
        result = _evaluate_loaded(YOLO(str(snapshot or weight_path), task=config['task']), config, run_dir, data_path, split, name)
        result['checkpoint'] = weight_path.relative_to(root).as_posix()
        if snapshot is not None:
            result['checkpoint_snapshot'] = snapshot.relative_to(root).as_posix()
            result['dataset_snapshot'] = dataset_snapshot.relative_to(root).as_posix()
        write_json(run_dir / 'evaluation' / name / 'summary.json', result)
    return run_dir


def evaluate_frozen_test(root, config, weights, run_dir, name=None):
    """Evaluate the frozen V4 selection once, retaining rejection and failure state."""
    from .freeze_gate import frozen_dataset_path

    root = Path(root).resolve()
    if run_dir is None:
        raise ValueError('V4 final test requires the run containing a frozen selection')
    run_dir = resolve(root, run_dir, allow_absolute=True)
    if 'runs' not in run_dir.relative_to(root).parts or not run_dir.is_dir():
        raise ValueError('V4 final test run must be an existing run inside runs')
    name = name or 'test'
    if Path(name).name != name or name in {'.', '..'}:
        raise ValueError('Invalid evaluation name')
    status_path = run_dir / 'metadata/final_test_status.json'
    attempts = run_dir / 'metadata/final_test_attempts'
    attempts.mkdir(parents=True, exist_ok=True)
    attempt_path = attempts / (uuid.uuid4().hex + '.json')
    previous = json.loads(status_path.read_text(encoding='utf-8')) if status_path.is_file() else {}
    if (run_dir / 'evaluation' / name).exists() or (run_dir / 'metadata/evaluation_inputs' / name).exists():
        write_json(attempt_path, {'status': 'rejected', 'reason': 'Evaluation name already used'})
        raise FileExistsError('V4 evaluation name already used: ' + name)
    if previous.get('status') in ('complete', 'running'):
        write_json(attempt_path, {'status': 'rejected', 'reason': 'Final test already completed or running'})
        raise ValueError('Final test already completed or running')
    try:
        if config.get('test_unavailable'):
            raise ValueError('Final test is unavailable; add an audited test set before testing')
        data_path = frozen_dataset_path(root, config, run_dir, weights)
    except BaseException as error:
        write_json(attempt_path, {'status': 'rejected', 'error': f'{type(error).__name__}: {error}'})
        if not status_path.exists():
            write_json(status_path, {'status': 'not_executed'})
        raise
    state = {'status': 'running', 'evaluation': 'evaluation/' + name,
             'selection': 'frozen_selection/record.json'}
    write_json(status_path, state); write_json(attempt_path, state)
    try:
        with run_context(root, config, run_dir, 'evaluate_' + name):
            YOLO = prepare_ultralytics(run_dir)
            selected = resolve(root, weights, allow_absolute=True)
            result = _evaluate_loaded(YOLO(str(selected), task=config['task']), config, run_dir, data_path, 'test', name)
            result['checkpoint'] = selected.relative_to(root).as_posix()
            result['selection'] = 'frozen_selection/record.json'
            write_json(run_dir / 'evaluation' / name / 'summary.json', result)
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                     error=f'{type(error).__name__}: {error}')
        write_json(status_path, state); write_json(attempt_path, state)
        raise
    state['status'] = 'complete'
    write_json(status_path, state); write_json(attempt_path, state)
    return run_dir


def compare_runs(root, config, paths, run_name=None):
    root = Path(root).resolve(); run_dir = new_run(root, config, run_name)
    with run_context(root, config, run_dir, 'compare'):
        rows, definitions = [], []
        for path in paths:
            directory = resolve(root, path, allow_absolute=True)
            environment_path = directory / 'metadata/environment.json'
            env = json.loads(environment_path.read_text(encoding='utf-8')) if environment_path.exists() else {}
            for summary_path in sorted(directory.glob('evaluation/*/summary.json')):
                summary = json.loads(summary_path.read_text(encoding='utf-8'))
                row = {'run': directory.relative_to(root).as_posix(), 'evaluation': summary_path.parent.name,
                       'model_key': summary.get('model_key'), 'version': summary.get('version'), 'split': summary['split'],
                       'images': summary['images'], 'torch': env.get('packages', {}).get('torch'),
                       'ultralytics': env.get('packages', {}).get('ultralytics'), **summary['metrics']}
                rows.append(row)
                definitions.append({'run': row['run'], 'evaluation': row['evaluation'],
                                    'model_key': row['model_key'], 'split': row['split'],
                                    'selection': summary.get('data_selection'), 'source_notes': summary.get('source_notes'),
                                    'sources': summary.get('evaluation_sources'), 'evaluation_config': summary.get('evaluation_config'),
                                    'metric_library': row['ultralytics']})
        if not rows: raise ValueError('No evaluation summaries found in supplied runs')
        fields = list(dict.fromkeys(key for row in rows for key in row))
        write_csv(run_dir / 'comparison.csv', rows, fields)
        groups = {}
        for definition in definitions:
            key = str(definition['model_key']) + '/' + definition['split']
            groups.setdefault(key, []).append(definition)
        comparable = {}
        for key, items in groups.items():
            known = all(item['sources'] is not None for item in items)
            same_sources = known and all(item['sources'] == items[0]['sources'] for item in items)
            same_classes = all((item.get('evaluation_config') or {}).get('class_names') ==
                               (items[0].get('evaluation_config') or {}).get('class_names') for item in items)
            same_library = all(item.get('metric_library') == items[0].get('metric_library') for item in items)
            comparable[key] = {'same_evaluation_sources': bool(same_sources), 'same_class_order': same_classes,
                               'same_metric_library_version': same_library,
                               'comparable': bool(same_sources and same_classes and same_library),
                               'label_review': 'Source identities compared; annotation semantics remain subject to recorded data review.'}
        write_json(run_dir / 'comparison.json', {'runs': rows, 'definitions': definitions, 'groups': comparable,
                   'interpretation': 'Direct comparison requires matching evaluation samples and class order; environment and training budgets are reported.'})
        import matplotlib.pyplot as plt
        metric_keys = [key for key in fields if key.startswith('metrics/mAP50-95')]
        for group_key, compatibility in comparable.items():
            if not compatibility['comparable']: continue
            selected_rows = [row for row in rows if str(row['model_key']) + '/' + row['split'] == group_key]
            for index, key in enumerate(metric_keys):
                available = [(row['version'] + '/' + row['evaluation'], row.get(key)) for row in selected_rows if row.get(key) is not None]
                if not available: continue
                fig, ax = plt.subplots(figsize=(max(7, len(available)*1.1), 5))
                ax.bar([r[0] for r in available], [r[1] for r in available]); ax.set_ylim(0, 1)
                ax.set_title(group_key); ax.set_ylabel(key); ax.tick_params(axis='x', rotation=30)
                filename = 'comparison_' + group_key.replace('/', '_') + '_' + str(index) + '.png'
                fig.tight_layout(); fig.savefig(run_dir / filename, dpi=150); plt.close(fig)
    return run_dir


def _mark_budget_exhausted(run_dir, phase):
    path = Path(run_dir) / 'metadata' / (phase + '_status.json')
    state = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'phase': phase}
    state.update(status='budget_exhausted', error='Parent wall-clock deadline stopped the experiment')
    write_json(path, state)
    write_json(Path(run_dir) / 'metadata/status.json', state)


def _record_abrupt_training_stop(run_dir, status, message, *, attempt=None):
    """Record forced termination without claiming a complete optimizer-step count."""
    run_dir = Path(run_dir)
    metadata = run_dir / 'metadata'
    path = metadata / ((attempt or 'train') + '_status.json')
    entered_run = not attempt or path.exists()
    try:
        state = json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError):
        state = {'phase': attempt or 'train'}
    if state.get('status') not in ('complete', 'failed', 'interrupted', 'budget_exhausted'):
        state.update(status=status, error=message)
        write_json(path, state)
        if entered_run:
            write_json(metadata / 'status.json', state)
    suffix = '_' + attempt if attempt else ''
    finalization_path = metadata / 'post_training.json'
    try:
        finalization = json.loads(finalization_path.read_text(encoding='utf-8'))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError):
        finalization = {}
    if finalization.get('status') == 'running':
        finalization.update(status=status, error=message)
        write_json(finalization_path, finalization)
    update_log = metadata / ('optimizer_updates' + suffix + '.jsonl')
    summary_path = metadata / ('optimization_summary' + suffix + '.json')
    if not summary_path.exists():
        recorded = 0
        if update_log.is_file():
            for line in update_log.read_text(encoding='utf-8').splitlines():
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    break
                recorded += 1
        summary = {'status': status, 'recorded_optimizer_steps': recorded,
                   'actual_optimizer_steps': None,
                   'step_log': update_log.name if update_log.is_file() else None,
                   'scope': 'An abrupt stop can interrupt recording; completed records are a lower bound.'}
        write_json(summary_path, summary)
        if attempt and entered_run:
            write_json(metadata / 'optimization_summary.json', summary)
    if not (run_dir / 'metadata/final_test_status.json').exists():
        write_json(run_dir / 'metadata/final_test_status.json', {'status': 'not_executed'})


def bounded_train(root, config, config_path, wall_seconds, run_name=None, *, resume=None):
    """Enforce an external deadline without changing framework scheduler settings."""
    root = Path(root).resolve()
    attempt = None
    if resume:
        if run_name:
            raise ValueError('Resume uses the original run directory')
        checkpoint = _resume_checkpoint(root, resume)
        resume = checkpoint.relative_to(root).as_posix()
        run_dir = checkpoint.parents[2]
        output = run_dir.parent
        name = None
        attempt = _phase_name('resume')
        config = config if config is not None else recorded_resume_config(root, checkpoint)
    else:
        name = run_name or (config['version'] + '_' + uuid.uuid4().hex)
        if Path(name).name != name or name in {'.', '..'}:
            raise ValueError('run-name must be one directory name')
        output = resolve(root, config['output_root'])
        run_dir = output / name
        if run_dir.exists():
            raise FileExistsError('Run already exists: ' + str(run_dir))
    validate(config, require_epochs=True)
    seconds = float(wall_seconds)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('wall_seconds must be finite and positive')
    controller = output / '_controllers' / uuid.uuid4().hex
    controller.mkdir(parents=True, exist_ok=False)
    state = {'phase': 'train_controller', 'status': 'running', 'wall_seconds': seconds,
             'run': run_dir.relative_to(root).as_posix(), 'elapsed_seconds': None, 'attempt': attempt}
    write_json(controller / 'status.json', state)
    started = time.monotonic()
    try:
        if resume or config.get('training_policy'):
            # Freeze the effective budget, including any CLI override, before launching the child.
            snapshot = controller / 'train.yaml'
            snapshot.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
                                encoding='utf-8')
            config_path = snapshot.relative_to(root).as_posix()
        arguments = child_command('train', root, config_path, run_name=name, resume=resume, attempt=attempt,
                                  controller_wall_seconds=seconds)
        run_bounded(arguments, started + seconds, controller / 'child.log', root, display_output=True)
        state['status'] = 'complete'
    except BudgetExhausted as error:
        state.update(status='budget_exhausted', error=str(error))
        _record_abrupt_training_stop(run_dir, 'budget_exhausted', str(error), attempt=attempt)
        raise
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                     error=f'{type(error).__name__}: {error}')
        if resume or config['version'].lower() == 'v4':
            _record_abrupt_training_stop(run_dir, state['status'], state['error'], attempt=attempt)
        raise
    finally:
        state['elapsed_seconds'] = time.monotonic() - started
        write_json(controller / 'status.json', state)
    return run_dir


def tune_thresholds(root, config, weights, run_name=None):
    """Bound the whole validation scan, including staging and diagnostic output."""
    root = Path(root).resolve(); validate(config)
    hours = float(config.get('tune', {}).get('max_hours', 1))
    if not math.isfinite(hours) or hours <= 0: raise ValueError('max_hours must be finite and positive')
    deadline = time.monotonic() + hours * 3600
    run_dir = new_run(root, config, run_name)
    with run_context(root, config, run_dir, 'threshold_controller'):
        config_path = run_dir / 'config/threshold_scan.yaml'
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
        arguments = child_command('thresholds', root, config_path.relative_to(root).as_posix(),
                                  run_dir=run_dir.relative_to(root).as_posix(), weights=weights)
        try:
            run_bounded(arguments, deadline, run_dir / 'logs/threshold_child.log', run_dir)
        except BudgetExhausted:
            partial = run_dir / 'thresholds_partial.json'
            contents = json.loads(partial.read_text(encoding='utf-8')) if partial.exists() else {'candidates': []}
            contents.update(status='budget_exhausted', split='val', test_used_for_selection=False,
                            selected_validation_candidate=None)
            write_json(run_dir / 'thresholds.json', contents)
            _mark_budget_exhausted(run_dir, 'threshold_tuning')
            raise
        if not (run_dir / 'thresholds.json').is_file():
            raise RuntimeError('Threshold child exited without its result summary')
    return run_dir


def threshold_scan_in_process(root, config, weights, run_name=None, run_dir=None):
    """Scan confidence and NMS IoU on val only, without retraining or test access."""
    root = Path(root).resolve(); validate(config)
    tune = config.get('tune', {})
    allowed = {'confidence', 'iou', 'max_trials', 'max_hours', 'objective', 'min_precision'}
    if set(tune) - allowed: raise ValueError('Unknown threshold tuning options')
    confidences = tune.get('confidence', [.1, .25, .4, .6])
    ious = tune.get('iou', [.5, .7])
    if any(not isinstance(values, list) or not values or len(values) > 32 for values in (confidences, ious)):
        raise ValueError('Threshold lists must contain 1 to 32 values')
    if any(not 0 < float(value) <= 1 for value in confidences + ious):
        raise ValueError('Confidence and IoU must be in (0, 1]')
    maximum = tune.get('max_trials', len(confidences)*len(ious))
    hours = tune.get('max_hours', 1)
    if type(maximum) is not int or not 1 <= maximum <= 100 or float(hours) <= 0:
        raise ValueError('Threshold tuning requires max_trials 1..100 and positive max_hours')
    objective = tune.get('objective', 'f1')
    if objective not in ('f1', 'recall', 'precision'): raise ValueError('Invalid threshold tuning objective')
    minimum_precision = float(tune.get('min_precision', 0))
    if not 0 <= minimum_precision <= 1: raise ValueError('min_precision must be in [0, 1]')
    checkpoint = resolve(root, weights, allow_absolute=True)
    if not checkpoint.is_file(): raise ValueError('Threshold tuning checkpoint missing')
    run_dir = resolve(root, run_dir, allow_absolute=True) if run_dir is not None else new_run(root, config, run_name)
    with run_context(root, config, run_dir, 'threshold_tuning'):
        data_path, _ = stage_dataset(root, config, run_dir, training=False)
        YOLO = prepare_ultralytics(run_dir); model = YOLO(str(checkpoint), task=config['task'])
        assert_model_contract(model, data_path, config)
        started = time.monotonic(); rows = []
        for index, nms in enumerate(ious):
            if len(rows) >= maximum: break
            probe = deepcopy(config); probe.update(diagnostic_conf=min(confidences), nms_iou=nms)
            destination = run_dir / 'thresholds' / f'nms_{index+1:02d}'
            diagnostics(model, data_path, 'val', probe, destination)
            predictions = [json.loads(line) for line in (destination / 'predictions/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
            for confidence in confidences:
                if len(rows) >= maximum: break
                tp = fp = fn = negative_fp = 0
                for sample in predictions:
                    selected = [pred for pred in sample['predictions'] if pred['confidence'] >= confidence]
                    matches, missed, extra = match_detections(sample['ground_truth'], selected, config.get('match_iou', .5))
                    tp += len(matches); fp += len(extra); fn += len(missed)
                    negative_fp += int(not sample['ground_truth'] and bool(selected))
                precision = tp/(tp+fp) if tp+fp else 0.0
                recall = tp/(tp+fn) if tp+fn else 0.0
                rows.append({'confidence': confidence, 'nms_iou': nms, 'images': len(predictions),
                             'tp': tp, 'fp': fp, 'fn': fn, 'negative_images_with_fp': negative_fp,
                             'precision': precision, 'recall': recall,
                             'f1': 2*precision*recall/(precision+recall) if precision+recall else 0.0})
                write_json(run_dir / 'thresholds_partial.json', {'status': 'partial', 'split': 'val', 'candidates': rows})
        eligible = [row for row in rows if row['precision'] >= minimum_precision]
        if not rows: raise RuntimeError('No threshold candidates evaluated')
        write_csv(run_dir / 'thresholds.csv', rows, list(rows[0]))
        write_json(run_dir / 'thresholds.json', {'status': 'complete', 'split': 'val', 'objective': objective,
                    'minimum_precision': minimum_precision, 'candidates': rows,
                    'selected_validation_candidate': max(eligible, key=lambda row:row[objective]) if eligible else None,
                    'elapsed_hours': (time.monotonic()-started)/3600, 'test_used_for_selection': False,
                    'scope': 'Fixed-threshold detection diagnostics; downstream safety and task-chain acceptance remain separate.'})
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for nms in ious:
            series = sorted((row for row in rows if row['nms_iou'] == nms), key=lambda row:row['confidence'])
            if series: ax.plot([row['confidence'] for row in series], [row['f1'] for row in series], marker='o', label='NMS IoU ' + str(nms))
        ax.set(xlabel='Confidence', ylabel='Validation F1', ylim=(0, 1)); ax.legend()
        fig.tight_layout(); fig.savefig(run_dir / 'thresholds.png', dpi=150); plt.close(fig)
    return run_dir


def tune_runs(root, config, run_name=None):
    """Explicit finite configuration grid; selection is based on validation only."""
    root = Path(root).resolve(); validate(config)
    tune = config.get('tune', {})
    if set(tune) - {'grid', 'max_trials', 'max_hours', 'metric'}:
        raise ValueError('Unknown tuning options')
    grid = tune.get('grid', {})
    if not grid or any(key not in TRAIN_KEYS | {'imgsz', 'batch', 'epochs'} for key in grid):
        raise ValueError('tune.grid requires explicitly supported training parameters')
    if any(not isinstance(values, list) or not values for values in grid.values()):
        raise ValueError('tune.grid entries must be nonempty lists')
    maximum, hours = tune.get('max_trials', 3), tune.get('max_hours', 2)
    if type(maximum) is not int or maximum < 1 or not math.isfinite(float(hours)) or float(hours) <= 0:
        raise ValueError('Positive max_trials and max_hours are required')
    metric = tune.get('metric', 'metrics/mAP50-95(P)' if config['task'] == 'pose' else 'metrics/mAP50-95(B)')
    started = time.monotonic(); deadline = started + float(hours)*3600
    run_dir = new_run(root, config, run_name)
    rows = []; state = 'running'
    with run_context(root, config, run_dir, 'tuning_controller'):
        try:
            for index, values in enumerate(itertools.product(*grid.values())):
                if index >= maximum: break
                remaining = (deadline - time.monotonic())/3600
                if remaining <= 0: raise BudgetExhausted('Tuning wall-clock budget exhausted')
                trial = deepcopy(config); trial.update(dict(zip(grid, values)))
                trial['output_root'] = (run_dir / 'trials').relative_to(root).as_posix()
                trial['time_hours'] = min(float(trial.get('time_hours') or remaining), remaining)
                trial['evaluate_after_training'] = True
                trial_name = f'trial_{index+1:02d}'
                trial_dir = run_dir / 'trials' / trial_name
                config_path = run_dir / 'config' / (trial_name + '.yaml')
                config_path.write_text(yaml.safe_dump(trial, allow_unicode=True, sort_keys=False), encoding='utf-8')
                arguments = child_command('train', root, config_path.relative_to(root).as_posix(), run_name=trial_name)
                try:
                    run_bounded(arguments, deadline, run_dir / 'logs' / (trial_name + '.log'), run_dir)
                except BudgetExhausted:
                    _mark_budget_exhausted(trial_dir, 'train')
                    rows.append({'trial': trial_dir.relative_to(root).as_posix(), 'parameters': dict(zip(grid, values)),
                                 'metric': metric, 'value': None, 'status': 'budget_exhausted'})
                    raise
                summary = json.loads((trial_dir / 'evaluation/val/summary.json').read_text(encoding='utf-8'))
                rows.append({'trial': trial_dir.relative_to(root).as_posix(), 'parameters': dict(zip(grid, values)),
                             'metric': metric, 'value': summary['metrics'].get(metric), 'status': 'complete'})
            state = 'complete'
        except BudgetExhausted:
            state = 'budget_exhausted'; raise
        except BaseException:
            state = 'failed'; raise
        finally:
            valid = [row for row in rows if row['value'] is not None]
            write_json(run_dir / 'tuning.json', {'status': state, 'trials': rows,
                       'best_validation': max(valid, key=lambda r:r['value']) if valid else None,
                       'wall_budget_hours': float(hours), 'elapsed_hours': (time.monotonic()-started)/3600,
                       'test_used_for_selection': False})
    return run_dir
