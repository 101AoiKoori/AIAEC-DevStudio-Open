"""Bounded training, explicit evaluation and validation-only experiment helpers."""
from pathlib import Path
from copy import deepcopy
import itertools
import json
import math
import shutil
import time
import yaml
from .config import TRAIN_KEYS, load_config, resolve, validate
from .data import inspect_dataset, stage_dataset, relocate_staged_data
from .runtime import environment, new_run, prepare_ultralytics, run_context, write_json
from .reporting import diagnostics, metric_report, tensorboard_report, write_csv, match_detections


def preflight(root, config):
    root = Path(root).resolve(); validate(config)
    model = resolve(root, config['model'])
    if model.suffix.lower() != '.pt' or not model.is_file():
        raise ValueError('Local pretrained/checkpoint .pt missing: ' + str(model))
    header, records, report = inspect_dataset(root, config)
    return {**report, 'class_names': header['names'], 'task': config['task'],
            'model': config['model'], 'environment': environment(root)}


def _training_options(config, data_path, run_dir):
    options = {'data': str(data_path), 'imgsz': config['imgsz'], 'epochs': config['epochs'],
               'batch': config['batch'], 'workers': config.get('workers', 0),
               'patience': config.get('patience', 30), 'seed': config.get('seed', 42),
               'device': config.get('device', 'cpu'), 'project': str(run_dir), 'name': 'training',
               'exist_ok': False, 'plots': True, 'save': True, 'cache': False,
               'amp': config.get('amp', False), 'max_det': config.get('max_det', 300),
               'val': True, 'verbose': True}
    options.update({key: config[key] for key in TRAIN_KEYS if key in config})
    options.update(config.get('augment', {}))
    if config['task'] == 'pose': options.update(fliplr=0.0, flipud=0.0)
    if config.get('time_hours'): options['time'] = config['time_hours']
    return options


def _attach_tensorboard(model, run_dir):
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(Path(run_dir) / 'tensorboard/train'))
    def epoch_metrics(trainer):
        epoch = trainer.epoch + 1
        metrics = dict(getattr(trainer, 'metrics', {}) or {})
        metrics.update(getattr(trainer, 'lr', {}) or {})
        loss = getattr(trainer, 'tloss', None)
        if loss is not None:
            for index, value in enumerate(loss.detach().cpu().flatten().tolist()):
                metrics['train/loss_' + str(index)] = value
        for key, value in metrics.items():
            try: value = float(value)
            except (TypeError, ValueError): continue
            if math.isfinite(value): writer.add_scalar(key, value, epoch)
        writer.flush()
    model.add_callback('on_fit_epoch_end', epoch_metrics)
    return writer


def _evaluate_loaded(model, config, run_dir, data_path, split, name):
    if Path(name).name != name or name in {'.', '..'}: raise ValueError('Invalid evaluation name')
    destination = Path(run_dir) / 'evaluation' / name
    if destination.exists(): raise FileExistsError('Evaluation already exists: ' + str(destination))
    options = {'data': str(data_path), 'split': split, 'imgsz': config['imgsz'],
               'batch': config['batch'], 'workers': config.get('workers', 0),
               'device': config.get('device', 'cpu'), 'conf': .001, 'max_det': config.get('max_det', 300),
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
    summary['data_selection'] = json.loads((Path(run_dir) / 'data/selection.json').read_text(encoding='utf-8'))
    selected_records = json.loads((Path(run_dir) / 'data/samples.json').read_text(encoding='utf-8'))
    summary['evaluation_sources'] = sorted(row.get('workspace_image', row['image']) for row in selected_records if row['split'] == split)
    summary['evaluation_config'] = {'imgsz': config['imgsz'], 'class_names': config.get('class_names'),
                                    'diagnostic_conf': config.get('diagnostic_conf', .25), 'match_iou': config.get('match_iou', .5)}
    write_json(actual / 'summary.json', summary)
    tensorboard_report(run_dir, name, summary, class_rows, diagnostic)
    return summary


def train_run(root, config, run_name=None, resume=None):
    root = Path(root).resolve(); config = validate(deepcopy(config))
    model_path = resolve(root, resume or config['model'])
    if not model_path.is_file() or model_path.suffix.lower() != '.pt':
        raise ValueError('Local model/checkpoint missing: ' + str(model_path))
    if resume:
        run_dir = model_path.parents[2]
        if model_path.parent.name != 'weights' or model_path.parent.parent.name != 'training':
            raise ValueError('Resume needs this engine run/training/weights/last.pt')
        previous = json.loads((run_dir / 'config/effective.json').read_text(encoding='utf-8'))
        allowed = {'batch', 'workers', 'device', 'output_root'}
        if {k:v for k,v in previous.items() if k not in allowed} != {k:v for k,v in config.items() if k not in allowed}:
            raise ValueError('Resume config differs from recorded run; only batch/workers/device/output_root may change')
    else:
        run_dir = new_run(root, config, run_name)
    print('RUN_DIR:', run_dir, flush=True)
    with run_context(root, config, run_dir, 'resume' if resume else 'train'):
        if resume:
            data_path = relocate_staged_data(run_dir)
        else:
            data_path, selection = stage_dataset(root, config, run_dir)
            print('Data selection:', selection, flush=True)
        config_yaml = run_dir / 'config/effective.yaml'
        config_yaml.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
        if config.get('amp', False):
            if not config.get('amp_check_model'):
                raise ValueError('amp=True requires a local amp_check_model for offline AMP checks')
            shutil.copy2(resolve(root, config['amp_check_model']), run_dir / 'yolov8n.pt')
        YOLO = prepare_ultralytics(run_dir)
        import torch
        write_json(run_dir / 'metadata/hardware.json', {'cuda_available': torch.cuda.is_available(),
                    'torch_cuda': torch.version.cuda,
                    'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
        model = YOLO(str(model_path), task=config['task'])
        writer = _attach_tensorboard(model, run_dir)
        options = _training_options(config, data_path, run_dir)
        write_json(run_dir / 'config/framework_options.json', options)
        try:
            if resume:
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
                model.train(**options)
        finally:
            writer.close()
        best = run_dir / 'training/weights/best.pt'
        if not best.is_file(): raise RuntimeError('Training completed without best.pt')
        write_json(run_dir / 'metadata/artifacts.json', {'best': 'training/weights/best.pt',
                    'last': 'training/weights/last.pt', 'training_metrics': 'training/results.csv',
                    'tensorboard': 'tensorboard', 'dataset': 'data/data.yaml'})
        if config.get('evaluate_after_training', True):
            name = 'val' if not resume else 'val_resume_' + str(int(time.time()))
            _evaluate_loaded(YOLO(str(best), task=config['task']), config, run_dir, data_path, 'val', name)
    return run_dir


def evaluate_run(root, config, weights, split='val', run_dir=None, name=None):
    if split not in ('val', 'test'): raise ValueError('Evaluation split must be val or test')
    root = Path(root).resolve(); config = validate(deepcopy(config))
    weight_path = resolve(root, weights, allow_absolute=True)
    if not weight_path.is_file(): raise ValueError('Evaluation checkpoint missing: ' + str(weight_path))
    if run_dir is None:
        run_dir = new_run(root, config)
    else:
        run_dir = resolve(root, run_dir, allow_absolute=True)
        if 'runs' not in run_dir.relative_to(root).parts:
            raise ValueError('Evaluation run must be inside runs')
    name = name or split
    with run_context(root, config, run_dir, 'evaluate_' + name):
        data_path = run_dir / 'data/data.yaml'
        if not data_path.exists(): data_path, _ = stage_dataset(root, config, run_dir)
        else: data_path = relocate_staged_data(run_dir)
        YOLO = prepare_ultralytics(run_dir)
        result = _evaluate_loaded(YOLO(str(weight_path), task=config['task']), config, run_dir, data_path, split, name)
        result['checkpoint'] = weight_path.relative_to(root).as_posix()
        write_json(run_dir / 'evaluation' / name / 'summary.json', result)
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
        key = next((key for key in fields if key.startswith('metrics/mAP50-95')), None)
        if key and len(comparable) == 1 and all(group['comparable'] for group in comparable.values()):
            fig, ax = plt.subplots(figsize=(max(7, len(rows)*1.1), 5))
            available = [(row['version'] + '/' + row['evaluation'], row.get(key)) for row in rows if row.get(key) is not None]
            ax.bar([r[0] for r in available], [r[1] for r in available]); ax.set_ylim(0, 1)
            ax.set_ylabel(key); ax.tick_params(axis='x', rotation=30)
            fig.tight_layout(); fig.savefig(run_dir / 'comparison.png', dpi=150); plt.close(fig)
    return run_dir


def tune_thresholds(root, config, weights, run_name=None):
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
    run_dir = new_run(root, config, run_name)
    with run_context(root, config, run_dir, 'threshold_tuning'):
        data_path, _ = stage_dataset(root, config, run_dir)
        YOLO = prepare_ultralytics(run_dir); model = YOLO(str(checkpoint), task=config['task'])
        started = time.monotonic(); rows = []
        for index, nms in enumerate(ious):
            if len(rows) >= maximum or (time.monotonic()-started)/3600 >= hours: break
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
        eligible = [row for row in rows if row['precision'] >= minimum_precision]
        if not rows: raise RuntimeError('No threshold candidates evaluated')
        write_csv(run_dir / 'thresholds.csv', rows, list(rows[0]))
        write_json(run_dir / 'thresholds.json', {'split': 'val', 'objective': objective,
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
    root = Path(root).resolve(); run_dir = new_run(root, config, run_name)
    tune = config.get('tune', {})
    if set(tune) - {'grid', 'max_trials', 'max_hours', 'metric'}:
        raise ValueError('Unknown tuning options')
    grid = tune.get('grid', {})
    if not grid or any(key not in TRAIN_KEYS | {'imgsz', 'batch', 'epochs'} for key in grid):
        raise ValueError('tune.grid requires explicitly supported training parameters')
    if any(not isinstance(values, list) or not values for values in grid.values()):
        raise ValueError('tune.grid entries must be nonempty lists')
    maximum, hours = tune.get('max_trials', 3), tune.get('max_hours', 2)
    if type(maximum) is not int or maximum < 1 or float(hours) <= 0:
        raise ValueError('Positive max_trials and max_hours are required')
    metric = tune.get('metric', 'metrics/mAP50-95(P)' if config['task'] == 'pose' else 'metrics/mAP50-95(B)')
    started = time.monotonic(); rows = []
    for index, values in enumerate(itertools.product(*grid.values())):
        remaining = hours - (time.monotonic()-started)/3600
        if index >= maximum or remaining <= 0: break
        trial = deepcopy(config); trial.update(dict(zip(grid, values)))
        trial['output_root'] = (run_dir / 'trials').relative_to(root).as_posix()
        trial['time_hours'] = min(float(trial.get('time_hours') or remaining), remaining)
        trial['evaluate_after_training'] = True
        trial_dir = train_run(root, trial, f'trial_{index+1:02d}')
        summary = json.loads((trial_dir / 'evaluation/val/summary.json').read_text(encoding='utf-8'))
        rows.append({'trial': trial_dir.relative_to(root).as_posix(), 'parameters': dict(zip(grid, values)),
                     'metric': metric, 'value': summary['metrics'].get(metric)})
    valid = [row for row in rows if row['value'] is not None]
    write_json(run_dir / 'tuning.json', {'trials': rows, 'best_validation': max(valid, key=lambda r:r['value']) if valid else None,
               'elapsed_hours': (time.monotonic()-started)/3600, 'test_used_for_selection': False})
    return run_dir
