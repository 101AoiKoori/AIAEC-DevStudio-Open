"""Project recorded YOLO training and evaluation metrics into TensorBoard events."""
import csv
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import re
import time


def _number(value):
    try:
        result = float(value)
        return result if not isinstance(value, bool) and math.isfinite(result) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _json(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _csv(path):
    try:
        content = path.read_text(encoding='utf-8-sig')
        # While the trainer is appending, consume only complete rows.
        if not content.endswith('\n'):
            content = content[:content.rfind('\n') + 1]
        return [{key.strip(): value for key, value in row.items() if key is not None}
                for row in csv.DictReader(io.StringIO(content)) if None not in row and None not in row.values()]
    except (OSError, ValueError, csv.Error):
        return None


def _metric_tag(key):
    match = re.fullmatch(r'metrics/([^()]+)\(([BP])\)', key)
    if match:
        metric, kind = match.groups()
        return ('Box/' if kind == 'B' else 'Pose/') + metric
    return key


def _started(run, phase, directory=None):
    receipt = _json(run / 'metadata' / f'{phase}_status.json')
    try:
        value = datetime.fromisoformat(receipt['started_at_utc'])
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc)
    except (KeyError, TypeError, ValueError):
        pass
    if directory is not None:
        for event in sorted(directory.glob('events.out.tfevents.*')):
            try:
                return datetime.fromtimestamp(int(event.name.split('.')[3]), timezone.utc)
            except (ValueError, OSError, OverflowError):
                continue
    return datetime.fromtimestamp((directory or run).stat().st_ctime, timezone.utc)


def _run_name(run):
    match = re.fullmatch(r'(v\d+)_([\da-f]{32})', run.name)
    if match:
        version, identifier = match.groups()
        return f'{version.upper()} {_started(run, "train"):%Y-%m-%d %H.%M.%S UTC} ({identifier[:8]})'
    return run.name


def _training_phases(run):
    phases = []
    for directory in (run / 'tensorboard').glob('*'):
        if not directory.is_dir() or directory.is_symlink():
            continue
        name = directory.name
        if name == 'train' or re.fullmatch(r'train_\d{8}T\d{6}_\d{6}Z', name):
            phase = 'train'
        elif name.startswith('resume_') and ((run / 'metadata' / f'{name}_status.json').is_file()
                or re.fullmatch(r'resume_(?:[\da-f]{32}|\d{8}T\d{6}_\d{6}Z)', name)):
            phase = name
        else:
            continue
        try:
            phases.append((phase != 'train', _started(run, phase, directory), directory))
        except OSError:
            continue  # A log may be replaced between discovery and reading.
    return sorted(phases)


def training_rows(run):
    rows = _csv(run / 'training/results.csv')
    if rows is None:
        return None
    config = _json(run / 'config/effective.json')
    policy = config.get('training_policy')
    fixed = isinstance(policy, dict) and policy.get('mode') == 'fixed'
    series = {}
    for row in rows:
        epoch = _number(row.get('epoch'))
        if epoch is None or epoch < 1 or not epoch.is_integer():
            continue
        values = {}
        for key, value in row.items():
            if key == 'time':
                tag = 'train/elapsed_seconds'
            elif key.startswith(('train/', 'lr/')) or (not fixed and key.startswith(('val/', 'metrics/'))):
                tag = _metric_tag(key)
            else:
                continue
            if (number := _number(value)) is not None:
                values[tag] = number
        if values:
            series[int(epoch)] = values
    return series


def evaluation_rows(directory):
    summary = _json(directory / 'summary.json')
    metrics = summary.get('metrics', {})
    if not isinstance(metrics, dict) or not metrics:
        return None
    values = {_metric_tag(key): value for key, value in metrics.items()}
    for field, group in [('speed_ms_per_image', 'speed'), ('diagnostics', 'diagnostics')]:
        fields = summary.get(field)
        if isinstance(fields, dict):
            values.update({f'{group}/{key}': value for key, value in fields.items()})
    for row in _csv(directory / 'per_class_metrics.csv') or ():
        if (index := row.get('class_id')) is not None:
            label = (row.get('name') or 'class').replace('/', '_')
            values.update({f'Class {index} {label}/{key}': value for key, value in row.items()
                           if key.startswith(('box_', 'pose_'))})
    for key in ('images', 'negative_images', 'annotated_instances'):
        values['dataset/' + key] = summary.get(key)
    return {0: {key: number for key, value in values.items() if (number := _number(value)) is not None}}


class Projection:
    """Present one canonical history and chronological recorded attempts."""
    def __init__(self, source, destination, *, single_run=False):
        self.source = Path(source)
        self.destination = Path(destination)
        self.single_run = single_run
        self.writers = {}
        self.recorded = {}
        self.attempt_readers = {}

    def _sync(self, name, series, times=None):
        from tensorboard.compat.proto import event_pb2, summary_pb2
        from tensorboard.plugins.scalar.summary_v2 import scalar_pb
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        if series is None:
            return
        previous = self.recorded.get(name, {})
        changes = sorted(step for step in previous.keys() | series.keys() if previous.get(step) != series.get(step))
        if not changes:
            return
        first = changes[0]
        writer = self.writers.get(name)
        if writer is None:
            writer = self.writers[name] = EventFileWriter(str(self.destination / name))
            writer.add_event(event_pb2.Event(step=first, session_log=event_pb2.SessionLog(status=event_pb2.SessionLog.START)))
        elif any(step >= first for step in previous):
            writer.add_event(event_pb2.Event(step=first, session_log=event_pb2.SessionLog(status=event_pb2.SessionLog.START)))
        for step, values in sorted(series.items()):
            if step >= first:
                scalars = (item for key, value in values.items() for item in scalar_pb(key, value).value)
                wall_time = times.get(step, time.time()) if times else time.time()
                writer.add_event(event_pb2.Event(step=step, wall_time=wall_time, summary=summary_pb2.Summary(value=scalars)))
        writer.flush()
        self.recorded[name] = series

    def _attempt_rows(self, directory, config):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, SCALARS, TENSORS
        from tensorboard.util.tensor_util import make_ndarray

        reader = self.attempt_readers.get(directory)
        if reader is None:
            reader = self.attempt_readers[directory] = EventAccumulator(str(directory), size_guidance={SCALARS: 0, TENSORS: 0})
        reader.Reload()
        losses = (('box_loss', 'pose_loss', 'kobj_loss', 'cls_loss', 'dfl_loss') if config.get('task') == 'pose'
                  else ('box_loss', 'cls_loss', 'dfl_loss'))
        policy = config.get('training_policy')
        fixed = isinstance(policy, dict) and policy.get('mode') == 'fixed'
        series, times = {}, {}
        for kind, tags in reader.Tags().items():
            if kind not in {SCALARS, TENSORS}:
                continue
            for tag in tags:
                if not tag.startswith(('train/', 'lr/')) and (fixed or not tag.startswith(('val/', 'metrics/'))):
                    continue
                alias = re.fullmatch(r'train/loss_(\d+)', tag)
                index = int(alias[1]) if alias else len(losses)
                output = 'train/' + losses[index] if index < len(losses) else _metric_tag(tag)
                events = reader.Scalars(tag) if kind == SCALARS else reader.Tensors(tag)
                for event in events:
                    if kind == SCALARS:
                        raw = event.value
                    else:
                        array = make_ndarray(event.tensor_proto)
                        if array.ndim != 0:
                            continue
                        raw = array.item()
                    if (number := _number(raw)) is not None:
                        series.setdefault(event.step, {})[output] = number
                        times[event.step] = event.wall_time
        return series, times

    def refresh(self):
        runs = [self.source] if self.single_run else sorted(self.source.glob('*'))
        for run in runs:
            if not run.is_dir() or run.name.startswith('_') or run.is_symlink():
                continue
            config = _json(run / 'config/effective.json')
            model = config.get('model_key')
            if not isinstance(model, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', model):
                model = self.source.parent.name if self.single_run else self.source.name
            title = f'{model}/{_run_name(run)}'
            self._sync(f'{title}/00 Overall', training_rows(run))
            for index, (resumed, started, directory) in enumerate(_training_phases(run), start=1):
                kind = 'Resume' if resumed else 'Train'
                try:
                    series, times = self._attempt_rows(directory, config)
                except OSError:
                    continue  # Retry on the next refresh if the writer temporarily holds the file.
                self._sync(f'{title}/{index:02d} {kind} {started:%Y-%m-%d %H.%M.%S UTC}', series, times)
            for evaluation in sorted(run.glob('evaluation/*')):
                if evaluation.is_dir() and evaluation.resolve().is_relative_to(run.resolve()):
                    self._sync(f'{title}/Evaluation/{evaluation.name}', evaluation_rows(evaluation))

    def close(self):
        for writer in self.writers.values():
            writer.close()
