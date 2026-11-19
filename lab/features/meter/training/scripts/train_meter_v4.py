"""Meter pose training and named keypoint error analysis."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
PACKAGE_ROOT = HERE.parents[2]
PACKAGED = (PACKAGE_ROOT / 'scripts/common/engine').is_dir()
ROOT = PACKAGE_ROOT if PACKAGED else HERE.parents[5]
COMMON = ROOT / 'scripts/common' if PACKAGED else ROOT / 'lab/training_common'
sys.path.insert(0, str(COMMON))
from engine.cli import main as engine_main
from engine.config import resolve
from engine.meter_report import write_meter_report

DEFAULT_CONFIG = 'lab/features/meter/training/config/meter_v4.yaml'
if PACKAGED:
    DEFAULT_CONFIG = 'config/train.yaml'
POINTS = ('temperature_pivot', 'temperature_tip', 'humidity_pivot', 'humidity_tip',
          'temperature_min', 'temperature_max', 'humidity_min', 'humidity_max')


def keypoint_errors(rows):
    """Report measured errors for labeled points on matched detections only."""
    values = [[] for _ in POINTS]
    visibility = [{1: 0, 2: 0} for _ in POINTS]
    for row in rows:
        index = int(row['keypoint_index'])
        state = int(float(row['visibility']))
        error = float(row['error_pixels'])
        if not 0 <= index < len(POINTS) or state not in (1, 2) or not math.isfinite(error) or error < 0:
            raise ValueError('Invalid keypoint index, visibility, or error')
        values[index].append(error)
        visibility[index][state] += 1
    return [{'index': index, 'name': name, 'matched_points': len(values[index]),
             'visibility_1': visibility[index][1], 'visibility_2': visibility[index][2],
             'mean_error_pixels': sum(values[index]) / len(values[index]) if values[index] else None,
             'max_error_pixels': max(values[index]) if values[index] else None}
            for index, name in enumerate(POINTS)]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'meter-report':
        parser = argparse.ArgumentParser(description=write_meter_report.__doc__)
        parser.add_argument('--root', type=Path, default=ROOT)
        parser.add_argument('--evaluation', required=True)
        args = parser.parse_args(argv[1:])
        try:
            root = args.root.resolve()
            evaluation = resolve(root, args.evaluation)
            if 'runs' not in evaluation.relative_to(root).parts:
                raise ValueError('Evaluation must be inside runs')
            print(write_meter_report(evaluation).name + ' ready')
            return 0
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            print(f'ERROR: {type(error).__name__}: {error}', file=sys.stderr)
            return 1
    if argv and argv[0] == 'keypoint-report':
        parser = argparse.ArgumentParser(description=keypoint_errors.__doc__)
        parser.add_argument('--root', type=Path, default=ROOT)
        parser.add_argument('--evaluation', required=True, help='Root-relative meter evaluation directory inside runs')
        args = parser.parse_args(argv[1:])
        try:
            root = args.root.resolve()
            evaluation = resolve(root, args.evaluation)
            if 'runs' not in evaluation.relative_to(root).parts:
                raise ValueError('Evaluation must be inside runs')
            summary = json.loads((evaluation / 'summary.json').read_text(encoding='utf-8'))
            if summary.get('model_key') != 'meter' or summary.get('task') != 'pose':
                raise ValueError('Evaluation is not meter pose')
            with (evaluation / 'keypoint_errors.csv').open(encoding='utf-8', newline='') as stream:
                points = keypoint_errors(csv.DictReader(stream))
            result = {'split': summary['split'], 'points': points,
                      'detection_false_negatives': summary['diagnostics']['false_negatives'],
                      'scope': ('Truth visibility 1 or 2 on matched boxes and present predicted point entries only; '
                                'missing predictions are skipped and no predicted-point confidence filter is applied. '
                                'Missed meters and unmeasured points are not zero-error points.')}
            destination = evaluation / 'meter_named_keypoint_report.json'
            with destination.open('x', encoding='utf-8') as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write('\n')
            print(json.dumps({'status': 'complete', 'output': str(destination)}, ensure_ascii=False))
            return 0
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
            return 1
    return engine_main(default_root=ROOT, default_config=DEFAULT_CONFIG, argv=argv)


if __name__ == '__main__':
    raise SystemExit(main())
