"""Electric-bike training and upright/fallen confusion analysis."""
import argparse
import json
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
from engine.reporting import box_iou

DEFAULT_CONFIG = 'lab/features/e_bike/training/config/e_bike_v4.yaml'
if PACKAGED:
    DEFAULT_CONFIG = 'config/train.yaml'
NAMES = ('upright', 'fallen')


def state_confusions(rows, threshold):
    """Pair only unmatched opposite-state boxes by descending IoU in each image."""
    if not 0 < threshold <= 1:
        raise ValueError('IoU threshold must be in (0, 1]')
    counts = {name: {'ground_truth': 0, 'matched_same_state': 0, 'missed': 0,
                     'opposite_state_prediction': 0} for name in NAMES}
    paired = 0
    for row in rows:
        truths, predictions = row['ground_truth'], row['predictions']
        for truth in truths:
            cls = truth['class_id']
            if type(cls) is not int or cls not in (0, 1):
                raise ValueError('Expected upright/fallen ground truth class IDs 0 and 1')
            counts[NAMES[cls]]['ground_truth'] += 1
        for prediction in predictions:
            if type(prediction['class_id']) is not int or prediction['class_id'] not in (0, 1):
                raise ValueError('Expected upright/fallen prediction class IDs 0 and 1')
        for gi, pi, _ in row['matches']:
            if truths[gi]['class_id'] != predictions[pi]['class_id']:
                raise ValueError('Input matches must already be same-class matches')
            counts[NAMES[truths[gi]['class_id']]]['matched_same_state'] += 1
        for gi in row['missed_gt_indices']:
            counts[NAMES[truths[gi]['class_id']]]['missed'] += 1
        candidates = []
        for gi in row['missed_gt_indices']:
            for pi in row['false_prediction_indices']:
                if truths[gi]['class_id'] == predictions[pi]['class_id']:
                    continue
                iou = box_iou(truths[gi]['xyxy'], predictions[pi]['xyxy'])
                if iou >= threshold:
                    candidates.append((iou, gi, pi))
        used_truth, used_prediction = set(), set()
        for _, gi, pi in sorted(candidates, reverse=True):
            if gi in used_truth or pi in used_prediction:
                continue
            used_truth.add(gi)
            used_prediction.add(pi)
            counts[NAMES[truths[gi]['class_id']]]['opposite_state_prediction'] += 1
            paired += 1
    return {'images': len(rows), 'match_iou': threshold, 'states': counts,
            'opposite_state_pairs': paired,
            'scope': 'Unmatched detection pairs only; does not assess physical entities or tracking.'}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'state-confusions':
        parser = argparse.ArgumentParser(description=state_confusions.__doc__)
        parser.add_argument('--root', type=Path, default=ROOT)
        parser.add_argument('--predictions', required=True, help='Root-relative engine predictions.jsonl')
        args = parser.parse_args(argv[1:])
        try:
            root = args.root.resolve()
            path = resolve(root, args.predictions)
            if 'runs' not in path.relative_to(root).parts or path.name != 'predictions.jsonl' or path.parent.name != 'predictions':
                raise ValueError('Input must be an engine predictions.jsonl inside runs')
            evaluation = path.parent.parent
            summary = json.loads((evaluation / 'summary.json').read_text(encoding='utf-8'))
            if summary.get('model_key') != 'e_bike':
                raise ValueError('Evaluation summary is not an e_bike model')
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
            if not rows:
                raise ValueError('Predictions are empty')
            result = state_confusions(rows, float(summary['diagnostics']['match_iou']))
            result['split'] = summary['split']
            destination = evaluation / 'e_bike_state_confusions.json'
            with destination.open('x', encoding='utf-8') as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write('\n')
            print(json.dumps({'status': 'complete', 'output': str(destination)}, ensure_ascii=False))
            return 0
        except (OSError, ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
            return 1
    return engine_main(default_root=ROOT, default_config=DEFAULT_CONFIG, argv=argv)


if __name__ == '__main__':
    raise SystemExit(main())
