"""Bin state training and reviewed colour-by-state confusion diagnostics."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
PACKAGE_ROOT = HERE.parents[2]
PACKAGED = (PACKAGE_ROOT / 'scripts/common/engine').is_dir()
ROOT = PACKAGE_ROOT if PACKAGED else HERE.parents[5]
sys.path.insert(0, str(ROOT / 'scripts/common' if PACKAGED else ROOT / 'lab/training_common'))
from engine.cli import main as engine_main
from engine.config import resolve
from engine.bin_state_report import write_bin_state_report
from engine.reporting import box_iou

DEFAULT_CONFIG = 'config/train.yaml' if PACKAGED else 'lab/features/waste_bin/training/config/waste_bin_state_v4.yaml'
COLORS = ('blue', 'red', 'green', 'gray', 'black', 'unknown')
STATES = ('open', 'closed')


def reviewed_color_states(rows, reviews, threshold):
    """Count same-state matches and opposite-state overlaps for reviewed bins."""
    if not 0 < threshold <= 1: raise ValueError('Invalid IoU threshold')
    images = {}
    for row in rows:
        image = row['source_image']
        if image in images: raise ValueError('Duplicate evaluation image identity')
        images[image] = row
    groups = {color: {state: {'instances': 0, 'matched': 0, 'missed': 0, 'opposite_state': 0}
                      for state in STATES} for color in COLORS}
    paired_by_image = {}
    seen = set()
    for item in reviews:
        image, index, color = item['source_image'], item['gt_index'], item['color']
        if image not in images or type(index) is not int or color not in COLORS:
            raise ValueError('Unknown image, truth index or colour')
        if item.get('review_status') != 'confirmed' or not item.get('review_evidence'):
            raise ValueError('Colour requires confirmed review and evidence')
        if (image, index) in seen: raise ValueError('Duplicate bin review')
        row = images[image]; truths, predictions = row['ground_truth'], row['predictions']
        if index < 0 or index >= len(truths) or truths[index]['class_id'] not in (0, 1):
            raise ValueError('Bin truth index/class mismatch')
        if image not in paired_by_image:
            candidates = []
            for gi in row['missed_gt_indices']:
                for pi in row['false_prediction_indices']:
                    if truths[gi]['class_id'] not in (0, 1) or predictions[pi]['class_id'] not in (0, 1):
                        raise ValueError('Unexpected state class')
                    if truths[gi]['class_id'] != predictions[pi]['class_id']:
                        iou = box_iou(truths[gi]['xyxy'], predictions[pi]['xyxy'])
                        if iou >= threshold: candidates.append((iou, gi, pi))
            used_gt, used_prediction = set(), set()
            for _, gi, pi in sorted(candidates, reverse=True):
                if gi not in used_gt and pi not in used_prediction:
                    used_gt.add(gi); used_prediction.add(pi)
            paired_by_image[image] = used_gt
        group = groups[color][STATES[truths[index]['class_id']]]
        matched = {pair[0] for pair in row['matches']}
        group['instances'] += 1
        group['matched'] += index in matched
        group['missed'] += index in row['missed_gt_indices']
        group['opposite_state'] += index in paired_by_image[image]
        seen.add((image, index))
    if not seen: raise ValueError('No reviewed bin instances')
    for states in groups.values():
        for group in states.values():
            group['recall'] = group['matched'] / group['instances'] if group['instances'] else None
    return {'reviewed_instances': len(seen), 'match_iou': threshold, 'colors': groups,
            'scope': 'Only reviewed labelled state boxes; unknown physical state, entity association and unreviewed bins are not evaluated.'}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'bin-state-report':
        parser = argparse.ArgumentParser(description=write_bin_state_report.__doc__)
        parser.add_argument('--root', type=Path, default=ROOT)
        parser.add_argument('--evaluation', required=True)
        parser.add_argument('--dataset-manifest', required=True)
        args = parser.parse_args(argv[1:])
        try:
            root = args.root.resolve()
            evaluation = resolve(root, args.evaluation)
            dataset_manifest = resolve(root, args.dataset_manifest)
            path = write_bin_state_report(evaluation, dataset_manifest, root=root)
            print(path.name + ' ready')
            return 0
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            print(f'ERROR: {type(error).__name__}: {error}', file=sys.stderr)
            return 1
    if argv and argv[0] == 'color-state-report':
        parser = argparse.ArgumentParser(description=reviewed_color_states.__doc__)
        parser.add_argument('--root', type=Path, default=ROOT); parser.add_argument('--evaluation', required=True)
        parser.add_argument('--review', required=True, help='JSON list: source_image, gt_index, color, review_status, review_evidence')
        args = parser.parse_args(argv[1:])
        try:
            root = args.root.resolve(); evaluation = resolve(root, args.evaluation)
            if 'runs' not in evaluation.relative_to(root).parts: raise ValueError('Evaluation must be inside runs')
            summary = json.loads((evaluation / 'summary.json').read_text(encoding='utf-8'))
            if summary.get('model_key') != 'waste_bin_state': raise ValueError('Evaluation is not bin state')
            rows = [json.loads(line) for line in (evaluation / 'predictions/predictions.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
            reviews = json.loads(resolve(root, args.review).read_text(encoding='utf-8'))
            if not isinstance(reviews, list): raise ValueError('Review must be a JSON list')
            for review in reviews:
                if not resolve(root, review['review_evidence']).is_file(): raise ValueError('Review evidence missing')
            report = reviewed_color_states(rows, reviews, float(summary['diagnostics']['match_iou']))
            report['split'] = summary['split']; report['review_file'] = args.review
            with (evaluation / 'bin_color_state_report.json').open('x', encoding='utf-8') as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2); stream.write('\n')
            print('bin_color_state_report.json created'); return 0
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr); return 1
    return engine_main(default_root=ROOT, default_config=DEFAULT_CONFIG, argv=argv)


if __name__ == '__main__':
    raise SystemExit(main())
