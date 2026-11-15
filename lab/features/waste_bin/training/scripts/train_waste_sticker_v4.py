"""Waste-sticker training and per-class plus multi-sticker diagnostics."""
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

DEFAULT_CONFIG = 'config/train.yaml' if PACKAGED else 'lab/features/waste_bin/training/config/waste_sticker_v4.yaml'


def sticker_class_report(rows, names):
    """Report class-specific matched instances and complete multi-sticker images."""
    if not isinstance(names, list) or len(names) != 24 or len(set(names)) != 24 or not rows:
        raise ValueError('Expected 24 distinct class names and nonempty predictions')
    classes = [{'class_id': i, 'name': name, 'ground_truth': 0, 'matched': 0,
                'false_predictions': 0, 'recall': None} for i, name in enumerate(names)]
    multi = complete = complete_no_false = 0
    for row in rows:
        truths, predictions = row['ground_truth'], row['predictions']
        for item in truths + predictions:
            cls = item['class_id']
            if type(cls) is not int or cls not in range(24): raise ValueError('Invalid sticker class')
        for item in truths: classes[item['class_id']]['ground_truth'] += 1
        seen_gt, seen_prediction = set(), set()
        for gi, pi, _ in row['matches']:
            if gi in seen_gt or pi in seen_prediction or not 0 <= gi < len(truths) or not 0 <= pi < len(predictions):
                raise ValueError('Invalid or duplicate matching indices')
            if truths[gi]['class_id'] != predictions[pi]['class_id']: raise ValueError('Matched class mismatch')
            seen_gt.add(gi); seen_prediction.add(pi); classes[truths[gi]['class_id']]['matched'] += 1
        false = row['false_prediction_indices']
        if len(false) != len(set(false)) or set(false) != set(range(len(predictions))) - seen_prediction:
            raise ValueError('False prediction indices disagree with matches')
        for index in false: classes[predictions[index]['class_id']]['false_predictions'] += 1
        if len(truths) >= 2:
            multi += 1
            if len(seen_gt) == len(truths):
                complete += 1
                complete_no_false += not bool(false)
    for item in classes:
        if item['ground_truth']: item['recall'] = item['matched'] / item['ground_truth']
    observed = [item for item in classes if item['ground_truth']]
    return {'images': len(rows), 'classes': classes,
            'observed_class_macro_recall': sum(item['recall'] for item in observed) / len(observed) if observed else None,
            'absent_class_ids': [item['class_id'] for item in classes if not item['ground_truth']],
            'multi_sticker_images': multi, 'all_stickers_matched_images': complete,
            'all_stickers_matched_without_extra_predictions_images': complete_no_false,
            'scope': 'Fixed-threshold detector matches, not physical bin association or disposal correctness; absent classes are excluded from macro recall.'}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'class-report':
        parser = argparse.ArgumentParser(description=sticker_class_report.__doc__)
        parser.add_argument('--root', type=Path, default=ROOT); parser.add_argument('--evaluation', required=True)
        args = parser.parse_args(argv[1:])
        try:
            root = args.root.resolve(); evaluation = resolve(root, args.evaluation)
            if 'runs' not in evaluation.relative_to(root).parts: raise ValueError('Evaluation must be inside runs')
            summary = json.loads((evaluation / 'summary.json').read_text(encoding='utf-8'))
            if summary.get('model_key') != 'waste_sticker': raise ValueError('Evaluation is not waste sticker')
            rows = [json.loads(line) for line in (evaluation / 'predictions/predictions.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
            result = sticker_class_report(rows, summary['evaluation_config']['class_names'])
            result['split'] = summary['split']
            with (evaluation / 'waste_sticker_class_report.json').open('x', encoding='utf-8') as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2); stream.write('\n')
            print('waste_sticker_class_report.json created'); return 0
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr); return 1
    return engine_main(default_root=ROOT, default_config=DEFAULT_CONFIG, argv=argv)


if __name__ == '__main__':
    raise SystemExit(main())
