"""Name-paper training and size-stratified detector diagnostics."""
import argparse
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
PACKAGE_ROOT = HERE.parents[2]
PACKAGED = (PACKAGE_ROOT / 'scripts/common/engine').is_dir()
ROOT = PACKAGE_ROOT if PACKAGED else HERE.parents[5]
sys.path.insert(0, str(ROOT / 'scripts/common' if PACKAGED else ROOT / 'lab/training_common'))
sys.path.insert(0, str(HERE.parent))
from engine.cli import main as engine_main
from engine.config import resolve
from paper_report import write_paper_report

DEFAULT_CONFIG = 'config/train.yaml' if PACKAGED else 'lab/features/fire/training/config/building_name_paper_v4.yaml'


def paper_size_report(rows, imgsz):
    """Count box recall by estimated unaugmented model-input short-side pixels."""
    if type(imgsz) is not int or imgsz <= 0 or not rows:
        raise ValueError('A positive model input size and nonempty evaluation are required')
    buckets = {key: {'ground_truth': 0, 'matched': 0, 'recall': None} for key in ('0_8', '8_16', '16_32', 'over_32')}
    negative, negative_fp = 0, 0
    for row in rows:
        height, width = row['shape_hw']
        if height <= 0 or width <= 0:
            raise ValueError('Invalid image size')
        truths = row['ground_truth']
        matched = {int(pair[0]) for pair in row['matches']}
        if any(index < 0 or index >= len(truths) for index in matched):
            raise ValueError('Matched truth index outside image')
        if not truths:
            negative += 1
            negative_fp += bool(row['false_prediction_indices'])
        ratio = imgsz / max(width, height)
        resized_w, resized_h = min(math.ceil(width * ratio), imgsz), min(math.ceil(height * ratio), imgsz)
        for index, truth in enumerate(truths):
            if truth['class_id'] != 0:
                raise ValueError('Name-paper task requires class ID 0')
            x1, y1, x2, y2 = truth['xyxy']
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError('Invalid name-paper box')
            size = min((x2 - x1) * resized_w / width, (y2 - y1) * resized_h / height)
            key = '0_8' if size <= 8 else '8_16' if size <= 16 else '16_32' if size <= 32 else 'over_32'
            buckets[key]['ground_truth'] += 1
            buckets[key]['matched'] += index in matched
    for value in buckets.values():
        if value['ground_truth']:
            value['recall'] = value['matched'] / value['ground_truth']
    return {'images': len(rows), 'input_size': imgsz, 'short_side_pixels': buckets,
            'empty_label_images': negative, 'empty_label_images_with_predictions': negative_fp,
            'scope': 'Input sizes are unaugmented resize estimates, not sensor detail; empty labels require verified negatives. OCR is not evaluated.'}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0]=='paper-report':
        parser=argparse.ArgumentParser(description=write_paper_report.__doc__)
        parser.add_argument('--root',type=Path,default=ROOT)
        parser.add_argument('--evaluation',required=True)
        parser.add_argument('--dataset-manifest',required=True)
        args=parser.parse_args(argv[1:])
        try:
            root=args.root.resolve()
            path=write_paper_report(resolve(root,args.evaluation),resolve(root,args.dataset_manifest),root=root)
            print(path.name+' ready');return 0
        except (OSError,ValueError,KeyError,IndexError,TypeError) as error:
            print(f'ERROR: {type(error).__name__}: {error}',file=sys.stderr);return 1
    if argv and argv[0] == 'paper-sizes':
        parser = argparse.ArgumentParser(description=paper_size_report.__doc__)
        parser.add_argument('--root', type=Path, default=ROOT)
        parser.add_argument('--evaluation', required=True, help='Root-relative evaluation directory inside runs')
        args = parser.parse_args(argv[1:])
        try:
            root = args.root.resolve()
            evaluation = resolve(root, args.evaluation)
            if 'runs' not in evaluation.relative_to(root).parts:
                raise ValueError('Evaluation must be inside runs')
            summary = json.loads((evaluation / 'summary.json').read_text(encoding='utf-8'))
            if summary.get('model_key') != 'building_name_paper':
                raise ValueError('Evaluation is not the building name-paper model')
            rows = [json.loads(line) for line in (evaluation / 'predictions/predictions.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
            report = paper_size_report(rows, summary['evaluation_config']['imgsz'])
            report['split'] = summary['split']
            with (evaluation / 'name_paper_size_report.json').open('x', encoding='utf-8') as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2); stream.write('\n')
            print('name_paper_size_report.json created'); return 0
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr); return 1
    return engine_main(default_root=ROOT, default_config=DEFAULT_CONFIG, argv=argv)


if __name__ == '__main__':
    raise SystemExit(main())
