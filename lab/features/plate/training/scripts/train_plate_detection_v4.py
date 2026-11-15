"""Plate-detector training and same-development-data initialization comparison."""
import argparse
import filecmp
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

DEFAULT_CONFIG = 'lab/features/plate/training/config/plate_detection_v4.yaml'
if PACKAGED:
    DEFAULT_CONFIG = 'config/train.yaml'


def development_records(evaluation):
    """Load evaluated split and its staged image/label pairs."""
    evaluation = Path(evaluation).resolve()
    if evaluation.parent.name != 'evaluation':
        raise ValueError('Expected run/evaluation/name')
    run = evaluation.parent.parent
    summary = json.loads((evaluation / 'summary.json').read_text(encoding='utf-8'))
    if summary.get('model_key') != 'plate_detection' or summary.get('split') != 'val':
        raise ValueError('Initialization comparison requires plate validation, not test')
    if not summary.get('checkpoint'):
        raise ValueError('Evaluation must record its actual checkpoint')
    samples = json.loads((run / 'data/samples.json').read_text(encoding='utf-8'))
    selected = {}
    for sample in samples:
        if sample['split'] != 'val':
            continue
        key = sample['image']
        if key in selected:
            raise ValueError('Duplicate source image in evaluation dataset')
        image = resolve(run, sample['run_image'])
        label = resolve(run, sample['run_label'])
        if not image.is_file() or not label.is_file():
            raise ValueError('Evaluation input image or label missing')
        selected[key] = (image, label)
    if not selected:
        raise ValueError('Validation inputs are empty')
    environment_path = run / 'metadata' / ('environment_evaluate_' + evaluation.name + '.json')
    environment = json.loads(environment_path.read_text(encoding='utf-8'))
    versions = {name: environment['packages'].get(name) for name in
                ('torch', 'torchvision', 'ultralytics', 'numpy', 'opencv-python')}
    if any(value is None for value in versions.values()):
        raise ValueError('Evaluation environment versions are incomplete')
    return summary, selected, {'python': environment['python'], 'packages': versions}


def compare_initializations(left_evaluation, right_evaluation):
    """Compare records and configuration without selecting the winning checkpoint."""
    left, left_data, left_environment = development_records(left_evaluation)
    right, right_data, right_environment = development_records(right_evaluation)
    if set(left_data) != set(right_data):
        raise ValueError('Validation image identities differ')
    for key in left_data:
        if any(not filecmp.cmp(a, b, shallow=False) for a, b in zip(left_data[key], right_data[key])):
            raise ValueError('Validation image or label bytes differ: ' + key)
    for field in ('evaluation_config',):
        if left.get(field) != right.get(field):
            raise ValueError('Evaluation configuration differs: ' + field)
    if left['diagnostics'].get('nms_iou') != right['diagnostics'].get('nms_iou'):
        raise ValueError('NMS threshold differs')
    if left_environment != right_environment:
        raise ValueError('Evaluation Python or dependency versions differ')
    if left['checkpoint'] == right['checkpoint']:
        raise ValueError('Initialization checkpoints are the same path')
    output = []
    for item in (left, right):
        d = item['diagnostics']
        output.append({'checkpoint': item['checkpoint'], 'images': item['images'],
                       'annotated_instances': item['annotated_instances'], 'metrics': item['metrics'],
                       'true_positives': d['true_positives'], 'false_positives': d['false_positives'],
                       'false_negatives': d['false_negatives'], 'negative_images': d['negative_images'],
                       'negative_images_with_false_positives': d['negative_images_with_false_positives']})
    return {'split': 'val', 'compared_images': len(left_data), 'environment': left_environment, 'candidates': output,
            'selected_checkpoint': None,
            'scope': 'Same staged images and labels by byte comparison; grouped small-plate and scene review still required.'}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'compare-initializations':
        parser = argparse.ArgumentParser(description=compare_initializations.__doc__)
        parser.add_argument('--root', type=Path, default=ROOT)
        parser.add_argument('--left-evaluation', required=True)
        parser.add_argument('--right-evaluation', required=True)
        parser.add_argument('--output', required=True, help='New JSON path inside runs')
        args = parser.parse_args(argv[1:])
        try:
            root = args.root.resolve()
            left, right = resolve(root, args.left_evaluation), resolve(root, args.right_evaluation)
            output = resolve(root, args.output)
            if any('runs' not in item.relative_to(root).parts for item in (left, right, output)):
                raise ValueError('Inputs and output must be inside runs')
            if output.suffix.lower() != '.json' or not output.parent.is_dir():
                raise ValueError('Output must be a new JSON file in an existing runs directory')
            result = compare_initializations(left, right)
            with output.open('x', encoding='utf-8') as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write('\n')
            print(json.dumps({'status': 'complete', 'output': str(output)}, ensure_ascii=False))
            return 0
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
            return 1
    return engine_main(default_root=ROOT, default_config=DEFAULT_CONFIG, argv=argv)


if __name__ == '__main__':
    raise SystemExit(main())
