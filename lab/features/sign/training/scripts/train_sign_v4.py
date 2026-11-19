"""Sign-fixture training and reviewed view-stratified detection diagnostics."""
import argparse
import json
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
from sign_report import write_sign_report

DEFAULT_CONFIG = 'config/train.yaml' if PACKAGED else 'lab/features/sign/training/config/sign_v4.yaml'
VIEWS = ('front', 'oblique', 'side', 'back', 'unknown')


def reviewed_view_report(rows, reviews):
    """Stratify fixture boxes using confirmed machine or human view evidence."""
    images = {}
    for row in rows:
        if row['source_image'] in images:
            raise ValueError('Duplicate evaluation image identity')
        images[row['source_image']] = row
    groups = {view: {'instances': 0, 'matched': 0, 'recall': None} for view in VIEWS}
    seen = set()
    for item in reviews:
        image, index, view = item['source_image'], item['gt_index'], item['view']
        if image not in images or type(index) is not int or view not in VIEWS:
            raise ValueError('Unknown image, truth index or view')
        if item.get('review_status') != 'confirmed' or not item.get('review_evidence'):
            raise ValueError('View needs confirmed review and evidence')
        if (image, index) in seen:
            raise ValueError('Duplicate fixture review')
        row = images[image]
        if index < 0 or index >= len(row['ground_truth']) or row['ground_truth'][index]['class_id'] != 0:
            raise ValueError('Fixture truth index/class mismatch')
        seen.add((image, index))
        matched = {pair[0] for pair in row['matches']}
        groups[view]['instances'] += 1
        groups[view]['matched'] += index in matched
    if not seen:
        raise ValueError('No reviewed fixture instances')
    for group in groups.values():
        if group['instances']:
            group['recall'] = group['matched'] / group['instances']
    return {'reviewed_instances': len(seen), 'views': groups,
            'scope': 'Only reviewed ground-truth fixtures; whole-fixture coverage, position and stop/route gate require separate checks.'}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'sign-report':
        parser=argparse.ArgumentParser(description=write_sign_report.__doc__)
        parser.add_argument('--root',type=Path,default=ROOT)
        parser.add_argument('--evaluation',required=True)
        parser.add_argument('--dataset-manifest',required=True)
        args=parser.parse_args(argv[1:])
        try:
            root=args.root.resolve()
            path=write_sign_report(resolve(root,args.evaluation),resolve(root,args.dataset_manifest),root=root)
            print(path.name+' ready');return 0
        except (OSError,ValueError,KeyError,IndexError,TypeError) as error:
            print(f'ERROR: {type(error).__name__}: {error}',file=sys.stderr);return 1
    if argv and argv[0] == 'view-report':
        parser = argparse.ArgumentParser(description=reviewed_view_report.__doc__)
        parser.add_argument('--root', type=Path, default=ROOT)
        parser.add_argument('--evaluation', required=True)
        parser.add_argument('--review', required=True, help='JSON list: source_image, gt_index, view, review_status, review_evidence')
        args = parser.parse_args(argv[1:])
        try:
            root = args.root.resolve(); evaluation = resolve(root, args.evaluation)
            if 'runs' not in evaluation.relative_to(root).parts:
                raise ValueError('Evaluation must be inside runs')
            summary = json.loads((evaluation / 'summary.json').read_text(encoding='utf-8'))
            if summary.get('model_key') != 'sign':
                raise ValueError('Evaluation is not the sign model')
            rows = [json.loads(line) for line in (evaluation / 'predictions/predictions.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
            reviews = json.loads(resolve(root, args.review).read_text(encoding='utf-8'))
            if not isinstance(reviews, list): raise ValueError('Review must be a JSON list')
            for review in reviews:
                if not resolve(root, review['review_evidence']).is_file(): raise ValueError('Review evidence missing')
            report = reviewed_view_report(rows, reviews); report['split'] = summary['split']; report['review_file'] = args.review
            with (evaluation / 'sign_view_report.json').open('x', encoding='utf-8') as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2); stream.write('\n')
            print('sign_view_report.json created'); return 0
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr); return 1
    return engine_main(default_root=ROOT, default_config=DEFAULT_CONFIG, argv=argv)


if __name__ == '__main__':
    raise SystemExit(main())
