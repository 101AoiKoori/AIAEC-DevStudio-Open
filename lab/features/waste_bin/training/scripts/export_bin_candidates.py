"""Export audited, fully labelled bin-lid images as unsplit YOLO candidates."""
import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bin_render_contract import STATE_CONTRACT


ROOT = Path.cwd().resolve()
BINS = {'blue', 'red', 'green', 'gray'}


def contained_file(root, relative):
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError('missing or invalid source file: ' + str(relative))
    return path


def plan_samples(samples):
    """Only accept four explicitly eligible lids and an existing candidate label."""
    included, excluded = [], []
    for sample in samples:
        labels = sample.get('labels', [])
        reasons = []
        if len(labels) != 4 or {x.get('bin') for x in labels} != BINS:
            reasons.append('missing_or_invalid_lid_records')
        for item in labels:
            if item.get('label_eligible') is not True or item.get('in_frame') is not True:
                details = item.get('rejection_reasons') or ['eligibility_unknown']
                reasons.extend(f"{item.get('bin', 'unknown')}:{reason}" for reason in details)
        if not sample.get('candidate_label'):
            reasons.append('no_candidate_label')
        if reasons:
            excluded.append({'source_image': sample['image'],
                             'source_candidate_label': sample.get('candidate_label'),
                             'group': sample.get('group'), 'trajectory': sample.get('trajectory'),
                             'scenario': sample.get('scenario'), 'reasons': reasons})
        else:
            included.append(sample)
    return included, excluded


def validate_pair(source, sample):
    image = contained_file(source, sample['image'])
    label = contained_file(source, sample['candidate_label'])
    if image.suffix.lower() != '.png' or label.suffix.lower() != '.txt' or image.stem != label.stem:
        raise ValueError('image and label must have matching YOLO stems and PNG/TXT extensions')
    rows = label.read_text(encoding='utf-8').splitlines()
    if len(rows) != 4:
        raise ValueError('each exported image must have four labels')
    for row in rows:
        fields = row.split()
        if len(fields) != 5 or fields[0] not in ('0', '1'):
            raise ValueError('invalid YOLO label')
        x, y, w, h = (float(value) for value in fields[1:])
        if not all(math.isfinite(value) for value in (x, y, w, h)) or not (
            w > 0 and h > 0 and x-w/2 >= -1e-6 and x+w/2 <= 1+1e-6
            and y-h/2 >= -1e-6 and y+h/2 <= 1+1e-6
        ):
            raise ValueError('invalid YOLO coordinates')
    return image, label


def run_source_audit(source):
    script = Path(__file__).resolve().with_name('audit_bin_renders.py')
    result = subprocess.run([sys.executable, str(script), str(source)],
                            capture_output=True, text=True, encoding='utf-8', errors='replace')
    if result.returncode:
        raise RuntimeError('source audit failed: ' + (result.stdout + result.stderr).strip())
    audit = json.loads(result.stdout)
    if audit.get('errors') != []:
        raise RuntimeError('source audit did not return an empty error list')
    return audit


def copy_exclusive(source, destination):
    with source.open('rb') as src, destination.open('xb') as dst:
        shutil.copyfileobj(src, dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = (ROOT / args.source).resolve()
    out = (ROOT / args.out).resolve()
    allowed = ((ROOT / 'datasets/exports/waste_bin').resolve(), (ROOT / 'tmp').resolve())
    if not any(source.is_relative_to(p) for p in allowed) or not source.is_dir():
        parser.error('source must be an existing directory under datasets/exports/waste_bin or tmp')
    if not any(out.is_relative_to(p) for p in allowed):
        parser.error('output must be under datasets/exports/waste_bin or tmp')
    if out.is_relative_to(source) or source.is_relative_to(out):
        parser.error('source and output must not overlap')
    if out.exists():
        parser.error('output already exists')
    data = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    if data.get('status') != 'complete_candidates':
        raise ValueError('source render run is incomplete or failed')
    if data.get('label_contract_id') != STATE_CONTRACT:
        raise ValueError('source uses an outdated or unknown label contract; export refused')
    if any('mask_visible' not in sample or 'mask_state_visible' not in sample
           for sample in data['samples']):
        raise ValueError('all source samples require target and state visibility mask evidence')
    audit = run_source_audit(source)
    included, excluded = plan_samples(data['samples'])
    if not included:
        raise ValueError('source has no fully eligible labelled images')
    pairs = [(sample, *validate_pair(source, sample)) for sample in included]
    if len({image.name for _, image, _ in pairs}) != len(pairs):
        raise ValueError('source image names collide in the output')
    out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'source_directory': source.relative_to(ROOT).as_posix(),
              'source_scene': data.get('scene_file'), 'source_seed': data.get('seed'),
              'classes': data.get('classes'), 'source_audit': audit,
              'split_policy': 'unsplit synthetic candidates; group by source trajectory; no independent test split',
              'included': [], 'excluded': excluded}

    def save():
        (out / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    save()
    try:
        (out / 'images').mkdir()
        (out / 'labels').mkdir()
        for sample, image, label in pairs:
            image_out, label_out = out / 'images' / image.name, out / 'labels' / label.name
            copy_exclusive(image, image_out)
            copy_exclusive(label, label_out)
            report['included'].append({'image': image_out.relative_to(out).as_posix(),
                                       'label': label_out.relative_to(out).as_posix(),
                                       'source_image': sample['image'],
                                       'source_candidate_label': sample['candidate_label'],
                                       'group': sample.get('group'), 'trajectory': sample.get('trajectory'),
                                       'scenario': sample.get('scenario'), 'states': sample.get('states')})
            save()
        report['status'] = 'complete'
    except BaseException as exc:
        report['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        report['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        save()
    print(json.dumps({'status': report['status'], 'included': len(report['included']),
                      'excluded': len(excluded), 'out': out.relative_to(ROOT).as_posix()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
