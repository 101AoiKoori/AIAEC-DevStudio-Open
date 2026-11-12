"""Assemble existing person train/val/test candidates into one YOLO layout.

This only copies existing images and labels and writes metadata. It does not
train a model; the user has prohibited starting training. Evaluation labels and
recording-session independence remain candidates requiring review.
"""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from extend_real_scene_candidates import checked_file, read_rows, validate_yolo

ROOT = Path(__file__).resolve().parents[5]
EXPORTS = ROOT / 'datasets/exports/person'
EXPECTED = {'train': 4000, 'val': 200, 'test': 200}
CLASS_NAMES = {0: 'community', 1: 'noncommunity'}


def package(train, evaluation, output):
    train, evaluation, output = (p.resolve() for p in (train, evaluation, output))
    allowed = EXPORTS.resolve()
    if any(not p.is_relative_to(allowed) for p in (train, evaluation, output)) or output.exists():
        raise ValueError('sources and new output must be under person exports')
    if train == evaluation or any(output.is_relative_to(p) or p.is_relative_to(output) for p in (train, evaluation)):
        raise ValueError('inputs and output must not overlap')
    ts = json.loads((train/'summary.json').read_text(encoding='utf8'))
    es = json.loads((evaluation/'summary.json').read_text(encoding='utf8'))
    if ts.get('status') != 'complete_candidates' or ts.get('completed') != EXPECTED['train']:
        raise ValueError('incomplete training candidates')
    if es.get('status') != 'complete_candidates' or es.get('completed') != {k: EXPECTED[k] for k in ('val', 'test')}:
        raise ValueError('incomplete evaluation candidates')
    plans = {}
    for split in EXPECTED:
        base = train if split == 'train' else evaluation/split
        rows = read_rows(base/'manifest.jsonl')
        if len(rows) != EXPECTED[split] or len({r['image'] for r in rows}) != len(rows):
            raise ValueError('wrong count or repeated input: ' + split)
        plans[split] = []
        for row in rows:
            image = checked_file(base, row['image'])
            label = checked_file(base, row['label'])
            number = validate_yolo(label)
            if split == 'train':
                scene, split_tag = row['scenario'], row['split']
            else:
                scene, split_tag = row['scenario']['kind'], row['split_candidate']
            if split_tag != ('train_candidate' if split == 'train' else split):
                raise ValueError('incorrect source split: ' + split)
            if scene not in ('single', 'multi', 'occlusion', 'no_person') or (scene == 'no_person') != (number == 0):
                raise ValueError('scenario/label mismatch: ' + split)
            plans[split].append((image, label, row, scene))
    output.mkdir(parents=True)
    summary = {'status': 'running', 'expected': EXPECTED, 'completed': {},
               'class_names': CLASS_NAMES, 'train_permission': 'not_authorized',
               'usage': 'candidate package, labels and session independence require review',
               'sources': {'train': train.relative_to(ROOT).as_posix(),
                           'evaluation': evaluation.relative_to(ROOT).as_posix()},
               'limits': 'val/test each use 2 known figure designs; same physical designs and vehicle model; source recording sessions and links to train remain unverified',
               'source_groups': {'train': ts['source_group'], 'evaluation': es['source_groups']},
               'scenarios': {}}
    def save():
        (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    save()
    try:
        for split in EXPECTED:
            for parent in ('images', 'labels'):
                (output/parent/split).mkdir(parents=True)
            counts = Counter()
            with (output/('manifest_'+split+'.jsonl')).open('w', encoding='utf8') as manifest:
                for index, (image, label, row, scene) in enumerate(plans[split], 1):
                    stem = f'{split}_{index:06d}'
                    image_dest = f'images/{split}/{stem}{image.suffix.lower()}'
                    label_dest = f'labels/{split}/{stem}.txt'
                    shutil.copyfile(image, output/image_dest)
                    shutil.copyfile(label, output/label_dest)
                    record = {'image': image_dest, 'label': label_dest, 'split': split,
                              'source_image': image.relative_to(ROOT).as_posix(),
                              'source_label': label.relative_to(ROOT).as_posix(),
                              'scenario': scene, 'source_record': row}
                    manifest.write(json.dumps(record, ensure_ascii=False)+'\n')
                    counts[scene] += 1
            summary['completed'][split] = len(plans[split])
            summary['scenarios'][split] = dict(counts)
            save()
        # Passing the absolute YAML filename lets Ultralytics resolve these
        # relative directories from the YAML location on either OS.
        yaml_text = ('train: images/train\nval: images/val\n'
                     'test: images/test\nnames:\n  0: community\n  1: noncommunity\n')
        (output/'data.yaml').write_text(yaml_text, encoding='utf8')
        summary['status'] = 'complete_candidates'
    except BaseException as exc:
        summary['status'] = 'failed_or_interrupted'
        summary['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(ROOT/args.train, ROOT/args.evaluation, ROOT/args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
