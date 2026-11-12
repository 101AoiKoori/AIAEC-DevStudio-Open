"""Package the reviewed mixed person candidates and an additional scene batch.

This prepares candidate images and labels only; it does not train a model or
create an evaluation split. Both batches belong to the same source group.
"""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
PERSON_EXPORTS = ROOT / 'datasets/exports/person'
SCENES = {'single': 1000, 'multi': 1200, 'occlusion': 1200, 'no_person': 600}
GROUP = 'scene_video_49s_shared'


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def validate_yolo(path):
    lines = path.read_text(encoding='utf-8').splitlines()
    for line in lines:
        parts = line.split()
        if len(parts) != 5 or parts[0] not in ('0', '1'):
            raise ValueError('invalid person label: ' + str(path))
        x, y, width, height = map(float, parts[1:])
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1
                and x - width / 2 >= -1e-6 and x + width / 2 <= 1 + 1e-6
                and y - height / 2 >= -1e-6 and y + height / 2 <= 1 + 1e-6):
            raise ValueError('person box outside image: ' + str(path))
    return len(lines)


def checked_file(base, relative):
    path = (base / relative).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError('missing or outside input: ' + str(relative))
    return path


def package(mixed, extra, output):
    mixed, extra, output = (p.resolve() for p in (mixed, extra, output))
    allowed = PERSON_EXPORTS.resolve()
    if any(not p.is_relative_to(allowed) for p in (mixed, extra, output)) or output.exists():
        raise ValueError('inputs and new output must be under person exports')
    if mixed == extra or any(output.is_relative_to(p) or p.is_relative_to(output) for p in (mixed, extra)):
        raise ValueError('input and output directories must not overlap')
    base_summary = json.loads((mixed / 'summary.json').read_text(encoding='utf-8'))
    extra_summary = json.loads((extra / 'summary.json').read_text(encoding='utf-8'))
    base_rows, extra_rows = read_rows(mixed / 'manifest.jsonl'), read_rows(extra / 'manifest.jsonl')
    if (base_summary.get('status') != 'complete_candidates' or base_summary.get('count') != 3500
            or extra_summary.get('status') != 'complete_candidates' or extra_summary.get('completed') != 500
            or len(base_rows) != 3500 or len(extra_rows) != 500):
        raise ValueError('input batches incomplete')
    planned, scenes = [], Counter()
    for kind, base, rows in (('mixed', mixed, base_rows), ('extra', extra, extra_rows)):
        seen = set()
        for row in rows:
            if kind == 'mixed':
                image, label, scene, group = row['image'], row['label'], row['scenario'], row['source_group']
            else:
                image, label = row['person_image'], row['person_label']
                scene, group = row['scenario']['kind'], row['background']['source_group']
            if row.get('split') != 'train_candidate' or group != GROUP or scene not in SCENES or image in seen:
                raise ValueError('invalid source identity, group or scenario')
            seen.add(image)
            image_file, label_file = checked_file(base, image), checked_file(base, label)
            labels = validate_yolo(label_file)
            if (scene == 'no_person') != (labels == 0):
                raise ValueError('scenario does not match person labels')
            scenes[scene] += 1
            planned.append((kind, image_file, label_file, row, scene))
    if dict(scenes) != SCENES:
        raise ValueError('scenario quotas do not match: ' + repr(dict(scenes)))
    output.mkdir(parents=True)
    (output / 'images').mkdir()
    (output / 'labels').mkdir()
    summary = {'status': 'running', 'requested': len(planned), 'completed': 0,
               'source_group': GROUP, 'split': 'train_candidate', 'scenarios': dict(scenes),
               'purpose': 'person detection candidates only; no independent evaluation',
               'sources': [mixed.relative_to(ROOT).as_posix(), extra.relative_to(ROOT).as_posix()]}
    try:
        with (output / 'manifest.jsonl').open('w', encoding='utf-8') as manifest:
            for index, (kind, image, label, row, scene) in enumerate(planned, 1):
                stem = f'person_{index:06d}'
                image_dest = 'images/' + stem + image.suffix.lower()
                label_dest = 'labels/' + stem + '.txt'
                shutil.copyfile(image, output / image_dest)
                shutil.copyfile(label, output / label_dest)
                manifest.write(json.dumps({'image': image_dest, 'label': label_dest,
                                           'source_batch': kind, 'source_record': row,
                                           'source_group': GROUP, 'scenario': scene,
                                           'split': 'train_candidate'}, ensure_ascii=False) + '\n')
                summary['completed'] = index
        summary['status'] = 'complete_candidates'
    except BaseException as exc:
        summary['status'] = 'failed_or_interrupted'
        summary['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mixed', type=Path, required=True)
    parser.add_argument('--extra', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(ROOT / args.mixed, ROOT / args.extra, ROOT / args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
