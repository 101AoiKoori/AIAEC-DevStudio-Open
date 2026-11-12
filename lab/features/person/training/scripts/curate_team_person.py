"""Assemble review-only YOLO person candidates from two documented sources.

One teammate submission contains synthetic images and original JSONL; the
other source contains locally generated gray-background preview images. Those
previews are recorded as excluded and never copied into the output images or
labels. Source video frames and illustrations are shared: no independent
validation/test split is created.
"""
import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[5]


def inside(root, path):
    p = (root / path).resolve()
    if not p.is_relative_to(root.resolve()):
        raise ValueError('path outside source')
    return p


def teammate_reasons(row, time_seconds):
    reasons = []
    aug = row.get('aug')
    if not isinstance(aug, dict):
        aug = {}
    flip = aug.get('flip')
    if flip is True:
        reasons.append('mirrored_person_illustration')
    elif flip is not False:
        reasons.append('mirror_status_unknown')
    if 27 <= time_seconds <= 33:
        reasons.append('background_may_contain_unlabelled_standees')
    if aug.get('occlude'):
        reasons.append('occlusion_not_reflected_in_bbox')
    if aug.get('crop'):
        reasons.append('crop_visibility_unverified')
    if row.get('label') == 2:
        reasons.append('negative_requires_full_resolution_review')
    if row.get('label') not in (0, 1, 2):
        reasons.append('unknown_class')
    return reasons


def read_mask(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('L')) > 127


def bbox_from_mask(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError('empty visible mask')
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def yolo_row(category, bbox, width, height):
    if category not in (0, 1) or len(bbox) != 4:
        raise ValueError('bad category or bbox')
    x0, y0, x1, y1 = bbox
    if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
        raise ValueError('bbox out of image')
    return f'{category} {(x0+x1)/(2*width):.6f} {(y0+y1)/(2*height):.6f} {(x1-x0)/width:.6f} {(y1-y0)/height:.6f}\n'


def preview_exclusion(row, image):
    """Record gray-background preview provenance without creating a label."""
    cls = {'community': 0, 'noncommunity': 1}.get(row.get('person_class'))
    if cls is None:
        raise ValueError('unknown preview person class')
    return {'source': 'local_occlusion_candidates', 'source_image': image,
            'source_group': row['source_group'], 'source_video': row['vehicle_video'],
            'source_identity': row['person_source'], 'class_id': cls,
            'reasons': ['gray_background_preview_excluded_from_real_scene_candidates']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission', type=Path, required=True)
    parser.add_argument('--own', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    submission, own, output = ((ROOT / x).resolve() for x in (args.submission, args.own, args.output))
    if not submission.is_relative_to(ROOT / 'datasets/raw/person') or not submission.is_dir():
        parser.error('submission must be inside datasets/raw/person')
    if not own.is_relative_to(ROOT / 'datasets/exports/e_bike') or not own.is_dir():
        parser.error('own source must be inside datasets/exports/e_bike')
    if not output.is_relative_to(ROOT / 'datasets/annotated/person') or output.exists():
        parser.error('output must be new and inside datasets/annotated/person')
    dataset = submission / 'dataset'
    with (dataset / 'reports/backgrounds_manifest.csv').open(encoding='utf8', newline='') as f:
        backgrounds = {x['file']: float(x['t']) for x in csv.DictReader(f)}
    own_records = json.loads((own / 'manifest.json').read_text(encoding='utf8'))['samples']
    plan, excluded = [], []
    for split in ('train', 'val', 'test'):
        for line in (dataset / 'synth/labels' / f'{split}.jsonl').read_text(encoding='utf8').splitlines():
            row = json.loads(line)
            image = inside(dataset / 'synth/images', row['image'])
            if not image.is_file() or row['background'] not in backgrounds:
                raise ValueError('missing team image or background record')
            t = backgrounds[row['background']]
            reasons = teammate_reasons(row, t)
            meta = {'source': 'teammate_synthetic', 'source_image': image.relative_to(ROOT).as_posix(),
                    'original_split': split, 'source_video_group': 'scene_video_49s_shared',
                    'source_background': row['background'], 'source_time_seconds': t,
                    'source_identity': row.get('identity'), 'class_id': row.get('label'),
                    'source_augmentation': row.get('aug'),
                    'intended_use': 'review_candidate_not_yet_ground_truth'}
            if reasons:
                excluded.append({**meta, 'reasons': reasons})
                continue
            with Image.open(image) as im:
                width, height = im.size
                im.verify()
            label = yolo_row(row['label'], row['bbox'], width, height)
            name = f'team_{split}_{image.stem}'
            plan.append((image, name, label, {**meta, 'bbox_xyxy': row['bbox'],
                        'box_semantics': 'unoccluded pasted-person rectangle; pending visual review'}))
    for row in own_records:
        image = inside(own, row['image'])
        if not image.is_file():
            raise ValueError('missing own gray-background preview')
        excluded.append(preview_exclusion(row, image.relative_to(ROOT).as_posix()))
    if not plan:
        raise ValueError('no eligible candidates')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'images').mkdir()
    (output / 'labels').mkdir()
    records = []
    try:
        for image, name, label, meta in plan:
            image_dest = output / 'images' / (name + image.suffix.lower())
            label_dest = output / 'labels' / (name + '.txt')
            shutil.copyfile(image, image_dest)
            label_dest.write_text(label, encoding='utf8')
            records.append({**meta, 'image': 'images/' + image_dest.name, 'label': 'labels/' + label_dest.name,
                            'status': 'AI_review_candidate_not_ground_truth', 'split': None})
        for name, data in [('manifest.jsonl', records), ('excluded.jsonl', excluded)]:
            (output / name).write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in data), encoding='utf8')
        counts = Counter(str(x['class_id']) for x in records)
        summary = {'status': 'complete_candidates', 'purpose': 'unsplit positive detection candidates; no verified negatives',
                   'label_names': {'0': 'community', '1': 'noncommunity'},
                   'included': len(records), 'classes': counts,
                   'excluded_team_samples': sum(x['source'] == 'teammate_synthetic' for x in excluded),
                   'excluded_preview_samples': sum(x['source'] == 'local_occlusion_candidates' for x in excluded),
                   'source_counts': Counter(x['source'] for x in records),
                   'note': 'All shared-source; visual review pending. Gray-background local previews are excluded from real-scene training. Not an independent real validation or test set.'}
        (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    except BaseException as exc:
        (output / 'FAILED.txt').write_text(f'{type(exc).__name__}: {exc}\n', encoding='utf8')
        raise
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
