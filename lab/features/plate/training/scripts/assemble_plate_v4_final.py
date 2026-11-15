"""Merge existing V4 training data with a completed 432-image render batch.

This entry does not render, train or evaluate. Its test split contains 108 new
rendered images; materials and the source frame are shared with training.
"""
from collections import Counter, defaultdict
import filecmp
import json
import math
from pathlib import Path
import re
import shutil
import sys

from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard

BASE = ROOT / 'datasets/exports/plate'
ORIGINAL = BASE / 'v4_delivery_20261113/manifest.json'
BATCH = BASE / 'v4_empty_scene_gpu_batch2_20261113/manifest.json'
OUT = BASE / 'v4_final_20261113'
PROFILES = ('normal', 'dark', 'low_saturation')
TEST_DARK_GAIN = 0.7


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rel(path):
    return path.relative_to(ROOT).as_posix()


def plan():
    original = read(ORIGINAL)
    if original.get('counts', {}).get('train') != 1905 or original.get('counts', {}).get('val') != 313:
        raise ValueError('Existing V4 delivery train/val counts differ from expected 1905/313')
    rows = []
    for index, row in enumerate(original['records']):
        if row['split'] in ('train', 'val'):
            rows.append(dict(row, input_manifest=rel(ORIGINAL), input_record_index=index))
    if Counter(r['split'] for r in rows) != {'train': 1905, 'val': 313}:
        raise ValueError('Existing V4 records disagree with declared train/val counts')

    batch = read(BATCH)
    if batch.get('status') != 'rendered_review_pending':
        raise ValueError('GPU batch is not complete; do not assemble a running or failed batch')
    if batch.get('skipped') or len(batch.get('records', [])) != 432:
        raise ValueError('Expected 432 rendered images and no skipped images; inspect the batch before assigning splits')
    indexed = {}
    for index, row in enumerate(batch['records']):
        match = re.fullmatch(r'plate_empty_(\d{4})\.png', Path(row['image']).name)
        if not match:
            raise ValueError('Unexpected render name: ' + str(row.get('image')))
        number = int(match.group(1))
        if number in indexed or not 0 <= number < 432:
            raise ValueError('Duplicate or invalid render index: ' + str(number))
        if row.get('illumination_profile') != PROFILES[number % 3]:
            raise ValueError('Profile does not match deterministic render index: ' + str(number))
        if row.get('illumination_profile') == 'low_saturation' and not math.isclose(float(row.get('texture_saturation', -1)), 0.4, rel_tol=0, abs_tol=1e-6):
            raise ValueError('Missing expected low-saturation texture metadata: ' + str(number))
        indexed[number] = (index, row)
    if set(indexed) != set(range(432)):
        raise ValueError('Incomplete render index range')
    for number in range(432):
        index, row = indexed[number]
        split = 'train' if number < 324 else 'test'
        rows.append(dict(row, split=split, input_manifest=rel(BATCH), input_record_index=index,
                         test_design='last 108 render outputs held for final test' if split == 'test' else None))
    if Counter(r['split'] for r in rows) != {'train': 2229, 'val': 313, 'test': 108}:
        raise ValueError('Unexpected planned split counts')
    images = [r['image'] for r in rows]
    if len(images) != len(set(images)):
        raise ValueError('A source image is assigned more than once')
    for split, expected in (('train', 108), ('test', 36)):
        profiles = Counter(r.get('illumination_profile') for r in rows
                           if r.get('input_manifest') == rel(BATCH) and r['split'] == split)
        if profiles != dict.fromkeys(PROFILES, expected):
            raise ValueError('New-batch profile counts do not match planned balance: ' + str(profiles))
    return rows


def validate(rows):
    sizes = defaultdict(list)
    for row in rows:
        image, label = ROOT / row['image'], ROOT / row['label']
        if not image.is_file() or not label.is_file():
            raise ValueError('Missing image or label: ' + str(image))
        with Image.open(image) as opened:
            opened.verify()
        with Image.open(image) as opened:
            width, height = opened.size
        if width > 1280 or height > 720:
            raise ValueError('Image exceeds 720p: ' + str(image))
        lines = label.read_text(encoding='utf-8').splitlines()
        if row['input_manifest'] == rel(BATCH) and len(lines) != 1:
            raise ValueError('Expected one generated plate box: ' + str(label))
        for line in lines:
            tokens = line.split()
            if len(tokens) != 5 or tokens[0] != '0':
                raise ValueError('Invalid plate label: ' + str(label))
            x, y, w, h = map(float, tokens[1:])
            if not all(math.isfinite(v) for v in (x, y, w, h)) or not (
                    0 < w <= 1 and 0 < h <= 1 and -1e-6 <= x-w/2 <= x+w/2 <= 1+1e-6
                    and -1e-6 <= y-h/2 <= y+h/2 <= 1+1e-6):
                raise ValueError('Invalid plate box: ' + str(label))
        sizes[image.stat().st_size].append((image, row['split']))
    ensure_no_cross_split_duplicates(sizes)


def ensure_no_cross_split_duplicates(sizes):
    """Compare same-size image bytes across splits without hashes."""
    for entries in sizes.values():
        for i, (left, left_split) in enumerate(entries):
            for right, right_split in entries[i+1:]:
                if left_split != right_split and filecmp.cmp(left, right, shallow=False):
                    raise ValueError('Identical image content crosses splits: ' + str(left) + ' / ' + str(right))


def write_image(row, source_image, destination):
    """Apply a recorded brightness change to the test low-saturation stratum."""
    if row['split'] == 'test' and row.get('illumination_profile') == 'low_saturation':
        with Image.open(source_image) as original:
            ImageEnhance.Brightness(original.convert('RGB')).enhance(TEST_DARK_GAIN).save(destination)
        return {'operation': 'brightness_multiply', 'brightness_gain': TEST_DARK_GAIN,
                'illumination_profile': 'simulated_low_light_and_low_saturation',
                'low_saturation_source': 'rendered_texture_saturation_0.4',
                'geometry_and_label': 'unchanged'}
    shutil.copy2(source_image, destination)
    return None


def assemble(rows, source_refs):
    validate(rows)
    if OUT.exists():
        raise ValueError('Output exists without a reusable completed receipt')
    for split in ('train', 'val', 'test'):
        for kind in ('images', 'labels'):
            (OUT / kind / split).mkdir(parents=True, exist_ok=False)
    records, counts, profiles = [], Counter(), Counter()
    output_sizes = defaultdict(list)
    for i, row in enumerate(rows):
        source_image, source_label = ROOT / row['image'], ROOT / row['label']
        split, stem = row['split'], f'plate_final_{i:06d}'
        image = OUT / 'images' / split / (stem + source_image.suffix.lower())
        label = OUT / 'labels' / split / (stem + '.txt')
        treatment = write_image(row, source_image, image)
        shutil.copy2(source_label, label)
        output_sizes[image.stat().st_size].append((image, split))
        record = dict(row)
        record.update(image=rel(image), label=rel(label), split=split,
                      input_image=row['image'], input_label=row['label'],
                      source_evidence=source_refs.get(row['image'], []),
                      final_image_processing=treatment)
        if not record['source_evidence']:
            raise ValueError('No indexed evidence for source: ' + row['image'])
        records.append(record)
        counts[split] += 1
        if row.get('illumination_profile'):
            profiles[(split, row['illumination_profile'])] += 1
    ensure_no_cross_split_duplicates(output_sizes)
    (OUT / 'data.yaml').write_text('train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: plate\n', encoding='utf-8')
    report = {
        'status': 'assembled_candidate', 'class_names': ['plate'], 'counts': dict(counts),
        'new_batch_profile_counts': [{'split': k[0], 'profile': k[1], 'images': v}
                                     for k, v in sorted(profiles.items())],
        'scope': 'Programmatic decode, YOLO label and cross-split identical-byte checks before and after image transformation; no model evaluation.',
        'source_relationships': [
            'All new rendered images use one shared source frame and plate texture; test is not independent acquisition.',
            'The 36 low-saturation test images receive simulated low-light brightness gain 0.7; labels stay unchanged.',
            'First 324 new renders are training, last 108 are held for final test; appearance and source material are shared.',
            'Historical candidate test images from the previous delivery are not assigned in this dataset.',
            'Physical plate dimensions are 0.36 x 0.24 m; source photograph is mapped nonuniformly; support size is illustrative.'
        ],
        'evaluation': {'status': 'not_run', 'weights': None, 'metrics': None, 'evidence_paths': None},
        'records': records,
    }
    (OUT / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': rel(OUT), 'counts': dict(counts)}, ensure_ascii=False))


def main():
    rows = plan()
    media = sorted({ROOT / row['image'] for row in rows})
    procedural = sorted({ROOT / row['label'] for row in rows} | {ORIGINAL, BATCH})
    parameters = {'base_train': 1905, 'base_val': 313, 'batch_images': 432,
                  'batch_train_indices': [0, 323], 'batch_test_indices': [324, 431],
                  'profile_cycle': list(PROFILES), 'dimensions': [640, 480],
                  'test_low_saturation_brightness_gain': TEST_DARK_GAIN}
    with generation_guard('plate_v4_final_20261113', media, [OUT], parameters,
                          code_paths=[Path(__file__)], procedural_inputs=procedural, root=ROOT) as run:
        if run.reused:
            print('Reused complete indexed plate V4 final dataset')
            return
        assemble(rows, run.source_refs)


if __name__ == '__main__':
    main()
