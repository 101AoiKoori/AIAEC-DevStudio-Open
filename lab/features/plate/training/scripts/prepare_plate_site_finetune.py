"""Export a traceable detector adaptation dataset and separate OCR manifests."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import sys
import os
from math import isfinite

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'lab/training_common'))

from engine.data import records


EXPORTS = ROOT / 'datasets/exports/plate'
SITE = ROOT / 'datasets/annotated/site_capture_20261112/field_eval_v1'
HISTORICAL = EXPORTS / 'provincial_synthetic'
FULL = {'visual_candidate', 'visual_negative'}
SITE_PLATES = {'京GV7T6J', '吉TWC818', '豫EUW2Q3'}

# Complete strings read from the unaugmented car photos. Video strings are
# preserved per box in the original diagnosis file.
PHOTO_TEXT = {
    'vehicle_0009': '京GV7T6J', 'vehicle_0011': '豫EUW2Q3',
    'vehicle_0029': '吉TWC818', 'vehicle_0030': '吉TWC818',
    'vehicle_0105': '京GV7T6J', 'vehicle_0109': '京GV7T6J',
    'vehicle_0116': '豫EUW2Q3',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    with path.open('x', encoding='utf-8') as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write('\n')


def image_path(value):
    path = (ROOT / value).resolve(strict=True)
    if not path.is_relative_to(ROOT / 'datasets') or not path.is_file():
        raise ValueError(f'Invalid dataset image: {value}')
    return path


def thumb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('L').resize((96, 72), Image.Resampling.BOX),
                          dtype=np.float32).ravel()


def near_holdout(training, holdout):
    sources = [(value, thumb(image_path(value))) for value in sorted(holdout)]
    for row in training:
        pixels = thumb(image_path(row['image']))
        for value, other in sources:
            if np.abs(pixels - other).mean() <= 14:
                if np.array_equal(pixels, other) or (
                        pixels.std() and other.std() and np.corrcoef(pixels, other)[0, 1] >= .975):
                    raise ValueError(f'Training image near frozen diagnostic: {row["image"]} ~ {value}')


def checked_boxes(row, width, height):
    result = []
    if (row['review_status'] == 'visual_candidate') != bool(row['boxes']):
        raise ValueError(f'Status and boxes disagree: {row["asset_id"]}')
    for box in row['boxes']:
        x1, y1, x2, y2 = box['xyxy']
        if (box['class_id'] != 0 or not all(isfinite(x) for x in (x1, y1, x2, y2))
                or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height)):
            raise ValueError(f'Plate box outside the original image: {row["asset_id"]}')
        values = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                  (x2 - x1) / width, (y2 - y1) / height)
        result.append('0 ' + ' '.join(f'{value:.8f}' for value in values) + '\n')
    return ''.join(result)


def detector_entry(destination, split, name, row, generation, role, label_text=None):
    original = image_path(row['image'])
    image = destination / 'images' / split / f'{name}{original.suffix.lower()}'
    label = destination / 'labels' / split / f'{name}.txt'
    shutil.copy2(original, image)
    if label_text is None:
        shutil.copy2(ROOT / row['label'], label)
    else:
        label.write_text(label_text, encoding='utf-8')
    return {
        'image': image.relative_to(ROOT).as_posix(),
        'label': label.relative_to(ROOT).as_posix(),
        'split': split, 'source_group': row['source_group'],
        'workspace_image': row['image'], 'workspace_label': row.get('label'),
        'generation_method': generation, 'role': role,
        'review_status': row.get('review_status', 'inherited_synthetic_annotation'),
        'asset_id': row.get('asset_id'), 'filename_time_hint': row.get('filename_time_hint'),
        'video_seconds': row.get('video_seconds'),
        'parking_slot_point_hint': row.get('point_hint'), 'parking_slot_verified': None,
    }


def legacy_records():
    result = []
    for name, folder in [('round2', HISTORICAL / 'round2_multi_car'),
                         ('procedural', HISTORICAL / 'synthetic_train/detector')]:
        config = {'data': (folder / 'dataset.yaml').relative_to(ROOT).as_posix(), 'task': 'detect'}
        for split in ('train', 'val'):
            for row in records(ROOT, config, split):
                result.append((name, split, row))
    return result


def ocr_data(destination, training_rows, diagnostic_rows):
    historical = HISTORICAL / 'synthetic_train/ocr'
    legacy = {split: read(historical / f'{split}_annotations.json') for split in ('train', 'val')}
    validation_strings = {row['text'] for row in legacy['val']['annotations']}
    charset = read(ROOT / 'ws/src/aec_vision/config/plate/charset.json')
    classes = [set(charset[name]) for name in charset['positions']]

    def valid(text):
        return len(text) == len(classes) and all(char in allowed for char, allowed in zip(text, classes))

    counts = {}
    for split, value in legacy.items():
        excluded = SITE_PLATES | (validation_strings if split == 'train' else set())
        candidates = [row for row in value['annotations'] if row['text'] not in excluded]
        for row in candidates:
            if not valid(row['text']) or not (historical / row['image']).is_file():
                raise ValueError(f'Invalid synthetic OCR sample: {row["image"]}')
        value.update(image_root=Path(os.path.relpath(historical, destination)).as_posix(),
                     count=len(candidates), annotations=candidates)
        write(destination / f'{split}_annotations.json', value)
        counts[split] = len(candidates)

    site_review, site_annotations = [], []
    for row in training_rows + diagnostic_rows:
        for index, box in enumerate(row['boxes']):
            text = (box.get('text_candidate') or box.get('visible_plate_text_candidate')
                    or PHOTO_TEXT.get(row['asset_id']) or '').replace('·', '')
            partial = row['review_status'] == 'partial'
            purpose = 'existing_site_diagnostic' if row in diagnostic_rows else 'adaptation_reference'
            ocr_status = ('partial' if partial else 'visual_text_candidate' if text
                          else 'text_unresolved')
            record = {
                'id': f'{row["asset_id"]}_{index + 1}', 'source_image': row['image'],
                'source_group': row['source_group'], 'video_seconds': row.get('video_seconds'),
                'source_box': box['xyxy'], 'role': purpose, 'status': ocr_status,
                'text_candidate': text or None,
                'physical_plate_group': text if text in SITE_PLATES else None,
                'parking_slot_point_hint': row.get('point_hint'), 'parking_slot_verified': None,
                'notes': row['notes'],
            }
            if text and not partial:
                if not valid(text) or text not in SITE_PLATES:
                    raise ValueError(f'Unexpected site OCR text: {text}')
                with Image.open(image_path(row['image'])) as image:
                    if any(value != int(value) for value in box['xyxy']):
                        raise ValueError(f'OCR crop requires integer original coordinates: {record}')
                    crop = image.convert('RGB').crop(tuple(int(value) for value in box['xyxy']))
                    name = f'{record["id"]}.png'
                    crop.save(destination / 'site_crops' / name)
                record['crop'] = 'site_crops/' + name
                if purpose == 'existing_site_diagnostic':
                    site_annotations.append({'image': record['crop'], 'text': text, 'id': record['id'],
                                             'physical_plate_group': text, 'review_status': ocr_status})
            site_review.append(record)
    write(destination / 'site_review.json', site_review)
    write(destination / 'site_test_annotations.json', {
        'schema_version': '1.0', 'task': 'license_plate_recognition', 'split': 'test',
        'image_root': '.', 'count': len(site_annotations), 'annotations': site_annotations,
        'scope': 'visually readable candidates of three known physical plates at the competition site',
    })
    return {**counts, 'site_test_crops': len(site_annotations),
            'site_review_boxes': len(site_review),
            'site_groups': sorted({row['physical_plate_group'] for row in site_annotations})}


def prepare(batch):
    reviewed = read(batch / 'review/site_train_boxes.json')
    diagnosis = read(SITE / 'evaluation_candidates.json')
    holdout = {row['image'] for row in diagnosis}
    diagnostics = [row for row in diagnosis if row['model_key'] == 'plate_detection']
    training = [row for row in reviewed if row['review_status'] in FULL]
    with (SITE / 'photo_selection.csv').open(encoding='utf-8-sig', newline='') as stream:
        excluded = {row['image'] for row in csv.DictReader(stream)
                    if row['disposition'] in {'selected', 'near_duplicate', 'excluded'}}
    if any(row['image'] in holdout | excluded for row in reviewed):
        raise ValueError('Reviewed adaptation photos overlap the frozen diagnostic or excluded photos')
    near_holdout(training, holdout)

    legacy = legacy_records()
    detector, ocr = batch / 'detector', batch / 'ocr'
    if any(path.exists() for path in (detector, ocr, batch / 'summary.json')):
        raise ValueError('Export destinations already exist')
    for split in ('train', 'val', 'test'):
        (detector / 'images' / split).mkdir(parents=True, exist_ok=False)
        (detector / 'labels' / split).mkdir(parents=True, exist_ok=False)
    (ocr / 'site_crops').mkdir(parents=True, exist_ok=False)
    exported = []
    for name, split, row in legacy:
        exported.append(detector_entry(detector, split, name + '__' + Path(row['image']).stem,
                                       row, 'provincial_synthetic', 'legacy_pretraining'))
    for split, batch_rows in (('train', training), ('test', diagnostics)):
        for row in batch_rows:
            if row['review_status'] not in FULL:
                continue
            with Image.open(image_path(row['image'])) as image:
                image.load()
                label = checked_boxes(row, *image.size)
            role = 'site_detector_adaptation' if split == 'train' else 'frozen_site_diagnostic'
            exported.append(detector_entry(detector, split, row['asset_id'], row,
                                           'direct_camera_capture', role, label))

    data = detector / 'data.yaml'
    data.write_text('train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: plate\n',
                    encoding='utf-8')
    write(detector / 'workspace_sources.json', {'schema_version': 1, 'records': exported})
    ocr_counts = ocr_data(ocr, reviewed, diagnostics)
    config = {'data': data.relative_to(ROOT).as_posix(), 'task': 'detect'}
    counts = {}
    for split in ('train', 'val', 'test'):
        items = records(ROOT, config, split)
        counts[split] = {'images': len(items), 'positive': sum(bool(r['instances']) for r in items),
                         'negative': sum(not r['instances'] for r in items),
                         'boxes': sum(r['instances'] for r in items)}
    summary = {
        'status': 'exported_visual_candidates', 'detector': counts, 'ocr': ocr_counts,
        'holdout_images_screened': len(holdout),
        'adaptation': dict(Counter(row['review_status'] for row in reviewed)),
        'diagnostic': dict(Counter(row['review_status'] for row in diagnostics)),
        'source_notes': [
            'The 18 car photos form a route inferred from filename times; session identity is unverified.',
            'Synthetic detector train/val retain the original two provincial source families.',
            'Site detector train/test share three physical plate instances and the same venue.',
            'Site test is the fixed diagnostic regression, with both original videos reserved for review and test.',
            'OCR training uses synthetic crops only, excluding site strings and validation-string overlap.',
            'Site OCR crops are visually read candidates; parking-slot ground truth is unresolved.',
        ],
    }
    write(batch / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    args = parser.parse_args()
    batch = (ROOT / args.batch).resolve()
    if batch == EXPORTS or not batch.is_relative_to(EXPORTS):
        parser.error('batch must be within datasets/exports/plate')
    try:
        prepare(batch)
    except (OSError, KeyError, ValueError) as exc:
        parser.exit(2, f'Export failed: {exc}\n')


if __name__ == '__main__':
    main()
