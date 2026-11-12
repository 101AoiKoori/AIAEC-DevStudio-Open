"""Build person retraining candidates from reviewed existing capture regions and masks."""

import argparse
from collections import Counter, defaultdict
import csv
from itertools import chain
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw

from extend_real_scene_candidates import checked_file, read_rows, validate_yolo
from synthesize_real_ground import yolo_row

ROOT = Path(__file__).resolve().parents[5]
EXPORTS = ROOT / 'datasets/exports/person'
CAPTURE = ROOT / 'datasets/raw/site_capture_20261112/vehicle_images'
ARCHIVE = ROOT / 'Archive/Assets/电动车与人物'
DIAGNOSTIC = ROOT / 'datasets/annotated/site_capture_20261112/field_eval_v1/evaluation_candidates.json'
CLASS_NAMES = {'community': 0, 'noncommunity': 1}
SYNTHETIC = ('real_scene_candidates_3500_20261110', 'real_scene_extra_500_20261110')
SPLITS = ('train', 'val', 'test')

# Crop and visible figure coordinates in the native vehicle camera image.
CAR_REGIONS = (
    (48, (100, 245, 216, 480), ((0, '15', (145, 279, 201, 469)),)),
    (53, (207, 250, 306, 478), ((0, '12', (224, 280, 293, 467)),)),
    (53, (391, 249, 529, 480), ((0, '10', (402, 280, 480, 470)),)),
    (63, (128, 247, 210, 443), ((1, 'F2', (132, 273, 194, 413)),)),
    (63, (366, 249, 444, 435), ((0, '14', (370, 269, 430, 408)),)),
    (83, (136, 248, 247, 450), ((0, '4', (171, 272, 241, 421)),)),
    (83, (455, 248, 591, 466), ((0, '13', (471, 269, 536, 440)),)),
    (93, (73, 249, 177, 466), ((1, 'F1', (101, 280, 156, 437)),)),
)

ARCHIVE_REGIONS = (
    (1, (238, 253, 331, 480), ((0, '5', (253, 278, 323, 471)),)),
    (1, (456, 247, 570, 480), ((0, '16', (467, 276, 547, 471)),)),
    (4, (43, 243, 141, 448), ((1, 'F2', (72, 274, 138, 423)),)),
    (5, None, ()),
    (6, None, ()),
)

# Review was at 960×540. Coordinates are mapped back to 1920×1080.
VIDEO_REGIONS_HALF = (
    (4.5, None, ()),
    (25, (0, 0, 455, 540), ()),
    (40.5, (138, 43, 305, 399), ((1, 'F2', (141, 62, 269, 363)),)),
    (40.5, (646, 40, 827, 407), ((0, '14', (651, 73, 800, 369)),)),
    (60, (53, 162, 401, 481), ((0, '6', (78, 204, 169, 458)),
                               (0, '16', (248, 196, 351, 443)))),
)


def relative(path):
    return path.relative_to(ROOT).as_posix()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def scene_candidates():
    for source_name in SYNTHETIC:
        source = EXPORTS / source_name
        for row in read_rows(source / 'manifest.jsonl'):
            background = row['background']
            if background['kind'] not in {'ground_only', 'track_context'}:
                continue
            if (background['source_group'] != 'scene_video_49s_shared'
                    or background['frame'] > 179
                    or background['person_supervision'] != 'AI_visual_review_no_obvious_background_person'):
                raise ValueError('unexpected background provenance: ' + str(background))
            image = checked_file(source, row['person_image'])
            old_label = checked_file(source, row['person_label'])
            with Image.open(image) as photo:
                width, height = photo.size
                photo.load()
            lines, people = [], []
            for item in row['objects']:
                if item['kind'] != 'person':
                    continue
                mask_path = checked_file(source, item['visible_mask'])
                with Image.open(mask_path) as mask:
                    if mask.size != (width, height) or mask.mode != 'L':
                        raise ValueError('visible mask geometry mismatch: ' + str(mask_path))
                    box = mask.getbbox()
                if box is None or list(box) != item['visible_bbox_xyxy']:
                    raise ValueError('visible mask box mismatch: ' + str(mask_path))
                cls = CLASS_NAMES[item['category']]
                lines.append(yolo_row(cls, box, width, height))
                people.append({'class_id': cls, 'xyxy': list(box), 'template': item['source'],
                               'visible_mask': relative(mask_path), 'visible_fraction': item['visible_fraction']})
            if ((not people) != (row['scenario']['kind'] == 'no_person')
                    or len(people) != row['scenario']['people']
                    or lines != old_label.read_text(encoding='utf-8').splitlines()):
                raise ValueError('label or scenario disagrees with visible masks: ' + str(image))
            yield 'train', image, None, lines, people, {
                'source_group': background['source_group'], 'capture_source': background['source_video'],
                'source_frame': background['frame'], 'background': background,
                'scenario': row['scenario']['kind'], 'generation_method': 'real_scene_composite',
                'review_status': 'mask_geometry_verified_background_contact_sheet_reviewed',
                'workspace_image': relative(image), 'workspace_label': relative(old_label),
            }


def thumbnail(image):
    return np.asarray(image.convert('L').resize((96, 72), Image.Resampling.BILINEAR), dtype=np.float32)


def read_holdout():
    records = json.loads(DIAGNOSTIC.read_text(encoding='utf-8'))
    paths = {(ROOT / row['image']).resolve() for row in records}
    if len(paths) != 60 or any(not path.is_relative_to(ROOT) for path in paths):
        raise ValueError('unexpected diagnostic holdout inventory')
    tiles = []
    for path in sorted(paths):
        if path.parent == CAPTURE:
            with Image.open(path) as image:
                tiles.append(thumbnail(image))
    return paths, tiles


def from_capture(split, source, image, crop, figures, group, extra):
    width, height = image.size
    crop = crop or (0, 0, width, height)
    x0, y0, x1, y1 = crop
    if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
        raise ValueError('crop outside original: ' + str(source))
    lines, people = [], []
    for cls, template, box in figures:
        local = (box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0)
        lines.append(yolo_row(cls, local, x1 - x0, y1 - y0))
        people.append({'class_id': cls, 'template': template, 'xyxy': list(local), 'source_xyxy': list(box)})
    image = image.crop(crop)
    meta = {'source_group': group, 'capture_source': relative(source), 'source_size': [width, height],
            'source_crop_xyxy': list(crop), 'generation_method': (
                'direct_capture' if crop == (0, 0, width, height) else 'real_image_crop'),
            'scenario': 'real_positive' if people else 'real_negative',
            'review_status': 'visual_candidate_complete_within_region', **extra}
    return split, source, image, lines, people, meta


def capture_candidates():
    holdout, tiles = read_holdout()
    inventory = ROOT / 'datasets/annotated/site_capture_20261112/source_inventory.csv'
    with inventory.open(encoding='utf-8-sig', newline='') as stream:
        source_names = {row['asset_id']: row['original_name'] for row in csv.DictReader(stream)}
    for number, crop, figures in CAR_REGIONS:
        stem = f'vehicle_{number:04d}'
        path = checked_file(CAPTURE, stem + '.png')
        with Image.open(path) as source:
            image = source.convert('RGB')
        if path in holdout or image.size != (640, 480):
            raise ValueError('capture candidate overlaps holdout or geometry differs: ' + str(path))
        tile = thumbnail(image)
        for other in tiles:
            if (np.abs(tile - other).mean() <= 14
                    and np.corrcoef(tile.ravel(), other.ravel())[0, 1] >= .975):
                raise ValueError('near-duplicate of diagnostic image: ' + str(path))
        name = source_names[stem]
        if '_20261111_120' not in name:
            raise ValueError('unexpected inferred route source: ' + name)
        yield from_capture('train', path, image, crop, figures, 'vehicle_inferred_20261111_1205_route', {
            'original_name': name, 'route_basis': 'filename time and waypoint sequence; session unverified',
            'checked_against_vehicle_diagnostic_thumbnails': len(tiles),
        })
    for number, crop, figures in ARCHIVE_REGIONS:
        path, = (ARCHIVE / '照片').glob(f'*_{number:03d}.ppm')
        with Image.open(path) as source:
            image = source.convert('RGB')
        yield from_capture('val', path, image, crop, figures, 'archive_pics_zip_vehicle', {})
    video, = (ARCHIVE / '视频').glob('*_012.mp4')
    cap = cv2.VideoCapture(str(video))
    try:
        for seconds, crop, figures in VIDEO_REGIONS_HALF:
            cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
            ok, bgr = cap.read()
            if not ok or bgr.shape[:2] != (1080, 1920):
                raise ValueError(f'failed to read archive video at {seconds}s')
            image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            doubled_crop = tuple(value * 2 for value in crop) if crop is not None else None
            doubled_figures = [(cls, template, tuple(value * 2 for value in box)) for cls, template, box in figures]
            yield from_capture('test', video, image, doubled_crop, doubled_figures, 'archive_person_scan_012', {
                'requested_seconds': seconds, 'source_frame': int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1,
            })
    finally:
        cap.release()


def review_sheet(output, records):
    """Save the actual real regions with exported boxes for review."""
    width, height, columns = 220, 205, 5
    canvas = Image.new('RGB', (columns * width, ((len(records) + columns - 1) // columns) * height), '#19232f')
    caption = ImageDraw.Draw(canvas)
    for index, record in enumerate(records):
        x, y = (index % columns) * width, (index // columns) * height
        with Image.open(ROOT / record['image']) as source:
            picture = source.convert('RGB')
        old_width, old_height = picture.size
        picture.thumbnail((width - 4, height - 20))
        sx, sy = picture.width / old_width, picture.height / old_height
        draw = ImageDraw.Draw(picture)
        for person in record['objects']:
            x0, y0, x1, y1 = person['xyxy']
            draw.rectangle((x0 * sx, y0 * sy, x1 * sx, y1 * sy), outline='#53f478', width=2)
            draw.text((x0 * sx + 3, y0 * sy + 2), person['template'], fill='#53f478', stroke_width=1, stroke_fill='black')
        caption.text((x + 3, y + 1), Path(record['image']).stem, fill='white')
        canvas.paste(picture, (x, y + 17))
    canvas.save(output / 'reviewed_real_regions.jpg', quality=90)


def export(output):
    if not output.is_relative_to(EXPORTS.resolve()) or output == EXPORTS.resolve():
        raise ValueError('output must be a child of datasets/exports/person')
    targets = ['images', 'labels', 'data.yaml', 'workspace_sources.json', 'summary.json', 'reviewed_real_regions.jpg']
    if any((output / name).exists() for name in targets):
        raise ValueError('output already contains a dataset export')
    output.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        for name in ('images', 'labels'):
            (output / name / split).mkdir(parents=True)
    counts, classes, methods = Counter(), defaultdict(Counter), defaultdict(Counter)
    groups, negatives = defaultdict(set), Counter()
    records, reviewed = [], []
    summary = {'status': 'running', 'class_names': CLASS_NAMES}
    write_json(output / 'summary.json', summary)
    try:
        for split, source, picture, lines, people, meta in chain(scene_candidates(), capture_candidates()):
            if any(meta['source_group'] in groups[other] for other in SPLITS if other != split):
                raise ValueError('source group appears in two splits: ' + meta['source_group'])
            counts[split] += 1
            kind = 'real' if picture is not None else 'synth'
            name = f'{split}_{kind}_{counts[split]:05d}'
            image = output / 'images' / split / (name + '.png')
            label = output / 'labels' / split / (name + '.txt')
            if picture is None:
                shutil.copyfile(source, image)
            else:
                picture.save(image)
            label.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
            if validate_yolo(label) != len(people):
                raise ValueError('invalid exported label count: ' + str(label))
            record = {'image': relative(image), 'label': relative(label), 'split': split,
                      'workspace_image': relative(image), 'workspace_label': relative(label),
                      'objects': people, **meta}
            records.append(record)
            if picture is not None:
                reviewed.append(record)
            classes[split].update(p['class_id'] for p in people)
            methods[split][meta['generation_method']] += 1
            groups[split].add(meta['source_group'])
            negatives[split] += not people
        if any(set(classes[split]) != {0, 1} or not negatives[split] for split in SPLITS):
            raise ValueError('every split must cover both classes and contain an audited negative')
        write_json(output / 'workspace_sources.json', {'records': records})
        (output / 'data.yaml').write_text('train: images/train\nval: images/val\ntest: images/test\n'
                                          'names:\n  0: community\n  1: noncommunity\n', encoding='utf-8')
        review_sheet(output, reviewed)
        summary['status'] = 'complete_candidates'
    except BaseException as exc:
        summary['status'] = 'failed_or_interrupted'
        summary['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        summary.update(images=dict(counts), instances=classes, generation_methods=methods, negatives=dict(negatives),
                       source_groups={k: sorted(v) for k, v in groups.items()},
                       holdout=relative(DIAGNOSTIC),
                       scope='Real evaluation uses visually reviewed crops and negatives. '
                             'Complete-street labels and physical figure independence remain unverified.')
        write_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export((ROOT / args.output).resolve()), ensure_ascii=False))


if __name__ == '__main__':
    main()
