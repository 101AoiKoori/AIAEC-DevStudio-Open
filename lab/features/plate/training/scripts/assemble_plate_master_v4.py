"""Assemble V4 plate data using recorded Blender scenes and real backgrounds."""
from collections import Counter, defaultdict
import filecmp
import json
import math
from pathlib import Path
import shutil
import sys

from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard  # noqa: E402

BASE = ROOT / 'datasets/exports/plate'
ORIGINAL = BASE / 'v4_bg40_20261113/manifest.json'
OUT = BASE / 'v4_master_v4_20261113'
MASTER = 'datasets/assets/blender/shared/arena_4m2_pbr_20261112/arena_master_user_20261113.blend'
MASTER_BATCH = BASE / 'v4_master_background_composites_20261113/manifest.json'
MASTER_TEST = BASE / 'v4_master_test_background_composites_20261113/manifest.json'
EMPTY = 'lab/features/plate/training/scripts/render_plate_empty_scene.py'
MEDIA = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp',
         '.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.mts'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def plan():
    original = read(ORIGINAL)['records']
    first = read(MASTER_BATCH)['records']
    extra = read(MASTER_TEST)['records']
    if len(original) != 2680 or len(first) != 150 or len(extra) != 90:
        raise ValueError('Expected 2680 original, 150 master and 90 master-test records')
    master = [(MASTER_BATCH, i, row) for i, row in enumerate(first)] + [(MASTER_TEST, i, row) for i, row in enumerate(extra)]
    selected, excluded = [], []
    for index, row in enumerate(original):
        bg = row.get('background') or {}
        if bg.get('source_group') == EMPTY:
            excluded.append({'image': row['image'], 'split': row['split'],
                             'manifest': rel(ORIGINAL), 'record_index': index,
                             'reason': 'single-color procedural background excluded from this delivery'})
            continue
        if bg.get('kind') not in {'real', 'blender'}:
            raise ValueError('Original image lacks established background: ' + row['image'])
        selected.append((row, ORIGINAL, index, bg))
    for master_manifest, index, row in master:
        view = row.get('view_group')
        if row.get('mother_scene') != MASTER or not row.get('blender_background') or not view:
            raise ValueError('Master scene or recorded view missing: ' + row['image'])
        processing = row.get('image_processing') or {}
        if processing.get('tone_profile') == 'low_saturation':
            saturation = processing.get('saturation_gain')
            if not isinstance(saturation, (int, float)) or not 0 <= saturation <= .6:
                raise ValueError('Low-saturation parameter is missing: ' + row['image'])
        bg = {'kind': 'blender', 'source_group': MASTER, 'view_group': view}
        selected.append((row, master_manifest, index, bg))
    counts = Counter(row['split'] for row, _, _, _ in selected)
    dropped = Counter(row['split'] for row in excluded)
    new_counts = Counter(row['split'] for _, _, row in master)
    if dropped != {'train': 432, 'test': 108} or new_counts != {'train': 90, 'val': 30, 'test': 120}:
        raise ValueError('Unexpected excluded/master splits: ' + str((dropped, new_counts)))
    if counts != {'train': 1867, 'val': 393, 'test': 120}:
        raise ValueError('Unexpected final splits: ' + str(counts))
    if len({r['image'] for r, _, _, _ in selected}) != len(selected):
        raise ValueError('Same source output appears more than once')
    return selected, excluded


def reject_cross_split_duplicates(sizes):
    for images in sizes.values():
        for i, (left, ls) in enumerate(images):
            for right, rs in images[i+1:]:
                if ls != rs and filecmp.cmp(left, right, shallow=False):
                    raise ValueError('Identical image bytes across splits: ' + str(left) + ' / ' + str(right))


def verify(selected):
    sizes = defaultdict(list)
    for row, _, _, bg in selected:
        image, label = ROOT / row['image'], ROOT / row['label']
        source = ROOT / bg['source_group']
        if not image.is_file() or not label.is_file() or not source.is_file():
            raise ValueError('Missing image, label or actual background source: ' + row['image'])
        if bg['kind'] == 'blender' and (source.suffix.lower() != '.blend' or not bg.get('view_group')):
            raise ValueError('Blender scene and recorded view required: ' + row['image'])
        with Image.open(image) as obj:
            obj.verify()
        with Image.open(image) as obj:
            width, height = obj.size
        if width > 1280 or height > 720:
            raise ValueError('Image exceeds 720p: ' + row['image'])
        for line in label.read_text(encoding='utf-8').splitlines():
            fields = line.split()
            if len(fields) != 5 or fields[0] != '0':
                raise ValueError('Invalid plate label: ' + row['label'])
            x, y, w, h = map(float, fields[1:])
            if not all(math.isfinite(v) for v in (x, y, w, h)) or not (
                    0 < w <= 1 and 0 < h <= 1 and -1e-6 <= x-w/2 <= x+w/2 <= 1+1e-6
                    and -1e-6 <= y-h/2 <= y+h/2 <= 1+1e-6):
                raise ValueError('Plate box outside image: ' + row['label'])
        sizes[image.stat().st_size].append((image, row['split']))
    reject_cross_split_duplicates(sizes)


def write_image(source, target, row, manifest):
    tone = (row.get('image_processing') or {}).get('tone_profile')
    if manifest in (MASTER_BATCH, MASTER_TEST) and tone == 'low_saturation':
        with Image.open(source) as original:
            ImageEnhance.Brightness(original.convert('RGB')).enhance(.7).save(target)
        return {'operation': 'brightness_multiply', 'brightness_gain': .7,
                'source_image': row['image'], 'input_tone_profile': tone,
                'input_saturation_gain': row['image_processing']['saturation_gain'],
                'conditions': ['low_light', 'low_saturation'],
                'condition_basis': 'simulated; no physical illuminance measurement',
                'geometry_and_label': 'unchanged'}
    shutil.copy2(source, target)
    return None


def assemble(selected, excluded, refs):
    verify(selected)
    if OUT.exists():
        raise ValueError('Output exists without a reusable completed receipt')
    for split in ('train', 'val', 'test'):
        for kind in ('images', 'labels'):
            (OUT / kind / split).mkdir(parents=True, exist_ok=False)
    records = []
    counts, blender, combined_conditions = Counter(), Counter(), Counter()
    background_groups, view_groups = defaultdict(Counter), defaultdict(set)
    for i, (row, manifest, index, bg) in enumerate(selected):
        split = row['split']
        source, label = ROOT / row['image'], ROOT / row['label']
        name = 'plate_master_%05d' % i
        target = OUT / 'images' / split / (name + source.suffix.lower())
        target_label = OUT / 'labels' / split / (name + '.txt')
        processing = write_image(source, target, row, manifest)
        if processing:
            combined_conditions[split] += 1
        shutil.copy2(label, target_label)
        evidence = refs.get(row['image'], [])
        if not evidence:
            raise ValueError('No indexed source evidence for ' + row['image'])
        record = dict(row, image=rel(target), label=rel(target_label), background=bg,
                      input_image=row['image'], input_label=row['label'],
                      input_manifest=rel(manifest), input_record_index=index,
                      source_evidence=evidence,
                      final_image_processing=processing if processing is not None else row.get('final_image_processing'))
        records.append(record)
        counts[split] += 1
        background_groups[split][bg['source_group']] += 1
        if bg['kind'] == 'blender':
            blender[split] += 1
            view_groups[bg['source_group']].add(bg['view_group'])
    if combined_conditions != {'train': 30, 'val': 10, 'test': 40}:
        raise ValueError('Unexpected combined simulated lighting conditions: ' + str(combined_conditions))
    # Verify the final bytes as brightness conversion may change equality.
    final_sizes = defaultdict(list)
    for row in records:
        image = ROOT / row['image']
        final_sizes[image.stat().st_size].append((image, row['split']))
    reject_cross_split_duplicates(final_sizes)
    total = sum(counts.values())
    ratio = sum(blender.values()) / total
    if ratio < .4 or sum(len(v) for v in view_groups.values()) < 2:
        raise ValueError('Blender share or view groups do not satisfy delivery policy')
    non_blender = {bg['source_group'] for _, _, _, bg in selected if bg['kind'] != 'blender'}
    if len(non_blender) < 2:
        raise ValueError('At least two original non-Blender background groups are required')
    (OUT / 'data.yaml').write_text('train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: plate\n', encoding='utf-8')
    result = {'status': 'assembled_candidate', 'counts': dict(counts),
              'blender_backgrounds': dict(blender), 'blender_total': sum(blender.values()),
              'total': total, 'blender_ratio': ratio,
              'background_groups_by_split': {k: dict(v) for k, v in background_groups.items()},
              'blender_view_groups': {k: sorted(v) for k, v in view_groups.items()},
              'simulated_low_light_low_saturation_counts': dict(combined_conditions),
              'excluded_single_color_procedural': excluded,
              'scope': 'Generated imagery shares actual source materials; no independent acquisition or model result claimed.',
              'evaluation': {'status': 'not_run', 'weights': None, 'metrics': None},
              'records': records}
    (OUT / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'counts': dict(counts), 'blender_total': sum(blender.values()),
                      'total': total, 'blender_ratio': ratio}, ensure_ascii=False))


def main(argv=None):
    selected, excluded = plan()
    media = {ROOT / row['image'] for row, _, _, _ in selected}
    procedural = {ORIGINAL, MASTER_BATCH, MASTER_TEST} | {ROOT / row['label'] for row, _, _, _ in selected}
    for row, _, _, bg in selected:
        values = [bg['source_group'], row.get('parent_image'), row.get('source_video'),
                  row.get('parent_video'), row.get('source_background')]
        values.extend(row.get('source_assets') or [])
        for value in values:
            if isinstance(value, str) and Path(value).suffix.lower() in MEDIA:
                media.add(ROOT / value)
        if bg['kind'] == 'blender':
            procedural.add(ROOT / bg['source_group'])
    params = {'old_excluded': {'train': 432, 'test': 108},
              'master_additions': {'train': 90, 'val': 30, 'test': 120},
              'low_saturation_brightness_gain': .7,
              'minimum_blender_ratio': .4, 'preserve_parent_splits': True}
    with generation_guard('plate_master_v4_20261113', sorted(media), [OUT], params,
                          code_paths=[Path(__file__)], procedural_inputs=sorted(procedural), root=ROOT) as run:
        if run.reused:
            print('Reused indexed plate master V4 dataset')
            return
        assemble(selected, excluded, run.source_refs)


if __name__ == '__main__':
    main()
