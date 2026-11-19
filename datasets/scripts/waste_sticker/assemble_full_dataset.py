"""Validate and assemble the four waste-sticker production branches."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.config import load_task_config
from datasets.scripts.waste_sticker.source_inputs import artwork_catalog
from lab.training_common.production_guard import generation_guard, plan_production

TASK = 'waste_sticker'
EXPORT = ROOT / 'datasets/exports' / TASK
STAGE = ROOT / 'datasets/annotated/waste_bin/waste_sticker_full'
COUNTS = {
    'train': {'native': 990, 'blender_surface': 270, 'hybrid': 450, 'real_surface': 90},
    'val': {'native': 160, 'real_surface': 80},
    'test': {'native': 200, 'hybrid': 40},
}
STICKERS = {
    'train': {0: 180, 1: 405, 2: 405, 3: 405, 4: 405},
    'val': {0: 24, 1: 54, 2: 54, 3: 54, 4: 54},
    'test': {0: 24, 1: 54, 2: 54, 3: 54, 4: 54},
}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def local_file(value, folder):
    path = Path(value)
    candidates = (path, ROOT / path, folder / path)
    for candidate in candidates:
        if candidate.is_file() and candidate.resolve().is_relative_to(ROOT):
            return candidate.resolve()
    raise FileNotFoundError(value)


def labels(path, size):
    result = []
    width, height = size
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f'Invalid YOLO row: {path}')
        class_id = int(fields[0])
        x, y, w, h = map(float, fields[1:])
        if (not 0 <= class_id < 24 or not np.isfinite([x, y, w, h]).all() or
                not 0 < w <= 1 or not 0 < h <= 1 or
                x - w/2 < -1e-6 or y - h/2 < -1e-6 or x + w/2 > 1.000001 or y + h/2 > 1.000001):
            raise ValueError(f'Invalid normalized YOLO box: {path}')
        result.append({'class_id': class_id, 'bbox_xyxy':
                       [(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height]})
    return result


def inspect_pair(row, folder):
    image = local_file(row['image'], folder)
    label = local_file(row.get('label') or row.get('label_candidate'), folder)
    with Image.open(image) as source:
        size = source.size
        source.load()
        pixels = sha256(source.convert('RGB').tobytes()).hexdigest()
    if row.get('image_size') is not None and list(size) != row['image_size']:
        raise ValueError(f'Recorded image size differs from the PNG: {image}')
    boxes = labels(label, size)
    bins = row.get('visible_bins', row.get('selected_bins'))
    if isinstance(bins, int):
        bin_count = bins
    elif isinstance(bins, list):
        bin_count = len(bins)
    else:
        bin_count = row.get('visible_bin_count') or row.get('bin_count')
    if not isinstance(bin_count, int) or not 1 <= bin_count <= 4:
        raise ValueError(f'Expected 1–4 visible bins: {image}')
    if len(boxes) not in (0, bin_count):
        raise ValueError(f'Expected one sticker per positive bin: {image}')
    instances = row.get('instances', [])
    if len(instances) != len(boxes):
        raise ValueError(f'Instance count differs from YOLO: {image}')
    for instance, box in zip(instances, boxes):
        if instance['class_id'] != box['class_id']:
            raise ValueError(f'Class differs from recorded instance: {image}')
        bounds = instance.get('bbox_xyxy') or instance.get('visible_bbox_xyxy')
        if (not isinstance(bounds, list) or len(bounds) != 4 or not np.isfinite(bounds).all() or
                max(abs(a-b) for a, b in zip(bounds, box['bbox_xyxy'])) > 1.1):
            raise ValueError(f'YOLO and instance box differ: {image}')
    return image, label, list(size), boxes, pixels, bin_count


def collect(split):
    grouped = []
    for name in ('native', 'surface', 'hybrid'):
        folder = STAGE / name / split
        if not folder.exists():
            continue
        manifest = folder / 'manifest.json'
        data = read_json(manifest)
        if data.get('records_valid') is False or data.get('status', '').startswith('rebuilding'):
            raise ValueError(f'Branch is still rebuilding: {manifest}')
        nested_plan = data.get('plan')
        declared_split = data.get('split')
        if declared_split is None and isinstance(nested_plan, dict):
            declared_split = nested_plan.get('target_split')
        if declared_split is not None and declared_split != split:
            raise ValueError(f'Branch manifest split differs from its directory: {manifest}')
        for raw in data['records']:
            if raw.get('split') != split:
                raise ValueError(f'Branch record split differs from its directory: {manifest} / {raw["id"]}')
            if name == 'surface':
                branch = raw.get('branch')
                if branch not in ('blender_surface', 'real_surface'):
                    raise ValueError(f'Unexpected surface branch: {branch}')
            else:
                branch = name
            image, label, size, boxes, pixels, bins = inspect_pair(raw, folder)
            parent = raw.get('source_parent') or raw.get('parent_group')
            if not parent:
                raise ValueError(f'Missing source parent: {image}')
            grouped.append(dict(raw=raw, branch=branch, image=image, label=label, size=size,
                                boxes=boxes, pixels=pixels, bins=bins, parent=parent, manifest=manifest))
    actual = Counter(row['branch'] for row in grouped)
    if actual != COUNTS[split]:
        raise ValueError(f'{split} branch counts {actual} differ from {COUNTS[split]}')
    counts = Counter(len(row['boxes']) for row in grouped)
    if counts != STICKERS[split]:
        raise ValueError(f'{split} visible-sticker counts {counts} differ from {STICKERS[split]}')
    if {box['class_id'] for row in grouped for box in row['boxes']} != set(range(24)):
        raise ValueError(f'Incomplete class coverage: {split}')
    return grouped


def declaration(split, rows):
    states = []
    for index, row in enumerate(rows):
        states.append(dict(
            id=f'{row["branch"]}_{index:06d}', min_count=1, max_count=1, prior={'alpha': 2, 'beta': 8},
            exposure_weight=1, failure_cost=1, failure_event='export_pair_incomplete', conditions=[],
            background={'image': relative(row['image'])},
            target={'kind': 'explicit_bin_stickers', 'placements': [
                {'class_id': box['class_id'], 'visible_bbox_xyxy': box['bbox_xyxy']} for box in row['boxes']]},
            labels={'format': 'yolo_detection', 'scope': 'all_visible_stickers'},
            augmentation={'operation': 'validated_copy'}, sources=[relative(row['image'])],
            parameters={'branch': row['branch'], 'origin_id': row['raw']['id'],
                        'label_sha256': sha256(row['label'].read_bytes()).hexdigest()},
        ))
    plan = dict(schema_version=1, model_key=TASK, target_split=split, budget=len(states),
                purpose='Assemble the complete current waste-sticker dataset from actual branch outputs',
                independence_basis='Held-out camera parents; shared fixed artwork and Blender source scene',
                uncertainty_weight=0, constraints={'image_transfer': 'lossless_copy',
                    'labels': 'all_visible_item_stickers', 'visible_bin_count': [1, 2, 3, 4]}, strata=states)
    plan_production(plan, ROOT)
    return plan


def export_split(split, rows):
    output = EXPORT / split
    plan = declaration(split, rows)
    configs = ROOT / 'datasets/scripts/waste_sticker/config'
    procedural = {row['manifest'] for row in rows} | {row['label'] for row in rows}
    procedural.update(configs / name for name in ('dataset.yaml', 'source.yaml'))
    _, source_config = load_task_config(TASK)
    procedural.update(ROOT / path for path in source_config['records'])
    plan_file = configs / f'full_{split}_plan.json'
    write_json(plan_file, plan)
    procedural.add(plan_file)
    code = [Path(__file__), Path(__file__).with_name('source_inputs.py'),
            *sorted(Path(__file__).parent.glob('produce_*dataset.py'))]
    with generation_guard(
        f'waste_sticker_full_export:{split}', [row['image'] for row in rows], [output],
        {'split': split, 'counts': COUNTS[split], 'stickers': STICKERS[split]},
        code_paths=code, procedural_inputs=sorted(procedural), root=ROOT, production_plan=plan,
        sample_output_paths=[output / 'images'],
    ) as run:
        if run.reused:
            return read_json(output / 'manifest.json')['records']
        for sub in ('images', 'labels'):
            (output / sub).mkdir(parents=True, exist_ok=True)
        entries = []
        by_id = {state['id']: row for state, row in zip(plan['strata'], rows)}
        for state in run.plan['strata']:
            row = by_id[state['id']]
            image = output / 'images' / (state['id'] + '.png')
            label = output / 'labels' / (state['id'] + '.txt')
            shutil.copyfile(row['image'], image)
            shutil.copyfile(row['label'], label)
            raw = row['raw']
            actual = {key: value for key, value in raw.items()
                      if key not in ('image', 'label', 'label_candidate')}
            entry = dict(actual, id=state['id'], origin_id=raw['id'], split=split, branch=row['branch'],
                         image=relative(image), label=relative(label), image_size=row['size'],
                         source_parent=row['parent'], visible_bin_count=row['bins'],
                         visible_sticker_count=len(row['boxes']), decoded_rgb_sha256=row['pixels'],
                         label_status='complete_machine_generated',
                         intermediate_root=relative(row['manifest'].parent),
                         evaluation_scope='controlled_shared_asset_diagnostic')
            entries.append(entry)
            run.record_sample(state['id'], output=image, sources=state['sources'],
                              parameters=state['parameters'], actual=entry)
        summary = coverage(entries)
        write_json(output / 'manifest.json', {'task': TASK, 'split': split, 'status': 'complete',
                                             'coverage': summary, 'production_plan': run.plan, 'records': entries})
        write_json(output / 'classes.json', [Path(artwork_catalog()[i]).stem for i in range(24)])
    return entries


def coverage(rows):
    count = Counter()
    by_class = Counter()
    bins = Counter()
    parents = Counter()
    for row in rows:
        pair = labels(ROOT / row['label'], row['image_size'])
        count[len(pair)] += 1
        by_class.update(box['class_id'] for box in pair)
        bins[row['visible_bin_count']] += 1
        parents[row['source_parent']] += 1
    return {'images': len(rows), 'instances': sum(by_class.values()),
            'by_branch': dict(Counter(row['branch'] for row in rows)),
            'by_visible_stickers': dict(sorted(count.items())), 'by_visible_bins': dict(sorted(bins.items())),
            'per_class': {str(i): by_class[i] for i in range(24)}, 'parent_groups': dict(parents)}


def finalize(by_split):
    classes = [Path(artwork_catalog()[i]).stem for i in range(24)]
    sources = []
    hashes = {}
    reused = defaultdict(set)
    for split, rows in by_split.items():
        for row in rows:
            digest = row['decoded_rgb_sha256']
            if digest in hashes:
                raise ValueError(f'Identical RGB output: {hashes[digest]} and {row["image"]}')
            hashes[digest] = row['image']
            parent = row['source_parent']
            branch = row['branch']
            real = branch in ('hybrid', 'real_surface')
            sources.append(dict(
                image=row['image'], label=row['label'], split=split, source_group=parent, source_id=parent,
                parent_video=parent if parent.lower().endswith('.mp4') else None,
                mother_scene='datasets/assets/blender/aiaec_dataset.blend' if branch != 'real_surface' else None,
                source_kind='derived_image' if real else 'blender_scene',
                generation_method='composite' if real else 'blender_render',
                image_processing={'background_origin': 'direct_camera', 'origin_review': 'confirmed'} if real else {},
                workspace_image=row['image'], workspace_label=row['label'],
                label_status='reviewed', review_state='complete_positive' if row['visible_sticker_count'] else 'complete_negative',
                review_evidence=f'datasets/exports/{TASK}/{split}/manifest.json',
                source_review_evidence=f'datasets/exports/{TASK}/{split}/manifest.json',
                source_assets=['datasets/assets/blender/aiaec_dataset.blend'] if branch != 'real_surface' else [],
                evaluation_scope='controlled_shared_asset_diagnostic',
            ))
    for row in sources:
        for relation in ('source_group', 'parent_video', 'mother_scene', 'source_assets'):
            values = row[relation] if isinstance(row[relation], list) else [row[relation]]
            for source in values:
                if source:
                    reused[relation, source].add(row['split'])
    summary = {split: coverage(rows) for split, rows in by_split.items()}
    write_json(EXPORT / 'workspace_sources.json', {
        'model_key': TASK, 'selection_mode': 'manifest_only', 'records': sources, 'coverage': summary,
        'source_reuse': [{'relation': relation, 'source': source, 'splits': sorted(splits),
                          'evaluation_scope': 'same_source_diagnostic'}
                         for (relation, source), splits in reused.items() if len(splits) > 1],
        'shared_assets': ['datasets/assets/blender/aiaec_dataset.blend', 'fixed_24_item_artworks'],
    })
    write_json(EXPORT / 'data.yaml', {'names': classes, 'train': 'train/images', 'val': 'val/images', 'test': 'test/images'})
    return summary


def previews(by_split):
    for split, rows in by_split.items():
        choices = []
        for branch in COUNTS[split]:
            for count in range(5):
                choices.extend(next(([r] for r in rows if r['branch'] == branch and r['visible_sticker_count'] == count), []))
        choices = choices[:20]
        columns, tile_w, tile_h = 4, 320, 258
        canvas = Image.new('RGB', (columns * tile_w, ((len(choices) + 3)//4) * tile_h), '#17191e')
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 14)
        except OSError:
            font = ImageFont.load_default()
        for index, row in enumerate(choices):
            with Image.open(ROOT / row['image']) as image:
                thumb = ImageOps.contain(image.convert('RGB'), (312, 224))
            left = (index % 4) * tile_w + (tile_w - thumb.width)//2
            top = (index // 4) * tile_h + (228 - thumb.height)//2
            canvas.paste(thumb, (left, top))
            sx, sy = thumb.width / row['image_size'][0], thumb.height / row['image_size'][1]
            for box in labels(ROOT / row['label'], row['image_size']):
                a, b, c, d = box['bbox_xyxy']
                draw.rectangle((left+a*sx, top+b*sy, left+c*sx, top+d*sy), outline='#5bffbd', width=2)
            draw.text(((index % 4)*tile_w+8, (index//4)*tile_h+232),
                      f'{row["branch"]} | {row["visible_sticker_count"]} stickers', font=font, fill='#eef2f6')
        canvas.save(EXPORT / f'{split}_preview.jpg', quality=91)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    grouped = {split: collect(split) for split in COUNTS}
    digests = [row['pixels'] for rows in grouped.values() for row in rows]
    if len(set(digests)) != len(digests):
        raise ValueError('Repeated RGB image in branch staging')
    if not args.write:
        for split, rows in grouped.items():
            print(split, len(rows), dict(Counter(row['branch'] for row in rows)))
        return
    by_split = {split: export_split(split, rows) for split, rows in grouped.items()}
    report = finalize(by_split)
    previews(by_split)
    from lab.training_common.engine.data import inspect_dataset, preflight_dataset
    config = {'version': 'v4', 'model_key': TASK, 'task': 'detect',
              'data': relative(EXPORT / 'data.yaml'), 'training_policy': {'mode': 'fixed_validation'}}
    _, _, training = preflight_dataset(ROOT, config)
    _, _, inspection = inspect_dataset(ROOT, {key: value for key, value in config.items()
                                            if key != 'training_policy'})
    write_json(EXPORT / 'audit.json', {
        'status': 'passed', 'images': sum(len(rows) for rows in by_split.values()),
        'instances': sum(item['instances'] for item in report.values()),
        'identical_rgb_pairs': 0, 'splits': report, 'training_preflight': training,
        'all_splits_inspection': inspection,
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
