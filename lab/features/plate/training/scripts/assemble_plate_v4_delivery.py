"""Assemble reviewed plate V4 sources into an indexed, guarded delivery dataset."""
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
from PIL import Image

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard

BASE = ROOT / 'datasets/exports/plate'
OUT = BASE / 'v4_delivery_20261113'
PRODUCTION = BASE / 'v4_production_draft_20261113/manifest.json'
EXTRA = [BASE / 'v4_reviewed_scene_negatives_20261113/manifest.json',
         BASE / 'v4_reviewed_oblique_20261113/manifest.json']
RENDER = BASE / 'v4_empty_scene_gpu_20261113/manifest.json'
SYNTH = BASE / 'provincial_synthetic/synthetic_train/detector'
META = SYNTH / 'metadata.jsonl'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def plan():
    source = read(PRODUCTION)['records']
    rows = [dict(r, source_manifest=PRODUCTION.relative_to(ROOT).as_posix(),
                 source_record_index=i) for i, r in enumerate(source) if r['split'] in ('train', 'val')]
    used = {Path(r[k]).stem for r in rows for k in ('source_image', 'parent_image', 'image') if r.get(k)}
    for path in EXTRA:
        for i, record in enumerate(read(path)['records']):
            rows.append(dict(record, split='train', source_manifest=path.relative_to(ROOT).as_posix(), source_record_index=i))
    render = read(RENDER)
    if render['status'] != 'rendered_review_pending' or len(render['records']) != 108:
        raise ValueError('Expected the previously reviewed 108 renders')
    for i, record in enumerate(render['records']):
        rows.append(dict(record, split='train', source_manifest=RENDER.relative_to(ROOT).as_posix(),
                         source_record_index=i, review_status='108 projection labels checked; 9 representative renders reviewed'))
    metadata = {row['id']: row for line in META.read_text(encoding='utf-8').splitlines() for row in [json.loads(line)]}
    options = {'positive': [], 'negative': []}
    for image in sorted((SYNTH / 'images/val').glob('*.jpg')):
        if image.stem in used:
            continue
        item = metadata.get(image.stem, {})
        kind = item.get('kind')
        label = SYNTH / 'labels/val' / (image.stem + '.txt')
        if kind not in options or not label.is_file():
            continue
        with Image.open(image) as photo:
            width, height = photo.size
        raw = label.read_text(encoding='utf-8').splitlines()
        if kind == 'negative':
            correct = not raw
        else:
            vals = raw[0].split() if len(raw) == 1 else []
            box = item.get('bbox_xyxy')
            correct = bool(box) and len(vals) == 5 and vals[0] == '0'
            if correct:
                x1, y1, x2, y2 = box
                expected = ((x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height)
                correct = all(abs(float(a)-b) < 1e-6 for a, b in zip(vals[1:], expected))
        if correct:
            options[kind].append((image, label, item))
    selected = options['positive'][:90] + options['negative'][:10]
    if len(selected) != 100:
        raise ValueError('Need 90 positive and 10 negative non-overlapping synthetic test candidates')
    for image, label, item in selected:
        rows.append({'image': image.relative_to(ROOT).as_posix(), 'label': label.relative_to(ROOT).as_posix(),
                     'split': 'test', 'family': 'provincial_single_scene_procedural',
                     'generation_method': 'historical_procedural_plate_synthesis',
                     'parent_image': image.relative_to(ROOT).as_posix(),
                     'source_metadata': META.relative_to(ROOT).as_posix(), 'source_metadata_id': item['id'],
                     'review_status': 'labels_match_generator_metadata',
                     'source_history': 'historical development use; earlier initialization ancestry and siblings unknown'})
    return rows


def main():
    rows = plan()
    media = sorted({ROOT / r['image'] for r in rows})
    labels = {ROOT / r['label'] for r in rows}
    dependencies = sorted(labels | {PRODUCTION, *EXTRA, RENDER, META})
    with generation_guard('plate_v4_delivery_20261113', media, [OUT],
                          {'test_positive': 90, 'test_negative': 10, 'class_names': ['plate'],
                           'source_splits': ['train', 'val'], 'render_assignment': 'train'},
                          code_paths=[Path(__file__)], procedural_inputs=dependencies, root=ROOT) as run:
        if run.reused:
            print('Reused prior complete indexed plate V4 delivery')
            return
        if OUT.exists():
            raise ValueError('Output exists without reusable receipt')
        _assemble(rows, run.source_refs)


def _assemble(rows, source_refs):
    counts, records, exposure = Counter(), [], Counter()
    for split in ('train', 'val', 'test'):
        for kind in ('images', 'labels'):
            (OUT / kind / split).mkdir(parents=True, exist_ok=False)
    for index, row in enumerate(rows):
        src, label = ROOT / row['image'], ROOT / row['label']
        split = row['split']
        with Image.open(src) as image:
            image.verify()
        with Image.open(src) as image:
            w, h = image.size
            if w > 1280 or h > 720:
                raise ValueError('Image exceeds 720p bounds: ' + str(src))
        text = label.read_text(encoding='utf-8').strip()
        for line in text.splitlines():
            tokens = line.split()
            if len(tokens) != 5 or tokens[0] != '0':
                raise ValueError('Invalid plate label: ' + str(label))
            x, y, width, height = map(float, tokens[1:])
            if not (0 < width <= 1 and 0 < height <= 1 and -1e-6 <= x-width/2 <= x+width/2 <= 1+1e-6 and -1e-6 <= y-height/2 <= y+height/2 <= 1+1e-6):
                raise ValueError('Plate box out of bounds: ' + str(label))
        name = f'plate_v4_{index:06d}'
        image_out = OUT / 'images' / split / (name + src.suffix.lower())
        label_out = OUT / 'labels' / split / (name + '.txt')
        shutil.copy2(src, image_out); shutil.copy2(label, label_out)
        exposure_value = row.get('tone_profile') or row.get('processing', {}).get('brightness') or row.get('value_gain')
        exposure[str(exposure_value) if exposure_value is not None else 'not_recorded'] += 1
        records.append({'image': image_out.relative_to(ROOT).as_posix(), 'label': label_out.relative_to(ROOT).as_posix(),
                        'split': split, 'source_image': row['image'], 'source_label': row['label'],
                        'generation_method': row.get('generation_method'), 'source_group': row.get('source_group'),
                        'parent_image': row.get('parent_image'), 'source_video': row.get('source_video'),
                        'source_time_seconds': row.get('source_time_seconds'), 'review_status': row.get('review_status'),
                        'source_manifest': row.get('source_manifest'), 'source_record_index': row.get('source_record_index'),
                        'source_evidence': source_refs.get(row['image'], []), 'source_metadata': row.get('source_metadata'),
                        'source_metadata_id': row.get('source_metadata_id'), 'source_history': row.get('source_history'),
                        'source_tone_profile': exposure_value, 'image_size': [w, h]})
        counts[split] += 1
    if dict(counts) != {'train': 1905, 'val': 313, 'test': 100}:
        raise ValueError('Unexpected split totals: ' + str(dict(counts)))
    (OUT / 'data.yaml').write_text('train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: plate\n', encoding='utf-8')
    manifest = {'status': 'configured_v4_data', 'counts': dict(counts), 'records': records,
                'class_names': ['plate'], 'recorded_tone_distribution': dict(exposure),
                'known_source_relationships': [
                    'Site video frames and 108 Blender renders share the same source video and physical props.',
                    'The 108 renders use one 46x18-pixel plate crop mapped to a 36x24-cm face; support is illustrative and appears floating.',
                    'The test candidates share synthetic assets/backgrounds and have historical development use; source-image identities differ from current train and val.',
                    'Prior provincial weight ancestry and sibling reuse are not fully known; report test scope accordingly.'],
                'evaluation': {'status': 'not_run', 'metrics': None, 'weights': None}}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': OUT.relative_to(ROOT).as_posix(), 'counts': dict(counts),
                      'recorded_tone_distribution': dict(exposure)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
