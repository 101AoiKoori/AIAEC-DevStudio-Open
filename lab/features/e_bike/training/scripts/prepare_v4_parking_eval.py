"""Select non-diagnostic fallen-bike composites for bounded V4 development/test.

These are shared-source composites, not independent camera sessions. Existing
labels and pixels are copied without resampling or additional tone variants.
"""
import argparse
import filecmp
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'lab/training_common'), str(HERE)]
from camera_tone import source_role  # noqa: E402
from e_bike_queue_inputs import queue_inputs  # noqa: E402
from engine.config import load_config  # noqa: E402
from engine.data import inspect_dataset  # noqa: E402
from lab.training_common.production_guard import generation_guard  # noqa: E402
from lab.scripts.data import v4_source_gate as gate  # noqa: E402

SOURCE = ROOT / 'datasets/exports/e_bike/v4_scaled_parking_additional_20261113'
REVIEW = ROOT / 'datasets/annotated/e_bike/v4_low_angle_multibike_review_20261112/review_manifest.json'
PARENT_REVIEW = ROOT / 'datasets/exports/e_bike/v4_production_candidate_20261113/positive_parent_review_20261113.json'
CONFIG = ROOT / 'lab/features/e_bike/training/config/e_bike_v4_reviewed_858.yaml'
STAGED = ROOT / 'datasets/annotated/e_bike/v4_858_source_20261113/staged_source_gate.json'
TONES = ('original', 'mild', 'moderate')


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def select_rows(records, excluded, restricted, occupied, val_count, test_count, seed):
    """Balance tones and plans, excluding exact copies and restricted parents."""
    pools = defaultdict(list)
    skipped = Counter()
    for index, row in enumerate(records):
        foreground = [obj.get('foreground_source', {}).get('path', '') for obj in row['objects']]
        if (row['background_parent'] in restricted or row['background_source_video'] in restricted or
                any(Path(path).stem in excluded for path in foreground)):
            skipped['restricted_or_bad_parent'] += 1
            continue
        if row.get('tone') not in TONES or not any(obj['class'] == 'fallen' for obj in row['objects']):
            skipped['tone_or_fallen'] += 1
            continue
        pools[(row['tone'], row['plan'])].append((index, row))
    rng = random.Random(seed)
    for bucket in pools.values():
        rng.shuffle(bucket)
    plans = sorted({plan for _, plan in pools})
    schedule = [(tone, plan) for tone in TONES for plan in plans]
    occupied_by_size = defaultdict(list)
    for path in occupied:
        occupied_by_size[path.stat().st_size].append(path)
    chosen = []
    while len(chosen) < val_count + test_count:
        progress = False
        for tone, plan in schedule:
            bucket = pools[(tone, plan)]
            while bucket:
                index, row = bucket.pop()
                path = ROOT / row['image']
                size = path.stat().st_size
                if any(filecmp.cmp(path, prior, shallow=False) for prior in occupied_by_size[size]):
                    skipped['identical_output'] += 1
                    continue
                occupied_by_size[size].append(path)
                chosen.append((index, row))
                progress = True
                break
            if len(chosen) == val_count + test_count:
                break
        if not progress:
            raise ValueError('Insufficient distinct eligible fallen-bike samples: ' + str(len(chosen)))
    return [('val' if i < val_count else 'test', index, row)
            for i, (index, row) in enumerate(chosen)], dict(skipped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--val-count', type=int, default=24)
    parser.add_argument('--test-count', type=int, default=48)
    parser.add_argument('--seed', type=int, default=2026111303)
    args = parser.parse_args()
    out = (ROOT / args.output).resolve()
    owner = (ROOT / 'datasets/exports/e_bike').resolve()
    if not out.is_relative_to(owner) or out == owner:
        parser.error('output must be an e-bike export subdirectory')
    if args.val_count < 1 or args.test_count < 1 or args.val_count + args.test_count > 186:
        parser.error('positive val/test counts must total at most 186')
    manifest, qa = SOURCE / 'manifest.json', SOURCE / 'qa.json'
    data = json.loads(manifest.read_text(encoding='utf8'))
    check = json.loads(qa.read_text(encoding='utf8'))
    if check.get('error_count') != 0 or check.get('images') != len(data['records']):
        raise ValueError('Candidate structural QA must cover every input image')
    excluded = set(json.loads(PARENT_REVIEW.read_text(encoding='utf8'))['excluded_foreground_stems'])
    config = load_config(ROOT, rel(CONFIG))
    _, active, _ = inspect_dataset(ROOT, config)
    occupied = [ROOT / item['image'] for split in ('train', 'val') for item in active[split]]
    active_labels = [ROOT / item['label'] for split in ('train', 'val') for item in active[split]]
    queue = gate.build_queue(ROOT)
    parents = {ROOT / row[key] for row in data['records'] for key in ('background_parent', 'background_source_video')}
    restricted = {rel(path) for path in parents if source_role(ROOT, path, queue)[0] == 'regression'}
    selected, skipped = select_rows(data['records'], excluded, restricted, occupied,
                                    args.val_count, args.test_count, args.seed)
    # Selection checks candidate bytes before choosing the final subset, so
    # all candidate images participate in the guarded input inventory.
    media = list(occupied) + [ROOT / row['image'] for row in data['records']]
    for _, _, row in selected:
        media += [ROOT / row['image'], ROOT / row['background_image'], ROOT / row['background_parent'],
                  ROOT / row['background_source_video']]
        for obj in row['objects']:
            if obj.get('visible_mask'):
                media.append(ROOT / obj['visible_mask'])
            foreground = obj.get('foreground_source')
            if foreground:
                media += [ROOT / foreground[key] for key in ('path', 'source_frame', 'source_video')]
        media += [ROOT / actor['source'] for actor in row.get('other_actors', [])]
    media.append(ROOT / check['visual_evidence'])
    procedural = [manifest, qa, REVIEW, PARENT_REVIEW, CONFIG, STAGED,
                  ROOT / config['data'], *active_labels]
    procedural += [ROOT / row['label'] for _, _, row in selected]
    extra_media, extra_procedural = queue_inputs(ROOT, queue, parents)
    media = list(dict.fromkeys(media + extra_media))
    procedural = list(dict.fromkeys(procedural + extra_procedural))
    code = [Path(__file__), HERE / 'e_bike_queue_inputs.py', ROOT / 'lab/training_common/camera_tone.py',
            ROOT / 'lab/training_common/engine/config.py', ROOT / 'lab/training_common/engine/data.py',
            ROOT / 'lab/training_common/train.py', ROOT / 'lab/training_common/production_guard.py',
            ROOT / 'lab/scripts/data/v4_source_gate.py', ROOT / 'lab/scripts/media/media_index.py']
    with generation_guard('e_bike.parking_eval:' + rel(out), media, [out],
                          {'val_count': args.val_count, 'test_count': args.test_count, 'seed': args.seed},
                          procedural_inputs=procedural, code_paths=code, root=ROOT) as run:
        if run.reused:
            return
        if out.exists():
            raise ValueError('Output exists; select a new directory')
        _write(out, selected, skipped, args.seed, run.source_refs)


def _write(out, selected, skipped, seed, refs):
    for split in ('val', 'test'):
        for folder in ('images', 'labels'):
            (out / folder / split).mkdir(parents=True, exist_ok=True)
    records, counts = [], Counter()
    for split, index, row in selected:
        src_image, src_label = ROOT / row['image'], ROOT / row['label']
        name = src_image.name
        image, label = out / 'images' / split / name, out / 'labels' / split / (src_image.stem + '.txt')
        shutil.copy2(src_image, image)
        shutil.copy2(src_label, label)
        item = dict(row)
        item.update({'image': rel(image), 'label': rel(label), 'split': split,
                     'source_image': row['image'], 'source_label': row['label'],
                     'source_manifest': rel(SOURCE / 'manifest.json'), 'source_record_index': index,
                     'source_refs': refs.get(row['image'], []),
                     'label_review': 'AI source/background review and per-image mask/box structural QA; not human accepted'})
        records.append(item)
        counts[split] += 1
        counts[split + '_' + row['tone']] += 1
        counts[split + '_' + row['plan']] += 1
    manifest = {'status': 'shared_source_eval_candidates', 'seed': seed, 'counts': dict(counts),
                'skipped': skipped, 'generation_method': 'real_scene_composite',
                'source_manifest': rel(SOURCE / 'manifest.json'), 'source_qa': rel(SOURCE / 'qa.json'),
                'source_relationship': 'Background video, frames, and foreground assets are shared; '
                                       'selected outputs differ exactly from current train/val and each other.',
                'evaluation_scope': 'Synthetic low-camera parking composition; no independent-session claim.',
                'evaluation_status': 'not_run', 'records': records}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'output': rel(out), 'counts': dict(counts), 'skipped': skipped}, ensure_ascii=False))


if __name__ == '__main__':
    main()
