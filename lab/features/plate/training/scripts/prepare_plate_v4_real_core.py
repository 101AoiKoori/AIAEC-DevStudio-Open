"""Assemble a small, traceable real-camera plate dataset draft.

The split proposal is deliberately marked incomplete; it is not the final
training package or a frozen test.  Each parent and its tone variants stay in
one split.  Run only after the referenced task-specific reviews exist.
"""
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT / 'lab/training_common')]
from camera_tone import source_role  # noqa: E402
from lab.scripts.data.v4_source_gate import build_queue  # noqa: E402
OUT = ROOT / 'datasets/exports/plate/v4_real_core_draft_20261112'
PHOTO_REVIEW = ROOT / 'datasets/annotated/plate/v4_real_photo_video_candidates_20261112/review_manifest.json'
VIDEO_REVIEW = ROOT / 'datasets/annotated/site_capture_20261112/sign_plate_v4_review/batch2/review_manifest.json'
TONE_SOURCES = (
    ROOT / 'datasets/exports/mixed_scene/v4_dark_plate_real_photo_pilot_20261112',
    ROOT / 'datasets/exports/mixed_scene/v4_dark_plate_real_photo_batch2_20261112',
    ROOT / 'datasets/exports/mixed_scene/v4_dark_plate_negative_batch_20261112',
)
PHOTO_SPLITS = {'vehicle_0016': 'train', 'vehicle_0035': 'train', 'vehicle_0107': 'train',
                'vehicle_0111': 'train', 'vehicle_0112': 'train', 'vehicle_0118': 'train',
                'vehicle_0072': 'val', 'vehicle_0076': 'test', 'vehicle_0079': 'test'}
VIDEO_SPLITS = {'frame_002850': 'train', 'frame_002000': 'val', 'frame_007800': 'test'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rel(path):
    return path.relative_to(ROOT).as_posix()


def _generate():
    if OUT.exists():
        raise ValueError('Output already exists; choose a new version instead of overwriting')
    photos = {item['id']: item for item in read(PHOTO_REVIEW)['samples']}
    videos = {Path(item['image']).stem: item for item in read(VIDEO_REVIEW)['samples']}
    tones = {}
    for directory in TONE_SOURCES:
        for item in read(directory / 'manifest.json')['samples']:
            tones.setdefault(item['parent_image'], []).append((directory / item['image'], item))
    if set(PHOTO_SPLITS) - photos.keys() or set(VIDEO_SPLITS) - videos.keys():
        raise ValueError('A reviewed real source is missing')
    known = build_queue(ROOT)
    for key in PHOTO_SPLITS:
        parent = ROOT / photos[key]['parent_image']
        role, reason = source_role(ROOT, parent, known)
        if role == 'regression':
            raise ValueError(f'{key} is restricted to regression: {reason}')
    records, counts = [], Counter()
    try:
        for split in ('train', 'val', 'test'):
            for part in ('images', 'labels'):
                (OUT / part / split).mkdir(parents=True, exist_ok=True)
        for key, split in list(PHOTO_SPLITS.items()) + list(VIDEO_SPLITS.items()):
            photo = key in PHOTO_SPLITS
            item = photos[key] if photo else videos[key]
            if photo:
                if item.get('state') != 'candidate_complete_positive' or not item.get('independent_review'):
                    raise ValueError('Photo has not had both recorded reviews: ' + key)
                parent = ROOT / item['parent_image']
                label = ROOT / item['label']
                scene_group = item['scene_group']
                evidence = rel(PHOTO_REVIEW)
                video, seconds = None, None
            else:
                task = item['tasks']['plate_detection']
                if task.get('state') != 'complete_negative' or not item.get('independent_review'):
                    raise ValueError('Video frame is not a reviewed negative: ' + key)
                parent = ROOT / item['image']
                label = ROOT / task['label']
                scene_group = item['segment_id']
                evidence = rel(VIDEO_REVIEW)
                video, seconds = item['parent_video'], item['source_time_seconds']
            text = label.read_text(encoding='utf-8').strip()
            if bool(text) != photo:
                raise ValueError('Source label does not match reviewed positive/negative state: ' + key)
            variants = [(parent, None)] + tones.get(rel(parent), [])
            if len(variants) != 3:
                raise ValueError('Each source requires its reviewed original and two tone candidates: ' + key)
            for image, tone in variants:
                with Image.open(image) as decoded:
                    decoded.load()
                    width, height = decoded.size
                if min(width, height) > 720 or max(width, height) > 1280:
                    raise ValueError('Image exceeds 720p: ' + rel(image))
                profile = tone['profile'] if tone else 'original'
                stem = key + '_' + profile
                dest = OUT / 'images' / split / (stem + image.suffix.lower())
                target = OUT / 'labels' / split / (stem + '.txt')
                shutil.copy2(image, dest)
                target.write_text(text + ('\n' if text else ''), encoding='utf-8')
                records.append({'image': rel(dest), 'label': rel(target), 'split': split,
                    'state': 'positive' if photo else 'negative', 'parent_image': rel(parent),
                    'source_image': rel(image), 'source_video': video, 'source_time_seconds': seconds,
                    'source_scene_group': scene_group, 'source_review': evidence,
                    'generation_method': 'camera_capture' if photo and tone is None else
                                         'provided_video_frame' if tone is None else 'augmentation',
                    'tone_profile': profile, 'tone_processing': tone['processing'] if tone else None,
                    'image_size': [width, height], 'label_review': 'candidate_two_AI_reviews',
                    'split_status': 'provisional_not_frozen'})
                counts[(split, 'positive' if photo else 'negative')] += 1
        (OUT / 'data.yaml').write_text(yaml.safe_dump({'names': {0: 'plate'},
            **{split: 'images/' + split for split in ('train', 'val', 'test')}}, allow_unicode=True), encoding='utf-8')
        manifest = {'status': 'draft_not_deliverable', 'purpose': 'Real-camera and video-frame core; '
            'more source groups, task conditions and source-gate approval are required before packaging.',
            'counts': {split: {state: counts[(split, state)] for state in ('positive', 'negative')}
                       for split in ('train', 'val', 'test')},
            'distinct_parent_images': len(PHOTO_SPLITS) + len(VIDEO_SPLITS),
            'notes': ['Tone variants do not create independent parents.',
                      'Scene/material reuse is recorded in the linked review; identity may be unknown.',
                      'Proposed test is not frozen and has not been evaluated.'], 'records': records}
        (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'output': rel(OUT), 'counts': manifest['counts'],
                          'parents': manifest['distinct_parent_images']}, ensure_ascii=False))
    except BaseException:
        (OUT / 'FAILED.txt').write_text('Draft assembly failed; inspect before reuse.\n', encoding='utf-8')
        raise


def _guard_inputs():
    photos = {x['id']: x for x in read(PHOTO_REVIEW)['samples']}
    videos = {Path(x['image']).stem: x for x in read(VIDEO_REVIEW)['samples']}
    parents = [photos[k]['parent_image'] for k in PHOTO_SPLITS] + [videos[k]['image'] for k in VIDEO_SPLITS]
    media = [ROOT / x for x in parents]
    labels = [ROOT / photos[k]['label'] for k in PHOTO_SPLITS] + [ROOT / videos[k]['tasks']['plate_detection']['label'] for k in VIDEO_SPLITS]
    manifests = [PHOTO_REVIEW, VIDEO_REVIEW] + [x / 'manifest.json' for x in TONE_SOURCES]
    for directory in TONE_SOURCES:
        for row in read(directory / 'manifest.json')['samples']:
            if row['parent_image'] in parents:
                media.append(directory / row['image'])
    return media, labels + manifests, {'photo_splits': PHOTO_SPLITS, 'video_splits': VIDEO_SPLITS}


def main():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from lab.training_common.production_guard import generation_guard
    media, procedural, parameters = _guard_inputs()
    from lab.scripts.data import v4_source_gate as gate
    procedural += [ROOT / value for value in (gate.INVENTORY, gate.SEGMENTS, gate.SELECTION,
        gate.EXPOSURE, gate.DUPLICATES, gate.DEVELOPMENT_EXPOSURE, gate.REVIEW_SUMMARY,
        'config/perception_tasks.json', *gate.ANNOTATIONS, *gate.SITE_SELECTIONS)]
    from lab.features.plate.training.scripts.plate_guard_inputs import source_gate_dependencies
    gate_files, gate_code = source_gate_dependencies(ROOT, gate.build_queue(ROOT))
    procedural += list(gate_files)
    media = sorted(set(media))
    procedural = sorted(set(procedural))
    with generation_guard("plate.prepare_plate_v4_real_core", media, [OUT], parameters,
                          code_paths=sorted(gate_code | {Path(__file__), ROOT / 'lab/training_common/camera_tone.py'}), procedural_inputs=procedural, root=ROOT) as run:
        if run.reused:
            return
        _generate()
        manifest_path = OUT / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['source_refs'] = run.source_refs
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
