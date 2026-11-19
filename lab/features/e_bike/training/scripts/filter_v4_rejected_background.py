"""Copy V4 samples with permitted camera backgrounds into an isolated candidate batch.

The original dataset and source gate are never changed. The retained batch is
still a candidate: vehicle contact and the low-angle evaluation scenes require
their own review before any updated distribution is released.
"""
import argparse
import filecmp
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT / 'lab/training_common')]
from engine.config import load_config  # noqa: E402
from engine.data import inspect_dataset  # noqa: E402
from lab.training_common.production_guard import generation_guard  # noqa: E402
from camera_tone import source_role  # noqa: E402
from e_bike_queue_inputs import queue_inputs  # noqa: E402
from lab.scripts.data.v4_source_gate import build_queue  # noqa: E402

SOURCE = ROOT / 'datasets/annotated/e_bike/v4_858_source_20261113/staged_source_gate.json'
DISTRIBUTION = ROOT / 'datasets/annotated/e_bike/v4_858_source_20261113/distribution_manifest.json'
CONFIG = ROOT / 'lab/features/e_bike/training/config/e_bike_v4_reviewed_858.yaml'
REJECTED_VIDEO = 'Archive/Assets/working_sources/legacy/site/场景/综合视频/空场与场景物料_001.mp4'
FIELD_VIDEO = 'Archive/Assets/captures/site_capture_20261112/videos/full_scene.mp4'
EVAL_VIDEO = 'Archive/Assets/working_sources/legacy/e_bike/视频/场地多辆电动车_直立与倒伏混合_较低视角_003.mp4'
FIELD_FRAMES = 'datasets/annotated/site_capture_20261112/sequence_640_v1/full_scene/'
ALLOWED_TRAIN = FIELD_FRAMES + 'frame_005751.png'
REJECTED_TRAIN = {FIELD_FRAMES + 'frame_007000.png', FIELD_FRAMES + 'frame_007045.png'}
REJECTED_VAL = {FIELD_FRAMES + 'frame_002800.png', FIELD_FRAMES + 'frame_008000.png'}
EVAL_PARENT = 'datasets/annotated/e_bike/v4_low_angle_multibike_review_20261112/images/'
LOW_REVIEW = ROOT / 'datasets/exports/e_bike/v4_low_angle_reviewed_20261113/manifest.json'
FRAME_REVIEW = ROOT / 'datasets/annotated/e_bike/v4_low_angle_multibike_review_20261112/review_manifest.json'


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def classify(row, allowed_train, allowed_eval):
    """Fail closed for unrecognised backgrounds; never override rejected families."""
    split = row.get('split')
    provenance = row.get('provenance')
    if not isinstance(provenance, dict):
        raise ValueError('Missing provenance: ' + str(row.get('id')))
    background = provenance.get('background_image')
    video = row.get('parent_video')
    if not isinstance(video, str) or not isinstance(background, str):
        raise ValueError('Missing background or parent video: ' + str(row.get('id')))
    if video == REJECTED_VIDEO:
        return 'rejected_user_background_family'
    if video == FIELD_VIDEO and split == 'train' and background in REJECTED_TRAIN:
        return 'rejected_overhead_train'
    if video == FIELD_VIDEO and split == 'val' and background in REJECTED_VAL:
        return 'rejected_nonvehicle_view_val'
    if video == FIELD_VIDEO and split == 'train' and background in allowed_train:
        state = row.get('tasks', {}).get('e_bike', {}).get('state')
        if state == 'complete_negative':
            return 'retained_reviewed_negative'
        if state == 'complete_positive':
            return 'rejected_suspended_composite_family'
        raise ValueError('Unknown supervision state: ' + str(row.get('id')))
    if (video in allowed_eval and split in ('val', 'test') and
            background.startswith(EVAL_PARENT) and Path(background).name in
            {'frame_001.png', 'frame_002.png', 'frame_003.png', 'frame_004.png'}):
        return 'rejected_suspended_composite_family'
    raise ValueError(f'Unknown background policy: {row.get("id")} {split} {video} {background}')


def plan_rows(source, active, allowed_train, allowed_eval):
    selected = [(index, row) for index, row in enumerate(source['samples'])
                if row.get('split') in ('train', 'val', 'test') and
                'e_bike' in row.get('selected_tasks', [])]
    by_image = {}
    for index, row in selected:
        image = row.get('image')
        if image in by_image:
            raise ValueError('Duplicate selected source image: ' + str(image))
        by_image[image] = (index, row)
    actual = [item for split in ('train', 'val', 'test') for item in active[split]]
    if len(actual) != len(by_image) or {x['image'] for x in actual} != set(by_image):
        raise ValueError('Source gate does not exactly cover the active dataset')
    result = []
    for item in actual:
        index, row = by_image[item['image']]
        if row['split'] != item['split'] or row['tasks']['e_bike']['label'] != item['label']:
            raise ValueError('Active split/label differs from source gate: ' + item['image'])
        result.append((index, row, item, classify(row, allowed_train, allowed_eval)))
    return result


def checked_real_rows(review, frames, known):
    """Verify the twelve source frames and labels without inventing ground truth."""
    if review.get('status') != 'candidate_review_passed' or review.get('errors') or len(review['records']) != 12:
        raise ValueError('Expected twelve AI-reviewed source candidates without recorded errors')
    by_parent = {row['source_frame']: row for row in frames['rows']}
    if len(by_parent) != 4:
        raise ValueError('Expected four reviewed low-angle parent frames')
    checked = set()
    seen = set()
    for row in review['records']:
        parent = by_parent.get(row['parent_image'])
        if (not parent or row['source_video'] != EVAL_VIDEO or row['split'] != 'train_candidate' or
                'AI_visual' not in row['review'] or row['boxes'] != parent['boxes'] or
                row['source_frame_index'] != parent['source_frame_index']):
            raise ValueError('Low-angle candidate does not match reviewed parent')
        if row['image'] in seen:
            raise ValueError('Duplicate low-angle image')
        seen.add(row['image'])
        for value in (row['parent_image'], row['source_video']):
            if value not in checked and source_role(ROOT, ROOT / value, known)[0] == 'regression':
                raise ValueError('Low-angle source is restricted to regression: ' + value)
            checked.add(value)
        with Image.open(ROOT / row['image']) as image:
            width, height = image.size
            image.verify()
        if [width, height] != parent['output_image_size']:
            raise ValueError('Low-angle image dimensions differ from reviewed parent')
        lines = (ROOT / row['label']).read_text(encoding='utf8').splitlines()
        if len(lines) != len(row['boxes']):
            raise ValueError('Low-angle label count differs from reviewed boxes')
        for box, line in zip(row['boxes'], lines):
            values = [float(value) for value in line.split()]
            x1, y1, x2, y2 = box['xyxy']
            expected = [0 if box['class'] == 'upright' else 1,
                        (x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                        (x2 - x1) / width, (y2 - y1) / height]
            if len(values) != 5 or any(abs(a - b) > 0.00002 for a, b in zip(values, expected)):
                raise ValueError('Low-angle label differs from reviewed box')
    if {row['tone'] for row in review['records']} != {'original', 'mild', 'moderate'}:
        raise ValueError('Low-angle tone coverage changed')
    return review['records']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--allow-train-parent', action='append', default=[])
    parser.add_argument('--allow-eval-video', action='append', default=[])
    args = parser.parse_args()
    out = (ROOT / args.output).resolve()
    owner = (ROOT / 'datasets/exports/e_bike').resolve()
    if not out.is_relative_to(owner) or out == owner:
        parser.error('output must be an e-bike export subdirectory')
    allowed_train = {ALLOWED_TRAIN, *args.allow_train_parent}
    allowed_eval = {EVAL_VIDEO, *args.allow_eval_video}
    source = json.loads(SOURCE.read_text(encoding='utf8'))
    config = load_config(ROOT, rel(CONFIG))
    _, active, _ = inspect_dataset(ROOT, config)
    planned = plan_rows(source, active, allowed_train, allowed_eval)
    review = json.loads(LOW_REVIEW.read_text(encoding='utf8'))
    frames = json.loads(FRAME_REVIEW.read_text(encoding='utf8'))
    known = build_queue(ROOT)
    real_rows = checked_real_rows(review, frames, known)

    # All active images and labels participate in the decision, even rejected
    # images. Declare the known source media and review evidence as inputs.
    media = {ROOT / item['image'] for _, _, item, _ in planned}
    procedural = {SOURCE, DISTRIBUTION, CONFIG, ROOT / config['data'], LOW_REVIEW, FRAME_REVIEW,
                  ROOT / review['source_manifest']}
    media.add(ROOT / review['review_contact'])
    for row in real_rows:
        media.update(ROOT / row[key] for key in ('image', 'parent_image', 'source_video'))
        procedural.add(ROOT / row['label'])
    extra_media, extra_procedural = queue_inputs(ROOT, known,
        [ROOT / row[key] for row in real_rows for key in ('parent_image', 'source_video')])
    media.update(extra_media)
    procedural.update(extra_procedural)
    for _, row, item, _ in planned:
        procedural.add(ROOT / item['label'])
        for value in [row.get('parent_video'), row['provenance'].get('background_image'),
                      *(row.get('source_assets') or [])]:
            if value:
                media.add(ROOT / value)
        task = row['tasks']['e_bike']
        for evidence in (row.get('review_evidence'), task.get('review_evidence'),
                         task.get('strata_review_evidence')):
            if evidence:
                procedural.add(ROOT / evidence)
    codes = [Path(__file__), ROOT / 'lab/training_common/production_guard.py',
             ROOT / 'lab/training_common/engine/config.py', ROOT / 'lab/training_common/engine/data.py',
             ROOT / 'lab/training_common/train.py', ROOT / 'lab/training_common/camera_tone.py',
             ROOT / 'lab/features/e_bike/training/scripts/e_bike_queue_inputs.py',
             ROOT / 'lab/scripts/media/media_index.py', ROOT / 'lab/scripts/data/v4_source_gate.py']
    params = {'allowed_train_parents': sorted(allowed_train), 'allowed_eval_videos': sorted(allowed_eval),
              'rejected_video': REJECTED_VIDEO, 'rejected_train_parents': sorted(REJECTED_TRAIN),
              'rejected_val_parents': sorted(REJECTED_VAL)}
    with generation_guard('e_bike.filter_background:' + rel(out), sorted(media), [out], params,
                          procedural_inputs=sorted(procedural), code_paths=codes, root=ROOT) as run:
        if run.reused:
            print(json.dumps({'status': 'reused', 'output': rel(out)}, ensure_ascii=False))
            return
        if out.exists():
            raise ValueError('Output exists; select a new directory')
        write_batch(out, planned, real_rows, params, run.source_refs)


def write_batch(out, planned, real_rows, policy, refs):
    for split in ('train', 'val', 'test'):
        for kind in ('images', 'labels'):
            (out / kind / split).mkdir(parents=True, exist_ok=True)
    counts, decisions, records = Counter(), [], []
    for index, row, item, decision in planned:
        counts[decision + '_' + row['split']] += 1
        source_record = {'source_gate_manifest': rel(SOURCE), 'source_gate_record_index': index,
                         'source_image': item['image'], 'source_label': item['label'],
                         'source_split': row['split'], 'decision': decision,
                         'background_image': row['provenance']['background_image'],
                         'parent_video': row['parent_video'], 'source_group': row.get('source_group'),
                         'source_time_seconds': row.get('source_time_seconds'),
                         'source_parent_ids': row.get('parent_ids', []),
                         'source_generation_method': row.get('generation_method'),
                         'image_size': row.get('image_size')}
        decisions.append(source_record)
        if decision != 'retained_reviewed_negative':
            continue
        old_image, old_label = ROOT / item['image'], ROOT / item['label']
        # Distinct prefixes preserve unique stems even when sources have the same filename.
        stem = f'{row["split"]}_{index:06d}_{old_image.stem}'
        image = out / 'images' / row['split'] / (stem + old_image.suffix)
        label = out / 'labels' / row['split'] / (stem + '.txt')
        shutil.copy2(old_image, image)
        shutil.copy2(old_label, label)
        if not filecmp.cmp(old_image, image, shallow=False) or not filecmp.cmp(old_label, label, shallow=False):
            raise IOError('Copied sample differs from its source: ' + item['image'])
        record = dict(source_record, image=rel(image), label=rel(label), split=row['split'],
                      generation_method='unchanged_copy', source_database_refs=refs.get(item['image'], []),
                      label_state=row['tasks']['e_bike']['state'],
                      source_gate_review_evidence=row.get('review_evidence'),
                      label_review_evidence=row['tasks']['e_bike'].get('review_evidence'),
                      review_status='source_negative_review_retained')
        records.append(record)
    for index, row in enumerate(real_rows):
        old_image, old_label = ROOT / row['image'], ROOT / row['label']
        stem = f'real_{index:02d}_{old_image.stem}'
        image, label = out / 'images/train' / (stem + old_image.suffix), out / 'labels/train' / (stem + '.txt')
        shutil.copy2(old_image, image)
        shutil.copy2(old_label, label)
        if not filecmp.cmp(old_image, image, shallow=False) or not filecmp.cmp(old_label, label, shallow=False):
            raise IOError('Copied low-angle sample differs from its source: ' + row['image'])
        with Image.open(old_image) as source_image:
            image_size = list(source_image.size)
        records.append({'image': rel(image), 'label': rel(label), 'split': 'train',
                        'source_image': row['image'], 'source_label': row['label'],
                        'source_manifest': rel(LOW_REVIEW), 'source_record_index': index,
                        'source_database_refs': refs.get(row['image'], []),
                        'background_image': row['parent_image'], 'parent_video': row['source_video'],
                        'source_pts_seconds': row['source_pts_seconds'], 'tone': row['tone'],
                        'image_size': image_size, 'reviewed_boxes': row['boxes'],
                        'generation_method': row['generation_method'],
                        'label_state': 'complete_positive', 'review_status': row['review'],
                        'source_group': 'one shared low-angle video and physical arrangement'})
        counts['retained_reviewed_real_train'] += 1
    value = {'status': 'retained_negative_candidates', 'policy': policy,
             'counts': dict(counts), 'records': records, 'decisions': decisions,
             'scope': 'Original samples retained unchanged; source assets and original split remain in the existing source gate. '
                      'All 005751 composites with a vehicle and all 003 composite evaluation images are withheld for suspended-grounding defects. '
                      'Twelve direct-frame and tone variants of four AI-reviewed parents share one video and are assigned to train only.'}
    (out / 'manifest.json').write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'output': rel(out), 'counts': dict(counts), 'retained': len(records)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
