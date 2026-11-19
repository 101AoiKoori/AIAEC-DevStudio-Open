"""Join the CarRealPicture observations and task labels by their existing image paths."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.scripts.common.config import load_task_config, read_yaml


BATCH = 'car_real_capture_20261112'
RAW = f'Archive/Assets/captures/{BATCH}'
ANNOTATED = f'datasets/annotated/{BATCH}'
ORIGINALS = 'Archive/Assets/CarRealPicture'
PARENT = f'{RAW}/V_20261112_152304.mp4'
PERSON = f'{ANNOTATED}/person_direct_captures.json'
ALIGNMENT = f'{ANNOTATED}/waypoint_video_alignment.json'
OUTPUT = f'{ANNOTATED}/direct_real_manifest.json'
SOURCE_COPIES = 'datasets/source_copy_manifest.jsonl'
SEGMENTS = f'{ANNOTATED}/segments_v1/segments.json'
TARGETS = 'config/task_targets.yaml'

# Transcribed from the printed paper in each original 640x480 waypoint photo.
PAPER_TEXT = {'P09_B1': '晨光大厦', 'P10_B2': '开心大厦', 'P24_B3': '锦绣大厦'}
STATES = {'visual_candidate': 'complete_positive', 'visual_negative': 'complete_negative',
          'partial': 'partial', 'partial_unlabeled': 'partial', 'unknown': 'unknown'}


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8-sig'))


def lines(name):
    with (ROOT / name).open(encoding='utf-8-sig') as file:
        for index, line in enumerate(file):
            if line.strip():
                yield index, json.loads(line)


def task_key(name):
    return 'fire' if name == 'fire_sticker' else name


def task_state(instances):
    return 'partial' if instances else 'unknown'


def instance(box, names):
    result = dict(box)
    if 'bbox_xyxy' in result:
        result['xyxy'] = result.pop('bbox_xyxy')
    cls = box.get('class_id')
    if type(cls) is int and cls in range(len(names)):
        result.setdefault('class_name', str(names[cls]))
    return result


def seed():
    segments = {frame: segment for segment in read(SEGMENTS)['segments']
                for frame in range(segment['start_frame'], segment['end_frame'] + 1)}
    rows, canonical = {}, {}
    for index, copy in lines(SOURCE_COPIES):
        path, original = copy['path'], copy.get('original_path', '')
        if not path.startswith(RAW + '/') or Path(path).suffix.lower() not in {'.jpg', '.png'}:
            continue
        if '/train/' in path:
            continue
        stem = Path(path).stem
        frame = int(stem.rsplit('_', 1)[-1]) if '/frames_' in path else None
        rows[path] = {'id': f'{BATCH}:{stem}', 'image': path, 'original_path': original,
                      'parent_video': PARENT, 'source_group': BATCH, 'image_size': [640, 480],
                      'point': stem.split('_', 1)[0] if frame is None else None,
                      'segment': segments[frame]['segment_id'] if frame else None,
                      'frame_index': frame, 'timestamp_seconds': None, 'tasks': {},
                      'evidence': {'source_copy': {'manifest': SOURCE_COPIES, 'record_index': index},
                                   'video_relation': 'same capture batch; exact timestamp unlocated',
                                   'segment_manifest': SEGMENTS if frame else None}}
        canonical[original] = path
    if len(rows) != len(segments) + 18:
        raise ValueError('CarRealPicture source copy coverage differs from its frame and waypoint inventory')
    return rows, canonical


def add_observations(rows, targets):
    for filename in ('target_instances_manifest.jsonl', 'sequence_instances_part1_manifest.jsonl',
                     'sequence_instances_part2_manifest.jsonl'):
        source = f'{ANNOTATED}/{filename}'
        for index, observation in lines(source):
            row = rows.get(observation['path'])
            if row is None:
                continue
            tasks = defaultdict(list)
            reviews = observation.get('task_review', observation.get('task_observations', {}))
            for key, review in reviews.items():
                tasks[task_key(key)].extend(review.get('instances', []) if isinstance(review, dict) else [])
            for box in observation.get('observed_instances', observation.get('instances', [])):
                tasks[task_key(box['task'])].append(box)
            for task in targets.keys() & tasks.keys():
                boxes = [instance(box, targets[task]['classes']) for box in tasks[task]]
                row['tasks'][task] = {'state': task_state(boxes), 'label_level': 'auto_candidate' if boxes else 'unlabeled',
                                      'uses': ['visual_reference'], 'instances': boxes,
                                      'evidence': {'manifest': source, 'record_index': index}}


def field_truth(annotation, names, evidence, *, text=None):
    task = task_key(annotation['model_key'])
    state = STATES[annotation['review_status']]
    boxes = [instance(b, names) for b in annotation['boxes']]
    truth = {'state': state, 'label_level': 'auto_candidate', 'instances': boxes,
             'uses': ['development_diagnostic'], 'evidence': evidence}
    if detection_only := (state == 'complete_positive' and
                          (task == 'meter' or (task == 'building_name_paper' and not text))):
        truth.update(state='partial', detection_state=state)
        truth['missing_supervision'] = 'pose_keypoints' if task == 'meter' else 'legible_ocr_text'
    if text:
        boxes[0]['text'] = text
        truth['ocr_evidence'] = 'visual transcription of the paper in the native waypoint image'
    if not detection_only and state in {'complete_positive', 'complete_negative'}:
        truth['uses'].append('direct_sample')
    return truth


def add_field_labels(rows, canonical, targets):
    for filename in ('annotations_root.json', 'annotations_frames.json'):
        source = f'{ANNOTATED}/field_eval_v1/{filename}'
        for index, annotation in enumerate(read(source)):
            task = task_key(annotation['model_key'])
            if task == 'person' or task not in targets:
                continue
            image = canonical.get(annotation['image'], annotation['image'])
            row = rows[image]
            text = PAPER_TEXT.get(Path(image).stem) if task == 'building_name_paper' and annotation['boxes'] else None
            truth = field_truth(annotation, targets[task]['classes'], {'manifest': source, 'record_index': index}, text=text)
            if prior := row['tasks'].get(task):
                truth['prior_observation'] = prior['evidence']
            row['tasks'][task] = truth


def add_sticker_labels(rows, targets):
    _, sources = load_task_config('waste_sticker')
    path = sources['real_annotations']
    selected = set(sources['current_selection']['direct_images'])
    for index, annotation in enumerate(read(path)):
        image = annotation['image']
        truth = field_truth(annotation, targets['waste_sticker']['classes'],
                            {'manifest': path, 'record_index': index})
        if image in selected:
            selection = next(x for x in annotation['task_use_status'] if x['role'] == 'direct_sample')
            truth.update(split=selection['split'], split_basis=selection['split_basis'], label=selection['label'])
        else:
            truth['uses'] = ['development_diagnostic']
        if annotation.get('unknown_regions'):
            truth['unknown_regions'] = annotation['unknown_regions']
        rows[image]['tasks']['waste_sticker'] = truth


def add_waypoint_alignment(rows):
    if not (ROOT / ALIGNMENT).is_file():
        return
    for index, entry in enumerate(read(ALIGNMENT)['records']):
        waypoint = rows[entry['image']]
        waypoint['evidence']['waypoint_alignment'] = {'manifest': ALIGNMENT, 'record_index': index}
        if not entry.get('matched_image'):
            continue
        frame = rows[entry['matched_image']]
        for key in ('frame_index', 'timestamp_seconds', 'segment'):
            waypoint[key] = entry.get(key)
        for row in (waypoint, frame):
            row['point'] = waypoint['point']
            row['waypoint_label'] = Path(waypoint['image']).stem
            row['timestamp_seconds'] = entry.get('timestamp_seconds')
            row['evidence']['waypoint_alignment'] = waypoint['evidence']['waypoint_alignment']
            row['evidence']['video_relation'] = ('content-aligned MP4 frame' if entry.get('timestamp_seconds') is not None
                                                 else 'same capture batch; exact timestamp unlocated')
            row['capture_identity'] = waypoint['capture_identity'] = f'{BATCH}:jpeg:{entry["frame_index"]}'
        frame['duplicate_of'] = waypoint['image']


def assemble():
    targets = read_yaml(ROOT / TARGETS)['tasks']
    rows, canonical = seed()
    add_observations(rows, targets)
    add_field_labels(rows, canonical, targets)
    add_sticker_labels(rows, targets)
    add_waypoint_alignment(rows)
    person = read(PERSON)
    for addition in person['records']:
        if not addition.get('image'):
            continue
        row = rows[addition['image']]
        row['tasks'].update(addition['tasks'])
        row['evidence'].update(addition.get('evidence', {}))
        for key, value in addition.items():
            if key not in {'tasks', 'evidence'} and value is not None:
                row[key] = value
        if row.get('point'):
            row['waypoint_label'] = row['point']
            row['point'] = row['point'].split('_', 1)[0]
        if row.get('timestamp_seconds') is not None:
            row['evidence']['video_relation'] = 'content-aligned MP4 frame'
    records = sorted(rows.values(), key=lambda row: (row['frame_index'] or 0, row['image']))
    states = {task: dict(Counter(row['tasks'][task]['state'] for row in records if task in row['tasks']))
              for task in sorted(targets)}
    return {'schema_version': 1, 'source_group': BATCH, 'parent_video': PARENT,
            'person_capture_catalog': PERSON, 'segments_manifest': SEGMENTS,
            'counts_by_task': states, 'records': records}


def main():
    result = assemble()
    output = ROOT / OUTPUT
    with TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        with staged.open('w', encoding='utf-8') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
        staged.replace(output)
    print(json.dumps(result['counts_by_task'], ensure_ascii=False))


if __name__ == '__main__':
    main()
