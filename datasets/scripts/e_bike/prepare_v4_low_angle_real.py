"""Export explicitly planned tone variants of reviewed low-camera frames."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'lab/training_common')]
from camera_tone import apply_tone, read_image, source_role, write_image  # noqa: E402
from lab.scripts.data.v4_source_gate import build_queue  # noqa: E402
from datasets.scripts.common.config import source_path
from datasets.scripts.common.paths import sample_dir, split_dir
from lab.training_common.production_guard import generation_guard  # noqa: E402
from datasets.scripts.e_bike.e_bike_queue_inputs import queue_input_report

REVIEW = source_path('e_bike', 'direct_real_review_manifest').parent
TONES = {'original': (1., 1.), 'mild': (.9, .85), 'moderate': (.7, .6)}
LABELS = {'format': 'yolo_visible_box', 'classes': {'upright': 0, 'fallen': 1}}
CONSTRAINTS = {'complete_candidate_only': True, 'max_width': 640, 'max_height': 480}
TOP_FIELDS = {'schema_version', 'model_key', 'target_split', 'purpose', 'budget',
              'uncertainty_weight', 'independence_basis', 'constraints', 'strata'}
STATE_FIELDS = {'id', 'failure_event', 'min_count', 'max_count', 'prior', 'exposure_weight',
                'failure_cost', 'conditions', 'background', 'target', 'labels', 'augmentation',
                'parameters', 'sources'}


def relative(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('path is outside the workspace')
    return path.relative_to(ROOT).as_posix()


def validate_declaration(declaration):
    if not isinstance(declaration, dict) or set(declaration) != TOP_FIELDS:
        raise ValueError('unsupported or missing declaration fields')
    if (declaration['model_key'] != 'e_bike' or declaration['target_split'] != 'train' or
            declaration['constraints'] != CONSTRAINTS or
            type(declaration['constraints'].get('complete_candidate_only')) is not bool):
        raise ValueError('unsupported model, split, or constraints')
    if type(declaration['budget']) is not int or not 1 <= declaration['budget'] <= 12:
        raise ValueError('budget must be between 1 and 12 distinct reviewed tone states')
    if not isinstance(declaration['strata'], list) or not declaration['strata']:
        raise ValueError('a nonempty state plan is required')
    signatures = set()
    for state in declaration['strata']:
        if not isinstance(state, dict) or set(state) != STATE_FIELDS or state['conditions'] != []:
            raise ValueError('unsupported state fields or conditions')
        if type(state['max_count']) is not int or state['max_count'] > 1:
            raise ValueError('each exact state can generate at most one image')
        key = json.dumps({name: state[name] for name in
                          ('background', 'target', 'labels', 'augmentation', 'parameters')},
                         ensure_ascii=False, sort_keys=True, allow_nan=False)
        if key in signatures:
            raise ValueError('duplicate exact generation states are not allowed')
        signatures.add(key)


def check_state(state, rows, queue):
    """Check declared semantics against existing review records without generating pixels."""
    if not isinstance(state['background'], dict) or set(state['background']) != {'image', 'crop_xyxy'}:
        raise ValueError('unsupported background fields')
    matches = [row for row in rows if row['image'] == state['background']['image']]
    if len(matches) != 1:
        raise ValueError('background must identify exactly one reviewed frame')
    row = matches[0]
    if row['split'] != 'train_candidate' or 'complete_candidate' not in row['label_status']:
        raise ValueError('source truth is not reviewed as complete')
    for source in (row['source_frame'], row['source_video']):
        if source_role(ROOT, ROOT / source, queue)[0] == 'regression':
            raise ValueError('source is restricted to regression: ' + source)
    if state['background']['crop_xyxy'] != row['crop_xyxy']:
        raise ValueError('declared crop differs from the reviewed crop')
    classes = [box['class'] for box in row['boxes']]
    if any(cls not in LABELS['classes'] for cls in classes) or state['target'] != {'classes': classes}:
        raise ValueError('declared targets differ from reviewed box classes')
    if state['labels'] != LABELS:
        raise ValueError('unsupported label contract')
    augmentation = state['augmentation']
    if not isinstance(augmentation, dict) or set(augmentation) != {'tone'} or augmentation['tone'] not in TONES:
        raise ValueError('unsupported tone')
    gain, saturation = TONES[augmentation['tone']]
    if state['parameters'] != {'value_gain': gain, 'saturation_gain': saturation}:
        raise ValueError('tone parameters do not match the chosen tone')
    actual_sources = sorted({row['image'], row['source_frame'], row['source_video']})
    if state['sources'] != actual_sources:
        raise ValueError('sources must list the sorted actual image, frame and video paths')
    width, height = row['output_image_size']
    if not 0 < width <= 640 or not 0 < height <= 480:
        raise ValueError('reviewed dimensions outside supported range')
    expected = []
    for box in row['boxes']:
        a, b, c, d = box['xyxy']
        if not 0 <= a < c <= width or not 0 <= b < d <= height:
            raise ValueError('box outside source image')
        cls = LABELS['classes'][box['class']]
        expected.append(f'{cls} {(a+c)/(2*width):.8f} {(b+d)/(2*height):.8f} '
                        f'{(c-a)/width:.8f} {(d-b)/height:.8f}')
    if (ROOT / row['label']).read_text(encoding='utf-8').splitlines() != expected:
        raise ValueError('label differs from reviewed boxes: ' + row['id'])
    return row, expected


def allocated_states(plan):
    states = plan['strata']
    if any(type(state.get('planned_count')) is not int or state['planned_count'] not in (0, 1)
           for state in states):
        raise ValueError('allocation exceeds the one-image-per-exact-state limit')
    return states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, help='Workspace-relative production declaration')
    parser.add_argument('--output', required=True, help='New e-bike batch directory')
    args = parser.parse_args()
    plan_path = (ROOT / args.plan).resolve()
    out = (ROOT / args.output).resolve()
    relative(plan_path)
    declaration = json.loads(plan_path.read_text(encoding='utf-8-sig'))
    validate_declaration(declaration)
    if out not in {sample_dir('e_bike').resolve(), split_dir('e_bike', declaration['target_split']).resolve()}:
        raise ValueError('output must be the sample or matching split')
    review_path = REVIEW / 'review_manifest.json'
    rows = json.loads(review_path.read_text(encoding='utf-8'))['rows']
    if len(rows) != 4:
        raise ValueError('expected the four reviewed camera frames')
    checked = [ROOT / value for row in rows for value in (row['source_frame'], row['source_video'])]
    media = [ROOT / row['image'] for row in rows] + checked
    procedural = [plan_path, review_path, ROOT/'datasets/scripts/e_bike/config/source.yaml', ROOT/'datasets/scripts/e_bike/config/dataset.yaml'] + [ROOT / row['label'] for row in rows]
    queue = build_queue(ROOT)
    for state in declaration['strata']:
        check_state(state, rows, queue)
    source_roles = {row['id']: {key: source_role(ROOT, ROOT / row[key], queue)
                                for key in ('source_frame', 'source_video')} for row in rows}
    extra_media, extra_procedural, missing_inputs = queue_input_report(ROOT, queue, checked)
    media = list(dict.fromkeys(media + extra_media))
    procedural = list(dict.fromkeys(procedural + extra_procedural))
    history = {'exposure_status': 'incomplete' if missing_inputs or queue.get('missing_exposure_paths') or queue.get('unavailable_exposure_paths')
               or queue.get('exposure_status') == 'incomplete' else 'complete',
               'missing_input_paths': missing_inputs,
               'missing_exposure_paths': queue.get('missing_exposure_paths', []),
               'unavailable_exposure_paths': queue.get('unavailable_exposure_paths', [])}
    code = [__file__, ROOT / 'lab/training_common/camera_tone.py', ROOT / 'lab/scripts/data/v4_source_gate.py',
            ROOT / 'lab/training_common/train.py', ROOT / 'lab/training_common/engine/data.py',
            ROOT / 'lab/training_common/engine/config.py', Path(__file__).with_name('e_bike_queue_inputs.py')]
    with generation_guard('e_bike.low_angle_real:' + relative(out), media, [out],
                          {'plan': relative(plan_path), 'historical_exposure': history},
                          procedural_inputs=procedural, code_paths=code, root=ROOT,
                          production_plan=declaration, sample_output_paths=[out / 'images']) as run:
        if run.reused:
            return
        if out.exists() and any(out.iterdir()):
            raise ValueError('output exists; clear the outdated current result before producing')
        states = allocated_states(run.plan)
        # Decode and check every source before creating the output directory.
        for state in states:
            row, _ = check_state(state, rows, queue)
            image = read_image(ROOT / row['image'])
            if [image.shape[1], image.shape[0]] != row['output_image_size']:
                raise ValueError('actual image dimensions differ from review')
        (out / 'images').mkdir(parents=True)
        (out / 'labels').mkdir(parents=True)
        records, counts = [], Counter()
        for index, state in enumerate(item for item in states if item['planned_count']):
            row, expected = check_state(state, rows, queue)
            image = read_image(ROOT / row['image'])
            tone = state['augmentation']['tone']
            gain, saturation = TONES[tone]
            frame = image if tone == 'original' else apply_tone(image, value_gain=gain, saturation_gain=saturation)
            name = f'bike_{index + 1:06d}'
            img, label = out / 'images' / (name + '.png'), out / 'labels' / (name + '.txt')
            write_image(img, frame)
            label.write_text('\n'.join(expected) + ('\n' if expected else ''), encoding='utf-8')
            actual = {'image': row['image'], 'source_frame': row['source_frame'], 'source_video': row['source_video'],
                      'historical_source_roles': source_roles[row['id']],
                      'crop_xyxy': row['crop_xyxy'], 'boxes': row['boxes'], 'tone': tone,
                      'value_gain': gain, 'saturation_gain': saturation, 'image_size': row['output_image_size']}
            run.record_sample(state['id'], output=img, sources=state['sources'],
                              parameters=state['parameters'], actual=actual)
            records.append({'image': relative(img), 'label': relative(label), 'state_id': state['id'],
                            'split': 'train_candidate', 'parent_image': row['source_frame'],
                            'reviewed_crop': row['image'], 'crop_xyxy': row['crop_xyxy'],
                            'source_video': row['source_video'], 'source_group': row['source_group'],
                            'historical_source_roles': source_roles[row['id']],
                            'source_frame_index': row['source_frame_index'], 'source_pts_seconds': row['source_pts_seconds'],
                            'physical_instances': row['physical_instances'], 'boxes': row['boxes'],
                            'label_review': row['label_status'], 'tone': tone, 'tone_parameters': state['parameters'],
                            'generation_method': 'video_frame_and_crop' if tone == 'original' else 'video_frame_crop_and_tone_augmentation',
                            'image_size': row['output_image_size'], 'reflection_policy': row['reflection_policy']})
            counts['tone_' + tone] += 1
            for box in row['boxes']:
                counts['instances_' + box['class']] += 1
        manifest = {'status': 'training_candidates_not_deliverable', 'review_manifest': relative(review_path),
                    'historical_exposure': history,
                    'plan': relative(plan_path), 'counts': dict(counts), 'records': records,
                    'scope': 'Tone variants are derivatives, not independent captures.'}
        (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'output': relative(out), 'images': len(records), 'counts': dict(counts)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
