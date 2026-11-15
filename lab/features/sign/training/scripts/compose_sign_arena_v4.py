"""Compose explicitly planned sign candidates on Blender or real backgrounds."""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/sign/. Production remains paused.")

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import compose_sign_scene as base
import compose_sign_v4_measured as measured
from sign_production_guard import guarded, existing_result

ROOT = base.ROOT
BACKGROUNDS = ROOT / 'datasets/exports/sign/v4_arena_backgrounds_20261113'
MASKS = ROOT / 'datasets/annotated/sign/sam2_fixture_candidates_20261110'
REPAIRED = ROOT / 'datasets/exports/sign/v4_measured_multibackground_train_20261112/back_mask_repaired.png'
REAL_BACKGROUNDS = ROOT / 'datasets/annotated/sign/v4_training_backgrounds_20261112'
OUTPUT_ROOT = ROOT / 'runs/sign'


def crop_alpha(path):
    image = base.read_image(path, cv2.IMREAD_UNCHANGED)
    if image is None or image.shape[2] != 4:
        raise ValueError(f'Expected RGBA foreground: {path}')
    ys, xs = np.where(image[:, :, 3] > 127)
    if not len(xs):
        raise ValueError(f'Empty foreground: {path}')
    return image[ys.min():ys.max()+1, xs.min():xs.max()+1]


def _backgrounds(bg_data, real, background_root=BACKGROUNDS):
    rows = []
    for item in bg_data['samples']:
        image = (background_root / item['image']).resolve().relative_to(ROOT).as_posix()
        rows.append({'image': image, 'road_anchor_center_x': [140, 500],
                     'parent': bg_data['source_scene'], 'time': None, 'kind': 'blender_render',
                     'absence': 'unknown', 'sign_absence': item.get('sign_absence'),
                     'sign_absence_evidence': item.get('sign_absence_evidence')})
    if real is not None:
        if real.get('status') != 'AI_reviewed_sign_empty_candidates':
            raise ValueError('Expected reviewed real empty-background candidates')
        for item in real['frames']:
            if item.get('usage') == 'unassigned':
                image = (REAL_BACKGROUNDS / item['image']).resolve().relative_to(ROOT).as_posix()
                rows.append({'image': image, 'road_anchor_center_x': item['road_anchor_center_x'],
                             'parent': item['parent_image'], 'video': real['video'],
                             'time': item['pts_seconds'], 'kind': 'real_scene_composite',
                             'absence': 'candidate_reviewed', 'sign_absence': item.get('sign_absence'),
                             'sign_absence_evidence': item.get('sign_absence_evidence')})
    return rows


def _prepare(declaration, backgrounds, mask_data):
    if not isinstance(declaration, dict) or declaration.get('model_key') != 'sign':
        raise ValueError('Explicit sign production plan is required')
    if declaration.get('constraints') != {'image_size': [640, 480], 'label_kind': 'visible_fixture_box'}:
        raise ValueError('Unsupported sign composition constraints')
    states = declaration.get('strata')
    if not isinstance(states, list) or not states:
        raise ValueError('A sign production plan with states is required')
    prepared, seen = {}, set()
    for state in states:
        row = measured.validate_planned_stratum(state, backgrounds)
        if type(state.get('max_count')) is not int or not 0 <= state['max_count'] <= 1:
            raise ValueError('Each exact-placement sign state permits at most one sample')
        mask = state['target']['mask']
        view = dict(measured.VIEWS)[mask]
        if set(state.get('conditions', [])) != {state['target']['state'], view, state['augmentation']['tone']}:
            raise ValueError('Conditions must describe sign state, view and tone')
        origins = [r for r in mask_data['records'] if 'index' in r and f"train_{r['index']}.png" == mask]
        if len(origins) != 1:
            raise ValueError('Foreground mask source is absent or ambiguous')
        mask_path = REPAIRED if view == 'back' else MASKS / mask
        values = [row['image'], row['parent']]
        if row.get('video'):
            values.append(row['video'])
        values += [mask_path.relative_to(ROOT).as_posix(), origins[0]['source'], base.SIGN_VIDEO]
        expected = list(dict.fromkeys(values))
        if state.get('sources') != expected:
            raise ValueError('Declared sources do not match selected inputs')
        identity = json.dumps([state['background'], state['target'], state['augmentation'], state['parameters']],
                              sort_keys=True)
        if identity in seen or state['id'] in prepared:
            raise ValueError('Duplicate sign composition state')
        seen.add(identity)
        prepared[state['id']] = (row, origins[0], mask_path)
    return prepared


def _generate(output, plan, run, prepared, sampling_seed=None):
    output = Path(output).resolve()
    if not output.is_relative_to(OUTPUT_ROOT.resolve()) or output == OUTPUT_ROOT.resolve() or output.exists():
        raise ValueError('Output must be a new directory under runs/sign')
    rendered = {}
    for state in plan['strata']:
        if not state['planned_count']:
            continue
        row, origin, mask_path = prepared[state['id']]
        if state['target']['state'] == 'placed':
            measured.validate_planned_stratum(state, [row])
        frame = base.read_image(ROOT / row['image'], cv2.IMREAD_COLOR)
        if frame is None or frame.shape != (480, 640, 3):
            raise ValueError('Expected 640x480 BGR background')
        sprite = crop_alpha(mask_path)
        image, box, geometry = measured.compose(frame, sprite, state['target']['state'],
                                                 state['augmentation']['tone'], None,
                                                 row.get('road_anchor_center_x'), state['parameters'])
        rendered[state['id']] = (row, origin, mask_path, image, box, geometry)
    (output / 'images').mkdir(parents=True)
    (output / 'labels_candidate').mkdir()
    result = {'status': 'running', 'class': 'sign_fixture', 'target_split': plan['target_split'],
              'scope': 'shared foreground and background sources; labels remain candidates', 'samples': []}
    try:
        for state in plan['strata']:
            if not state['planned_count']:
                continue
            row, origin, mask_path, image, box, geometry = rendered[state['id']]
            name = f"sign_arena_{len(result['samples']):04d}"
            path = output / 'images' / (name + '.png')
            label = output / 'labels_candidate' / (name + '.txt')
            x1, y1, x2, y2 = box
            measured.write_png(path, image)
            label.write_text(f'0 {(x1+x2)/1280:.8f} {(y1+y2)/960:.8f} {(x2-x1)/640:.8f} {(y2-y1)/480:.8f}\n', encoding='utf8')
            actual = {'background': row['image'], 'background_parent': row['parent'],
                      'background_video': row.get('video'), 'background_time_seconds': row['time'],
                      'background_kind': row['kind'], 'background_sign_absence': row['absence'],
                      'foreground_mask': mask_path.relative_to(ROOT).as_posix(),
                      'foreground_source': origin['source'], 'foreground_video': base.SIGN_VIDEO,
                      'view': dict(measured.VIEWS)[state['target']['mask']], 'state': state['target']['state'],
                      'tone': state['augmentation']['tone'], 'bbox_xyxy': box, 'geometry': geometry,
                      'sampling_seed': sampling_seed}
            if state['target']['state'] == 'placed':
                actual['background_sign_absence'] = row['sign_absence']
                actual['background_sign_absence_evidence'] = row['sign_absence_evidence']
            run.record_sample(state['id'], output=path, sources=state['sources'],
                              parameters=state['parameters'], actual=actual)
            result['samples'].append({'id': name, 'state_id': state['id'], 'image': 'images/' + name + '.png',
                                      'label': 'labels_candidate/' + name + '.txt', 'sources': state['sources'],
                                      'actual': actual, 'review_status': 'candidate'})
        result['status'] = 'candidates_pending_review'
    except BaseException as exc:
        result.update(status='failed_or_interrupted', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (output / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    return result


def main():
    raise RuntimeError("Retired dataset entry; use datasets/scripts task entry and its source.yaml")


if __name__ == '__main__':
    main()
