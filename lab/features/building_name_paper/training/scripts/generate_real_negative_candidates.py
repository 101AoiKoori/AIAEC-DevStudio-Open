"""Create planned, unverified name-negative candidates from real signal views."""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/building_name_paper/. Production remains paused.")

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
FIRE = ROOT / 'lab/features/fire/training/scripts'
sys.path[:0] = [str(ROOT), str(FIRE)]
from compose_scene_fire import read_image  # noqa: E402
from lab.training_common.production_guard import generation_guard, plan_production  # noqa: E402

EVIDENCE = ROOT / 'datasets/exports/authoritative/fire_sticker/production/signal_negative_plan.json'


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def declaration(path):
    path = (ROOT / path).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError('plan must be an existing workspace file')
    source_plan = json.loads(EVIDENCE.read_text(encoding='utf8'))
    sources = {s['id']: s for s in source_plan['strata'] if 'no_visible_building' in s['conditions']}
    request = json.loads(path.read_text(encoding='utf8'))
    if request.get('model_key') != 'building_name_paper' or request.get('target_split') != 'train':
        raise ValueError('expected a building-name candidate plan')
    if request.get('constraints') != {'background_review': 'ai_candidate_not_human_accepted',
                                      'no_name_truth': 'candidate_not_confirmed'}:
        raise ValueError('unsupported source or label policy')
    for state in request.get('strata', []):
        if not isinstance(state.get('id'), str) or not re.fullmatch(r'[A-Za-z0-9_-]+', state['id']):
            raise ValueError('state id must be a safe file stem')
        source_id = state.get('background', {}).get('source_state')
        if source_id not in sources:
            raise ValueError('source state lacks no-building observation')
        source = sources[source_id]
        image = source['background']['image']
        if state.get('background') != {'source_state': source_id, 'image': image}:
            raise ValueError('source image differs from evidence')
        if state.get('target') != {'kind': 'building_name_paper', 'visibility': 'negative_candidate'}:
            raise ValueError('unsupported target state')
        if state.get('labels') != {'class_id': 0, 'format': 'yolo_v8', 'status': 'negative_candidate_unverified'}:
            raise ValueError('unsupported label state')
        gain = state.get('augmentation', {}).get('brightness_gain')
        if state.get('augmentation') != {'brightness_gain': gain} or gain not in (0.85, 1.0, 1.15):
            raise ValueError('unsupported brightness')
        if state.get('sources') != source['sources'] or state.get('parameters') != {'image': image}:
            raise ValueError('source roles or image differ')
        if state.get('conditions') != ['no_visible_building_candidate', 'name_absence_unverified']:
            raise ValueError('unsupported source conditions')
        if state.get('failure_event') != 'unrecognized_name_or_background_uncertainty':
            raise ValueError('unsupported uncertainty')
        if state.get('min_count') != 1 or state.get('max_count') != 1:
            raise ValueError('each state generates one image')
    return path, request


def generate(output, run):
    raise RuntimeError("Retired output API; use datasets/scripts/building_name_paper/compose_rooftop_papers.py")


def _legacy_generate(output, run):
    if output.exists():
        raise ValueError('output already exists')
    (output / 'images').mkdir(parents=True)
    (output / 'labels_candidate').mkdir()
    records = []
    for state in run.plan['strata']:
        if state['planned_count'] != 1:
            raise ValueError('state allocation differs from plan')
        source = ROOT / state['background']['image']
        gain = state['augmentation']['brightness_gain']
        image = np.clip(read_image(source).astype(np.float32) * gain, 0, 255).astype(np.uint8)
        image_out = output / 'images' / (state['id'] + '.png')
        label_out = output / 'labels_candidate' / (state['id'] + '.txt')
        ok, encoded = cv2.imencode('.png', image)
        if not ok:
            raise ValueError('image encoding failed')
        encoded.tofile(str(image_out))
        label_out.write_text('', encoding='utf8')
        actual = {'source_image': relative(source), 'brightness_gain': gain,
                  'source_evidence': relative(EVIDENCE),
                  'name_absence': 'candidate_not_confirmed', 'label_status': 'negative_candidate_unverified'}
        records.append({'image': relative(image_out), 'label': relative(label_out),
                        'production_state_id': state['id'], **actual})
        run.record_sample(state['id'], output=image_out, sources=state['sources'],
                          parameters=state['parameters'], actual=actual)
    result = {'status': 'negative_candidates_not_training_approved',
              'generation_method': 'real_camera_image_brightness_augmentation',
              'scope': 'Name absence is an unconfirmed candidate; augmented views share their parents.',
              'count': len(records), 'records': records}
    (output / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to((ROOT / 'runs/building_name_paper').resolve()):
        raise ValueError('output outside building-name mainline')
    plan_path, request = declaration(args.plan)
    planned = plan_production(request, ROOT)
    images = [ROOT / path for path in sorted({s['parameters']['image'] for s in request['strata']})]
    metadata = [plan_path, EVIDENCE]
    code = [Path(__file__), FIRE / 'compose_scene_fire.py', ROOT / 'lab/training_common/bayesian_allocation.py',
            ROOT / 'datasets/scripts/common/images.py', ROOT / 'datasets/scripts/common/annotations.py']
    with generation_guard('building_name.real_negative:' + relative(output), images + metadata,
                          [output], {}, code_paths=code, procedural_inputs=metadata, root=ROOT,
                          production_plan=request, sample_output_paths=[output / 'images']) as run:
        if run.plan != planned:
            raise ValueError('guard plan differs from requested allocation')
        if not run.reused:
            generate(output, run)
    print(json.dumps({'planned': planned['planned_total'], 'output': relative(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
