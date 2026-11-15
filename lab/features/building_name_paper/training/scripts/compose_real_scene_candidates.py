"""Compose planned name-paper candidates on existing real-image backgrounds."""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/building_name_paper/. Production remains paused.")

import argparse
import json
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
FIRE = ROOT / 'lab/features/fire/training/scripts'
sys.path[:0] = [str(ROOT), str(FIRE)]
from compose_scene_fire_eval import BACKGROUND, FONT, SOURCES, name_paper, paste  # noqa: E402
from compose_scene_fire import read_image  # noqa: E402
from fire_blender_labels import yolo_row  # noqa: E402
from vary_name_papers import CHARS, SUFFIX, make_name  # noqa: E402
from lab.training_common.production_guard import generation_guard, plan_production  # noqa: E402

SOURCE_MANIFEST = ROOT / 'datasets/annotated/fire/real_review_candidates_20261110/manifest.json'


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def declaration(path):
    path = Path(path)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError('plan must be an existing workspace file')
    request = json.loads(path.read_text(encoding='utf8'))
    if request.get('model_key') != 'building_name_paper' or request.get('target_split') != 'train':
        raise ValueError('expected a building-name training candidate plan')
    if request.get('constraints') != {'existing_background_targets': 'unknown_partial',
                                      'ocr_truth': 'not_verified', 'text_mirror': False}:
        raise ValueError('unsupported source or OCR constraints')
    for state in request.get('strata', []):
        if not isinstance(state.get('id'), str) or not re.fullmatch(r'[A-Za-z0-9_-]+', state['id']):
            raise ValueError('state id must be a safe file stem')
        target, background = state.get('target', {}), state.get('background', {})
        key = background.get('source_key')
        if key not in SOURCES:
            raise ValueError('unsupported background')
        source = SOURCES[key]
        image = BACKGROUND / source['image']
        if background != {'source_key': key, 'source_image': relative(image),
                          'source_group': source['source_group']}:
            raise ValueError('background does not match source')
        exact_text = set(target) == {'kind', 'text', 'quad_xy'}
        random_text = set(target) == {'kind', 'text_policy', 'name_index_range', 'quad_xy'}
        if target.get('kind') != 'building_name_paper' or target.get('quad_xy') != source['name_quad']:
            raise ValueError('unsupported paper geometry')
        if exact_text:
            text = target['text']
            if not isinstance(text, str) or not text.strip() or len(text) > 8 or any(ch in text for ch in '\r\n\t'):
                raise ValueError('unsupported exact text')
        elif random_text:
            if (target['text_policy'] != 'unique_seeded_generator_indices' or
                    target['name_index_range'] != [0, len(CHARS) * len(CHARS) * len(SUFFIX)]):
                raise ValueError('unsupported text distribution')
        else:
            raise ValueError('unsupported text policy')
        if state.get('labels') != {'task': 'paper_detection', 'class_id': 0,
                                   'format': 'yolo_v8', 'status': 'partial_candidate',
                                   'ocr_status': 'not_verified'}:
            raise ValueError('background has unknown other targets; only partial labels are supported')
        augmentation = state.get('augmentation', {})
        if exact_text:
            gain = augmentation.get('brightness_gain')
            if augmentation != {'brightness_gain': gain, 'output_size': [960, 540]} or gain not in (0.85, 1.0):
                raise ValueError('unsupported fixed augmentation')
        elif augmentation != {'brightness_gain_range': [0.85, 1.05], 'distribution': 'uniform',
                              'output_size': [960, 540]}:
            raise ValueError('unsupported brightness distribution')
        if state.get('sources') != ['real_scene_background', 'name_font_asset']:
            raise ValueError('unsupported source roles')
        parameters = {'source_image': relative(image), 'source_key': key}
        if random_text:
            seed = state.get('parameters', {}).get('seed')
            if type(seed) is not int or not 0 <= seed < 2**63:
                raise ValueError('a bounded integer seed is required')
            parameters['seed'] = seed
        if state.get('parameters') != parameters:
            raise ValueError('source parameters differ')
        if state.get('conditions') != ['background_other_targets_unknown', 'text_not_mirrored']:
            raise ValueError('unsupported conditions')
        if state.get('failure_event') != 'paper_or_background_label_uncertainty':
            raise ValueError('unsupported uncertainty')
        if exact_text and (state.get('min_count') != 1 or state.get('max_count') != 1):
            raise ValueError('each exact state generates one image')
    modes = {'text_policy' in state['target'] for state in request.get('strata', [])}
    if len(modes) != 1:
        raise ValueError('fixed and random text states cannot be mixed in a batch')
    if modes == {True} and len({s['parameters']['seed'] for s in request['strata']}) != 1:
        raise ValueError('all randomized states must use the same batch seed')
    return path, request


def generate(output, run):
    raise RuntimeError("Retired output API; use datasets/scripts/building_name_paper/compose_rooftop_papers.py")


def _legacy_generate(output, run):
    if output.exists():
        raise ValueError('output already exists')
    (output / 'images').mkdir(parents=True)
    (output / 'labels_partial').mkdir()
    records = []
    schedule = [(state, index) for state in run.plan['strata'] for index in range(state['planned_count'])]
    randomized = bool(schedule and 'text_policy' in schedule[0][0]['target'])
    if randomized:
        seed = schedule[0][0]['parameters']['seed']
        rng = random.Random(seed)
        rng.shuffle(schedule)
        indices = rng.sample(range(len(CHARS) * len(CHARS) * len(SUFFIX)), len(schedule))
        sample_seeds = rng.sample(range(2**62), len(schedule))
    for sample_index, (state, state_index) in enumerate(schedule):
        key = state['background']['source_key']
        cfg = SOURCES[key]
        image_path = BACKGROUND / cfg['image']
        image = cv2.resize(read_image(image_path), (960, 540))
        sample_seed = sample_seeds[sample_index] if randomized else None
        text = make_name(indices[sample_index]) if randomized else state['target']['text']
        box = paste(image, name_paper(text), state['target']['quad_xy'])
        gain = (random.Random(sample_seed).uniform(*state['augmentation']['brightness_gain_range'])
                if randomized else state['augmentation']['brightness_gain'])
        image = np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        stem = state['id'] + (f'_{state_index:06d}' if randomized else '')
        image_out = output / 'images' / (stem + '.png')
        label_out = output / 'labels_partial' / (stem + '.txt')
        ok, encoded = cv2.imencode('.png', image)
        if not ok:
            raise ValueError('image encoding failed')
        encoded.tofile(str(image_out))
        label_out.write_text(yolo_row(box, 960, 540) + '\n', encoding='utf8')
        actual = {'source_image': relative(image_path), 'source_video': cfg['source_video'],
                  'source_group': cfg['source_group'], 'generated_text': text,
                  'paper_quad_xy': state['target']['quad_xy'], 'paper_box_xyxy': box,
                  'brightness_gain': gain, 'background_other_targets': 'unknown',
                  'ocr_status': 'not_verified', 'label_status': 'partial_candidate',
                  'batch_seed': seed if randomized else None, 'sample_seed': sample_seed,
                  'name_index': indices[sample_index] if randomized else None}
        records.append({'image': relative(image_out), 'label': relative(label_out),
                        'production_state_id': state['id'], **actual})
        run.record_sample(state['id'], output=image_out, sources=state['sources'],
                          parameters=state['parameters'], actual=actual)
    result = {'status': 'partial_candidates_not_training_approved',
              'generation_method': 'real_scene_plus_generated_name_paper',
              'source_manifest': relative(SOURCE_MANIFEST),
              'scope': 'Generated paper boxes only; other original-image targets and OCR readability remain unknown.',
              'count': len(records), 'records': records}
    (output / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return result


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
    images = [BACKGROUND / SOURCES[key]['image'] for key in
              sorted({s['background']['source_key'] for s in request['strata']})]
    metadata = [plan_path, FONT, SOURCE_MANIFEST]
    code = [Path(__file__), FIRE / 'compose_scene_fire_eval.py', FIRE / 'compose_scene_fire.py',
            FIRE / 'vary_name_papers.py',
            FIRE / 'fire_blender_labels.py', ROOT / 'lab/training_common/bayesian_allocation.py',
            ROOT / 'datasets/scripts/common/images.py', ROOT / 'datasets/scripts/common/annotations.py',
            ROOT / 'datasets/scripts/common/names.py', ROOT / 'datasets/scripts/common/paper.py',
            ROOT / 'datasets/scripts/common/perspective.py']
    with generation_guard('building_name.compose_real_scene:' + relative(output), images + metadata,
                          [output], {}, code_paths=code, procedural_inputs=metadata, root=ROOT,
                          production_plan=request, sample_output_paths=[output / 'images']) as run:
        if run.plan != planned:
            raise ValueError('guard plan differs from requested allocation')
        if not run.reused:
            generate(output, run)
    print(json.dumps({'planned': planned['planned_total'], 'output': relative(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
