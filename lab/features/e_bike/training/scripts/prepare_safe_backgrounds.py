"""Create two visually reviewed, e-bike-free crops of one legacy source video."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT / 'lab/training_common')]
from camera_tone import source_role  # noqa: E402
from lab.training_common.production_guard import generation_guard  # noqa: E402
from lab.scripts.data.v4_source_gate import build_queue  # noqa: E402

VIDEO = 'datasets/raw/site_archive_legacy/场景/综合视频/空场与场景物料_001.mp4'
PARENTS = [f'datasets/raw/person/teammate_submission_20261110/dataset/backgrounds/bg_{i:05d}_f{j:05d}.jpg'
           for i, j in ((0, 1), (1, 4))]
CROP = [800, 200, 1760, 920]


def declaration():
    states = []
    for i, parent in enumerate(PARENTS):
        states.append({'id': f'legacy_crop_{i+1}', 'failure_event': 'background_coverage',
            'min_count': 1, 'max_count': 1, 'prior': {'alpha': 1, 'beta': 1},
            'exposure_weight': 1, 'failure_cost': 1, 'conditions': [],
            'background': {'image': parent}, 'target': {'e_bike': 'none_in_reviewed_crop'},
            'labels': {'status': 'visually_reviewed_empty_crop_candidate'},
            'augmentation': {'resize': 'linear_640x480'},
            'parameters': {'crop_xyxy': CROP, 'output_size': [640, 480]},
            'sources': sorted([parent, VIDEO])})
    return {'schema_version': 1, 'model_key': 'e_bike', 'target_split': 'train',
            'purpose': 'Prepare visually reviewed ground crops from one legacy source video.',
            'budget': len(states), 'uncertainty_weight': 0,
            'independence_basis': 'Two source frames share one parent video.',
            'constraints': {'crop_xyxy': CROP, 'size': [640, 480]}, 'strata': states}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = (ROOT / args.output).resolve()
    owner = (ROOT / 'runs/e_bike').resolve()
    if not out.is_relative_to(owner) or out == owner or out.exists():
        parser.error('output must be a new candidate directory under runs/e_bike')
    plan = declaration()
    known = build_queue(ROOT)
    for name in PARENTS + [VIDEO]:
        if not (ROOT / name).is_file() or source_role(ROOT, ROOT / name, known)[0] == 'regression':
            raise ValueError('missing or regression-only source: ' + name)
    op = 'e_bike.safe_legacy_crops:' + out.relative_to(ROOT).as_posix()
    with generation_guard(op, [ROOT / n for n in PARENTS + [VIDEO]], [out],
                          {'review': 'full crop reviewed; no visible e-bike', 'crop_xyxy': CROP},
                          code_paths=[Path(__file__), ROOT / 'lab/training_common/production_guard.py'],
                          root=ROOT, production_plan=plan, sample_output_paths=[out / 'images']) as run:
        if run.reused:
            return
        (out / 'images').mkdir(parents=True)
        records = []
        for state in run.plan['strata']:
            parent = state['background']['image']
            frame = cv2.imdecode(np.fromfile(str(ROOT / parent), np.uint8), cv2.IMREAD_COLOR)
            if frame is None or frame.shape[:2] != (1080, 1920):
                raise ValueError('unexpected legacy frame dimensions')
            left, top, right, bottom = CROP
            crop = cv2.resize(frame[top:bottom, left:right], (640, 480), interpolation=cv2.INTER_LINEAR)
            image = out / 'images' / (state['id']+'.png')
            cv2.imencode('.png', crop)[1].tofile(str(image))
            actual = {'parent_image': parent, 'source_video': VIDEO, 'crop_xyxy': CROP,
                      'review': 'visually reviewed full crop; no visible e-bike',
                      'source_history': 'unknown/historical_exposure_incomplete'}
            run.record_sample(state['id'], output=image, sources=state['sources'],
                              parameters=state['parameters'], actual=actual)
            records.append({'id': state['id'], 'image': image.relative_to(ROOT).as_posix(),
                'parent_image': parent, 'source_video': VIDEO, 'task_state': 'candidate_complete_negative',
                'source_group': 'legacy_scene_001_video', 'generation_method': 'frame_crop_resize',
                'ground_zone': {'poly': [[0, 180], [520, 180], [520, 479], [0, 479]]},
                'safe_regions': [[[20, 250], [500, 250], [500, 460], [20, 460]]],
                'forbidden_boxes': [], **actual})
        (out / 'backgrounds.json').write_text(json.dumps({'backgrounds': records}, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
        print(json.dumps({'output': out.relative_to(ROOT).as_posix(), 'images': len(records)}))


if __name__ == '__main__':
    main()
