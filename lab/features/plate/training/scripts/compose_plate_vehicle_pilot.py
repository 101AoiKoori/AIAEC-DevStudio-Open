"""Make low-resolution plate-replacement candidates on exposed car frames.

These are development-regression examples, not independent test data. Existing
candidate boxes locate the replacement; every resulting image needs visual review.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard  # noqa: E402
EXPORTS = ROOT / 'datasets/exports/plate'
ANNOTATIONS = ROOT / 'datasets/annotated/car_real_capture_20261112/field_eval_v1/annotations_root.json'
BACKGROUNDS = ('P09_B1.png', 'P10_B2.png')
PLATES = ('车牌一.png', '车牌二.png', '车牌三.png')


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot decode {path}')
    return image


def write_image(path, value):
    ok, encoded = cv2.imencode('.png', value)
    if not ok:
        raise OSError(f'Cannot encode {path}')
    encoded.tofile(str(path))


def generate(output):
    output = output.resolve()
    if not output.is_relative_to(EXPORTS.resolve()) or output == EXPORTS.resolve():
        raise ValueError('output must be a new directory under datasets/exports/plate')
    backgrounds = [ROOT / 'Archive/Assets/captures/car_real_capture_20261112' / name for name in BACKGROUNDS]
    plates = [ROOT / 'Archive/Assets/working_sources/digital/plate/provincial_source_pic' / name for name in PLATES]
    with generation_guard(
        'plate_vehicle_pilot:' + output.relative_to(ROOT).as_posix(),
        [*backgrounds, *plates, ANNOTATIONS], [output],
        {'backgrounds': BACKGROUNDS, 'plates': PLATES, 'image_size': [640, 480],
         'interpolation': 'area', 'gaussian_kernel': [5, 5], 'blur_sigma': 1.0,
         'brightness_factor': 0.8},
        code_paths=[Path(__file__)], procedural_inputs=[ANNOTATIONS], root=ROOT,
    ) as run:
        if run.reused:
            return json.loads((output / 'manifest.json').read_text(encoding='utf8'))
        return _generate(output, run.source_refs)


def _generate(output, source_refs):
    if output.exists():
        raise ValueError('output must be a new directory under datasets/exports/plate')
    annotations = json.loads(ANNOTATIONS.read_text(encoding='utf8'))
    output.mkdir(parents=True)
    for part in ('images', 'labels_candidate', 'overlays'):
        (output / part).mkdir()
    report = {'status': 'running', 'role': 'development_regression', 'source_refs': source_refs,
              'review': 'pending', 'image_size': [640, 480], 'samples': []}
    try:
        for background_name in BACKGROUNDS:
            background = ROOT / 'Archive/Assets/captures/car_real_capture_20261112' / background_name
            records = [r for r in annotations if r['model_key'] == 'plate_detection'
                       and r['image'] == background.relative_to(ROOT).as_posix()]
            if len(records) != 1 or len(records[0]['boxes']) != 1:
                raise ValueError('Expected one recorded plate on selected background')
            x1, y1, x2, y2 = records[0]['boxes'][0]['xyxy']
            if not (0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480):
                raise ValueError('Invalid candidate plate box')
            for index, plate_name in enumerate(PLATES, 1):
                plate_path = ROOT / 'Archive/Assets/working_sources/digital/plate/provincial_source_pic' / plate_name
                frame = read_image(background)
                if frame.shape != (480, 640, 3):
                    raise ValueError('Expected native 640x480 car frame')
                plate = cv2.resize(read_image(plate_path), (x2-x1, y2-y1), interpolation=cv2.INTER_AREA)
                plate = cv2.GaussianBlur(plate, (5, 5), 1.0)
                frame[y1:y2, x1:x2] = np.clip(plate.astype(np.float32) * .8, 0, 255).astype(np.uint8)
                name = f'{background.stem}_plate_{index}'
                write_image(output / 'images' / (name + '.png'), frame)
                label = f'0 {(x1+x2)/1280:.6f} {(y1+y2)/960:.6f} {(x2-x1)/640:.6f} {(y2-y1)/480:.6f}\n'
                (output / 'labels_candidate' / (name + '.txt')).write_text(label, encoding='utf8')
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)
                write_image(output / 'overlays' / (name + '.png'), overlay)
                report['samples'].append({'id': name, 'image': f'images/{name}.png',
                    'label': f'labels_candidate/{name}.txt', 'background': background.relative_to(ROOT).as_posix(),
                    'background_annotation': ANNOTATIONS.relative_to(ROOT).as_posix(),
                    'plate_source': plate_path.relative_to(ROOT).as_posix(), 'placement_xyxy': [x1, y1, x2, y2],
                    'generation': 'area downscale, Gaussian blur sigma 1.0, brightness factor 0.8',
                    'split': 'regression', 'status': 'pending_visual_review', 'human_reviewed': False})
        report['status'] = 'complete_candidates'
    except BaseException as exc:
        report.update(status='failed_or_interrupted', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (output / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = generate(ROOT / args.output)
    print(json.dumps({'status': result['status'], 'count': len(result['samples'])}, ensure_ascii=False))
