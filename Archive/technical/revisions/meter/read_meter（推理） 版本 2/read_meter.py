#!/usr/bin/env python3
"""Read a calibrated image and save experimental needle overlay and JSON."""
import argparse
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'ws' / 'src' / 'aec_vision' / 'src'))
from aec_vision.meter import annotate, annotate_ticks, detect_needle, match_template, project_points


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('image')
    p.add_argument('--config', required=True)
    p.add_argument('--template', required=True, help='Image used for calibration')
    p.add_argument('--ticks', required=True, help='Estimated tick segments in template coordinates')
    p.add_argument('--output', required=True, help='Output image path')
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    ticks = json.loads(Path(args.ticks).read_text(encoding='utf-8-sig'))
    if ticks['image_size'] != config['image_size']:
        p.error('tick calibration size differs from template')
    image = cv2.imread(args.image)
    if image is None:
        p.error('image could not be read')
    template = cv2.imread(args.template)
    if template is None or list(template.shape[1::-1]) != config['image_size']:
        p.error('template missing or size differs from calibration')
    matrix, match = match_template(template, image)
    if matrix is None:
        results = {name: {'status': 'unknown', 'reason': 'template matching failed'}
                   for name in ('temperature', 'humidity')}
        output = image.copy()
    else:
        width, height = config['image_size']
        aligned = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        results = {name: detect_needle(aligned, config[name]) for name in ('temperature', 'humidity')}
        for result in results.values():
            points = [result['center']]
            if result['endpoint'] is not None:
                points.append(result['endpoint'])
            mapped = project_points(points, matrix)
            result['center'] = [round(float(v), 1) for v in mapped[0]]
            if result['endpoint'] is not None:
                result['endpoint'] = [round(float(v), 1) for v in mapped[1]]
        output = annotate(annotate_ticks(image, ticks, matrix), results)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), output):
        raise RuntimeError('could not write overlay')
    print(json.dumps({'match': match, 'needles': results}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
