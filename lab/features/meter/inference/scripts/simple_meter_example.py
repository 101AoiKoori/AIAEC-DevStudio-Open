#!/usr/bin/env python3
"""Minimal offline reference for the PDF meter; not a final ROS node."""
# PDF example only. Recalibrate for the actual meter before deployment.
SCALE = {
    'temperature': {'minimum': -30.0, 'maximum': 50.0, 'unit': 'C'},
    'humidity': {'minimum': 0.0, 'maximum': 100.0, 'unit': '%RH'},
}

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'ws/src/aec_vision/src'))
from aec_vision.meter import (annotate, annotate_ticks, detect_needle, match_template,
                              project_points, reading_from_ticks, tick_visibility)

CONFIG = ROOT / 'lab/features/meter/inference/config/pdf_page15.json'
TICKS = ROOT / 'lab/features/meter/inference/config/pdf_ticks.json'
PDF_EXAMPLE = ROOT / 'datasets/raw/meter/pdf/page15-005.jpg'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', nargs='?', default=str(PDF_EXAMPLE))
    parser.add_argument('--template', default=str(PDF_EXAMPLE))
    parser.add_argument('--output', default=str(ROOT / 'lab/features/meter/inference/outputs/simple_reading.png'))
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding='utf-8-sig'))
    ticks = json.loads(TICKS.read_text(encoding='utf-8-sig'))
    image, template = cv2.imread(args.image), cv2.imread(args.template)
    if image is None or template is None:
        parser.error('input image or template cannot be read')
    if list(template.shape[1::-1]) != config['image_size'] or ticks['image_size'] != config['image_size']:
        parser.error('template and calibration sizes differ')
    matrix, match = match_template(template, image)
    if matrix is None:
        print(json.dumps({'match': match, 'readings': {name: {'status': 'unknown', 'value': None, 'unit': scale['unit']} for name, scale in SCALE.items()}}, indent=2))
        return 1
    aligned = cv2.warpPerspective(image, matrix, tuple(config['image_size']),
                                  flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    results, readings = {}, {}
    for name, scale in SCALE.items():
        result = detect_needle(aligned, config[name])
        value = reading_from_ticks(result['angle_deg'], config[name]['center'], ticks[name],
                                   scale['minimum'], scale['maximum'])
        readings[name] = {'status': 'estimated' if value is not None else 'unknown',
                          'value': round(value, 2) if value is not None else None,
                          'unit': scale['unit'], 'angle_deg': result['angle_deg']}
        points = [result['center']] + ([result['endpoint']] if result['endpoint'] is not None else [])
        mapped = project_points(points, matrix)
        result['center'] = mapped[0].tolist()
        if result['endpoint'] is not None:
            result['endpoint'] = mapped[1].tolist()
        results[name] = result
    visibility = tick_visibility(template, aligned, ticks)
    output = annotate(annotate_ticks(image, ticks, matrix, visibility), results)
    for index, (name, reading) in enumerate(readings.items()):
        text = '{}: {} {}'.format(name, reading['value'] if reading['value'] is not None else 'unknown', reading['unit'])
        cv2.putText(output, text, (10, 80 + index * 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output, 'Green dashed: estimated tick', (10, max(20, image.shape[0] - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 0), 1, cv2.LINE_AA)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), output):
        raise RuntimeError('cannot write output image')
    print(json.dumps({'match': match, 'readings': readings}, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
