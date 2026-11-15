#!/usr/bin/env python3
"""Offline OpenCV-only experiment. No PDF template or fixed image coordinates."""
import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'ws/src/aec_vision/src'))
from aec_vision.meter_auto import analyze_meter

# Values are experimental assumptions; actual instrument scale needs checking.
SCALES = {
    'temperature': {'minimum': -30.0, 'maximum': 50.0, 'unit': 'C'},
    'humidity': {'minimum': 0.0, 'maximum': 100.0, 'unit': '%RH'},
}


def draw(image, result):
    output = image.copy()
    for name, dial in result.get('dials', {}).items():
        color = (0, 0, 255) if name == 'temperature' else (255, 0, 0)
        for tick in dial.get('ticks', []):
            start = tuple(int(round(x)) for x in tick['start'])
            end = tuple(int(round(x)) for x in tick['end'])
            cv2.line(output, start, end, (0, 255, 0), 2, cv2.LINE_AA)
        needle = dial.get('needle') or {}
        if needle.get('status') == 'detected':
            center = tuple(int(round(x)) for x in needle['center'])
            endpoint = tuple(int(round(x)) for x in needle['endpoint'])
            cv2.circle(output, center, 4, color, 2)
            cv2.line(output, center, endpoint, color, 2, cv2.LINE_AA)
        arc = dial.get('arc') or {}
        for key, text in (('start_point', 'END A?'), ('end_point', 'END B?')):
            if arc.get(key) is not None:
                p = tuple(int(round(x)) for x in arc[key])
                cv2.circle(output, p, 5, (0, 255, 255), 2)
                cv2.putText(output, text, p, cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1)
    cv2.putText(output, 'Green: candidates, not verified', (8, max(20, output.shape[0] - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 0), 1, cv2.LINE_AA)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('image', help='Input image')
    p.add_argument('--output', required=True, help='Output image')
    p.add_argument('--assume-clockwise', action='store_true',
                   help='Experiment only: assume arc start is minimum and increases clockwise')
    args = p.parse_args()
    image = cv2.imread(args.image)
    if image is None:
        p.error('cannot read image')
    scales = {name: dict(scale, direction='clockwise') if args.assume_clockwise else dict(scale)
              for name, scale in SCALES.items()}
    result = analyze_meter(image, scales)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), draw(image, result)):
        raise RuntimeError('cannot write output')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get('status') != 'unknown' else 1


if __name__ == '__main__':
    sys.exit(main())

