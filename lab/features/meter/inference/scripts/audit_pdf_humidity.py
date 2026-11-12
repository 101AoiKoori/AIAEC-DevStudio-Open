#!/usr/bin/env python3
"""Report PDF humidity candidate geometry, not ground-truth accuracy."""
import argparse
import json
import math
from pathlib import Path


def humidity_diagnostics(ticks, center):
    if len(ticks) != 51 or [x['index'] for x in ticks] != list(range(51)):
        raise ValueError('expected 51 ordered humidity candidates')
    angles = []
    for tick in ticks:
        x, y = tick['end']
        angle = math.degrees(math.atan2(y - center[1], x - center[0]))
        if angles:
            while angle < angles[-1] - 180:
                angle += 360
            while angle > angles[-1] + 180:
                angle -= 360
        angles.append(angle)
    gaps = [b - a for a, b in zip(angles, angles[1:])]
    if min(gaps) <= 0:
        raise ValueError('tick angles are not strictly increasing')
    segments = []
    for segment in range(10):
        first = 5 * segment
        start, end = angles[first], angles[first + 5]
        residual = max(abs(angles[first + j] - (start + (end - start) * j / 5))
                       for j in range(1, 5))
        segments.append({'index_start': first, 'index_end': first + 5,
                         'span_degrees': round(end - start, 2),
                         'max_internal_residual_degrees': round(residual, 2)})
    return {'candidate_count': len(ticks), 'segments': segments,
            'warning': 'Candidate geometry only; this is not independent ground truth.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ticks', required=True)
    parser.add_argument('--center', nargs=2, type=float, default=(325, 497))
    args = parser.parse_args()
    data = json.loads(Path(args.ticks).read_text(encoding='utf-8-sig'))
    print(json.dumps(humidity_diagnostics(data['humidity'], args.center), indent=2))


if __name__ == '__main__':
    main()
