#!/usr/bin/env python3
"""Reproduce experimental PDF temperature tick proposal, not real-world calibration.

Nine visually located major tick angles and ring radii are sample-specific.
Subdivide each major interval into ten, then search near each predicted angle.
All outputs remain unverified, including occluded or ambiguous ticks.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

CENTER = (320.0, 355.0)
MAJOR_ANGLES = [-211.1, -183.3, -156.4, -127.3, -98.1, -67.7, -37.4, -6.2, 25.2]
MAJOR_RADII = [231.0, 234.0, 237.0, 238.0, 240.0, 238.0, 235.0, 232.0, 228.0]


def calibrate(image, ticks):
    if image.shape[:2] != (672, 641):
        raise ValueError('only the 641x672 PDF sample is supported')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if len(ticks) != 81:
        raise ValueError('expected 81 temperature proposals')
    cx, cy = CENTER
    chosen = []
    for i, tick in enumerate(ticks):
        k = min(i // 10, 7)
        frac = (i - 10 * k) / 10.0
        predicted = (1 - frac) * MAJOR_ANGLES[k] + frac * MAJOR_ANGLES[k + 1]
        radius = (1 - frac) * MAJOR_RADII[k] + frac * MAJOR_RADII[k + 1]
        inner = int(round(radius))
        candidates = []
        for delta in np.arange(-1.0, 1.001, 0.1):
            angle = predicted + delta
            radians = math.radians(angle)
            radii = np.arange(inner + 1, inner + 24)
            x = np.rint(cx + radii * math.cos(radians)).astype(int)
            y = np.rint(cy + radii * math.sin(radians)).astype(int)
            if np.any(x < 0) or np.any(x >= gray.shape[1]) or np.any(y < 0) or np.any(y >= gray.shape[0]):
                continue
            darkness = float(np.mean(gray[y, x] < 115))
            candidates.append((darkness - .005 * abs(delta), angle, darkness))
        if not candidates:
            raise ValueError('candidate lies outside image')
        _, angle, support = max(candidates)
        chosen.append(angle)
        radians = math.radians(angle)
        outer = inner + (34 if i % 10 == 0 else 19)
        tick['start'] = [round(cx + inner * math.cos(radians), 1), round(cy + inner * math.sin(radians), 1)]
        tick['end'] = [round(cx + outer * math.cos(radians), 1), round(cy + outer * math.sin(radians), 1)]
        tick['source'] = 'major-anchor interpolated candidate; unverified'
        tick['dark_support'] = round(support, 3)
    gaps = np.diff(chosen)
    if np.any(gaps < 1.5) or np.any(gaps > 4.5):
        raise ValueError('duplicate or implausibly spaced candidates: manual review required')
    return float(gaps.min()), float(gaps.max())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', required=True)
    p.add_argument('--ticks', required=True, help='Existing JSON, humidity entries preserved')
    p.add_argument('--output', required=True, help='Output JSON, review before use')
    args = p.parse_args()
    image = cv2.imread(args.image)
    if image is None:
        p.error('image could not be read')
    data = json.loads(Path(args.ticks).read_text(encoding='utf-8-sig'))
    lo, hi = calibrate(image, data['temperature'])
    Path(args.output).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('81 unverified candidate ticks; angular gaps {:.2f}..{:.2f} degrees'.format(lo, hi))


if __name__ == '__main__':
    main()
