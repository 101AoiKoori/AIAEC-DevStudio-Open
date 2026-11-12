"""Configurable single-image analog needle detection for offline experiments.

Angles are degrees in image coordinates: zero points right and positive angles
point down. Configuration and calibration are supplied by the caller.
"""
import math

import cv2
import numpy as np


def detect_needle(image, cfg):
    """Score dark radial strokes inside a calibrated annulus and angular sector."""
    if image is None or image.size == 0:
        raise ValueError('image is empty')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    cx, cy = cfg['center']
    r0, r1 = cfg['search_radius']
    lo, hi = cfg['angle_range_deg']
    if r0 < 1 or r1 <= r0 or hi <= lo or hi - lo > 360:
        raise ValueError('invalid search region')
    # Scan a narrow strip around each radius; use the median to tolerate noise.
    radii = np.arange(r0, r1 + 1, dtype=np.float32)
    angles = np.arange(lo, hi + 0.001, cfg.get('angle_step_deg', 0.5))
    scores = []
    threshold = cfg.get('dark_threshold', 80)
    h, w = gray.shape[:2]
    for angle in angles:
        a = math.radians(angle)
        dx, dy = math.cos(a), math.sin(a)
        samples = []
        for offset in (-1, 0, 1):
            x = np.rint(cx + radii * dx - offset * dy).astype(int)
            y = np.rint(cy + radii * dy + offset * dx).astype(int)
            if np.any(x < 0) or np.any(x >= w) or np.any(y < 0) or np.any(y >= h):
                samples = []
                break
            samples.append(gray[y, x])
        if not samples:
            scores.append(0.0)
        else:
            dark = np.median(np.stack(samples), axis=0) < threshold
            scores.append(float(np.mean(dark)))
    scores = np.asarray(scores)
    best = int(np.argmax(scores))
    angle = float(angles[best])
    score = float(scores[best])
    # Only report confident detection; this is an experimental threshold, not
    # a calibrated probability or a guarantee against false positives.
    # A uniformly dark frame has no distinguishable needle even though every
    # radial sample would otherwise score perfectly.
    uniform_dark = float(np.mean(gray < threshold)) > 0.95
    ok = score >= cfg.get('min_support', 0.6) and not uniform_dark
    a = math.radians(angle)
    # Extend the detected line until it leaves the dark stroke. The endpoint
    # is an image estimate, and may be shortened by glare or merge into text.
    end_radius = r1
    missing = 0
    for radius in range(int(r1) + 1, int(cfg.get('max_tip_radius', r1)) + 1):
        x, y = int(round(cx + radius * math.cos(a))), int(round(cy + radius * math.sin(a)))
        if x < 0 or x >= w or y < 0 or y >= h:
            break
        if gray[y, x] < threshold:
            end_radius, missing = radius, 0
        else:
            missing += 1
            if missing > cfg.get('max_tip_gap', 3):
                break
    endpoint = [round(cx + end_radius * math.cos(a), 1), round(cy + end_radius * math.sin(a), 1)]
    return {'status': 'detected' if ok else 'unknown', 'angle_deg': round(angle, 2) if ok else None,
            'center': [cx, cy], 'endpoint': endpoint if ok else None,
            'support': round(score, 4), 'reason': None if ok else 'insufficient radial support'}


def annotate(image, results):
    output = image.copy()
    for index, (name, result) in enumerate(results.items()):
        color = (0, 0, 255) if index == 0 else (255, 0, 0)
        center = tuple(int(round(v)) for v in result['center'])
        cv2.circle(output, center, 5, color, 2)
        if result['endpoint'] is not None:
            tip = tuple(int(round(v)) for v in result['endpoint'])
            cv2.line(output, center, tip, color, 2)
        cv2.putText(output, '{}: {}'.format(name, result['angle_deg'] if result['angle_deg'] is not None else 'unknown'),
                    (10, 25 + 25 * index), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2, cv2.LINE_AA)
    return output
