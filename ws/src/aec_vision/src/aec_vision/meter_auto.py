"""Experimental geometry-only meter analysis; all detections are candidates.

Assumes a round outer dial and a smaller dial in its lower half. Circle geometry,
sector boundaries and semantic labels must be independently verified. No PDF
pixel coordinates, image template, model weights, or OCR are used.
"""
import math

import cv2
import numpy as np

from .meter import detect_needle


def _circles(gray):
    h, w = gray.shape
    size = min(h, w)
    result = cv2.HoughCircles(cv2.medianBlur(gray, 5), cv2.HOUGH_GRADIENT,
                              dp=1.2, minDist=size * .085, param1=120,
                              param2=30, minRadius=max(3, int(size * .03)),
                              maxRadius=int(size * .49))
    if result is None:
        return None, None
    circles = np.asarray(result[0], dtype=float)
    # Hough returns strongest first; only a substantial circle can be the dial.
    outer = next((c for c in circles if .27 * size <= c[2] <= .49 * size), None)
    if outer is None:
        return None, None
    x, y, radius = outer
    # A dual dial with a smaller, vertically separated lower subdial is a
    # sample-derived topology hypothesis, not a general gauge classifier.
    options = [c for c in circles if .22 * radius < c[2] < .42 * radius
               and abs(c[0] - x) < .25 * radius
               and .25 * radius < c[1] - y < .75 * radius
               and np.hypot(c[0] - x, c[1] - y) + c[2] < radius * 1.02]
    inner = min(options, key=lambda c: abs(c[0] - x)) if options else None
    return outer, inner


def _tick_candidates(gray, circle, inner=False):
    cx, cy, radius = circle
    # Sample radial strokes in an annulus relative to automatically found circle.
    step = .25
    angles = np.arange(0., 360., step)
    radians = np.deg2rad(angles)
    # Inner dial has a shorter, lighter printed scale in the PDF sample.
    r0, r1, threshold, fraction = (.85, .99, 125, .25) if inner else (.81, .90, 65, .30)
    radii = np.linspace(radius * r0, radius * r1, 20)
    x = np.rint(cx + np.cos(radians[:, None]) * radii).astype(int)
    y = np.rint(cy + np.sin(radians[:, None]) * radii).astype(int)
    valid = (x >= 0) & (x < gray.shape[1]) & (y >= 0) & (y < gray.shape[0])
    dark = np.zeros(x.shape, dtype=bool)
    dark[valid] = gray[y[valid], x[valid]] < threshold
    active = np.mean(dark, axis=1) >= fraction
    # Runs wider than 3 degrees usually represent letters, shadows or needles.
    indexes = np.flatnonzero(active & ~np.roll(active, 1))
    candidates = []
    for start in indexes:
        length = 0
        while length < len(active) and active[(start + length) % len(active)]:
            length += 1
        if not 1 <= length <= int(3 / step):
            continue
        angle = ((start + (length - 1) / 2) * step) % 360
        a = math.radians(angle)
        candidates.append({'angle_deg': round(angle, 2),
                           'start': [round(cx + radius * r0 * math.cos(a), 1),
                                     round(cy + radius * r0 * math.sin(a), 1)],
                           'end': [round(cx + radius * r1 * math.cos(a), 1),
                                   round(cy + radius * r1 * math.sin(a), 1)]})
    return sorted(candidates, key=lambda t: t['angle_deg'])


def _candidate_arc(ticks):
    # Longest sequence of relatively close radial strokes; a heuristic that
    # can mistake text and shadows for ticks. It is NOT a calibrated scale.
    if len(ticks) < 35:
        return {'status': 'unknown', 'reason': 'too few tick candidates'}
    a = [t['angle_deg'] for t in ticks]
    gaps = [(a[(i + 1) % len(a)] - a[i]) % 360 for i in range(len(a))]
    cut = [i for i, gap in enumerate(gaps) if gap > 18]
    if not cut:
        return {'status': 'unknown', 'reason': 'no identifiable empty angular gap'}
    clusters = []
    for i, prev in enumerate(cut):
        end = cut[(i + 1) % len(cut)]
        count = (end - prev) % len(a)
        if len(cut) == 1:
            count = len(a)
        start = (prev + 1) % len(a)
        if count > len(a):
            continue
        span = (a[end] - a[start]) % 360
        clusters.append((count, span, start, end))
    count, span, start, end = max(clusters)
    if count < 10 or not 50 <= span <= 330:
        return {'status': 'unknown', 'reason': 'no sufficiently long candidate tick arc'}
    return {'status': 'candidate', 'start_deg': a[start], 'end_deg': a[end],
            'start_point': ticks[start]['start'], 'end_point': ticks[end]['start'],
            'candidate_count': count, 'span_deg': round(span, 2),
            'reason': 'candidate arc; tick identities and endpoints are unconfirmed'}


def analyze_meter(image, ranges=None):
    """Find dial/needle/tick candidates; ranges alone do not identify direction.

    Returns pixel coordinates in the input image. A candidate is not a valid
    competition reading. Each optional range may include minimum, maximum and
    an explicit value direction ('clockwise' or 'counterclockwise').
    Even then, the value is only a hypothesis based on an unverified arc.
    """
    if image is None or image.size == 0:
        raise ValueError('image is empty')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if gray.ndim != 2 or min(gray.shape) < 80:
        raise ValueError('image must be grayscale or BGR with minimum dimension 80')
    # Normalize resolution only for image processing, not fixed pixel positions.
    scale = 640.0 / min(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    outer, inner = _circles(small)
    answer = {'status': 'unknown', 'reason': 'dual dial geometry not confirmed', 'dials': {}}
    for name, circle in [('temperature', outer), ('humidity', inner)]:
        if circle is None:
            answer['dials'][name] = {'status': 'unknown', 'reason': 'circle not found', 'ticks': [],
                                     'arc': {'status': 'unknown'}, 'reading': {'status': 'unknown', 'value': None}}
            continue
        cx, cy, radius = (float(v) for v in circle)
        cfg = {'center': [cx, cy], 'search_radius': [max(1, int(radius * .25)), max(2, int(radius * .80))],
               'angle_range_deg': [-180, 180], 'dark_threshold': 80, 'min_support': .55,
               'max_tip_radius': int(radius * .86)}
        needle = detect_needle(small, cfg)
        ticks = _tick_candidates(small, circle, inner=name == 'humidity')
        arc = _candidate_arc(ticks)
        reading = {'status': 'unknown', 'value': None,
                   'unit': 'C' if name == 'temperature' else '%RH',
                   'reason': 'tick arc, endpoint values and direction are unconfirmed'}
        bounds = (ranges or {}).get(name, {})
        if bounds.get('direction') in ('clockwise', 'counterclockwise') and arc['status'] == 'candidate' and needle['status'] == 'detected':
            minimum, maximum = float(bounds['minimum']), float(bounds['maximum'])
            if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum >= maximum:
                raise ValueError('scale minimum and maximum must be finite and ordered')
            offsets = sorted((t['angle_deg'] - arc['start_deg']) % 360 for t in ticks
                             if (t['angle_deg'] - arc['start_deg']) % 360 <= arc['span_deg'] + .01)
            gaps = np.diff(offsets)
            low, high = np.quantile(gaps, [.1, .9]) if len(gaps) >= 10 else (0., float('inf'))
            offset = (needle['angle_deg'] - arc['start_deg']) % 360
            if low <= 0 or high / low > 1.45:
                reading['reason'] = 'candidate tick spacing inconsistent with uniform angular scale'
            elif offset <= arc['span_deg']:
                ratio = offset / arc['span_deg']
                if bounds['direction'] == 'counterclockwise':
                    ratio = 1 - ratio
                reading = {'status': 'estimated', 'value': round(minimum + ratio * (maximum - minimum), 2),
                           'unit': reading['unit'],
                           'reason': 'unverified tick arc and explicitly assumed direction'}
        for t in ticks:
            for key in ('start', 'end'):
                t[key] = [round(v / scale, 1) for v in t[key]]
        if arc['status'] == 'candidate':
            for key in ('start_point', 'end_point'):
                arc[key] = [round(v / scale, 1) for v in arc[key]]
        for key in ('center', 'endpoint'):
            if needle[key] is not None:
                needle[key] = [round(v / scale, 1) for v in needle[key]]
        answer['dials'][name] = {'status': 'candidate',
                                 'center': [round(cx / scale, 1), round(cy / scale, 1)],
                                 'radius': round(radius / scale, 1), 'needle': needle,
                                 'ticks': ticks, 'arc': arc, 'reading': reading}
    if outer is not None or inner is not None:
        answer['status'] = 'partial'
        answer['reason'] = 'circle positions and dial identities are hypotheses; candidate tick arc unverified'
    return answer
