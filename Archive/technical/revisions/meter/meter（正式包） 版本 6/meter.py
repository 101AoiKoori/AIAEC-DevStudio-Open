"""Configurable single-image analog needle detection for offline experiments.

Angles are degrees in image coordinates: zero points right and positive angles
point down. Configuration and calibration are supplied by the caller.
"""
import math

import cv2
import numpy as np


def match_template(template, image):
    """Estimate template-to-image homography; reject weak or degenerate fits."""
    orb = cv2.ORB_create(nfeatures=3000)
    def gray(value):
        return cv2.cvtColor(value, cv2.COLOR_BGR2GRAY) if value.ndim == 3 else value
    # Calibration images are centred dial crops. Ignore the outer border and
    # corners, which often show the unchanged room rather than the instrument.
    # This assumption must be revisited for a differently framed template.
    h, w = template.shape[:2]
    dial_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(dial_mask, (w // 2, h // 2),
                (max(1, int(w * .42)), max(1, int(h * .42))),
                0, 0, 360, 255, -1)
    keys1, desc1 = orb.detectAndCompute(gray(template), dial_mask)
    keys2, desc2 = orb.detectAndCompute(gray(image), None)
    if desc1 is None or desc2 is None:
        return None, {'status': 'unknown', 'reason': 'insufficient features'}
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc1, desc2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < .75 * p[1].distance]
    if len(good) < 12:
        return None, {'status': 'unknown', 'reason': 'insufficient matches', 'matches': len(good)}
    src = np.float32([keys1[m.queryIdx].pt for m in good])
    dst = np.float32([keys2[m.trainIdx].pt for m in good])
    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if matrix is None or mask is None:
        return None, {'status': 'unknown', 'reason': 'homography failed'}
    valid = mask.ravel().astype(bool)
    count = int(valid.sum())
    coverage = cv2.contourArea(cv2.convexHull(src[valid])) / float(w * h) if count >= 3 else 0
    # Require evidence on both sides of both axes, rather than letting one
    # small shared logo or part of the surrounding scene locate the whole dial.
    inliers = src[valid]
    quadrants = set((int(x >= w / 2), int(y >= h / 2)) for x, y in inliers)
    corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    mapped = cv2.perspectiveTransform(corners.reshape(1, -1, 2), matrix)[0]
    area = abs(cv2.contourArea(mapped))
    image_area = float(image.shape[0] * image.shape[1])
    # A valid geometric fit can still place much of the dial outside the
    # image. Reject it before generating tick marks for invisible regions.
    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    dial_edge = np.column_stack((w / 2 + w * .42 * np.cos(angles),
                                 h / 2 + h * .42 * np.sin(angles))).astype(np.float32)
    projected_edge = cv2.perspectiveTransform(dial_edge.reshape(1, -1, 2), matrix)[0]
    visible = (np.isfinite(projected_edge).all() and
               np.all(projected_edge[:, 0] >= 0) and np.all(projected_edge[:, 0] < image.shape[1]) and
               np.all(projected_edge[:, 1] >= 0) and np.all(projected_edge[:, 1] < image.shape[0]))
    if (count < 12 or count / len(good) < .5 or coverage < .04 or len(quadrants) < 3 or
            not np.isfinite(mapped).all() or not cv2.isContourConvex(mapped) or
            area < .02 * image_area or area > 4 * image_area or not visible):
        return None, {'status': 'unknown', 'reason': 'unreliable geometric fit',
                      'inliers': count, 'coverage': round(coverage, 4)}
    # Geometry alone can match a visible half of a dial while the other half
    # is hidden. Compare textured blocks in template coordinates so a broad
    # occlusion cannot silently receive projected tick marks. Small changes,
    # including moving needles, do not by themselves reject the full dial.
    aligned = cv2.warpPerspective(gray(image), matrix, (w, h),
                                  flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    reference = gray(template)
    compared, inconsistent = 0, 0
    for row in range(8):
        for col in range(8):
            if ((col + .5 - 4) / 3.36) ** 2 + ((row + .5 - 4) / 3.36) ** 2 > 1:
                continue
            x0, x1 = int(w * col / 8), int(w * (col + 1) / 8)
            y0, y1 = int(h * row / 8), int(h * (row + 1) / 8)
            ref = reference[y0:y1, x0:x1]
            sample = aligned[y0:y1, x0:x1]
            if ref.size == 0 or float(np.std(ref)) < 12:
                continue
            compared += 1
            if float(cv2.matchTemplate(ref, sample, cv2.TM_CCOEFF_NORMED)[0, 0]) < .25:
                inconsistent += 1
    if compared >= 6 and inconsistent >= 3:
        return None, {'status': 'unknown', 'reason': 'dial appearance inconsistent or occluded',
                      'inconsistent_blocks': inconsistent, 'compared_blocks': compared}
    return matrix, {'status': 'matched', 'inliers': count, 'matches': len(good),
                    'coverage': round(coverage, 4)}


def project_points(points, matrix):
    """Map template pixel points to input image coordinates."""
    pts = np.asarray(points, dtype=np.float32).reshape(1, -1, 2)
    return cv2.perspectiveTransform(pts, matrix)[0]


def tick_visibility(template, aligned, ticks, threshold=.65):
    """Compare local appearance in template coordinates, independently per tick.

    Similarity is a visibility heuristic, not proof a proposed tick is correct.
    An already occluded template tick is not treated as observable.
    """
    def gray(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    ref, sample = gray(template), gray(aligned)
    if ref.shape != sample.shape:
        raise ValueError('template and aligned image size differ')
    h, w = ref.shape
    result = {}
    for group in ('temperature', 'humidity'):
        visible, unknown, estimated = [], [], []
        for tick in ticks[group]:
            index = tick['index']
            if tick.get('template_occluded', False):
                estimated.append(index)
                continue
            coords = np.asarray([tick['start'], tick['end']], dtype=np.float32)
            if not np.isfinite(coords).all():
                unknown.append(index)
                continue
            x0, y0 = np.floor(coords.min(axis=0) - 4).astype(int)
            x1, y1 = np.ceil(coords.max(axis=0) + 5).astype(int)
            if x0 < 0 or y0 < 0 or x1 > w or y1 > h or x1 <= x0 or y1 <= y0:
                unknown.append(index)
                continue
            base = ref[y0:y1, x0:x1]
            current = sample[y0:y1, x0:x1]
            if float(np.std(base)) < 8 or float(np.std(current)) < 8:
                unknown.append(index)
                continue
            similarity = float(cv2.matchTemplate(base, current, cv2.TM_CCOEFF_NORMED)[0, 0])
            (visible if math.isfinite(similarity) and similarity >= threshold else unknown).append(index)
        result[group] = {'visible_indices': visible, 'unknown_indices': unknown,
                         'estimated_indices': estimated, 'visible_count': len(visible),
                         'unknown_count': len(unknown), 'estimated_count': len(estimated)}
    return result


def annotate_ticks(image, ticks, matrix, visibility=None):
    """Draw estimated calibrated tick segments in bright green on input image."""
    output = image.copy()
    for group in ('temperature', 'humidity'):
        allowed = None if visibility is None else set(visibility[group]['visible_indices'])
        for tick in ticks[group]:
            if allowed is not None and tick['index'] not in allowed:
                continue
            points = project_points([tick['start'], tick['end']], matrix)
            start, end = (tuple(int(round(v)) for v in point) for point in points)
            cv2.line(output, start, end, (0, 255, 0), 2, cv2.LINE_AA)
    return output


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
