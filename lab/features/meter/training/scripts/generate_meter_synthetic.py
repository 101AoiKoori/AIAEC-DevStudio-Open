#!/usr/bin/env python3
"""Generate a self-drawn dual-needle YOLO Pose smoke dataset (train only).

This is a synthetic training experiment, not a copy of the competition meter.
The output directory must not exist. No real or independent validation data
are created. Python 3.8+, NumPy and OpenCV are required.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

SIZE = 640
KEYPOINTS = ('temperature_pivot', 'temperature_tip', 'humidity_pivot', 'humidity_tip',
             'temperature_min', 'temperature_max', 'humidity_min', 'humidity_max')
SCALES = {'temperature': {'center': (320, 278), 'radius': 220, 'angles': (150, 390),
                          'minimum': -30.0, 'maximum': 50.0, 'intervals': 80},
          'humidity': {'center': (320, 429), 'radius': 85, 'angles': (-45, 225),
                       'minimum': 0.0, 'maximum': 100.0, 'intervals': 50}}


def polar(center, radius, angle):
    a = math.radians(angle)
    return (center[0] + radius * math.cos(a), center[1] + radius * math.sin(a))


def ip(point):
    return tuple(int(round(v)) for v in point)


def draw_dial(rng):
    image = np.full((SIZE, SIZE, 3), (231, 235, 237), np.uint8)
    cv2.circle(image, (320, 320), 300, (50, 55, 60), -1, cv2.LINE_AA)
    cv2.circle(image, (320, 320), 287, (245, 245, 240), -1, cv2.LINE_AA)
    points, tick_lines, values, needle_segments = {}, {}, {}, []
    for name, scale in SCALES.items():
        center, radius = scale['center'], scale['radius']
        a0, a1 = scale['angles']
        lines = []
        for i in range(scale['intervals'] + 1):
            angle = a0 + (a1 - a0) * i / scale['intervals']
            major = i % (10 if name == 'temperature' else 5) == 0
            start = polar(center, radius - (20 if major else 11), angle)
            end = polar(center, radius, angle)
            cv2.line(image, ip(start), ip(end), (30, 30, 30), 2 if major else 1, cv2.LINE_AA)
            lines.append([list(start), list(end)])
            if major:
                value = scale['minimum'] + i * (scale['maximum'] - scale['minimum']) / scale['intervals']
                pos = polar(center, radius - 35, angle)
                cv2.putText(image, str(int(value)), (int(pos[0]) - 9, int(pos[1]) + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, .38 if name == 'temperature' else .27,
                            (40, 40, 40), 1, cv2.LINE_AA)
        tick_lines[name] = lines
        frac = float(rng.uniform(.03, .97))
        angle = a0 + frac * (a1 - a0)
        tip = polar(center, radius - 25, angle)
        tail = polar(center, 15, angle + 180)
        points[name + '_pivot'] = center
        points[name + '_tip'] = tip
        points[name + '_min'] = polar(center, radius, a0)
        points[name + '_max'] = polar(center, radius, a1)
        values[name] = scale['minimum'] + frac * (scale['maximum'] - scale['minimum'])
        needle_segments.append((name, tail, tip))
    # Draw hands above the common face; save their true locations independently.
    needle_masks = {}
    for name, tail, tip in needle_segments:
        center = SCALES[name]['center']
        color = (20, 20, 180) if name == 'temperature' else (150, 35, 20)
        cv2.line(image, ip(tail), ip(tip), color, 5, cv2.LINE_AA)
        cv2.circle(image, ip(center), 8, (20, 20, 20), -1, cv2.LINE_AA)
        mask = np.zeros((SIZE, SIZE), np.uint8)
        cv2.line(mask, ip(tail), ip(tip), 255, 7)
        cv2.circle(mask, ip(center), 8, 255, -1)
        needle_masks[name] = mask
    return image, points, tick_lines, values, needle_masks


def project_plane(rng):
    # Intrinsics and angles are synthetic experiment ranges, not field limits.
    yaw, pitch, roll = rng.uniform(-20, 20, size=3)
    yaw, pitch, roll = map(math.radians, (yaw, pitch, roll))
    rx = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)],
                   [0, math.sin(pitch), math.cos(pitch)]], dtype=np.float64)
    ry = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0],
                   [-math.sin(yaw), 0, math.cos(yaw)]], dtype=np.float64)
    rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                   [math.sin(roll), math.cos(roll), 0], [0, 0, 1]], dtype=np.float64)
    rotation = rz @ ry @ rx
    distance = float(rng.uniform(750, 920))
    focal = float(rng.uniform(610, 740))
    translation = np.array([rng.uniform(-20, 20), rng.uniform(-20, 20), distance])
    corners = np.array([[-320, -320, 0], [319, -320, 0], [319, 319, 0], [-320, 319, 0]], np.float64)
    camera = (rotation @ corners.T).T + translation
    if np.any(camera[:, 2] <= 0):
        raise ValueError('invalid camera geometry')
    dest = np.column_stack((focal * camera[:, 0] / camera[:, 2] + SIZE / 2,
                            focal * camera[:, 1] / camera[:, 2] + SIZE / 2)).astype(np.float32)
    source = np.float32([[0, 0], [639, 0], [639, 639], [0, 639]])
    matrix = cv2.getPerspectiveTransform(source, dest)
    params = {'yaw_deg': math.degrees(yaw), 'pitch_deg': math.degrees(pitch),
              'roll_deg': math.degrees(roll), 'distance': distance, 'focal_px': focal,
              'translation': translation.tolist(), 'homography': matrix.tolist()}
    return matrix, params


def map_points(points, matrix):
    return cv2.perspectiveTransform(np.asarray(points, dtype=np.float32).reshape(1, -1, 2), matrix)[0]


def apply_glare(image, rng, mask):
    alpha = np.zeros(image.shape[:2], np.uint8)
    ellipses = []
    if rng.random() < .8:
        # Elliptical glass highlights are an approximation, not optical simulation.
        for _ in range(int(rng.integers(1, 4))):
            pos = tuple(int(v) for v in rng.integers(120, 520, size=2))
            axes = (int(rng.integers(25, 130)), int(rng.integers(8, 45)))
            angle, intensity = float(rng.uniform(-90, 90)), int(rng.integers(90, 245))
            cv2.ellipse(alpha, pos, axes, angle, 0, 360, intensity, -1, cv2.LINE_AA)
            ellipses.append({'center': pos, 'axes': axes, 'angle_deg': angle, 'alpha': intensity})
        alpha = cv2.GaussianBlur(alpha, (0, 0), 5)
    alpha = cv2.bitwise_and(alpha, mask)
    weight = alpha.astype(np.float32)[..., None] / 255.0
    return np.clip(image * (1.0 - weight) + 255.0 * weight, 0, 255).astype(np.uint8), alpha, ellipses


def sample(rng):
    face, points, tick_lines, values, needle_masks = draw_dial(rng)
    matrix, params = project_plane(rng)
    # Include random brightness without implying a real scene distribution.
    background = np.full_like(face, tuple(int(v) for v in rng.integers(80, 180, size=3)))
    face_mask = np.zeros((SIZE, SIZE), np.uint8)
    cv2.circle(face_mask, (320, 320), 300, 255, -1)
    warped_mask = cv2.warpPerspective(face_mask, matrix, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
    warped = cv2.warpPerspective(face, matrix, (SIZE, SIZE), flags=cv2.INTER_LINEAR)
    image = np.where((warped_mask > 0)[..., None], warped, background)
    image, glare, ellipses = apply_glare(image, rng, warped_mask)
    gain = float(rng.uniform(.8, 1.12))
    image = np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    xy = map_points([points[k] for k in KEYPOINTS], matrix)
    warped_needles = {name: cv2.warpPerspective(mask, matrix, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
                      for name, mask in needle_masks.items()}
    labels, visibility = [], []
    for name, (x, y) in zip(KEYPOINTS, xy):
        if not (0 <= x < SIZE and 0 <= y < SIZE):
            labels.extend((0.0, 0.0, 0))
            visibility.append(0)
        else:
            # Strong highlight indicates uncertain/occluded, not a confirmed point.
            # Humidity is drawn last; it may hide the temperature needle.
            # Both hands may cover the printed scale endpoint landmarks.
            occluders = (('humidity',) if name.startswith('temperature_') and
                         name.endswith(('pivot', 'tip')) else
                         ('temperature', 'humidity') if name.endswith(('min', 'max')) else ())
            covered = any(warped_needles[k][int(y), int(x)] > 0 for k in occluders)
            v = 1 if glare[int(y), int(x)] >= 140 or covered else 2
            labels.extend((float(x / SIZE), float(y / SIZE), v))
            visibility.append(v)
    x, y, w, h = cv2.boundingRect(warped_mask)
    if w < 2 or h < 2:
        raise ValueError('projected dial is not visible')
    bbox = [(x + w / 2) / SIZE, (y + h / 2) / SIZE, w / SIZE, h / SIZE]
    # Store exact pixel-space geometry for future visual/round-trip auditing.
    projected_ticks = {name: map_points(np.asarray(lines).reshape(-1, 2), matrix).reshape(-1, 2, 2).tolist()
                       for name, lines in tick_lines.items()}
    return image, bbox, labels, {'parameters': params, 'brightness_gain': gain,
                                'values': values, 'keypoints': xy.tolist(),
                                'visibility': visibility, 'tick_lines': projected_ticks,
                                'glare_ellipses': ellipses, 'glare_max_alpha': int(glare.max())}


def sample_reference(rng, source_image, reference):
    """Augment one annotated real image; its needle positions stay unchanged."""
    height, width = source_image.shape[:2]
    if reference.get('image_size') != [width, height]:
        raise ValueError('reference image size differs from annotation')
    annotated = reference.get('keypoints', [])
    if len(annotated) != len(KEYPOINTS) or [p['name'] for p in annotated] != list(KEYPOINTS):
        raise ValueError('reference keypoint order must match the eight-point schema')
    if any(p.get('visibility') not in (1, 2) or not (0 <= p['x'] < width and 0 <= p['y'] < height)
           for p in annotated):
        raise ValueError('reference points must be inside the image with visibility 1 or 2')
    bbox = reference.get('bbox_xyxy')
    if bbox is None or len(bbox) != 4 or not (0 <= bbox[0] < bbox[2] <= width and 0 <= bbox[1] < bbox[3] <= height):
        raise ValueError('invalid reference bounding box')
    s = min(SIZE / width, SIZE / height)
    normalize = np.array([[s, 0, (SIZE - width * s) / 2],
                          [0, s, (SIZE - height * s) / 2], [0, 0, 1]], dtype=np.float64)
    camera_matrix, params = project_plane(rng)
    matrix = camera_matrix @ normalize
    # This projects the existing photograph as a flat image. It cannot invent
    # new needle positions or remove the reflection already in the photograph.
    warped = cv2.warpPerspective(source_image, matrix, (SIZE, SIZE), flags=cv2.INTER_LINEAR)
    source_mask = np.full((height, width), 255, np.uint8)
    warped_mask = cv2.warpPerspective(source_mask, matrix, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
    background = np.full_like(warped, tuple(int(v) for v in rng.integers(80, 180, size=3)))
    image = np.where((warped_mask > 0)[..., None], warped, background)
    x0, y0, x1, y1 = bbox
    corners = map_points([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], matrix)
    meter_mask = np.zeros((SIZE, SIZE), np.uint8)
    cv2.fillConvexPoly(meter_mask, np.rint(corners).astype(np.int32), 255)
    image, glare, ellipses = apply_glare(image, rng, cv2.bitwise_and(warped_mask, meter_mask))
    gain = float(rng.uniform(.8, 1.12))
    image = np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    xy = map_points([[p['x'], p['y']] for p in annotated], matrix)
    labels, visibility = [], []
    for (x, y), point in zip(xy, annotated):
        if not (0 <= x < SIZE and 0 <= y < SIZE):
            labels.extend((0.0, 0.0, 0))
            visibility.append(0)
        else:
            v = 1 if point['visibility'] == 1 or glare[int(y), int(x)] >= 140 else 2
            labels.extend((float(x / SIZE), float(y / SIZE), v))
            visibility.append(v)
    x, y, w, h = cv2.boundingRect(meter_mask)
    if w < 2 or h < 2:
        raise ValueError('projected meter is not visible')
    bounds = [(x + w / 2) / SIZE, (y + h / 2) / SIZE, w / SIZE, h / SIZE]
    params['homography'] = matrix.tolist()
    return image, bounds, labels, {'parameters': params, 'brightness_gain': gain,
                                  'values': {'temperature': None, 'humidity': None},
                                  'keypoints': xy.tolist(), 'visibility': visibility,
                                  'tick_lines': None, 'glare_ellipses': ellipses,
                                  'glare_max_alpha': int(glare.max()),
                                  'reference_status': reference.get('status', 'unverified')}


def generate(output, count, seed, source_image=None, reference=None):
    output = Path(output)
    if count < 1:
        raise ValueError('count must be positive')
    if (source_image is None) != (reference is None):
        raise ValueError('source image and reference must be provided together')
    # Never delete or overwrite a previous run, including partial runs.
    output.mkdir(parents=True, exist_ok=False)
    image_dir, label_dir, preview_dir = (output / 'images' / 'train', output / 'labels' / 'train', output / 'previews')
    for directory in (image_dir, label_dir, preview_dir):
        directory.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    manifest = {'purpose': 'augmentation smoke test only; no independent validation',
                'source_type': 'annotated_single_image' if source_image is not None else 'self_drawn',
                'source_description': reference.get('source', 'unspecified') if reference else 'procedurally drawn',
                'seed': seed, 'keypoint_order': KEYPOINTS, 'samples': []}
    for i in range(count):
        image, box, keypoints, info = (sample_reference(rng, source_image, reference)
                                       if source_image is not None else sample(rng))
        name = 'meter_{:05d}'.format(i)
        if not cv2.imwrite(str(image_dir / (name + '.png')), image):
            raise RuntimeError('failed to write image ' + name)
        values = [0] + box + keypoints
        (label_dir / (name + '.txt')).write_text(' '.join(str(v) if isinstance(v, int) else '{:.6f}'.format(v)
                                                               for v in values) + '\n', encoding='utf-8')
        if i < 8:
            preview = image.copy()
            for k, ((x, y), visible) in enumerate(zip(info['keypoints'], info['visibility'])):
                if visible:
                    cv2.circle(preview, ip((x, y)), 4, (0, 255, 0) if visible == 2 else (0, 165, 255), -1)
                    cv2.putText(preview, str(k), ip((x + 4, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 255), 1)
            if not cv2.imwrite(str(preview_dir / (name + '.png')), preview):
                raise RuntimeError('failed to write preview ' + name)
        info['id'] = name
        manifest['samples'].append(info)
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New output directory; must not already exist')
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--seed', type=int, default=20261106)
    parser.add_argument('--source-image', help='Optional annotated source photo')
    parser.add_argument('--reference', help='JSON reference annotation for source photo')
    args = parser.parse_args()
    try:
        if bool(args.source_image) != bool(args.reference):
            raise ValueError('--source-image and --reference are required together')
        image, reference = None, None
        if args.source_image:
            image = cv2.imread(args.source_image)
            if image is None:
                raise ValueError('cannot read source image')
            reference = json.loads(Path(args.reference).read_text(encoding='utf-8-sig'))
        print('Generated {} experimental samples in {}'.format(args.count, generate(args.output, args.count, args.seed, image, reference)))
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
