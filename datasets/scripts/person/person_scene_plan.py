"""In-memory person scene composition from explicit, caller-reviewed inputs.

Coordinates are pixels in the supplied background. This module does not load
sources, write outputs, allocate quotas or provide a production entry point.
Callers must connect source records and a production guard before production.
"""

if __name__ == "__main__":
    raise SystemExit("Task helper is import-only; use the primary entry in datasets/scripts/person/.")

import math

import cv2
import numpy as np

from datasets.scripts.person.synthesize_real_ground import alpha_place, planar_warp, visible_masks, yolo_row


CLASSES = {'community': 0, 'noncommunity': 1}


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be nonempty text')
    return value


def _point(value, name):
    if (not isinstance(value, (list, tuple)) or len(value) != 2 or
            any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value)):
        raise ValueError(f'{name} must be a finite pixel point')
    return tuple(float(v) for v in value)


def _polygon(points, width, height, name):
    if not isinstance(points, (list, tuple)) or len(points) < 3:
        raise ValueError(f'{name} requires at least three vertices')
    poly = np.asarray([_point(p, name) for p in points], dtype=np.float32)
    if (any(not (0 <= x < width and 0 <= y < height) for x, y in poly) or
            abs(cv2.contourArea(poly)) < 1 or not cv2.isContourConvex(poly)):
        raise ValueError(f'{name} must be a nondegenerate convex in-frame polygon')
    return poly


def _box(mask):
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError('person has no visible pixels')
    return [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]


def _segment_intersects_polygon(start, end, polygon):
    """Check the whole contact segment, including touching an obstacle edge."""
    if any(cv2.pointPolygonTest(polygon, p, False) >= 0 for p in (start, end)):
        return True

    def cross(a, b, c):
        return (b[0]-a[0]) * (c[1]-a[1]) - (b[1]-a[1]) * (c[0]-a[0])

    def on_segment(a, b, p):
        return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and
                min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))

    for i, a in enumerate(polygon):
        b = polygon[(i + 1) % len(polygon)]
        c1, c2 = cross(start, end, a), cross(start, end, b)
        c3, c4 = cross(a, b, start), cross(a, b, end)
        if ((c1 == 0 and on_segment(start, end, a)) or
                (c2 == 0 and on_segment(start, end, b)) or
                (c3 == 0 and on_segment(a, b, start)) or
                (c4 == 0 and on_segment(a, b, end)) or
                ((c1 < 0 < c2 or c2 < 0 < c1) and (c3 < 0 < c4 or c4 < 0 < c3))):
            return True
    return False


def _foreground(image, identity_mask, actor):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 4:
        raise ValueError('foreground must be a uint8 BGRA image')
    if image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError('foreground is too small')
    size = actor.get('size_wh')
    if (not isinstance(size, (list, tuple)) or len(size) != 2 or
            any(type(v) is not int or not 2 <= v <= 8192 for v in size)):
        raise ValueError('size_wh must be two bounded positive integers')
    angles = actor.get('angles_degrees')
    if not isinstance(angles, dict) or set(angles) != {'yaw', 'pitch', 'roll'}:
        raise ValueError('yaw, pitch and roll are required')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in angles.values()):
        raise ValueError('angles must be finite numbers')
    projected = planar_warp(image, **angles)
    sprite = cv2.resize(projected, tuple(size), interpolation=cv2.INTER_LINEAR)
    evidence = None
    upper = None
    if identity_mask is not None:
        if (not isinstance(identity_mask, np.ndarray) or identity_mask.shape != image.shape[:2] or
                identity_mask.dtype != np.uint8):
            raise ValueError('identity mask must be uint8 and match the source image')
        if not np.any(identity_mask > 127) or np.any((identity_mask > 127) & (image[:, :, 3] <= 127)):
            raise ValueError('identity mask must be nonempty and inside the opaque foreground')
        marker = np.zeros_like(image)
        marker[:, :, 0] = identity_mask
        marker[:, :, 3] = image[:, :, 3]
        marker = planar_warp(marker, **angles)
        marker = cv2.resize(marker, tuple(size), interpolation=cv2.INTER_LINEAR)
        evidence = marker[:, :, 0] > 127
        if not np.any(evidence):
            raise ValueError('identity evidence vanished during transformation')
    if actor.get('min_upper_visible') is not None:
        # This is the upper quarter of the source foreground, not a face detector.
        # It also covers faces near either side of irregular artwork.
        marker = np.zeros_like(image)
        upper_end = max(1, round(image.shape[0] * 0.25))
        marker[:upper_end, :, 0] = image[:upper_end, :, 3]
        marker[:, :, 3] = image[:, :, 3]
        marker = planar_warp(marker, **angles)
        marker = cv2.resize(marker, tuple(size), interpolation=cv2.INTER_LINEAR)
        upper = marker[:, :, 0] > 127
        if not np.any(upper):
            raise ValueError('upper foreground evidence vanished during transformation')
    return sprite, evidence, upper


def compose_scene(background, plan, foregrounds, identity_masks):
    """Compose one candidate; arrays and source assessments are supplied by caller.

    plan has background_source, background_assessment, ground_polygon,
    obstacle_polygons, actors and required_overlaps. Each actor declares id,
    kind (person/vehicle), source, foot_xy, contact_width, size_wh, angles,
    and for person: view (front/back/side), identity and category. Front
    persons additionally declare identity_evidence and min_identity_visible.
    Actors and required overlap pairs are ordered from back to front. Returned
    masks use this order; labels contain only identifiable frontal people.
    """
    if (not isinstance(background, np.ndarray) or background.dtype != np.uint8 or
            background.ndim != 3 or background.shape[2] != 3):
        raise ValueError('background must be a uint8 BGR image')
    if not isinstance(plan, dict):
        raise ValueError('scene plan must be an object')
    _text(plan.get('background_source'), 'background source')
    assessment = plan.get('background_assessment')
    if (not isinstance(assessment, dict) or
            assessment.get('person_scope') not in ('candidate_empty', 'confirmed_empty', 'complete_existing', 'unknown') or
            not isinstance(assessment.get('evidence'), str) or not assessment['evidence'].strip()):
        raise ValueError('background requires evidence of complete person supervision')
    height, width = background.shape[:2]
    existing = assessment.get('existing_people', [])
    if not isinstance(existing, list) or (assessment['person_scope'] not in ('complete_existing', 'unknown') and existing):
        raise ValueError('background person declarations conflict with their scope')
    if assessment['person_scope'] == 'complete_existing' and not existing:
        raise ValueError('complete_existing requires existing people')
    existing_labels, background_boxes = [], []
    for item in existing:
        if not isinstance(item, dict) or set(item) != {'class_id', 'xyxy'}:
            raise ValueError('existing people require class_id and xyxy')
        if type(item['class_id']) is not int or item['class_id'] not in (0, 1):
            raise ValueError('unsupported existing person class')
        box = item['xyxy']
        if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box):
            raise ValueError('existing person box must use integer pixels')
        existing_labels.append(yolo_row(item['class_id'], box, width, height))
        background_boxes.append(box)
    avoid_boxes = assessment.get('avoid_boxes', [])
    if not isinstance(avoid_boxes, list):
        raise ValueError('background avoid boxes must be a list')
    for box in avoid_boxes:
        if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box):
            raise ValueError('avoid box must use integer pixels')
        yolo_row(0, box, width, height)
        background_boxes.append(box)
    ground = _polygon(plan.get('ground_polygon'), width, height, 'ground polygon')
    obstacles = plan.get('obstacle_polygons')
    if not isinstance(obstacles, list):
        raise ValueError('obstacle polygons must be a list')
    obstacles = [_polygon(p, width, height, 'obstacle polygon') for p in obstacles]
    actors = plan.get('actors')
    overlaps = plan.get('required_overlaps')
    if not isinstance(actors, list) or not isinstance(overlaps, list):
        raise ValueError('actors and required overlaps must be lists')
    if not isinstance(foregrounds, dict) or not isinstance(identity_masks, dict):
        raise ValueError('foreground and identity masks must be mappings')
    canvas = background.copy()
    masks, markers, upper_markers, records, ids, identities = [], [], [], [], {}, set()
    for actor in actors:
        if not isinstance(actor, dict):
            raise ValueError('actor must be an object')
        actor_id = _text(actor.get('id'), 'actor id')
        if actor_id in ids:
            raise ValueError('duplicate actor id')
        ids[actor_id] = len(records)
        kind = actor.get('kind')
        if kind not in ('person', 'vehicle'):
            raise ValueError('unsupported actor kind')
        source = _text(actor.get('source'), 'foreground source')
        if actor_id not in foregrounds:
            raise ValueError('missing foreground for actor')
        x, y = _point(actor.get('foot_xy'), 'foot_xy')
        contact = actor.get('contact_width')
        if isinstance(contact, bool) or not isinstance(contact, (int, float)) or not math.isfinite(contact) or contact <= 0:
            raise ValueError('contact_width must be positive')
        # Convex regions permit endpoint checking for the complete contact segment.
        segment = ((x-contact/2, y), (x+contact/2, y))
        if any(cv2.pointPolygonTest(ground, p, False) < 0 for p in segment):
            raise ValueError('contact segment outside ground')
        if any(_segment_intersects_polygon(*segment, p) for p in obstacles):
            raise ValueError('contact segment intersects obstacle')
        category, view, identity = actor.get('category'), actor.get('view'), actor.get('identity')
        front = kind == 'person' and view == 'front'
        if kind == 'person':
            if view not in ('front', 'back', 'side'):
                raise ValueError('person view must be front, back or side')
            if front:
                if category not in CLASSES:
                    raise ValueError('frontal person requires a known category')
                _text(identity, 'person identity')
                if identity in identities:
                    raise ValueError('duplicate frontal person identity')
                identities.add(identity)
                _text(actor.get('identity_evidence'), 'identity evidence')
                threshold = actor.get('min_identity_visible')
                if (isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or
                        not math.isfinite(threshold) or not 0 < threshold <= 1):
                    raise ValueError('frontal person requires an identity visibility threshold')
                if actor_id not in identity_masks:
                    raise ValueError('missing identity evidence mask')
            elif category is not None:
                raise ValueError('back and side people must not have a target category')
        elif category not in ('upright', 'fallen'):
            raise ValueError('vehicle category must be upright or fallen')
        sprite, marker, upper = _foreground(foregrounds[actor_id], identity_masks.get(actor_id) if front else None, actor)
        if contact > sprite.shape[1]:
            raise ValueError('contact width exceeds foreground width')
        mask = alpha_place(canvas, sprite, x, y)
        if any(np.any(mask[b[1]:b[3], b[0]:b[2]]) for b in background_boxes):
            raise ValueError('foreground overlaps a protected background object')
        marker_canvas = np.zeros((height, width), dtype=bool)
        upper_canvas = np.zeros((height, width), dtype=bool)
        if marker is not None:
            left, top = round(x-sprite.shape[1]/2), round(y-sprite.shape[0])
            marker_canvas[top:top+sprite.shape[0], left:left+sprite.shape[1]] = marker & (sprite[:, :, 3] > 127)
            if not marker_canvas.any():
                raise ValueError('identity evidence is not visible in the foreground')
        if upper is not None:
            left, top = round(x-sprite.shape[1]/2), round(y-sprite.shape[0])
            upper_canvas[top:top+sprite.shape[0], left:left+sprite.shape[1]] = upper & (sprite[:, :, 3] > 127)
        masks.append(mask)
        markers.append(marker_canvas)
        upper_markers.append(upper_canvas)
        records.append({'id': actor_id, 'kind': kind, 'category': category, 'view': view,
                        'identity': identity, 'source': source, 'foot_xy': [x, y],
                        'contact_width': contact, 'size_wh': list(actor['size_wh']),
                        'angles_degrees': dict(actor['angles_degrees'])})
    for pair in overlaps:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2 or
                any(not isinstance(value, str) for value in pair) or
                pair[0] not in ids or pair[1] not in ids or ids[pair[0]] >= ids[pair[1]]):
            raise ValueError('overlap must name distinct actors in back-to-front order')
        if not np.any(masks[ids[pair[0]]] & masks[ids[pair[1]]]):
            raise ValueError('required overlap has no intersecting pixels')
    visible = visible_masks(masks)
    labels = existing_labels[:]
    for i, (actor, record) in enumerate(zip(actors, records)):
        record['visible_fraction'] = float(visible[i].sum() / masks[i].sum())
        record['visible_bbox_xyxy'] = _box(visible[i]) if visible[i].any() else None
        if record['kind'] == 'person' and record['view'] == 'front':
            evidence_visible = int((markers[i] & visible[i]).sum()) / int(markers[i].sum())
            if evidence_visible < actor['min_identity_visible']:
                raise ValueError('identity evidence is obscured beyond the declared threshold')
            upper_min = actor.get('min_upper_visible')
            if upper_min is not None:
                upper_visible = int((upper_markers[i] & visible[i]).sum()) / int(upper_markers[i].sum())
                if upper_visible < upper_min:
                    raise ValueError('upper foreground is obscured beyond the declared threshold')
                record['upper_visible_fraction'] = upper_visible
                record['upper_visibility_basis'] = 'top quarter of source foreground; geometric approximation'
            occlusion_range = actor.get('occlusion_fraction_range')
            if occlusion_range is not None:
                occlusion = 1.0 - record['visible_fraction']
                # Fully unoccluded people remain valid. Real occlusion is bounded.
                if occlusion > 1e-9 and not occlusion_range[0] <= occlusion <= occlusion_range[1]:
                    raise ValueError('person occlusion outside the declared range')
                record['occlusion_fraction'] = occlusion
                record['occlusion_fraction_range'] = list(occlusion_range)
            record['identity_evidence'] = actor['identity_evidence']
            record['identity_visible_fraction'] = evidence_visible
            record['min_identity_visible'] = actor['min_identity_visible']
            labels.append(yolo_row(CLASSES[record['category']], record['visible_bbox_xyxy'], width, height))
    return {'image': canvas, 'labels': labels, 'full_masks': masks, 'visible_masks': visible,
            'objects': records, 'background_source': plan['background_source'],
            'background_assessment': dict(assessment), 'ground_polygon': plan['ground_polygon'],
            'obstacle_polygons': plan['obstacle_polygons'], 'required_overlaps': overlaps,
            'label_status': 'candidate_caller_assessed'}
