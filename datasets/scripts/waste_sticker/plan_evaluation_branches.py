"""Compile the real-surface Val and real-ground Test branches from assigned stock."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.config import source_path
from datasets.scripts.common.frame_environment import image_digest, query_frames


def working_image(row):
    paths = sorted(row['paths'], key=lambda path: not path.startswith('datasets/raw/shared/'))
    path = next(path for path in paths if (ROOT / path).is_file())
    if image_digest(ROOT / path) != row['content_sha256']:
        raise ValueError(f'Stored frame identity differs: {path}')
    return path


def surface_placement(face, class_id, image_size, crop, rng):
    width, height = image_size
    quad = np.float32(face['allowed_polygon'])
    face_w = np.linalg.norm(quad[1] - quad[0])
    face_h = np.linalg.norm(quad[3] - quad[0])
    warp = cv2.getPerspectiveTransform(np.float32([[0, 0], [face_w, 0], [face_w, face_h], [0, face_h]]), quad)
    safe = _mask(face['allowed_polygon'], width, height)
    for vertices in face['exclude_polygons'] + face['occluder_polygons']:
        safe &= ~_mask(vertices, width, height)
    for attempt in range(100):
        half = min(face_w, face_h) * rng.uniform(.48, .72) / 2 * (.87 ** (attempt // 25))
        center = (rng.uniform(.21, .79) * face_w, rng.uniform(.38, .73) * face_h)
        rotation = cv2.getRotationMatrix2D(center, rng.uniform(-22, 22), 1)
        offsets = np.float32([[-half, -half], [half, -half], [half, half], [-half, half]])
        rotated = cv2.transform(np.float32([offsets + center]), rotation)
        projected = np.rint(cv2.perspectiveTransform(rotated, warp)[0]).astype(int)
        if (np.any(projected < crop[:2]) or np.any(projected >= crop[2:]) or
                cv2.contourArea(projected, oriented=True) <= 0):
            continue
        paper = _mask(projected.tolist(), width, height, quad=True)
        if not np.any(paper & ~safe):
            return dict(bin_id=face['bin_id'], class_id=class_id, paper_quad=projected.tolist(),
                        identifiability='identifiable')
    raise ValueError(f'No valid placement on {face["bin_id"]}')


def surface_plan():
    from datasets.scripts.waste_sticker.produce_surface_dataset import compile_plan
    return compile_plan('val')


def ground_plan():
    assigned = [row for row in query_frames(task='waste_sticker', role='composite_background', split='test')
                if row['geometry'].get('validated_placement') and not row['geometry'].get('bins')]
    rows = []
    for row in assigned:
        ground = row['geometry']
        span = np.ptp(ground['ground_polygon'], axis=0)
        _, _, subject_width, subject_height = ground['validated_placement']['nominal_output_xywh']
        # Make room for the projected volume at a comparable scale. Narrow
        # alpha-contact probes remain assigned and available for other poses.
        if span[0] >= .70 * subject_width and span[1] >= .22 * subject_height:
            rows.append(row)
    if not rows:
        raise ValueError('No assigned Test whole-bin placement records')
    plates = [dict(source_photo=working_image(row), frame_id=row['frame_id'],
                   record_id=row['record_id'], parent=row['parent_group'], size=row['image_size'],
                   geometry=dict(ground_polygon=row['geometry']['ground_polygon']),
                   evidence_ids=[e['evidence_id'] for e in row['evidence']]) for row in rows]
    animation = json.loads(source_path('waste_sticker', 'animated_source_plan').read_text(encoding='utf8'))
    poses = {class_id: [] for class_id in range(24)}
    for sample in animation['samples']:
        for bin_record in sample['bins']:
            poses[bin_record['class_id']].append((sample['frame'], bin_record['bin_color']))
    jobs = []
    for i in range(40):
        index = i % len(rows)
        ground = rows[index]['geometry']
        points = ground['validated_placement']['contacts_output_xy']
        left, top, _, _ = ground['validated_placement']['output_content_xyxy']
        polygon = np.asarray(ground['ground_polygon'])
        width, height = rows[index]['image_size']
        frame, color = poses[(7*i + 9) % 24][i // 24 * 3]
        center_x = np.mean([p[0] - left for p in points])
        center_y = np.mean([p[1] - top for p in points])
        xmin, ymin = polygon.min(axis=0)
        xmax, ymax = polygon.max(axis=0)
        inset_x, inset_y = min(27, (xmax - xmin) // 2), min(9, (ymax - ymin) // 2)
        x = int(np.clip(center_x, xmin + inset_x, xmax - inset_x))
        y = int(np.clip(center_y - 6, ymin + inset_y, ymax - inset_y))
        # The contact evidence supplies the local anchor. Camera pitch is fitted
        # within the micro-sample's low-angle range and remains an approximation.
        horizon = ground.get('horizon_y')
        if horizon is None:
            horizon = round(.21 * height)
        jobs.append([index, frame, {color: [x, y]}, 82 + 6*(i // len(rows) % 3),
                     horizon, round(.78 * width), (-7, 3, 8)[i // len(rows) % 3]])
    return dict(model_key='waste_sticker', target_split='test', budget=len(jobs),
                independence_basis='One held-out parent video; real-scene diagnostic',
                assigned_ground_count=len(assigned), plates=plates, jobs=jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='write the validated plan declarations')
    args = parser.parse_args()
    val = surface_plan()
    test = ground_plan()
    print(f'Val: {val["budget"]} planned; {len({s["parameters"]["photo"] for s in val["strata"]})} assigned surfaces; '
          f'{dict(Counter(len(s["parameters"]["placements"]) for s in val["strata"]))}')
    print(f'Test: {test["budget"]} planned; {len(test["plates"])} of {test["assigned_ground_count"]} assigned real sites')
    if args.write:
        config = Path(__file__).resolve().parent / 'config'
        for name, plan in [('val_surface_plan.json', val), ('test_ground_plan.json', test)]:
            (config / name).write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
