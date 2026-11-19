"""Compile and produce the current full-run perspective surface branches."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.frame_environment import query_frames
from datasets.scripts.waste_sticker.generate_sticker_real_photo_v4 import (
    COMPLETE_SURFACE_CONSTRAINTS, _mask, _produce_surface, validate_surface_plan, artwork_generator,
)
from datasets.scripts.common.config import source_path
from datasets.scripts.common.working_inputs import WorkingInputs
from datasets.scripts.waste_sticker.plan_evaluation_branches import working_image
from datasets.scripts.waste_sticker.source_inputs import artwork_catalog, surface_background_catalog
from lab.training_common.production_guard import plan_production

OUTPUT = ROOT / 'datasets/annotated/waste_bin/waste_sticker_full/surface'
CONFIG = Path(__file__).resolve().parent / 'config/surface_full_plan.json'


def paper_on_face(face, class_id, size, crop, rng, material_mask=None):
    width, height = size
    quad = np.float32(face['allowed_polygon'])
    face_width = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
    face_height = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
    projected = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [face_width, 0], [face_width, face_height], [0, face_height]]), quad)
    safe = _mask(face['allowed_polygon'], width, height)
    for polygon in face['exclude_polygons'] + face['occluder_polygons']:
        safe &= ~_mask(polygon, width, height)
    if material_mask is not None:
        safe &= material_mask
    scale = rng.uniform(.62, .88) if face.get('source_material_check') else rng.uniform(.53, .79)
    for attempt in range(180):
        half = min(face_width, face_height) * scale * (.90 ** (attempt // 45)) / 2
        angle = math.radians(rng.uniform(-23, 23))
        rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        center = (rng.uniform(.20, .80) * face_width, rng.uniform(.19, .82) * face_height)
        corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]]) @ rotation.T + center
        paper = np.rint(cv2.perspectiveTransform(np.float32([corners]), projected)[0]).astype(int)
        if (np.any(paper < crop[:2]) or np.any(paper >= crop[2:]) or
                cv2.contourArea(paper, oriented=True) <= 0):
            continue
        mask = _mask(paper.tolist(), width, height, quad=True)
        if not np.any(mask & ~safe):
            return dict(bin_id=face['bin_id'], class_id=class_id,
                        paper_quad=paper.tolist(), identifiability='identifiable')
    raise ValueError(f'No feasible placement for {face["bin_id"]} at crop {crop}')


def source_plastic_masks(row, photo):
    if not any(face.get('source_material_check') for face in row['geometry']['bins']):
        return {}
    image = cv2.imdecode(np.fromfile(ROOT / photo, np.uint8), cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    colors = {
        'green': (hue > 42) & (hue < 90) & (saturation > 80) & (value > 35),
        'red': ((hue < 20) | (hue > 170)) & (saturation > 120) & (value > 85),
        'blue': (hue > 95) & (hue < 135) & (saturation > 100) & (value > 65),
    }
    result = {}
    for face in row['geometry']['bins']:
        color = face['bin_color']
        if color not in colors:
            continue
        mask = cv2.morphologyEx(colors[color].astype(np.uint8), cv2.MORPH_CLOSE,
                               np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        width, height = row['image_size']
        expected = _mask(face['allowed_polygon'], width, height)
        candidates = []
        for contour in contours:
            if cv2.contourArea(contour) <= 100:
                continue
            hull = np.zeros_like(mask)
            cv2.drawContours(hull, [cv2.convexHull(contour)], -1, 1, cv2.FILLED)
            candidates.append((int(np.count_nonzero(hull.astype(bool) & expected)), hull))
        if candidates:
            filled = max(candidates, key=lambda item: item[0])[1]
        result[face['bin_id']] = filled.astype(bool)
    return result


def crop_faces(row, count, box_catalog, rng):
    width, height = row['image_size']
    faces = sorted(row['geometry']['bins'], key=lambda face: np.mean(np.asarray(face['allowed_polygon'])[:, 0]))
    starts = list(range(len(faces) - count + 1))
    rng.shuffle(starts)
    boxes = box_catalog.get(row['frame_id'])
    for start in starts:
        stop = start + count
        selected = faces[start:stop]
        selected_points = np.concatenate([np.asarray(face['allowed_polygon']) for face in selected])
        if boxes:
            left = max([boxes[face['bin_id']][2] for face in faces[:start]], default=0)
            right = min([boxes[face['bin_id']][0] for face in faces[stop:]], default=width)
            top = max(0, min(boxes[face['bin_id']][1] for face in selected) - round(height * .08))
            bottom = min(height, max(boxes[face['bin_id']][3] for face in selected) + round(height * .09))
        else:
            x_min, y_min = selected_points.min(axis=0)
            x_max, y_max = selected_points.max(axis=0)
            face_height = max(np.ptp(face['allowed_polygon'], axis=0)[1] for face in selected)
            left = (max(x for x, _ in faces[start-1]['allowed_polygon']) + x_min) // 2 if start else max(0, x_min - 65)
            right = (min(x for x, _ in faces[stop]['allowed_polygon']) + x_max) // 2 if stop < len(faces) else min(width, x_max + 65)
            top = max(0, y_min - round(face_height * 1.20))
            bottom = min(height, y_max + round(face_height * .35))
        centers = [np.mean(np.asarray(face['allowed_polygon'])[:, 0]) for face in selected]
        body_coverage = (not boxes or all(
            max(0, min(right, boxes[face['bin_id']][2]) - max(left, boxes[face['bin_id']][0])) >=
            .65 * (boxes[face['bin_id']][2] - boxes[face['bin_id']][0])
            for face in selected))
        if left < min(centers) and max(centers) < right and body_coverage:
            return selected, [int(left), int(top), int(right), int(bottom)]
    raise ValueError(f'No {count}-bucket output crop for {row["frame_id"]}')


def compile_plan(split):
    rng = random.Random(2026111803 + (split == 'val'))
    available = [row for row in query_frames(task='waste_sticker', role='composite_background', split=split)
                 if row.get('geometry', {}).get('bins')]
    box_catalog = {
        row['frame_id']: {item['object_id']: item['xyxy'] for item in row['label']['instances']
                          if item.get('object_id') and item.get('xyxy')}
        for row in query_frames(task='waste_bin_state', role='direct_target', split=split)
    }
    images = {row['frame_id']: working_image(row) for row in available}
    material_masks = {row['frame_id']: source_plastic_masks(row, images[row['frame_id']])
                      for row in available}
    catalog = surface_background_catalog(images=images.values(), split=split)
    art = artwork_catalog()
    classes = [Path(art[index]).stem for index in range(24)]
    branch_specs = [('blender_surface', {1: 67, 2: 67, 3: 68, 4: 68}),
                    ('real_surface', {1: 23, 2: 22, 3: 23, 4: 22})] if split == 'train' else [
                        ('real_surface', {1: 20, 2: 20, 3: 20, 4: 20})]
    instances = 901 if split == 'train' else 200
    class_ids = [index % 24 for index in range(instances)]
    rng.shuffle(class_ids)
    states = []
    for branch, counts in branch_specs:
        rows = [row for row in available if (row.get('source_kind') == 'blender_render') == (branch == 'blender_surface')]
        if not rows:
            raise ValueError(f'Missing {split}/{branch} input')
        requests = [count for count, amount in counts.items() for _ in range(amount)]
        rng.shuffle(requests)
        for index, count in enumerate(requests):
            # Some perspective crops overlap the neighboring bucket's rectangle.
            # Retry the contiguous alternatives while keeping all added paper in view.
            for attempt in range(max(16, len(rows) * 2)):
                row = rows[(index + attempt // 2) % len(rows)]
                photo = images[row['frame_id']]
                pending = class_ids[-count:]
                try:
                    selected, crop = crop_faces(row, count, box_catalog, rng)
                    placements = [paper_on_face(face, class_id, row['image_size'], crop, rng,
                                                material_masks[row['frame_id']].get(face['bin_id']))
                                  for face, class_id in zip(selected, pending)]
                    break
                except ValueError:
                    if attempt == max(16, len(rows) * 2) - 1:
                        raise
            del class_ids[-count:]
            tone = dict(gain=round(rng.uniform(.91, 1), 4), saturation=round(rng.uniform(.90, 1), 4))
            parameters = dict(photo=photo, placements=placements, output_crop_xyxy=crop, **tone)
            used = sorted({placement['class_id'] for placement in placements})
            states.append(dict(
                id=f'{split}_{branch}_{index:04d}', min_count=1, max_count=1,
                prior=dict(alpha=2, beta=8), exposure_weight=1, failure_cost=1,
                failure_event='visible_sticker_label_incomplete', conditions=[branch],
                background=dict(kind=catalog[photo]['background_kind'], image=photo, record_id=catalog[photo]['record_id']),
                target=dict(kind='explicit_bin_stickers', placements=placements),
                labels=dict(format='yolo_detection', status=COMPLETE_SURFACE_CONSTRAINTS['labels']),
                augmentation=dict(placement='explicit_surface_quads', **tone),
                parameters=parameters, sources=[photo, *[art[identity] for identity in used]]))
    assert not class_ids
    plan = dict(schema_version=1, model_key='waste_sticker', target_split=split, budget=len(states),
                purpose='Current full-dataset perspective surface branches',
                independence_basis='Qualified source frames and assigned real parent videos; Blender shares one asset',
                uncertainty_weight=0, constraints=COMPLETE_SURFACE_CONSTRAINTS, strata=states)
    validate_surface_plan(plan, classes, None, background_roles=catalog)
    plan_production(plan, ROOT)
    return plan


def normalize(split):
    out = OUTPUT / split
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf8'))
    states = {state['id']: state for state in manifest['plan']['strata']}
    records = []
    for record in manifest['records']:
        actual = record['actual']
        state = states[record['state_id']]
        branch = state['conditions'][0]
        instances = actual['instances']
        records.append(dict(
            id=record['state_id'], image=record['image'], label=record['label_candidate'],
            branch=branch, split=split, source_photo=actual['source_photo'],
            source_parent=actual['background_role']['parent_group'],
            source_decision=actual['background_role']['record_id'],
            source_evidence=actual['background_role']['evidence_ids'],
            placement_geometry=actual['background_role']['geometry'],
            source_crop_xyxy=actual['output_crop_xyxy'], image_size=actual['output_size'],
            class_ids=[item['class_id'] for item in instances], instances=instances,
            retained_background_instances=actual['retained_background_instances'],
            visible_bins=[item['bin_id'] for item in state['parameters']['placements']],
            gain=actual['gain'], saturation=actual['saturation']))
    manifest['records'] = records
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return records


def contact_sheet(split, records):
    out = OUTPUT / split
    indices = np.linspace(0, len(records) - 1, 24, dtype=int)
    sheet = Image.new('RGB', (1200, 1000), (25, 27, 30))
    draw = ImageDraw.Draw(sheet)
    for index, record_index in enumerate(indices):
        record = records[record_index]
        with Image.open(ROOT / record['image']) as im:
            thumb = ImageOps.contain(im.convert('RGB'), (290, 135))
        left, top = 300 * (index % 4), 166 * (index // 4)
        sheet.paste(thumb, (left + (300 - thumb.width) // 2, top))
        draw.text((left + 5, top + 138), record['id'], fill=(235, 238, 240))
    sheet.save(out / 'contact_sheet.jpg', quality=90)


def produce(split, plan):
    out = OUTPUT / split
    out.mkdir(parents=True, exist_ok=True)
    art = artwork_catalog()
    toolkit = source_path('waste_sticker', 'artwork_toolkit')
    names = [Path(art[index]).stem for index in range(24)]
    working = WorkingInputs('waste_sticker',
                            backgrounds=[state['parameters']['photo'] for state in plan['strata']],
                            targets=art.values(), split=split)
    icons_paths = [working.path(art[index]) for index in range(24)]
    if len({path.parent for path in icons_paths}) != 1:
        raise ValueError('The fixed artwork working set needs one material directory')
    cache = OUTPUT / 'asset_cache'
    artwork_generator.init_worker(toolkit, cache, out, 1280, 720, False, icon_dir=icons_paths[0].parent)
    icons = []
    for name in names:
        with np.load(cache / (name + '.npz')) as asset:
            icons.append((asset['rgb'].copy(), asset['alpha'].astype(np.float32)))
    catalog = surface_background_catalog(images={state['parameters']['photo'] for state in plan['strata']}, split=split)
    records_path = out / 'batch.jsonl'
    with records_path.open('w', encoding='utf8') as batch:
        def record_sample(state_id, *, output, sources, parameters, actual):
            batch.write(json.dumps(dict(state_id=state_id, output=output.relative_to(ROOT).as_posix(),
                                        sources=sources, source_decision=actual['background_role']['record_id'],
                                        parameters=parameters), ensure_ascii=False) + '\n')
        # These are current-run working images. The parent publishes all four
        # branches through one production_guard and registers the final receipt.
        run = SimpleNamespace(plan=plan_production(plan, ROOT), record_sample=record_sample)
        _produce_surface(out, run, names, None, icons, 2026111803, formal=True,
                         background_roles=catalog, working=working)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--split', choices=('train', 'val'), action='append')
    args = parser.parse_args()
    plans = json.loads(CONFIG.read_text(encoding='utf8')) if CONFIG.is_file() else {}
    selected_splits = args.split or ('train', 'val')
    for split in selected_splits:
        plans[split] = compile_plan(split)
        if split == 'val':
            (CONFIG.parent / 'val_surface_plan.json').write_text(
                json.dumps(plans[split], ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    CONFIG.write_text(json.dumps(plans, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for split in selected_splits:
        plan = plans[split]
        plan_path = OUTPUT / f'{split}_plan.json'
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        print(split, plan['budget'], dict(Counter(state['conditions'][0] for state in plan['strata'])), flush=True)
        if args.plan_only:
            continue
        produce(split, plan)
        records = normalize(split)
        contact_sheet(split, records)
        print('COMPLETE', split, len(records), flush=True)


if __name__ == '__main__':
    main()
