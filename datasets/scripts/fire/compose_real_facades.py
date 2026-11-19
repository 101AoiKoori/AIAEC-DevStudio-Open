"""Produce the current fire sample from database-backed camera and Blender branches."""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json
import random
import re
import subprocess
import sys
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.annotations import yolo_row
from datasets.scripts.common.config import load_task_config, source_path
from datasets.scripts.common.frame_environment import image_digest, query_frames
from datasets.scripts.common.images import read_image
from datasets.scripts.common.paper import paste
from datasets.scripts.common.paths import sample_dir
from datasets.scripts.fire.wall_projection import place_on_wall, transform_points
from lab.training_common.production_guard import generation_guard

TASK = 'fire'
HERE = Path(__file__).resolve().parent
OUTPUT = sample_dir(TASK)


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def write_json(path, data):
    contents = json.dumps(data, ensure_ascii=False, indent=2) + '\n'
    if not path.is_file() or path.read_text(encoding='utf8') != contents:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding='utf8')


def identify_building(row, source):
    assignment = source.get('frame_buildings', {}).get(row['frame_id'])
    if assignment:
        return assignment
    matches = [key for key, name in source['building_aliases'].items()
               if name in row['parent_group']]
    if len(matches) != 1:
        raise ValueError(f'Building attribution needs a source observation: {row["frame_id"]}')
    return {'target_building': matches[0], 'visible_buildings': matches}


def current_inputs(source):
    database = ROOT / source['databases']['frame_environment']
    recipe = source['sample_recipe']
    direct = [row for row in query_frames(task=TASK, role=source['direct_query']['role'], database=database)
              if row['split'] in source['direct_query']['source_splits']
              and row['source_kind'] in source['direct_query']['source_kinds']]
    composite = [row for row in query_frames(task=TASK, role=source['composite_query']['role'], database=database)
                 if row['source_kind'] in source['composite_query']['source_kinds']
                 and row['split'] in source['composite_query']['source_splits']]
    uses = {}
    for branch, rows in (('direct_real', direct), ('real_background_composite', composite)):
        for row in sorted(rows, key=lambda r: r['frame_id']):
            label = row['label']
            if (row['scope'] != {'kind': 'frame'} or label['completeness'] != 'complete'
                    or image_digest(row['image']) != row['content_sha256']):
                raise ValueError(f'Invalid current fire source: {row["frame_id"]}')
            attribution = identify_building(row, source)
            if branch == 'direct_real':
                if label['presence'] != 'present' or not label['instances']:
                    raise ValueError('Direct positive must contain complete fire instances')
                variants = [(len(label['instances']), 'as_captured')]
            else:
                geometry = row['geometry']
                slots = [r for r in geometry['regions'] if r.get('kind') == source['composite_query']['regions_kind']]
                if not slots or not geometry.get('supervision_crop_xyxy'):
                    raise ValueError(f'Current facade needs reviewed paper slots: {row["frame_id"]}')
                walls = {wall['wall_id'] for wall in geometry.get('wall_surfaces', [])}
                if not walls or any(slot.get('wall_id') not in walls for slot in slots):
                    raise ValueError(f'Current facade needs a registered plane for each paper anchor: {row["frame_id"]}')
                variants = [(count, tone) for count in recipe['counts'] for tone in recipe['tones']]
                if max(recipe['counts']) > len(slots):
                    raise ValueError(f'Insufficient distinct wall slots: {row["frame_id"]}')
            for count, tone in variants:
                key = f'{branch}_{row["frame_id"][:12]}_{count}_{tone}'
                uses[key] = {'row': row, 'branch': branch, 'count': count, 'tone': tone, **attribution}
    if {u['branch'] for u in uses.values()} != {'direct_real', 'real_background_composite'}:
        raise ValueError('Current database returned no input for a requested camera branch')
    return uses


def renderer_plan(source):
    recipe = source['sample_recipe']
    parameters = {'scenario_index': None, 'scenario_offset': 0, 'seed': recipe['seed'],
                  'resolution': recipe.get('render_image_size', [640, 480]), 'samples': recipe['blender_samples'],
                  'device': recipe['device'], 'mode': 'standard'}
    targets = []
    # Nine spherical-strip positions per building. Each horizontal row contains 0/1/3 stickers.
    for offset, building in enumerate('ABC'):
        for position in range(9):
            count = recipe['counts'][(position + offset) % len(recipe['counts'])]
            targets.append(([building], {building: count}, position))
    targets += [(['A', 'B'], {'A': 0, 'B': 2}, 4),
                (['A', 'B'], {'A': 2, 'B': 0}, 4),
                (['B', 'C'], {'B': 1, 'C': 2}, 5),
                (['B', 'C'], {'B': 2, 'C': 1}, 5)]
    states = []
    for active, counts, position in targets:
        view = f'orange_peel_{position:02d}'
        states.append({
            'id': f'blender_explicit_{"".join(active)}_{"_".join(map(str, counts.values()))}_{view}',
            'failure_event': 'renderer_state_or_label_uncertainty', 'min_count': 1, 'max_count': 1,
            'prior': {'alpha': 1, 'beta': 1}, 'exposure_weight': 1, 'failure_cost': 1,
            'conditions': ['scenario:explicit', 'visibility:no_added_occluder', 'traffic_signal_injection:none'],
            'background': {'domain': 'synthetic_building_arena', 'traffic_signal_injection': 'none'},
            'target': {'scenario': 'explicit', 'active_buildings': active, 'fire_counts': counts,
                       'view': view, 'placement': 'all', 'visibility': 'no_added_occluder'},
            'labels': {name: {'class_id': 0, 'format': 'yolo_v8', 'status': 'candidate_requires_visual_review'}
                       for name in ('fire', 'name')},
            'augmentation': {'camera': view, 'lighting': 'seeded_random', 'sticker_placement': 'all',
                             'traffic_signal_injection': 'none'},
            'sources': ['blender_building_library', 'fire_sticker_artwork'], 'parameters': parameters})
    return {'schema_version': 1, 'model_key': 'fire_sticker', 'target_split': 'train',
            'purpose': 'fire spherical-strip camera sample with per-building counts',
            'budget': len(states), 'uncertainty_weight': .5,
            'independence_basis': 'shared Buildings scene; development sample',
            'constraints': {'route_context': 'building_facade',
                            'traffic_signal_cooccurrence': 'forbidden_separate_route',
                            'traffic_signal_generation': 'no_traffic_signal_injection',
                            'unknown_visual_targets': 'suppress_labels'}, 'strata': states}


def production_plan(uses, renderer, artwork, source):
    states = deepcopy(renderer['strata'])
    for key, use in uses.items():
        row = use['row']
        sources = [row['image']]
        if use['branch'] == 'real_background_composite' and use['count']:
            sources.append(artwork)
        states.append({'id': key, 'failure_event': 'sticker_miss_or_false_positive',
            'min_count': 1, 'max_count': 1, 'prior': {'alpha': 1, 'beta': 1},
            'exposure_weight': 1, 'failure_cost': 1,
            'conditions': [use['branch'], f'building:{use["target_building"]}',
                           f'count:{use["count"]}', f'tone:{use["tone"]}'],
            'background': {'image': row['image'], 'parent_group': row['parent_group'],
                           'record_id': row['record_id']},
            'target': {'building': use['target_building'], 'count': use['count']},
            'labels': {'class_id': 0, 'scope': 'entire_output'},
            'augmentation': {'tone': use['tone'], 'max_image_side': source['sample_recipe']['max_image_side'],
                             'sticker_size': source['sample_recipe']['sticker_size'],
                             'wall_projection': source['sample_recipe']['wall_projection']},
            'sources': sources, 'parameters': {'seed': source['sample_recipe']['seed'],
                'record_id': row['record_id'], 'frame_id': row['frame_id'],
                'source_sha256': row['content_sha256'], 'source_split': row['split']}})
    return {'schema_version': 1, 'model_key': 'fire_sticker', 'target_split': 'train',
            'purpose': 'current three-branch representative fire sample',
            'budget': len(states), 'uncertainty_weight': .5,
            'independence_basis': 'development sample; preserve source splits and historical contact',
            'constraints': {'supervision': 'complete_visible_targets_in_output',
                            'building_counts': 'scoped_to_covered_visible_surfaces'},
            'strata': states}


@lru_cache(maxsize=3)
def source_image(path, digest):
    return read_image(ROOT/path)


def compose_camera(use, paper, recipe, seed):
    row = use['row']
    image = source_image(row['image'], row['content_sha256'])
    if list(image.shape[1::-1]) != row['image_size']:
        raise ValueError('Decoded image dimensions differ from the database')
    crop = [0, 0, *row['image_size']]
    if use['branch'] == 'real_background_composite':
        crop = row['geometry']['supervision_crop_xyxy']
    x0, y0, x1, y1 = crop
    if not 0 <= x0 < x1 <= image.shape[1] or not 0 <= y0 < y1 <= image.shape[0]:
        raise ValueError('Fire supervision crop leaves the current source')
    image = image[y0:y1, x0:x1]
    ratio = min(1, recipe['max_image_side'] / max(image.shape[:2]))
    size = [round(image.shape[1] * ratio), round(image.shape[0] * ratio)]
    scale = np.asarray(size) / image.shape[1::-1]
    image = cv2.resize(image, tuple(size), interpolation=cv2.INTER_AREA)
    instances = []
    for item in row['label']['instances']:
        a, b, c, d = item['xyxy']
        if not x0 <= a < c <= x1 or not y0 <= b < d <= y1:
            raise ValueError('An existing fire target leaves the output supervision scope')
        box = ((np.asarray([[a, b], [c, d]]) - [x0, y0]) * scale).reshape(-1).tolist()
        instances.append({'xyxy': box, 'building': use['target_building'],
                          'object_id': item.get('object_id'), 'origin': 'existing'})
    rng = random.Random(seed)
    gain = 1
    if use['branch'] == 'real_background_composite':
        slots = [r for r in row['geometry']['regions'] if r.get('kind') == 'fire_paper_slot']
        facade = np.float32(next(r['polygon'] for r in row['geometry']['regions'] if r.get('kind') == 'facade'))
        facade_height = float((np.linalg.norm(facade[3]-facade[0]) + np.linalg.norm(facade[2]-facade[1])) / 2)
        policy = recipe['sticker_size']
        lower, upper = policy['height_fraction'][use['target_building']]
        surfaces = {surface['wall_id']: surface for surface in row['geometry']['wall_surfaces']}
        placements = []
        for attempt in range(policy['placement_attempts']):
            placements = []
            for slot in rng.sample(slots, use['count']):
                fraction = rng.uniform(lower, upper)
                projected = place_on_wall(surfaces[slot['wall_id']], slot['quad_px'], fraction,
                    facade_height, policy['anchor_jitter_fraction'], paper.shape[1]/paper.shape[0],
                    recipe['wall_projection']['margin_source_px'], rng)
                if projected is None:
                    break
                quad = (np.float32(projected['quad_source_px']) - [x0, y0]) * scale
                if (not np.isfinite(quad).all() or
                        any(not 1 <= x < size[0]-1 or not 1 <= y < size[1]-1 for x, y in quad) or
                        any(cv2.intersectConvexConvex(quad.astype('float32'), item['quad'])[0] > 0
                            for item in placements)):
                    break
                placements.append({'slot': slot, 'quad': quad.astype('float32'),
                                   **projected})
            if len(placements) == use['count']:
                break
        if len(placements) != use['count']:
            raise ValueError(f'No non-overlapping scaled sticker layout for {row["frame_id"]}')
        for placement in placements:
            slot, quad = placement['slot'], placement['quad']
            cx, cy = map(round, quad.mean(axis=0))
            illumination = float(np.clip(np.median(image[cy-3:cy+4, cx-3:cx+4]) / 190, .73, 1.03))
            box = paste(image, np.clip(paper * illumination, 0, 255).astype('uint8'), quad)
            if min(box[2]-box[0], box[3]-box[1]) < 12:
                raise ValueError('The output sticker is below this batch discernibility size')
            instances.append({'xyxy': box, 'quad_xy': quad.tolist(), 'building': slot['building'],
                              'origin': 'placed', 'slot_id': slot['slot_id'],
                              **{key: value for key, value in placement.items() if key not in ('slot', 'quad')},
                              'reference_facade_height_px': facade_height})
        gain = rng.uniform(*recipe['tones'][use['tone']])
        image = np.clip(image.astype('float32') * gain, 0, 255).astype('uint8')
    if len(instances) != use['count']:
        raise ValueError('Actual camera-branch instances differ from the plan')
    return image, instances, {'crop_xyxy': crop, 'output_size': size, 'brightness_gain': gain}


def counts_for(buildings, instances, scope, complete=False):
    return {building: {'visible_count': sum(i['building'] == building for i in instances),
            'whole_building_count': None, 'covered_placement_count': (
                sum(i['building'] == building for i in instances) if complete else None),
            'scope': scope, 'coverage_complete': complete} for building in buildings}


def output_records():
    path = OUTPUT / 'manifest.json'
    if path.is_file():
        return json.loads(path.read_text(encoding='utf8'))['records']
    path = OUTPUT / 'manifest.jsonl'
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()] if path.is_file() else []


def generated_files():
    owned = set()
    for row in output_records():
        for field in ('image', 'label', 'fire_label', 'instance_pass', 'overlay', 'render_layer'):
            if row.get(field):
                path = (OUTPUT / row[field]).resolve()
                if not path.is_relative_to(OUTPUT):
                    raise ValueError('Existing fire manifest escapes the task sample')
                owned.add(path)
    owned.update(OUTPUT / name for name in ('manifest.json', 'manifest.jsonl', 'run.json',
                                           'validation.json', 'contact_sheet.jpg'))
    # An interrupted encode/render can leave its image before the manifest row.
    # These are the task's own generated filenames in the declared output folders.
    for directory in ('images/train', 'images/val', 'images/test', 'labels/train', 'labels/val', 'labels/test',
                      'labels/fire/train', 'instance_pass', 'overlays', 'render_layers'):
        for path in (OUTPUT / directory).glob('*'):
            if (path.suffix.lower() in {'.png', '.jpg', '.txt'} and
                    re.fullmatch(r'(fire_\d{6}|hard_negative_\d{6}|contrast_positive_\d{6}|small_sticker_\d{6}|direct_real_\d{4}|real_background_composite_\d{4})', path.stem)):
                owned.add(path)
    return {path for path in owned if path.is_file()}


def validate_records(records):
    histogram = Counter()
    projected_count, contact_count = 0, 0
    for row in records:
        image = read_image(OUTPUT / row['image'])
        height, width = image.shape[:2]
        lines = (OUTPUT / row['label']).read_text(encoding='utf8').splitlines()
        if len(lines) != len(row['instances']):
            raise ValueError('Fire label count differs from the instance manifest')
        expected = [yolo_row(i['xyxy'], width, height) for i in row['instances']]
        if lines != expected:
            raise ValueError('Fire YOLO geometry differs from measured instances')
        if sum(c['visible_count'] for c in row['building_counts'].values()) != len(lines):
            raise ValueError('Fire building attribution does not partition the labels')
        for instance in row['instances']:
            if instance['origin'] == 'placed':
                matrix = np.array(instance['wall_homography'])
                restored = transform_points(instance['quad_source_px'], np.linalg.inv(matrix))
                wall_quad = np.array(instance['quad_wall_xy'])
                if (not np.allclose(restored, wall_quad, atol=.003, rtol=0) or
                        not np.allclose(wall_quad[[0, 1], 1], wall_quad[0, 1], atol=.003) or
                        not np.allclose(wall_quad[[2, 3], 1], wall_quad[2, 1], atol=.003) or
                        not np.allclose(wall_quad[[0, 3], 0], wall_quad[0, 0], atol=.003) or
                        not np.allclose(wall_quad[[1, 2], 0], wall_quad[1, 0], atol=.003)):
                    raise ValueError('Fire paper is not a rectangle on its assigned wall plane')
                boundary = np.float32(instance['wall_boundary_px'])
                if any(cv2.pointPolygonTest(boundary, tuple(point), True) < 0
                       for point in instance['quad_source_px']):
                    raise ValueError('A fire paper crosses its assigned wall boundary')
                aspect = (wall_quad[1, 0]-wall_quad[0, 0])/(wall_quad[3, 1]-wall_quad[0, 1])
                if abs(aspect-instance['paper_aspect_ratio']) > 1e-5:
                    raise ValueError('Fire paper aspect differs from its rectified artwork')
                crop = row['transform']['crop_xyxy']
                scale = np.asarray(row['transform']['output_size']) / [crop[2]-crop[0], crop[3]-crop[1]]
                output_quad = (np.array(instance['quad_source_px'])-crop[:2])*scale
                if not np.allclose(output_quad, instance['quad_xy'], atol=.003, rtol=0):
                    raise ValueError('Fire paper source and output coordinate transforms disagree')
                projected_count += 1
            elif instance['origin'] == 'rendered':
                contact = instance['wall_contact']
                if contact['max_plane_error_m'] > 1e-6 or contact['offset_m'] < 0:
                    raise ValueError('Rendered fire paper is not aligned with its wall plane')
                contact_count += 1
        histogram[(row['branch'], len(lines))] += 1
    return {'images': len(records), 'instances': sum(len(r['instances']) for r in records),
            'by_branch': dict(Counter(r['branch'] for r in records)),
            'counts_by_branch': [{'branch': b, 'count': n, 'images': value} for (b, n), value in histogram.items()],
            'checks': ['decode', 'image_label_pairing', 'bbox_bounds', 'exact_yolo_rows',
                       'instance_count', 'per_building_partition', 'wall_homography',
                       'single_wall_boundary', 'rectified_paper_aspect', 'blender_wall_plane_contact'],
            'wall_projected_instances': projected_count, 'wall_contact_instances': contact_count,
            'whole_building_counts': 'unknown; frame or crop supervision covers visible detectable targets'}


def overview_indices(records, limit=240):
    if len(records) <= limit:
        return list(range(len(records)))
    selected = {}
    for index, row in enumerate(records):
        camera = row.get('camera', {})
        key = (row['branch'], row.get('frame_id') or row.get('photo_frame_id'), row.get('tone'),
               tuple((b, c['visible_count']) for b, c in sorted(row['building_counts'].items())),
               camera.get('view_profile'), camera.get('trajectory_index'))
        selected.setdefault(key, index)
    values = list(selected.values())
    return values if len(values) <= limit else [values[i] for i in np.linspace(0,len(values)-1,limit,dtype=int)]


def previews(records):
    folder = OUTPUT / 'overlays'
    folder.mkdir(exist_ok=True)
    colors = {'A': (30, 220, 70), 'B': (240, 180, 20), 'C': (40, 190, 240)}
    selected = {index: position for position, index in enumerate(overview_indices(records))}
    sheet = Image.new('RGB', (1600, ((len(selected)+3)//4)*320), '#ececec')
    for index, row in enumerate(records):
        image = Image.open(OUTPUT / row['image']).convert('RGB')
        draw = ImageDraw.Draw(image)
        for instance in row['instances']:
            box = instance['xyxy']
            draw.rectangle(box, outline=colors[instance['building']], width=2)
            draw.text((box[0], max(0, box[1]-13)), instance['building'], fill=colors[instance['building']],
                      stroke_width=1, stroke_fill='black')
        overlay = f'overlays/{Path(row["image"]).stem}.jpg'
        image.save(OUTPUT / overlay, quality=93)
        row['overlay'] = overlay
        if index not in selected:
            continue
        image.thumbnail((400, 286))
        position = selected[index]
        x, y = position % 4*400, position //4*320
        sheet.paste(image, (x+(400-image.width)//2, y))
        values = []
        for key, value in row['building_counts'].items():
            count = value['visible_count']
            crop = row['branch'] == 'real_background_composite'
            shown = '?' if count == 0 and not value['coverage_complete'] and not crop else str(count)
            values.append(f'{key} {"crop" if crop else "visible"}={shown}')
        summary = ' '.join(values)
        ImageDraw.Draw(sheet).text((x+5,y+288), f'{index+1} {row["branch"]} {summary}', fill='black')
    sheet.save(OUTPUT / 'contact_sheet.jpg', quality=91)


def main(argv=None):
    from datasets.scripts.fire.build_dataset import main as build
    return build(argv)


if __name__ == '__main__':
    main()
