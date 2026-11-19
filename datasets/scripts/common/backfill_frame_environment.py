"""Import current V4 source evidence into the durable frame environment database."""

from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import yaml

from .config import load_task_config
from .frame_environment import DEFAULT_DATABASE, SPLITS, edit_database, query_frames, query_split_candidates, quota_summary
from .paths import TASKS, WORKSPACE
from .source_history import historical_source, read_source_history


RAW = WORKSPACE / 'datasets/raw'
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif', '.ppm'}
SOURCE_COPIES = 'datasets/source_copy_manifest.jsonl'
WHITE_BACK = 'datasets/annotated/common/sources/targets/person/white_back/manifest.json'
CAR_PERSON = 'datasets/annotated/car_real_capture_20261112/person_direct_captures.json'
PERSON_DISPATCH = 'datasets/scripts/person/config/background_dispatch.json'
PERSON_ISSUES = 'datasets/scripts/person/config/sample_issues.yaml'
PERSON_B6 = 'datasets/annotated/common/review/person/v4_review_20261115_b6/metrics.jsonl'
ARENA_PERSON = 'datasets/annotated/person/arena_route_backgrounds/manifest.json'
CAMERA_SURFACES = (
    'datasets/annotated/car_real_capture_20261112/sequence_visual_review_manifest.jsonl',
    'datasets/annotated/car_real_capture_20261112/sequence_visual_review_part2_manifest.jsonl',
    'datasets/annotated/car_real_capture_20261112/waypoint_surface_manifest.jsonl',
    'datasets/annotated/car_real_capture_20261112/sequence_ground_surface_manifest.jsonl',
)
CAMERA_VIEWS = (
    'datasets/annotated/car_real_capture_20261112/sequence_view_part1_manifest.jsonl',
    'datasets/annotated/car_real_capture_20261112/sequence_view_part2_manifest.jsonl',
)
GROUND_TASKS = {'person', 'e_bike', 'sign', 'waste_bin_state'}


@lru_cache(maxsize=None)
def _read(path):
    return json.loads((WORKSPACE / path).read_text(encoding='utf-8-sig'))


@lru_cache(maxsize=None)
def _file_info(path):
    return _image_info((WORKSPACE / path).read_bytes())


def _image_info(raw):
    with Image.open(BytesIO(raw)) as image:
        size = list(image.size)
        properties = {'mode': image.mode, 'format': image.format, 'image_size': size}
    return size, sha256(raw).hexdigest(), properties


def _reference(manifest, index):
    return {'manifest': manifest, 'record_index': index}


def _path_observations(manifests):
    records = {}
    for manifest in manifests:
        with (WORKSPACE / manifest).open(encoding='utf-8-sig') as stream:
            for index, line in enumerate(stream):
                if line.strip():
                    row = json.loads(line)
                    records[row['path']] = (row, _reference(manifest, index))
    return records


def _split(value):
    return value if value in SPLITS else 'unassigned'


def _rect(value):
    x0, y0, x1, y1 = value
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _ground_record(writer, frame_id, metadata, task, geometry, label, evidence, pixels):
    """Run only for a declared ground-composite candidate with missing contact evidence."""
    import cv2
    import numpy as np

    from .review.placement import assess_placeable_ground
    from .review.placement_record import ALGORITHM_VERSION, encode_placement_record

    native = geometry.get('native_ground') or {}
    candidate = native.get('polygon') or geometry.get('placement_patch') or geometry.get('ground_polygon')
    if not candidate and geometry.get('support_rectangle_xyxy'):
        candidate = _rect(geometry['support_rectangle_xyxy'])
    if not candidate:
        return None, None, 'ground_polygon_missing'
    width, height = metadata['image_size']
    base_width, base_height = [width, height] if native else geometry.get('coordinate_size', [width, height])
    crop = None if native else geometry.get('crop_xyxy')
    if crop:
        sx0, sy0, sx1, sy1 = crop
        if not (0 <= sx0 < sx1 <= width and 0 <= sy0 < sy1 <= height):
            return None, None, 'ground_coordinate_basis_unresolved'
    elif [base_width, base_height] == [width, height] or (
        task == 'person' and abs(base_width * height - width * base_height) < max(width, height)
    ):
        sx0, sy0, sx1, sy1 = 0, 0, width, height
    else:
        return None, None, 'ground_coordinate_basis_unresolved'

    def source_xy(point):
        x, y = point
        return [
            round(min(sx1 - 1, max(sx0, sx0 + (x + .5) * (sx1 - sx0) / base_width - .5)), 4),
            round(min(sy1 - 1, max(sy0, sy0 + (y + .5) * (sy1 - sy0) / base_height - .5)), 4),
        ]

    polygon = [source_xy(point) for point in candidate]
    boxes = []
    for item in label['instances']:
        if box := item.get('xyxy') or item.get('box_xyxy'):
            boxes.append(box)
    for box in geometry.get('protected_rectangles_xyxy', []):
        boxes.append([*source_xy(box[:2]), *source_xy(box[2:])])

    # Version the input evidence and basis along with the space algorithm.
    from .review.ground_space import GroundSpace

    source_quad = native.get('reference_quad_px') or (polygon if len(polygon) == 4 else None)
    horizon = native.get('horizon_y', geometry.get('horizon_y'))
    floor_basis = native.get('floor_mask_basis')
    if source_quad:
        try:
            GroundSpace(source_quad, image_shape=(height, width), horizon_y=horizon)
        except ValueError:
            source_quad = None
    source_basis = {'reference': evidence, 'legacy_coordinate_size': [base_width, base_height],
                    'crop_xyxy': crop, 'polygon_source_pixels': polygon, 'avoid_boxes': boxes,
                    'reference_quad_px': source_quad, 'horizon_y': horizon,
                    'floor_mask_basis': floor_basis or 'near_floor_appearance',
                    'native_review': native.get('evidence')}
    input_hash = sha256(json.dumps(source_basis, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    key = 'placeable_ground'
    cached = writer.current_observation(frame_id, task=task, kind='geometry', scope='placement', evidence_key=key)
    if cached and cached[1].get('input_hash') == input_hash and (
        cached[1]['placeable_ground']['algorithm_version'] == ALGORITHM_VERSION
    ):
        record = cached[1]['placeable_ground']
        return cached[0], record, record['basis']
    if 'bgr' not in pixels:
        pixels['bgr'] = cv2.imdecode(np.frombuffer(pixels['bytes'], dtype=np.uint8), cv2.IMREAD_COLOR)
    image = pixels['bgr']
    if image is None or image.shape[:2] != (height, width):
        return None, None, 'ground_image_decode_failed'
    # Legacy perimeter coordinates may land at width/height. source_xy
    # converts them to pixel centers; a collapsed region supplies no evidence.
    if cv2.contourArea(np.asarray(polygon, np.float32)) <= 0:
        return None, None, 'ground_polygon_degenerate'
    floor_mask = None
    if floor_basis == 'scene_camera_floor_projection':
        floor_mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(floor_mask, [np.asarray(polygon, np.int32)], 1)
    assessment = assess_placeable_ground(image, polygon, boxes=boxes, floor_mask=floor_mask)
    record = encode_placement_record(assessment, reference_quad_px=source_quad, horizon_y=horizon)
    observation = writer.observe(
        frame_id, task=task, kind='geometry', scope='placement', evidence_key=key,
        reference=source_basis, algorithm_version=ALGORITHM_VERSION,
        payload={'coordinate_size': [width, height], 'coordinate_space': 'source_image_pixels',
                 'input_hash': input_hash, 'placeable_ground': record},
    )
    return observation, record, record['basis']


def _background_label(task, entry):
    state = entry.get('label_state', 'unknown')
    instances = entry.get('existing_targets') or []
    geometry = entry.get('geometry') or {}
    scope = {'kind': 'frame'}
    if crop := geometry.get('crop_xyxy'):
        scope = {'kind': 'crop', 'crop_xyxy': crop}
    if geometry.get('mode') == 'alpha_composite_identity':
        scope.update(projection='alpha_composite_identity', description=geometry.get('scope'))
    presence, completeness = ('present' if instances else 'unknown'), 'unknown'
    if entry.get('negative_exclusion'):
        presence, completeness = 'present', 'partial'
    elif state == 'confirmed_empty' or entry.get('person_scope_review'):
        presence, completeness = 'absent', 'complete'
    elif state == 'complete_positive' and instances:
        presence, completeness = 'present', 'complete'
    elif state == 'candidate_complete_negative_in_crop' and crop and entry.get('review_verdict') == 'ready':
        presence, completeness = 'absent', 'complete'
    elif state == 'candidate_complete_negative':
        detail = entry.get('evidence') or {}
        # A source manifest locates a candidate; it does not establish target
        # absence. Preserve an explicit per-frame machine verdict when present.
        if entry.get('review_verdict') == 'ready' and detail.get('manifest'):
            presence, completeness = 'absent', 'complete'
    elif state == 'complete_candidate' and task == 'building_name_paper' and instances:
        replacements = {region.get('replacement_of') for region in geometry.get('regions', [])}
        if all(target.get('building_id') in replacements for target in instances):
            completeness = 'complete'
            scope = {'kind': 'replace_all_labelled_instances', 'label_layers': ['detection'],
                     'original_ocr': 'unknown'}
    elif state == 'partial':
        completeness = 'partial'
    evidence = entry.get('person_scope_review') or entry.get('negative_exclusion') or entry.get('evidence')
    return {'presence': presence, 'completeness': completeness, 'scope': scope, 'instances': instances,
            'label_state': state, 'evidence': evidence}


def _background(writer, frame_id, metadata, task, entry, reference, pixels):
    geometry = dict(entry.get('geometry') or {})
    geometry['coordinate_space'] = 'legacy_review_pixels'
    geometry.setdefault('coordinate_size', metadata['image_size'])
    if geometry.get('mode') == 'alpha_composite_identity':
        width, height = metadata['image_size']
        crop_width, crop_height = min(width, height * 4 // 3), min(height, width * 3 // 4)
        left, top = (width - crop_width) // 2, (height - crop_height) // 2
        geometry.update(coordinate_size=[width, height], coordinate_space='source_image_pixels',
                        output_canvas_size=[640, 480],
                        station_crop_xyxy=[left, top, left + crop_width, top + crop_height],
                        resize_method='PIL_LANCZOS')
    label = _background_label(task, entry)
    details = {**metadata, 'scope': label['scope'], 'geometry': geometry, 'label': label,
               'legacy_state': entry.get('state'), 'historical_role': entry.get('historical_role'),
               'legacy_uses': entry.get('uses', []), 'accepted_by': entry.get('accepted_by')}
    label_id = writer.observe(
        frame_id, task=task, kind='label', scope=label['scope']['kind'],
        evidence_key='background_pool', reference=reference, payload=label, origin='legacy',
    )
    legacy_id = writer.observe(
        frame_id, task=task, kind='geometry', scope='legacy_review', evidence_key='background_pool',
        reference=reference, payload=entry, origin='legacy',
    )
    ids = [label_id, legacy_id]
    split = _split(entry.get('split'))
    placement_declared = 'composite_background' in entry.get('uses', [])
    geometry_fields = {'ground_polygon', 'placement_patch', 'regions', 'bins', 'support_rectangle_xyxy', 'mode'}
    has_geometry = any(geometry.get(name) for name in geometry_fields)
    if has_geometry:
        writer.decide(frame_id, task=task, role='geometry_reference', split=split, status='eligible',
                      evidence_ids=[legacy_id], details={**details, 'reason_codes': ['legacy_region_available']},
                      origin='legacy')
    if task == 'waste_sticker' and geometry.get('bins'):
        writer.decide(frame_id, task=task, role='surface', split=split, status='eligible',
                      evidence_ids=[legacy_id], details={**details, 'scope': {'kind': 'bin_surfaces'},
                                                        'reason_codes': ['recorded_sticker_surfaces']},
                      origin='legacy')

    reasons = []
    if not placement_declared:
        reasons.append('background_use_unreviewed')
    if label['completeness'] != 'complete':
        reasons.append('task_label_coverage_unresolved')
    if not has_geometry:
        reasons.append('placement_region_unresolved')
    if placement_declared and has_geometry and task in GROUND_TASKS:
        geometry_id, ground, basis = _ground_record(writer, frame_id, metadata, task, geometry, label, reference, pixels)
        if geometry_id:
            ids.append(geometry_id)
            geometry['placeable_ground'] = ground
            geometry['placeable_ground_coordinate_size'] = metadata['image_size']
        if not ground or ground['status'] == 'unknown':
            reasons.append(basis)
        elif task == 'person':
            geometry['placement_scale'] = {'far_scale': 0.70, 'basis': 'person_plan_apparent_height_ratio'}
            if not ground['reference_quad_px']:
                reasons.append('placement_reference_quad_unresolved')
    if task in {'e_bike', 'sign'} and label['presence'] == 'present':
        reasons.append('existing_targets_unsupported_by_entry')
    if entry.get('negative_exclusion'):
        reasons.append('source_exclusion')
    status = 'blocked' if 'source_exclusion' in reasons else ('unknown' if reasons else 'eligible')
    writer.decide(
        frame_id, task=task, role='composite_background', split=split, status=status, evidence_ids=ids,
        details={**details, 'reason_codes': reasons or ['placement_and_labels_cover_use']}, origin='legacy',
    )
    # Whole-frame empty supervision does not require room to add another target.
    is_empty = label['presence'] == 'absent' and label['scope']['kind'] == 'frame'
    if label['presence'] == 'present':
        empty_status, reason = 'blocked', 'task_target_present'
    elif is_empty and label['completeness'] == 'complete' and status != 'blocked':
        empty_status, reason = 'eligible', 'full_frame_target_absence'
    else:
        empty_status, reason = 'unknown', 'full_frame_target_absence_unproven'
    writer.decide(
        frame_id, task=task, role='empty_negative', split=split, status=empty_status, evidence_ids=[label_id],
        details={**metadata, 'scope': label['scope'], 'label': label, 'reason_codes': [reason],
                 'role_basis': 'background_review'}, origin='legacy',
    )


def _direct_review(entry):
    state = entry.get('detection_state') or entry.get('state', 'unknown')
    instances = entry.get('instances') or []
    presence = 'present' if instances else 'unknown'
    completeness = 'partial' if state == 'partial' else 'unknown'
    if state == 'complete_negative' and not instances:
        presence, completeness = 'absent', 'complete'
    if state == 'complete_positive' and instances:
        presence, completeness = 'present', 'complete'
    scope = {'kind': 'frame'}
    if entry.get('detection_state') or entry.get('selection_evidence'):
        scope['label_layers'] = ['detection']
    return {**entry, 'presence': presence, 'completeness': completeness, 'instances': instances,
            'scope': scope, 'label_state': state}


def _selected_detection(task, entry, metadata):
    evidence = entry.get('selection_evidence')
    if task != 'building_name_paper' or not evidence:
        return False
    selected = _read(evidence['manifest'])['records'][evidence['record_index']]
    return (selected['image'] == metadata['source_image']
            and selected['source_label'] == entry.get('label')
            and selected['source_state'] == (entry.get('detection_state') or entry['state'])
            and selected['split'] == entry.get('split'))


def _target(writer, frame_id, metadata, row, task, entry, reference):
    evidence_key = reference['manifest']
    if entry is None:
        # These resources are artwork, cutouts or texture inputs rather than frame-level detection labels.
        role = row.get('role', 'reference')
        label = {'presence': 'unknown', 'completeness': 'unknown', 'scope': {'kind': 'material'},
                 'class_id': row.get('class_id'), 'class_name': row.get('class_name')}
        evidence_id = writer.observe(
            frame_id, task=task, kind='material', scope='material', evidence_key=evidence_key,
            reference=reference, payload=row, origin='legacy',
        )
        writer.decide(frame_id, task=task, role=role, split=None, status='eligible', evidence_ids=[evidence_id],
                      details={**metadata, 'label': label, 'scope': label['scope'],
                               'reason_codes': ['declared_material_role']}, origin='legacy')
        return
    label = _direct_review(entry)
    label_id = writer.observe(
        frame_id, task=task, kind='label', scope='frame', evidence_key=evidence_key,
        reference=reference, payload=label, origin='legacy',
    )
    details = {**metadata, 'label': label, 'scope': label['scope'], 'legacy_uses': entry.get('uses', []),
               'role_basis': 'direct_annotation',
               'legacy_state': entry.get('state'), 'historical_role': entry.get('split')}
    split = _split(entry.get('split') or row.get('split'))
    direct_use = {'direct_sample', 'direct_target', 'empty_negative'} & set(entry.get('uses', []))
    if label['completeness'] != 'complete' or not direct_use:
        if set(entry.get('uses', [])) & {'visual_reference', 'review_reference', 'artwork_reference',
                                        'pose_reference', 'same_capture_reference', 'held_out_reference'}:
            writer.decide(frame_id, task=task, role='reference', split=split, status='eligible',
                          evidence_ids=[label_id], details={**details, 'reason_codes': ['task_label_reference']},
                          origin='legacy')
        return
    review_level = entry.get('label_level')
    checked = (review_level in {'auto_validated', 'human_confirmed', 'manual_confirmed'}
               or _selected_detection(task, entry, metadata))
    reasons = ['full_frame_annotation_verified'] if checked else ['direct_annotation_candidate']
    label_path = entry.get('label')
    if label_path:
        file = WORKSPACE / label_path
        if not file.is_file():
            checked = False
            reasons = ['declared_label_file_missing']
        elif bool(file.read_text(encoding='utf-8-sig').strip()) != (label['presence'] == 'present'):
            checked = False
            reasons = ['declared_label_presence_mismatch']
    role = 'empty_negative' if label['presence'] == 'absent' else 'direct_target'
    writer.decide(
        frame_id, task=task, role=role, split=split, status='eligible' if checked else 'unknown',
        evidence_ids=[label_id], details={**details, 'reason_codes': reasons}, origin='legacy',
    )


class Importer:
    def __init__(self, writer):
        self.writer = writer
        self.history = read_source_history(WORKSPACE)
        with (WORKSPACE / SOURCE_COPIES).open(encoding='utf-8-sig') as stream:
            self.copies = {
                row['path']: row['original_path'] for line in stream
                if line.strip() and (row := json.loads(line)).get('original_path')
            }
        self.person_review = {
            row['image']: (i, row) for i, row in enumerate(_read(PERSON_DISPATCH)['records'])
        }
        self.person_exclusions = {
            row['image']: (i, row) for i, row in enumerate(yaml.safe_load(
                (WORKSPACE / PERSON_ISSUES).read_text(encoding='utf-8')
            ).get('exclude_backgrounds', []))
        }
        with (WORKSPACE / PERSON_B6).open(encoding='utf-8-sig') as stream:
            self.person_ground = {
                row['frame']['path']: (i, row) for i, line in enumerate(stream)
                if line.strip() and (row := json.loads(line)).get('frame')
            }
        self.arenas = {row['image']: (i, row) for i, row in enumerate(_read(ARENA_PERSON)['records'])}
        self.surface_rows = _path_observations(CAMERA_SURFACES)
        self.view_rows = _path_observations(CAMERA_VIEWS)
        self.environment = {}
        self.raw_paths = set()
        self.summary = Counter()
        self.missing = set()
        self.copy_mismatches = set()

    def register(self, row, reference, pixels=None):
        image_path = row.get('image')
        if not image_path or Path(image_path).suffix.lower() not in IMAGE_SUFFIXES:
            return None
        if not (WORKSPACE / image_path).is_file():
            self.missing.add(image_path)
            return None
        if pixels is None:
            size, digest, properties = _file_info(image_path)
        else:
            pixels['bytes'] = (WORKSPACE / image_path).read_bytes()
            size, digest, properties = _image_info(pixels['bytes'])
        aliases = []
        for path in sorted({row.get('source_image') or image_path, row.get('original_path') or image_path} - {image_path}):
            if Path(path).suffix.lower() not in IMAGE_SUFFIXES or not (WORKSPACE / path).is_file():
                continue
            if _file_info(path)[1] == digest:
                aliases.append(path)
            else:
                if path == row.get('source_image') and row.get('generation_method') == 'file_copy':
                    self.copy_mismatches.add((image_path, path))
        parent = (row.get('parent_group') or row.get('mother_scene') or row.get('source_blend') or
                  row.get('parent_video') or row.get('video') or row.get('source_group') or row.get('parent_artwork'))
        if parent:
            parent = historical_source(self.copies.get(parent, parent), self.history)
        frame_id = self.writer.register(image_path, image_size=size, parent_group=parent,
                                        verified_copy_paths=aliases, content_sha256=digest)
        geometry_basis_valid = row.get('image_size') in (None, size)
        metadata = {
            'image': image_path, 'image_size': size, 'image_attributes': properties,
            'source_image_size': size, 'content_sha256': digest,
            'source_image': row.get('source_image') or image_path,
            'original_path': row.get('original_path') or self.copies.get(row.get('source_image') or image_path),
            'working_path': image_path if image_path.startswith('datasets/raw/') else None,
            'capture_identity': row.get('capture_identity') or row.get('id') or historical_source(
                row.get('source_image') or image_path, self.history),
            'parent_group': parent, 'parent_video': row.get('parent_video') or row.get('video'),
            'source_group': row.get('source_group'), 'source_kind': row.get('source_kind'),
            'source_blend': row.get('source_blend') or row.get('source_scene'),
            'mother_scene': row.get('mother_scene'), 'parent_image': row.get('parent_image'),
            'parent_artwork': row.get('parent_artwork'), 'generation_method': row.get('generation_method'),
            'frame_index': row.get('frame_index', row.get('source_frame_index')),
            'timestamp_seconds': row.get('timestamp_seconds', row.get('source_time_seconds')),
            'point': row.get('point'), 'segment': row.get('segment'),
            'duplicate_group': row.get('duplicate_group'), 'duplicate_of': row.get('duplicate_of'),
            'provenance': {**reference, 'legacy_id': row.get('id'), 'row_evidence': row.get('evidence'),
                           'verified_copy_paths': aliases, 'declared_image_size': row.get('image_size')},
        }
        self.writer.observe(frame_id, kind='lineage', scope='frame',
                            evidence_key=f'{reference["manifest"]}:{image_path}', reference=reference,
                            payload=metadata, origin='legacy')
        self.remember_environment(frame_id, metadata, row, reference, aliases)
        if image_path.startswith('datasets/raw/'):
            self.raw_paths.add(image_path)
        self.summary[reference['manifest']] += 1
        if not geometry_basis_valid:
            self.summary['declared_dimensions_differ'] += 1
            return None
        return frame_id, metadata

    def remember_environment(self, frame_id, metadata, row, reference, aliases):
        source_kind = metadata.get('source_kind')
        if source_kind in {'digital', 'cutout'}:
            domain = 'material'
        elif source_kind == 'blender_render':
            domain = 'rendered'
        elif metadata.get('parent_video') or source_kind in {'camera', 'camera_photo', 'archive_camera_photo'}:
            domain = 'camera'
        else:
            domain = 'unknown'
        payload = {'source_domain': domain, 'coordinate_size': metadata['image_size'],
                   'scope': {'kind': 'frame'}}
        if domain == 'material':
            score = 1
            payload.update(status='not_applicable', reason_codes=['non_scene_material'])
        else:
            score = 0 if domain == 'unknown' else 1
            payload.update(status='unknown', reason_codes=['shared_scene_evidence_unavailable'])
        for path in (metadata['image'], metadata['source_image'], *aliases):
            observed = self.surface_rows.get(path)
            view = self.view_rows.get(path)
            if view and view[0].get('image_size') == metadata['image_size']:
                view_row, view_ref = view
                score, reference = 3, view_ref
                payload.update(source_domain='camera', status='observed', view=view_row.get('view'),
                               camera_angles=view_row.get('camera_angles'), angle_basis=view_row.get('angle_basis'),
                               view_evidence=view_ref, view_review_method=view_row.get('review_method'),
                               reason_codes=['recorded_visual_view'])
            if observed and observed[0].get('image_size') == metadata['image_size']:
                surface, source_ref = observed
                score, reference = 4, source_ref
                observed_view = payload.get('view') or surface.get('view')
                payload.update(source_domain='camera',
                               status='observed' if surface.get('regions') or observed_view else 'unknown',
                               view=observed_view, surface_review_method=surface.get('review_method'),
                               regions=surface.get('regions', []), placement_status=surface.get('placement_status'),
                               projection_reference=surface.get('projection_reference'),
                               camera_angles=payload.get('camera_angles') or surface.get('camera_angles'),
                               uncertainties=surface.get('uncertainties', []),
                               reason_codes=['recorded_scene_surface_reference'])
                break
        if score < 3:
            geometries = {
                task: {key: value for key, value in review['geometry'].items() if key != 'placeable_ground'}
                for task, review in row.get('tasks', {}).items() if review.get('geometry')
            }
            if geometries:
                score = 2
                payload.update(status='observed', task_region_references=geometries,
                               reason_codes=['recorded_task_region_references'])
        if score > self.environment.get(frame_id, (-1,))[0]:
            self.environment[frame_id] = (score, reference, payload)

    def write_environment(self):
        for frame_id, (_, reference, payload) in self.environment.items():
            self.writer.observe(frame_id, kind='environment', scope='frame', evidence_key='legacy_scene',
                                reference=reference, payload=payload, algorithm_version='legacy_scene_import_v1',
                                origin='legacy')

    def person_entry(self, metadata, entry):
        source = metadata['source_image']
        matched = self.person_review.get(source)
        result = dict(entry)
        if matched:
            number, row = matched
            evidence = row.get('person_scope_review') or {}
            if (evidence.get('result') == 'complete_negative' and evidence.get('scope') == 'whole_frame'
                    and evidence.get('image_sha256') == metadata['content_sha256']):
                result['person_scope_review'] = {**_reference(PERSON_DISPATCH, number), **evidence}
                result['split'] = row.get('split') or result.get('split')
                result['uses'] = sorted(set(entry.get('uses', [])) | {'composite_background'})
                result['existing_targets'] = []
                geometry = dict(entry.get('geometry') or {})
                measurement = self.person_ground.get(source)
                arena = self.arenas.get(source)
                if measurement:
                    number, metric = measurement
                    ground = metric.get('ground') or {}
                    patch = ground.get('placement_patch')
                    if patch and metric.get('extra', {}).get('image_shape') == metadata['image_size'][::-1]:
                        geometry['native_ground'] = {
                            'polygon': patch, 'reference_quad_px': patch[::-1], 'horizon_y': ground.get('horizon_y'),
                            'evidence': {**_reference(PERSON_B6, number), 'code_hash': metric['code_hash']},
                        }
                elif arena:
                    number, render = arena
                    if render['image_size'] == metadata['image_size']:
                        native = {'polygon': render['ground_polygon_640x480'],
                                  'floor_mask_basis': 'scene_camera_floor_projection',
                                  'evidence': _reference(ARENA_PERSON, number)}
                        # The dispatch explicitly records a conservative reference quad
                        # scaled independently to its 640x360 authoring view.
                        quad = row.get('ground_polygon_640x360') or []
                        if len(quad) == 4 and 'quadrilateral' in row.get('placement_region_basis', ''):
                            width, height = metadata['image_size']
                            native['reference_quad_px'] = [[x * (width - 1) / 639, y * (height - 1) / 359]
                                                           for x, y in quad]
                        geometry['native_ground'] = native
                result['geometry'] = geometry
            if row['state'] == 'excluded':
                result['state'] = 'excluded'
        if source in self.person_exclusions and not result.get('person_scope_review'):
            number, evidence = self.person_exclusions[source]
            reference = {**_reference(PERSON_ISSUES, number), **evidence, 'collection': 'exclude_backgrounds'}
            # The legacy list also includes outside people, which are out of
            # scope for current standee labels. Keep that review as history.
            reason = evidence.get('reason', '')
            field = 'negative_exclusion' if '立牌' in reason and '漏标' in reason else 'historical_exclusion'
            result[field] = reference
        return result

    @staticmethod
    def ebike_entry(metadata, entry):
        evidence = entry.get('evidence') or {}
        if evidence.get('manifest') != 'datasets/annotated/e_bike/v4_complex_backgrounds_20261112/backgrounds.json':
            return entry
        row = _read(evidence['manifest'])['backgrounds'][evidence['record_index']]
        if row.get('image') != metadata['source_image'] or not row.get('crop_xyxy'):
            return entry
        transform = {'parent_image': row['parent_image'], 'parent_crop_xyxy': row['crop_xyxy'],
                     'output_size': metadata['image_size'], 'resize_method': 'bilinear'}
        return {**entry, 'geometry': {**entry['geometry'], 'parent_transform': transform}}

    def import_manifest(self, manifest, *, background=False, only_tasks=None, material_task=None):
        for index, row in enumerate(_read(manifest)['records']):
            reference = _reference(manifest, index)
            pixels = {} if background else None
            result = self.register(row, reference, pixels=pixels)
            if not result:
                continue
            frame_id, metadata = result
            for task, entry in row.get('tasks', {}).items():
                if only_tasks is not None and task not in only_tasks:
                    continue
                if task not in TASKS:
                    continue
                if background:
                    if task == 'person':
                        legacy_entry = entry
                        entry = self.person_entry(metadata, entry)
                        if legacy_entry != entry:
                            self.writer.observe(
                                frame_id, task=task, kind='source_review', scope='legacy_pool',
                                evidence_key=manifest, reference=reference, payload=legacy_entry, origin='legacy',
                            )
                    if task == 'e_bike':
                        entry = self.ebike_entry(metadata, entry)
                    _background(self.writer, frame_id, metadata, task, entry, reference, pixels)
                else:
                    _target(self.writer, frame_id, metadata, row, task, entry, reference)
            if material_task and material_task not in row.get('tasks', {}) and row.get('role'):
                _target(self.writer, frame_id, metadata, row, material_task, None, reference)

    def import_white_backs(self):
        for index, row in enumerate(_read(WHITE_BACK)['records']):
            reference = _reference(WHITE_BACK, index)
            row = {**row, 'source_kind': 'digital', 'parent_group': row['parent_artwork']}
            result = self.register(row, reference)
            if result:
                frame_id, metadata = result
                evidence = self.writer.observe(
                    frame_id, task='person', kind='material', scope='material', evidence_key=WHITE_BACK,
                    reference=reference, payload=row, origin='legacy',
                )
                self.writer.decide(
                    frame_id, task='person', role='hard_negative', split=None, status='eligible',
                    evidence_ids=[evidence], origin='legacy',
                    details={**metadata, 'scope': {'kind': 'material'},
                             'label': {'presence': 'absent', 'completeness': 'complete',
                                       'label_state': row['status_for_model'], 'instances': []},
                             'reason_codes': ['documented_white_back_occluder_artwork']},
                )

    def inventory_raw(self):
        files = {p.relative_to(WORKSPACE).as_posix() for p in RAW.rglob('*')
                 if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()}
        unmatched = sorted(files - self.raw_paths)
        for path in unmatched:
            self.register({'image': path}, {'manifest': 'datasets/raw', 'inventory_only': True})
        return len(files), unmatched


def backfill(database=DEFAULT_DATABASE):
    """Backfill the currently configured, persisted observations, plus raw image inventory."""
    _read.cache_clear()
    _file_info.cache_clear()
    sources = {task: load_task_config(task)[1] for task in sorted(TASKS)}
    background_manifests = {cfg['background_pool_manifest'] for cfg in sources.values()}
    direct_manifests = {}
    for task, cfg in sources.items():
        for path in cfg.get('direct_capture_manifests', []):
            direct_manifests.setdefault(path, set()).add(task)
    direct_manifests.setdefault(CAR_PERSON, set()).add('person')
    with edit_database(database) as writer:
        importer = Importer(writer)
        for task, cfg in sources.items():
            importer.import_manifest(cfg['target_working_manifest'], material_task=task)
        for path, tasks in direct_manifests.items():
            importer.import_manifest(path, only_tasks=tasks)
        for path in sorted(background_manifests):
            importer.import_manifest(path, background=True)
        importer.import_white_backs()
        raw_count, unmatched = importer.inventory_raw()
        importer.write_environment()
    return {
        'raw_images': raw_count, 'raw_images_without_source_manifest': unmatched,
        'registered_manifest_images': dict(importer.summary), 'missing_declared_images': sorted(importer.missing),
        'copy_content_differences': sorted(importer.copy_mismatches), **quota_summary(database=database),
    }


def _native_ground(image, source_image):
    from .review import geometry

    ground = geometry.estimate_ground_region(image)
    if not ground.ok or not ground.placement_patch:
        raise ValueError(f'Native ground placement remains unresolved: {source_image}')
    code = Path(geometry.__file__)
    return {
        'polygon': ground.placement_patch, 'reference_quad_px': ground.placement_patch[::-1],
        'horizon_y': ground.horizon_y,
        'evidence': {
            'source_image': source_image, 'method': 'estimate_ground_region',
            'code': code.relative_to(WORKSPACE).as_posix(), 'code_sha256': sha256(code.read_bytes()).hexdigest(),
            'reason_codes': ground.reason_codes, 'residual': ground.residual,
        },
    }


def refresh_ground(selections, *, contacts=None, database=DEFAULT_DATABASE):
    """Refresh the named composite decisions using their existing label and source evidence."""
    import cv2
    import numpy as np

    from .review.placement_record import ALGORITHM_VERSION, decode_placement_record

    selections = dict(selections)
    contacts = dict(contacts or {})
    if not selections or contacts.keys() - selections.keys():
        raise ValueError('Select at least one frame, and specify contacts only for selected frames')
    records = {}
    for task, split in sorted({(task, split) for task, split, _ in selections}):
        if task not in GROUND_TASKS or split not in SPLITS:
            raise ValueError(f'Choose a ground task and an assigned split: {task}/{split}')
        for row in query_frames(task=task, role='composite_background', split=split, status=None, database=database):
            key = task, split, Path(row['source_image']).name
            if key in selections:
                if key in records:
                    raise ValueError(f'Ambiguous selected source filename: {key}')
                records[key] = row
    if missing := selections.keys() - records.keys():
        raise ValueError(f'Selected source frames are missing: {sorted(missing)}')
    results = []
    with edit_database(database) as writer:
        for key, row in records.items():
            task, split, _ = key
            previous = writer.conn.execute(
                'SELECT details_json FROM decisions WHERE record_id = ? AND current = 1', (row['record_id'],),
            ).fetchone()
            if previous is None:
                raise ValueError(f'Selected decision changed during refresh: {key}')
            if row['origin'] == 'human' or any(not item['current'] or item['origin'] == 'human'
                                               for item in row['evidence']):
                raise ValueError(f'Selected decision uses historical or human evidence: {key}')
            result = {'task': task, 'split': split, 'image': row['source_image'], 'frame_id': row['frame_id'],
                      'previous_record_id': row['record_id'], 'previous_status': row['status'],
                      'previous_reasons': row['reason_codes']}
            details = json.loads(previous['details_json'])
            geometry = details['geometry']
            old_ground = geometry.get('placeable_ground') or {}
            if old_ground.get('algorithm_version') == ALGORITHM_VERSION:
                results.append({**result, 'record_id': row['record_id'], 'status': row['status'],
                                'reason_codes': row['reason_codes'], 'reused': True})
                continue
            if row['status'] != 'unknown' or 'composite_background' not in details.get('legacy_uses', []):
                raise ValueError(f'Selected frame has no pending declared ground-composite use: {key}')
            old_placement = next((item for item in row['evidence'] if item['scope'] == 'placement'), None)
            legacy = next(item for item in row['evidence'] if item['scope'] == 'legacy_review')
            if not selections[key] and old_placement is None:
                raise ValueError(f'Selected frame has no source-pixel placement evidence: {key}')

            raw = (WORKSPACE / row['image']).read_bytes()
            if sha256(raw).hexdigest() != row['content_sha256']:
                raise ValueError(f'Frame bytes changed since the former decision: {row["image"]}')
            image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or list(image.shape[:2]) != row['image_size'][::-1]:
                raise ValueError(f'Cannot decode the registered source pixel version: {row["image"]}')
            if selections[key]:
                geometry['native_ground'] = _native_ground(image, row['source_image'])
            evidence_id, ground, basis = _ground_record(
                writer, row['frame_id'], row, task, geometry, row['label'], legacy['reference'], {'bgr': image},
            )
            if ground is None:
                raise ValueError(f'Cannot refresh the selected geometry {key}: {basis}')
            geometry.update(placeable_ground=ground, placeable_ground_coordinate_size=row['image_size'])
            placement = decode_placement_record(ground)
            prior_ground_reasons = {old_ground.get('basis'), 'ground_coordinate_basis_unresolved',
                                    'placement_reference_quad_unresolved'}
            reasons = [reason for reason in row['reason_codes'] if reason not in prior_ground_reasons]
            if placement.status != 'candidate':
                reasons.append(basis)
            elif task == 'person':
                if not ground.get('reference_quad_px'):
                    reasons.append('placement_reference_quad_unresolved')
                else:
                    geometry['placement_scale'] = {'far_scale': 0.70, 'basis': 'person_plan_apparent_height_ratio'}
            if key in contacts:
                x, y, width = contacts[key]
                contact_status = placement.contact_status((x, y), contact_width=width)
                geometry['placement_contact_probe'] = {
                    'foot_xy': [x, y], 'contact_width': width, 'status': contact_status,
                    'coordinate_space': 'source_image_pixels',
                }
                if contact_status != 'candidate':
                    reasons.append(f'placement_contact_probe_{contact_status}')

            status = 'unknown' if reasons else 'eligible'
            final_reasons = reasons or ['placement_and_labels_cover_use']
            retained_evidence = [item['evidence_id'] for item in row['evidence'] if item != old_placement]
            record_id = writer.decide(
                row['frame_id'], task=task, role='composite_background', split=split, status=status,
                evidence_ids=[*retained_evidence, evidence_id], origin='machine',
                details={**details, 'reason_codes': final_reasons},
            )
            results.append({**result, 'record_id': record_id, 'status': status, 'reason_codes': final_reasons,
                            'placement_basis': basis, 'candidate_pixels': int(placement.candidate_mask.sum()),
                            'contact_probe': geometry.get('placement_contact_probe')})
    return {'algorithm_version': ALGORITHM_VERSION, 'records': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['backfill', 'summary', 'refresh-ground', 'candidates', 'assign'])
    parser.add_argument('--database', type=Path, default=DEFAULT_DATABASE)
    parser.add_argument('--task', help='任务（candidates）')
    parser.add_argument('--role', help='用途角色（candidates）')
    parser.add_argument('--split', choices=SPLITS, help='目标集合（candidates）')
    parser.add_argument('--record-id', action='append', default=[], help='明确选择的待分配记录（assign，可重复）')
    parser.add_argument('--frame', action='append', nargs='+', default=[],
                        help='TASK SPLIT SOURCE_FILENAME [SOURCE_FILENAME ...]')
    parser.add_argument('--native-frame', action='append', nargs='+', default=[],
                        help='TASK SPLIT SOURCE_FILENAME [SOURCE_FILENAME ...], remeasure in native pixels')
    parser.add_argument('--contact', action='append', nargs=6, default=[],
                        metavar=('TASK', 'SPLIT', 'SOURCE_FILENAME', 'X', 'Y', 'WIDTH'))
    args = parser.parse_args()
    exit_code = 0
    if args.command == 'assign':
        if not all((args.task, args.role, args.split, args.record_id)):
            parser.error('assign requires --task, --role, --split and at least one --record-id')
        if args.task not in TASKS:
            parser.error('Unknown task: ' + args.task)
        if args.frame or args.native_frame or args.contact:
            parser.error('Frame and contact arguments require the refresh-ground command')
        from .assign_split import assign_selected

        records = assign_selected(args.task, args.role, args.split, args.record_id, database=args.database)
        assigned = sum(row['status'] == 'assigned' for row in records)
        unchanged = sum(row['status'] == 'unchanged' for row in records)
        failed = len(records) - assigned - unchanged
        status = 'complete' if not failed else 'failed' if not (assigned or unchanged) else 'partial'
        exit_code = {'complete': 0, 'failed': 1, 'partial': 2}[status]
        result = {'task': args.task, 'role': args.role, 'requested_split': args.split,
                  'status': status, 'counts': {'assigned': assigned, 'unchanged': unchanged, 'failed': failed},
                  'records': records}
    elif args.record_id:
        parser.error('--record-id requires the assign command')
    elif args.command == 'candidates':
        if not all((args.task, args.role, args.split)):
            parser.error('candidates requires --task, --role and --split')
        if args.task not in TASKS:
            parser.error('Unknown task: ' + args.task)
        if args.frame or args.native_frame or args.contact:
            parser.error('Frame and contact arguments require the refresh-ground command')
        from .source_ledger.config import build_groups, load_source_config
        from .source_ledger.reservations import candidate_reservation

        rows = query_split_candidates(task=args.task, role=args.role, split=args.split, database=args.database)
        try:
            source = load_source_config(WORKSPACE, args.task)
            groups = build_groups(args.task, source.get('pairing') or {})
            history = read_source_history(WORKSPACE)
            for row in rows:
                row['reservation'] = candidate_reservation(row, args.split, groups, history)
            policy = {'status': 'read', 'task': args.task}
        except (OSError, ValueError, TypeError, KeyError, AttributeError, yaml.YAMLError) as exc:
            for row in rows:
                row['reservation'] = {'status': 'unknown', 'matched_groups': [], 'conflicts': []}
            policy = {'status': 'unknown', 'task': args.task, 'reason': str(exc)}
        result = {'task': args.task, 'role': args.role, 'requested_split': args.split,
                  'reservation_policy': policy, 'records': rows}
    elif args.task or args.role or args.split:
        parser.error('--task, --role and --split require the candidates command')
    elif args.command == 'refresh-ground':
        selections = {}
        for groups, native in ((args.frame, False), (args.native_frame, True)):
            for group in groups:
                if len(group) < 3:
                    parser.error('Each frame selection needs a task, split and at least one filename')
                task, split, *names = group
                for name in names:
                    if (key := (task, split, name)) in selections:
                        parser.error(f'Duplicate frame selection: {key}')
                    selections[key] = native
        contacts = {tuple(values[:3]): tuple(map(float, values[3:])) for values in args.contact}
        result = refresh_ground(selections, contacts=contacts, database=args.database)
    elif args.frame or args.native_frame or args.contact:
        parser.error('Frame and contact arguments require the refresh-ground command')
    else:
        result = backfill(args.database) if args.command == 'backfill' else quota_summary(database=args.database)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
