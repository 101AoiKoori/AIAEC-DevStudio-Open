"""Review and register one original image for any number of V4 tasks."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import argparse
import json
import math
from pathlib import Path

from .frame_environment import DEFAULT_DATABASE, edit_database, workspace_path
from .paths import TASKS, WORKSPACE
from .source_history import historical_source, read_source_history


_GROUND_TASKS = {'person', 'e_bike', 'sign', 'waste_bin_state'}
_DIRECT_SOURCES = {'camera', 'camera_photo', 'video_frame', 'archive_camera_photo',
                   'archive_camera_video_frame', 'archive_video_frame'}


def _bounds(rect, size):
    if not isinstance(rect, (list, tuple)) or len(rect) != 4:
        return False
    x1, y1, x2, y2 = rect
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in rect) and (
        0 <= x1 < x2 <= size[0] and 0 <= y1 < y2 <= size[1]
    )


def _has_absence_evidence(evidence):
    """Accept a declared image review or a locator into an annotation manifest."""
    if not isinstance(evidence, dict):
        return False
    method = evidence.get('method')
    if isinstance(method, str):
        terms = set(method.casefold().replace('-', ' ').replace('_', ' ').split())
        image_review = terms & {'visual', 'inspection', 'annotation'} or (
            'review' in terms and terms & {'frame', 'image', 'machine', 'manual', 'complete'}
        )
        if (image_review and
                not terms & {'directory', 'folder', 'path', 'provenance', 'inventory', 'lineage', 'metadata'}):
            return True

    manifest = evidence.get('manifest')
    if not isinstance(manifest, str):
        return False
    path, _, locator = manifest.partition('#')
    manifest_file = Path(path)
    if manifest_file.suffix.casefold() not in {'.json', '.jsonl', '.yaml', '.yml', '.csv', '.tsv'}:
        return False
    if manifest_file.name.casefold() in {
        'source.yaml', 'source_copy_map.json', 'source_copy_manifest.jsonl',
        'raw_source_relocations.json', 'background_pool_manifest.json',
    }:
        return False
    index = evidence.get('record_index')
    record_id = evidence.get('record_id')
    return bool(locator.strip() or (type(index) is int and index >= 0) or
                (isinstance(record_id, str) and record_id.strip()))


def _label(objects, annotation, size):
    scope = annotation.get('scope', {'kind': 'frame'})
    if scope['kind'] == 'crop' and not _bounds(scope.get('crop_xyxy'), size):
        raise ValueError('Annotation crop must use source image pixels')
    if scope['kind'] not in {'frame', 'crop', 'surface'}:
        raise ValueError(f'Unsupported annotation scope: {scope}')
    presence, completeness = annotation['presence'], annotation['completeness']
    if presence not in {'present', 'absent', 'unknown'} or completeness not in {'complete', 'partial', 'unknown'}:
        raise ValueError('Use explicit presence and completeness values')
    instances, excluded, references = [], [], []
    seen = set()
    for row in annotation.get('instances', []):
        object_id = row['object_id']
        if object_id not in objects or object_id in seen:
            raise ValueError(f'Unknown or repeated object_id in task: {object_id}')
        seen.add(object_id)
        disposition = row.get('disposition', 'target')
        if disposition not in {'target', 'ignore', 'hard_negative'}:
            raise ValueError(f'Invalid task disposition: {disposition}')
        combined = {**objects[object_id], **row, 'disposition': disposition}
        if 'xyxy' in combined and not _bounds(combined['xyxy'], size):
            raise ValueError(f'Task object box outside the source frame: {object_id}')
        references.append(dict(row))
        if disposition == 'target':
            instances.append(combined)
        else:
            excluded.append(combined)
    if completeness == 'complete':
        if not annotation.get('evidence') or presence not in {'present', 'absent'}:
            raise ValueError('Complete labels require scope-specific target presence and evidence')
        if presence == 'absent' and not _has_absence_evidence(annotation['evidence']):
            raise ValueError('Complete absence requires per-image review or annotation manifest evidence')
        if bool(instances) != (presence == 'present'):
            raise ValueError('Task target references disagree with the complete presence label')
        if any(row.get('class_id') is None and not row.get('class_name') for row in instances):
            raise ValueError('Complete target references require a task class')
    return {**annotation, 'scope': scope, 'instances': instances, 'excluded_instances': excluded,
            'object_references': references}


def _bgr(image):
    if 'bgr' not in image:
        import numpy as np

        image['bgr'] = np.asarray(image['pixels'].convert('RGB'), dtype=np.uint8)[:, :, ::-1].copy()
    return image['bgr']


def _geometry(image, size, task, geometry, objects):
    if geometry.get('coordinate_size', size) != size:
        raise ValueError(f'{task} geometry does not use exact source image coordinates')
    geometry = {**geometry, 'coordinate_size': size, 'coordinate_space': 'source_image_pixels'}
    status = geometry.get('status', 'unknown')
    if status not in {'candidate', 'unknown', 'blocked', 'not_applicable'}:
        raise ValueError(f'Invalid geometry status: {status}')
    has_ground = bool(geometry.get('ground_polygon') or geometry.get('placeable_ground'))
    sticker_ground = task == 'waste_sticker' and has_ground
    if (task in _GROUND_TASKS or sticker_ground) and has_ground:
        from .review.placement import assess_placeable_ground
        from .review.placement_record import decode_placement_record, encode_placement_record

        if sticker_ground and geometry.get('bins'):
            if status == 'candidate' and not geometry.get('evidence'):
                raise ValueError('waste_sticker surface geometry requires review evidence')
            geometry['surface_status'] = status
        record = geometry.get('placeable_ground')
        if record is not None:
            assessment = decode_placement_record(record)
            if list(record['image_shape']) != size[::-1]:
                raise ValueError('Placement evidence does not match the exact image dimensions')
        else:
            assessment = assess_placeable_ground(
                _bgr(image), geometry['ground_polygon'], boxes=[
                    row['xyxy'] for row in objects.values() if 'xyxy' in row
                ],
            )
            record = encode_placement_record(
                assessment, reference_quad_px=geometry.get('reference_quad_px'), horizon_y=geometry.get('horizon_y'),
            )
        if task == 'person' and assessment.status == 'candidate' and not record.get('reference_quad_px'):
            from .review.geometry import contact_reference_quad

            # The contact mask is the placement evidence. A small inscribed quad
            # supplies local coordinates without claiming a measured scale.
            if quad := contact_reference_quad(assessment.candidate_mask):
                record = {
                    **record,
                    **encode_placement_record(assessment, reference_quad_px=quad,
                                              horizon_y=record.get('horizon_y')),
                    'reference_quad_basis': 'inscribed_contact_mask_rectangle',
                }
        geometry['placeable_ground'] = record
        if status not in {'blocked', 'not_applicable'}:
            status = assessment.status
            if task == 'person' and not record.get('reference_quad_px'):
                status = 'unknown'
            if sticker_ground and geometry.get('surface_status') == 'candidate':
                status = 'candidate'
    elif task in _GROUND_TASKS and status == 'candidate':
        status = 'unknown'
    elif status == 'candidate':
        has_region = geometry.get('regions') or geometry.get('bins') or geometry.get('station_crop_xyxy')
        if not has_region or not geometry.get('evidence'):
            raise ValueError(f'{task} geometry requires an applicable region and review evidence')
    reason = (geometry.get('placeable_ground') or {}).get('basis')
    if status == 'unknown' and task == 'person' and geometry.get('placeable_ground', {}).get('status') == 'candidate':
        reason = 'placement_reference_quad_unresolved'
    fallback = {
        'candidate': 'recorded_placement_region', 'unknown': 'placement_evidence_unavailable',
        'blocked': 'placement_blocked', 'not_applicable': 'placement_not_applicable',
    }
    return {**geometry, 'status': status, 'reason_codes': geometry.get('reason_codes') or [reason or fallback[status]]}


def _shared_ground(image):
    from .review.geometry import estimate_ground_region

    ground = estimate_ground_region(_bgr(image))
    return {
        'status': 'observed' if ground.polygon else 'unknown',
        'algorithm_version': 'ground_region_v1',
        'reason_codes': ground.reason_codes or ['ground_region_unresolved'],
        'ground_polygon': ground.polygon, 'placement_patch': ground.placement_patch,
        'horizon_y': ground.horizon_y, 'residual': ground.residual,
    }


def ingest_frame(image, *, objects, task_annotations, environment, task_geometry, lineage,
                 database=DEFAULT_DATABASE, origin='machine', review=None,
                 expected_frame_id=None, expected_sha256=None):
    """Decode one image and atomically store its shared objects, evidence and per-task roles.

    Pixel rectangles are half-open in the source frame. `review`, when supplied,
    receives the already-decoded PIL image and may return the fields
    `objects`, `task_annotations`, `environment`, or `task_geometry` for this review.
    `environment.primary_subject_task` can explicitly exclude the frame from
    that task's composite backgrounds.
    """
    from PIL import Image

    image = workspace_path(image)
    raw_bytes = (WORKSPACE / image).read_bytes()
    digest = sha256(raw_bytes).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f'Image bytes changed since machine review: {image}')
    with Image.open(BytesIO(raw_bytes)) as pixels:
        pixels.load()
        size = list(pixels.size)
        image_properties = {'mode': pixels.mode, 'format': pixels.format, 'image_size': size}
        if review is not None:
            reviewed = review(pixels)
            objects = reviewed.get('objects', objects)
            task_annotations = reviewed.get('task_annotations', task_annotations)
            environment = reviewed.get('environment', environment)
            task_geometry = reviewed.get('task_geometry', task_geometry)
        primary_subject_task = environment.get('primary_subject_task')
        if primary_subject_task not in (None, '') and (
            not isinstance(primary_subject_task, str) or primary_subject_task not in TASKS
        ):
            raise ValueError('primary_subject_task must name a V4 task or be empty')
        object_map = {}
        for item in objects:
            object_id = item.get('object_id')
            if not isinstance(object_id, str) or not object_id or object_id in object_map:
                raise ValueError(f'Missing or repeated shared object_id: {object_id}')
            if 'xyxy' in item and not _bounds(item['xyxy'], size):
                raise ValueError(f'Shared object box outside the source frame: {object_id}')
            object_map[object_id] = dict(item)
        if invalid := (set(task_annotations) | set(task_geometry)) - TASKS:
            raise ValueError(f'Unknown V4 tasks: {sorted(invalid)}')
        tasks = sorted(set(task_annotations) | set(task_geometry))
        decoded = {'pixels': pixels}
        annotations = {
            task: _label(object_map, task_annotations.get(task, {'presence': 'unknown', 'completeness': 'unknown'}), size)
            for task in tasks
        }
        if primary_subject_task in annotations and annotations[primary_subject_task]['presence'] == 'absent':
            raise ValueError('primary_subject_task cannot have absent target presence')
        if not (environment.keys() - {'primary_subject_task'}) and _GROUND_TASKS.intersection(tasks):
            environment = {**_shared_ground(decoded), **environment}
        reviewed_geometry = dict(task_geometry)
        if patch := environment.get('placement_patch'):
            for task in _GROUND_TASKS.intersection(tasks) - reviewed_geometry.keys():
                reviewed_geometry[task] = {
                    'ground_polygon': patch, 'horizon_y': environment.get('horizon_y'),
                    'reference_quad_px': patch[::-1] if len(patch) == 4 else None,
                    'evidence': {'basis': 'shared_ground_placement_patch'},
                }
        geometry = {
            task: _geometry(decoded, size, task, reviewed_geometry.get(task, {}), object_map)
            for task in tasks
        }

    parent = (lineage.get('parent_group') or lineage.get('mother_scene') or lineage.get('source_blend') or
              lineage.get('parent_video') or lineage.get('source_group'))
    metadata = {
        **lineage, 'image': image, 'image_size': size, 'source_image_size': size,
        'image_attributes': image_properties, 'source_image': lineage.get('source_image', image),
        'original_path': lineage.get('original_path', image),
        'working_path': image if image.startswith('datasets/raw/') else None,
        'capture_identity': lineage.get('capture_identity', image),
        'parent_group': historical_source(parent, read_source_history(WORKSPACE)) if parent else None,
        'provenance': lineage.get('provenance', {'image': image, 'content_sha256': digest}),
    }
    source_kind = metadata.get('source_kind')
    direct_source = source_kind in _DIRECT_SOURCES
    if direct_source:
        domain = 'camera'
    elif source_kind == 'blender_render':
        domain = 'rendered'
    elif source_kind in {'digital', 'cutout', 'artwork', 'texture'}:
        domain = 'material'
    else:
        domain = 'unknown'
    environment_status = environment.get('status', 'not_applicable' if domain == 'material' else 'unknown')
    scene_reasons = {
        'observed': 'recorded_scene_observation', 'unknown': 'scene_evidence_missing',
        'blocked': 'scene_blocked', 'not_applicable': 'non_scene_material',
    }
    if environment_status not in scene_reasons or environment.get('coordinate_size', size) != size:
        raise ValueError('Shared scene observation must use a valid status and exact image size')
    environment = {
        'scope': {'kind': 'frame'}, 'algorithm_version': 'scene_observation_v1', **environment,
        'coordinate_size': size, 'source_domain': domain, 'status': environment_status,
        'reason_codes': environment.get('reason_codes') or [scene_reasons[environment_status]],
    }
    output = {}
    with edit_database(database) as writer:
        frame_id = writer.register(image, image_size=size, content_sha256=digest,
                                   parent_group=metadata.get('parent_group'))
        if expected_frame_id is not None and frame_id != expected_frame_id:
            raise ValueError(f'Machine review refers to a different frame identity: {image}')
        writer.observe(frame_id, kind='lineage', scope='frame', evidence_key=f'intake:{image}',
                       reference=metadata['provenance'], payload=metadata, origin=origin)
        writer.observe(
            frame_id, kind='objects', scope='frame', evidence_key='intake',
            reference=environment.get('evidence', {'image': image}),
            payload={'coordinate_size': size, 'objects': list(object_map.values())}, origin=origin,
        )
        environment_id = writer.observe(
            frame_id, kind='environment', scope='frame', evidence_key='intake',
            reference=environment.get('evidence', {'image': image}),
            payload=environment, origin=origin, algorithm_version=environment.get('algorithm_version'),
        )
        for task in tasks:
            label, placement = annotations[task], geometry[task]
            split = label.get('split') or placement.get('split')
            labels_id = writer.observe(
                frame_id, kind='label', task=task, scope=label['scope']['kind'], evidence_key='intake',
                reference=label.get('evidence', {'image': image}), payload=label, origin=origin,
                algorithm_version=label.get('algorithm_version'),
            )
            placement_id = writer.observe(
                frame_id, kind='geometry', task=task, scope='placement', evidence_key='intake',
                reference=placement.get('evidence', {'image': image}), payload=placement, origin=origin,
                algorithm_version=(placement.get('placeable_ground') or placement).get('algorithm_version'),
            )
            full = label['scope']['kind'] == 'frame' and label['completeness'] == 'complete'
            positive, absent = label['presence'] == 'present', label['presence'] == 'absent'
            direct_ready = direct_source and full and positive
            empty_ready = domain in {'camera', 'rendered'} and full and absent
            placement_roles = {'candidate': 'eligible', 'blocked': 'blocked', 'unknown': 'unknown',
                               'not_applicable': 'blocked'}
            placement_role_status = placement_roles[placement['status']]
            composite = placement_role_status
            if composite == 'eligible' and label['completeness'] != 'complete':
                composite = 'unknown'
            if domain == 'material':
                composite = 'blocked'
            reasons = [] if label['completeness'] == 'complete' else ['task_label_coverage_unresolved']
            if label['scope']['kind'] != 'frame' and placement.get('scope') != label['scope']:
                composite = 'unknown' if composite == 'eligible' else composite
                reasons.append('placement_and_label_scopes_differ')
            if positive and task in {'sign', 'e_bike'} and not placement.get('existing_target_transform_evidence'):
                composite = 'unknown' if composite == 'eligible' else composite
                reasons.append('existing_target_transform_unresolved')
            if placement['status'] != 'candidate':
                reasons.extend(placement['reason_codes'])
            composite_evidence_ids = [labels_id, placement_id]
            if primary_subject_task == task:
                composite = 'blocked'
                reasons = ['primary_task_target_is_frame_subject']
                composite_evidence_ids.append(environment_id)
            details = {**metadata, 'scope': label['scope']}
            decision_specs = {
                'direct_target': ('eligible' if direct_ready else 'blocked' if absent else 'unknown',
                                  [labels_id], ['complete_direct_annotation'] if direct_ready else
                                  ['direct_positive_unavailable']),
                'empty_negative': ('eligible' if empty_ready else 'blocked' if positive else 'unknown',
                                   [labels_id], ['full_frame_target_absence'] if empty_ready else
                                   ['full_frame_target_absence_unproven']),
                'composite_background': (composite, composite_evidence_ids,
                                         reasons or ['placement_and_labels_cover_use']),
            }
            if task == 'waste_sticker' and placement.get('bins'):
                surface_status = placement.get('surface_status', placement['status'])
                surface_reasons = (['recorded_sticker_surfaces'] if surface_status == 'candidate' else
                                   ['sticker_surface_placement_unresolved']) if 'surface_status' in placement else placement['reason_codes']
                decision_specs['surface'] = (placement_roles[surface_status], [placement_id], surface_reasons)
            if any(placement.get(key) for key in ('ground_polygon', 'regions', 'station_crop_xyxy')):
                decision_specs['geometry_reference'] = (
                    'eligible', [placement_id], ['recorded_task_coordinate_reference'],
                )
            output[task] = {
                role: writer.decide(frame_id, task=task, role=role,
                                    # Coordinates belong to the frame; split-specific quota roles keep their guard.
                                    split='unassigned' if role == 'geometry_reference' else split,
                                    status=status,
                                    evidence_ids=evidence, details={
                                        **details, 'reason_codes': reason,
                                        'role_basis': ('direct_annotation' if direct_source and
                                                       role in {'direct_target', 'empty_negative'} else 'background_review'),
                                        **({'label_evidence_id': labels_id} if labels_id in evidence else {}),
                                        **({'geometry_evidence_id': placement_id} if placement_id in evidence else {}),
                                    }, origin=origin)
                for role, (status, evidence, reason) in decision_specs.items()
            }
    return {'frame_id': frame_id, 'content_sha256': digest, 'image_size': size, 'records': output}


def ingest_review_batch(rows, *, database=DEFAULT_DATABASE):
    """Write explicit multi-task machine reviews, one image transaction at a time.

    An existing image may carry its reviewed frame ID and digest. No omitted task
    receives a default unknown decision; every selected task must provide both
    its semantic label and its task-specific placement conclusion. A failed row
    leaves earlier image transactions intact and the batch can be retried.
    """
    rows = list(rows)
    seen_images, seen_frames = set(), set()
    required = {'image', 'objects', 'task_annotations', 'task_geometry', 'environment', 'lineage'}
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f'Machine review row {index} must be a JSON object')
        if required - row.keys():
            raise ValueError(f'Machine review row {index} lacks {sorted(required - row.keys())}')
        image = workspace_path(row['image'])
        if image in seen_images or not (WORKSPACE / image).is_file():
            raise ValueError(f'Machine review row {index} repeats or lacks its image: {image}')
        seen_images.add(image)
        frame_id, digest = row.get('frame_id'), row.get('content_sha256')
        if (frame_id is None) != (digest is None):
            raise ValueError(f'Machine review row {index} needs both frame ID and digest')
        if frame_id is not None:
            if frame_id in seen_frames:
                raise ValueError(f'Machine review repeats a frame identity: {frame_id}')
            seen_frames.add(frame_id)
        annotations, geometry = row['task_annotations'], row['task_geometry']
        if (not isinstance(row['objects'], list) or not isinstance(annotations, dict) or
                not annotations or not isinstance(geometry, dict) or annotations.keys() != geometry.keys()):
            raise ValueError(f'Machine review row {index} needs objects and matching task labels/geometry')
        if not isinstance(row['environment'], dict) or not isinstance(row['lineage'], dict):
            raise ValueError(f'Machine review row {index} needs environment and lineage objects')
        if not row['lineage'].get('source_kind') or not row['lineage'].get('capture_identity'):
            raise ValueError(f'Machine review row {index} needs source kind and capture identity')
        for task, label in annotations.items():
            if task not in TASKS or not isinstance(label, dict) or not isinstance(geometry[task], dict):
                raise ValueError(f'Machine review row {index} has an invalid task: {task}')
            if label.get('split') not in {'train', 'val', 'test', 'unassigned'}:
                raise ValueError(f'Machine review row {index} must state the {task} split')
            evidence = label.get('evidence')
            if not isinstance(evidence, dict) or not evidence.get('method') or not (
                    evidence.get('model_version') or evidence.get('algorithm_version')):
                raise ValueError(f'Machine review row {index} needs versioned {task} visual evidence')
    results = []
    for row in rows:
        results.append(ingest_frame(
            row['image'], objects=row['objects'], task_annotations=row['task_annotations'],
            environment=row['environment'], task_geometry=row['task_geometry'], lineage=row['lineage'],
            expected_frame_id=row.get('frame_id'), expected_sha256=row.get('content_sha256'),
            database=database, origin='machine',
        ))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', nargs='?')
    parser.add_argument('--review-json', type=Path)
    parser.add_argument('--batch-jsonl', type=Path, help='One explicit multi-task machine review per image')
    parser.add_argument('--database', type=Path, default=DEFAULT_DATABASE)
    args = parser.parse_args()
    if args.batch_jsonl is not None:
        if args.image is not None or args.review_json is not None:
            parser.error('--batch-jsonl cannot be combined with a single image')
        with args.batch_jsonl.open(encoding='utf-8-sig') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        result = ingest_review_batch(rows, database=args.database)
        print(json.dumps({'images': len(result), 'records': result}, indent=2, ensure_ascii=False))
        return
    if args.image is None or args.review_json is None:
        parser.error('an image and --review-json are required outside batch mode')
    review = json.loads(args.review_json.read_text(encoding='utf-8-sig'))
    result = ingest_frame(args.image, **review, database=args.database)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
