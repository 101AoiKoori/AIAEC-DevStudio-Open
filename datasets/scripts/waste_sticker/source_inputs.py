"""Resolve the designated sticker photos, frames, artwork and annotation records."""
import json
from pathlib import Path

from datasets.scripts.common.config import load_task_config
from datasets.scripts.common.frame_environment import image_digest, query_frames
from datasets.scripts.common.paths import SPLITS

ROOT = Path(__file__).resolve().parents[3]


def workspace_path(value, root=ROOT):
    if not isinstance(value, str) or Path(value).is_absolute():
        raise ValueError('source paths must be workspace relative')
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('source path escapes the workspace')
    return path


def artwork_catalog(config=None, root=ROOT):
    if config is None:
        _, config = load_task_config('waste_sticker')
    items = {}
    for ref in config['records']:
        for line in workspace_path(ref, root).read_text(encoding='utf8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for use in row.get('task_use_status', []):
                if (use.get('task') == 'waste_sticker' and use.get('role') == 'target_foreground'
                        and use.get('status') == 'ready'):
                    class_id = use['class_id']
                    path = workspace_path(row['image'], root)
                    if not path.is_relative_to(workspace_path(config['artwork_toolkit'], root)) or not path.is_file():
                        raise ValueError('sticker artwork is missing or outside the configured toolkit')
                    if class_id in items and items[class_id] != row['image']:
                        raise ValueError('sticker class resolves to multiple artworks')
                    items[class_id] = row['image']
    if set(items) != set(range(24)):
        raise ValueError('configured artwork records must cover exactly the 24 sticker classes')
    return items


def real_source_catalog(config=None, root=ROOT):
    """Return all designated sources, retaining missing annotations and split restrictions."""
    if config is None:
        _, config = load_task_config('waste_sticker')
    indices = config['frame_indices']
    first, last, stationary_first, stationary_last = (
        indices[k] for k in ('first', 'last', 'stationary_first', 'stationary_last'))
    if any(type(i) is not int for i in (first, last, stationary_first, stationary_last)) or not (
            0 <= first <= stationary_first <= stationary_last <= last):
        raise ValueError('invalid source frame interval')
    pattern = config['frame_filename_pattern']
    filenames = [pattern.format(index=i) for i in range(first, last + 1)]
    if len(set(filenames)) != len(filenames) or any(Path(name).name != name for name in filenames):
        raise ValueError('frame pattern must name distinct files')
    if config['real_video_frames'] != filenames[0] + '..' + filenames[-1]:
        raise ValueError('frame description differs from configured numeric interval')
    stationary = pattern.format(index=stationary_first) + '..' + pattern.format(index=stationary_last)
    if config['stationary_frame_range'] != stationary:
        raise ValueError('stationary description differs from configured numeric interval')
    annotations_path = workspace_path(config['real_annotations'], root)
    annotations = json.loads(annotations_path.read_text(encoding='utf8'))
    by_image = {}
    for index, record in enumerate(annotations):
        if record['image'] in by_image:
            raise ValueError('duplicate real sticker annotation')
        by_image[record['image']] = (index, record)
    photo = config['real_sticker_source']
    sources = [(photo, None)] + [
        ((Path(config['real_video_frames_root']) / name).as_posix(), number)
        for name, number in zip(filenames, range(first, last + 1))]
    result = []
    for image, number in sources:
        match = by_image.get(image)
        result.append({'image': image, 'exists': workspace_path(image, root).is_file(),
                       'frame_number': number,
                       'stationary': number is not None and stationary_first <= number <= stationary_last,
                       'parent': config['real_video_parent'] if number is not None else photo,
                       'annotation_manifest': config['real_annotations'],
                       'record_index': None if match is None else match[0],
                       'annotation': None if match is None else match[1],
                       'selected_for_current_batch': (image in config['current_selection']['direct_images']) if config.get('current_selection') else None,
                       'gap': 'annotation_missing' if match is None else None})
    return result


def select_direct_source(catalog, image, split, *, formal=False, reviewed=None):
    matches = [row for row in catalog if row['image'] == image]
    if len(matches) != 1 or not matches[0]['exists']:
        raise ValueError('direct sticker image is missing or outside the designated source interval')
    selected = matches[0]
    if selected.get('selected_for_current_batch') is False:
        raise ValueError('direct sticker source is excluded or partial in the current source.yaml batch')
    annotation = selected['annotation']
    if not annotation:
        raise ValueError('direct sticker image lacks an annotation record')
    boxes = annotation.get('boxes')
    width, height = annotation['image_size']
    if not isinstance(boxes, list) or any(not isinstance(box.get('class_id'), int) or not 0 <= box['class_id'] < 24
                        for box in boxes):
        raise ValueError('direct sticker labels contain unknown classes')
    for box in boxes:
        x1, y1, x2, y2 = box['xyxy']
        if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
            raise ValueError('direct sticker label is outside its source image')
    if formal:
        if not reviewed or image not in reviewed:
            raise ValueError('direct sticker frame has no eligible complete task label in the frame database: ' + image)
        environment, truth = reviewed[image]
        if (environment['split'] != split or truth['state'] not in ('complete_positive', 'complete_negative') or
                environment['role'] != ('direct_target' if boxes else 'empty_negative') or
                sorted((box['class_id'], tuple(box['xyxy'])) for box in boxes) !=
                sorted((box['class_id'], tuple(box['xyxy'])) for box in truth['instances'])):
            raise ValueError('direct sticker annotations differ from the complete stored task label')
        selected = {**selected, 'frame_environment': environment}
    else:
        uses = [use for use in annotation.get('task_use_status', [])
                if use.get('task') == 'waste_sticker' and use.get('split') == split
                and use.get('role') == 'direct_sample']
        if len(uses) != 1 or uses[0].get('status') not in ('candidate', 'ready', 'accepted') or not boxes:
            raise ValueError('direct sticker source is excluded, partial or unassigned for this split')
    return selected


def surface_background_catalog(config=None, root=ROOT, images=None, *, split='train'):
    """Read persisted task eligibility for this batch's selected bin surfaces."""
    if config is None:
        _, config = load_task_config('waste_sticker')
    selected = set(config['current_selection']['surface_images'] if images is None else images)
    if split not in SPLITS or not selected or any(not isinstance(image, str) or not image for image in selected):
        raise ValueError('sticker surface batch must select image paths and a dataset split')
    database = workspace_path(config['databases']['frame_environment'], root)
    eligible = query_frames(task='waste_sticker', role='composite_background', split=split,
                            database=database)
    result = {}
    for record in eligible:
        matches = selected.intersection(record['paths'])
        if not matches:
            continue
        label, geometry = record.get('label') or {}, record.get('geometry') or {}
        bins = geometry.get('bins') or []
        if (record['image_size'] != geometry.get('coordinate_size') or
                label.get('presence') not in ('absent', 'present') or
                label.get('completeness') != 'complete' or
                not bins or any(not item.get('allowed_polygon') or
                                not isinstance(item.get('exclude_polygons'), list) or
                                not isinstance(item.get('occluder_polygons'), list) for item in bins)):
            raise ValueError('stored sticker background lacks complete labels or bin surface geometry')
        for image in matches:
            if image in result or image_digest(workspace_path(image, root)) != record['content_sha256']:
                raise ValueError('stored sticker background identity differs: ' + image)
            kind = 'blender_render' if record.get('source_kind') == 'blender_render' else 'camera_photo'
            result[image] = {'frame_id': record['frame_id'], 'record_id': record['record_id'],
                             'evidence_ids': [item['evidence_id'] for item in record['evidence']],
                             'parent_group': record['parent_group'], 'source_video': record.get('parent_video'),
                             'background_kind': kind, 'source_blend': record.get('source_blend'),
                             'scene': record.get('scene'), 'route': record.get('route'),
                             'render_frame': record.get('render_frame'),
                             'split': split, 'geometry': geometry, 'label': label}
    if set(result) != selected:
        raise ValueError('selected sticker surfaces lack eligible stored task records: '
                         + ', '.join(sorted(selected - set(result))))
    return result
