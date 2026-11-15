"""Resolve the designated sticker photos, frames, artwork and annotation records."""
import json
from pathlib import Path

from datasets.scripts.common.config import load_task_config

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


def select_direct_source(catalog, image, split, *, formal=False):
    matches = [row for row in catalog if row['image'] == image]
    if len(matches) != 1 or not matches[0]['exists']:
        raise ValueError('direct sticker image is missing or outside the designated source interval')
    selected = matches[0]
    if selected.get('selected_for_current_batch') is False:
        raise ValueError('direct sticker source is excluded or partial in the current source.yaml batch')
    annotation = selected['annotation']
    if not annotation:
        raise ValueError('direct sticker image lacks an annotation record')
    uses = [use for use in annotation.get('task_use_status', [])
            if use.get('task') == 'waste_sticker' and use.get('split') == split
            and use.get('role') == 'direct_sample']
    if len(uses) != 1 or uses[0].get('status') not in ('candidate', 'ready', 'accepted'):
        raise ValueError('direct sticker source is excluded, partial or unassigned for this split')
    boxes = annotation.get('boxes')
    width, height = annotation['image_size']
    if not boxes or any(not isinstance(box.get('class_id'), int) or not 0 <= box['class_id'] < 24
                        for box in boxes):
        raise ValueError('direct sticker labels contain unknown classes')
    for box in boxes:
        x1, y1, x2, y2 = box['xyxy']
        if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
            raise ValueError('direct sticker label is outside its source image')
    return selected


def surface_background_catalog(config=None, root=ROOT):
    """Read the existing empty-sticker observations for this batch's selected surfaces."""
    if config is None:
        _, config = load_task_config('waste_sticker')
    manifest = config['surface_background_roles']
    rows = [json.loads(line) for line in workspace_path(manifest, root).read_text(encoding='utf8').splitlines() if line.strip()]
    selected = set(config['current_selection']['surface_images'])
    result = {}
    for index, row in enumerate(rows):
        if row['image'] not in selected:
            continue
        uses = [use for use in row.get('task_use_status', []) if use.get('task') == 'waste_sticker'
                and use.get('split') == 'train' and use.get('role') == 'background' and use.get('status') == 'ready'
                and use.get('background_item_status') == 'ai_visual_no_visible_waste_item_sticker']
        if len(uses) != 1 or not workspace_path(row['image'], root).is_file() or not workspace_path(row['source_video'], root).is_file():
            raise ValueError('selected surface lacks its recorded usable background or parent video')
        result[row['image']] = {'manifest': manifest, 'record_index': index,
                                'source_video': row['source_video'], 'observation': uses[0]}
    if set(result) != selected:
        raise ValueError('selected surface is absent from the configured background role manifest')
    return result
