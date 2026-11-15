"""Composite a rendered meter foreground onto indexed real backgrounds as candidates.

Inputs must name existing manifest records that directly associate each background
with its raw photograph or video. The result is not an accepted training set:
ground contact, scene occlusion, reflections and other meter instances need review.
"""


import argparse
import filecmp
import json
import math
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'lab/scripts/media')]
from lab.training_common.production_guard import generation_guard
import media_index
from datasets.scripts.common.config import load_task_config
from datasets.scripts.common.paths import sample_dir, split_dir

POINTS = ('temperature_pivot', 'temperature_tip', 'humidity_pivot', 'humidity_tip',
          'temperature_min', 'temperature_max', 'humidity_min', 'humidity_max')
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
PARENT_SUFFIXES = IMAGE_SUFFIXES | {'.mp4', '.mov', '.avi', '.mkv'}


def workspace(value, *, existing=True):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    if path.is_symlink():
        raise ValueError('symlink input or output is not supported')
    path = path.resolve()
    if path == ROOT or not path.is_relative_to(ROOT):
        raise ValueError('path must be inside workspace: ' + str(value))
    if existing and not path.is_file():
        raise ValueError('input is missing: ' + str(value))
    return path


def rel(path):
    return path.relative_to(ROOT).as_posix()


def record_at(path, index):
    if path.suffix.lower() == '.json':
        data = json.loads(path.read_text(encoding='utf8'))
        if isinstance(data, dict) and isinstance(data.get('images'), list):
            return data['images'][index] if 0 <= index < len(data['images']) else None
    return next((row for i, row, _ in media_index.manifest_rows(path) if i == index and row is not None), None)


def declared(row, manifest, keys):
    result = set()
    for key in keys:
        values = row.get(key)
        for value in values if isinstance(values, list) else [values]:
            if isinstance(value, str):
                result.update(media_index.path_candidates(value, manifest))
    return result


def configured_backgrounds(config):
    """Resolve only the explicitly selected existing review rows, without a scan."""
    manifest = workspace(config['background_review'])
    rows = json.loads(manifest.read_text(encoding='utf8'))['images']
    selected = config['selected_background_records']
    if not selected or config.get('background_usage') != 'candidate_reuse_of_recorded_regression':
        raise ValueError('meter real_composite requires explicit candidate background reuse')
    result = []
    seen = set()
    for choice in selected:
        index = choice['record_index']
        if type(index) is not int or index < 0 or index >= len(rows) or index in seen:
            raise ValueError('invalid or duplicate configured background record index')
        seen.add(index)
        row = rows[index]
        if row.get('id') != choice['id'] or row.get('pose_state') != 'candidate_complete_negative':
            raise ValueError('selected background identity or candidate label state changed')
        if not row.get('label') or workspace(row['label']).read_text(encoding='utf8').strip():
            raise ValueError('configured negative background must have its recorded empty candidate label')
        result.append([row['image'], row['parent_video'], rel(manifest), str(index), 'review_record'])
    return result


def background_spec(values, *, candidate_reuse=False):
    image, parent, manifest = (workspace(value) for value in values[:3])
    index, relation = int(values[3]), values[4]
    if image.suffix.lower() not in IMAGE_SUFFIXES or parent.suffix.lower() not in PARENT_SUFFIXES:
        raise ValueError('background must be an image and parent must be a photo or video')
    if index < 0 or relation not in ({'describes', 'referenced_as', 'review_record'} if candidate_reuse else {'describes', 'referenced_as'}):
        raise ValueError('background evidence must be an indexed describes or referenced_as record')
    row = record_at(manifest, index)
    if not isinstance(row, dict):
        raise ValueError('manifest record does not exist')
    declared_images = declared(row, manifest, ('image', 'image_path', 'path', 'frame', 'file',
                                               'parent_image', 'source_image', 'background_parent'))
    if rel(image) not in declared_images:
        raise ValueError('manifest record does not describe the requested background')
    restricted = {row.get('usage'), row.get('split'), row.get('disposition')}
    if 'exclude' in restricted:
        raise ValueError('background is excluded')
    if not candidate_reuse and 'regression' in restricted:
        raise ValueError('background is restricted to regression or excluded')
    if candidate_reuse and row.get('pose_state') != 'candidate_complete_negative':
        raise ValueError('candidate reuse cannot promote a partial or unknown background')
    return {'image': image, 'parent': parent, 'manifest': manifest, 'record_index': index,
            'relation': relation, 'row': row}


def lineage_link(values):
    """One explicitly recorded subject-to-parent edge; paths alone do not prove it."""
    subject, parent, manifest = (workspace(value) for value in values[:3])
    index, relation = int(values[3]), values[4]
    if index < 0 or relation not in {'describes', 'referenced_as'}:
        raise ValueError('invalid indexed lineage link')
    row = record_at(manifest, index)
    if not isinstance(row, dict):
        raise ValueError('lineage record does not exist')
    subject_keys = ('image', 'image_path', 'path', 'frame', 'file', 'parent_image',
                    'source_image', 'background_parent')
    parent_keys = ('parent_image', 'source_image', 'parent_video', 'source_video', 'video')
    if (rel(subject) not in declared(row, manifest, subject_keys) or
            rel(parent) not in declared(row, manifest, parent_keys) or subject == parent):
        raise ValueError('lineage record does not name the subject and an explicit parent')
    return {'subject': subject, 'parent': parent, 'manifest': manifest, 'record_index': index,
            'relation': relation, 'row': row}


def validate_lineages(backgrounds, links):
    for item in backgrounds:
        queue = [item['image']]
        visited = set()
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            for link in links:
                if link['subject'] == node:
                    queue.append(link['parent'])
            # A background record may name both its described/referenced image
            # and a direct source parent, including a parent video.
            parent_keys = ('parent_video', 'source_video', 'parent_image', 'source_image', 'video')
            if node == item['image'] and rel(item['parent']) in declared(item['row'], item['manifest'], parent_keys):
                queue.append(item['parent'])
        if item['parent'] not in visited:
            raise ValueError('background-to-parent lineage lacks explicitly recorded links')
        parent = item['parent']
        relevant = [item] + [link for link in links if link['parent'] == parent]
        if parent.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv'}:
            if not any(rel(parent) in declared(link['row'], link['manifest'], ('parent_video', 'source_video', 'video'))
                       for link in relevant):
                raise ValueError('parent video is not explicitly recorded as a video source')
        elif not any(link['row'].get('generation_method') == 'camera_capture' or
                     link['row'].get('source_kind') in {'photo', 'task_point'} for link in relevant):
            raise ValueError('parent photo lacks an explicit camera-capture declaration')


def pose_values(path, record):
    lines = path.read_text(encoding='utf8').splitlines()
    if len(lines) != 1:
        raise ValueError('source must have one pose object')
    values = [float(value) for value in lines[0].split()]
    if len(values) != 29 or values[0] != 0 or not all(math.isfinite(value) for value in values):
        raise ValueError('invalid 8-point pose label')
    if not all(0 <= value <= 1 for value in values[1:5]) or min(values[3:5]) <= 0:
        raise ValueError('invalid source bounding box')
    mask = record.get('dial_raster_evidence', {})
    box = mask.get('bbox_xyxy_pixels')
    if mask.get('bbox_status') != 'candidate_from_isolated_raster_and_visible_fraction' or not box:
        raise ValueError('source bounding box lacks raster evidence')
    x1, y1, x2, y2 = box
    expected = ((x1+x2)/1280, (y1+y2)/960, (x2-x1)/640, (y2-y1)/480)
    if any(abs(a-b) > 1e-6 for a, b in zip(values[1:5], expected)):
        raise ValueError('source bounding box and raster evidence differ')
    for index, name in enumerate(POINTS):
        x, y, visibility = values[5 + index * 3:8 + index * 3]
        info = record['keypoint_visibility'][name]
        if visibility not in (0, 1, 2) or visibility != info['visibility']:
            raise ValueError('source visibility and evidence do not match')
        if visibility == 0:
            if (x, y) != (0, 0):
                raise ValueError('outside point must have zero coordinates')
        else:
            projected = record['keypoints_xy_depth_normalized_candidate'][name]
            if not (0 <= x < 1 and 0 <= y < 1 and abs(x-projected[0]) < 1e-6 and abs(y-projected[1]) < 1e-6):
                raise ValueError('source point differs from its geometric projection')
            # Changing the background changes reflected/visible image evidence.
            values[7 + index * 3] = 1
    return values


def source_rows(batch):
    manifest_path = batch / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if manifest.get('status') != 'complete_synthetic_pose_candidates' or not manifest.get('foreground_policy'):
        raise ValueError('source batch lacks complete pose and full-station RGBA evidence')
    if manifest.get('resolution') != [640, 480] or manifest.get('keypoint_order') != list(POINTS):
        raise ValueError('source pose order or resolution differs')
    records_path = batch / 'samples.jsonl'
    rows = [(index, json.loads(line)) for index, line in enumerate(records_path.read_text(encoding='utf8').splitlines())
            if line.strip()]
    if len(rows) != manifest.get('completed_count') or len(rows) != manifest.get('requested_count'):
        raise ValueError('source batch is incomplete')
    selected = []
    for index, row in rows:
        if not row.get('pose_label'):
            continue
        stem = row.get('id')
        if not isinstance(stem, str) or not stem.startswith('arena_meter_') or not stem[12:].isdigit():
            raise ValueError('invalid source identifier')
        if (row.get('image') != 'images/' + stem + '.png' or
                row.get('foreground_rgba') != 'foreground/' + stem + '.png' or
                row.get('pose_label') != 'labels/' + stem + '.txt' or
                row.get('annotation_status') != 'synthetic_pose_geometric'):
            raise ValueError('inconsistent source references')
        image = workspace(batch / row['image'])
        foreground = workspace(batch / row['foreground_rgba'])
        label = workspace(batch / row['pose_label'])
        annotation = workspace(batch / 'annotations' / (stem + '.json'))
        if json.loads(annotation.read_text(encoding='utf8')) != row:
            raise ValueError('source annotation differs from batch record')
        values = pose_values(label, row)
        selected.append((index, row, image, foreground, label, annotation, values))
    if not selected or len(selected) != manifest.get('pose_label_count'):
        raise ValueError('source label count is missing or inconsistent')
    return manifest_path, records_path, manifest, selected


def compose(background, foreground):
    """Center-crop background to 4:3; leave the 640x480 foreground untransformed."""
    if foreground.size != (640, 480) or foreground.mode != 'RGBA':
        raise ValueError('foreground must be native 640x480 RGBA')
    width, height = background.size
    if min(width, height) < 240:
        raise ValueError('background is too small for bounded 640x480 composition')
    crop_width = min(width, int(height * 4 / 3))
    crop_height = min(height, int(width * 3 / 4))
    left, top = (width-crop_width)//2, (height-crop_height)//2
    crop = (left, top, left+crop_width, top+crop_height)
    fitted = background.convert('RGB').crop(crop).resize((640, 480), Image.Resampling.LANCZOS)
    alpha = foreground.getchannel('A')
    lo, hi = alpha.getextrema()
    if lo != 0 or hi < 128 or alpha.getbbox() is None:
        raise ValueError('foreground has no clear transparency or useful opaque station')
    return Image.alpha_composite(fitted.convert('RGBA'), foreground).convert('RGB'), {
        'source_size': [width, height], 'crop_xyxy': list(crop), 'output_size': [640, 480],
        'upsampled': crop_width < 640 or crop_height < 480,
        'note': 'Cropping and resampling do not add captured detail.'}


def validate_placement(placement):
    """Validate an output-image region and a uniform foreground transform.

    The rectangle is a proposed placement region, not a measured 3-D plane.
    Coordinates use 640x480 output pixels with a top-left origin.
    """
    if not isinstance(placement, dict) or set(placement) != {'scale', 'offset_xy', 'region_xyxy'}:
        raise ValueError('placement requires scale, offset_xy and region_xyxy')
    scale, offset, region = placement['scale'], placement['offset_xy'], placement['region_xyxy']
    def numeric(value):
        return type(value) in (int, float) and math.isfinite(value)
    if not numeric(scale) or not .25 <= scale <= 2:
        raise ValueError('placement scale must be between 0.25 and 2')
    if not isinstance(offset, list) or len(offset) != 2 or not all(map(numeric, offset)):
        raise ValueError('placement offset must contain two finite pixel coordinates')
    if not isinstance(region, list) or len(region) != 4 or not all(map(numeric, region)):
        raise ValueError('placement region must contain four finite pixel coordinates')
    x1, y1, x2, y2 = region
    if not (0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480):
        raise ValueError('placement region must be a nonempty rectangle inside the output')


def place_foreground(foreground, values, placement):
    """Transform RGBA and candidate box/points together without clipping."""
    validate_placement(placement)
    if foreground.size != (640, 480) or foreground.mode != 'RGBA':
        raise ValueError('foreground must be native 640x480 RGBA')
    alpha = foreground.getchannel('A')
    lo, hi = alpha.getextrema()
    bounds = alpha.getbbox()
    if lo != 0 or hi < 128 or bounds is None:
        raise ValueError('foreground has no clear transparency or useful opaque station')
    scale = placement['scale']
    dx, dy = placement['offset_xy']
    region = placement['region_xyxy']
    transformed = [bounds[0]*scale+dx, bounds[1]*scale+dy,
                   bounds[2]*scale+dx, bounds[3]*scale+dy]
    if (transformed[0] < region[0] or transformed[1] < region[1] or
            transformed[2] > region[2] or transformed[3] > region[3]):
        raise ValueError('transformed foreground falls outside placement region')
    if len(values) != 29 or values[0] != 0 or not all(math.isfinite(v) for v in values):
        raise ValueError('invalid source pose candidate')
    result = list(values)
    cx, cy, width, height = values[1:5]
    x1, x2 = (cx-width/2)*640*scale+dx, (cx+width/2)*640*scale+dx
    y1, y2 = (cy-height/2)*480*scale+dy, (cy+height/2)*480*scale+dy
    if not (0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480):
        raise ValueError('transformed dial box falls outside output')
    result[1:5] = [(x1+x2)/1280, (y1+y2)/960, (x2-x1)/640, (y2-y1)/480]
    for index in range(8):
        pos = 5 + 3*index
        x, y, visibility = values[pos:pos+3]
        if visibility == 0:
            if (x, y) != (0, 0):
                raise ValueError('outside point must have zero coordinates')
            continue
        if visibility not in (1, 2) or not (0 <= x < 1 and 0 <= y < 1):
            raise ValueError('invalid candidate point')
        x, y = x*640*scale+dx, y*480*scale+dy
        if not (0 <= x < 640 and 0 <= y < 480):
            raise ValueError('transformed candidate point falls outside output')
        result[pos:pos+3] = [x/640, y/480, 1]
    inverse = (1/scale, 0, -dx/scale, 0, 1/scale, -dy/scale)
    image = foreground.transform((640, 480), Image.Transform.AFFINE, inverse,
                                 resample=Image.Resampling.BICUBIC)
    if image.getchannel('A').getextrema()[1] < 128:
        raise ValueError('transformed foreground has no useful opaque station')
    return image, result, {'kind': 'uniform_scale_translate', 'scale': scale,
                           'offset_xy': [dx, dy], 'region_xyxy': list(region),
                           'foreground_bounds_xyxy': transformed,
                           'region_semantics': 'declared image-space placement region; physical fit unconfirmed'}


def validate_plan(plan, count, backgrounds, selected):
    """Check declared background/foreground pairs without creating images."""
    if not isinstance(plan, dict) or plan.get('model_key') != 'meter' or plan.get('target_split') != 'train':
        raise ValueError('composite plan must target meter train')
    if type(count) is not int or count < 1 or plan.get('budget') != count:
        raise ValueError('count must equal the positive plan budget')
    constraints = plan.get('constraints')
    if constraints not in ({'resolution': [640, 480], 'geometry': 'identity', 'label_policy': 'candidate_pose'},
                            {'resolution': [640, 480], 'geometry': 'uniform_scale_translate',
                             'label_policy': 'candidate_pose'}):
        raise ValueError('unsupported composite constraints')
    geometry = constraints['geometry']
    background_map = {rel(item['image']): item for item in backgrounds}
    foreground_map = {row['id']: item for item in selected for row in [item[1]]}
    if len(background_map) != len(backgrounds) or len(foreground_map) != len(selected):
        raise ValueError('backgrounds and foreground IDs must be unique')
    states = plan.get('strata')
    if not isinstance(states, list) or not states:
        raise ValueError('composite plan requires strata')
    seen = set()
    for state in states:
        if not isinstance(state, dict):
            raise ValueError('composite state must be an object')
        expected = {'background': {'kind': 'real_captured_scene'},
                    'target': {'kind': 'rendered_station'}, 'labels': {'kind': 'candidate_pose'},
                    'augmentation': {'kind': 'alpha_composite_identity' if geometry == 'identity'
                                     else 'alpha_composite_uniform_scale_translate'}}
        if any(state.get(key) != value for key, value in expected.items()) or state.get('conditions') != []:
            raise ValueError('unsupported composite state semantics')
        if type(state.get('max_count')) is not int or not 0 <= state['max_count'] <= 1:
            raise ValueError('each exact foreground/background pair has capacity at most one')
        params = state.get('parameters')
        required = {'background_image', 'foreground_id'} | ({'placement'} if geometry != 'identity' else set())
        if not isinstance(params, dict) or set(params) != required:
            raise ValueError('composite parameters do not match the declared geometry')
        if geometry != 'identity':
            validate_placement(params['placement'])
        bg, fg = params['background_image'], params['foreground_id']
        if not isinstance(bg, str) or not isinstance(fg, str) or bg not in background_map or fg not in foreground_map:
            raise ValueError('planned composite source pair is unavailable')
        item, source = background_map[bg], foreground_map[fg]
        expected_sources = list(dict.fromkeys([bg, rel(item['parent']), rel(source[2]), rel(source[3])]))
        if state.get('sources') != expected_sources:
            raise ValueError('composite sources do not match the selected background and foreground')
        if (bg, fg) in seen:
            raise ValueError('duplicate exact foreground/background pair')
        seen.add((bg, fg))
    return background_map, foreground_map


def background_meter_review(path, backgrounds):
    """Preserve recorded candidate-negative status and its historical use."""
    raw = path.read_bytes()
    data = json.loads(raw.decode('utf-8-sig'))
    rows = data.get('images') if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise ValueError('meter review must contain image records')
    result = {}
    for item in backgrounds:
        image = rel(item['image'])
        matches = [(i, row) for i, row in enumerate(rows) if isinstance(row, dict) and row.get('image') == image]
        if len(matches) != 1:
            raise ValueError('background must have exactly one meter review record')
        index, row = matches[0]
        if row.get('pose_state') != 'candidate_complete_negative' or row.get('parent_video') != rel(item['parent']):
            raise ValueError('background is not recorded as a candidate negative from this parent')
        label = workspace(row.get('label', ''))
        if label.read_bytes().strip():
            raise ValueError('candidate negative label is not empty')
        result[image] = {'manifest': rel(path), 'record_index': index, 'label': rel(label),
                         'pose_state': row['pose_state'], 'prior_split': row.get('split'),
                         'human_accepted': row.get('human_accepted'),
                         'parent_video': row['parent_video']}
    return raw, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    _, source = load_task_config('meter')
    defaults = source['real_composite']
    parser.add_argument('--batch', default=defaults['foreground_batch'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--count', type=int, default=defaults['count'])
    parser.add_argument('--seed', type=int, default=20261113)
    parser.add_argument('--production-plan', default=defaults['production_plan'], help='Workspace JSON declaration')
    parser.add_argument('--background-meter-review', default=defaults['background_review'], help='Workspace JSON with candidate negative background records')
    parser.add_argument('--background', nargs=5, action='append',
                        metavar=('IMAGE', 'RAW_PARENT', 'MANIFEST', 'RECORD_INDEX', 'RELATION'))
    parser.add_argument('--lineage-link', nargs=5, action='append', default=[],
                        metavar=('SUBJECT', 'PARENT', 'MANIFEST', 'RECORD_INDEX', 'RELATION'))
    args = parser.parse_args()
    configured = configured_backgrounds(defaults)
    args.background = args.background if args.background is not None else configured
    args.lineage_link = args.lineage_link or defaults.get('lineage_links', [])
    if not args.background or any(list(row) not in configured for row in args.background):
        raise ValueError('backgrounds must be selected from meter.source.yaml review records')
    batch = workspace(args.batch, existing=False)
    if not batch.is_dir() or not batch.is_relative_to(ROOT / 'datasets'):
        raise ValueError('configured rendered meter foreground batch is missing')
    output = workspace(args.output, existing=False)
    if output not in {sample_dir('meter'), split_dir('meter', 'train')} or (output.exists() and any(output.iterdir())):
        raise ValueError('compositor output must be the empty current meter sample or train directory')
    if batch == output:
        raise ValueError('foreground batch and composite output must be different directories')
    if not 1 <= args.count <= 5000:
        raise ValueError('count must be between 1 and 5000')
    backgrounds = [background_spec(values, candidate_reuse=True) for values in args.background]
    links = [lineage_link(values) for values in args.lineage_link]
    validate_lineages(backgrounds, links)
    if len({item['image'] for item in backgrounds}) != len(backgrounds):
        raise ValueError('background image is repeated')
    review_path = workspace(args.background_meter_review) if args.background_meter_review else None
    review_bytes, review_context = background_meter_review(review_path, backgrounds) if review_path else (None, {})
    parents = sorted({item['parent'] for item in backgrounds})
    for index, parent in enumerate(parents):
        for other in parents[index + 1:]:
            if parent.stat().st_size == other.stat().st_size and filecmp.cmp(parent, other, shallow=False):
                raise ValueError('distinct parent paths contain identical original media')
    manifest_path, records_path, manifest, selected = source_rows(batch)
    plan_path = workspace(args.production_plan)
    plan_bytes = plan_path.read_bytes()
    declaration = json.loads(plan_bytes.decode('utf-8-sig'))
    sampling_evidence = (workspace(declaration['sampling_evidence'])
                         if declaration.get('sampling_evidence') else None)
    background_map, foreground_map = validate_plan(declaration, args.count, backgrounds, selected)
    sampling_seed, sampling_groups = None, {}
    if sampling_evidence:
        sampling = json.loads(sampling_evidence.read_text(encoding='utf-8-sig'))
        sampling_seed = sampling.get('sampling_seed')
        selections = sampling.get('selected')
        if type(sampling_seed) is not int or not isinstance(selections, list):
            raise ValueError('sampling evidence must contain an integer seed and selected pairs')
        for selection in selections:
            if (not isinstance(selection, list) or len(selection) != 3 or
                    any(not isinstance(v, str) or not v for v in selection)):
                raise ValueError('invalid selected sampling pair')
            key = tuple(selection[:2])
            if key in sampling_groups:
                raise ValueError('sampling evidence contains a duplicate pair')
            sampling_groups[key] = selection[2]
        planned_pairs = {(s['parameters']['background_image'], s['parameters']['foreground_id'])
                         for s in declaration['strata']}
        if set(sampling_groups) != planned_pairs:
            raise ValueError('sampling evidence and planned pairs differ')
    media = {item[key] for item in backgrounds for key in ('image', 'parent')}
    media.update(item[key] for item in links for key in ('subject', 'parent'))
    media.update(path for _, _, image, foreground, _, _, _ in selected for path in (image, foreground))
    configs = [ROOT / 'datasets/scripts/meter/config' / name for name in ('dataset.yaml', 'source.yaml')]
    procedural = {*configs, manifest_path, records_path, plan_path, *(item['manifest'] for item in backgrounds + links)}
    if sampling_evidence:
        procedural.add(sampling_evidence)
    if review_path:
        procedural.add(review_path)
        procedural.update(workspace(item['label']) for item in review_context.values())
    procedural.update(path for _, _, _, _, label, annotation, _ in selected for path in (label, annotation))
    params = {'count': args.count, 'seed': args.seed,
              'backgrounds': [{key: rel(item[key]) if key in ('image', 'parent', 'manifest') else item[key]
                               for key in ('image', 'parent', 'manifest', 'record_index', 'relation')}
                              for item in backgrounds]}
    params['lineage_links'] = [{key: rel(item[key]) if key in ('subject', 'parent', 'manifest') else item[key]
                                for key in ('subject', 'parent', 'manifest', 'record_index', 'relation')}
                               for item in links]
    params['background_meter_review'] = rel(review_path) if review_path else None
    params['sampling_evidence'] = rel(sampling_evidence) if sampling_evidence else None
    operation = 'meter.compose_real_background_v4:' + rel(output)
    with generation_guard(operation, sorted(media), [output], params,
                          procedural_inputs=sorted(procedural), code_paths=[Path(__file__)], root=ROOT,
                          production_plan=declaration, sample_output_paths=[output / 'images']) as run:
        if plan_path.read_bytes() != plan_bytes:
            raise ValueError('production plan changed while entering the guard')
        if review_path and review_path.read_bytes() != review_bytes:
            raise ValueError('background meter review changed while entering the guard')
        if run.reused:
            print('METER_REAL_COMPOSITE_REUSED', json.dumps(run.value, ensure_ascii=False))
            return
        for item in backgrounds:
            refs = run.source_refs[rel(item['image'])]
            if item['relation'] != 'review_record' and not any(ref.get('manifest') == rel(item['manifest']) and
                       ref.get('record_index') == item['record_index'] and
                       ref.get('relation') == item['relation'] for ref in refs):
                raise ValueError('background manifest record is absent from synchronized index')
            if record_at(item['manifest'], item['record_index']) != item['row']:
                raise ValueError('background source record changed while entering the guard')
        for item in links:
            refs = run.source_refs[rel(item['subject'])]
            if not any(ref.get('manifest') == rel(item['manifest']) and
                       ref.get('record_index') == item['record_index'] and
                       ref.get('relation') == item['relation'] for ref in refs):
                raise ValueError('lineage link is absent from synchronized index')
            if record_at(item['manifest'], item['record_index']) != item['row']:
                raise ValueError('lineage record changed while entering the guard')
        for index, _, image, _, _, _, _ in selected:
            if not any(ref.get('manifest') == rel(records_path) and ref.get('record_index') == index and
                       ref.get('relation') == 'describes' for ref in run.source_refs[rel(image)]):
                raise ValueError('foreground source sample record is absent from synchronized index')
        _, _, current_manifest, current_selected = source_rows(batch)
        if current_manifest != manifest or current_selected != selected:
            raise ValueError('foreground batch changed while entering the guard')
        states = []
        for state in run.plan['strata']:
            if type(state['planned_count']) is not int or state['planned_count'] not in (0, 1):
                raise ValueError('each planned pair can be used at most once')
            if state['planned_count']:
                states.append(state)
        if len(states) != args.count:
            raise ValueError('allocated count differs from budget')
        if declaration['constraints']['geometry'] == 'uniform_scale_translate':
            for state in states:
                source = foreground_map[state['parameters']['foreground_id']]
                with Image.open(source[3]) as foreground:
                    place_foreground(foreground, source[6], state['parameters']['placement'])
        if output.exists() and any(output.iterdir()):
            raise ValueError('output already contains samples')
        output.mkdir(parents=True, exist_ok=True)
        label_folder = 'labels' if output == split_dir('meter', 'train') else 'labels_candidate'
        for folder in ('images', label_folder, 'annotations'):
            (output / folder).mkdir()
        result = {'status': 'running', 'requested_count': args.count, 'completed_count': 0,
                  'source_batch': rel(manifest_path), 'seed': args.seed, 'resolution': [640, 480],
                  'production_plan': rel(plan_path),
                  'sampling_evidence': params['sampling_evidence'],
                  'samples': 'samples.jsonl', 'generation_method': 'composite',
                  'scope': ('Synthetic meter on real backgrounds; pose and box candidates only. '
                            'Background occlusion, reflection, existing meter instances, placement and ground contact '
                            'have not been verified; not an independent camera capture.')}
        manifest_output = output / 'manifest.json'
        def save():
            manifest_output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        save()
        try:
            for index in range(args.count):
                state = states[index]
                source_index, row, source_image, foreground, label, annotation, values = foreground_map[state['parameters']['foreground_id']]
                background = background_map[state['parameters']['background_image']]
                geometry = 'identity'
                with Image.open(background['image']) as bg, Image.open(foreground) as fg:
                    if declaration['constraints']['geometry'] == 'uniform_scale_translate':
                        fg, values, geometry = place_foreground(fg, values, state['parameters']['placement'])
                    composed, background_transform = compose(bg, fg)
                name = 'meter_real_composite_%05d' % index
                composed.save(output / 'images' / (name + '.png'))
                (output / label_folder / (name + '.txt')).write_text(
                    ' '.join(str(int(value)) if i == 0 or (i >= 7 and (i - 7) % 3 == 0)
                             else f'{value:.8f}' for i, value in enumerate(values)) + '\n', encoding='utf8')
                record = {'id': name, 'image': 'images/' + name + '.png',
                          'label_candidate': label_folder + '/' + name + '.txt', 'split': 'train' if label_folder == 'labels' else 'unassigned',
                          'background': rel(background['image']), 'background_parent': rel(background['parent']),
                          'background_evidence': {key: rel(background[key]) if key == 'manifest' else background[key]
                                                  for key in ('manifest', 'record_index', 'relation')},
                          'background_lineage_links': params['lineage_links'],
                          'background_meter_review': review_context.get(rel(background['image'])),
                          'sampling_seed': sampling_seed,
                          'sampling_group': sampling_groups.get((rel(background['image']), row['id'])),
                          'background_transform': background_transform,
                          'background_source_refs': [ref for ref in run.source_refs[rel(background['image'])]
                                                     if ref.get('manifest') == rel(background['manifest']) and
                                                     ref.get('record_index') == background['record_index'] and
                                                     ref.get('relation') == background['relation']],
                          'foreground': rel(foreground), 'foreground_source_image': rel(source_image),
                          'foreground_annotation': rel(annotation), 'foreground_label': rel(label),
                          'foreground_source_evidence': {'manifest': rel(records_path),
                                                         'record_index': source_index, 'relation': 'describes'},
                          'image_size': [640, 480], 'geometry_transform': geometry,
                          'visibility_policy': 'source v0 remains v0; all in-frame points v1 after background change',
                          'keypoint_readability': dict.fromkeys(POINTS, 'unknown'),
                          'reflection_status': 'unknown', 'physical_reading': 'unknown',
                          'status': 'candidate; visual scene and occlusion review required'}
                (output / 'annotations' / (name + '.json')).write_text(
                    json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
                with (output / 'samples.jsonl').open('a', encoding='utf8') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                run.record_sample(state['id'], output=output / 'images' / (name + '.png'),
                                  sources=state['sources'], parameters=state['parameters'],
                                  actual={'background': rel(background['image']),
                                          'background_parent': rel(background['parent']),
                                          'background_evidence': record['background_evidence'],
                                          'background_meter_review': record['background_meter_review'],
                                          'sampling_seed': record['sampling_seed'],
                                          'sampling_group': record['sampling_group'],
                                          'foreground': rel(foreground),
                                          'foreground_annotation': rel(annotation),
                                          'foreground_label': rel(label),
                                          'foreground_source_evidence': record['foreground_source_evidence'],
                                          'background_transform': background_transform,
                                          'geometry_transform': geometry,
                                          'keypoint_readability': record['keypoint_readability'],
                                          'reflection_status': record['reflection_status'],
                                          'physical_reading': record['physical_reading'],
                                          'visibility_policy': record['visibility_policy']})
                result['completed_count'] = index + 1
                if (index + 1) % 25 == 0:
                    save()
            result['status'] = 'complete_candidates'
        except BaseException as exc:
            result.update(status='failed_or_interrupted', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            save()
        run.set_result({'count': result['completed_count'], 'status': result['status']})
        print('METER_REAL_COMPOSITE_CANDIDATES', json.dumps(run.value, ensure_ascii=False))


if __name__ == '__main__':
    main()
