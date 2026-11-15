"""Overlay generated ink onto visible name-paper masks from arena renders."""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/fire/. Production remains paused.")

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT / 'lab/features/fire/training/scripts'), str(ROOT / 'lab/training_common')]
from camera_tone import read_image, write_image  # noqa: E402
from compose_v4_tracked_facade_papers import FONT, paper_graphic  # noqa: E402
from vary_name_papers import make_name  # noqa: E402
from fire_blender_labels import decode_ids, srgb_to_linear, visible_bbox, yolo_row  # noqa: E402
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard, plan_production  # noqa: E402


def rel(path):
    return path.relative_to(ROOT).as_posix()


def quad_from_mask(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 1:
        raise ValueError('paper mask has multiple or missing components')
    hull = cv2.convexHull(contours[0])
    polygon = cv2.approxPolyDP(hull, .02 * cv2.arcLength(hull, True), True).reshape(-1, 2)
    if len(polygon) != 4:
        polygon = cv2.boxPoints(cv2.minAreaRect(hull))
    pts = np.asarray(polygon, np.float32)
    sums, diffs = pts.sum(axis=1), pts[:, 1] - pts[:, 0]
    corners = np.array([pts[np.argmin(sums)], pts[np.argmin(diffs)],
                        pts[np.argmax(sums)], pts[np.argmax(diffs)]], np.float32)
    if len(set(map(tuple, corners))) != 4 or abs(cv2.contourArea(corners)) < 200:
        raise ValueError('paper quadrilateral is degenerate')
    return corners


def _candidate_specs(sources, variants):
    result = []
    for source_index, source in enumerate(sources):
        run = json.loads((source / 'run.json').read_text(encoding='utf8'))
        if run.get('status') != 'complete_candidates':
            raise ValueError('source render batch incomplete')
        purpose = run.get('purpose', '')
        if not isinstance(purpose, str) or 'not_training' in purpose.casefold():
            raise ValueError('source batch is marked as not intended for training')
        manifest = source / 'manifest.jsonl'
        for row_index, line in enumerate(manifest.read_text(encoding='utf8').splitlines()):
            row = json.loads(line)
            if row.get('split') != 'train':
                raise ValueError('source render record is not assigned to train')
            signs = [x for x in row['instances'] if x['kind'] == 'name_sign']
            if (row['occluder'] or len(signs) != 1 or signs[0]['state'] != 'visible' or
                    signs[0]['truncated'] or not signs[0].get('text')):
                continue
            sign = signs[0]
            if not isinstance(sign.get('building'), str) or not sign['building'].strip():
                raise ValueError('visible name-paper instance requires a building identifier')
            tone = row.get('tone')
            if tone is not None and (not isinstance(tone, dict) or
                    type(tone.get('exposure_ev')) not in (int, float) or
                    not np.isfinite(tone['exposure_ev'])):
                raise ValueError('source tone must provide a finite exposure_ev when present')
            frame = int(Path(row['image']).stem.split('_')[-1])
            image = (source / row['image']).resolve()
            mask = (source / row['instance_pass']).resolve()
            if not image.is_file() or not mask.is_file() or not image.is_relative_to(source.resolve()) or not mask.is_relative_to(source.resolve()):
                raise ValueError('eligible source image or instance mask is missing or outside its batch')
            for variant in range(max(1, variants)):
                name = (make_name(10000 + source_index * 1000 + (frame // 2) * 2 + variant)
                        if variants else sign['text'])
                stem = (f's{source_index:02d}_r{row_index:06d}_' + Path(row['image']).stem +
                        (f'_v{variant:02d}' if variants else ''))
                state_id = f'arena_name_{source_index:02d}_{row_index:06d}_v{variant:02d}'
                result.append({'state_id': state_id, 'source_index': source_index,
                    'source': source, 'row': row, 'sign': sign, 'frame': frame, 'variant': variant,
                    'name': name, 'stem': stem, 'image': image, 'mask': mask,
                    'manifest': manifest, 'building': sign['building'], 'tone': tone,
                    'source_assets': row.get('source_assets', [run['source_scene']] if run.get('source_scene') else [])})
    if not result:
        raise ValueError('no eligible arena name-paper records')
    return result


def _production_declaration(path, candidates, variants):
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    if path.is_symlink():
        raise ValueError('production plan symlinks are not supported')
    path = path.resolve()
    if not path.is_relative_to(ROOT) or not path.is_file() or path.suffix.lower() != '.json':
        raise ValueError('production plan must be an existing workspace JSON file')
    declaration = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(declaration, dict):
        raise ValueError('production declaration must be a JSON object')
    if set(declaration) != {'schema_version', 'model_key', 'target_split', 'purpose', 'budget',
                            'uncertainty_weight', 'independence_basis', 'constraints', 'strata'}:
        raise ValueError('production declaration has missing or unsupported top-level fields')
    constraints = {'route_context': 'building_facade',
                   'traffic_signal_cooccurrence': 'forbidden_separate_route',
                   'traffic_signal_generation': 'no_traffic_signal_injection',
                   'ocr_truth': 'not_verified'}
    if (declaration.get('model_key') != 'building_name_paper' or
            declaration.get('target_split') != 'train' or
            declaration.get('constraints') != constraints):
        raise ValueError('unsupported building-name model, split, budget or route constraints')
    by_id = {item['state_id']: item for item in candidates}
    if len(by_id) != len(candidates):
        raise ValueError('duplicate executable candidate state')
    states = declaration.get('strata')
    if not isinstance(states, list) or not states:
        raise ValueError('production plan must contain feasible arena name states')
    for state in states:
        if not isinstance(state, dict) or set(state) != {
                'id', 'failure_event', 'min_count', 'max_count', 'prior', 'exposure_weight',
                'failure_cost', 'conditions', 'background', 'target', 'labels', 'augmentation',
                'sources', 'parameters'}:
            raise ValueError('production state has missing or unsupported fields')
        if not isinstance(state.get('id'), str):
            raise ValueError('production state id must be a string')
        item = by_id.get(state['id'])
        if item is None:
            raise ValueError('production plan references an unsupported or unavailable candidate state')
        sources = ['blender_arena_name_paper', 'blender_instance_mask', 'name_font_asset']
        expected = {
            'failure_event': 'name_paper_visibility_or_ink_projection_uncertainty',
            'conditions': [f"source:{rel(item['image'])}", f"variant:{item['variant']}",
                           'traffic_signal_injection:none'],
            'background': {'domain': 'synthetic_blender_building_arena',
                           'source_image': rel(item['image']), 'traffic_signal_injection': 'none'},
            'target': {'kind': 'building_name_paper', 'building': item['sign']['building'],
                       'text': item['name'], 'paper_bbox_xyxy': item['sign']['pixel_bbox_xyxy'],
                       'visible_pixels': item['sign']['visible_pixels']},
            'labels': {'class_id': 0, 'format': 'yolo_v8', 'task': 'paper_detection',
                       'ocr_status': 'not_verified'},
            'augmentation': {'method': 'fitted_perspective_ink_projection',
                             'geometry': 'verified_instance_mask_quad', 'alpha': 0.88},
            'sources': sources,
            'parameters': {'source_image': rel(item['image']), 'source_mask': rel(item['mask']),
                           'source_manifest': rel(item['manifest']), 'variant_index': item['variant'],
                           'variant_count': variants},
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError('production state differs from the actual source, label or ink-composition behavior')
        if type(state.get('max_count')) is not int or state['max_count'] > 1:
            raise ValueError('each arena source/variant state produces at most one distinct image')
    if any(state['id'] not in by_id for state in states):
        raise ValueError('production plan contains a state outside the executable candidate set')
    return path, declaration


def _generate(out, candidates, run):
    if not out.is_relative_to((ROOT / 'datasets/exports/fire').resolve()) or out.exists():
        raise ValueError('Use a new task-owned fire export directory')
    out.mkdir(parents=True)
    records, rejected, counts = [], [], Counter()
    try:
        planned_states = run.plan['strata']
        by_id = {item['state_id']: item for item in candidates}
        if sum(state['planned_count'] for state in planned_states) != run.plan['budget']:
            raise ValueError('planned name-paper total differs from declared budget')
        for state in planned_states:
            if state['planned_count'] == 0:
                continue
            item = by_id[state['id']]
            source, r, sign = item['source'], item['row'], item['sign']
            img = read_image(item['image'])
            mask_png = read_image(item['mask'])
            if img.shape != (480, 640, 3) or mask_png.shape != img.shape:
                raise ValueError('unexpected dimensions')
            rgb = cv2.cvtColor(mask_png, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            ids = decode_ids(srgb_to_linear(rgb), [x['id'] for x in r['instances']])
            mask = ids == sign['id']
            box, pixels = visible_bbox(mask)
            if box != sign['pixel_bbox_xyxy'] or pixels != sign['visible_pixels']:
                rejected.append({'source': rel(item['image']), 'reason': 'mask_record_mismatch'})
                continue
            if box[2] - box[0] < 35 or box[3] - box[1] < 12:
                rejected.append({'source': rel(item['image']), 'reason': 'paper_too_small_for_ink'})
                continue
            corners = quad_from_mask(mask)
            transform = cv2.getPerspectiveTransform(np.float32([[0, 0], [119, 0], [119, 35], [0, 35]]), corners)
            graphic = paper_graphic(item['name'])
            gray = cv2.cvtColor(graphic, cv2.COLOR_BGR2GRAY)
            ink = np.clip((190.0 - gray.astype(np.float32)) / 150.0, 0, 1)
            alpha = cv2.warpPerspective(ink, transform, (640, 480), flags=cv2.INTER_LINEAR)
            alpha *= mask.astype(np.float32)
            if np.count_nonzero(alpha > .3) < 8:
                rejected.append({'source': rel(item['image']), 'reason': 'insufficient_visible_ink', 'name': item['name']})
                continue
            composed = np.clip(img.astype(np.float32) * (1 - .88 * alpha[:, :, None]), 0, 255).astype(np.uint8)
            image_path = out / 'images/train' / (item['stem'] + '.png')
            label_path = out / 'labels/train' / (item['stem'] + '.txt')
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            write_image(image_path, composed)
            label_path.write_text(yolo_row(box, 640, 480) + '\n', encoding='utf8')
            records.append({'id': item['stem'], 'image': rel(image_path), 'label': rel(label_path),
                'source_image': rel(item['image']), 'source_instance_pass': rel(item['mask']),
                'source_manifest': rel(item['manifest']), 'source_assets': item['source_assets'],
                'split': 'train', 'generation_method': 'fitted_arena_render_plus_2d_ink_projection',
                'name_requested': item['name'], 'ocr_status': 'not_verified', 'paper_box_xyxy': box,
                'paper_quad_xy': corners.tolist(), 'paper_visible_pixels': pixels,
                'tone': item['tone'], 'tone_status': 'recorded' if item['tone'] is not None else 'unknown',
                'building': item['building'], 'parent_occluder': r['occluder'],
                'review_status': 'candidate_not_human_accepted', 'shared_parent_pair': f"{source.name}:{item['frame'] // 2}",
                'production_state_id': state['id']})
            run.record_sample(state['id'], output=image_path, sources=state['sources'],
                parameters=state['parameters'], actual={'source_image': rel(item['image']),
                    'source_instance_pass': rel(item['mask']), 'source_manifest': rel(item['manifest']),
                    'variant_index': item['variant'], 'name_requested': item['name'],
                    'paper_bbox_xyxy': box, 'paper_quad_xy': corners.tolist(),
                    'paper_visible_pixels': pixels, 'tone': item['tone'], 'building': item['building'],
                    'alpha': 0.88, 'visible_ink_pixels_over_threshold': int(np.count_nonzero(alpha > .3)),
                    'traffic_signal_added': False})
            counts['tone_unknown' if item['tone'] is None else
                   'negative_exposure' if item['tone']['exposure_ev'] < 0 else 'nonnegative_exposure'] += 1
        result = {'status': 'train_candidate_not_frozen', 'scope': 'Existing fitted arena renders with generated 2D ink projected on visible paper; no new camera capture, no accepted OCR truth. Generated labels require semantic and source review.',
                  'counts': dict(counts), 'rejected': rejected, 'records': records}
        (out / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        print(json.dumps({'generated': len(records), 'counts': dict(counts), 'rejected': len(rejected)}, ensure_ascii=False))
    except BaseException as exc:
        (out / 'FAILED.txt').write_text(f'{type(exc).__name__}: {exc}\n', encoding='utf8')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--production-plan', required=True,
                        help='workspace JSON declaration with explicit state priors, sources and constraints')
    parser.add_argument('--source', action='append', required=True,
                        help='existing workspace fire export batch containing run.json and manifest.jsonl; repeat as needed')
    parser.add_argument('--variants', type=int, default=0, choices=range(0, 9))
    args = parser.parse_args()
    out = (ROOT / args.output).resolve()
    sources = [(ROOT / value).resolve() for value in args.source]
    source_roots = ((ROOT / 'datasets/exports/fire').resolve(),
                    (ROOT / 'datasets/exports/authoritative/fire_sticker').resolve())
    if (len(set(sources)) != len(sources) or any(
            not any(source.is_relative_to(root) for root in source_roots) or not source.is_dir()
            for source in sources)):
        raise ValueError('sources must be distinct existing fire export batches')
    candidates = _candidate_specs(sources, args.variants)
    plan_path, declaration = _production_declaration(args.production_plan, candidates, args.variants)
    planned = plan_production(declaration, ROOT)
    if planned['planned_total'] != declaration['budget']:
        raise ValueError('planned name-paper total differs from declared budget')
    metadata, media = [FONT], []
    for source in sources:
        metadata.extend([source / 'run.json', source / 'manifest.jsonl'])
    media.extend(path for item in candidates for path in (item['image'], item['mask']))
    metadata.append(plan_path)
    code = [Path(__file__), ROOT / 'lab/features/fire/training/scripts/vary_name_papers.py',
            ROOT / 'lab/features/fire/training/scripts/compose_scene_fire.py', ROOT / 'lab/features/fire/training/scripts/compose_scene_fire_eval.py',
            ROOT / 'lab/features/fire/training/scripts/compose_v4_tracked_facade_papers.py',
            ROOT / 'lab/features/fire/training/scripts/fire_blender_labels.py', ROOT / 'lab/training_common/camera_tone.py',
            ROOT / 'datasets/scripts/common/images.py', ROOT / 'datasets/scripts/common/annotations.py',
            ROOT / 'datasets/scripts/common/names.py', ROOT / 'datasets/scripts/common/paper.py',
            ROOT / 'datasets/scripts/common/perspective.py']
    with generation_guard('fire.compose_arena_name:' + str(out.relative_to(ROOT)),
                          list(dict.fromkeys([*metadata, *media])), [out], {'variants': args.variants},
                          code_paths=[*code, ROOT / 'lab/training_common/bayesian_allocation.py'],
                          procedural_inputs=metadata, root=ROOT, production_plan=declaration,
                          sample_output_paths=[out / 'images/train']) as run:
        if run.plan != planned:
            raise ValueError('guard plan differs from the pre-generation feasibility plan')
        if not run.reused:
            _generate(out, candidates, run)
            manifest = out / 'manifest.json'
            result = json.loads(manifest.read_text(encoding='utf8'))
            result['batch_source_refs'] = run.source_refs
            manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
