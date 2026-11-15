"""Create inspectable person/e-bike occlusion previews, not a training dataset."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.images import box, load_rgba, resized

HEIGHT, WIDTH = 768, 1024


def workspace_path(value):
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError('路径必须位于工作空间内') from exc
    return path


def mask_and_place(canvas, rgba, x, y):
    h, w = rgba.shape[:2]
    if x < 0 or y < 0 or x + w > WIDTH or y + h > HEIGHT:
        raise ValueError('前景放置超出画布')
    opacity = rgba[:, :, 3:4].astype(np.float32) / 255
    target = canvas[y:y + h, x:x + w]
    target[:] = (rgba[:, :, :3] * opacity + target * (1 - opacity)).astype(np.uint8)
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    mask[y:y + h, x:x + w] = (rgba[:, :, 3] > 127).astype(np.uint8)
    return mask


def write_image(path, image):
    ok, data = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f'无法编码图片：{path}')
    data.tofile(str(path))


PREVIEW_CONSTRAINTS = {'width': 1024, 'height': 768, 'person_vehicle_overlap': True}
PREVIEW_LABELS = {'format': 'visible_masks', 'person': 'person_visible', 'vehicle': 'vehicle'}


def validate_preview_plan(plan, available):
    top = {'schema_version', 'model_key', 'target_split', 'purpose', 'budget', 'uncertainty_weight',
           'independence_basis', 'constraints', 'strata'}
    fields = {'id', 'failure_event', 'min_count', 'max_count', 'prior', 'exposure_weight', 'failure_cost',
              'conditions', 'background', 'target', 'labels', 'augmentation', 'parameters', 'sources'}
    if not isinstance(plan, dict) or set(plan) != top or plan['model_key'] != 'e_bike' or plan['target_split'] != 'train':
        raise ValueError('unsupported declaration fields, model or split')
    if plan['constraints'] != PREVIEW_CONSTRAINTS or type(plan['constraints'].get('person_vehicle_overlap')) is not bool:
        raise ValueError('unsupported preview constraints')
    if type(plan['budget']) is not int or not 1 <= plan['budget'] <= 10000:
        raise ValueError('invalid budget')
    if not isinstance(plan['strata'], list) or not plan['strata']:
        raise ValueError('a nonempty state plan is required')
    seen = set()
    for state in plan['strata']:
        if not isinstance(state, dict) or set(state) != fields or state['conditions'] != []:
            raise ValueError('unsupported state fields or conditions')
        if type(state['max_count']) is not int or state['max_count'] > 1:
            raise ValueError('each exact state can generate at most one image')
        bg, target, params = state['background'], state['target'], state['parameters']
        if bg != {'kind': 'synthetic_flat'} or state['labels'] != PREVIEW_LABELS or state['augmentation'] != {'style': 'none'}:
            raise ValueError('unsupported background, labels or augmentation')
        if not isinstance(target, dict) or set(target) != {'person_class', 'person_source', 'vehicle_class', 'vehicle_source'}:
            raise ValueError('unsupported target fields')
        if target['person_class'] not in ('community', 'noncommunity') or target['vehicle_class'] not in ('upright', 'fallen'):
            raise ValueError('unsupported target classes')
        if target['person_source'] != available[target['person_class']] or target['vehicle_source'] != available[target['vehicle_class']]:
            raise ValueError('target sources differ from declared class inputs')
        if not isinstance(params, dict) or set(params) != {'person_height', 'bike_width', 'floor', 'person_x', 'bike_x'}:
            raise ValueError('unsupported layout parameters')
        if any(type(v) is not int for v in params.values()) or not (1 <= params['person_height'] <= HEIGHT and
                1 <= params['bike_width'] <= WIDTH and 1 <= params['floor'] <= HEIGHT and
                0 <= params['person_x'] < WIDTH and 0 <= params['bike_x'] < WIDTH):
            raise ValueError('invalid layout parameters')
        sources = sorted({target['person_source'], target['vehicle_source']})
        if len(sources) != 2 or state['sources'] != sources:
            raise ValueError('sources must be the sorted distinct person and vehicle paths')
        signature = json.dumps({k: state[k] for k in ('background', 'target', 'labels', 'augmentation', 'parameters')},
                               ensure_ascii=False, sort_keys=True, allow_nan=False)
        if signature in seen:
            raise ValueError('duplicate exact generation states are not allowed')
        seen.add(signature)


def render_planned_preview(state):
    """Render one explicit state after the planned production guard has started."""
    target, params = state['target'], state['parameters']
    person = resized(load_rgba(workspace_path(target['person_source'])), height=params['person_height'])
    bike = resized(load_rgba(workspace_path(target['vehicle_source'])), width=params['bike_width'])
    canvas = np.full((HEIGHT, WIDTH, 3), (208, 211, 211), np.uint8)
    canvas[580:] = (150, 151, 151)
    full = mask_and_place(canvas, person, params['person_x'], params['floor'] - person.shape[0])
    vehicle = mask_and_place(canvas, bike, params['bike_x'], params['floor'] - bike.shape[0])
    visible = np.logical_and(full, np.logical_not(vehicle)).astype(np.uint8)
    if not visible.sum() or visible.sum() == full.sum():
        raise ValueError('requested visible person/vehicle occlusion is infeasible')
    actual = {'person_class': target['person_class'], 'vehicle_class': target['vehicle_class'],
              'person_source': target['person_source'], 'vehicle_source': target['vehicle_source'],
              'parameters': params, 'person_full_bbox_xyxy': box(full),
              'person_visible_bbox_xyxy': box(visible), 'vehicle_bbox_xyxy': box(vehicle),
              'person_visible_fraction': int(visible.sum()) / int(full.sum())}
    return canvas, full, visible, vehicle, actual


def preview(community, noncommunity, upright, fallen, output, *, production_plan=None):
    if production_plan is None:
        raise ValueError('an explicit production plan is required')
    sys.path.insert(0, str(ROOT))
    from lab.training_common.production_guard import generation_guard
    paths = [workspace_path(x) for x in (community, noncommunity, upright, fallen)]
    available = {name: path.relative_to(ROOT).as_posix() for name, path in
                 zip(('community', 'noncommunity', 'upright', 'fallen'), paths)}
    plan_path, output = workspace_path(production_plan), workspace_path(output)
    parents = (ROOT / 'datasets/annotated/e_bike', ROOT / 'datasets/exports/research_samples/e_bike')
    if not any(output.is_relative_to(parent) and output != parent for parent in parents):
        raise ValueError('preview output must be a new e-bike candidate or research-sample batch')
    declaration = json.loads(plan_path.read_text(encoding='utf-8-sig'))
    validate_preview_plan(declaration, available)
    operation = 'e_bike.preview_person_occlusions:' + output.relative_to(ROOT).as_posix()
    with generation_guard(operation, paths, [output], {'plan': plan_path.relative_to(ROOT).as_posix()},
                          procedural_inputs=[plan_path], code_paths=[Path(__file__), ROOT / 'datasets/scripts/common/images.py'], root=ROOT,
                          production_plan=declaration, sample_output_paths=[output / 'images'],
                          auxiliary_output_paths=[output / 'masks']) as run:
        if run.reused:
            return json.loads((output / 'manifest.json').read_text(encoding='utf-8'))['samples']
        if output.exists():
            raise ValueError('output exists; choose a new batch')
        states = run.plan['strata']
        if any(type(s.get('planned_count')) is not int or s['planned_count'] not in (0, 1) for s in states):
            raise ValueError('allocation exceeds one image per exact state')
        for state in states:
            render_planned_preview(state)
        (output / 'images').mkdir(parents=True)
        (output / 'masks').mkdir(parents=True)
        records = []
        for index, state in enumerate(s for s in states if s['planned_count']):
            canvas, full, visible, vehicle, actual = render_planned_preview(state)
            stem = f'preview_{index+1:06d}'
            image = output / 'images' / (stem + '.png')
            write_image(image, canvas)
            paths_out = {}
            for name, mask in [('person_full', full), ('person_visible', visible), ('vehicle', vehicle)]:
                path = output / 'masks' / f'{stem}_{name}.png'
                write_image(path, mask * 255)
                paths_out[name] = path.relative_to(ROOT).as_posix()
            run.record_sample(state['id'], output=image, sources=state['sources'],
                              parameters=state['parameters'], actual=actual)
            records.append({'image': image.relative_to(ROOT).as_posix(), 'state_id': state['id'],
                            'masks': paths_out, 'actual': actual, 'status': 'candidate_not_ground_truth'})
        manifest = {'status': 'complete_preview', 'purpose': 'occlusion candidates, not a training dataset',
                    'background': 'programmatically generated, not a real scene',
                    'limitation': 'vehicle masks are candidates; their quality is not established by generation',
                    'split': None, 'plan': plan_path.relative_to(ROOT).as_posix(), 'samples': records}
        (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return records


def _preview(community, noncommunity, upright, fallen, output):
    raise ValueError('unplanned generation is unavailable; use preview with a production plan')


def main():
    parser = argparse.ArgumentParser(description='人物和电动车遮挡的少量候选预览')
    parser.add_argument('--community', required=True)
    parser.add_argument('--noncommunity', required=True)
    parser.add_argument('--upright', required=True)
    parser.add_argument('--fallen', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        sources = [workspace_path(p) for p in (args.community, args.noncommunity, args.upright, args.fallen)]
        result = preview(*sources, workspace_path(args.output), production_plan=args.plan)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f'失败：{exc}\n')
    print(f'生成 {len(result)} 张遮挡预览及对应掩码；不是训练集')


if __name__ == '__main__':
    main()
