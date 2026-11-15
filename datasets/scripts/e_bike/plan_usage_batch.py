"""Plan e-bike training states from task/split/role use relations; no images are generated."""

if __name__ == "__main__":
    raise SystemExit("Task helper is import-only; use the primary entry in datasets/scripts/e_bike/. Production remains paused.")

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
from datasets.scripts.e_bike import compose_v4_grounded_train as ground
from datasets.scripts.e_bike import prepare_v4_low_angle_real as real
from lab.scripts.data.v4_source_gate import build_queue  # noqa: E402
from datasets.scripts.common.bayesian import allocate  # noqa: E402
from datasets.scripts.common.perspective import planar_warp  # noqa: E402

USAGE = ROOT / 'datasets/annotated/e_bike/v4_complex_backgrounds_20261112/image_task_use_manifest_20261114.json'
DESIGN = ROOT / 'datasets/annotated/e_bike/v4_car_sequence_review_20261113/manifest.json'
REVIEW = real.REVIEW / 'review_manifest.json'
CLASSES = ((), ('upright',), ('fallen',), ('upright', 'upright'),
           ('fallen', 'fallen'), ('upright', 'fallen'))


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def relation(record, role, status):
    return [r for r in record['task_use_status'] if r.get('task') == 'e_bike'
            and r.get('split') == 'train' and r.get('role') == role and r.get('status') in status]


def state_base(index, event):
    return {'id': f'state_{index:06d}', 'failure_event': event, 'min_count': 1,
            'max_count': 1, 'prior': {'alpha': 1, 'beta': 1}, 'exposure_weight': 1,
            'failure_cost': 1, 'conditions': []}


def header(states, constraints, purpose):
    return {'schema_version': 1, 'model_key': 'e_bike', 'target_split': 'train',
            'purpose': purpose, 'budget': len(states), 'uncertainty_weight': 0,
            'independence_basis': 'Parents and derived variants may recur; none are additional independent captures.',
            'constraints': constraints, 'strata': states}


def eligible_sources(usage, backgrounds, entries, queue):
    # Only this task/split/role relationship confers readiness. A reviewed candidate
    # background is admitted by its explicit full-frame negative review and the
    # user's current authorization to use these already reviewed alternatives.
    bg_by_path = {b['image']: b for b in backgrounds}
    selected_bg, selected_fg = [], {'upright': [], 'fallen': []}
    for record in usage['records']:
        bg_rel = relation(record, 'background', {'ready', 'candidate'})
        if bg_rel and record['image'] in bg_by_path:
            bg = bg_by_path[record['image']]
            review = bg.get('v4_input_review_20261114', {})
            if bg['task_state'] != 'candidate_complete_negative' or not review:
                continue
            if '整幅裁切图目视未见电动车' not in bg_rel[0].get('label_status', ''):
                continue
            if any(not (ROOT / bg[k]).is_file() for k in ('image', 'parent_image', 'source_video')):
                continue
            if ground.source_role(ROOT, ROOT / bg['parent_image'], queue)[0] == 'regression':
                continue
            selected_bg.append(bg)
        fg_rel = relation(record, 'target_foreground', {'ready'})
        if not fg_rel:
            continue
        for item in entries:
            if item['path'] != record['image'] or item['class'] != fg_rel[0]['posture']:
                continue
            if any(not (ROOT / item[k]).is_file() for k in ('path', 'source_frame', 'source_video')):
                continue
            if any(ground.source_role(ROOT, ROOT / item[k], queue)[0] == 'regression'
                   for k in ('source_frame', 'source_video')):
                continue
            selected_fg[item['class']].append(item)
    return selected_bg, selected_fg


def make_plans(seed):
    rng = random.Random(seed)
    usage, design = load(USAGE), load(DESIGN)['v4_combination_source_plan_20261114']
    backgrounds = load(ground.BACKGROUND)['backgrounds']
    entries = load(ground.bike.SELECTION)['vehicles']
    queue = build_queue(ROOT)
    bgs, foregrounds = eligible_sources(usage, backgrounds, entries, queue)
    if not bgs or any(not foregrounds[k] for k in foregrounds):
        raise ValueError('reviewed eligible background and both ready posture sources are required')
    # Transform foreground alpha only, with the same operations as the production
    # entry. No background is composited and no candidate image is produced.
    sprites = {}
    for cls, items in foregrounds.items():
        for item in items:
            src = ground.bike.scene.assets.load_rgba(ROOT / item['path'])
            warped = planar_warp(src, 0, 0, 0)
            for size in (65, 80, 95):
                sprite = ground.bike.scene.assets.resized(warped, **({'height': size} if cls == 'upright' else {'width': size}))
                if int((sprite[:, :, 3] > 127).sum()) < 120:
                    raise ValueError('foreground too small')
                sprites[(item['path'], size)] = sprite
    states, counts = [], {}
    for branch, classes in zip(design['branches'], CLASSES):
        # User authorized reallocating unavailable parent groups to the ready
        # group. Negative states still require distinct background/tone states.
        goal = branch['real_background_composite_images']
        created, seen = 0, set()
        # For empty images only actual background/tone combinations differ.
        attempts = 0
        while created < goal and attempts < max(2000, goal * 200):
            attempts += 1
            bg = rng.choice(bgs)
            tone = rng.choice(tuple(ground.TONES))
            gain, sat = ground.TONES[tone]
            actors, targets, boxes = [], [], []
            for i, cls in enumerate(classes):
                item = rng.choice(foregrounds[cls])
                size = rng.choice((65, 80, 95))
                sprite = sprites[(item['path'], size)]
                # Wide separation guarantees no target obscures another target.
                if bg['id'] == 'frame_005751':
                    x = rng.randrange(65, 95) if i == 0 else rng.randrange(205, 225)
                    y = rng.randrange(425, 454)
                else:
                    x = rng.randrange(85, 190) if i == 0 else rng.randrange(350, 545)
                    y = rng.randrange(390, 455)
                actor = {'foot_x': x, 'foot_y': y, 'size_px': size, 'yaw': 0, 'pitch': 0,
                         'roll': 0, 'relative_exposure': rng.choice((.95, 1., 1.05))}
                h, w = sprite.shape[:2]
                left, top = round(x - w / 2), round(y - h)
                if left < 0 or top < 0 or left + w > 640 or top + h > 480:
                    break
                placement = dict(actor, sprite=sprite)
                try:
                    ground.bike.scene.validate_actor_placement(placement, 'ground_only')
                except ValueError:
                    break
                if not ground.grounded(placement, ground.ground_zone(bg)):
                    break
                box = (left, top, left+w, top+h)
                if any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in boxes):
                    break
                boxes.append(box)
                actors.append(actor)
                targets.append({'class': cls, 'source': item['path']})
            if len(actors) != len(classes):
                continue
            signature = json.dumps([bg['id'], targets, actors, tone], sort_keys=True)
            if signature in seen:
                continue
            seen.add(signature)
            used = {bg[k] for k in ('image', 'parent_image', 'source_video')}
            for target in targets:
                item = next(x for x in foregrounds[target['class']] if x['path'] == target['source'])
                used.update(item[k] for k in ('path', 'source_frame', 'source_video', 'source_image') if k in item)
            state = state_base(len(states) + 1, branch['combination_id'])
            state.update({'background': {'id': bg['id'], 'image': bg['image']},
                          'target': {'vehicles': targets}, 'labels': ground.GROUND_LABELS,
                          'augmentation': {'tone': tone},
                          'parameters': {'actors': actors, 'value_gain': gain, 'saturation_gain': sat},
                          'sources': sorted(used)})
            states.append(state)
            created += 1
        counts[branch['combination_id']] = created
    plan = header(states, ground.GROUND_CONSTRAINTS, 'Seeded reviewed-background e-bike combinations; exact states only.')
    # Validate against the real entry, without calling the producer or guard.
    _, loaded_vehicles = ground.bike.sources(include_people=False)
    # The entry checks the same parent once per state. Cache that identical
    # read-only query during validation to avoid rescanning exposure history.
    original_role, cached_roles = ground.source_role, {}
    def cached_role(root, source, known):
        if source not in cached_roles:
            cached_roles[source] = original_role(root, source, known)
        return cached_roles[source]
    ground.source_role = cached_role
    try:
        ground.validate_ground_plan(plan, backgrounds, loaded_vehicles, queue)
    finally:
        ground.source_role = original_role
    allocate(plan, ROOT)
    frame_states = []
    rows = load(REVIEW)['rows']
    use_rows = {r['image']: r for r in usage['records'] if relation(r, 'direct_frame', {'candidate', 'ready'})}
    for row in rows:
        if row['image'] not in use_rows or 'complete_candidate' not in row['label_status']:
            continue
        for tone, (gain, sat) in real.TONES.items():
            state = state_base(len(frame_states) + 1, 'EB-C5-mixed-multiple')
            state.update({'background': {'image': row['image'], 'crop_xyxy': row['crop_xyxy']},
                          'target': {'classes': [b['class'] for b in row['boxes']]}, 'labels': real.LABELS,
                          'augmentation': {'tone': tone},
                          'parameters': {'value_gain': gain, 'saturation_gain': sat},
                          'sources': sorted({row['image'], row['source_frame'], row['source_video']})})
            real.check_state(state, rows, queue)
            frame_states.append(state)
    direct_plan = header(frame_states, real.CONSTRAINTS, 'Reviewed candidate video crops and tone derivatives.')
    real.validate_declaration(direct_plan)
    allocate(direct_plan, ROOT)
    branches = []
    for branch in design['branches']:
        key = branch['combination_id']
        # The four unchanged-tone video crops have pre-existing candidate boxes.
        # Tone variants are listed separately as derivatives, not new captures.
        direct = sum(1 for state in frame_states if key == 'EB-C5-mixed-multiple'
                     and state['augmentation']['tone'] == 'original'
                     and set(state['target']['classes']) == {'upright', 'fallen'})
        direct_goal = branch['direct_images']
        composite_goal = branch['real_background_composite_images']
        blender_goal = branch['blender_background_images']
        branches.append({'combination_id': key, 'direct_candidate_frames': direct,
                         'direct_target': direct_goal, 'direct_shortfall': direct_goal-direct,
                         'real_composite_states': counts[key], 'real_composite_target': composite_goal,
                         'real_composite_shortfall': composite_goal-counts[key],
                         'original_real_parent_quotas': branch['real_background_group_quota'],
                         'planned_real_parent_counts': {'G_full_scene': counts[key], 'G_car_capture': 0, 'G_legacy_scene': 0},
                         'blender_states': 0, 'blender_target': blender_goal,
                         'blender_shortfall': blender_goal})
    report = {'seed': seed, 'backgrounds': [b['image'] for b in bgs],
              'foregrounds': {k: [i['path'] for i in v] for k, v in foregrounds.items()},
              'source_history': queue.get('exposure_status', 'unknown'),
              'counts': counts, 'real_frame_states': len(frame_states), 'branches': branches,
              'direct_frame_tone_derivatives_excluding_original': sum(
                  state['augmentation']['tone'] != 'original' for state in frame_states),
              'limitations': ['No output generated and production guard not run.',
                  'Four background frames share one video; 12 empty states are four frames with three tones.',
                  'Unavailable real-background parent-group shares were reassigned to G_full_scene; the other groups lack complete e-bike labels.',
                  'Blender entry is a fixed pilot, includes a blocked foreground and cannot consume a batch plan.',
                  'Four direct-frame candidates have AI-labelled boxes, not human verification; all are crops from one video.',
                  'Eight additional frame tone states are derivatives and do not count as new direct captures.',
                  'Source exposure history is incomplete; unknown is not an independently cleared history.',
                  'Foreground contact, in-frame and alpha occupancy were checked without background composition; generated labels and images remain unverified.']}
    return {'composite': plan, 'real_frames': direct_plan, 'report': report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20261114)
    parser.add_argument('--stdout', action='store_true', help='Print plans as JSON; do not write files')
    parser.add_argument('--output-dir', type=Path, help='New directory under e-bike authoritative plans')
    args = parser.parse_args()
    result = make_plans(args.seed)
    if args.stdout:
        print(json.dumps(result, ensure_ascii=False))
        return
    if args.output_dir is None:
        parser.error('supply --stdout or --output-dir')
    out = (ROOT / args.output_dir).resolve()
    owner = (ROOT / 'datasets/scripts/e_bike/config/plans').resolve()
    if not out.is_relative_to(owner) or out == owner:
        parser.error('output directory must be under e-bike authoritative plans')
    out.mkdir(parents=True, exist_ok=False)
    for name, payload in result.items():
        (out / (name + '.json')).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output_dir': out.relative_to(ROOT).as_posix(), **result['report']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
