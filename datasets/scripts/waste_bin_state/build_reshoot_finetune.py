"""Build the current bin fine-tuning set from reviewed photos and Train-only replay."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.frame_environment import query_frames
from datasets.scripts.common.review.placement_record import decode_placement_record
from datasets.scripts.waste_bin_state import build_full_dataset as full
from datasets.scripts.waste_bin_state import generate_waste_v4_grouped as compose
from lab.training_common.production_guard import generation_guard

DEFAULT_PLAN = 'datasets/scripts/waste_bin_state/config/reshoot_finetune_plan.json'


def path(value):
    result = (ROOT / value).resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError('Input or output escapes workspace: ' + str(value))
    return result


def read(value):
    return json.loads(path(value).read_text(encoding='utf-8-sig'))


def sha(file):
    with file.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def choose_replay(records, total=128, negative_count=16, seed=42):
    """Keep every direct Train positive, then round-robin actual composition strata."""
    rng = random.Random(seed)
    if any(row['split'] != 'train' for row in records):
        raise ValueError('Replay selection accepts only Train records')
    direct = [r for r in records if r['source_kind'] == 'direct_real']
    negatives = [r for r in records if not r['bins']]
    negatives.sort(key=lambda r: (r['source_kind'], r['source_record_id']))
    rng.shuffle(negatives)
    # Keep the scarce observed real empty scenes before filling with rendered
    # negatives. A larger negative quota must not silently discard real context.
    negatives.sort(key=lambda r: r['source_kind'] == 'blender_empty_negative')
    if len(negatives) < negative_count or len(direct) + negative_count > total:
        raise ValueError('Insufficient compatible replay quota')
    chosen = direct + negatives[:negative_count]
    buckets = defaultdict(list)
    for row in records:
        if row in direct or not row['bins']:
            continue
        key = (row['source_kind'], len(row['bins']),
               tuple(sorted((b['color'], b['state']) for b in row['bins'])))
        buckets[key].append(row)
    for group in buckets.values():
        rng.shuffle(group)
    keys = sorted(buckets)
    rng.shuffle(keys)
    while len(chosen) < total:
        previous = len(chosen)
        for key in keys:
            if buckets[key]:
                chosen.append(buckets[key].pop())
                if len(chosen) == total:
                    break
        if len(chosen) == previous:
            raise ValueError('Insufficient distinct Train positives for replay')
    return chosen


def composition_schedule(annotations, heights, per_color_state=48):
    """Balance actual color-state strata and sizes, fairly reusing eligible photos."""
    if not heights or per_color_state % len(heights):
        raise ValueError('Each color-state quota must divide evenly over requested heights')
    schedule = []
    for color in sorted({r['color'] for r in annotations}):
        for state in ('closed', 'open'):
            sources = [r for r in annotations if r['color'] == color and r['state'] == state
                       and r.get('composite_eligible', True)]
            if not sources:
                raise ValueError(f'No eligible photos for {color}:{state}')
            base, remainder = divmod(per_color_state, len(sources))
            # A contiguous fair quota gives every reused view all sizes, avoiding
            # source-index/size correlation when their cycle lengths share factors.
            i = 0
            for source_index, source in enumerate(sources):
                for _ in range(base + (source_index < remainder)):
                    schedule.append((source, heights[i % len(heights)], i))
                    i += 1
    return schedule


def baseline_snapshot(plan, destination):
    """Store source paths and label text before the current export is replaced."""
    snapshot = destination / 'baseline_manifest.json'
    if snapshot.is_file():
        return json.loads(snapshot.read_text(encoding='utf8'))
    base = path(plan.get('baseline_export', 'datasets/exports/waste_bin_state'))
    report = {'task': full.TASK, 'origin': full.relative(base), 'splits': {}}
    for split in full.SPLIT_ORDER:
        report['splits'][split] = []
        for row in json.loads((base / split / 'manifest.json').read_text(encoding='utf8'))['records']:
            source = path(row['source_image'])
            if not source.is_file() or sha(source) != row['image_sha256']:
                raise ValueError('Baseline source no longer matches exported pixels: ' + str(source))
            report['splits'][split].append({
                'source': full.relative(source),
                'label_text': (base / split / row['label']).read_text(encoding='utf8'),
                'label_sha256': sha(base / split / row['label']),
                'record': row,
            })
    full.write_json(snapshot, report)
    return report


def admitted_backgrounds(plan, train_rows):
    """Use prior Train empty pixels with existing complete ground observations."""
    _, source_config = full.load_task_config(full.TASK)
    database = path(source_config['databases']['frame_environment'])
    qualifications = query_frames(task=full.TASK, role='composite_background', split='train', database=database)
    by_hash = {row['content_sha256']: row for row in qualifications
               if row.get('label', {}).get('presence') == 'absent'}
    eligible = []
    for item in train_rows:
        row = item['record']
        qualified = by_hash.get(row['image_sha256'])
        if row['bins'] or qualified is None:
            continue
        geometry = qualified.get('geometry') or {}
        if not geometry.get('placeable_ground') or not geometry.get('ground_polygon'):
            continue
        observed = decode_placement_record(geometry['placeable_ground'])
        assessment = compose.output_ground(observed, qualified['image_size'])
        floor = assessment.candidate_mask
        image = compose.output_frame(compose.base.read_image(path(item['source'])))
        x0, y0, x1, y1 = compose.output_content_rectangle(qualified['image_size'])
        sw, sh = qualified['image_size']
        polygon = [[round(x0+x*(x1-x0)/sw), round(y0+y*(y1-y0)/sh)]
                   for x, y in geometry['ground_polygon']]
        background = {'id': qualified['frame_id'], 'image': item['source'], 'split': 'train',
                      'ground_polygon_xy': polygon, 'existing_bins': [],
                      'source_kind': row['source_kind'],
                      'parent_group': row['physical_parent_group'], 'evidence': qualified['record_id']}
        if floor.any():
            eligible.append((background, image, floor))
    random.Random(plan.get('seed', 42)).shuffle(eligible)
    count = plan.get('background_count', 64)
    if len(eligible) < count:
        raise ValueError(f'Need {count} prior Train empty ground backgrounds; found {len(eligible)}')
    return eligible[:count]


def background_order(backgrounds, state_index, *, height_count=6):
    """Pair every color-state stratum with the same alternating source-domain recipe."""
    pools = {'camera': [], 'blender': []}
    for item in backgrounds:
        kind = 'blender' if item[0]['source_kind'] == 'blender_empty_negative' else 'camera'
        pools[kind].append(item)
    # Alternate across both height and repetition, so each size occurs in both domains.
    requested = 'camera' if (state_index + state_index // height_count) % 2 == 0 else 'blender'
    alternative = 'blender' if requested == 'camera' else 'camera'
    offset = state_index // 2
    ordered = []
    for kind in (requested, alternative):
        candidates = pools[kind]
        if candidates:
            index = offset % len(candidates)
            ordered.extend(candidates[index:] + candidates[:index])
    if not ordered:
        raise ValueError('No admitted backgrounds for composition')
    return requested, ordered[0][0]['image'], ordered


def _make_single_composites(plan, annotations, baseline, output):
    backgrounds = admitted_backgrounds(plan, baseline['splits']['train'])
    train = plan['train']
    heights = train['target_heights_px']
    jobs = composition_schedule(annotations, heights, train['composites_per_color_state'])
    if len(jobs) != train['real_cutout_composites']:
        raise ValueError('Composite state allocations disagree with the plan total')
    manifest = output / 'manifest.json'
    records, entries = [], []
    for number, (source, height, state_index) in enumerate(jobs):
        color = source['color']
        rgba = compose.base.read_image(path(source['cutout']), cv2.IMREAD_UNCHANGED)
        if rgba.ndim != 3 or rgba.shape[2] != 4 or not np.any(rgba[:, :, 3]):
            raise ValueError('Source cutout requires nonempty alpha')
        # Each color/state uses the same 48 background/tone/height recipes. Real
        # camera backgrounds are reused explicitly, rather than counted as new captures.
        requested_domain, preferred_background, ordered = background_order(
            backgrounds, state_index, height_count=len(heights))
        rng = random.Random(plan.get('seed', 42) + state_index)
        selected = None
        for background in ordered:
            try:
                selected = compose.ground_placement(rgba, {color: rgba[:, :, 3] > 32},
                                                     [background], rng, height_range=(height, height))
                break
            except ValueError:
                continue
        if selected is None:
            raise ValueError(f'No admitted floor placement for target height {height}')
        background, pixels, floor, left, top, width, actual_height, masks, supports = selected
        actual_domain = 'blender' if background['source_kind'] == 'blender_empty_negative' else 'camera'
        fallback = actual_domain != requested_domain or background['image'] != preferred_background
        foreground = cv2.resize(rgba, (width, actual_height), interpolation=cv2.INTER_AREA)
        values, saturations = plan['augmentation']['value_gain'], plan['augmentation']['saturation_gain']
        tone = state_index // len(heights)
        value, saturation = values[tone % len(values)], saturations[(tone // len(values)) % len(saturations)]
        hsv = cv2.cvtColor(foreground[:, :, :3], cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= saturation
        hsv[:, :, 2] *= value
        foreground[:, :, :3] = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        result = compose.add_contact_shadows(pixels.copy(), floor, masks, supports)
        alpha = foreground[:, :, 3:4].astype(np.float32) / 255
        crop = result[top:top+actual_height, left:left+width]
        result[top:top+actual_height, left:left+width] = np.clip(
            crop * (1-alpha) + foreground[:, :, :3] * alpha, 0, 255).astype(np.uint8)
        mx, my, mw, mh = cv2.boundingRect(masks[color])
        box = [left+mx, top+my, left+mx+mw, top+my+mh]
        image = output / 'train/images' / f'reshoot_composite_{number:04d}.png'
        label = output / 'train/labels' / f'{image.stem}.txt'
        compose.write_png(image, result)
        text = compose.box_line(0 if source['state'] == 'open' else 1, box) + '\n'
        label.write_text(text, encoding='utf8')
        row = {'source_kind': 'real_target_composite', 'split': 'train',
               'source_image': full.relative(image), 'source_manifest': full.relative(manifest),
               'source_record_id': f'reshoot_composite_{number:04d}',
               'parent_group': background['parent_group'], 'physical_parent_group': background['parent_group'],
               'target_parent_group': source['physical_parent_group'],
               'capture_session': source['capture_session'], 'foreground_source': source['image'],
               'foreground_mask': source['mask'], 'background_image': background['image'],
               'background_source_kind': background['source_kind'],
               'background_recipe_index': state_index, 'requested_background_domain': requested_domain,
               'preferred_background_image': preferred_background,
               'background_placement_fallback': fallback,
               'background_fallback_reason': ('requested_domain_unavailable_or_no_floor_valid_placement'
                                              if actual_domain != requested_domain else
                                              'no_floor_valid_placement_for_preferred_background'
                                              if fallback else None),
               'background_evidence': background['evidence'], 'background_split': 'train',
               'generation_method': 'real_cutout_composite', 'scale_basis': 'planned_approximation',
               'ground_support_xy': supports[color],
               'allocation_axes': {'color': color, 'view': source.get('view', 'unknown'),
                                   'occlusion': 'separated', 'target_height': height,
                                   'tone': f'value_{value:g}_saturation_{saturation:g}'},
               'augmentation': {'value_gain': value, 'saturation_gain': saturation},
               'bins': [{'color': color, 'state': source['state'], 'bbox_xyxy': box}]}
        records.append(row)
        entries.append((image, text, row, {image, path(source['image']), path(background['image'])},
                        {manifest, path(source['mask']), path(source['cutout'])}))
    return records, entries


def make_composites(plan, annotations, baseline, output):
    if 'composition_groups' not in plan['train']:
        return _make_single_composites(plan, annotations, baseline, output)
    from datasets.scripts.waste_bin_state import reshoot_groups
    backgrounds = admitted_backgrounds(plan, baseline['splits']['train'])
    jobs = reshoot_groups.schedule(annotations, plan)
    if len(jobs) != plan['train']['real_cutout_composites']:
        raise ValueError('Composite group allocations disagree with the plan total')
    manifest = output / 'manifest.json'
    records, entries = [], []
    for number, job in enumerate(jobs):
        pixels, bins, trace = reshoot_groups.render(job, backgrounds, plan)
        image = output / 'train/images' / f'real_priority_composite_{number:04d}.png'
        label = output / 'train/labels' / f'{image.stem}.txt'
        compose.write_png(image, pixels)
        text = ''.join(compose.box_line(0 if b['state'] == 'open' else 1, b['bbox_xyxy']) + '\n'
                       for b in bins)
        label.write_text(text, encoding='utf8')
        background = trace['background']
        sources = job['sources']
        row = {'source_kind': 'real_target_composite', 'split': 'train',
               'source_image': full.relative(image), 'source_manifest': full.relative(manifest),
               'source_record_id': image.stem,
               'parent_group': background['parent_group'], 'physical_parent_group': background['parent_group'],
               'target_parent_group': sources[0]['physical_parent_group'],
               'capture_session': sources[0]['capture_session'],
               'foreground_sources': trace['foreground_sources'],
               'background_image': background['image'], 'background_source_kind': background['source_kind'],
               'background_evidence': background['evidence'], 'background_split': 'train',
               'generation_method': 'real_cutout_composite', 'scale_basis': 'planned_approximation',
               'placement': trace,
               'allocation_axes': {'colors': [b['color'] for b in bins], 'view': 'normal' if
                                   trace['view_group'].startswith('medium_') else 'low' if
                                   trace['view_group'].startswith('low_') else 'unknown',
                                   'camera_view_group': trace['view_group'],
                                   'count': len(bins), 'layout': trace['actual_layout'],
                                   'target_height': job['height'],
                                   'occlusion': 'present' if any(b.get('occlusion_fraction', 0) > 0 for b in bins)
                                                else 'none'},
               'augmentation': {'value_gain': trace['value_gain'], 'saturation_gain': trace['saturation_gain'],
                                'scope': 'whole_composite_after_alpha_blending'}, 'bins': bins}
        records.append(row)
        media = {image, path(background['image']), *[path(s['image']) for s in sources]}
        procedural = {manifest, *[path(s[k]) for s in sources for k in ('mask', 'cutout')]}
        entries.append((image, text, row, media, procedural))
        if (number + 1) % 64 == 0:
            print(f'Composed {number + 1}/{len(jobs)} real-bin scenes', file=sys.stderr, flush=True)
    return records, entries


def prepare(plan_path):
    plan = read(plan_path)
    output = path(plan.get('output', 'datasets/annotated/waste_bin_state/reshoot_finetune'))
    for directory in (output, output / 'train/images', output / 'train/labels'):
        directory.mkdir(parents=True, exist_ok=True)
    baseline = baseline_snapshot(plan, output)
    annotation_path = path(plan.get('annotation_manifest', 'datasets/annotated/waste_bin_state/reshoot/manifest.json'))
    annotations = json.loads(annotation_path.read_text(encoding='utf8'))['records']
    if plan.get('view_observations'):
        observations = read(plan['view_observations'])['observations_by_source_image']
        for source in annotations:
            observed = observations[source['source_image']]
            if (observed['source_sha256'] != source['source_sha256'] or
                    not observed['state_consistent'] or observed['observed_state'] != source['state']):
                raise ValueError('View observation differs from reviewed source: ' + source['image'])
            source['view_observation'] = observed
    train = plan['train']
    if len(annotations) != train['source_photos'] or any(r['split'] != 'train' for r in annotations):
        raise ValueError('All requested reshoot photos must remain in Train')
    result = {split: [] for split in full.SPLIT_ORDER}
    by_id = {r['record']['source_record_id']: r for r in baseline['splits']['train']}
    replay = choose_replay([r['record'] for r in baseline['splits']['train']],
                           train['replay_images'], train['replay_empty_images'], plan.get('seed', 42))
    for split in full.SPLIT_ORDER:
        selected = [by_id[r['source_record_id']] for r in replay] if split == 'train' else baseline['splits'][split]
        for entry in selected:
            row = deepcopy(entry['record'])
            image = path(entry['source'])
            result[split].append((image, entry['label_text'], row, {image}, {path(row['source_manifest'])}))
    for source in annotations:
        image = path(source['image'])
        digest = sha(image)
        if digest != source['source_sha256']:
            raise ValueError('Reviewed reshoot pixels changed')
        record = {**deepcopy(source), 'source_kind': 'direct_real',
                  'source_image': source['image'], 'source_manifest': full.relative(annotation_path),
                  'parent_group': source['physical_parent_group'],
                  'annotation': {**source.get('annotation', {}), 'completeness': 'complete', 'presence': 'present',
                                 'evidence': {'coordinate_size': source['image_size'],
                                              'method': 'codex_visual_original_full_frame_review'}},
                  'frame_environment': {'content_sha256': digest, 'split': 'train'}}
        if 'view_observation' in source:
            observed = source['view_observation']
            record['allocation_axes'] = {'view': 'low' if observed['elevation'] == 'low' else 'normal',
                                         'camera_view_group': observed['compatibility_group']}
        result['train'].append((image, path(source['label']).read_text(encoding='utf8'), record,
                                {image}, {annotation_path, path(source['label'])}))
    records, composites = make_composites(plan, annotations, baseline, output)
    result['train'].extend(composites)
    if len(result['train']) != train['total']:
        raise ValueError('Actual fine-tuning Train size differs from plan')
    full.write_json(output / 'manifest.json', {'task': full.TASK, 'status': 'complete_finetune_sources',
                                             'plan': plan_path, 'generation_plan': deepcopy(plan), 'records': records,
                                             'view_observations': plan.get('view_observations'),
                                             'reshoot_session': annotations[0]['capture_session'],
                                             'replay_records': [r['source_record_id'] for r in replay],
                                             'evaluation_scope': 'existing Val/Test, no reshoot session held out'})
    return result, plan, output


def publish(rows, plan_path, plan, output):
    """Reuse canonical image validation and atomic publication with truthful new plan references."""
    base = full.dataset_dir(full.TASK).resolve()
    outputs = [*(full.split_dir(full.TASK, s) for s in full.SPLIT_ORDER),
               *(base / n for n in ('data.yaml', 'manifest.json', 'workspace_sources.json'))]
    if any(p.resolve() != p or p.parent != base for p in outputs):
        raise ValueError('Publication targets must stay in the canonical export')
    media = sorted({p for entries in rows.values() for entry in entries for p in entry[3]})
    procedural = sorted({path(plan_path), output / 'baseline_manifest.json',
                         ROOT / 'datasets/scripts/waste_bin_state/config/dataset.yaml',
                         ROOT / 'datasets/scripts/waste_bin_state/config/source.yaml',
                         *{p for entries in rows.values() for entry in entries for p in entry[4]}})
    if plan.get('view_observations'):
        procedural.append(path(plan['view_observations']))
    code_paths = [Path(__file__), Path(full.__file__)]
    if 'composition_groups' in plan['train']:
        code_paths.append(ROOT / 'datasets/scripts/waste_bin_state/reshoot_groups.py')
    backup = full.HISTORY / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')

    def verify_existing(workspace, output_state, previous):
        # This entry replaces the user's current generated export and archives
        # it intact. History is shared by full and fine-tuning publications, so
        # it legitimately grows outside this operation's last receipt.
        if (workspace.resolve() != ROOT.resolve() or full.HISTORY.is_symlink() or
                backup.exists() or any(p.is_symlink() for p in outputs)):
            raise ValueError('Unexpected export/history target during replacement')
        if (base / 'manifest.json').is_file():
            full.training_sources(base, base)  # Validate ownership, hashes and label geometry.
        elif any(p.exists() for p in outputs):
            raise ValueError('Existing export cannot be identified by its task manifest')
        return True

    installed, moved = [], []
    with TemporaryDirectory(dir=ROOT / 'tmp', prefix='bin_reshoot_publish_') as temporary:
        stage = Path(temporary)
        try:
            with generation_guard('waste_bin_state.build_reshoot_finetune', media, [*outputs, full.HISTORY],
                                  {'plan': plan_path}, code_paths=code_paths,
                                  procedural_inputs=procedural, root=ROOT,
                                  replace_outputs=True, replace_verifier=verify_existing) as run:
                if run.reused:
                    return full.verified.read_json(base / 'manifest.json')
                summary = full.verify_and_summarize(rows)
                summary['train'].update(composition_manifest=full.relative(output / 'manifest.json'),
                                        composition_plan=plan_path, allocation_audit=plan_path)
                full.write_output(stage, rows, summary, [])
                document = full.verified.read_json(stage / 'manifest.json')
                document.update(phase='four_color_reshoot_finetune', plan=plan_path,
                                generation_plan=deepcopy(plan),
                                train_source_budget=dict(Counter(r[2]['source_kind'] for r in rows['train'])),
                                baseline_manifest=full.relative(output / 'baseline_manifest.json'),
                                annotation_manifest=plan.get('annotation_manifest'),
                                source_scope='One reshoot session of four physical colored bins; derivatives are not independent captures.')
                full.write_json(stage / 'manifest.json', document)
                full.write_json(stage / 'workspace_sources.json', full.training_sources(stage, base))
                full.HISTORY.mkdir(exist_ok=True)
                backup.mkdir()
                full.install(stage, outputs, backup, installed, moved)
                full.write_json(backup / 'publication_backup.json', {'task': full.TASK,
                                'original_root': full.relative(base), 'paths': [full.relative(p) for _, p in moved]})
                run.set_result({'images': document['images'], 'splits': summary,
                                'previous_export_archive': full.relative(backup)})
                return document
        except BaseException:
            for original, staged in reversed(installed):
                if original.exists() and not staged.exists():
                    original.rename(staged)
            for saved, original in reversed(moved):
                if saved.exists() and not original.exists():
                    saved.rename(original)
            if backup.is_dir():
                (backup / 'publication_backup.json').unlink(missing_ok=True)
                if not any(backup.iterdir()):
                    backup.rmdir()
            raise


def verify_publication(report, plan, output):
    """Read back installed labels and held-out bytes; record this revision only."""
    base = full.dataset_dir(full.TASK)
    baseline = read(full.relative(output / 'baseline_manifest.json'))
    unchanged = {}
    for split in ('val', 'test'):
        current = read(full.relative(base / split / 'manifest.json'))['records']
        previous = {e['record']['source_record_id']: e for e in baseline['splits'][split]}
        if len(current) != len(previous):
            raise ValueError('Held-out set count changed: ' + split)
        for row in current:
            old = previous[row['source_record_id']]
            if (sha(base / split / row['image']) != old['record']['image_sha256'] or
                    sha(base / split / row['label']) != old['label_sha256']):
                raise ValueError('Held-out image/label bytes changed: ' + row['source_record_id'])
        unchanged[split] = len(current)
    train = read(full.relative(base / 'train/manifest.json'))['records']
    composite = [r for r in train if r['source_kind'] == 'real_target_composite']
    instance_counts = Counter((b['color'], b['state']) for r in composite for b in r['bins'])
    all_instances = sum(len(r['bins']) for r in train)
    real_instances = sum(len(r['bins']) for r in train if r['source_kind'] in ('direct_real', 'real_target_composite'))
    source_index = read(full.relative(base / 'workspace_sources.json'))['records']
    if len(source_index) != report['images']:
        raise ValueError('Published source-index count mismatch')
    if 'composition_groups' in plan['train']:
        quota = sum(g['images'] * g['count'] for g in plan['train']['composition_groups']) // 8
        if instance_counts != Counter({(c, s): quota for c in ('blue', 'gray', 'green', 'red')
                                       for s in ('open', 'closed')}):
            raise ValueError('Actual real composite color-state instances differ from the balanced recipe')
    verification = {'task': full.TASK, 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
                    'scope': 'Current real-priority data revision; no training started for this revision.',
                    'plan': report.get('plan', DEFAULT_PLAN), 'images': report['images'],
                    'splits': {s: report['splits'][s]['images'] for s in full.SPLIT_ORDER},
                    'evaluation_bytes_unchanged': unchanged, 'training_started': False,
                    'source_index_records': len(source_index), 'train_summary': report['splits']['train'],
                    'real_target_instances': real_instances, 'all_train_instances': all_instances,
                    'real_target_instance_fraction': real_instances / all_instances,
                    'composite_color_state_instances': {f'{c}:{s}': n for (c, s), n in sorted(instance_counts.items())},
                    'composite_backgrounds': dict(Counter(r['background_source_kind'] for r in composite)),
                    'composite_view_groups': dict(Counter(r['placement']['view_group'] for r in composite
                                                         if 'placement' in r)),
                    'requested_layouts': dict(Counter(r['placement']['requested_layout'] for r in composite
                                                      if 'placement' in r)),
                    'actual_layouts': dict(Counter(r['placement']['actual_layout'] for r in composite
                                                   if 'placement' in r)),
                    'images_with_measured_occlusion': sum(any(b.get('occlusion_fraction', 0) > 0
                                                              for b in r['bins']) for r in composite),
                    'view_observations': plan.get('view_observations'),
                    'evaluation_scope': plan['evaluation_scope']}
    full.write_json(output / 'verification.json', verification)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default=DEFAULT_PLAN)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    rows, plan, output = prepare(args.plan)
    report = publish(rows, args.plan, plan, output) if args.publish else full.verify_and_summarize(rows)
    if args.publish:
        verify_publication(report, plan, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
