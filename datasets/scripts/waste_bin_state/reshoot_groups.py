"""Arrange reviewed real-bin cutouts without inventing camera views or lid geometry."""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations, product
from pathlib import Path
import random

import cv2
import numpy as np

from datasets.scripts.waste_bin_state import generate_waste_v4_grouped as compose

ROOT = Path(__file__).resolve().parents[3]
COLORS = ('blue', 'gray', 'green', 'red')
STATES = ('closed', 'open')


def schedule(annotations, plan):
    """Cross every requested state pattern with the same view/size/layout recipes."""
    eligible = [s for s in annotations if s.get('composite_eligible', True)]
    pools = {(c, s): sorted((r for r in eligible if r['color'] == c and r['state'] == s),
                           key=lambda r: r['source_record_id']) for c in COLORS for s in STATES}
    if any(not pool for pool in pools.values()):
        raise ValueError('Every color and state needs an eligible real cutout')
    view_groups = plan.get('multi_view_groups', ['medium_front', 'medium_rear'])
    layouts = plan.get('layouts', ['separated', 'adjacent', 'light_overlap'])
    if not view_groups or not layouts or any(v not in ('separated', 'adjacent', 'light_overlap') for v in layouts):
        raise ValueError('A real group needs observed views and supported layout recipes')
    jobs = []
    for spec in plan['train']['composition_groups']:
        count, images, heights = spec['count'], spec['images'], spec['heights_px']
        if count not in (1, 2, 4) or not heights:
            raise ValueError('Supported real-bin group counts are 1, 2 and 4')
        color_sets = list(combinations(COLORS, count))
        patterns = list(product(STATES, repeat=count))
        strata = len(color_sets) * len(patterns)
        if images % strata:
            raise ValueError('Group quota must cover every color/state combination equally')
        repeats = images // strata
        for colors in color_sets:
            paired_views = []
            if count == 1:
                observed = [{r.get('view_observation', {}).get('compatibility_group')
                             for r in pools[colors[0], state]} for state in STATES]
                paired_views = sorted((observed[0] & observed[1]) - {None, 'unknown'})
                if not paired_views:
                    raise ValueError(f'No paired-state observed views for {colors[0]}')
            for states in patterns:
                single_order = []
                # Both states receive the same ordered view quotas. A one-state
                # view (for example low closed only) must not be amplified by
                # cutout reuse. Within each paired view, share its quota fairly.
                for view_index, view in enumerate(paired_views):
                    quota = repeats // len(paired_views) + (view_index < repeats % len(paired_views))
                    sources = [r for r in pools[colors[0], states[0]]
                               if r.get('view_observation', {}).get('compatibility_group') == view]
                    single_order.extend(source for i, source in enumerate(sources)
                                        for _ in range(quota // len(sources) + (i < quota % len(sources))))
                for recipe in range(repeats):
                    height_index = recipe % len(heights) if count == 1 else (recipe // len(view_groups)) % len(heights)
                    view_group = None if count == 1 else view_groups[recipe % len(view_groups)]
                    sources = []
                    for color, state in zip(colors, states):
                        matching = pools[color, state] if count == 1 else [
                            r for r in pools[color, state]
                            if r.get('view_observation', {}).get('compatibility_group') == view_group]
                        if not matching:
                            raise ValueError(f'No compatible observed view for {color}/{state}/{view_group}')
                        sources.append(single_order[recipe] if count == 1 else matching[(recipe // 8) % len(matching)])
                    rotation = recipe % count
                    sources = sources[rotation:] + sources[:rotation]
                    view_group = view_group or sources[0].get('view_observation', {}).get('compatibility_group', 'unknown')
                    jobs.append({'sources': sources, 'recipe_index': recipe, 'height': heights[height_index],
                                 'height_index': height_index, 'layout': 'single' if count == 1 else
                                 layouts[recipe % len(layouts)],
                                 'view_group': view_group, 'count': count})
    if len(jobs) != plan['train']['real_cutout_composites']:
        raise ValueError('Real-cutout schedule and total disagree')
    return jobs


@lru_cache(maxsize=256)
def _patch(cutout, height):
    rgba = compose.base.read_image(ROOT / cutout, cv2.IMREAD_UNCHANGED)
    if rgba.ndim != 3 or rgba.shape[2] != 4 or not (rgba[:, :, 3] > 32).any():
        raise ValueError('Real cutout needs a nonempty alpha channel: ' + str(cutout))
    width = max(1, round(rgba.shape[1] * height / rgba.shape[0]))
    return cv2.resize(rgba, (width, height), interpolation=cv2.INTER_AREA)


def _alpha_over(destination, foreground):
    a = foreground[:, :, 3:4].astype(np.float32) / 255
    old_a = destination[:, :, 3:4].astype(np.float32) / 255
    new_a = a + old_a * (1-a)
    rgb = (foreground[:, :, :3] * a + destination[:, :, :3] * old_a * (1-a)) / np.maximum(new_a, 1e-8)
    destination[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    destination[:, :, 3:4] = np.rint(new_a * 255).astype(np.uint8)


def _group(job, layout):
    """Build an RGBA group and full masks before its common ground placement."""
    height = job['height']
    patches = [_patch(s['cutout'], height) for s in job['sources']]
    # Fixed slots cover all front/rear source aspect ratios, so lid-state changes
    # do not themselves move a bin to a different scene position.
    slot = max(round(height * .94), max(p.shape[1] for p in patches))
    step = slot + (max(14, round(height * .12)) if layout == 'separated' else 5)
    if layout == 'light_overlap':
        step = round(slot * .83)
    # A shallow side overlap covers this top-heavy bin's lid first. A deeper
    # stagger confines any overlap to the rear bin's body instead.
    depth = round(height * .55) if layout == 'light_overlap' else 0
    width = step * (len(patches)-1) + slot
    canvas = np.zeros((height + depth, width, 4), dtype=np.uint8)
    masks, protected, placements = {}, {}, {}
    for i, (source, patch) in enumerate(zip(job['sources'], patches)):
        color = source['color']
        top = depth if i % 2 else 0
        left = i * step + (slot-patch.shape[1]) // 2
        mask = np.zeros(canvas.shape[:2], dtype=bool)
        mask[top:top+height, left:left+patch.shape[1]] = patch[:, :, 3] > 32
        evidence = mask.copy()
        evidence[top+round(height*.45):] = False
        masks[color], protected[color] = mask, evidence
        placements[color] = [left, top, patch.shape[1], height]
    draw_order = sorted(range(len(patches)), key=lambda i: (placements[job['sources'][i]['color']][1], i))
    for i in draw_order:
        color = job['sources'][i]['color']
        left, top, width, height = placements[color]
        _alpha_over(canvas[top:top+height, left:left+width], patches[i])
    visible, covered = {}, np.zeros(canvas.shape[:2], dtype=bool)
    for i in reversed(draw_order):
        color = job['sources'][i]['color']
        if np.any(protected[color] & covered):
            raise ValueError('Group would cover the conservative lid-evidence region')
        visible[color] = masks[color] & ~covered
        if not visible[color].any():
            raise ValueError('Group completely hides a bin')
        covered |= masks[color]
    return canvas, masks, visible, placements


def _background_order(backgrounds, job, plan):
    cycle = plan.get('background_domain_cycle', ['camera', 'camera', 'camera', 'blender'])
    recipe = job['recipe_index']
    requested = cycle[(recipe + job['height_index']) % len(cycle)]
    pools = {'camera': [], 'blender': []}
    for entry in backgrounds:
        kind = 'blender' if entry[0]['source_kind'] == 'blender_empty_negative' else 'camera'
        pools[kind].append(entry)
    ordered = []
    for kind in (requested, 'blender' if requested == 'camera' else 'camera'):
        pool = pools[kind]
        if pool:
            index = (recipe // len(cycle)) % len(pool)
            ordered.extend(pool[index:] + pool[:index])
    if not ordered:
        raise ValueError('No admitted Train ground backgrounds')
    return requested, ordered


def render(job, backgrounds, plan):
    """Render a real-cutout group; return pixels, visible whole-bin boxes and trace."""
    requested_layout = job['layout']
    fallback_reasons = []
    requested_domain, ordered = _background_order(backgrounds, job, plan)
    preferred = ordered[0][0]['image']
    selected = None
    layouts = [requested_layout, 'adjacent'] if requested_layout == 'light_overlap' else [requested_layout]
    for actual_layout in layouts:
        try:
            group, masks, visible, placements = _group(job, actual_layout)
        except ValueError:
            if actual_layout != 'light_overlap':
                raise
            fallback_reasons.append('light_overlap_would_obscure_conservative_lid_evidence')
            continue
        for background in ordered:
            try:
                selected = compose.ground_placement(
                    group, masks, [background], random.Random(plan.get('seed', 42) + job['recipe_index']),
                    height_range=(group.shape[0], group.shape[0]))
                break
            except ValueError:
                continue
        if selected is not None:
            break
        if actual_layout == 'light_overlap':
            fallback_reasons.append('no_floor_valid_placement_for_depth_staggered_overlap')
    if selected is None:
        raise ValueError('No admitted ground fits the real-bin group')
    background, pixels, floor, left, top, width, height, scaled_masks, supports = selected
    result = compose.add_contact_shadows(pixels.copy(), floor, scaled_masks, supports)
    foreground = cv2.resize(group, (width, height), interpolation=cv2.INTER_AREA)
    alpha = foreground[:, :, 3:4].astype(np.float32) / 255
    region = result[top:top+height, left:left+width]
    region[:] = np.clip(region * (1-alpha) + foreground[:, :, :3] * alpha, 0, 255).astype(np.uint8)
    bins, source_traces = [], []
    for source in job['sources']:
        color = source['color']
        raster = cv2.resize(visible[color].astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        x, y, w, h = cv2.boundingRect(raster)
        box = [left+x, top+y, left+x+w, top+y+h]
        compose.box_line(0 if source['state'] == 'open' else 1, box)
        full = scaled_masks[color]
        fx, fy, fw, fh = cv2.boundingRect(full)
        fraction = float(raster.sum() / max(1, full.sum()))
        bins.append({'color': color, 'state': source['state'], 'bbox_xyxy': box,
                     'visible_fraction': fraction, 'occlusion_fraction': 1-fraction,
                     'actual_height_px': fh})
        source_traces.append({'image': source['image'], 'mask': source['mask'], 'cutout': source['cutout'],
                              'source_record_id': source['source_record_id'], 'color': color, 'state': source['state'],
                              'view_observation': source.get('view_observation', {}),
                              'full_bbox_xyxy': [left+fx, top+fy, left+fx+fw, top+fy+fh],
                              'visible_bbox_xyxy': box, 'visible_fraction': fraction,
                              'occlusion_fraction': 1-fraction, 'actual_height_px': fh,
                              'group_placement_xywh': placements[color],
                              'transform': {'group_scale_xy': [width/group.shape[1], height/group.shape[0]],
                                            'group_translation_xy': [left, top]}})
    values, saturations = plan['augmentation']['value_gain'], plan['augmentation']['saturation_gain']
    recipe = job['recipe_index']
    value, saturation = values[recipe % len(values)], saturations[(recipe // len(values)) % len(saturations)]
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= saturation
    hsv[:, :, 2] *= value
    result = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    actual_domain = 'blender' if background['source_kind'] == 'blender_empty_negative' else 'camera'
    if background['image'] != preferred:
        fallback_reasons.append('preferred_background_has_no_floor_valid_group_placement')
    if actual_domain != requested_domain:
        fallback_reasons.append('requested_background_domain_has_no_floor_valid_group_placement')
    trace = {'background': background, 'foreground_sources': source_traces, 'supports': supports,
             'actual_layout': actual_layout, 'requested_layout': requested_layout,
             'view_group': job['view_group'], 'value_gain': value, 'saturation_gain': saturation,
             'tone_scope': 'whole_composite', 'background_domain': actual_domain,
             'requested_background_domain': requested_domain, 'preferred_background_image': preferred,
             'background_placement_fallback': background['image'] != preferred or actual_domain != requested_domain,
             'layout_fallback': actual_layout != requested_layout, 'fallback_reasons': fallback_reasons,
             'group_placement_xywh': [left, top, width, height],
             'state_protection': 'conservative_upper_45_percent_of_each_source_mask; not a measured mouth segmentation',
             'perspective': 'same_visual_view_group; planned_scale_and_depth; no_camera_calibration'}
    return result, bins, trace
