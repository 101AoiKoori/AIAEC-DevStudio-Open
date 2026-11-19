"""Meter selection and photometry over the shared pool's current task decisions."""
from collections import defaultdict
import itertools
import random

import numpy as np
from PIL import Image, ImageEnhance

from datasets.scripts.common.frame_environment import image_digest, query_frames


def select_backgrounds(selection, *, rows=None):
    """Keep source split and evidence; evenly sample each rendered scene/route."""
    rows = query_frames(task='meter', role='composite_background', split=selection['split']) if rows is None else rows
    groups = defaultdict(list)
    real = []
    for row in rows:
        if row['split'] != selection['split'] or row.get('status') != 'eligible':
            continue
        if row['label']['presence'] != 'absent' or row['label']['completeness'] != 'complete':
            continue  # Existing positive objects need a separate label-preserving compositor.
        if row['source_kind'] == 'blender_render':
            groups[(row['scene'], row['route'])].append(row)
        else:
            real.append(row)
    selected = sorted(real, key=lambda r: r['image'])
    n = selection['frames_per_scene_route']
    for _, candidates in sorted(groups.items()):
        candidates.sort(key=lambda r: (r['render_frame'], r['image']))
        count = min(n, len(candidates))
        selected.extend(candidates[round(i*(len(candidates)-1)/max(1,count-1))] for i in range(count))
    if not selected:
        raise ValueError('No eligible meter backgrounds in selected source split')
    # Interleave real and rendered inputs, keeping deterministic camera/material pairs.
    selected.sort(key=lambda r: r['record_id'])
    result = []
    for row in selected:
        if image_digest(row['image']) != row['content_sha256']:
            raise ValueError('Background bytes differ from current evidence: '+row['image'])
        result.append({key: row.get(key) for key in (
            'record_id','frame_id','image','image_size','content_sha256','parent_group',
            'source_kind','split','scene','route','render_frame','geometry','label','evidence')})
    return result


def photometric_schedule(recipe, count):
    """Balanced shuffled 3-way cross, independent of the paired face material."""
    levels = recipe['photometry']
    cells = list(itertools.product(*(levels[key] for key in ('saturation','exposure','light_relation'))))
    rng = random.Random(recipe['render_seed'])
    schedule = []
    while len(schedule) < count:
        cycle = cells.copy()
        rng.shuffle(cycle)
        schedule.extend(dict(zip(('saturation','exposure','light_relation'), cell)) for cell in cycle)
    return schedule[:count]


def adjust(image, *, saturation=1., brightness=1.):
    """Modify RGB only, retaining the exact alpha and all pixel coordinates."""
    alpha = image.getchannel('A') if image.mode == 'RGBA' else None
    result = ImageEnhance.Color(image.convert('RGB')).enhance(saturation)
    result = ImageEnhance.Brightness(result).enhance(brightness)
    if alpha is not None:
        result.putalpha(alpha)
    return result


def pixel_stats(image, mask=None):
    rgb = np.asarray(image.convert('RGB'),dtype=float)/255
    values = rgb[mask] if mask is not None else rgb.reshape(-1,3)
    if not len(values):
        raise ValueError('Empty photometry measurement mask')
    return {'mean_luma':float(np.mean(values @ np.array([.2126,.7152,.0722]))),
            'mean_chroma':float(np.mean(values.max(axis=1)-values.min(axis=1)))}


def placement(background, foreground):
    """Fit the station above a reviewed footprint, with a uniform image transform."""
    width,height = background['image_size']
    crop_width = min(width,height*4/3); crop_height = min(height,width*3/4)
    left,top = (width-crop_width)/2,(height-crop_height)/2
    factor = 640/crop_width
    support = next((r['xyxy'] for r in background['geometry']['regions'] if r['kind']=='meter_contact_support'), None)
    if support is None:
        support = [40,416,600,464]  # General empty-scene qualification; image-space near-floor approximation.
    else:
        support = [(support[0]-left)*factor,(support[1]-top)*factor,
                   (support[2]-left)*factor,(support[3]-top)*factor]
    x1,y1,x2,y2 = foreground.getchannel('A').getbbox()
    scale = min(1.,(support[2]-support[0]-8)/(x2-x1), (support[3]-12)/(y2-y1))
    foot = min(support[3]-6, max(support[1]+6,444))
    dx=(support[0]+support[2])/2-(x1+x2)*scale/2
    dy=foot-y2*scale
    if scale < .25:
        raise ValueError('Reviewed contact region too narrow for current station')
    return {'scale':scale,'offset_xy':[dx,dy],'region_xyxy':[0,0,640,480]}, support
