"""Compose sticker candidates on a recorded real-scene frame with an indexed bin mask."""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/waste_bin_state or waste_sticker/. Production remains paused.")

import argparse
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
import composite_bin_real_scene as base
import generate_waste_v4_grouped as grouped
import generate_stickers as artwork_generator

ROOT = base.ROOT
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard

BACKGROUND = ROOT / 'datasets/annotated/fire/real_background_candidates_20261110/images/0001.png'
BACKGROUND_MANIFEST = ROOT / 'datasets/annotated/fire/real_background_candidates_20261110/manifest.json'
PARENT_VIDEO = 'datasets/raw/site_archive_legacy/场景/综合视频/空场与场景物料_001.mp4'
CONSTRAINTS = {'source_bin_color': 'red', 'height_px': [165, 210],
               'bottom_px': [410, 462], 'paper_size_rule': 'max(12,round(bin_width*0.22))'}


def validate_plan(plan, names, src, mask):
    """Reject states that this fixed-background, single-bin compositor cannot realize."""
    if plan['model_key'] != 'waste_sticker' or plan['constraints'] != CONSTRAINTS:
        raise ValueError('unsupported task or compositor constraints')
    if len(names) != 24 or len(set(names)) != 24:
        raise ValueError('expected 24 distinct sticker classes')
    background = BACKGROUND.relative_to(ROOT).as_posix()
    image = src.relative_to(ROOT).as_posix()
    mask_name = mask.relative_to(ROOT).as_posix()
    for state in plan['strata']:
        params = state['parameters']
        if set(params) != {'class_id', 'gain', 'saturation'}:
            raise ValueError('unsupported or missing compositor parameters')
        cls = params['class_id']
        if cls is not None and (type(cls) is not int or not 0 <= cls < len(names)):
            raise ValueError('class_id must identify a sticker or be null for a negative')
        for key in ('gain', 'saturation'):
            value = params[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 2:
                raise ValueError(key + ' must be finite and between 0 and 2')
        name = names[cls] if cls is not None else None
        target = {'kind': 'sticker', 'class_id': cls, 'name': name} if name else {'kind': 'negative'}
        artwork = [f'datasets/raw/waste_bin/toolkit/SourcePic/{name}.png'] if name else []
        if (state['background'] != {'kind': 'real_frame', 'image': background} or
                state['target'] != target or state['labels'] != {'format': 'yolo_detection', 'class_id': cls} or
                state['augmentation'] != {'gain': params['gain'], 'saturation': params['saturation'],
                                          'variation': 'seeded_position_scale_and_paper'} or
                state['sources'] != [background, image, mask_name] + artwork):
            raise ValueError('state declarations do not match supported compositor inputs: ' + state['id'])
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--scene-image', required=True)
    parser.add_argument('--scene-mask', required=True)
    parser.add_argument('--scene-manifest', required=True)
    parser.add_argument('--plan', required=True, help='workspace-relative generation declaration')
    parser.add_argument('--seed', type=int, default=20261130)
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    exports = (ROOT / 'datasets/exports').resolve()
    if not out.is_relative_to(exports) or not (out.name.startswith('waste_sticker_') or out == exports / 'authoritative/waste_sticker'):
        parser.error('output must be the waste_sticker mainline or a waste_sticker_ directory under exports')
    src, mask, scene_manifest = [(ROOT / x).resolve() for x in (args.scene_image, args.scene_mask, args.scene_manifest)]
    plan_path = (ROOT / args.plan).resolve()
    if any(not path.is_relative_to(ROOT.resolve()) or not path.is_file() for path in (src, mask, scene_manifest, plan_path)):
        parser.error('scene inputs and plan must be existing files inside the workspace')
    declaration = json.loads(plan_path.read_text(encoding='utf8'))
    classes = yaml.safe_load(base.CLASSES.read_text(encoding='utf8'))['items']
    names = [x['name'] for x in classes]
    validate_plan(declaration, names, src, mask)
    media = [BACKGROUND, src, mask]
    media += [ROOT / 'datasets/raw/waste_bin/toolkit/SourcePic' / (name + '.png') for name in names]
    toolkit = ROOT / 'datasets/raw/waste_bin/toolkit'
    procedural = [BACKGROUND_MANIFEST, scene_manifest, base.CLASSES, ROOT / 'config/perception_tasks.json',
                  toolkit / 'configs/gen_default.yaml', plan_path]
    with generation_guard('waste_sticker_real_parent_v4:' + out.relative_to(ROOT).as_posix(), media, [out],
                          vars(args), code_paths=[__file__, base.__file__, grouped.__file__, artwork_generator.__file__,
                                                 toolkit / 'tools/gen_dataset.py', toolkit / 'tools/matting.py',
                                                 ROOT / 'lab/training_common/camera_tone.py'],
                          procedural_inputs=procedural, root=ROOT, production_plan=declaration,
                          sample_output_paths=[out / 'images']) as run:
        if run.reused:
            print('existing output reused'); return
        cache = out / '.asset_cache'
        artwork_generator.init_worker(toolkit, cache, out, 640, 480, False)
        previous_cache = base.CACHE
        try:
            base.CACHE = cache
            names_loaded, icons, _, _ = base.load_icons()
        finally:
            base.CACHE = previous_cache
        if names_loaded != names:
            raise ValueError('sticker class mapping changed during admission')
        produce(out, src, mask, scene_manifest, args.seed, run, names_loaded, icons)


def place_paper_avoiding_bright_detail(image, mask, icon, rng):
    """Avoid the bounding rectangle of visible pale details; this is not a semantic mask."""
    x, y, width, height = cv2.boundingRect(mask.astype(np.uint8))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    band = np.zeros(mask.shape, dtype=np.uint8)
    band[y + round(height * .30):y + round(height * .90), x:x+width] = 1
    detail = (mask & (band > 0) & (hsv[:, :, 2] > 120) & (hsv[:, :, 1] < 125))
    yy, xx = np.where(detail)
    if len(xx) < 20:
        raise ValueError('visible pale bin details are insufficient for conservative avoidance')
    forbidden = [max(0, int(xx.min()) - 3), max(0, int(yy.min()) - 3),
                 min(image.shape[1], int(xx.max()) + 4), min(image.shape[0], int(yy.max()) + 4)]
    safe = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8))
    size = max(12, round(width * .22))
    candidates = []
    for py in range(y + 4, y + height - size, 2):
        for px in range(x + 4, x + width - size, 2):
            if not np.all(safe[py:py+size, px:px+size]): continue
            if px < forbidden[2] and forbidden[0] < px+size and py < forbidden[3] and forbidden[1] < py+size: continue
            if py < y + round(height * .20): continue
            candidates.append((px, py))
    if not candidates:
        raise ValueError('no paper position clear of detected bright details')
    px, py = rng.choice(candidates)
    paper = np.full((size, size, 3), rng.randint(228, 247), dtype=np.uint8)
    rgb, alpha = icon
    margin = max(1, round(size * .12)); inner = size - 2 * margin
    scale = min(inner / rgb.shape[1], inner / rgb.shape[0])
    iw, ih = max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale))
    ink = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_AREA)
    ink = np.clip(237.0 + (ink.astype(np.float32) - 237.0) * 1.35, 0, 255)
    a = cv2.resize(alpha, (iw, ih), interpolation=cv2.INTER_AREA)[:, :, None]
    ix, iy = (size-iw)//2, (size-ih)//2
    paper[iy:iy+ih, ix:ix+iw] = np.clip(ink * a + paper[iy:iy+ih, ix:ix+iw] * (1-a), 0, 255).astype(np.uint8)
    image[py:py+size, px:px+size] = paper
    return [px, py, px+size, py+size], forbidden


def produce(out, src, mask_path, scene_manifest, seed, run, names=None, icons=None):
    raise RuntimeError("Retired production entry; use the task scripts in datasets/scripts with its source.yaml")


if __name__ == '__main__': main()
