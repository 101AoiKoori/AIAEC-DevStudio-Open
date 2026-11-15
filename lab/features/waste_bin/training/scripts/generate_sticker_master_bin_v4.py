"""Composite planned sticker candidates onto declared master-scene bin renders."""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/waste_bin_state or waste_sticker/. Production remains paused.")

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

import composite_bin_real_scene as base
import generate_waste_v4_grouped as grouped

ROOT = base.ROOT
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard

CONSTRAINTS = {'canvas': [640, 480], 'min_paper_side_px': 26,
               'placement': 'visible_mask_bright_detail_avoidance', 'source_selection': 'seeded_pool_exhaustion'}


def decode_masks(path, colors):
    mask = base.read_image(path, cv2.IMREAD_UNCHANGED)
    if mask.shape != (480, 640, 4):
        raise ValueError('unexpected mask dimensions')
    rgb = mask[:, :, [2, 1, 0]]
    output = {}
    for color in colors:
        hit = mask[:, :, 3] >= 191
        for channel, value in enumerate(base.COLORS[color]):
            hit &= rgb[:, :, channel] >= 191 if value else rgb[:, :, channel] <= 63
        if not hit.any():
            raise ValueError('missing target instance mask: ' + color)
        output[color] = hit
    return output


def place_avoiding_visible_graphics(original, mask, icon, rng):
    """Avoid visible bright details; this does not identify semantic symbols."""
    trial = original.copy()
    initial = base.paper_on_bin(trial, mask, icon, rng)
    if initial is None:
        return None
    x0, y0, x1, y1 = initial
    w, h = x1-x0, y1-y0
    patch = trial[y0:y1, x0:x1].copy()
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((11, 11), dtype=np.uint8))
    visible_graphics = cv2.dilate((top_hat >= 14).astype(np.uint8), np.ones((5, 5), dtype=np.uint8))
    safe = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), dtype=np.uint8))
    bx, by, bw, bh = cv2.boundingRect(mask.astype(np.uint8))
    possibilities = []
    for yy in range(max(by, by + bh//2), min(by + bh-h, original.shape[0]-h)+1, 2):
        for xx in range(max(0,bx), min(bx+bw-w,original.shape[1]-w)+1, 2):
            if np.all(safe[yy:yy+h,xx:xx+w]) and not visible_graphics[yy:yy+h,xx:xx+w].any():
                possibilities.append((xx, yy))
    if not possibilities:
        return None
    xx, yy = rng.choice(posibilities)
    result = original.copy()
    result[yy:yy+h,xx:xx+w] = patch
    return result, (xx,yy,xx+w,yy+h)


def validate_plan(plan, names, manifests):
    if plan['model_key'] != 'waste_sticker' or plan['constraints'] != CONSTRAINTS:
        raise ValueError('unsupported task or master-bin constraints')
    if len(names) != 24 or len(set(names)) != 24:
        raise ValueError('expected 24 distinct sticker classes')
    manifest_names = [path.relative_to(ROOT).as_posix() for path in manifests]
    if not manifest_names or len(manifest_names) != len(set(manifest_names)):
        raise ValueError('provide distinct source manifests')
    for state in plan['strata']:
        params = state['parameters']
        if set(params) != {'class_id', 'bin_color', 'gain', 'saturation'}:
            raise ValueError('unsupported or missing master-bin parameters')
        cls, color = params['class_id'], params['bin_color']
        if cls is not None and (type(cls) is not int or not 0 <= cls < len(names)):
            raise ValueError('class_id must identify a sticker or be null for a candidate negative')
        if (cls is None and color is not None) or (cls is not None and color not in base.COLORS):
            raise ValueError('positive requires a supported bin_color; negative requires null')
        for key in ('gain', 'saturation'):
            value = params[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 2:
                raise ValueError(key + ' must be finite and between 0 and 2')
        name = names[cls] if cls is not None else None
        target = {'kind': 'sticker', 'class_id': cls, 'name': name} if name else {'kind': 'candidate_negative'}
        artwork = [f'datasets/raw/waste_bin/toolkit/SourcePic/{name}.png'] if name else []
        if (state['background'] != {'kind': 'blender_render_pool', 'manifests': manifest_names} or
                state['target'] != target or state['labels'] != {'format': 'yolo_detection', 'class_id': cls} or
                state['augmentation'] != {'gain': params['gain'], 'saturation': params['saturation'],
                                          'variation': 'seeded_source_and_paper_placement'} or
                state['sources'] != manifest_names + artwork):
            raise ValueError('state declarations do not match supported master-bin inputs: ' + state['id'])
    return plan


def source_files(manifests):
    """Load source declarations and collect candidate input files without synthesizing pixels."""
    sources, media, procedural = [], set(), set(manifests)
    for manifest in manifests:
        report = json.loads(manifest.read_text(encoding='utf8'))
        if report.get('status') != 'complete_candidates':
            raise ValueError('source batch incomplete: ' + str(manifest))
        for key in ('scene_file', 'bin_source'):
            path = (ROOT / report[key]).resolve()
            if not path.is_relative_to(ROOT.resolve()):
                raise ValueError('scene and bin asset paths must stay in the workspace')
            procedural.add(path)
        for index, sample in enumerate(report['samples']):
            eligible = [x['bin'] for x in sample['labels'] if x.get('target_eligible')]
            if not sample['bin_count'] or not sample.get('target_mask') or not eligible:
                continue
            image = (manifest.parent / sample['image']).resolve()
            mask = (manifest.parent / sample['target_mask']).resolve()
            if not image.is_relative_to(manifest.parent) or not mask.is_relative_to(manifest.parent):
                raise ValueError('source image and mask must remain inside their manifest directory')
            media.update((image, mask))
            sources.append((manifest, report, sample, eligible, index, image, mask))
    if not sources:
        raise ValueError('no eligible source render in declared manifests')
    return sources, media, procedural


def produce(out, seed, run, names, icons, sources):
    raise RuntimeError("Retired production entry; use the task scripts in datasets/scripts with its source.yaml")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--source-manifest', action='append', required=True)
    parser.add_argument('--seed', type=int, default=20261124)
    parser.add_argument('--variants', type=int, default=None)
    parser.add_argument('--training-batch', action='store_true')
    args = parser.parse_args()
    if args.variants is not None or args.training_batch:
        parser.error('variants and training-batch are replaced by planned state quotas')
    out = (ROOT / args.out).resolve()
    exports = (ROOT / 'datasets/exports').resolve()
    if not out.is_relative_to(exports) or not (out.name.startswith('waste_sticker_') or out == exports/'authoritative/waste_sticker'):
        parser.error('output must be the waste_sticker mainline or a waste_sticker_ directory under exports')
    plan_path = (ROOT / args.plan).resolve()
    manifests = [(ROOT / path).resolve() for path in args.source_manifest]
    if any(not path.is_relative_to(ROOT.resolve()) or not path.is_file() for path in [plan_path] + manifests):
        parser.error('plan and source manifests must be existing workspace files')
    classes = yaml.safe_load(base.CLASSES.read_text(encoding='utf8'))
    names = [item['name'] for item in classes['items']]
    declaration = json.loads(plan_path.read_text(encoding='utf8'))
    validate_plan(declaration, names, manifests)
    sources, media, procedural = source_files(manifests)
    for state in declaration['strata']:
        cls, color = state['parameters']['class_id'], state['parameters']['bin_color']
        if state['min_count'] and cls is not None and not any(color in source[3] for source in sources):
            raise ValueError('no source render supports planned state: ' + state['id'])
    procedural.update({base.CLASSES, ROOT/'config/perception_tasks.json', plan_path})
    procedural.update(base.CACHE / (name + '.npz') for name in names)
    media.update(ROOT / 'datasets/raw/waste_bin/toolkit/SourcePic' / (name + '.png') for name in names)
    codes = [__file__, base.__file__, grouped.__file__, ROOT/'lab/training_common/camera_tone.py']
    with generation_guard('waste_sticker_master_bin_v4:' + out.relative_to(ROOT).as_posix(),
                          sorted(media | procedural), [out], vars(args), code_paths=codes,
                          procedural_inputs=sorted(procedural), root=ROOT, production_plan=declaration,
                          sample_output_paths=[out/'images']) as run:
        if run.reused:
            print('Existing complete output reused:', out)
            return
        planned = validate_plan(run.plan, names, manifests)
        for state in planned['strata']:
            cls, color = state['parameters']['class_id'], state['parameters']['bin_color']
            if state['planned_count'] and cls is not None and not any(color in source[3] for source in sources):
                raise ValueError('no source render supports planned state: ' + state['id'])
        names_loaded, icons, _, _ = base.load_icons()
        if names_loaded != names:
            raise ValueError('sticker class names changed after plan admission')
        try:
            produce(out, args.seed, run, names, icons, sources)
        except BaseException:
            out.mkdir(parents=True, exist_ok=True)
            (out/'FAILED.txt').write_text('Generation stopped; inspect before reuse.\n', encoding='utf8')
            raise


if __name__ == '__main__':
    main()
