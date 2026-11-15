"""Produce planned single-sticker candidates on a synthetic flat panel.

Each stratum declares concrete supported fields; unspecified renderer states
cannot be inferred from a seed or counted as planned coverage.
"""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/waste_bin_state or waste_sticker/. Production remains paused.")

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'lab/features/waste_bin/training/scripts'))
from lab.training_common.production_guard import generation_guard
import generate_stickers as existing


CONSTRAINTS = {'paper_short_side_min_px': 120, 'image_margin_px': 2, 'max_attempts': 40}
TOOLKIT = 'datasets/raw/waste_bin/toolkit'


def validate_plan(plan, names):
    """Accept only states whose declared meaning this generator can enforce."""
    if plan['model_key'] != 'waste_sticker' or plan['constraints'] != CONSTRAINTS:
        raise ValueError('flat generator requires waste_sticker and its exact supported constraints')
    if len(names) != 24 or len(set(names)) != 24:
        raise ValueError('expected 24 distinct sticker classes')
    for state in plan['strata']:
        params = state['parameters']
        if set(params) != {'class_id', 'bin_color_index', 'gain', 'saturation'}:
            raise ValueError('unsupported or missing flat generator parameters')
        cls, color = params['class_id'], params['bin_color_index']
        if cls is not None and (type(cls) is not int or not 0 <= cls < len(names)):
            raise ValueError('class_id must identify a sticker or be null for a negative')
        if type(color) is not int or color not in range(4):
            raise ValueError('bin_color_index must be an integer from 0 to 3')
        for key in ('gain', 'saturation'):
            value = params[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 2:
                raise ValueError(key + ' must be finite and between 0 and 2')
        name = names[cls] if cls is not None else None
        background = {'kind': 'synthetic_flat_panel', 'color_index': color}
        target = {'kind': 'sticker', 'class_id': cls, 'name': name} if name else {'kind': 'negative'}
        labels = {'format': 'yolo_detection', 'class_id': cls}
        augmentation = {'gain': params['gain'], 'saturation': params['saturation'],
                        'gamma': 1.0, 'vignette': 0.0, 'color_temp_std': 0.0,
                        'blur_sigma': 0.0, 'noise_sigma': 0.0,
                        'variation': 'seeded_toolkit_geometry_and_texture'}
        sources = ([f'{TOOLKIT}/SourcePic/{name}.png'] if name else
                   [f'{TOOLKIT}/configs/gen_default.yaml'])
        if (state['background'] != background or state['target'] != target or state['labels'] != labels
                or state['augmentation'] != augmentation or state['sources'] != sources):
            raise ValueError('state declarations do not match supported parameters and sources: ' + state['id'])
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--plan', required=True, help='workspace-relative JSON generation declaration')
    parser.add_argument('--seed', type=int, default=20261117)
    parser.add_argument('--split', choices=('train', 'val', 'test'), default=None)
    # Old counts cannot override the allocation; keep explicit errors for callers.
    parser.add_argument('--per-class', type=int, default=None)
    parser.add_argument('--negatives', type=int, default=None)
    args = parser.parse_args()
    if args.per_class is not None or args.negatives is not None:
        parser.error('per-class and negatives are replaced by explicit planned state quotas')
    out = (ROOT / args.out).resolve()
    allowed = (ROOT / 'datasets/exports').resolve()
    if not out.is_relative_to(allowed) or not (out.name.startswith('waste_sticker_') or
            out == allowed / 'authoritative/waste_sticker'):
        parser.error('output must be the waste_sticker mainline or a waste_sticker_ directory under exports')
    plan_path = (ROOT / args.plan).resolve()
    if not plan_path.is_relative_to(ROOT.resolve()) or not plan_path.is_file():
        parser.error('plan must be an existing workspace file')
    declaration = json.loads(plan_path.read_text(encoding='utf-8'))
    if args.split is not None and args.split != declaration.get('target_split'):
        parser.error('split must match the generation declaration')
    toolkit = ROOT / TOOLKIT
    configs = [toolkit / 'configs/gen_default.yaml', toolkit / 'configs/classes.yaml']
    config = yaml.safe_load(configs[0].read_text(encoding='utf-8'))
    classes = yaml.safe_load(configs[1].read_text(encoding='utf-8'))
    names = [item['name'] for item in classes['items']]
    icon_dir = (toolkit / config['paths']['icons']).resolve()
    if icon_dir != (toolkit / 'SourcePic').resolve():
        raise ValueError('toolkit icons must match the declared SourcePic directory')
    icons = sorted(icon_dir.glob('*.png'))
    if {p.stem for p in icons} != set(names):
        raise ValueError('toolkit source icons do not match the 24 class names')
    validate_plan(declaration, names)
    code = [Path(__file__).resolve(), Path(existing.__file__).resolve(),
            toolkit / 'tools/gen_dataset.py', toolkit / 'tools/matting.py']
    with generation_guard('waste_sticker_flat_controlled:' + out.relative_to(ROOT).as_posix(),
                          icons + configs + [plan_path], [out], vars(args), code_paths=code,
                          procedural_inputs=configs + [plan_path], root=ROOT,
                          production_plan=declaration,
                          sample_output_paths=[out / 'images']) as run:
        if run.reused:
            print('Existing output reused:', out.relative_to(ROOT).as_posix())
            return
        validate_plan(run.plan, names)
        _generate(args, out, run)


def _generate(args, out, run):
    cv2.setNumThreads(1)
    toolkit = ROOT / TOOLKIT
    cache = out / '.asset_cache'
    existing.init_worker(toolkit, cache, out, 640, 480, False)
    cfg, gen, names = existing.CFG, existing.GEN, existing.NAMES
    plan = validate_plan(run.plan, names)
    split = plan['target_split']
    cfg['scene']['layout']['mode'] = 'single'
    cfg['scene']['panel']['color_jitter_hsv'] = [0, 0, 0]
    cfg['scene']['sticker']['ink_contrast'] = 1.0
    cfg['scene']['layout']['sticker']['ink_contrast'] = 1.0
    pc = cfg['photometric']
    pc['gamma_single'] = {'min': 1.0, 'max': 1.0}
    pc['vignette_single'] = {'min': 0.0, 'max': 0.0}
    pc['color_temp']['std'] = 0.0
    pc['blur_p'] = 0.0
    pc['noise_sigma'] = {'min': 0.0, 'max': 0.0}
    rng = np.random.default_rng(args.seed)
    used_seeds = set()
    records = []
    for folder in (f'images/{split}', f'labels/{split}'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    try:
        for state in plan['strata']:
            params = state['parameters']
            cls, color = params['class_id'], params['bin_color_index']
            gain, saturation = params['gain'], params['saturation']
            name = names[cls] if cls is not None else None
            pc['gain_single'] = {'min': gain, 'max': gain}
            for _ in range(state['planned_count']):
                ordinal = len(records)
                for attempt in range(CONSTRAINTS['max_attempts']):
                    for _ in range(16):
                        seed = int(rng.integers(0, 2**62))
                        if seed not in used_seeds:
                            used_seeds.add(seed)
                            break
                    else:
                        raise RuntimeError('could not allocate a distinct generation seed')
                    result = gen.gen_one({'idx': ordinal, 'seed': seed, 'items': [name] if name else [],
                                          'n_stickers': int(name is not None), 'negative': name is None,
                                          'panel_color': color})
                    if name is None:
                        if result['inst'] or result['det']:
                            raise RuntimeError('negative unexpectedly contains target labels')
                        item, box, quad = None, None, None
                        break
                    if len(result['inst']) != 1 or len(result['det']) != 1:
                        continue
                    item = result['inst'][0]
                    if item['name'] != name:
                        raise ValueError('generated sticker differs from the planned class')
                    box, quad = item['xyxy'], item['quad']
                    x1, y1, x2, y2 = box
                    if min(x2-x1, y2-y1) >= 120 and all(2 < x < 637 and 2 < y < 477 for x, y in quad):
                        break
                else:
                    raise RuntimeError('planned state could not be generated: ' + state['id'])
                image = result['img']
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 1] *= saturation
                image = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
                stem = f'{split}_{ordinal:06d}'
                im_path = out / 'images' / split / (stem + '.png')
                label_path = out / 'labels' / split / (stem + '.txt')
                ok, encoded = cv2.imencode('.png', image)
                if not ok:
                    raise RuntimeError('PNG encode failed')
                encoded.tofile(str(im_path))
                if name:
                    parts = result['det'][0].split()
                    parts[0] = str(cls)
                    label_path.write_text(' '.join(parts) + '\n', encoding='utf-8')
                else:
                    label_path.write_text('', encoding='utf-8')
                records.append({'id': stem, 'state_id': state['id'], 'split': split,
                    'image': im_path.relative_to(ROOT).as_posix(), 'label': label_path.relative_to(ROOT).as_posix(),
                    'class_id': cls, 'source_name': name, 'sources': state['sources'],
                    'source_kind': 'synthetic flat panel', 'seed': seed, 'attempt': attempt,
                    'parameters': params, 'paper_bbox_xyxy': box, 'projected_paper_quad': quad,
                    'item_theta': item.get('theta') if item else None,
                    'item_tilt_deg': item.get('tilt_deg') if item else None,
                    'distractor': result['distractor'], 'compression': 'PNG lossless',
                    'status': 'candidate_semantic_review_pending' if name else 'candidate_negative_closed_scene'})
                run.record_sample(state['id'], output=im_path, sources=state['sources'], parameters=params,
                                  actual={'seed': seed, 'attempt': attempt, 'paper_bbox_xyxy': box,
                                          'projected_paper_quad': quad, 'image_size': [640, 480],
                                          'item_theta': item.get('theta') if item else None,
                                          'item_tilt_deg': item.get('tilt_deg') if item else None,
                                          'distractor': result['distractor'], 'used_files': state['sources']})
        report = {'status': 'candidate_semantic_review_pending', 'seed': args.seed,
                  'plan': plan, 'counts': dict(Counter(r['source_name'] for r in records if r['source_name'])),
                  'negative_count': sum(r['source_name'] is None for r in records),
                  'source_reuse': '24 existing source artworks; generator and artworks may be shared with prior splits',
                  'field_scope': 'synthetic flat panel, not a physical bin or measured vehicle distance',
                  'records': records}
        (out / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        (out / 'classes.json').write_text(json.dumps(names, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'images': len(records), 'classes': len(report['counts'])}, ensure_ascii=False))
    except BaseException:
        (out / 'FAILED.txt').write_text('Generation stopped; inspect before use.\n', encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
