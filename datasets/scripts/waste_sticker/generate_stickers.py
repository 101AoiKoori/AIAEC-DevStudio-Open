"""Generate sticker images from explicit, planned toolkit states."""

if __name__ == "__main__":
    raise SystemExit("Task helper is import-only; use the primary entry in datasets/scripts/waste_bin_state or waste_sticker/.")

import argparse
import json
import multiprocessing as mp
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

GEN = CFG = NAMES = OUT = None
WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE))
from lab.training_common.production_guard import generation_guard


def init_worker(root, cache, out, width, height, vehicle_scale, *, icon_dir=None):
    """Initialize the workspace toolkit; also used by the flat sticker adapter."""
    global GEN, CFG, NAMES, OUT
    root, cache, OUT = Path(root), Path(cache), Path(out)
    source_path = (root / 'tools/gen_dataset.py').resolve()
    try:
        source_path.relative_to(WORKSPACE.resolve())
        (root / 'tools').resolve().relative_to(WORKSPACE.resolve())
        (root / 'tools/matting.py').resolve().relative_to(WORKSPACE.resolve())
    except ValueError:
        raise ValueError('Toolkit scripts must be within the workspace')
    sys.path.insert(0, str(root / 'tools'))
    source = source_path.read_text(encoding='utf-8')
    old = 'order = list(range(K)) * 8'
    if source.count(old) != 1:
        raise ValueError('Unsupported toolkit version: cannot enforce one sticker per panel')
    GEN = types.ModuleType('original_generator')
    GEN.__file__ = str(source_path)
    exec(compile(source.replace(old, 'order = list(range(K))'), str(source_path), 'exec'), GEN.__dict__)
    GEN.cv2.setNumThreads(1)
    CFG = yaml.safe_load((root / 'configs/gen_default.yaml').read_text(encoding='utf-8'))
    CFG['canvas']['width'], CFG['canvas']['height'] = width, height
    if vehicle_scale:
        CFG['scene']['sticker']['side_ratio'] = {'mean': .32, 'std': .07, 'min': .18, 'max': .50}
        CFG['scene']['layout']['sticker']['side_ratio'] = {'mean': .29, 'std': .08, 'min': .16, 'max': .50}
    classes = yaml.safe_load((root / 'configs/classes.yaml').read_text(encoding='utf-8'))
    NAMES = [item['name'] for item in classes['items']]
    if len(NAMES) != 24 or len(set(NAMES)) != 24:
        raise ValueError('Expected 24 distinct source icons')
    CFG['negative']['distractors']['kinds'] = ['stain', 'scuff', 'tape']
    CFG['negative']['distractors']['kinds_single'] = ['stain', 'scuff', 'tape']
    metas = GEN.prep_assets(str(icon_dir or root / CFG['paths']['icons']), str(cache))
    if {m['name'] for m in metas} != set(NAMES):
        raise ValueError('Missing or unexpected source icons')
    assets = []
    for m in metas:
        with np.load(cache / m['file']) as z:
            assets.append(dict(name=m['name'], cls=m['cls'], mode=m['mode'], rgb=z['rgb'], alpha=z['alpha'].astype(np.float32)))
    GEN._init_worker(CFG, assets)


def validate_plan(plan, names, toolkit, width, height, vehicle_scale):
    constraints = {'canvas': [width, height], 'vehicle_scale': vehicle_scale,
                   'max_attempts': 20, 'one_sticker_per_panel': True}
    if plan['model_key'] != 'waste_sticker' or plan['constraints'] != constraints:
        raise ValueError('unsupported task or toolkit generation constraints')
    if len(names) != 24 or len(set(names)) != 24:
        raise ValueError('expected 24 distinct sticker classes')
    toolkit_name = toolkit.relative_to(WORKSPACE).as_posix()
    for state in plan['strata']:
        params = state['parameters']
        if set(params) != {'class_ids', 'layout', 'panel_color'}:
            raise ValueError('unsupported or missing toolkit state parameters')
        ids, layout, color = params['class_ids'], params['layout'], params['panel_color']
        if (not isinstance(ids, list) or len(ids) > 4 or
                any(type(i) is not int or not 0 <= i < 24 for i in ids) or len(set(ids)) != len(ids)):
            raise ValueError('class_ids must contain zero to four distinct class indexes')
        if layout == 'single':
            if len(ids) > 1 or type(color) is not int or color not in range(4):
                raise ValueError('single layout requires at most one class and a panel color 0 to 3')
            palette = color
        elif layout == 'row':
            if color is not None:
                raise ValueError('row layout supports shuffled panel colors only')
            palette = 'shuffled'
        else:
            raise ValueError('layout must be single or row')
        names_selected = [names[i] for i in ids]
        sources = [f'{toolkit_name}/SourcePic/{name}.png' for name in names_selected]
        if not sources:
            sources = [f'{toolkit_name}/configs/gen_default.yaml']
        if (state['background'] != {'kind': 'synthetic_panels', 'layout': layout, 'palette': palette} or
                state['target'] != {'class_ids': ids, 'names': names_selected} or
                state['labels'] != {'format': 'yolo_detection', 'class_ids': ids} or
                state['augmentation'] != {'variation': 'seeded_toolkit_geometry_texture_photometric',
                                          'config': f'{toolkit_name}/configs/gen_default.yaml'} or
                state['sources'] != sources):
            raise ValueError('state declarations do not match supported toolkit inputs: ' + state['id'])
    return plan


def planned_jobs(plan, names, seed):
    rng = np.random.default_rng(seed)
    seeds, jobs = set(), []
    for state in plan['strata']:
        params = state['parameters']
        for _ in range(state['planned_count']):
            for attempt in range(16):
                block = int(rng.integers(0, 2**57))
                if block not in seeds:
                    seeds.add(block)
                    break
            else:
                raise RuntimeError('could not allocate a distinct generation seed block')
            jobs.append({'idx': len(jobs), 'split': plan['target_split'], 'state_id': state['id'],
                         'n': len(params['class_ids']), 'layout': params['layout'],
                         'items': [names[i] for i in params['class_ids']], 'color': params['panel_color'],
                         'seed': block * 20, 'sources': state['sources'], 'parameters': params})
    return jobs


def _generate_job(job):
    split, idx, n = job['split'], job['idx'], job['n']
    CFG['scene']['layout']['mode'] = job['layout']
    CFG['scene']['n_panels'] = {'values': [4], 'probs': [1.0]}
    stem = f'{split}_{idx:06d}'
    for retry in range(20):
        seed = job['seed'] + retry
        request = dict(idx=idx, seed=seed, items=job['items'], n_stickers=n, negative=n == 0)
        if job['color'] is not None:
            request['panel_color'] = job['color']
        result = GEN.gen_one(request)
        actual = [item['name'] for item in result['inst']]
        if result['n'] == len(result['det']) == len(actual) == n and sorted(actual) == sorted(job['items']):
            break
    else:
        raise RuntimeError(f'{stem}: could not render every planned class after 20 tries')
    image = OUT / 'images' / split / (stem + '.jpg')
    if not GEN.imwrite_u(str(image), result['img'], 90):
        raise RuntimeError(f'{stem}: image write failed')
    lines = []
    for label, inst in zip(result['det'], result['inst']):
        parts = label.split()
        parts[0] = str(NAMES.index(inst['name']))
        lines.append(' '.join(parts))
    (OUT / 'labels' / split / (stem + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
    record = dict(image=f'images/{split}/{stem}.jpg', state_id=job['state_id'], source='synthetic', seed=seed,
                  retries=retry, layout=job['layout'], requested=job['items'], panel_color=job['color'],
                  reported_panel=result.get('panel'), instances=result['inst'], distractor=result['distractor'])
    (OUT / 'metadata' / split / (stem + '.json')).write_text(json.dumps(record, ensure_ascii=False), encoding='utf-8')
    if idx < 12:
        vis = result['img'].copy()
        for inst in result['inst']:
            x1, y1, x2, y2 = map(int, inst['xyxy'])
            GEN.cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        if not GEN.imwrite_u(str(OUT / 'preview' / (stem + '.jpg')), vis, 90):
            raise RuntimeError(f'{stem}: preview write failed')
    return record


def _generate(args, root, cache, out, run, names):
    init_worker(root, cache, out, args.width, args.height, args.vehicle_scale)
    if NAMES != names:
        raise ValueError('toolkit class names changed after plan admission')
    plan = validate_plan(run.plan, names, root, args.width, args.height, args.vehicle_scale)
    split = plan['target_split']
    jobs = planned_jobs(plan, names, args.seed)
    by_id = {state['id']: state for state in plan['strata']}
    for sub in ('images', 'labels', 'metadata'):
        (out / sub / split).mkdir(parents=True, exist_ok=True)
    (out / 'preview').mkdir(exist_ok=True)
    records = []
    with mp.get_context('spawn').Pool(args.workers, initializer=init_worker,
                                       initargs=(str(root), str(cache), str(out), args.width, args.height, args.vehicle_scale)) as pool:
        for record in pool.imap_unordered(_generate_job, jobs, chunksize=2):
            state = by_id[record['state_id']]
            run.record_sample(state['id'], output=out / record['image'], sources=state['sources'],
                              parameters=state['parameters'], actual={'seed': record['seed'], 'retries': record['retries'],
                                  'instances': record['instances'], 'distractor': record['distractor'],
                                  'reported_panel': record['reported_panel'], 'used_files': state['sources']})
            records.append(record)
    records.sort(key=lambda r: r['image'])
    summary = {'images': len(records), 'object_counts': dict(Counter(str(len(r['instances'])) for r in records)),
               'classes': dict(Counter(x['name'] for r in records for x in r['instances']))}
    (out / 'manifest.json').write_text(json.dumps({'plan': plan, 'seed': args.seed, 'canvas': [args.width, args.height],
        'vehicle_scale_proposal': args.vehicle_scale, 'toolkit': root.relative_to(WORKSPACE).as_posix(),
        'summary': summary, 'images': records}, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'classes.json').write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--toolkit', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--cache', type=Path, default=None)
    p.add_argument('--count', type=int, default=None)
    p.add_argument('--val-count', type=int, default=None)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--vehicle-scale', action='store_true')
    p.add_argument('--seed', type=int, default=20261106)
    args = p.parse_args()
    if args.count is not None or args.val_count is not None:
        p.error('count and val-count are replaced by the explicit production plan')
    root, out, plan_path = args.toolkit.resolve(), args.out.resolve(), args.plan.resolve()
    exports = (WORKSPACE / 'datasets/exports').resolve()
    if not root.is_relative_to(WORKSPACE.resolve()) or not out.is_relative_to(exports):
        p.error('toolkit must be in the workspace and output under datasets/exports')
    if not (out.name.startswith('waste_sticker_') or out == exports / 'authoritative/waste_sticker'):
        p.error('output must be the waste_sticker mainline or a waste_sticker_ directory')
    cache = args.cache.resolve() if args.cache else out / '.asset_cache'
    if not cache.is_relative_to(out) or cache == out:
        p.error('cache must be a subdirectory of the output')
    if not plan_path.is_relative_to(WORKSPACE.resolve()) or not plan_path.is_file():
        p.error('plan must be an existing workspace file')
    if args.workers < 1 or min(args.width, args.height) < 1 or max(args.width, args.height) > 1280 or min(args.width, args.height) > 720:
        p.error('invalid workers or resolution exceeding 1280 by 720 in either orientation')
    configs = {root / 'configs/gen_default.yaml', root / 'configs/classes.yaml'}
    cfg = yaml.safe_load((root / 'configs/gen_default.yaml').read_text(encoding='utf-8'))
    names = [item['name'] for item in yaml.safe_load((root / 'configs/classes.yaml').read_text(encoding='utf-8'))['items']]
    icon_dir = (root / cfg['paths']['icons']).resolve()
    if icon_dir != (root / 'SourcePic').resolve():
        p.error('toolkit icons must match the declared SourcePic directory')
    icons = set(icon_dir.glob('*.png'))
    if {path.stem for path in icons} != set(names):
        p.error('toolkit source icons do not match the 24 class names')
    declaration = json.loads(plan_path.read_text(encoding='utf-8'))
    validate_plan(declaration, names, root, args.width, args.height, args.vehicle_scale)
    code = [__file__, root / 'tools/gen_dataset.py', root / 'tools/matting.py']
    params = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    with generation_guard('waste_stickers:' + out.relative_to(WORKSPACE).as_posix(),
             sorted(icons | configs | {plan_path}), [out], params, code_paths=code,
             procedural_inputs=sorted(configs | {plan_path}), root=WORKSPACE, production_plan=declaration,
             sample_output_paths=[out / 'images'], auxiliary_output_paths=[out / 'preview']) as run:
        if run.reused:
            print('Existing complete output reused:', out)
            return
        return _generate(args, root, cache, out, run, names)


if __name__ == '__main__':
    main()
