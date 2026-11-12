"""Generate synthetic sticker training and optional same-source diagnostic data."""
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


def init_worker(root, cache, out):
    global GEN, CFG, NAMES, OUT
    root, cache, OUT = Path(root), Path(cache), Path(out)
    sys.path.insert(0, str(root / 'tools'))
    source_path = root / 'tools/gen_dataset.py'
    source = source_path.read_text(encoding='utf-8')
    old = 'order = list(range(K)) * 8'
    if source.count(old) != 1:
        raise ValueError('Unsupported toolkit version: cannot enforce one sticker per panel')
    GEN = types.ModuleType('original_generator')
    GEN.__file__ = str(source_path)
    exec(compile(source.replace(old, 'order = list(range(K))'), str(source_path), 'exec'), GEN.__dict__)
    GEN.cv2.setNumThreads(1)
    CFG = yaml.safe_load((root / 'configs/gen_default.yaml').read_text(encoding='utf-8'))
    classes = yaml.safe_load((root / 'configs/classes.yaml').read_text(encoding='utf-8'))
    NAMES = [item['name'] for item in classes['items']]
    if len(NAMES) != 24 or len(set(NAMES)) != 24:
        raise ValueError('Expected 24 distinct source icons')
    CFG['negative']['distractors']['kinds'] = ['stain', 'scuff', 'tape']
    CFG['negative']['distractors']['kinds_single'] = ['stain', 'scuff', 'tape']
    # Parent prepares cache before workers start, avoiding concurrent writes.
    metas = GEN.prep_assets(str(root / CFG['paths']['icons']), str(cache))
    if {m['name'] for m in metas} != set(NAMES):
        raise ValueError('Missing or unexpected source icons')
    assets = []
    for m in metas:
        with np.load(cache / m['file']) as z:
            assets.append(dict(name=m['name'], cls=m['cls'], mode=m['mode'], rgb=z['rgb'], alpha=z['alpha'].astype(np.float32)))
    GEN._init_worker(CFG, assets)


def plan_split(count, seed, names, split):
    """15% negatives, 25% single, 15/20/25% with 2/3/4 stickers."""
    rng = np.random.default_rng(seed)
    probabilities = np.array([.15, .25, .15, .20, .25])
    raw = probabilities * count
    sizes = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - sizes))[:count - int(sizes.sum())]:
        sizes[i] += 1
    kinds = [k for k, size in enumerate(sizes) for _ in range(size)]
    rng.shuffle(kinds)
    occurrences = Counter()
    color_uses = {name: [0] * 4 for name in names}
    jobs = []
    negatives_seen = 0
    for idx, n in enumerate(kinds):
        # Greedy balancing avoids repeated classes within an image.
        order = rng.permutation(len(names)).tolist()
        order.sort(key=lambda i: occurrences[names[i]])
        items = [names[i] for i in order[:n]]
        occurrences.update(items)
        color = None
        if n == 1:
            usage = color_uses[items[0]]
            candidates = [i for i in range(4) if usage[i] == min(usage)]
            color = int(rng.choice(candidates))
            usage[color] += 1
        layout = 'single' if n == 1 or (n == 0 and negatives_seen % 2 == 0) else 'row'
        negatives_seen += n == 0
        jobs.append(dict(idx=idx, split=split, n=n, layout=layout, items=items, color=color,
                         seed=int(rng.integers(0, 2**62))))
    return jobs


def generate(job):
    split, idx, n = job['split'], job['idx'], job['n']
    CFG['scene']['layout']['mode'] = job['layout']
    CFG['scene']['n_panels'] = {'values': [4], 'probs': [1.0]}
    stem = f'{split}_{idx:06d}'
    result = None
    # Retry the same planned classes if a rendered target is omitted from labels.
    for retry in range(20):
        seed = job['seed'] + retry
        args = dict(idx=idx, seed=seed, items=job['items'], n_stickers=n, negative=n == 0)
        if job['color'] is not None:
            args['panel_color'] = job['color']
        result = GEN.gen_one(args)
        if len(result['names']) == result['n'] == n:
            break
    else:
        raise RuntimeError(f'{stem}: could not render every planned label after 20 tries')
    if not GEN.imwrite_u(str(OUT / 'images' / split / (stem + '.jpg')), result['img'], 90):
        raise RuntimeError(f'{stem}: image write failed')
    lines = []
    for label, inst in zip(result['det'], result['inst']):
        parts = label.split()
        parts[0] = str(NAMES.index(inst['name']))
        lines.append(' '.join(parts))
    (OUT / 'labels' / split / (stem + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
    record = dict(image=f'images/{split}/{stem}.jpg', source='synthetic', seed=seed,
                  retries=retry, layout=CFG['scene']['layout']['mode'], requested=job['items'],
                  single_panel_color=job['color'], instances=result['inst'], distractor=result['distractor'])
    (OUT / 'metadata' / split / (stem + '.json')).write_text(json.dumps(record, ensure_ascii=False), encoding='utf-8')
    if idx < 12:
        vis = result['img'].copy()
        for inst in result['inst']:
            x1, y1, x2, y2 = map(int, inst['xyxy'])
            GEN.cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        GEN.imwrite_u(str(OUT / 'preview' / (stem + '.jpg')), vis, 90)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--toolkit', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cache', type=Path, default=Path(__file__).resolve().parents[5] / 'datasets/exports/waste_bin/.cache/assets')
    p.add_argument('--count', type=int, default=500, help='training image count')
    p.add_argument('--val-count', type=int, default=0, help='same-source synthetic diagnostic images, not a real validation set')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=20261106)
    args = p.parse_args()
    if args.count < 1 or args.val_count < 0 or args.workers < 1 or args.out.exists():
        p.error('counts/workers must be valid and output must not already exist')
    root, cache, out = args.toolkit.resolve(), args.cache.resolve(), args.out.resolve()
    # Prepare cache once before forking/spawning worker processes.
    init_worker(root, cache, out)
    names = NAMES[:]
    splits = [('train', args.count)] + ([('val', args.val_count)] if args.val_count else [])
    jobs = []
    for number, (split, count) in enumerate(splits):
        for sub in ('images', 'labels', 'metadata'):
            (out / sub / split).mkdir(parents=True)
        jobs.extend(plan_split(count, args.seed + number * 1000003, names, split))
    (out / 'preview').mkdir()
    records = []
    with mp.get_context('spawn').Pool(args.workers, initializer=init_worker,
                                       initargs=(str(root), str(cache), str(out))) as pool:
        for record in pool.imap_unordered(generate, jobs, chunksize=2):
            records.append(record)
            if len(records) % 100 == 0 or len(records) == len(jobs):
                print(f'{len(records)}/{len(jobs)}', flush=True)
    records.sort(key=lambda r: r['image'])
    summary = {split: {'images': count,
                       'object_counts': dict(Counter(str(len(r['instances'])) for r in records if r['image'].startswith(f'images/{split}/'))),
                       'classes': dict(Counter(x['name'] for r in records if r['image'].startswith(f'images/{split}/') for x in r['instances']))}
               for split, count in splits}
    (out / 'manifest.json').write_text(json.dumps({'purpose': 'synthetic training; val is same-source diagnostic only, not real validation', 'seed': args.seed, 'summary': summary, 'images': records}, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'classes.json').write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.val_count:
        config = {'path': out.as_posix(), 'train': 'images/train', 'val': 'images/val', 'names': dict(enumerate(names))}
        (out / 'data.yaml').write_text('# Synthetic same-source diagnostic validation only.\n' + yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
