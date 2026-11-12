"""Generate training-only sticker detection data using the supplied toolkit.

The toolkit is external raw material; no synthetic split is an independent evaluation.
"""
import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--toolkit', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cache', type=Path, default=Path(__file__).resolve().parents[5] / 'datasets/exports/waste_bin/.cache/assets')
    p.add_argument('--count', type=int, default=500)
    p.add_argument('--seed', type=int, default=20261106)
    args = p.parse_args()
    if args.count < 1 or args.out.exists():
        p.error('count must be positive and output must not already exist')
    root = args.toolkit.resolve()
    sys.path.insert(0, str(root / 'tools'))
    source_path = root / 'tools/gen_dataset.py'
    source = source_path.read_text(encoding='utf-8')
    old = 'order = list(range(K)) * 8'
    if source.count(old) != 1:
        raise ValueError('Unsupported toolkit version: cannot enforce one sticker per panel')
    # Each selected panel receives at most one sticker. Reject changes to the source
    # layout instead of silently falling back to its duplicate-panel assignment.
    source = source.replace(old, 'order = list(range(K))')
    gen = types.ModuleType('original_generator')
    gen.__file__ = str(source_path)
    exec(compile(source, str(source_path), 'exec'), gen.__dict__)
    cfg = yaml.safe_load((root / 'configs/gen_default.yaml').read_text(encoding='utf-8'))
    classes = yaml.safe_load((root / 'configs/classes.yaml').read_text(encoding='utf-8'))
    names = [item['name'] for item in classes['items']]
    if len(names) != 24 or len(set(names)) != 24:
        raise ValueError('Expected 24 distinct source icons')
    # Do not render unlabelled target-like icons or white sticker-like distractors.
    cfg['negative']['distractors']['kinds'] = ['stain', 'scuff', 'tape']
    cfg['negative']['distractors']['kinds_single'] = ['stain', 'scuff', 'tape']
    cache = args.cache.resolve()
    metas = gen.prep_assets(str(root / cfg['paths']['icons']), str(cache))
    if {m['name'] for m in metas} != set(names):
        raise ValueError('Missing or unexpected source icons')
    assets = []
    for m in metas:
        with np.load(cache / m['file']) as z:
            assets.append(dict(name=m['name'], cls=m['cls'], mode=m['mode'], rgb=z['rgb'], alpha=z['alpha'].astype(np.float32)))
    out = args.out.resolve()
    for sub in ('images/train', 'labels/train', 'metadata/train', 'preview'):
        (out / sub).mkdir(parents=True)
    manifest = []
    rejected = 0
    for i in range(args.count):
        # Mix single and four-panel views, not a claim of real bin appearance.
        cfg['scene']['layout']['mode'] = 'single' if i % 2 == 0 else 'row'
        cfg['scene']['n_panels'] = {'values': [4], 'probs': [1.0]}
        gen._init_worker(cfg, assets)
        rng = np.random.default_rng(args.seed + i * 7919)
        neg = rng.random() < 0.15
        n = 0 if neg else (1 if i % 2 == 0 else int(rng.integers(2, 5)))
        items = rng.choice(names, size=n, replace=False).tolist()
        job = dict(idx=i, seed=args.seed + i * 7919, items=items, n_stickers=n, negative=neg)
        result = gen.gen_one(job)
        # Discard the whole image if a rendered sticker was omitted by minimum-size filtering.
        if len(result['names']) != result['n'] or result['n'] != n:
            rejected += 1
            continue
        stem = f'syn_{i:06d}'
        if not gen.imwrite_u(str(out / 'images/train' / (stem + '.jpg')), result['img'], 90):
            raise RuntimeError('Image write failed')
        lines = []
        for label, inst in zip(result['det'], result['inst']):
            parts = label.split()
            parts[0] = str(names.index(inst['name']))
            lines.append(' '.join(parts))
        (out / 'labels/train' / (stem + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
        record = dict(image=f'images/train/{stem}.jpg', source='synthetic', seed=job['seed'], layout=cfg['scene']['layout']['mode'], requested=items, instances=result['inst'], distractor=result['distractor'])
        (out / 'metadata/train' / (stem + '.json')).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        manifest.append(record)
        if len(manifest) <= 12:
            vis = result['img'].copy()
            for inst in result['inst']:
                x1, y1, x2, y2 = map(int, inst['xyxy'])
                gen.cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            gen.imwrite_u(str(out / 'preview' / (stem + '.jpg')), vis, 90)
        if (i + 1) % 50 == 0:
            print(f'{i+1}/{args.count}, accepted={len(manifest)}, rejected={rejected}', flush=True)
    (out / 'manifest.json').write_text(json.dumps({'purpose': 'synthetic training only; no independent validation', 'attempted': args.count, 'accepted': len(manifest), 'rejected': rejected, 'images': manifest}, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'classes.json').write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'accepted={len(manifest)} rejected={rejected}; no validation set generated')


if __name__ == '__main__':
    main()
