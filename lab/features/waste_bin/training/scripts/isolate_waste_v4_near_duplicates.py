"""Isolate cross-split near-duplicate source groups in a provisional candidate set.

This is conservative screening, not proof that the retained images are independent.
The source batch is preserved and a new filtered batch is created.
"""
import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
DESTINATION = ROOT / 'datasets/exports/waste_bin'


def _generate():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--mean-difference', type=float, default=6.0)
    args = parser.parse_args()
    source, out = (ROOT / args.source).resolve(), (ROOT / args.out).resolve()
    if not source.is_relative_to(DESTINATION.resolve()) or not out.is_relative_to(DESTINATION.resolve()) or out.exists():
        parser.error('source and new output must be under datasets/exports/waste_bin')
    if args.mean_difference <= 0:
        parser.error('mean-difference must be positive')
    original = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    nodes, record_nodes = {}, {}
    for record in original['records']:
        group = []
        if record.get('source_render'):
            key = ('bin_render', record['source_render'])
            signature = tuple(sorted(record['bin_states'].items()))
            nodes.setdefault(key, {'split': record['split'], 'signature': signature})
            group.append(key)
        bg = record['background']
        role = 'empty_render' if record['generation_method'] == 'blender_render' else 'real_background'
        key = (role, bg['image'])
        nodes.setdefault(key, {'split': record['split'], 'signature': ()})
        group.append(key)
        for node in group:
            if nodes[node]['split'] != record['split']:
                raise ValueError('identical source image used across splits: ' + node[1])
        record_nodes[record['id']] = group
    parent = {key: key for key in nodes}

    def find(key):
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    buckets = defaultdict(list)
    for key, item in nodes.items():
        image = cv2.imdecode(np.fromfile(str(ROOT / key[1]), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError('missing image ' + key[1])
        item['gray'] = cv2.resize(image, (80, 60), interpolation=cv2.INTER_AREA).astype(np.int16)
        buckets[(key[0], item['signature'])].append(key)
    links = []
    for keys in buckets.values():
        for index, left in enumerate(keys):
            for right in keys[index+1:]:
                distance = float(np.mean(np.abs(nodes[left]['gray'] - nodes[right]['gray'])))
                if distance <= args.mean_difference:
                    union(left, right)
                    if nodes[left]['split'] != nodes[right]['split']:
                        links.append({'left': list(left), 'right': list(right), 'mean_difference': distance})
    components = defaultdict(list)
    for key in nodes:
        components[find(key)].append(key)
    priority = {'train': 0, 'val': 1, 'test': 2}
    blocked, cross_components = {}, []
    for members in components.values():
        splits = {nodes[node]['split'] for node in members}
        if len(splits) < 2:
            continue
        kept_split = min(splits, key=priority.get)
        cross_components.append({'keep_split': kept_split,
            'members': [{'role': node[0], 'path': node[1], 'split': nodes[node]['split']} for node in members]})
        for node in members:
            if nodes[node]['split'] != kept_split:
                blocked[node] = kept_split
    retained, isolated = [], []
    for record in original['records']:
        reasons = [node for node in record_nodes[record['id']] if node in blocked]
        if reasons:
            isolated.append({'id': record['id'], 'split': record['split'],
                             'source_nodes': [{'role': n[0], 'path': n[1], 'kept_split': blocked[n]} for n in reasons]})
        else:
            retained.append(record.copy())
    out.mkdir(parents=True)
    for task in ('state', 'sticker'):
        for split in ('train', 'val', 'test'):
            for folder in ('images', 'labels'):
                (out / task / folder / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / task / 'data.yaml', out / task / 'data.yaml')
    shutil.copy2(source / 'classes.json', out / 'classes.json')
    for record in retained:
        for field in ('image_state', 'label_state', 'image_sticker', 'label_sticker'):
            if record[field]:
                old = (ROOT / record[field]).resolve()
                new = out / old.relative_to(source)
                shutil.copy2(old, new)
                record[field] = new.relative_to(ROOT).as_posix()
    counts, quantities, scenarios, class_colors = Counter(), Counter(), Counter(), Counter()
    for r in retained:
        counts[(r['split'], r['sticker_state'])] += 1
        quantities[(r['split'], str(len(r['bin_states'])))] += 1
        if r.get('state_code'):
            scenarios[(r['split'], r['state_code'], 'dark' if r['tone']['value_gain'] < .9 else 'normal')] += 1
        for item in r['sticker_candidates']:
            class_colors[(item['name'], item['bin_color'])] += 1
    result = dict(original)
    result.update({'status': 'draft_near_duplicate_screened_not_deliverable',
        'near_duplicate_isolation': {'original_batch': source.relative_to(ROOT).as_posix(),
            'screen': '80x60 grayscale mean absolute difference <= threshold; same visible bin colors/states; source graph connected components',
            'threshold': args.mean_difference, 'priority': ['train', 'val', 'test'],
            'limitations': 'heuristic, may miss visual duplicates and over-isolate; all Blender frames reuse one scene',
            'cross_split_components': len(cross_components), 'cross_split_links': len(links),
            'isolated_records': len(isolated)},
        'quantity_quota_achieved': False, 'records': retained,
        'counts': {'|'.join(k): v for k, v in counts.items()},
        'quantity_counts': {'|'.join(k): v for k, v in quantities.items()},
        'state_scenarios': {'|'.join(k): v for k, v in scenarios.items()},
        'class_color_instances': {'|'.join(k): v for k, v in class_colors.items()}})
    (out / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (out / 'isolation.json').write_text(json.dumps({'isolated_records': isolated, 'cross_split_components': cross_components,
        'cross_split_links': links}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'retained': len(retained), 'isolated': len(isolated),
        'quantity_counts': result['quantity_counts'], 'counts': result['counts'],
        'cross_split_components': len(cross_components), 'cross_split_links': len(links)}, ensure_ascii=False))


def _guard_inputs(source):
    manifest = source / 'manifest.json'
    data = json.loads(manifest.read_text(encoding='utf8'))
    media, procedural = set(), {manifest, source / 'classes.json',
                                 source / 'state/data.yaml', source / 'sticker/data.yaml'}
    for row in data['records']:
        for field in ('image_state', 'image_sticker', 'source_render'):
            if row.get(field):
                media.add(ROOT / row[field])
        for field in ('label_state', 'label_sticker'):
            if row.get(field):
                procedural.add(ROOT / row[field])
        background = row['background']
        for field in ('image', 'parent_image'):
            if background.get(field):
                media.add(ROOT / background[field])
    return sorted(media), sorted(procedural)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--mean-difference', type=float, default=6.0)
    args, _ = parser.parse_known_args()
    source, out = (ROOT / args.source).resolve(), (ROOT / args.out).resolve()
    media, procedural = _guard_inputs(source)
    sys.path.insert(0, str(ROOT))
    from lab.training_common.production_guard import generation_guard
    with generation_guard('waste_bin.isolate_duplicates:' + str(out.relative_to(ROOT)),
                          media, [out], {'argv': sys.argv[1:]}, code_paths=[Path(__file__).resolve()],
                          procedural_inputs=procedural, root=ROOT) as run:
        if run.reused:
            return
        _generate()


if __name__ == '__main__':
    main()
