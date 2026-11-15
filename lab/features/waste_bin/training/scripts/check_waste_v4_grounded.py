"""Check a provisional grounded waste candidate without treating it as a test result."""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]


def _generate():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', help='dataset path relative to workspace root')
    args = parser.parse_args()
    dataset = (ROOT / args.dataset).resolve()
    manifest = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
    errors, counts, quantities, strata = [], Counter(), Counter(), Counter()
    signatures, subsets = defaultdict(set), defaultdict(set)
    backgrounds, render_frames, negative_parents = defaultdict(set), defaultdict(set), defaultdict(set)
    for record in manifest['records']:
        key, split = record['id'], record['split']
        counts[(split, record['sticker_state'])] += 1
        quantities[(split, len(record['bin_states']))] += 1
        colors = ','.join(c for c in ('blue', 'red', 'green', 'gray') if c in record['bin_states']) or 'none'
        states = ','.join(c+':'+record['bin_states'][c] for c in ('blue', 'red', 'green', 'gray')
                          if c in record['bin_states']) or 'none'
        tone = 'dark_low_saturation' if record['tone']['value_gain'] < .9 else 'normal'
        strata[(split, str(len(record['bin_states'])), colors, states, tone)] += 1
        signatures[(split, str(len(record['bin_states'])))].add(states)
        subsets[(split, str(len(record['bin_states'])))].add(colors)
        bg = record['background']
        backgrounds[split].add(bg['parent_image'])
        if record.get('source_render'):
            render_frames[split].add(record['source_render'])
        else:
            if bg['parent_image'] in negative_parents[split]:
                errors.append(f'{key}: repeated pure-negative parent')
            negative_parents[split].add(bg['parent_image'])
        if record['bin_states']:
            mask = np.zeros((480, 640), dtype=np.uint8)
            cv2.fillPoly(mask, [np.asarray(bg['ground_polygon_xy'], dtype=np.int32)], 1)
            mask = cv2.erode(mask, np.ones((11, 11), dtype=np.uint8))
            if len(record.get('ground_support_xy', {})) != len(record['bin_states']):
                errors.append(f'{key}: missing bin support points')
            for color, points in record.get('ground_support_xy', {}).items():
                if len(points) != 3 or any(not (0 <= x < 640 and 0 <= y < 480 and mask[y, x]) for x, y in points):
                    errors.append(f'{key}: support outside floor for {color}')
        if record['sticker_state'] == 'partial' and (record['image_sticker'] or record['label_sticker']):
            errors.append(f'{key}: partial sticker truth included as complete')
        if record['sticker_state'] != 'partial' and any(s['readability'] == 'unknown' for s in record['sticker_candidates']):
            errors.append(f'{key}: unknown sticker in complete sample')
        state_image = None
        for task, class_count in (('state', 2), ('sticker', 24)):
            image_path, label_path = record['image_'+task], record['label_'+task]
            if image_path is None and label_path is None:
                continue
            if not image_path or not label_path:
                errors.append(f'{key}: incomplete {task} pair')
                continue
            try:
                image = cv2.imdecode(np.fromfile(str(ROOT / image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None or image.shape != (480, 640, 3):
                    errors.append(f'{key}: bad {task} image')
                elif task == 'state':
                    state_image = image
                elif state_image is not None and not np.array_equal(image, state_image):
                    errors.append(f'{key}: state and sticker images differ')
                rows = (ROOT / label_path).read_text(encoding='utf-8').splitlines()
                expected = len(record['bin_states']) if task == 'state' else 0
                if task == 'state' and len(rows) != expected:
                    errors.append(f'{key}: expected {expected} state boxes')
                if task == 'sticker' and len(rows) != len(record['sticker_candidates']):
                    errors.append(f'{key}: sticker count mismatch')
                actual_classes = []
                for row in rows:
                    fields = row.split()
                    cls, (x, y, w, h) = int(fields[0]), map(float, fields[1:])
                    actual_classes.append(cls)
                    if not (len(fields) == 5 and 0 <= cls < class_count and 0 < w <= 1 and 0 < h <= 1 and
                            -1e-7 <= x-w/2 and x+w/2 <= 1+1e-7 and
                            -1e-7 <= y-h/2 and y+h/2 <= 1+1e-7):
                        errors.append(f'{key}: invalid {task} box {row}')
                if task == 'state':
                    if set(record['bin_states'].values()) - {'open', 'closed'}:
                        errors.append(f'{key}: unknown state entered complete state labels')
                    expected_classes = [0 if value == 'open' else 1 for value in record['bin_states'].values()]
                    if sorted(actual_classes) != sorted(expected_classes):
                        errors.append(f'{key}: state class labels disagree with rendered bin states')
                elif sorted(actual_classes) != sorted(item['class_id'] for item in record['sticker_candidates']):
                    errors.append(f'{key}: sticker classes disagree with readable candidates')
            except (OSError, ValueError, TypeError) as exc:
                errors.append(f'{key}: {task} pair read error: {exc}')
    for kind, groups in (('background parent', backgrounds), ('render frame', render_frames)):
        for first, second in (('train', 'val'), ('train', 'test'), ('val', 'test')):
            overlap = groups[first] & groups[second]
            if overlap:
                errors.append(f'{kind} reused across {first}/{second}: {len(overlap)}')
    if manifest.get('render_source_directories') and manifest.get('quantity_quota_achieved', True):
        for split, total in (('train', 800), ('val', 100), ('test', 100)):
            for amount, percent in ((4, 40), (3, 25), (2, 20), (1, 10), (0, 5)):
                if quantities[(split, amount)] != total * percent // 100:
                    errors.append(f'{split}: quantity {amount} did not meet its assumed quota')
            for amount, required in ((4, 1), (3, 4), (2, 6), (1, 4)):
                if len(subsets[(split, str(amount))]) != required:
                    errors.append(f'{split}: incomplete color subsets for {amount} bins')
            if len(signatures[(split, '4')]) != 16:
                errors.append(f'{split}: fewer than 16 visible four-bin open/closed combinations')
    result = {'status': 'structural_check_pass' if not errors else 'structural_check_failed',
              'records': len(manifest['records']), 'counts': {'|'.join(k): v for k, v in counts.items()},
              'quantity_counts': {'|'.join((k[0], str(k[1]))): v for k, v in quantities.items()},
              'quantity_color_state_tone': {'|'.join(k): v for k, v in strata.items()},
              'distinct_visible_color_state_combinations': {'|'.join(k): len(v) for k, v in signatures.items()},
              'distinct_color_subsets': {'|'.join(k): sorted(v) for k, v in subsets.items()},
              'distinct_backgrounds': {k: len(v) for k, v in backgrounds.items()},
              'distinct_render_frames': {k: len(v) for k, v in render_frames.items()},
              'negative_backgrounds': {k: len(v) for k, v in negative_parents.items()},
              'scope': 'structural and ground-foot checks only; no photorealism, truth, accuracy, or independence claim',
              'errors': errors}
    (dataset / 'structural_check.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
    return bool(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', help='dataset path relative to workspace root')
    args = parser.parse_args()
    dataset = (ROOT / args.dataset).resolve()
    manifest_path = dataset / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    media, procedural = set(), {manifest_path}
    for record in manifest['records']:
        for task in ('state', 'sticker'):
            image, label = record['image_'+task], record['label_'+task]
            if image and label:
                media.add(ROOT / image)
                procedural.add(ROOT / label)
    output = dataset / 'structural_check.json'
    sys.path.insert(0, str(ROOT))
    from lab.training_common.production_guard import generation_guard
    with generation_guard('waste_bin.check_grounded:' + str(dataset.relative_to(ROOT)),
                          sorted(media), [output], {}, code_paths=[Path(__file__).resolve()],
                          procedural_inputs=sorted(procedural), root=ROOT) as run:
        if not run.reused:
            _generate()
    if json.loads(output.read_text(encoding='utf-8'))['status'] != 'structural_check_pass':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
