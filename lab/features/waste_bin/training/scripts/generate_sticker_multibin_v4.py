"""Generate multi-bin sticker candidates using the existing grounded pipeline."""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/waste_bin_state or waste_sticker/. Production remains paused.")

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import composite_bin_real_scene as base
import generate_sticker_closeups_v4 as closeups
import generate_waste_v4_grouped as grouped

ROOT = base.ROOT
CLASSES = (4, 7, 10, 18)


def _generate():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--count', type=int, default=120)
    parser.add_argument('--seed', type=int, default=20261119)
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    if not out.is_relative_to((ROOT / 'datasets/exports').resolve()) or not out.name.startswith('waste_sticker_') or out.exists():
        parser.error('output must be a new waste_sticker directory under datasets/exports')
    if not 1 <= args.count <= 1000:
        parser.error('count must be 1–1000')
    cv2.setNumThreads(1)
    names, icons, _, _ = base.load_icons()
    _, backgrounds = closeups.load_inputs()
    paths = [ROOT / 'datasets/exports/waste_bin/blender_quantity_candidates_160_20261112',
             ROOT / 'datasets/exports/waste_bin/blender_quantity_val_trajectory_20261112',
             ROOT / 'datasets/exports/waste_bin/blender_quantity_test_full_trajectory_20261112']
    pools = grouped.load_quantity_sources(paths)
    retained = json.loads((ROOT / 'datasets/exports/waste_bin/v4_quantity_isolated_20261112/manifest.json').read_text(encoding='utf-8'))
    allowed = {r['source_render'] for r in retained['records'] if r['split'] == 'train' and r.get('source_render')}
    sources = {n: [s for s in pools[('train', n)] if s['source_directory'] + '/' + s['name'] in allowed]
               for n in (2, 3, 4)}
    if any(not sources[n] for n in sources):
        raise ValueError('missing retained training renders')
    rng = random.Random(args.seed)
    color_counts = Counter()
    records = []
    out.mkdir(parents=True)
    for sub in ('images/train', 'labels/train'):
        (out / sub).mkdir(parents=True)
    try:
        for index in range(args.count):
            quantity = (2, 3, 4)[index % 3]
            dark = index % 2 == 1
            for attempt in range(120):
                source = rng.choice(sources[quantity])
                crop = source['image'].copy()
                masks = source['masks']
                choices = list(masks)
                rng.shuffle(choices)
                chosen = choices[:quantity-1]
                stickers = []
                for color in chosen:
                    order = list(CLASSES)
                    rng.shuffle(order)
                    order.sort(key=lambda c: color_counts[(c, color)])
                    cls = order[0]
                    paper = base.paper_on_bin(crop, masks[color], icons[cls], rng)
                    if paper is None:
                        break
                    stickers.append((cls, color, paper))
                if len(stickers) != quantity-1:
                    continue
                try:
                    bg, background, floor, left, top, width, height, scaled, supports = grouped.ground_placement(
                        crop, masks, backgrounds['train'], rng, height_range=(150, 220))
                except ValueError:
                    continue
                boxes = []
                for cls, color, (x1, y1, x2, y2) in stickers:
                    box = [round(left+x1*width/crop.shape[1]), round(top+y1*height/crop.shape[0]),
                           round(left+x2*width/crop.shape[1]), round(top+y2*height/crop.shape[0])]
                    boxes.append((cls, color, box))
                if any(min(b[2]-b[0], b[3]-b[1]) < 26 for _, _, b in boxes):
                    continue
                break
            else:
                raise RuntimeError(f'could not place {quantity} bins after 120 attempts')
            frame = grouped.add_contact_shadows(background, floor, scaled, supports)
            union = np.logical_or.reduce(list(masks.values())).astype(np.uint8) * 255
            alpha = cv2.resize(union, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            patch = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)
            region = frame[top:top+height, left:left+width].astype(np.float32)
            frame[top:top+height, left:left+width] = np.clip(patch*alpha[:, :, None] + region*(1-alpha[:, :, None]), 0, 255).astype(np.uint8)
            value, saturation = (0.72, 0.65) if dark else (0.96, 0.95)
            frame = grouped.apply_tone(frame, value_gain=value, saturation_gain=saturation)
            stem = f'train_{index:05d}'
            image, label = out/'images/train'/(stem+'.png'), out/'labels/train'/(stem+'.txt')
            grouped.write_png(image, frame)
            label.write_text(''.join(grouped.box_line(cls, box)+'\n' for cls, _, box in boxes), encoding='utf-8')
            bins = []
            for color, mask in scaled.items():
                x, y, w, h = cv2.boundingRect(mask)
                match = next(({'class_id': cls, 'name': names[cls], 'bbox_xyxy': box,
                               'paper_size_pixels': [box[2]-box[0], box[3]-box[1]],
                               'visibility': 'candidate_visible'} for cls, item_color, box in boxes if item_color == color), None)
                bins.append({'color': color, 'state': source['states'][color], 'bbox_xyxy': [left+x, top+y, left+x+w, top+y+h],
                             'item': match, 'fixed_symbol_visibility': 'not separately measured; item may cover part of symbol'})
            for cls, color, _ in boxes:
                color_counts[(cls, color)] += 1
            records.append({'id': stem, 'split': 'train', 'image': grouped.rel(image), 'label': grouped.rel(label),
                'source_render': source['source_directory']+'/'+source['name'], 'source_group': source['group'],
                'source_trajectory': source['trajectory'], 'background': bg, 'generation_method': 'real_scene_composite',
                'placement_xywh': [left, top, width, height], 'ground_support_xy': supports,
                'tone': {'value_gain': value, 'saturation_gain': saturation, 'status': 'simulated_not_camera_calibration'},
                'bins': bins, 'attempt': attempt, 'status': 'candidate_semantic_review_pending'})
        report = {'status': 'candidate_semantic_review_pending', 'seed': args.seed, 'images': len(records),
            'scope': 'synthetic bins composited on real-scene backgrounds; train only; all share a Blender mother scene, source icons and video',
            'selection': '2–4 visible bins, one kept without item sticker; classes carton, vegetable leaf, bread and cigarette; paper bbox >=26px',
            'class_color_counts': {'|'.join(map(str, k)): v for k, v in sorted(color_counts.items())}, 'records': records}
        (out/'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(json.dumps({'images': len(records), 'class_color_counts': report['class_color_counts']}, ensure_ascii=False))
    except BaseException:
        (out/'FAILED.txt').write_text('Generation stopped; inspect output before reuse.\n', encoding='utf-8')
        raise


def main():
    raise RuntimeError("Retired dataset entry; use datasets/scripts/waste_sticker with source.yaml")


if __name__ == '__main__':
    main()
