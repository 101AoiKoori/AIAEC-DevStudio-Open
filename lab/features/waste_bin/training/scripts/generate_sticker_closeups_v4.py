"""Make supplemental close-view sticker candidates from existing grouped renders.

The underlying bin was rendered in Blender with its recorded visibility. A
bin is cropped only when its bounds contain no pixels of any other bin. Partial
truth is saved separately and never written as an empty training label.
"""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/waste_bin_state or waste_sticker/.")

import argparse
import itertools
import json
import random
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np
import yaml

import composite_bin_real_scene as base
import generate_waste_v4_grouped as grouped

ROOT = base.ROOT
DESTINATION = ROOT / 'datasets/exports/waste_bin'
SOURCE = DESTINATION / 'v4_quantity_isolated_20261112'
QUANTITY_PATHS = [DESTINATION/'blender_quantity_candidates_160_20261112',
                  DESTINATION/'blender_quantity_val_trajectory_20261112',
                  DESTINATION/'blender_quantity_test_full_trajectory_20261112']
VISIBLE = (0, 2, 11, 23)
UNKNOWN = (13, 14, 19, 20)
COLORS = tuple(base.COLORS)
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard


def guard_sources(quantity_paths, source_manifest):
    """Resolve the files actually loaded by the quantity and icon helpers."""
    classes = yaml.safe_load(base.CLASSES.read_text(encoding='utf-8'))
    names = [item['name'] for item in classes['items']]
    procedural = {base.CLASSES, ROOT / 'config/perception_tasks.json', source_manifest}
    procedural.update(base.CACHE / (name + '.npz') for name in names)
    media = {ROOT / 'Archive/Assets/working_sources/digital/waste_sticker/toolkit/SourcePic' / (name + '.png') for name in names}
    for index, path in enumerate(quantity_paths):
        manifest = path / 'manifest.json'
        procedural.add(manifest)
        report = json.loads(manifest.read_text(encoding='utf-8'))
        for sample in report['samples']:
            group = sample['group']
            included = ((index == 0 and group < 8) or (index == 1 and 8 <= group <= 12)
                        or (index == 2 and 16 <= group <= 20))
            if sample['candidate_label'] and included:
                media.update((path / sample['image'], path / sample['mask_visible']))
    report = json.loads(source_manifest.read_text(encoding='utf-8'))
    for item in report['records']:
        if item['generation_method'] != 'blender_render':
            media.add(ROOT / item['background']['image'])
    return media, procedural


def single_bin_crop(source, color):
    target = source['masks'][color].astype(np.uint8)
    x, y, w, h = cv2.boundingRect(target)
    # Reject overlapping projected rectangles: an obscured shape cannot be
    # reconstructed by erasing its neighbour.
    for other, mask in source['masks'].items():
        if other == color:
            continue
        ox, oy, ow, oh = cv2.boundingRect(mask.astype(np.uint8))
        if x < ox+ow and ox < x+w and y < oy+oh and oy < y+h:
            return None
    ux, uy, _, _ = source['source_union_xywh']
    return source['image'][y:y+h, x:x+w].copy(), {color: target[y:y+h, x:x+w]}, [ux+x, uy+y, w, h]


def load_inputs():
    source_records = json.loads((SOURCE/'manifest.json').read_text(encoding='utf-8'))['records']
    allowed = defaultdict(set)
    backgrounds = defaultdict(dict)
    for item in source_records:
        split = item['split']
        if item.get('source_render'):
            allowed[split].add(item['source_render'])
        bg = item['background']
        if item['generation_method'] != 'blender_render':
            backgrounds[split][bg['parent_image']] = bg
    foregrounds = defaultdict(list)
    pools = grouped.load_quantity_sources(QUANTITY_PATHS)
    for (split, _), entries in pools.items():
        for entry in entries:
            render = entry['source_directory']+'/'+entry['name']
            if render not in allowed[split]:
                continue
            for color in entry['masks']:
                cropped = single_bin_crop(entry, color)
                if cropped:
                    foregrounds[(split, color)].append((entry, render, cropped))
    bg_pools = defaultdict(list)
    for split, entries in backgrounds.items():
        for item in entries.values():
            image = base.read_image(ROOT/item['image'])
            floor = np.zeros((480,640), dtype=np.uint8)
            cv2.fillPoly(floor, [np.asarray(item['ground_polygon_xy'], dtype=np.int32)], 1)
            floor = cv2.erode(floor, np.ones((11,11), np.uint8))
            bg_pools[split].append((item, image, floor))
    for split, color in itertools.product(('train','val','test'), COLORS):
        if not foregrounds[(split, color)] or not bg_pools[split]:
            raise ValueError('missing retained source for '+split+' '+color)
    return foregrounds, bg_pools


def screen_foregrounds(foregrounds, threshold=6.0):
    """Conservatively group near-identical normalized bin crops across splits."""
    nodes = []
    for (split, color), entries in foregrounds.items():
        for entry in entries:
            src, render, (crop, masks, _) = entry
            mask = masks[color].astype(bool)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray[~mask] = 128
            image = cv2.resize(gray, (64,80), interpolation=cv2.INTER_AREA).astype(np.int16)
            nodes.append({'split': split, 'color':color, 'state':src['states'][color],
                          'render':render, 'entry':entry, 'image':image})
    parent = list(range(len(nodes)))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    def union(first, second):
        first, second = find(first), find(second)
        if first != second:
            parent[second] = first
    buckets = defaultdict(list)
    for index, node in enumerate(nodes):
        buckets[(node['color'],node['state'])].append(index)
    links = 0
    for indices in buckets.values():
        for i, first in enumerate(indices):
            for second in indices[i+1:]:
                delta = float(np.mean(np.abs(nodes[first]['image']-nodes[second]['image'])))
                if delta <= threshold:
                    union(first,second)
                    links += int(nodes[first]['split'] != nodes[second]['split'])
    groups = defaultdict(list)
    for index in range(len(nodes)):
        groups[find(index)].append(index)
    priority = {'train':0,'val':1,'test':2}
    retained, blocked = defaultdict(list), []
    for indices in groups.values():
        keep = min((nodes[i]['split'] for i in indices), key=priority.get)
        for i in indices:
            item = nodes[i]
            if item['split'] == keep:
                retained[(item['split'], item['color'])].append(item['entry'])
            else:
                blocked.append({'split':item['split'], 'color':item['color'], 'state':item['state'],
                                'source_render':item['render'], 'group_kept_in':keep})
    return retained, blocked, links


def _generate():
    cv2.setNumThreads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=20261122)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    out = (ROOT/args.out).resolve()
    if not out.is_relative_to(DESTINATION.resolve()) or out == DESTINATION.resolve() or out.exists():
        parser.error('out must be a new directory under datasets/exports/waste_bin')
    names, icons, _, _ = base.load_icons()
    sources, backgrounds = load_inputs()
    screened, blocked, links = screen_foregrounds(sources)
    print('foreground screening:', json.dumps({'candidate_counts':{'|'.join(k):len(v) for k,v in sources.items()},
        'retained_counts':{'|'.join(k):len(v) for k,v in screened.items()},
        'blocked':len(blocked), 'cross_split_links':links},ensure_ascii=False))
    if args.audit_only:
        return
    for split, color in itertools.product(('train','val','test'), COLORS):
        if not screened[(split,color)]:
            raise ValueError('foreground screening removed all sources for '+split+' '+color)
    sources = screened
    rng = random.Random(args.seed)
    records, counts = [], Counter()
    if args.pilot:
        jobs = [('train', cls, COLORS[i % 4], dark, 0)
                for i, cls in enumerate(VISIBLE+UNKNOWN) for dark in (False, True)]
    else:
        jobs = [(split, cls, color, dark, rep)
                for split, repeats in (('train',3), ('val',1), ('test',1))
                for cls in VISIBLE for color in COLORS for dark in (False,True) for rep in range(repeats)]
        jobs += [(split, cls, color, dark, 0) for split in ('train','val','test')
                 for cls in UNKNOWN for color in COLORS for dark in (False,True)]
    try:
        for index, (split, cls, color, dark, repeat) in enumerate(jobs):
            for attempt in range(60):
                source, render, (crop, masks, crop_xywh) = rng.choice(sources[(split,color)])
                placed = grouped.ground_placement(crop, masks, backgrounds[split], rng, height_range=(250, 335))
                bg, background, floor, left, top, width, height, scaled_masks, supports = placed
                patch = cv2.resize(crop, (width,height), interpolation=cv2.INTER_LINEAR)
                paper = base.paper_on_bin(patch, scaled_masks[color], icons[cls], rng)
                if paper and 46 <= min(paper[2]-paper[0], paper[3]-paper[1]) <= 70:
                    break
            else:
                raise ValueError(f'Could not create sufficiently large sticker: {split} {color} {cls}')
            frame = grouped.add_contact_shadows(background, floor, scaled_masks, supports)
            mask = scaled_masks[color].astype(np.float32)
            target = frame[top:top+height,left:left+width].astype(np.float32)
            frame[top:top+height,left:left+width] = np.clip(patch*mask[:,:,None] + target*(1-mask[:,:,None]),0,255).astype(np.uint8)
            frame = grouped.apply_tone(frame, value_gain=.72 if dark else .96,
                                       saturation_gain=.65 if dark else .95)
            frame = cv2.GaussianBlur(frame,(3,3),.35)
            box = [paper[0]+left, paper[1]+top, paper[2]+left, paper[3]+top]
            known = cls in VISIBLE
            folder = 'images' if known else 'partial'
            image = out/folder/split/f'{split}_{index:05d}.png'
            image.parent.mkdir(parents=True, exist_ok=True)
            grouped.write_png(image,frame)
            label = None
            if known:
                label = out/'labels'/split/(image.stem+'.txt')
                label.parent.mkdir(parents=True, exist_ok=True)
                label.write_text(grouped.box_line(cls,box)+'\n',encoding='utf-8')
            records.append({'id': image.stem, 'split': split, 'image': grouped.rel(image),
                'label': grouped.rel(label) if label else None, 'status': 'candidate_visible' if known else 'partial_unknown',
                'icon_class_candidate': cls, 'icon_name': names[cls], 'bin_color': color,
                'bin_state': source['states'][color], 'source_bin_states': source['states'],
                'source_render': render, 'source_crop_xywh': crop_xywh,
                'source_trajectory': source['trajectory'], 'background': bg,
                'bbox_xyxy': box, 'paper_size_pixels': [box[2]-box[0],box[3]-box[1]],
                'bin_size_pixels': [width,height], 'ground_support_xy': supports,
                'generation_method': 'blender_render_crop_plus_real_scene_composite',
                'tone': {'value_gain': .72 if dark else .96, 'saturation_gain': .65 if dark else .95,
                         'status': 'simulated_not_camera_calibration'},
                'unknown_reason': None if known else 'image cannot establish full class semantics at this size',
                'status_note': 'synthetic close view; not a new camera capture or confirmed deployment scale'})
            counts[(split, str(cls), color, 'dark' if dark else 'normal', records[-1]['status'])] += 1
        for split in ('train','val','test'):
            for folder in ('images','labels','partial'):
                (out/folder/split).mkdir(parents=True,exist_ok=True)
        (out/'data.yaml').write_text(yaml.safe_dump({'names':dict(enumerate(names)),
            **{s:'images/'+s for s in ('train','val','test')}}, allow_unicode=True, sort_keys=False),encoding='utf-8')
        report = {'status':'draft_source_review_required', 'seed':args.seed, 'pilot':args.pilot,
            'source_base':grouped.rel(SOURCE), 'source_reuse':'same Blender mother scene and source sticker artwork; backgrounds from one video',
            'foreground_screen': {'method':'64x80 masked grayscale crop difference <=6/255; same color and state; connected groups kept train, then val, then test',
                'blocked_source_crops':len(blocked), 'cross_split_links':links,
                'retained_source_crops':{'|'.join(k):len(v) for k,v in screened.items()},
                'scope':'heuristic, not proof of independence; groups share a Blender mother scene'},
            'reviewed_visible_classes':list(VISIBLE), 'unknown_classes':list(UNKNOWN),
            'measured_reference_note':'existing field subset paper 26-29px wide; this supplemental 46-70px close view is hypothetical, not field-calibrated',
            'counts':{'|'.join(k):v for k,v in counts.items()}, 'records':records}
        (out/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (out/'foreground_isolation.json').write_text(json.dumps(blocked,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'output':grouped.rel(out),'records':len(records),'status':dict(Counter(r['status'] for r in records))},ensure_ascii=False))
    except BaseException:
        out.mkdir(parents=True,exist_ok=True)
        (out/'FAILED.txt').write_text('Generation failed; inspect before reuse.\n',encoding='utf-8')
        raise


def main():
    raise RuntimeError("Retired dataset entry; use datasets/scripts task entry and its source.yaml")


if __name__ == '__main__':
    main()
