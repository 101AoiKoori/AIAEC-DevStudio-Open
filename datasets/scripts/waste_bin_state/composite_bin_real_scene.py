"""Make two separate bin-state and waste-sticker candidate sets on real ground.

Source RGB and bin masks are existing Blender renders; the background is a
reviewed empty-ground crop from one recorded scene video. Same-source variants
are candidates, not independent validation/test images. No model is trained.
"""

if __name__ == "__main__":
    raise SystemExit("Task helper is import-only; use the primary entry in datasets/scripts/waste_bin_state/. Production remains paused.")

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'datasets/exports/waste_bin/blender_state_candidates_160_20261110'
EXPORT = ROOT / 'datasets/exports/waste_bin/blender_state_yolo_candidates_160_20261110'
BASE = ROOT / 'datasets/raw/person/teammate_submission_20261110/dataset'
CACHE = ROOT / 'datasets/exports/waste_bin/.cache/assets'
CLASSES = ROOT / 'datasets/raw/waste_bin/toolkit/configs/classes.yaml'
COLORS = {'blue': (1, 0, 0), 'red': (0, 1, 0), 'green': (0, 0, 1), 'gray': (1, 1, 0)}


def read_image(path, flags=cv2.IMREAD_COLOR):
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), flags)
    if image is None:
        raise ValueError('cannot decode ' + str(path))
    return image


def write_image(path, image):
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise OSError('image encoding failed')
    encoded.tofile(str(path))


def background_sources():
    with (BASE / 'reports/backgrounds_manifest.csv').open(encoding='utf-8', newline='') as file:
        rows = list(csv.DictReader(file))
    chosen = [row for row in rows if 0 <= int(row['frame']) <= 179]
    if len(chosen) != 50:
        raise ValueError('expected the 50 previously reviewed ground-crop backgrounds')
    result = []
    for row in chosen:
        path = (BASE / 'backgrounds' / row['file']).resolve()
        if not path.is_relative_to((BASE / 'backgrounds').resolve()) or not path.is_file():
            raise ValueError('invalid background path')
        image = read_image(path)
        if image.shape[:2] != (1080, 1920):
            raise ValueError('unexpected background dimensions')
        result.append((row, image[600:1080, 480:1440].copy()))
    return result


def synthetic_backgrounds(seed):
    rng = np.random.default_rng(seed)
    result = []
    for i in range(50):
        tone = rng.uniform(60, 115)
        noise = rng.normal(0, rng.uniform(2, 6), (480, 960)).astype(np.float32)
        blur = cv2.GaussianBlur(noise, (0, 0), rng.uniform(2, 7))
        shade = np.linspace(-12, 8, 480, dtype=np.float32)[:, None]
        gray = np.clip(tone + blur * 2 + shade, 0, 255).astype(np.uint8)
        image = np.dstack([gray, gray, gray])
        if i % 2 == 0:
            cv2.line(image, (0, rng.integers(300, 470).item()),
                     (959, rng.integers(300, 470).item()), (205, 205, 205), 5)
        result.append(({'file':f'procedural_val_{i:03}', 'frame':-1}, image))
    return result


def load_sources(split):
    render = json.loads((SRC / 'manifest.json').read_text(encoding='utf-8'))
    export = json.loads((EXPORT / 'manifest.json').read_text(encoding='utf-8'))
    if render.get('status') != 'complete_candidates' or export.get('status') != 'complete':
        raise ValueError('incomplete source')
    by_name = {item['image']: item for item in render['samples']}
    selected = []
    for entry in export['included']:
        sample = by_name[entry['source_image']]
        if (sample['trajectory'] == 0) != (split == 'train'):
            continue
        image = read_image(SRC / sample['image'])
        mask = read_image(SRC / sample['mask_visible'], cv2.IMREAD_UNCHANGED)
        if image.shape != (1024, 1024, 3) or mask.shape != (1024, 1024, 4):
            raise ValueError('unexpected source shape')
        rgb = mask[:, :, [2, 1, 0]]
        masks = {}
        for name, color in COLORS.items():
            hit = mask[:, :, 3] >= 191
            for channel, value in enumerate(color):
                hit &= rgb[:, :, channel] >= 191 if value else rgb[:, :, channel] <= 63
            if not hit.any():
                raise ValueError('missing mask: ' + name)
            masks[name] = hit
        union = np.logical_or.reduce(list(masks.values()))
        x, y, w, h = cv2.boundingRect(union.astype(np.uint8))
        info = {item['bin']: item['state'] for item in sample['labels']}
        if set(info) != set(COLORS) or set(info.values()) - {'open', 'closed'}:
            raise ValueError('unexpected source state')
        selected.append({'image': image[y:y+h, x:x+w].copy(),
                         'masks': {name: bit[y:y+h, x:x+w].copy() for name, bit in masks.items()},
                         'name': sample['image'], 'group': sample['group'], 'trajectory':sample['trajectory'],
                         'open_pose': sample['open_pose'], 'states': info})
    if len(selected) != (128 if split == 'train' else 32):
        raise ValueError('unexpected source trajectory split')
    return selected


def load_icons():
    config = yaml.safe_load(CLASSES.read_text(encoding='utf-8'))
    names = [item['name'] for item in config['items']]
    if len(names) != 24 or len(set(names)) != 24:
        raise ValueError('expected 24 unique sticker classes')
    icons = []
    for name in names:
        with np.load(CACHE / (name + '.npz')) as z:
            rgb, alpha = z['rgb'].copy(), z['alpha'].astype(np.float32)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or alpha.shape != rgb.shape[:2]:
            raise ValueError('invalid icon')
        # Despite the cache key name, prep_assets stores OpenCV BGR arrays.
        icons.append((rgb, alpha))
    categories = {item['name']: item['id'] for item in config['items']}
    if set(categories.values()) != {0, 1, 2, 3}:
        raise ValueError('expected the four source waste category identifiers')
    tasks = json.loads((ROOT / 'config/perception_tasks.json').read_text(encoding='utf-8'))['tasks']
    contract = next(task for task in tasks if task['id'] == 'waste_bin')['waste_items']
    category_id = {'recyclable': 0, 'food': 1, 'hazardous': 2, 'residual': 3}
    if set(contract) != set(names) or any(categories[name] != category_id[contract[name]['category']] for name in names):
        raise ValueError('source icons and authoritative waste-category mapping disagree')
    return names, icons, categories, contract


def sticker_count(i):
    # Exactly 15/25/15/20/25% for a multiple of 20.
    return (0, 1, 2, 3, 4, 1, 4, 3, 2, 1, 0, 4, 3, 2, 1, 4, 3, 0, 1, 4)[i % 20]


def paper_on_bin(image, mask, icon, rng):
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    size = max(6, round(w * rng.uniform(.24, .30)))
    cx = x + round(w * rng.uniform(.47, .53))
    # The competition illustration places the paper at the low front edge,
    # just below the bin's existing waste-category emblem.
    cy = y + h - size // 2 - max(5, round(h * rng.uniform(.08, .10)))
    x0, y0 = cx - size // 2, cy - size // 2
    x1, y1 = x0 + size, y0 + size
    if x0 < 0 or y0 < 0 or x1 > image.shape[1] or y1 > image.shape[0]:
        return None
    # Keep a real margin around the paper; the original object mask is the
    # visible silhouette. Erosion rejects placements touching any bin edge.
    safe = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8))
    if not np.all(safe[y0:y1, x0:x1]):
        return None
    # An opaque matte white paper square with the source icon printed on it.
    paper = np.full((size, size, 3), rng.randint(228, 247), dtype=np.uint8)
    rgb, alpha = icon
    margin = max(1, round(size * .12))
    inner = size - 2 * margin
    ratio = min(inner / rgb.shape[1], inner / rgb.shape[0])
    iw, ih = max(1, round(rgb.shape[1] * ratio)), max(1, round(rgb.shape[0] * ratio))
    ink = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_AREA)
    # Preserve hue while making pale source artwork visible after downscaling.
    ink = np.clip(237.0 + (ink.astype(np.float32) - 237.0) * 1.35, 0, 255).astype(np.uint8)
    a = cv2.resize(alpha, (iw, ih), interpolation=cv2.INTER_AREA)[:, :, None]
    ix, iy = (size-iw)//2, (size-ih)//2
    region = paper[iy:iy+ih, ix:ix+iw]
    paper[iy:iy+ih, ix:ix+iw] = np.clip(ink*a + region*(1-a), 0, 255).astype(np.uint8)
    image[y0:y1, x0:x1] = paper
    return x0, y0, x1, y1


def _generate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--split', choices=('train', 'val'), required=True)
    parser.add_argument('--seed', type=int, default=20261110)
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    if not out.is_relative_to((ROOT / 'datasets/exports/waste_bin').resolve()) or out.exists() or args.count < 1:
        parser.error('new output must be within datasets/exports/waste_bin and count > 0')
    sources = load_sources(args.split)
    backgrounds = background_sources() if args.split == 'train' else synthetic_backgrounds(args.seed + 17001)
    names, icons, categories, contract = load_icons()
    rng = random.Random(args.seed)
    pools = {}
    for source_index, src in enumerate(sources):
        combination = ''.join('1' if src['states'][color] == 'open' else '0' for color in COLORS)
        pools.setdefault((combination, src['open_pose']), []).append(source_index)
    jobs = []
    pose_strata = Counter()
    for i in range(args.count):
        # Each combination has equal frequency; within it select 80% hanging
        # and 20% upright lid poses, independently of sticker count.
        combination = format(i % 16, '04b')
        occurrence = i // 16
        desired = sticker_count(occurrence)
        pose_strata[(combination, desired)] += 1
        offset = (int(combination, 2) + desired * 3) % 5
        pose = 'upright' if (pose_strata[(combination, desired)] + offset) % 5 == 0 else 'hanging_back'
        jobs.append((rng.choice(pools[(combination, pose)]), i, occurrence, desired))
    rng.shuffle(jobs)
    counts = Counter()
    state_seen = Counter()
    state_counts = Counter()
    scenario_counts = Counter()
    lighting_counts = Counter()
    pose_counts = Counter()
    for task in ('state', 'sticker'):
        (out/task/'images').mkdir(parents=True)
        (out/task/'labels').mkdir()
    (out/'previews').mkdir()
    (out/'run.json').write_text(json.dumps({'status': 'running', 'count': args.count, 'seed': args.seed}), encoding='utf-8')
    with (out/'manifest.jsonl').open('x', encoding='utf-8') as manifest:
        for order, (source_index, plan_index, occurrence, desired) in enumerate(jobs):
            src = sources[source_index]
            row, bg = backgrounds[rng.randrange(len(backgrounds))]
            crop = src['image'].copy()
            state_crop = crop.copy()
            bitmaps = src['masks']
            scenario_counts[desired] += 1
            pose_counts[src['open_pose']] += 1
            order_colors = list(COLORS)
            rng.shuffle(order_colors)
            sticker_boxes = []
            for color in order_colors:
                if len(sticker_boxes) == desired:
                    break
                candidates = list(range(24)); rng.shuffle(candidates)
                candidates.sort(key=lambda c: counts[c])
                cls = candidates[0]
                box = None
                for _ in range(20):
                    box = paper_on_bin(crop, bitmaps[color], icons[cls], rng)
                    if box is not None:
                        break
                if box is not None:
                    counts[cls] += 1
                    sticker_boxes.append((cls, color, box))
            if len(sticker_boxes) != desired:
                raise ValueError('cannot fit planned sticker within visible bins')
            max_width = min(820, int(400 * crop.shape[1] / crop.shape[0]))
            if max_width < 180:
                raise ValueError('foreground cannot fit into the background')
            width = rng.randint(min(500, max_width), max_width)
            scale = width / crop.shape[1]
            height = round(crop.shape[0] * scale)
            x0 = rng.randint(15, 960-width-15)
            y0 = rng.randint(max(290, height + 5), 450) - height
            if y0 < 0: raise ValueError('foreground placement outside image')
            image = bg.copy()
            shadow = np.zeros((480, 960), dtype=np.uint8)
            for mask in bitmaps.values():
                x,y,w,h = cv2.boundingRect(mask.astype(np.uint8))
                center=(x0+round((x+w/2)*scale),y0+round((y+h)*scale))
                cv2.ellipse(shadow, center, (max(2,round(w*scale*.48)), max(2,round(h*scale*.035))), 0,0,360,90,-1)
            shadow=cv2.GaussianBlur(shadow,(21,21),5).astype(np.float32)/255
            image=(image.astype(np.float32)*(1-shadow[:,:,None]*.38)).astype(np.uint8)
            union=np.logical_or.reduce(list(bitmaps.values())).astype(np.uint8)*255
            rgba=cv2.resize(crop,(width,height),interpolation=cv2.INTER_AREA)
            alpha=cv2.resize(union,(width,height),interpolation=cv2.INTER_AREA).astype(np.float32)[:,:,None]/255
            state_patch=cv2.resize(state_crop,(width,height),interpolation=cv2.INTER_AREA)
            patch=image[y0:y0+height,x0:x0+width]
            state_image=image.copy()
            state_image[y0:y0+height,x0:x0+width]=np.clip(state_patch*alpha+patch*(1-alpha),0,255).astype(np.uint8)
            image[y0:y0+height,x0:x0+width]=np.clip(rgba*alpha+patch*(1-alpha),0,255).astype(np.uint8)
            light = ('dim','normal','bright')[plan_index % 3]
            lighting_counts[light] += 1
            gain = {'dim': rng.uniform(.72,.88), 'normal': rng.uniform(.94,1.06), 'bright': rng.uniform(1.10,1.22)}[light]
            image=np.clip(image.astype(np.float32)*gain,0,255).astype(np.uint8)
            state_image=np.clip(state_image.astype(np.float32)*gain,0,255).astype(np.uint8)
            state_lines=[]; sticker_lines=[]; sticker_records=[]; overlay=image.copy()
            for color,mask in bitmaps.items():
                x,y,w,h=cv2.boundingRect(mask.astype(np.uint8))
                x1,y1=x0+x*scale,y0+y*scale; x2,y2=x0+(x+w)*scale,y0+(y+h)*scale
                cls=0 if src['states'][color]=='open' else 1
                state_lines.append(f'{cls} {(x1+x2)/1920:.6f} {(y1+y2)/960:.6f} {(x2-x1)/960:.6f} {(y2-y1)/480:.6f}')
                cv2.rectangle(overlay,(round(x1),round(y1)),(round(x2),round(y2)),(0,255,0),1)
            for cls,color,(x1,y1,x2,y2) in sticker_boxes:
                x1,x2=x0+x1*scale,x0+x2*scale; y1,y2=y0+y1*scale,y0+y2*scale
                sticker_lines.append(f'{cls} {(x1+x2)/1920:.6f} {(y1+y2)/960:.6f} {(x2-x1)/960:.6f} {(y2-y1)/480:.6f}')
                sticker_records.append({'class_id':cls,'name':names[cls],'bin':color,
                                        'bbox_xyxy':[x1,y1,x2,y2]})
                cv2.rectangle(overlay,(round(x1),round(y1)),(round(x2),round(y2)),(0,200,255),1)
            stem=f'bin_scene_{order:06d}'
            combination = ''.join('1' if src['states'][color] == 'open' else '0' for color in COLORS)
            state_seen[combination] += 1
            state_empty = state_seen[combination] % 10 == 0 or (args.count < 160 and order % 10 == 0)
            state_counts['empty' if state_empty else combination] += 1
            empty_background = np.clip(bg.astype(np.float32)*gain,0,255).astype(np.uint8)
            for task,lines in (('state',[] if state_empty else state_lines),('sticker',sticker_lines)):
                target_image = (empty_background if state_empty else state_image) if task == 'state' else image
                write_image(out/task/'images'/(stem+'.jpg'),target_image)
                (out/task/'labels'/(stem+'.txt')).write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8')
            if order < 12: write_image(out/'previews'/(stem+'.jpg'),overlay)
            record={'image':stem+'.jpg','source_render':src['name'],'render_group':src['group'],
                    'background':row['file'],'background_frame':int(row['frame']),'open_pose':src['open_pose'],
                    'source_group':'scene_video_49s_shared' if args.split == 'train' else 'procedural_val_backgrounds',
                    'render_trajectory':src['trajectory'],
                    'task_splits':{'state':args.split,'sticker':'train' if args.split == 'train' else 'scene_diagnostic_shared_icons'},
                    'states':src['states'],
                    'stickers':sticker_records,
                    'state_empty':state_empty,'brightness':light,'gain':gain,'crop_xyxy':[480,600,1440,1080],
                    'placement':[x0,y0,width,height]}
            manifest.write(json.dumps(record,ensure_ascii=False)+'\n')
    (out/'classes.json').write_text(json.dumps({'state':['open','closed'],'sticker':names,
        'sticker_to_waste_category_id':categories,'canonical_waste_names':contract},ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps({'status':'complete_candidates','count':args.count,'seed':args.seed,
        'task_splits':{'state':args.split,'sticker':'train' if args.split == 'train' else 'scene_diagnostic_shared_icons'},
        'source_groups':['scene_video_49s_shared'] if args.split == 'train' else ['procedural_val_backgrounds'],
        'render_trajectory':0 if args.split == 'train' else 1,'class_counts':{names[i]:counts[i] for i in range(24)},
        'sticker_source_group':'waste_bin_toolkit_24icons_shared_across_splits',
        'sticker_evaluation_scope':'seen artwork and changed scene; not independent source artwork',
        'state_scenarios':dict(state_counts),'sticker_counts_per_image':dict(scenario_counts),
        'lighting':dict(lighting_counts),'open_poses':dict(pose_counts),
        'description':'rendered bins and printed sticker sprites with separate labels; source split by trajectory and background; visual review needed'},ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'run.json').write_text(json.dumps({'status': 'complete_candidates', 'count': args.count, 'seed': args.seed}), encoding='utf-8')
    print('complete:',args.count)


def _guard_icons():
    """Declare cached artwork, its original media, and its category contract."""
    classes = yaml.safe_load(CLASSES.read_text(encoding='utf8'))
    names = [item['name'] for item in classes['items']]
    media = {ROOT / 'datasets/raw/waste_bin/toolkit/SourcePic' / (name + '.png') for name in names}
    procedural = {CLASSES, ROOT / 'config/perception_tasks.json'}
    procedural.update(CACHE / (name + '.npz') for name in names)
    return media, procedural


def _guard_inputs(split):
    media, procedural = _guard_icons()
    render_manifest, export_manifest = SRC / 'manifest.json', EXPORT / 'manifest.json'
    procedural.update((render_manifest, export_manifest))
    render = json.loads(render_manifest.read_text(encoding='utf8'))
    if render.get('scene_file'):
        procedural.add(ROOT / render['scene_file'])
    exported = json.loads(export_manifest.read_text(encoding='utf8'))
    by_name = {item['image']: item for item in render['samples']}
    for entry in exported['included']:
        sample = by_name[entry['source_image']]
        if (sample['trajectory'] == 0) == (split == 'train'):
            media.update((SRC / sample['image'], SRC / sample['mask_visible']))
    if split == 'train':
        manifest = BASE / 'reports/backgrounds_manifest.csv'
        procedural.add(manifest)
        with manifest.open(encoding='utf8', newline='') as stream:
            media.update(BASE / 'backgrounds' / row['file'] for row in csv.DictReader(stream)
                         if 0 <= int(row['frame']) <= 179)
    return sorted(media), sorted(procedural)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--split', choices=('train', 'val'), required=True)
    parser.add_argument('--seed', type=int, default=20261110)
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    sys.path.insert(0, str(ROOT))
    from lab.training_common.production_guard import generation_guard
    media, procedural = _guard_inputs(args.split)
    with generation_guard('waste_bin.composite:' + str(out.relative_to(ROOT)),
                          media, [out], vars(args),
                          code_paths=[Path(__file__).resolve()], procedural_inputs=procedural, root=ROOT) as run:
        if run.reused:
            return
        _generate()


if __name__ == '__main__':
    main()
