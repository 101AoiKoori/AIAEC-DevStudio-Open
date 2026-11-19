"""Plan and compose full-bin Blender layers over assigned real ground frames."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.frame_environment import image_digest, query_frames
from datasets.scripts.common.images import read_image, write_image
from datasets.scripts.waste_sticker.compose_sticker_real_plate import composite, image8, linear
from datasets.scripts.waste_sticker.plan_evaluation_branches import ground_plan
from datasets.scripts.waste_sticker.source_inputs import artwork_catalog

CONFIG = ROOT / 'datasets/scripts/waste_sticker/config/hybrid_full_plan.json'
OUTPUT = ROOT / 'datasets/annotated/waste_bin/waste_sticker_full/hybrid'
COLORS = ('blue', 'red', 'green', 'gray')


def store_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def train_templates():
    records = [row for row in query_frames(task='waste_sticker', role='composite_background', split='train')
               if not row['geometry'].get('bins') and '空场与场景物料_001' in row['parent_group']]
    if len(records) != 1:
        raise ValueError('Expected the qualified low-view training plate')
    row = records[0]
    photo = next(path for path in row['paths'] if path.startswith('datasets/raw/shared/') and (ROOT/path).is_file())
    if image_digest(ROOT/photo) != row['content_sha256']:
        raise ValueError('Training photograph differs from its qualification')
    plate = dict(source_photo=photo, frame_id=row['frame_id'], record_id=row['record_id'],
                 parent=row['parent_group'], size=row['image_size'],
                 geometry=dict(ground_polygon=row['geometry']['ground_polygon']),
                 evidence_ids=[item['evidence_id'] for item in row['evidence']])
    jobs, details = [], []
    class_totals = Counter()
    for count in (1, 2, 3, 4):
        for index in range(8):
            selected = [COLORS[(index+j) % 4] for j in range(count)]
            repeats = 15 if count in (1, 2) and index == 0 else 14
            classes = sorted(range(24), key=lambda class_id: (class_totals[class_id], (class_id-index*7) % 24))[:count]
            for class_id in classes:
                class_totals[class_id] += repeats
            anchors = {1: [(470, 456)],
                       2: [(340, 454), (560, 458)],
                       3: [(255, 452), (450, 450), (645, 454)],
                       4: [(160, 465), (330, 465), (500, 465), (670, 465)]}[count]
            feet = dict(zip(selected, anchors))
            template_id = f'train_template_{count}_{index:02d}'
            jobs.append([0, 1 + ((index*29 + count*47) % 240), feet,
                         (112 if count == 4 else 136) + 3*(index % 3), 109, 750, (-8, 0, 7)[index % 3]])
            details.append(dict(id=template_id, class_ids=dict(zip(selected, classes)), repeats=repeats))
    return dict(model_key='waste_sticker', target_split='train', budget=len(jobs),
                plates=[plate], jobs=jobs, job_details=details)


def plan():
    train = train_templates()
    test = ground_plan()
    # Eight physical single-bin templates per selected real frame cover all 24
    # classes. Repeated composites retain their shared template identity.
    test['jobs'] = test['jobs'][:24]
    test['job_details'] = []
    for index, job in enumerate(test['jobs']):
        color = next(iter(job[2]))
        test['job_details'].append(dict(id=f'test_template_{index:02d}',
                                        class_ids={color: index}, repeats=2 if index < 16 else 1))
    test['budget'] = len(test['jobs'])
    value = dict(task='waste_sticker', branch='blender_on_real_site',
                 counts=dict(train=450, test=40), templates=dict(train=train, test=test),
                 seed=2026111803, scene='datasets/assets/blender/aiaec_dataset.blend',
                 reuse='Physical RGBA, receiver-shadow and instance masks retained per template; '
                       'ground-constrained uniform scale/translation and mild photometric variation per output')
    store_json(CONFIG, value)
    for split, declaration in value['templates'].items():
        store_json(OUTPUT/split/'template_plan.json', declaration)
    print('PLANNED', {split: len(d['jobs']) for split, d in value['templates'].items()}, flush=True)


def project_points(points, transform):
    return cv2.transform(np.float32([points]), transform)[0]


def make_variant(record, rng, split, variant):
    width, height = record['image_size']
    source_w, source_h = record['source_size']
    polygon = np.float32(record['geometry']) * [width/source_w, height/source_h]
    footprints = [point for entry in record['projected_feet'].values() for point in entry['footprint']]
    center = np.mean([entry['center'] for entry in record['projected_feet'].values()], axis=0)
    alpha_bounds = record['alpha_bounds']
    subject_corners = [[alpha_bounds[0], alpha_bounds[1]], [alpha_bounds[2], alpha_bounds[1]],
                       [alpha_bounds[2], alpha_bounds[3]], [alpha_bounds[0], alpha_bounds[3]]]
    for attempt in range(500):
        scale = rng.uniform(.82, 1.12) if split == 'train' else rng.uniform(.87, 1.07)
        dx = rng.uniform(-135, 135) if split == 'train' else rng.uniform(-48, 48)
        dy = rng.uniform(-20, 25) if split == 'train' else rng.uniform(-10, 10)
        transform = np.float32([[scale, 0, center[0]*(1-scale)+dx],
                               [0, scale, center[1]*(1-scale)+dy]])
        feet = project_points(footprints, transform)
        corners = project_points(subject_corners, transform)
        if (any(cv2.pointPolygonTest(polygon.astype(np.float32), tuple(map(float, point)), False) < 0 for point in feet)
                or np.any(corners < [0, 0]) or np.any(corners >= [width, height])):
            continue
        return transform, feet.tolist(), scale
    raise ValueError(f'No grounded image variant {record["id"]}/{variant}')


def compose_split(split):
    full_plan = json.loads(CONFIG.read_text('utf8'))
    declaration = full_plan['templates'][split]
    work = OUTPUT/split
    layers = work/'layers'
    composite(layers)
    rendered = json.loads((layers/'manifest.json').read_text('utf8'))
    by_id = {row['id']: row for row in rendered['records']}
    classes = artwork_catalog()
    class_names = [Path(classes[index]).stem for index in range(24)]
    for sub in ('images', 'labels'):
        (work/sub).mkdir(exist_ok=True)
    records, hashes = [], set()
    rng = random.Random(full_plan['seed'] + (split == 'test')*10000)
    for detail in declaration['job_details']:
        record = by_id[detail['id']]
        width, height = record['image_size']
        foreground = read_image(layers/record['foreground'], cv2.IMREAD_UNCHANGED)
        raw_mask = read_image(layers/record['visible_mask'], cv2.IMREAD_UNCHANGED)
        raw_mask = cv2.cvtColor(raw_mask, cv2.COLOR_BGRA2RGBA)
        shadow = read_image(layers/record['shadow_attenuation'], cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
        photo = cv2.resize(read_image(ROOT/record['source_photo']), (width, height), interpolation=cv2.INTER_AREA)
        ys, xs = np.nonzero(foreground[:, :, 3] > 8)
        record['alpha_bounds'] = [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]
        for variant in range(detail['repeats']):
            transform, footprints, scale = make_variant(record, rng, split, variant)
            rgba = cv2.warpAffine(foreground, transform, (width, height), flags=cv2.INTER_LINEAR)
            shade = cv2.warpAffine(shadow, transform, (width, height), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=1)
            mask = cv2.warpAffine(raw_mask, transform, (width, height), flags=cv2.INTER_NEAREST)
            alpha = rgba[:, :, 3:4].astype(np.float32)/255
            subject = rgba[:, :, :3].astype(np.float32)/255
            luma = subject @ np.array([.0722, .7152, .2126], np.float32)
            subject = .82*subject + .18*luma[:, :, None]
            result = image8(linear(photo.astype(np.float32)/255)*shade[:, :, None]*(1-alpha) +
                            .96*linear(subject)*alpha)
            gain, saturation = rng.uniform(.93, 1.035), rng.uniform(.92, 1.045)
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= saturation
            hsv[:, :, 2] *= gain
            result = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
            instances, labels = [], []
            for instance in record['instances']:
                color = rendered['palette'][instance['bin']]
                visible = mask[:, :, 3] >= 192
                for channel, value in enumerate(color):
                    visible &= mask[:, :, channel] >= 192 if value else mask[:, :, channel] <= 63
                y, x = np.nonzero(visible)
                if len(x) < 24 or np.mean(alpha[y, x, 0] > .5) < .98:
                    raise ValueError(f'{record["id"]}: the visible sticker is too small or mask differs')
                box = [int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)]
                a, b, c, d = box
                cls = instance['class_id']
                labels.append(f'{cls} {(a+c)/(2*width):.8f} {(b+d)/(2*height):.8f} '
                              f'{(c-a)/width:.8f} {(d-b)/height:.8f}')
                instances.append(dict(bin_id='bin_'+instance['bin'], class_id=cls, bbox_xyxy=box,
                                      visible_pixels=len(x), source_artwork=classes[cls]))
            stem = f'hybrid_{split}_{len(records):06d}'
            image_path, label_path = work/'images'/(stem+'.png'), work/'labels'/(stem+'.txt')
            write_image(image_path, result)
            label_path.write_text('\n'.join(labels)+'\n', encoding='utf8')
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            if digest in hashes:
                raise ValueError('Repeated image bytes in hybrid branch')
            hashes.add(digest)
            records.append(dict(
                id=stem, image=f'images/{stem}.png', label=f'labels/{stem}.txt', split=split,
                branch='blender_on_real_site', generation_method='physical_blender_bin_real_ground_composite',
                source_parent=record['parent_group'], source_photo=record['source_photo'],
                source_decision=record['source_decision'], source_evidence=record['source_evidence'],
                source_frame_id=record['frame_id'], source_size=record['source_size'],
                source_blend=full_plan['scene'], template_id=record['id'],
                shared_template_count=detail['repeats'], frame=record['frame'],
                image_size=[width, height], selected_bins=record['selected_bins'], visible_bins=record['selected_bins'],
                visible_sticker_count=len(instances), class_ids=[item['class_id'] for item in instances],
                instances=instances, camera=record['camera'],
                projected_feet=footprints, ground_polygon=record['geometry'],
                variant=dict(affine=transform.tolist(), scale=scale, gain=gain, saturation=saturation,
                             receiver_shadow_reuse='same physical render transformed with the complete bin layer'),
                sha256=digest,
            ))
            if len(records) % 25 == 0:
                print('COMPOSED', split, len(records), flush=True)
    expected = full_plan['counts'][split]
    if len(records) != expected:
        raise ValueError(f'Expected {expected}, generated {len(records)}')
    by_class = Counter(item['class_id'] for row in records for item in row['instances'])
    manifest = dict(task='waste_sticker', split=split, status='generated',
                    image_count=len(records), instance_count=sum(by_class.values()),
                    template_count=len(by_id), source_groups=sorted({r['source_parent'] for r in records}),
                    counts_by_visible_stickers=dict(Counter(row['visible_sticker_count'] for row in records)),
                    counts_by_class=dict(sorted(by_class.items())), records=records)
    store_json(work/'manifest.json', manifest)
    store_json(work/'classes.json', class_names)
    indices = sorted(set([0, len(records)//4, len(records)//2, 3*len(records)//4, len(records)-1]))
    sheet = Image.new('RGB', (960, 400), '#17191d')
    draw = ImageDraw.Draw(sheet)
    for j, index in enumerate(indices):
        row = records[index]
        image = Image.open(work/row['image']).convert('RGB')
        thumb = ImageOps.contain(image, (310, 174))
        x, y = (j % 3)*320+5, (j // 3)*200+5
        sheet.paste(thumb, (x, y))
        draw.text((x, y+178), row['id']+' '+str(row['class_ids']), fill='white')
    sheet.save(work/'contact_sheet.jpg', quality=92)
    print('DONE', split, len(records), manifest['counts_by_visible_stickers'], dict(sorted(by_class.items())), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'compose'))
    parser.add_argument('--split', choices=('train', 'test'))
    args = parser.parse_args()
    if args.action == 'plan':
        plan()
    else:
        if not args.split:
            parser.error('compose requires --split')
        compose_split(args.split)


if __name__ == '__main__':
    main()
