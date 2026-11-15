"""Make a source-tracked mixed plate development candidate, not a frozen test."""
import json
import hashlib
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT / 'lab/training_common')]
from camera_tone import apply_tone, read_image, source_role, write_image  # noqa: E402
from lab.scripts.data.v4_source_gate import build_queue  # noqa: E402
from lab.training_common.production_guard import generation_guard  # noqa: E402

LEGACY = ROOT / 'datasets/exports/plate/site_finetune_v1_20261112/detector'
ROAD = ROOT / 'datasets/exports/plate/provincial_ccpd/detection'
SITE = ROOT / 'datasets/exports/plate/v4_video_train_candidates_20261112'
OUT = ROOT / 'datasets/exports/plate/v4_mixed_development_candidates_v2_20261112'
SEED = 20261117
ROAD_CATEGORIES = ('ccpd_base', 'ccpd_blur', 'ccpd_db', 'ccpd_fn', 'ccpd_np',
                   'ccpd_rotate', 'ccpd_tilt', 'ccpd_weather')


def rel(path):
    return path.relative_to(ROOT).as_posix()


def valid_label(path):
    text = path.read_text(encoding='utf-8').strip()
    for line in text.splitlines():
        values = line.split()
        if len(values) != 5 or values[0] != '0':
            raise ValueError('Unexpected plate label: ' + rel(path))
        x, y, width, height = map(float, values[1:])
        # Original YOLO decimals round edge-touching boxes by up to 5e-9.
        if not (0 < width <= 1 and 0 < height <= 1 and -1e-6 <= x-width/2 <= x+width/2 <= 1+1e-6
                and -1e-6 <= y-height/2 <= y+height/2 <= 1+1e-6):
            raise ValueError('Plate label outside image: ' + rel(path))
    return text


def choices(items, count, rng, seen, source_image):
    items = sorted(items, key=lambda item: str(item))
    if len(items) < count:
        raise ValueError(f'Insufficient source inputs: {len(items)}/{count}')
    rng.shuffle(items)
    chosen = []
    for item in items:
        digest = hashlib.sha256(source_image(item).read_bytes()).digest()
        if digest not in seen:
            seen.add(digest)
            chosen.append(item)
            if len(chosen) == count:
                return chosen
    raise ValueError(f'Insufficient distinct source images: {len(chosen)}/{count}')


def _generate(known, source_refs):
    rng = random.Random(SEED)
    old = json.loads((LEGACY/'workspace_sources.json').read_text(encoding='utf-8'))['records']
    grouped = defaultdict(list)
    for row in old:
        if row['generation_method'] == 'provincial_synthetic' and row['split'] in ('train', 'val'):
            grouped[(row['split'], row['source_group'])].append(row)
    queue, seen = [], set()
    for split, count in (('train', 400), ('val', 80)):
        groups = sorted(key for key in grouped if key[0] == split)
        if len(groups) != 2:
            raise ValueError('Expected both historical synthetic families')
        for key in groups:
            for row in choices(grouped[key], count, rng, seen, lambda x: ROOT/x['workspace_image']):
                queue.append({'split':split, 'image':ROOT/row['workspace_image'],
                    'label':ROOT/row['workspace_label'], 'family':'provincial_synthetic',
                    'source_group':row['source_group'], 'source_split':split,
                    'review_status':'inherited_synthetic_annotation'})
    for split, count in (('train', 100), ('val', 20)):
        for category in ROAD_CATEGORIES:
            images = list((ROAD/'images'/split).glob(category+'__*'))
            for image in choices(images, count, rng, seen, lambda x: x):
                label = ROAD/'labels'/split/(image.stem+'.txt')
                queue.append({'split':split,'image':image,'label':label,'family':'ccpd_road_capture',
                    'source_group':f'CCPD/{category}', 'source_split':split,
                    'review_status':'inherited_CCPD_annotation'})
    site = json.loads((SITE/'manifest.json').read_text(encoding='utf-8'))['records']
    site_parents = {}
    for row in site:
        parent = ROOT/row['parent_image']
        if parent not in site_parents:
            role, reason = source_role(ROOT, parent, known)
            if role == 'regression':
                raise ValueError('Diagnostic source cannot enter training: ' + str(reason))
            site_parents[parent] = row['state']
        queue.append({'split':'train','image':ROOT/row['image'],'label':ROOT/row['label'],
            'family':'site_video_augmentation','source_group':row['source_video'],
            'source_split':'train_candidate','parent':row['parent_image'],
            'source_time_seconds':row['source_time_seconds'],'segment_id':row['segment_id'],
            'source_processing':row['processing'],'review_status':row['label_review']})
    if len(site_parents) != 5 or len(site) != 120:
        raise ValueError('Unexpected site parent/augmentation count')
    if OUT.exists():
        raise ValueError('Output exists; select a new version rather than overwrite')
    return _produce(queue, site_parents, source_refs)


def _produce(queue, site_parents, source_refs):
    counts, records, tones = Counter(), [], Counter()
    try:
        for split in ('train','val'):
            (OUT/'images'/split).mkdir(parents=True)
            (OUT/'labels'/split).mkdir(parents=True)
        for index, item in enumerate(queue):
            label = valid_label(item['label'])
            image = read_image(item['image'])
            h, w = image.shape[:2]
            scale = min(1., 640 / max(w, h))
            if scale < 1:
                image = cv2.resize(image, (max(1,round(w*scale)), max(1,round(h*scale))),
                                   interpolation=cv2.INTER_AREA)
            if item['family'] == 'site_video_augmentation':
                profile, value, saturation = 'inherited_site_tone', None, None
            else:
                profile = ('original','mild','moderate')[index % 3]
                value, saturation = {'original':(1.,1.), 'mild':(.9,.85), 'moderate':(.7,.6)}[profile]
                if profile != 'original':
                    image = apply_tone(image, value_gain=value, saturation_gain=saturation)
            split = item['split']
            name = f'plate_{index+1:06d}'
            output_image, output_label = OUT/'images'/split/(name+'.png'), OUT/'labels'/split/(name+'.txt')
            write_image(output_image, image)
            output_label.write_text((label+'\n') if label else '', encoding='utf-8')
            records.append({'image':rel(output_image), 'label':rel(output_label), 'split':split,
                'source_image':rel(item['image']), 'source_label':rel(item['label']),
                'family':item['family'], 'source_group':item['source_group'], 'source_split':item['source_split'],
                'parent_image':item.get('parent',rel(item['image'])),
                'source_time_seconds':item.get('source_time_seconds'), 'segment_id':item.get('segment_id'),
                'review_status':item['review_status'], 'image_size':[image.shape[1],image.shape[0]],
                'source_image_size':[w,h], 'resize':'downscale_to_longest_640' if scale<1 else 'none',
                'tone_profile':profile, 'value_gain':value, 'saturation_gain':saturation,
                'source_processing':item.get('source_processing'),
                'generation_method':'inherited_site_augmentation' if value is None else 'historical_image_resize_and_tone'})
            counts[(split,item['family'],'positive' if label else 'negative')] += 1
            tones[(split,profile)] += 1
        manifest = {'status':'draft_not_deliverable', 'seed':SEED, 'class_names':['plate'], 'source_refs':source_refs,
            'scope':'Train and previously-used development candidates only; no frozen test. '
                    'Site variants share five frames from one video and reused physical props. '
                    'CCPD is road imagery, not the competition camera. Synthetic source families share assets.',
            'site_parent_count':len(site_parents), 'site_positive_parent_count':sum(v=='positive' for v in site_parents.values()),
            'counts':[{'split':a,'source':b,'state':c,'images':n} for (a,b,c),n in sorted(counts.items())],
            'tone_counts':[{'split':a,'profile':b,'images':n} for (a,b),n in sorted(tones.items())],
            'records':records}
        (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (OUT/'data.draft.yaml').write_text('train: images/train\nval: images/val\nnames:\n  0: plate\n',encoding='utf-8')
        print(json.dumps({'output':rel(OUT),'counts':manifest['counts'],'tone_counts':manifest['tone_counts'],
                          'site_parents':len(site_parents)},ensure_ascii=False))
    except BaseException:
        (OUT/'FAILED.txt').write_text('Generation stopped; do not use before inspecting the failure.\n',encoding='utf-8')
        raise


def main():
    from lab.features.plate.training.scripts.plate_guard_inputs import source_gate_dependencies
    old_path, site_path = LEGACY/'workspace_sources.json', SITE/'manifest.json'
    old = json.loads(old_path.read_text(encoding='utf-8'))['records']
    site = json.loads(site_path.read_text(encoding='utf-8'))['records']
    known = build_queue(ROOT)
    gate_files, gate_code = source_gate_dependencies(ROOT, known)
    eligible = [r for r in old if r['generation_method'] == 'provincial_synthetic' and r['split'] in ('train', 'val')]
    media = {ROOT / row['workspace_image'] for row in eligible}
    procedural = {ROOT / row['workspace_label'] for row in eligible}
    procedural.update({old_path, site_path} | gate_files)
    for split in ('train', 'val'):
        for category in ROAD_CATEGORIES:
            for image in (ROAD/'images'/split).glob(category+'__*'):
                media.add(image)
                procedural.add(ROAD/'labels'/split/(image.stem+'.txt'))
    for row in site:
        media.add(ROOT / row['image'])
        procedural.add(ROOT / row['label'])
        media.update(ROOT / row[key] for key in ('parent_image', 'source_video') if row.get(key))
    with generation_guard('plate_mixed_candidates_v2', sorted(media), [OUT],
                          {'seed': SEED, 'road_categories': ROAD_CATEGORIES},
                          code_paths=sorted(gate_code | {Path(__file__), ROOT/'lab/training_common/camera_tone.py'}),
                          procedural_inputs=sorted(procedural), root=ROOT) as run:
        if run.reused:
            print('Reused indexed plate mixed candidates')
            return
        return _generate(known, run.source_refs)


if __name__ == '__main__':
    main()
