"""Assemble plate V4 using provenance-backed Blender and original backgrounds."""
import collections
import filecmp
import json
import math
from pathlib import Path
import shutil
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard

BASE = ROOT / 'datasets/exports/plate'
ORIGINAL = BASE / 'v4_final_20261113/manifest.json'
COMPOSITES = BASE / 'v4_arena_background_composites_20261113/manifest.json'
OUT = BASE / 'v4_bg40_20261113'
EMPTY_SOURCE = 'lab/features/plate/training/scripts/render_plate_empty_scene.py'
ARENA_SOURCE = 'datasets/assets/blender/shared/arena_4m2_pbr_20261112/arena_indoor_detailed.blend'
LEGACY_SYNTH = ('datasets/exports/plate/provincial_synthetic/round2_multi_car/',
                'datasets/exports/plate/provincial_synthetic/synthetic_train/')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def classify(row, old_manifests):
    group = row.get('source_group') or ''
    if any(group.startswith(prefix) for prefix in LEGACY_SYNTH):
        return None, 'original background/scene generator not established in workspace evidence'
    if row.get('generation_method') == 'blender_cycles_optix_empty_scene_real_photo_crop':
        ref = row.get('source_manifest') or row.get('input_manifest')
        index = row.get('source_record_index') if row.get('source_manifest') else row.get('input_record_index')
        source = old_manifests[ref]['records'][index]
        yaw = source.get('camera_yaw_degrees')
        if not isinstance(yaw, (int, float)) or not math.isfinite(yaw):
            raise ValueError('Missing recorded Blender camera yaw: ' + row['image'])
        view = 'empty_left' if yaw < -3 else 'empty_right' if yaw > 3 else 'empty_front'
        return {'kind': 'blender', 'source_group': EMPTY_SOURCE, 'view_group': view}, None
    if row.get('source_video'):
        return {'kind': 'real', 'source_group': row['source_video'], 'view_group': None}, None
    if group.startswith('CCPD/'):
        parent = row.get('parent_image')
        if not isinstance(parent, str) or not parent.startswith('datasets/exports/plate/provincial_ccpd/detection/images/'):
            raise ValueError('Missing imported CCPD original photo: ' + row['image'])
        return {'kind': 'real', 'source_group': parent, 'view_group': None}, None
    raise ValueError('Cannot establish original background source: ' + row['image'])


def plan():
    old = read(ORIGINAL)['records']
    composite = read(COMPOSITES)['records']
    if len(old) != 2650 or len(composite) != 990:
        raise ValueError('Expected existing 2650 images and 990 background composites')
    manifests = {}
    for row in old:
        if row.get('generation_method') == 'blender_cycles_optix_empty_scene_real_photo_crop':
            ref = row.get('source_manifest') or row.get('input_manifest')
            if ref not in manifests:
                manifests[ref] = read(ROOT / ref)
    selected, excluded = [], []
    for index, row in enumerate(old):
        bg, reason = classify(row, manifests)
        if reason:
            excluded.append({'image': row['image'], 'split': row['split'], 'reason': reason,
                             'manifest': rel(ORIGINAL), 'record_index': index})
            continue
        selected.append((row, rel(ORIGINAL), index, bg))
    for index, row in enumerate(composite):
        if row.get('mother_scene') != ARENA_SOURCE or not row.get('blender_background'):
            raise ValueError('Composite background is not backed by expected Blender scene')
        camera = row.get('background_camera') or {}
        view = camera.get('image')
        if view not in {'arena_00.png','arena_01.png','arena_02.png','arena_03.png'}:
            raise ValueError('Missing or unknown Blender camera view')
        selected.append((row, rel(COMPOSITES), index,
                         {'kind': 'blender', 'source_group': ARENA_SOURCE, 'view_group': Path(view).stem}))
    counts = collections.Counter(row['split'] for row, _, _, _ in selected)
    if counts != {'train': 2209, 'val': 363, 'test': 108} or len(excluded) != 960:
        raise ValueError('Unexpected selected/excluded counts: ' + str((counts, len(excluded))))
    return selected, excluded, sorted(manifests)


def verify(selected):
    sizes = collections.defaultdict(list)
    for row, _, _, bg in selected:
        image, label = ROOT / row['image'], ROOT / row['label']
        if not image.is_file() or not label.is_file() or not (ROOT / bg['source_group']).is_file():
            raise ValueError('Missing input, label, or original background: ' + row['image'])
        with Image.open(image) as im:
            im.verify()
        with Image.open(image) as im:
            width, height = im.size
        if width > 1280 or height > 720:
            raise ValueError('Image exceeds 720p: ' + row['image'])
        for line in label.read_text(encoding='utf-8').splitlines():
            parts = line.split()
            if len(parts) != 5 or parts[0] != '0':
                raise ValueError('Invalid class/label: ' + row['label'])
            x,y,w,h = map(float, parts[1:])
            if not all(math.isfinite(v) for v in (x,y,w,h)) or not (0<w<=1 and 0<h<=1 and
               -1e-6<=x-w/2<=x+w/2<=1+1e-6 and -1e-6<=y-h/2<=y+h/2<=1+1e-6):
                raise ValueError('Invalid label bounds: ' + row['label'])
        sizes[image.stat().st_size].append((image,row['split']))
    for group in sizes.values():
        for i,(a,sa) in enumerate(group):
            for b,sb in group[i+1:]:
                if sa != sb and filecmp.cmp(a,b,shallow=False):
                    raise ValueError('Identical pixels across splits: ' + str(a) + ' / ' + str(b))


def assemble(selected, excluded, source_refs):
    verify(selected)
    if OUT.exists():
        raise ValueError('Output already exists; no overwrite')
    for split in ('train','val','test'):
        for kind in ('images','labels'):
            (OUT/kind/split).mkdir(parents=True, exist_ok=False)
    records=[]; counts=collections.Counter(); background_counts=collections.Counter()
    views=collections.defaultdict(set); original_groups=collections.defaultdict(collections.Counter)
    for index,(row,manifest,record_index,bg) in enumerate(selected):
        original, label = ROOT/row['image'], ROOT/row['label']
        name = 'plate_bg40_%05d' % index
        image_out = OUT/'images'/row['split']/(name+original.suffix.lower())
        label_out = OUT/'labels'/row['split']/(name+'.txt')
        shutil.copy2(original,image_out); shutil.copy2(label,label_out)
        refs=source_refs.get(row['image'],[])
        if not refs:
            raise ValueError('Indexed source evidence missing: ' + row['image'])
        record=dict(row, image=rel(image_out), label=rel(label_out), background=bg,
                    input_image=row['image'], input_label=row['label'],
                    input_manifest=manifest, input_record_index=record_index,
                    source_evidence=refs)
        records.append(record); counts[row['split']]+=1
        if bg['kind']=='blender':
            background_counts[row['split']]+=1; views[bg['source_group']].add(bg['view_group'])
        else:
            original_groups[row['split']][bg['source_group']]+=1
    n=sum(counts.values()); bn=sum(background_counts.values())
    if bn/n < .4 or sum(map(len, views.values())) < 2:
        raise ValueError('Blender share or distinct camera view requirement not met')
    if len({g for groups in original_groups.values() for g in groups}) < 2:
        raise ValueError('Insufficient original non-Blender background groups')
    (OUT/'data.yaml').write_text('train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: plate\n',encoding='utf-8')
    result={'status':'assembled_candidate','counts':dict(counts),'blender_backgrounds':dict(background_counts),
            'blender_total':bn,'total':n,'blender_ratio':bn/n,
            'blender_view_groups':{k:sorted(v) for k,v in views.items()},
            'non_blender_background_counts':{k:dict(v) for k,v in original_groups.items()},
            'scope':'Blender indoor scene composites and empty scene renders; non-Blender site video and imported CCPD source photographs. CCPD imported photo paths identify available files, not proof of independently captured scenes or distinct underlying originals; no model-performance claim.',
            'excluded_background_unresolved':excluded,'evaluation':{'status':'not_run','weights':None,'metrics':None},
            'records':records}
    (OUT/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'counts':dict(counts),'blender_total':bn,'total':n,'ratio':bn/n},ensure_ascii=False))


def main():
    selected,excluded,manifests=plan()
    media = {ROOT/row['image'] for row,_,_,_ in selected}
    procedural = {ROOT/row['label'] for row,_,_,_ in selected}|{ORIGINAL,COMPOSITES}|{ROOT/m for m in manifests}
    media_suffixes = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp','.mp4','.mov','.avi','.mkv'}
    for row, _, _, bg in selected:
        references = [bg['source_group'], row.get('parent_image'), row.get('source_video'),
                      row.get('parent_video'), row.get('source_background')]
        references.extend(row.get('source_assets') or [])
        for value in references:
            if isinstance(value, str) and Path(value).suffix.lower() in media_suffixes:
                media.add(ROOT/value)
        if bg['kind']=='blender' and Path(bg['source_group']).suffix.lower()=='.blend':
            procedural.add(ROOT/bg['source_group'])
    media, procedural = sorted(media), sorted(procedural)
    parameters={'class':'plate','policy':'background_source_40_percent','excluded_reason':'background_source_unestablished',
                'input_split_policy':'preserve','output_counts':[2209,363,108]}
    with generation_guard('plate_v4_bg40_20261113',media,[OUT],parameters,
                          code_paths=[Path(__file__), ROOT/EMPTY_SOURCE],procedural_inputs=procedural,root=ROOT) as run:
        if run.reused:
            print('Reused indexed plate bg40 output')
            return
        assemble(selected,excluded,run.source_refs)


if __name__ == '__main__':
    main()
