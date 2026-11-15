"""Replace V4 sticker selection with indexed arena and real-video scene batches."""
import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT / 'lab/scripts/data')]
from lab.training_common.production_guard import generation_guard
import v4_source_gate as gate

SOURCE = 'datasets/annotated/waste_sticker_v4_handoff_20261113/staged_master_with_test_source_gate.json'
CONFIG = 'lab/features/waste_bin/training/config/waste_sticker_v4_reviewed_master.yaml'
DISTRIBUTION = 'datasets/annotated/waste_sticker_v4_handoff_20261113/distribution_master_candidate.json'
PRIOR = 'datasets/exports/waste_bin/waste_sticker_v4_reviewed_two_bin_20261113'
PRIOR_VIDEO = 'datasets/raw/site_capture_20261112/videos/full_scene.mp4'
CLASSES = 'datasets/raw/waste_bin/toolkit/configs/classes.yaml'
ARTWORK_DIR = 'datasets/raw/waste_bin/toolkit/SourcePic'
CACHE_DIR = 'datasets/exports/waste_bin/.cache/assets'
SPLITS = ('train', 'val', 'test')


def relative(root, value):
    path = Path(value)
    path = path if path.is_absolute() else root / path
    path = path.resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('path must be in workspace: ' + str(value))
    return path.relative_to(root).as_posix()


def facts(row):
    """Resolve the true background independently from the pasted foreground."""
    if row.get('arena_master') and row.get('source_image'):
        camera = row.get('source_camera')
        if (not isinstance(camera, dict) or not camera.get('location') or not camera.get('rotation')
                or camera.get('lens_mm') is None):
            raise ValueError('Blender background requires fitted camera location, rotation and lens')
        view = json.dumps({'location': camera['location'], 'rotation': camera['rotation'],
                           'lens_mm': camera['lens_mm']}, sort_keys=True, separators=(',', ':'))
        return {'blender': True, 'parent': row['source_image'], 'video': None,
                'scene': row['arena_master'], 'background_origin': 'synthetic', 'view_group': view}
    if row.get('background_image') and row.get('parent_video'):
        return {'blender': False, 'parent': row['background_image'], 'video': row['parent_video'],
                'scene': row.get('source_render'), 'background_origin': 'direct_camera'}
    raise ValueError('missing background image and real parent video or Blender mother scene')


def artwork_paths(root, row, class_names):
    """Resolve the exact image/cache inputs used by the verified sticker generators."""
    cls = row.get('class_id')
    if cls is None:
        return []
    if (not isinstance(cls, int) or isinstance(cls, bool) or not 0 <= cls < len(class_names)
            or row.get('source_name') != class_names[cls]):
        raise ValueError('class id and generator class name disagree: ' + row.get('image', '?'))
    name = class_names[cls]
    paths = [relative(root, root / ARTWORK_DIR / (name + '.png')),
             relative(root, root / CACHE_DIR / (name + '.npz'))]
    declared = row.get('source_artwork')
    if declared and declared != paths[0]:
        raise ValueError('source artwork does not match generator class: ' + row.get('image', '?'))
    if any(not (root / path).is_file() for path in paths):
        raise ValueError('generator artwork/cache is missing: ' + name)
    return paths


def tone_conditions(row):
    tone = row.get('tone')
    tone = tone.get('normal_or_dark') if isinstance(tone, dict) else tone
    if not tone:
        tone = row.get('normal_or_dark', row.get('brightness'))
    if tone == 'dark_low_saturation':
        tone = 'dark'
    if tone not in ('normal', 'dark'):
        raise ValueError('normal/dark condition is missing: ' + row.get('image', '?'))
    return ['dark', 'low_light', 'low_saturation'] if tone == 'dark' else ['normal']


def _reference(refs, image, manifest, index):
    selected = [entry for entry in refs.get(image, []) if entry.get('manifest') == manifest
                and entry.get('record_index') == index and entry.get('relation')]
    if not selected:
        raise ValueError('indexed batch record missing for ' + image)
    return selected


def reset_selection(source, batches, refs, retain_prior=True, artwork_by_image=None):
    """Keep original nodes, reselect exactly the 54 reviewed video composites, add new batches."""
    source = copy.deepcopy(source)
    artwork_by_image = artwork_by_image or {}
    nodes = source['samples']
    for node in nodes:
        if node.get('split') in SPLITS:
            node['split'] = 'unassigned'
    retained = []
    for node in nodes:
        if (retain_prior and node.get('image', '').startswith(PRIOR + '/images/train/') and
            node.get('parent_video') == PRIOR_VIDEO and
            node.get('tasks', {}).get('waste_sticker', {}).get('state') == 'complete_positive'):
            node['split'] = 'train'
            node['background'] = {'kind': 'real', 'source_group': PRIOR_VIDEO}
            retained.append(node)
    if retain_prior and len(retained) != 54:
        raise ValueError('expected exactly 54 prior reviewed real-background positives; found ' + str(len(retained)))
    image_to_id = {node['image']: node['id'] for node in nodes if node.get('image') and node.get('id')}
    ids = {node['id'] for node in nodes if node.get('id')}
    seen = set()
    report = {split: {'images': 0, 'positive': 0, 'negative': 0, 'blender_background': 0,
                      'non_blender_positive_videos': Counter(), 'non_blender_background_groups': Counter(),
                      'blender_view_groups': Counter(), 'dark_low_saturation': 0} for split in SPLITS}
    report['train']['images'] += len(retained)
    report['train']['positive'] += len(retained)
    if retained:
        report['train']['non_blender_positive_videos'][PRIOR_VIDEO] += len(retained)
        report['train']['non_blender_background_groups'][PRIOR_VIDEO] += len(retained)
    for old in retained:
        conditions = old['tasks']['waste_sticker'].get('strata', {}).get('conditions', [])
        if 'low_light' in conditions and 'low_saturation' in conditions:
            report['train']['dark_low_saturation'] += 1
    for manifest, entries in batches:
        for index, row in enumerate(entries):
            image, label, split = row['image'], row['label'], row['split']
            if split not in SPLITS or image in seen:
                raise ValueError('invalid split or repeated image: ' + image)
            seen.add(image)
            info = facts(row)
            background = info['parent']
            parent_id = image_to_id.get(background)
            if not parent_id:
                parent_id = 'sticker-scene-parent:' + background
                if parent_id in ids:
                    raise ValueError('duplicate parent node')
                node = {'id': parent_id, 'image': background, 'split': 'unassigned',
                        'source_review_evidence': row.get('source_manifest', manifest)}
                if info['video']:
                    node['parent_video'] = info['video']
                if info['scene']:
                    node['mother_scene'] = info['scene']
                nodes.append(node)
                image_to_id[background] = parent_id
                ids.add(parent_id)
            positive = row.get('class_id') is not None
            conditions = tone_conditions(row)
            state = 'complete_positive' if positive else 'complete_negative'
            processing = {'resize': 'background and visible objects composited at 640x480; see batch parameters',
                          'compression': 'PNG lossless', 'blur': 'see batch transformation',
                          'origin_review': 'confirmed'}
            # Negative scenes have no newly pasted target; their final tone change is augmentation.
            method = 'composite' if positive or not info['blender'] else 'augmentation'
            processing['background_origin' if method == 'composite' else 'parent_origin'] = info['background_origin']
            if info['blender']:
                conditions.append('blender_scene_background')
            else:
                conditions.append('real_video_frame_background')
            reference = _reference(refs, image, manifest, index)
            if positive and len(artwork_by_image.get(image, [])) != 2:
                raise ValueError('positive missing artwork and cache source: ' + image)
            sample = {'id': 'sticker-scene:' + image, 'image': image, 'split': split,
                      'image_size': row.get('image_size', [640, 480]),
                      'source_group': 'waste_sticker_blender_background' if info['blender'] else 'waste_sticker_real_video_background',
                      'source_kind': 'derived_image', 'generation_method': method,
                      'background': ({'kind': 'blender', 'source_group': info['scene'], 'view_group': info['view_group']}
                                     if info['blender'] else {'kind': 'real', 'source_group': info['video']}),
                      'parent_ids': [parent_id], 'source_assets': artwork_by_image.get(image, []) if positive else [],
                      'mother_scene': info['scene'], 'image_processing': processing,
                      'lineage_review': 'confirmed', 'near_duplicate_review': 'confirmed',
                      'near_duplicate_review_scope': 'source gate exact-image and known diagnostic checks; shared source recorded',
                      'review_evidence': manifest, 'selected_tasks': ['waste_sticker'],
                      'indexed_source_references': reference,
                      'tasks': {'waste_sticker': {'state': state, 'label': label, 'review_evidence': manifest,
                          'strata': {'difficulty': 'scene_visible_sticker' if positive else 'scene_negative',
                                     'conditions': conditions, 'target_states': [row['source_name']] if positive else []},
                          'strata_review': 'confirmed'}},
                      'selection_evidence': {'method': 'known synthetic placement and structural check',
                                             'per_image_human_review': False}}
            if info['video']:
                sample['parent_video'] = info['video']
                foreground = row.get('source_render')
                if foreground:
                    foreground_id = image_to_id.get(foreground)
                    if not foreground_id:
                        foreground_id = 'sticker-scene-parent:' + foreground
                        if foreground_id in ids:
                            raise ValueError('duplicate foreground parent node')
                        nodes.append({'id': foreground_id, 'image': foreground, 'split': 'unassigned',
                                      'source_review_evidence': row.get('source_scene_manifest', manifest)})
                        image_to_id[foreground] = foreground_id
                        ids.add(foreground_id)
                    sample['parent_ids'].append(foreground_id)
            if sample['id'] in ids:
                raise ValueError('scene selection already present: ' + image)
            ids.add(sample['id'])
            nodes.append(sample)
            stats = report[split]
            stats['images'] += 1
            stats['positive' if positive else 'negative'] += 1
            stats['blender_background'] += int(info['blender'])
            if info['blender']:
                stats['blender_view_groups'][info['view_group']] += 1
            elif info['video']:
                stats['non_blender_background_groups'][info['video']] += 1
            if positive and info['video']:
                stats['non_blender_positive_videos'][info['video']] += 1
            stats['dark_low_saturation'] += int('low_light' in conditions and 'low_saturation' in conditions)
    total = sum(report[s]['images'] for s in SPLITS)
    blender = sum(report[s]['blender_background'] for s in SPLITS)
    groups = set().union(*(set(report[s]['non_blender_background_groups']) for s in SPLITS))
    views = set().union(*(set(report[s]['blender_view_groups']) for s in SPLITS))
    if not total or blender / total < .4:
        raise ValueError('Blender background fraction is less than 40 percent')
    if len(views) < 2:
        raise ValueError('Blender backgrounds require at least two camera view groups')
    if groups and len(groups) < 2:
        raise ValueError('non-Blender backgrounds require at least two actual parent groups')
    if any(not report[s]['dark_low_saturation'] for s in SPLITS):
        raise ValueError('each split requires dark, low-saturation examples')
    for stats in report.values():
        stats['blender_background_fraction'] = stats['blender_background'] / stats['images'] if stats['images'] else 0
        stats['non_blender_positive_videos'] = dict(stats['non_blender_positive_videos'])
        stats['non_blender_background_groups'] = dict(stats['non_blender_background_groups'])
        stats['blender_view_groups'] = dict(stats['blender_view_groups'])
    source['purpose'] = ('V4 waste sticker: Blender backgrounds' +
                         (' and real-video background groups' if groups else ' only'))
    source['scope'] = ('Generated scene composites; shared fitted arena, artworks and video sources across splits '
                       'are recorded; test is not used for tuning or selection.')
    return source, report


def inspect_records(root, batches):
    for manifest, records in batches:
        for row in records:
            image = root / relative(root, row['image'])
            label = root / relative(root, row['label'])
            if row['split'] not in SPLITS:
                raise ValueError('invalid split')
            with Image.open(image) as im:
                im.load()
                if im.size != (640, 480):
                    raise ValueError('expected 640x480: ' + str(image))
            lines = [line.split() for line in label.read_text(encoding='utf8').splitlines() if line.strip()]
            cls = row.get('class_id')
            if cls is None:
                if lines:
                    raise ValueError('negative label is nonempty')
            elif (not isinstance(cls, int) or not 0 <= cls < 24 or len(lines) != 1 or len(lines[0]) != 5 or
                  int(lines[0][0]) != cls or any(not 0 <= float(value) <= 1 for value in lines[0][1:])):
                raise ValueError('invalid positive label: ' + str(label))


def configured_copy(config, distribution, batches, report, retain_prior=True):
    config, distribution = copy.deepcopy(config), copy.deepcopy(distribution)
    overrides = {s: [] for s in SPLITS}
    if retain_prior:
        overrides['train'].append({'images': PRIOR + '/images/train', 'labels': PRIOR + '/labels/train',
                                   'source_group': 'reviewed_two_bin_train'})
    for manifest, records in batches:
        for split in SPLITS:
            selected = [row for row in records if row['split'] == split]
            if not selected:
                continue
            images = {str(Path(row['image']).parent).replace('\\', '/') for row in selected}
            labels = {str(Path(row['label']).parent).replace('\\', '/') for row in selected}
            if len(images) != 1 or len(labels) != 1:
                raise ValueError('each batch split must use one images directory and one labels directory')
            overrides[split].append({'images': images.pop(), 'labels': labels.pop(), 'source_group': manifest})
    if any(not overrides[s] for s in SPLITS):
        raise ValueError('each split needs an image/label source')
    config['data_overrides'] = overrides
    real_present = any(report[split]['non_blender_background_groups'] for split in SPLITS)
    config['source_notes'] = ['来源范围、每 split 数量及 Blender 背景来源组见分发清单；测试不得用于调参或选模。',
                              'Blender 背景与屏幕空间目标为合成，母场景及图案可跨集合复用。']
    if real_present:
        config['source_notes'].append('实拍视频背景及其派生可跨集合复用，并共享原始父视频。')
    distribution['scope'] = ('Blender 背景' + ('与实拍视频背景' if real_present else '') +
                             '的贴纸合成；来源关系与各集合数量见 selection_summary；测试不用于调参或选模。')
    distribution['selection_summary'] = report
    # Current selection evidence only; the source gate retains historical parent nodes.
    distribution['additional_records'] = [manifest for manifest, _ in batches]
    if retain_prior:
        distribution['additional_records'] += [PRIOR + '/selection.json',
            'datasets/exports/waste_bin/waste_sticker_v4_multibin_candidate_20261113/two_bin_35px_group_evidence.json']
    return config, distribution


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', action='append', required=True, help='workspace-relative batch manifest; repeat for each batch')
    parser.add_argument('--exclude-prior-real', action='store_true', help='do not reselect the 54 old real-video composites')
    args = parser.parse_args(argv)
    manifests = [relative(ROOT, value) for value in args.manifest]
    if len(set(manifests)) != len(manifests):
        parser.error('duplicate manifest')
    source = json.loads((ROOT / SOURCE).read_text(encoding='utf8'))
    config = yaml.safe_load((ROOT / CONFIG).read_text(encoding='utf8'))
    distribution = json.loads((ROOT / DISTRIBUTION).read_text(encoding='utf8'))
    batches = [(name, json.loads((ROOT / name).read_text(encoding='utf8'))['records']) for name in manifests]
    classes = yaml.safe_load((ROOT / CLASSES).read_text(encoding='utf8'))
    names = [item['name'] for item in classes['items']]
    if len(names) != 24 or len(set(names)) != 24:
        raise ValueError('expected 24 unique generator classes')
    artwork_by_image = {}
    procedural = {ROOT / SOURCE, ROOT / CONFIG, ROOT / DISTRIBUTION, ROOT / CLASSES, *(ROOT / name for name in manifests)}
    media = set()
    for manifest, rows in batches:
        for row in rows:
            media.add(ROOT / relative(ROOT, row['image']))
            assets = artwork_paths(ROOT, row, names)
            artwork_by_image[row['image']] = assets
            media.update(ROOT / value for value in assets)
            info = facts(row)
            media.add(ROOT / relative(ROOT, info['parent']))
            if info['video']:
                media.add(ROOT / relative(ROOT, info['video']))
                if row.get('source_render'):
                    media.add(ROOT / relative(ROOT, row['source_render']))
            procedural.add(ROOT / relative(ROOT, row['label']))
    for path in gate._selection_files(ROOT, source):
        if path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.mp4', '.mov', '.avi', '.npz'):
            media.add(path)
        else:
            procedural.add(path)
    outputs = [ROOT / SOURCE, ROOT / CONFIG, ROOT / DISTRIBUTION]
    codes = [Path(__file__).resolve(), ROOT / 'lab/scripts/data/v4_source_gate.py']
    procedural.difference_update(codes)
    with generation_guard('waste_sticker:scene_selection_v4', sorted(media), outputs, {'manifests': manifests, 'retain_prior_real': not args.exclude_prior_real},
                          code_paths=codes, procedural_inputs=sorted(procedural), inplace_paths=outputs, root=ROOT) as run:
        if run.reused:
            print('Existing scene selection reused')
            return
        inspect_records(ROOT, batches)
        selected, report = reset_selection(source, batches, run.source_refs, retain_prior=not args.exclude_prior_real,
                                           artwork_by_image=artwork_by_image)
        configured, distribution = configured_copy(config, distribution, batches, report, retain_prior=not args.exclude_prior_real)
        checked = gate.audit(ROOT, selected)
        if checked['status'] != 'passed':
            raise ValueError('source gate failed: ' + json.dumps(checked['errors'][:10], ensure_ascii=False))
        (ROOT / SOURCE).write_text(json.dumps(selected, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        (ROOT / CONFIG).write_text(yaml.safe_dump(configured, allow_unicode=True, sort_keys=False), encoding='utf8')
        (ROOT / DISTRIBUTION).write_text(json.dumps(distribution, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
