"""Prepare the reviewed vehicle-view batches for the existing V4 source gate.

No generation or training occurs here. The input QA is a separate, explicit
review artifact; a candidate manifest alone is not an acceptance decision.
"""
import argparse
import copy
import filecmp
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard

SOURCE = ROOT / 'datasets/annotated/e_bike/v4_858_source_20261113/staged_source_gate.json'
DISTRIBUTION = SOURCE.with_name('distribution_manifest.json')
CONFIGS = [ROOT / 'lab/features/e_bike/training/config/e_bike_v4_reviewed_858.yaml',
           ROOT / 'lab/features/e_bike/training/config/e_bike_v4.yaml']
REAL = ROOT / 'datasets/exports/e_bike/v4_view_filtered_20261113/manifest.json'
BLENDER = ROOT / 'datasets/exports/e_bike/v4_vehicle_view_arena_20261113/manifest.json'
MASTER = 'datasets/assets/blender/shared/arena_4m2_pbr_20261112/arena_master_user_20261113.blend'
SPLITS = ('train', 'val', 'test')


def rel(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def workspace(path):
    p = (ROOT / path).resolve()
    if not p.is_relative_to(ROOT.resolve()) or not p.is_file():
        raise ValueError('Missing or non-workspace input: ' + str(path))
    return p


def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _labels(path, size):
    values = []
    for line in path.read_text(encoding='utf8').splitlines():
        fields = line.split()
        if len(fields) != 5 or fields[0] not in ('0', '1'):
            raise ValueError('Invalid YOLO label: ' + str(path))
        nums = list(map(float, fields[1:]))
        x, y, w, h = nums
        if not (0 < w <= 1 and 0 < h <= 1 and 0 <= x-w/2 and x+w/2 <= 1.000001
                and 0 <= y-h/2 and y+h/2 <= 1.000001):
            raise ValueError('Invalid YOLO box: ' + str(path))
        values.append((int(fields[0]), nums))
    return values


def _validate_rows(real, blender, expected=True):
    rows = real['records'] + blender['records']
    counts = Counter(x['split'] for x in rows)
    if expected and (Counter(x['split'] for x in real['records']) != {'train': 15}
                     or Counter(x['split'] for x in blender['records']) != {'train': 160, 'val': 40, 'test': 40}):
        raise ValueError('Unexpected batch split counts')
    if set(counts) != set(SPLITS):
        raise ValueError('All splits must be present')
    groups = defaultdict(set)
    sizes = defaultdict(list)
    for row in rows:
        split = row['split']
        image, label = workspace(row['image']), workspace(row['label'])
        with Image.open(image) as im:
            size = list(im.size)
            im.verify()
        labels = _labels(label, size)
        if row in real['records']:
            bg = {'kind': 'real', 'source_group': row['parent_video']}
            workspace(bg['source_group'])
            if row.get('source_manifest'):
                review = load(workspace(row['source_manifest']))['records'][row['source_record_index']]
                if review['image'] != row['source_image'] or review['label'] != row['source_label'] or review['source_video'] != row['parent_video']:
                    raise ValueError('Reviewed real source record does not match')
            if row['label_state'] not in ('complete_negative', 'complete_positive'):
                raise ValueError('Unreviewed real label state')
            if bool(labels) != (row['label_state'] == 'complete_positive'):
                raise ValueError('Real label state disagrees with label content')
            if not filecmp.cmp(image, workspace(row['source_image']), shallow=False) or not filecmp.cmp(label, workspace(row['source_label']), shallow=False):
                raise ValueError('Real copy differs from reviewed source')
        else:
            bg = row['background']
            if (bg.get('kind') != 'blender' or bg.get('source_group') != MASTER
                    or bg.get('view_group') != Path(row['background_image']).stem
                    or row.get('mother_scene') != MASTER or row.get('background_origin') != 'blender'):
                raise ValueError('Invalid Blender background provenance')
            workspace(MASTER)
            workspace(row['background_image'])
            if len(labels) != len(row['objects']):
                raise ValueError('Object/label count mismatch')
            for (cls, box), obj in zip(labels, row['objects']):
                if cls != {'upright': 0, 'fallen': 1}[obj['class']]:
                    raise ValueError('Object/label class mismatch')
                x1, y1, x2, y2 = obj['bbox_xyxy']
                expected_box = [(x1+x2)/(2*size[0]), (y1+y2)/(2*size[1]),
                                (x2-x1)/size[0], (y2-y1)/size[1]]
                if any(abs(a-b) > 0.00002 for a, b in zip(box, expected_box)):
                    raise ValueError('Object/label box mismatch')
                workspace(obj['visible_mask'])
        groups[bg['kind']].add(bg['source_group'])
        if bg['kind'] == 'blender':
            groups['views'].add(bg['view_group'])
        for prior_split, prior in sizes[image.stat().st_size]:
            if prior_split != split and filecmp.cmp(image, prior, shallow=False):
                raise ValueError('Identical image across splits')
        sizes[image.stat().st_size].append((split, image))
    if len(groups['real']) < 2 or len(groups['views']) < 2 or len(blender['records']) / len(rows) < .4:
        raise ValueError('Background diversity or Blender fraction below requirement')
    result = {}
    for split in (*SPLITS, 'total'):
        items = [r for r in rows if split == 'total' or r['split'] == split]
        n = sum(r in blender['records'] for r in items)
        result[split] = {'images': len(items), 'blender_count': n, 'blender_ratio': n / len(items),
                         'blender_view_groups': sorted({r['background']['view_group'] for r in items if r in blender['records']}),
                         'real_source_groups': sorted({r['parent_video'] for r in items if r in real['records']})}
    return result


def _plan(source, distribution, configs, real, blender, qa, qa_path, expected=True):
    stats = _validate_rows(real, blender, expected)
    if (qa.get('status') != 'passed' or qa.get('structural_error_count') != 0
            or qa.get('images_checked') != len(blender['records'])
            or qa.get('representative_visual_review') != 'passed'
            or qa.get('manifest') != rel(BLENDER)):
        raise ValueError('QA must explicitly cover all structures and pass representative visual review')
    workspace(qa['visual_evidence'])
    source, distribution = copy.deepcopy(source), copy.deepcopy(distribution)
    selected = [r for r in source['samples'] if r.get('split') in SPLITS and 'e_bike' in r.get('selected_tasks', [])]
    if expected and Counter(r['split'] for r in selected) != {'train': 930, 'val': 28, 'test': 48}:
        raise ValueError('Unexpected existing selected records')
    for r in selected:
        r['split'] = 'exclude'
    ids = {r['id'] for r in source['samples']}
    for batch_name, records in (('real', real['records']), ('blender', blender['records'])):
        for index, row in enumerate(records):
            parent_id, sample_id = f'vehicle_view_{batch_name}_parent_{index}', f'vehicle_view_{batch_name}_{index}'
            if parent_id in ids or sample_id in ids:
                raise ValueError('Batch already registered')
            parent_image = row['source_image'] if batch_name == 'real' else row['background_image']
            original = None
            if batch_name == 'real' and row.get('source_gate_manifest'):
                if row['source_gate_manifest'] != rel(SOURCE):
                    raise ValueError('Unexpected source gate')
                original = source['samples'][row['source_gate_record_index']]
                if original['image'] != row['source_image'] or original['tasks']['e_bike']['label'] != row['source_label']:
                    raise ValueError('Source gate parent does not match')
                parent_id = original['id']
            else:
                source['samples'].append({'id': parent_id, 'image': parent_image, 'parent_ids': [],
                                          'split': 'unassigned', 'existence': 'present'})
            bg = ({'kind': 'real', 'source_group': row['parent_video']} if batch_name == 'real' else row['background'])
            state = row['label_state'] if batch_name == 'real' else ('complete_positive' if row['objects'] else 'complete_negative')
            task = {'state': state, 'label': row['label'], 'review_evidence': rel(REAL if batch_name == 'real' else qa_path),
                    'label_review': ('Recorded source AI review, no per-image human verification' if batch_name == 'real'
                                     else 'Generated visible boxes, structural QA and representative AI visual review; no per-image human acceptance.')}
            if batch_name == 'blender' and row['tone_parameters']['value_gain'] < 1 and row['tone_parameters']['saturation_gain'] < 1:
                task.update({'strata': {'conditions': ['low_light', 'low_saturation']}, 'strata_review': 'confirmed',
                             'strata_review_evidence': rel(BLENDER), 'strata_scope': 'relative simulated gains; illumination not measured'})
            assets = []
            if batch_name == 'blender':
                assets = [x['foreground_source']['path'] for x in row['objects']]
                if row.get('person_source'):
                    assets.append(row['person_source']['path'])
            source['samples'].append({'id': sample_id, 'image': row['image'], 'split': row['split'],
                'source_group': bg['source_group'], 'background': bg, 'source_kind': 'derived_image',
                'generation_method': original['generation_method'] if original else ('augmentation' if batch_name == 'real' else 'composite'),
                'parent_ids': [parent_id], 'image_size': list(Image.open(workspace(row['image'])).size),
                'lineage_review': 'confirmed', 'near_duplicate_review': 'confirmed',
                'review_evidence': rel(REAL if batch_name == 'real' else qa_path),
                'physical_assets': None, 'source_assets': assets or None,
                'mother_scene': MASTER if batch_name == 'blender' else None,
                'image_processing': original['image_processing'] if original else {'resize': 'inherited' if batch_name == 'real' else 'recorded composite',
                                     'compression': 'PNG', 'blur': 'not recorded', 'tone_profile': row.get('tone')},
                'selected_tasks': ['e_bike'], 'tasks': {'e_bike': task},
                'provenance': {'batch_manifest': rel(REAL if batch_name == 'real' else BLENDER),
                               'batch_record_index': index, 'batch_record': row},
                'unknowns': ['Physical asset identity and unrecorded near duplicates remain unknown.',
                             'AI review is not per-image human acceptance; shared sources are not independent captures.']})
    source['scope'] = 'Candidate vehicle-view dataset; shared-source synthetic views and AI review; no independent-scene or deployment claim.'
    distribution['scope'] = source['scope']
    distribution['background_distribution'] = stats
    for p in (rel(REAL), rel(BLENDER), rel(qa_path)):
        if p not in distribution['additional_records']:
            distribution['additional_records'].append(p)
    result_configs = []
    for text in configs:
        cfg = yaml.safe_load(text)
        cfg['test_unavailable'] = False
        cfg['description'] = '已核有限母样的候选微调；开发和测试为同源 Blender 场景与车辆前景合成。'
        cfg['source_notes'] = [
            '训练 175 张、开发 40 张、测试 40 张；实拍训练图 15 张，其余为 Blender 场景背景合成。',
            '实拍来源含两个原始视频；Blender 背景来自同一母场景的多个视角，前景素材复用。',
            '标签经过记录的结构检查和代表性 AI 目视审核，未逐图人工验收；未知近重复与物料身份保留为未知。',
            '开发和测试同源，不代表独立采集、独立场次、部署表现或实车验收。',
            '车载参考 640×480；制作数据不超过 720p。放大不增加真实细节。']
        cfg['data_overrides'] = {split: ([{'images': rel(REAL.parent / 'images' / split), 'labels': rel(REAL.parent / 'labels' / split)}] if split == 'train' else []) +
            [{'images': rel(BLENDER.parent / 'images' / split), 'labels': rel(BLENDER.parent / 'labels' / split)}] for split in SPLITS}
        result_configs.append(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    return source, distribution, result_configs, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qa', type=Path, required=True, help='Explicit reviewed QA JSON in workspace')
    args = parser.parse_args()
    qa_path = workspace(args.qa)
    real, blender, qa = load(REAL), load(BLENDER), load(qa_path)
    media, procedural = [], [SOURCE, DISTRIBUTION, *CONFIGS, REAL, BLENDER, qa_path, ROOT / yaml.safe_load(CONFIGS[0].read_text(encoding='utf8'))['data']]
    media.append(workspace(qa['visual_evidence']))
    for row in real['records']:
        media.extend(workspace(row[k]) for k in ('image', 'source_image', 'background_image', 'parent_video'))
        procedural.extend(workspace(row[k]) for k in ('label', 'source_label'))
        for key in ('source_manifest', 'source_gate_manifest', 'source_gate_review_evidence', 'label_review_evidence'):
            if row.get(key): procedural.append(workspace(row[key]))
        procedural.extend(workspace(ref['manifest']) for ref in row.get('source_database_refs', []))
    procedural.append(workspace(MASTER))
    def add_refs(value):
        if isinstance(value, dict):
            if isinstance(value.get('manifest'), str) and 'record_index' in value:
                procedural.append(workspace(value['manifest']))
            for child in value.values(): add_refs(child)
        elif isinstance(value, list):
            for child in value: add_refs(child)
    add_refs(blender)
    for row in blender['records']:
        media.extend(workspace(row[k]) for k in ('image', 'background_image'))
        procedural.append(workspace(row['label']))
        for obj in row['objects']:
            media.append(workspace(obj['visible_mask']))
            media.extend(workspace(obj['foreground_source'][k]) for k in ('path', 'source_frame', 'source_video'))
        if row.get('person_source'):
            media.append(workspace(row['person_source']['path']))
    output = [SOURCE, DISTRIBUTION, *CONFIGS]
    with generation_guard('e_bike.integrate_v4_vehicle_view', list(dict.fromkeys(media)), output,
                          {'real_manifest': rel(REAL), 'blender_manifest': rel(BLENDER), 'qa': rel(qa_path), 'version': 1},
                          procedural_inputs=list(dict.fromkeys(procedural)), code_paths=[Path(__file__), ROOT / 'lab/training_common/production_guard.py'],
                          inplace_paths=output, root=ROOT) as run:
        if run.reused: return
        source, distribution, configs, stats = _plan(load(SOURCE), load(DISTRIBUTION),
            [p.read_text(encoding='utf8') for p in CONFIGS], real, blender, qa, qa_path)
        for path, obj in ((SOURCE, source), (DISTRIBUTION, distribution)):
            path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        for path, text in zip(CONFIGS, configs): path.write_text(text, encoding='utf8')
        print(json.dumps(stats, ensure_ascii=False))


if __name__ == '__main__':
    main()
