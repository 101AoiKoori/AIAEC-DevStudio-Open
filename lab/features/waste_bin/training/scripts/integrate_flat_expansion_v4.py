"""Select the existing guarded flat expansion in the waste-sticker V4 manifests."""
import copy
import json
import sys
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'lab/scripts/data'))
from lab.training_common.production_guard import generation_guard
import v4_source_gate as gate

BATCH = 'datasets/exports/waste_bin/waste_sticker_v4_flat_expansion_20261113'
MANIFEST = BATCH + '/manifest.json'
SOURCE = 'datasets/annotated/waste_sticker_v4_handoff_20261113/staged_master_with_test_source_gate.json'
CONFIG = 'lab/features/waste_bin/training/config/waste_sticker_v4_reviewed_master.yaml'
DISTRIBUTION = 'datasets/annotated/waste_sticker_v4_handoff_20261113/distribution_master_candidate.json'
GEN_CONFIG = 'Archive/Assets/working_sources/digital/waste_sticker/toolkit/configs/gen_default.yaml'
GENERATOR = 'lab/features/waste_bin/training/scripts/generate_flat_controlled_v4.py'


def candidate_rows(records, refs):
    """Describe generated labels, retaining indexed references to the batch records."""
    result = []
    for index, item in enumerate(records):
        image, label = item['image'], item['label']
        original = [x for x in refs.get(image, []) if x.get('manifest') == MANIFEST
                    and x.get('record_index') == index and x.get('relation')]
        if not original:
            raise ValueError('indexed original record missing for ' + image)
        positive = item['class_id'] is not None
        conditions = [item['normal_or_dark'], 'synthetic_flat_panel']
        if item['normal_or_dark'] == 'dark':
            conditions += ['low_light', 'low_saturation']
        result.append({
            'id': 'flat-expansion:' + image, 'image': image, 'image_size': [640, 480],
            'split': 'train', 'source_group': 'waste_sticker_controlled_flat',
            'source_kind': 'synthetic_scene', 'generation_method': 'synthetic_render',
            'parent_ids': [], 'source_assets': ([item['source_artwork']] if positive else []) + [GEN_CONFIG],
            'physical_assets': None, 'mother_scene': 'procedural single panel, 24 shared source artworks',
            'image_processing': {'resize': 'procedural panel rendered at 640x480' if not positive else
                'source artwork fitted to paper and projected to recorded quadrilateral at 640x480',
                'compression': 'PNG lossless', 'blur': 'none added', 'resolution_review': 'pending'},
            'lineage_review': 'confirmed', 'near_duplicate_review': 'confirmed',
            'near_duplicate_review_scope': 'source gate checks known exposure inventory and exact images; no exhaustive visual near-duplicate search',
            'review_evidence': MANIFEST, 'selected_tasks': ['waste_sticker'],
            'tasks': {'waste_sticker': {'state': 'complete_positive' if positive else 'complete_negative',
                'label': label, 'review_evidence': MANIFEST,
                'strata': {'difficulty': 'controlled_large_paper' if positive else 'synthetic_empty_panel',
                           'conditions': conditions, 'target_states': [item['source_name']] if positive else []},
                'strata_review': 'confirmed'}},
            'indexed_source_references': original,
            'selection_evidence': {'method': 'generator geometry and image/label structural check; one dark-condition sample visually inspected',
                                   'per_image_human_review': False},
        })
    return result


def integrate(source, config, distribution, records, refs):
    source, config, distribution = copy.deepcopy(source), copy.deepcopy(config), copy.deepcopy(distribution)
    rows = candidate_rows(records, refs)
    prior = {x['id'] for x in source['samples']}
    if any(x['id'] in prior for x in rows):
        raise ValueError('expansion already present; refuse duplicate selection')
    source['samples'].extend(rows)
    source['purpose'] = 'V4 2586 train / 48 development / 216 synthetic test; artwork and scene shared'
    source['scope'] = ('No direct generated image reuse across sets in checked inputs; 24 icon artworks '
                       'and procedural scene shared. Test not used for model selection or tuning. '
                       'Flat expansion uses generator labels and one representative visual inspection.')
    inputs = config['data_overrides']['train']
    images = BATCH + '/images/train'
    if any(x.get('images') == images for x in inputs):
        raise ValueError('expansion already present in train config')
    inputs.append({'images': images, 'labels': BATCH + '/labels/train', 'source_group': 'flat_expansion_train'})
    notes = config.get('source_notes', [])
    config['source_notes'] = [x for x in notes if not x.startswith('训练 1146')]
    config['source_notes'].insert(0, '训练 2586、开发 48、合成测试 216；新增平板 960 正例和 480 负例，测试不用于调参或选模。')
    distribution['scope'] = ('V4 训练 2586、开发 48、合成测试 216；三集合共享 24 图案母样及程序化场景。'
                             '两桶贴图为屏幕空间近似；测试仅用于最终测试，不用于调参或选模。')
    if MANIFEST not in distribution['additional_records']:
        distribution['additional_records'].append(MANIFEST)
    return source, config, distribution


def already_selected(source, config, distribution, records):
    """Check the existing selection without rewriting it or copying source records."""
    prefix = 'flat-expansion:'
    selected = [row for row in source['samples'] if row.get('id', '').startswith(prefix)]
    inputs = [row for row in config['data_overrides']['train']
              if row.get('images') == BATCH + '/images/train']
    if not selected and not inputs and MANIFEST not in distribution['additional_records']:
        return False
    by_id = {row['id']: row for row in selected}
    if (len(by_id) != len(records) or len(selected) != len(records) or len(inputs) != 1 or
            inputs[0].get('labels') != BATCH + '/labels/train' or
            distribution.get('source_manifest') != SOURCE or
            distribution['additional_records'].count(MANIFEST) != 1):
        raise ValueError('existing expansion selection is incomplete or inconsistent')
    for index, item in enumerate(records):
        row = by_id.get(prefix + item['image'])
        task = row.get('tasks', {}).get('waste_sticker', {}) if row else {}
        refs = row.get('indexed_source_references', []) if row else []
        expected_state = 'complete_positive' if item['class_id'] is not None else 'complete_negative'
        if (not row or row.get('image') != item['image'] or row.get('split') != 'train' or
                'waste_sticker' not in row.get('selected_tasks', []) or
                task.get('label') != item['label'] or task.get('state') != expected_state or
                not any(ref.get('manifest') == MANIFEST and ref.get('record_index') == index and
                        ref.get('relation') == 'describes' for ref in refs)):
            raise ValueError('existing expansion selection differs at record ' + str(index))
    return True


def validate_batch(root, batch):
    records = batch['records']
    if len(records) != 1440 or Counter(x['class_id'] for x in records) != {**{i: 40 for i in range(24)}, None: 480}:
        raise ValueError('expected 960 positives across 24 classes and 480 negatives')
    if Counter(x['normal_or_dark'] for x in records) != {'normal': 720, 'dark': 720}:
        raise ValueError('unexpected tone distribution')
    seeds = set()
    for row in records:
        if row['split'] != 'train' or row['seed'] in seeds:
            raise ValueError('invalid split or duplicate seed')
        seeds.add(row['seed'])
        image, label = root / row['image'], root / row['label']
        if not image.is_relative_to(root / BATCH / 'images/train') or not label.is_relative_to(root / BATCH / 'labels/train'):
            raise ValueError('unexpected batch path')
        with Image.open(image) as im:
            im.load()
            if im.size != (640, 480):
                raise ValueError('unexpected image dimensions: ' + row['image'])
        lines = [line.split() for line in label.read_text(encoding='utf8').splitlines() if line.strip()]
        if row['class_id'] is None:
            if lines:
                raise ValueError('negative has label: ' + row['label'])
        elif (len(lines) != 1 or len(lines[0]) != 5 or int(lines[0][0]) != row['class_id'] or
              any(not 0 <= float(value) <= 1 for value in lines[0][1:])):
            raise ValueError('invalid positive label: ' + row['label'])
    return records


def main():
    source = json.loads((ROOT / SOURCE).read_text(encoding='utf8'))
    config = yaml.safe_load((ROOT / CONFIG).read_text(encoding='utf8'))
    distribution = json.loads((ROOT / DISTRIBUTION).read_text(encoding='utf8'))
    batch = json.loads((ROOT / MANIFEST).read_text(encoding='utf8'))
    records = batch['records']
    old_refs = gate._selection_files(ROOT, source)
    procedural = {ROOT / SOURCE, ROOT / CONFIG, ROOT / DISTRIBUTION, ROOT / MANIFEST, ROOT / GENERATOR}
    media = {ROOT / row['image'] for row in records}
    procedural.update(ROOT / row['label'] for row in records)
    for path in old_refs:
        if path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.mp4', '.mov', '.avi', '.npz'):
            media.add(path)
        else:
            procedural.add(path)
    outputs = [ROOT / SOURCE, ROOT / CONFIG, ROOT / DISTRIBUTION]
    codes = [Path(__file__).resolve(), ROOT / GENERATOR, ROOT / 'lab/scripts/data/v4_source_gate.py']
    procedural.difference_update(codes)
    with generation_guard('waste_sticker:select_flat_expansion_20261113', sorted(media), outputs,
                          {'batch': MANIFEST, 'selection': 'all', 'split': 'train'},
                          code_paths=codes, procedural_inputs=sorted(procedural), inplace_paths=outputs,
                          root=ROOT) as run:
        if run.reused:
            print('Existing selected expansion reused')
            return
        if already_selected(source, config, distribution, records):
            print('Existing selected expansion verified; no files rewritten')
            return
        validate_batch(ROOT, batch)
        selected, configured, published = integrate(source, config, distribution, records, run.source_refs)
        checked = gate.audit(ROOT, selected)
        if checked['status'] != 'passed':
            raise ValueError('source gate failed: ' + json.dumps(checked['errors'][:10], ensure_ascii=False))
        (ROOT / SOURCE).write_text(json.dumps(selected, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        (ROOT / CONFIG).write_text(yaml.safe_dump(configured, allow_unicode=True, sort_keys=False), encoding='utf8')
        (ROOT / DISTRIBUTION).write_text(json.dumps(published, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        print(json.dumps({'status': 'selected', 'added': len(records), 'train': 2586,
                          'val': 48, 'test': 216}, ensure_ascii=False))


if __name__ == '__main__':
    main()
