"""Assemble the measured sign train candidates and real-frame eval draft.

The result remains provisional: its source video and sign material are shared,
coverage is incomplete, and no final test or model-selection claim is made.
"""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/sign/.")

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT / 'lab/training_common')]
from camera_tone import source_role  # noqa: E402
from lab.scripts.data.v4_source_gate import build_queue  # noqa: E402

TRAIN = ROOT / 'datasets/exports/sign/v4_measured_multibackground_train_20261112'
EVAL = ROOT / 'datasets/exports/sign/v4_real_valtest_draft_20261112'
OUT = ROOT / 'runs/sign/v4_measured_dataset_draft_20261112'


def _main():
    if OUT.exists():
        raise ValueError('Output already exists; choose a new version')
    train = json.loads((TRAIN/'manifest.json').read_text(encoding='utf8'))
    evaluation = json.loads((EVAL/'manifest.json').read_text(encoding='utf8'))
    if train['status'] != 'complete_candidates' or evaluation['status'] != 'provisional_not_frozen':
        raise ValueError('Expected the measured candidate batch and provisional real-frame review')
    selected = [(TRAIN, row, 'train') for row in train['samples'] if row.get('split') == 'unassigned']
    selected += [(EVAL, row, row['split']) for row in evaluation['samples']]
    known = build_queue(ROOT)
    splits_by_parent = {}
    for _, row, split in selected:
        parent = row.get('background_parent') or row['source_image']
        if parent not in splits_by_parent and source_role(ROOT, ROOT/parent, known)[0] == 'regression':
            raise ValueError('Source is restricted to regression: ' + parent)
        if parent in splits_by_parent and splits_by_parent[parent] != split:
            raise ValueError('A parent frame crosses splits: ' + parent)
        splits_by_parent[parent] = split
    records, counts = [], Counter()
    try:
        for split in ('train', 'val', 'test'):
            for kind in ('images', 'labels'):
                (OUT/kind/split).mkdir(parents=True, exist_ok=True)
        for source, row, split in selected:
            image, label = source/row['image'], source/row['label']
            with Image.open(image) as decoded:
                decoded.load()
                w, h = decoded.size
            if min(w, h) > 720 or max(w, h) > 1280:
                raise ValueError('Image exceeds 720p: ' + str(image))
            values = label.read_text(encoding='utf8').strip().split()
            positive = row.get('state') not in ('negative',)
            if bool(values) != positive or (values and (len(values) != 5 or values[0] != '0')):
                raise ValueError('Label/instance state mismatch: ' + str(label))
            name = f'{split}_{row["id"]}'
            dest = OUT/'images'/split/(name + image.suffix.lower())
            target = OUT/'labels'/split/(name + '.txt')
            shutil.copy2(image, dest)
            shutil.copy2(label, target)
            if split == 'train':
                method = 'composite' if positive else ('augmentation' if row['tone'] != 'normal' else 'crop')
            else:
                method = ('augmentation' if row['tone'] != 'original' else
                          'crop' if row.get('crop_xyxy') else 'provided_video_frame')
            records.append({'id': name, 'image': dest.relative_to(ROOT).as_posix(),
                'label': target.relative_to(ROOT).as_posix(), 'split': split, 'state': 'positive' if positive else 'negative',
                'parent_image': row.get('background_parent') or row['source_image'],
                'source_image': image.relative_to(ROOT).as_posix(),
                'source_time_seconds': row['source_time_seconds'], 'source_review':
                    (source/'manifest.json').relative_to(ROOT).as_posix(),
                'generation_method': method,
                'background_image': ('datasets/annotated/sign/v4_training_backgrounds_20261112/' + row['background'])
                    if split == 'train' else None, 'foreground_mask': row.get('foreground_mask'),
                'foreground_video': train['foreground_source'] if split == 'train' and positive else None,
                'foreground_view': row.get('foreground_view') or row.get('view'),
                'state_detail': row['state'], 'tone': row.get('tone'), 'image_size': [w, h],
                'parameters': row.get('parameters') or row.get('tone_parameters'),
                'crop_xyxy': row.get('crop_xyxy'), 'split_status': 'provisional_not_frozen'})
            counts[(split, 'positive' if positive else 'negative')] += 1
        (OUT/'data.yaml').write_text(yaml.safe_dump({'names': {0: 'sign_fixture'},
            **{split: 'images/'+split for split in ('train', 'val', 'test')}}, allow_unicode=True), encoding='utf8')
        result = {'status': 'draft_not_deliverable', 'scope': '10 training backgrounds plus 5 evaluation '
            'frames from one video, shared sign material. No final test or target-device run.',
            'source_manifests': [(TRAIN/'manifest.json').relative_to(ROOT).as_posix(),
                                 (EVAL/'manifest.json').relative_to(ROOT).as_posix()],
            'distinct_parent_frames': len(splits_by_parent),
            'counts': {s: {k: counts[(s, k)] for k in ('positive', 'negative')} for s in ('train', 'val', 'test')},
            'missing': ['Broader source, difficulty and hard-negative coverage; source-gate audit; '
                        'confirmed frozen test and production distribution checks.'], 'records': records}
        (OUT/'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
        print(json.dumps({'counts': result['counts'], 'parents': len(splits_by_parent)}, ensure_ascii=False))
    except BaseException:
        (OUT/'FAILED.txt').write_text('Draft assembly failed; inspect before reuse.\n', encoding='utf8')
        raise


def main():
    raise RuntimeError("Retired dataset entry; use datasets/scripts task entry and its source.yaml")


if __name__ == '__main__':
    main()
