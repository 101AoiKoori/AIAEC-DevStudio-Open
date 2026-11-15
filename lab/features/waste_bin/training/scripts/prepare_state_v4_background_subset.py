"""Select a bounded subset of existing labelled waste-state training images."""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    source = (ROOT / args.manifest).resolve()
    out = (ROOT / args.out).resolve()
    owner = (ROOT / 'runs/waste_bin_state').resolve()
    if not source.is_relative_to(ROOT) or not out.is_relative_to(owner) or out == owner or args.count < 1:
        parser.error('input must be in workspace; output must be under runs/waste_bin_state')
    records = json.loads(source.read_text(encoding='utf8'))['records']
    candidates = [row for row in records if row.get('split') == 'train' and row.get('source_type') == 'legacy_crop']
    if len(candidates) < args.count:
        raise ValueError('not enough legacy cropped records: ' + str(len(candidates)))
    selected = random.Random(args.seed).sample(sorted(candidates, key=lambda row: row['id']), args.count)
    images = [ROOT / row['image'] for row in selected]
    labels = [ROOT / row['label'] for row in selected]
    sys.path.insert(0, str(ROOT))
    from lab.training_common.production_guard import generation_guard
    with generation_guard('waste_bin_state.background_subset:' + out.relative_to(ROOT).as_posix(),
                          images, [out], {'count': args.count, 'seed': args.seed},
                          code_paths=[Path(__file__).resolve()], procedural_inputs=[source, *labels], root=ROOT) as run:
        if run.reused:
            return
        (out / 'images/train').mkdir(parents=True)
        (out / 'labels/train').mkdir(parents=True)
        result = []
        for row, image, label in zip(selected, images, labels):
            image_out = out / 'images/train' / image.name
            label_out = out / 'labels/train' / label.name
            if image_out.exists() or label_out.exists():
                raise ValueError('duplicate selected name: ' + image.name)
            shutil.copyfile(image, image_out)
            shutil.copyfile(label, label_out)
            result.append({'id': row['id'], 'image': image_out.relative_to(ROOT).as_posix(),
                           'label': label_out.relative_to(ROOT).as_posix(), 'split': 'train',
                           'source_image': row['image'], 'source_label': row['label']})
        (out / 'data.yaml').write_text('names:\n  0: open\n  1: closed\ntrain: images/train\n', encoding='utf8')
        (out / 'manifest.json').write_text(json.dumps({'source_manifest': args.manifest, 'seed': args.seed,
            'records': result}, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        print(json.dumps({'count': len(result), 'output': out.relative_to(ROOT).as_posix()}))


if __name__ == '__main__':
    main()
