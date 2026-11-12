"""Check generated images, YOLO labels, provenance, and split distributions."""
import argparse
import collections
import json
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=Path)
    args = parser.parse_args()
    root = args.dataset
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    names = json.loads((root / 'classes.json').read_text(encoding='utf-8'))
    errors = []
    counts = collections.defaultdict(collections.Counter)
    seen, seeds = set(), {}
    for rec in manifest['images']:
        path = Path(rec['image'])
        split, stem = path.parts[1], path.stem
        if str(path) in seen:
            errors.append(f'duplicate manifest path: {path}')
        seen.add(str(path))
        if rec['seed'] in seeds and seeds[rec['seed']] != split:
            errors.append(f'seed shared across splits: {rec["seed"]}')
        seeds[rec['seed']] = split
        image = root / path
        label = root / 'labels' / split / (stem + '.txt')
        meta = root / 'metadata' / split / (stem + '.json')
        try:
            with Image.open(image) as im:
                width, height = im.size
                im.verify()
            lines = label.read_text(encoding='utf-8').splitlines()
            stored = json.loads(meta.read_text(encoding='utf-8'))
            if stored != rec or len(lines) != len(rec['instances']) or len(lines) != len(rec['requested']):
                errors.append(f'provenance or instance mismatch: {path}')
            for line, instance in zip(lines, rec['instances']):
                cls, x, y, w, h = map(float, line.split())
                if not (cls.is_integer() and 0 <= cls < len(names) and names[int(cls)] == instance['name'] and
                        w > 0 and h > 0 and x-w/2 >= -1e-6 and x+w/2 <= 1.000001 and
                        y-h/2 >= -1e-6 and y+h/2 <= 1.000001):
                    errors.append(f'invalid label: {label}: {line}')
                actual_box = [(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height]
                if max(abs(a-b) for a, b in zip(actual_box, instance['xyxy'])) > 1.0:
                    errors.append(f'label and metadata box mismatch: {label}')
                counts[split][instance['name']] += 1
            counts[split][f'images_with_{len(lines)}_objects'] += 1
            counts[split][f'layout_{rec["layout"]}_with_{len(lines)}_objects'] += 1
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            errors.append(f'{path}: {exc}')
    for split in manifest['summary']:
        actual = {str(p.relative_to(root)) for p in (root / 'images' / split).glob('*.jpg')}
        expected = {s for s in seen if Path(s).parts[1] == split}
        if actual != expected:
            errors.append(f'{split}: image files and manifest differ')
        expected_stems = {Path(s).stem for s in expected}
        for directory, suffix in [('labels', '*.txt'), ('metadata', '*.json')]:
            found = {p.stem for p in (root / directory / split).glob(suffix)}
            if found != expected_stems:
                errors.append(f'{split}: {directory} files and manifest differ')
    print(json.dumps({'counts': counts, 'errors': errors}, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
