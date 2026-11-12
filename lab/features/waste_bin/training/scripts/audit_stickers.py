"""Inspect an original YOLO zip without modifying or extracting it."""
import argparse
import collections
import csv
import io
import json
import zipfile

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive')
    args = parser.parse_args()
    with zipfile.ZipFile(args.archive) as z:
        paths = z.namelist()
        prefix = next(p[:-len('manifest.csv')] for p in paths if p.endswith('/manifest.csv'))
        rows = list(csv.DictReader(io.StringIO(z.read(prefix + 'manifest.csv').decode('utf-8-sig').replace('\r', ''))))
        counts = collections.Counter()
        issues = []
        for split in ('train', 'val'):
            images = [p for p in paths if p.startswith(prefix + 'images/' + split + '/') and p.lower().endswith('.jpg')]
            for image in images:
                counts[split + '_images'] += 1
                try:
                    with Image.open(io.BytesIO(z.read(image))) as im:
                        im.verify()
                except Exception as exc:
                    issues.append(f'{image}: {exc}')
                label = image.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
                if label not in paths:
                    issues.append(f'missing label: {label}')
                    continue
                lines = z.read(label).decode('utf-8').splitlines()
                if not lines:
                    counts[split + '_negative'] += 1
                for line in lines:
                    try:
                        c, x, y, w, h = map(float, line.split())
                        assert c.is_integer() and 0 <= c < 24 and w > 0 and h > 0
                        assert 0 <= x-w/2 and x+w/2 <= 1 and 0 <= y-h/2 and y+h/2 <= 1
                    except (ValueError, AssertionError):
                        issues.append(f'invalid label: {label}: {line}')
                    counts[split + '_instances'] += 1
        source = {s: sorted({r['icon'] for r in rows if r['split'] == s}) for s in ('train', 'val')}
        print(json.dumps({'counts': counts, 'source_icons_per_split': {s: len(v) for s, v in source.items()}, 'shared_icons': len(set(source['train']) & set(source['val'])), 'manifest_items': dict(collections.Counter(r['icon'] for r in rows)), 'issues': issues}, ensure_ascii=False, indent=2))
        if issues:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
