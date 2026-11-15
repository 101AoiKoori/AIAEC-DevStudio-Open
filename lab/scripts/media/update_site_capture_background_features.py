"""Measure full-frame appearance for the site-capture photo review table.

Semantic fields stay unreviewed until a visual assessment supplies them. This
tool only updates reproducible image statistics and preserves existing notes.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
import tempfile

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[3]
TABLE = ROOT / 'datasets/annotated/site_capture_20261112/photo_review.csv'
RAW = ROOT / 'datasets/raw/site_capture_20261112/onboard_camera_images'
SEMANTIC_DEFAULTS = {
    'background_view': 'unknown',
    'background_placement_plane': 'unknown',
    'background_target_presence': 'unknown',
    'background_use': 'unreviewed',
    'background_review_state': 'unreviewed',
}
MEASURE_FIELDS = ('mean_luma_0_255', 'mean_saturation_0_255')


def measure(path: Path) -> tuple[float, float]:
    with Image.open(path) as source:
        rgb = source.convert('RGB')
        luma = ImageStat.Stat(rgb.convert('L')).mean[0]
        saturation = ImageStat.Stat(rgb.convert('HSV').getchannel('S')).mean[0]
    return round(luma, 2), round(saturation, 2)


def main() -> None:
    with TABLE.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if len(rows) != 160 or len({row.get('asset_id') for row in rows}) != len(rows):
        raise ValueError('photo review is not the expected 160 unique photos')
    for field in (*SEMANTIC_DEFAULTS, *MEASURE_FIELDS):
        if field not in fields:
            fields.append(field)
    for row in rows:
        path = (ROOT / row['current_path']).resolve()
        if not path.is_relative_to(RAW.resolve()) or not path.is_file():
            raise ValueError('photo path missing or outside onboard batch: ' + row['current_path'])
        for field, default in SEMANTIC_DEFAULTS.items():
            row[field] = row.get(field) or default
        luma, saturation = measure(path)
        row['mean_luma_0_255'] = f'{luma:.2f}'
        row['mean_saturation_0_255'] = f'{saturation:.2f}'
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='',
                                     dir=TABLE.parent, suffix='.csv.tmp', delete=False) as handle:
        staged = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(staged, TABLE)
    print(f'measured {len(rows)} photos in {TABLE.relative_to(ROOT).as_posix()}')


if __name__ == '__main__':
    main()
