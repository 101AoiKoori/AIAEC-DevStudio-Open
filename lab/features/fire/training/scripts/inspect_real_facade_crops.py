"""Produce three bounded facade inspection candidates from existing task-use records."""
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard
from generate_fire_dataset import _generate, read_image, write_image

USE_MANIFEST = ROOT / 'datasets/annotated/fire/fire_sticker_v4_arena_handoff_20261113/fire_image_use_manifest_20261114.json'
ARTWORK = ROOT / 'Archive/Assets/working_sources/digital/fire/team_assets/image1.png'
OUTPUT = ROOT / 'runs/fire_sticker/real_facade_bounded_inspection_v3_20261114'
CROP = (1354, 1452, 2430, 2259)
# Coordinates are in the 600 x 450 visual inspection of each 1076 x 807 crop.
# These are small wall patches, not descriptions of the entire facade.
PATCHES = {
    'facade_A_04': [(180, 80), (220, 82), (220, 138), (180, 136)],
    'facade_B_51': [(318, 150), (334, 150), (334, 174), (318, 174)],
    'facade_C_33': [(230, 125), (248, 125), (248, 150), (230, 150)],
}


def select_sources(data):
    selected = {}
    for record in data['records']:
        for relation in record['task_use_status']:
            key = relation.get('source_id')
            if (key in PATCHES and (relation.get('task'), relation.get('split'),
                    relation.get('role'), relation.get('status')) ==
                    ('fire_sticker', 'train', 'background', 'candidate')):
                if key in selected:
                    raise ValueError('ambiguous source relation: ' + key)
                selected[key] = record['path']
    if set(selected) != set(PATCHES) or len(set(selected.values())) != len(PATCHES):
        raise ValueError('three distinct recorded photo sources are required')
    return selected


def main():
    sources = select_sources(json.loads(USE_MANIFEST.read_text(encoding='utf8')))
    if OUTPUT.exists():
        raise ValueError('inspection output already exists')
    for path in sources.values():
        source = (ROOT / path).resolve()
        if not source.is_file() or not any(source.is_relative_to(ROOT / base) for base in (
                'datasets/raw/fire', 'Archive/Assets/working_sources/legacy/fire', 'Archive/Assets/captures/fire')):
            raise ValueError('source photo missing or outside fire camera sources')
    inputs = [USE_MANIFEST, ARTWORK, *(ROOT / path for path in sources.values())]
    with generation_guard('fire.inspect_real_facade_crops:' + OUTPUT.relative_to(ROOT).as_posix(),
            inputs, [OUTPUT], {'crop_xyxy': list(CROP), 'source_ids': list(PATCHES), 'seed': 431},
            code_paths=[Path(__file__), ROOT / 'lab/features/fire/training/scripts/generate_fire_dataset.py'],
            procedural_inputs=[USE_MANIFEST], root=ROOT) as run:
        if run.reused:
            return
        OUTPUT.mkdir(parents=True)
        (OUTPUT / 'crops').mkdir()
        evidence = []
        x0, y0, x1, y1 = CROP
        for index, (source_id, path) in enumerate(sources.items()):
            image = read_image(ROOT / path)
            if image.shape[:2] != (4640, 3472):
                raise ValueError('unexpected source dimensions: ' + source_id)
            crop = image[y0:y1, x0:x1]
            crop_path = OUTPUT / 'crops' / (source_id + '.png')
            write_image(crop_path, crop)
            quad = [[round(x * (x1-x0) / 600, 2), round(y * (y1-y0) / 450, 2)]
                    for x, y in PATCHES[source_id]]
            entry = {'image': crop_path.relative_to(ROOT).as_posix(), 'split': 'train',
                     'review_state': 'ai_reviewed_empty_candidate', 'sticker_count': 1, 'quad': quad}
            source_file = OUTPUT / (source_id + '.json')
            source_file.write_text(json.dumps([entry], ensure_ascii=False, indent=2), encoding='utf8')
            destination = OUTPUT / ('sample_' + source_id)
            _generate(source_file, ARTWORK, destination, 1, 431 + index)
            name_dir = destination / 'labels' / 'name_candidate'
            name_dir.mkdir(parents=True)
            name_label = name_dir / 'fire_000001.txt'
            name_label.write_text('', encoding='utf8')
            # Crop-level observations are candidates.  No claim is made about the
            # rest of the original image or targets concealed behind openings.
            evidence.append({'source_id': source_id, 'parent': path, 'crop_xyxy': list(CROP),
                'crop': crop_path.relative_to(ROOT).as_posix(), 'inspection_size': [600, 450],
                'wall_patch_xy': quad, 'window_policy': 'only the marked local patch may receive artwork',
                'fire_sticker': 'no_recognizable_target_in_crop_candidate',
                'building_name_paper': 'no_recognizable_target_in_crop_candidate',
                'building_name_label': name_label.relative_to(ROOT).as_posix(),
                'review_method': 'AI visual inspection of full cropped view',
                'scope_limit': 'original image outside crop and objects behind openings are unknown',
                'sample': destination.relative_to(ROOT).as_posix(), 'seed': 431 + index})
        (OUTPUT / 'crop_evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf8')
        run.set_result({'samples': len(evidence), 'review_status': 'candidate_requires_visual_review'})
    print('FACADE_INSPECTION_CANDIDATES', len(sources), OUTPUT)


if __name__ == '__main__':
    main()
