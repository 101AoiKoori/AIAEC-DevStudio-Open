"""Expand reviewed, non-diagnostic site-video plate frames for training only."""
import itertools
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT / 'lab/training_common'), str(Path(__file__).resolve().parent)]
from camera_tone import apply_tone, read_image, source_role, write_image  # noqa: E402
from expand_plate_v4_real_core import transform_box  # noqa: E402
from lab.scripts.data.v4_source_gate import SITE_VIDEO, build_queue, site_sequence_times  # noqa: E402

REVIEW = ROOT / 'datasets/annotated/plate/v4_video_approach_review_20261112'
OUT = ROOT / 'datasets/exports/plate/v4_video_train_candidates_20261112'
SEED = 20261113


def rel(path):
    return path.relative_to(ROOT).as_posix()


def _generate():
    if OUT.exists():
        raise ValueError('Output already exists; use a new version')
    positive = json.loads((REVIEW / 'review_manifest.json').read_text(encoding='utf-8'))['samples']
    positive = [r for r in positive if r['split'] == 'unassigned' and r['state'] == 'candidate_complete_positive']
    negative = json.loads((REVIEW / 'negative_review.json').read_text(encoding='utf-8'))['samples']
    if len(positive) != 3 or len(negative) != 2:
        raise ValueError('Expected three reviewed positive and two reviewed negative parents')
    known, times = build_queue(ROOT), site_sequence_times(ROOT)
    parents = []
    for row in positive:
        parents.append((ROOT / row['parent_image'], (ROOT / row['label']).read_text(encoding='utf-8').strip(),
                        row['segment_id'], rel(REVIEW / 'review_manifest.json')))
    for row in negative:
        parents.append((ROOT / row['image'], '', None, rel(REVIEW / 'negative_review.json')))
    for path, _, _, _ in parents:
        role, reason = source_role(ROOT, path, known)
        if role == 'regression':
            raise ValueError(f'Parent is restricted to regression: {rel(path)} ({reason})')
    rng, records, counts, skipped = np.random.default_rng(SEED), [], Counter(), []
    combinations = list(itertools.product((1.05, 1.15, 1.25), (.9, .7), (.85, .6), (0., .65)))
    try:
        (OUT / 'images/train').mkdir(parents=True)
        (OUT / 'labels/train').mkdir(parents=True)
        for parent_index, (path, label, segment, evidence) in enumerate(parents):
            image = read_image(path)
            height, width = image.shape[:2]
            if (width, height) != (640, 360):
                raise ValueError('Expected 640x360 source frame: ' + rel(path))
            index = int(path.stem.split('_')[-1])
            time = times[index]
            for combo_index, (scale, value, saturation, blur) in enumerate(combinations):
                angle = float(rng.uniform(-1., 1.))
                shift = rng.uniform(-4., 4., size=2)
                matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
                matrix[:, 2] += shift
                projected = transform_box(label, matrix, width, height)
                if projected is None:
                    skipped.append({'parent': rel(path), 'combination': combo_index, 'reason': 'cropped or too small'})
                    continue
                frame = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE)
                frame = apply_tone(frame, value_gain=value, saturation_gain=saturation)
                if blur:
                    frame = cv2.GaussianBlur(frame, (0, 0), blur)
                name = f'video_{index:06d}_{combo_index:02d}'
                image_out, label_out = OUT / 'images/train' / (name + '.png'), OUT / 'labels/train' / (name + '.txt')
                write_image(image_out, frame)
                label_out.write_text(projected + ('\n' if projected else ''), encoding='utf-8')
                state = 'positive' if label else 'negative'
                records.append({'image': rel(image_out), 'label': rel(label_out), 'split': 'train_candidate',
                    'state': state, 'parent_image': rel(path), 'source_video': SITE_VIDEO,
                    'source_time_seconds': time, 'segment_id': segment, 'source_review': evidence,
                    'generation_method': 'augmentation', 'image_size': [width, height],
                    'processing': {'scale': scale, 'rotation_degrees': angle, 'shift_xy': shift.tolist(),
                        'affine_matrix': matrix.tolist(), 'value_gain': value, 'saturation_gain': saturation,
                        'gaussian_sigma_px': blur, 'border': 'replicate', 'compression': 'PNG lossless'},
                    'label_review': 'projected_from_AI_reviewed_parent_batch_QA_pending'})
                counts[state] += 1
        manifest = {'status': 'train_candidates_not_deliverable', 'seed': SEED, 'source_video': SITE_VIDEO,
            'scope': 'One video, shared scene and physical props; variants are not independent captures. '
                     'No validation/test data or model result.',
            'parent_images': len(parents), 'requested_per_parent': len(combinations),
            'counts': dict(counts), 'skipped': skipped, 'records': records}
        (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(json.dumps({'output': rel(OUT), 'parents': len(parents), 'counts': dict(counts),
                          'skipped': len(skipped)}, ensure_ascii=False))
    except BaseException:
        (OUT / 'FAILED.txt').write_text('Generation failed; inspect before reuse.\n', encoding='utf-8')
        raise


def _guard_inputs():
    positive = json.loads((REVIEW / 'review_manifest.json').read_text(encoding='utf-8'))['samples']
    positive = [r for r in positive if r['split'] == 'unassigned' and r['state'] == 'candidate_complete_positive']
    negative = json.loads((REVIEW / 'negative_review.json').read_text(encoding='utf-8'))['samples']
    media = [ROOT / SITE_VIDEO] + [ROOT / r['parent_image'] for r in positive] + [ROOT / r['image'] for r in negative]
    return media, [REVIEW / 'review_manifest.json', REVIEW / 'negative_review.json'] + [ROOT / r['label'] for r in positive], {'seed': SEED}


def main():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from lab.training_common.production_guard import generation_guard
    media, procedural, parameters = _guard_inputs()
    from lab.scripts.data import v4_source_gate as gate
    procedural += [ROOT / value for value in (gate.INVENTORY, gate.SEGMENTS, gate.SELECTION,
        gate.EXPOSURE, gate.DUPLICATES, gate.DEVELOPMENT_EXPOSURE, gate.REVIEW_SUMMARY,
        'config/perception_tasks.json', *gate.ANNOTATIONS, *gate.SITE_SELECTIONS)]
    procedural.append(ROOT / gate.SITE_SEQUENCE / 'ffmpeg.log')
    from lab.features.plate.training.scripts.plate_guard_inputs import source_gate_dependencies
    gate_files, gate_code = source_gate_dependencies(ROOT, gate.build_queue(ROOT))
    procedural += list(gate_files)
    media = sorted(set(media))
    procedural = sorted(set(procedural))
    with generation_guard("plate.expand_plate_v4_video_train", media, [OUT], parameters,
                          code_paths=sorted(gate_code | {Path(__file__), ROOT / 'lab/training_common/camera_tone.py', ROOT / 'lab/features/plate/training/scripts/expand_plate_v4_real_core.py'}), procedural_inputs=procedural, root=ROOT) as run:
        if run.reused:
            return
        _generate()
        manifest_path = OUT / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['source_refs'] = run.source_refs
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
