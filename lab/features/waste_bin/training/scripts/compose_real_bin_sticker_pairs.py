"""Build a fixed three-frame, real-bin sticker pilot for development regression.

The source image and four physical bins are reused. Candidate sticker classes
remain unknown until a separate review of the output pixels supports them.
"""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/waste_bin_state or waste_sticker/. Production remains paused.")

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from compose_fire_bin_pilot import ROOT, TASKS, image, write_image

BACKGROUND = ROOT / 'datasets/annotated/site_capture_20261112/sequence_640_v1/full_scene/frame_004951.png'
BIN_REVIEW = ROOT / 'datasets/annotated/waste_bin/v4_review_20261112/review_manifest.json'
STICKER_DIR = ROOT / 'datasets/raw/waste_bin/toolkit/SourcePic'
BIN_COLORS = ('gray', 'green', 'blue', 'red')
# Hand-checked safe paper rectangles and conservative exclusion rectangles
# for fixed bin category markings, both in 640x360 output pixels.
PAPERS = ([75, 240, 102, 260], [214, 224, 241, 244],
          [350, 214, 377, 234], [480, 194, 507, 214])
FIXED_MARKINGS = ([45, 170, 119, 239], [198, 156, 258, 223],
                  [341, 148, 390, 214], [469, 127, 524, 193])
PAIRS = (('油漆', '苹果', '菜叶', '鱼骨'), ('鱼骨', '油漆', '苹果', '菜叶'))


def overlap(a, b):
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Resolve all consumed media before reserving any output.
    output = args.output.resolve()
    owner = (ROOT / 'datasets/exports/waste_bin').resolve()
    if not output.is_relative_to(owner) or output == owner:
        parser.error('output must be a directory under datasets/exports/waste_bin')
    media = [BACKGROUND] + [STICKER_DIR / (name + '.png') for name in sorted(set(sum((list(pair) for pair in PAIRS), [])))]
    procedural = [BIN_REVIEW, TASKS]
    parameters = {}
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from lab.training_common.production_guard import generation_guard
    declared = list(dict.fromkeys(media + procedural))
    with generation_guard('waste_bin/compose_real_bin_sticker_pairs.py/' + output.relative_to(ROOT).as_posix(),
                          declared, [output], parameters,
                          code_paths=[Path(__file__), ROOT / 'lab/features/waste_bin/training/scripts/compose_fire_bin_pilot.py', ROOT / 'lab/features/waste_bin/training/scripts/bin_render_contract.py'], procedural_inputs=procedural) as run:
        if run.reused:
            return
        return _produce(args)


def _produce(args):
    output = args.output.resolve()
    owner = (ROOT / 'datasets/exports/waste_bin').resolve()
    if not output.is_relative_to(owner) or output == owner:
        parser.error('output must be a new directory below datasets/exports/waste_bin')
    frame = image(BACKGROUND)
    if frame.shape != (360, 640, 3):
        raise ValueError('Expected 640x360 background')
    review = json.loads(BIN_REVIEW.read_text(encoding='utf8'))
    source = next(x for x in review['images'] if x['id'] == 'frame_004951')
    if source['split'] != 'regression' or source['tasks']['waste_bin_state']['state'] != 'candidate_complete_positive':
        raise ValueError('Expected reviewed regression bin states')
    if source['tasks']['waste_sticker']['state'] != 'candidate_complete_negative':
        raise ValueError('Background sticker negative must be recorded')
    bin_boxes = source['tasks']['waste_bin_state']['boxes']
    if len(bin_boxes) != 4 or [x[0] for x in bin_boxes] != [1, 0, 0, 1]:
        raise ValueError('Unexpected bin-state candidates')
    classes = list(next(x for x in json.loads(TASKS.read_text(encoding='utf8'))['tasks']
                        if x['id'] == 'waste_bin')['waste_items'])
    for (class_id, bin_box), paper, marking in zip(bin_boxes, PAPERS, FIXED_MARKINGS):
        if not (bin_box[0] <= paper[0] < paper[2] <= bin_box[2] and bin_box[1] <= paper[1] < paper[3] <= bin_box[3]):
            raise ValueError('Paper rectangle is outside candidate bin box')
        if overlap(paper, marking):
            raise ValueError('Paper rectangle intersects a fixed marking')
    output.mkdir(parents=True)
    record = {'purpose': 'real-bin photo composite candidate; regression only; human acceptance pending',
              'role': 'development_regression', 'source_frame': BACKGROUND.relative_to(ROOT).as_posix(),
              'source_review': BIN_REVIEW.relative_to(ROOT).as_posix(),
              'parent_video': source.get('parent_video'), 'source_time_seconds': source.get('source_time_seconds'),
              'physical_assets': 'same source image and four same-appearance bins for all three images; exact IDs unknown',
              'image_size': [640, 360], 'fixed_marking_exclusion_xyxy': FIXED_MARKINGS,
              'sampling': 'one background plus two rotations of four source icons; no independent-scene claim', 'samples': []}
    for i, names in enumerate(((), *PAIRS)):
        pic = frame.copy()
        candidates = []
        for color, name, paper in zip(BIN_COLORS, names, PAPERS):
            x1, y1, x2, y2 = paper
            source_icon = STICKER_DIR / (name + '.png')
            icon = image(source_icon)
            patch = cv2.resize(icon, (x2-x1, y2-y1), interpolation=cv2.INTER_AREA)
            patch = np.clip(patch.astype(np.float32)*.90, 0, 255).astype(np.uint8)
            patch = cv2.GaussianBlur(patch, (3, 3), .35)
            pic[y1:y2, x1:x2] = patch
            candidates.append({'bin_color': color, 'candidate_bbox_xyxy': paper,
                               'output_class_id': None, 'source_class_id': classes.index(name),
                               'source_class_name': name, 'source_image': source_icon.relative_to(ROOT).as_posix(),
                               'transform': 'area-resized to 27x20, brightness 0.90, Gaussian sigma 0.35'})
        sid = ('background', 'pair_a', 'pair_b')[i]
        image_path = output/'images'/(sid+'.png')
        image_path.parent.mkdir(exist_ok=True, parents=True)
        write_image(image_path, pic)
        state_label = output/'labels/waste_bin_state'/(sid+'.txt')
        state_label.parent.mkdir(exist_ok=True, parents=True)
        lines = []
        for cls, (a, b, c, d) in bin_boxes:
            lines.append(f'{cls} {(a+c)/1280:.6f} {(b+d)/720:.6f} {(c-a)/640:.6f} {(d-b)/360:.6f}')
        state_label.write_text('\n'.join(lines)+'\n', encoding='utf8')
        sticker_label = None
        if not names:
            path = output/'labels/waste_sticker'/(sid+'.txt')
            path.parent.mkdir(exist_ok=True, parents=True)
            path.write_text('', encoding='utf8')
            sticker_label = path.relative_to(output).as_posix()
        record['samples'].append({'id': sid, 'image': image_path.relative_to(output).as_posix(),
            'bin_state': {'state': 'candidate_complete_positive', 'label': state_label.relative_to(output).as_posix()},
            'waste_sticker': {'state': 'partial' if names else 'candidate_complete_negative',
                              'label': sticker_label, 'candidate_boxes': candidates,
                              'reason': 'output class visual review pending' if names else 'source negative reviewed'},
            'visual_review': 'pending', 'human_accepted': False})
    (output/'manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print('REAL_BIN_PAIRS', output)


if __name__ == '__main__':
    main()
