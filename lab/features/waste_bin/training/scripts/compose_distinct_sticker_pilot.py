"""Create two review-only real-bin images with five source icon appearances.

No sticker truth is exported until each rendered class is visually reviewed at
the actual output size. Both parent images are exposed development material.
"""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/waste_bin_state or waste_sticker/.")

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from compose_fire_bin_pilot import ROOT, TASKS, image, write_image
from compose_real_bin_sticker_pairs import BACKGROUND, BIN_REVIEW, FIXED_MARKINGS, overlap

EXPORTS = ROOT / 'datasets/exports/waste_bin'
ASSET_REVIEW = ROOT / 'datasets/annotated/waste_bin/v4_real_asset_review_20261112/review_manifest.json'
ASSET_FRAME = ROOT / 'datasets/annotated/waste_bin/v4_real_asset_review_20261112/images/0004.png'
ICONS = ROOT / 'Archive/Assets/working_sources/digital/waste_sticker/toolkit/SourcePic'
ASSET_PAPERS = [[164, 242, 178, 256], [285, 220, 299, 234],
                [354, 155, 374, 175], [453, 140, 473, 160]]
ASSET_FIXED = [[150, 179, 200, 241], [274, 159, 315, 219],
               [380, 135, 414, 191], [479, 124, 517, 177]]
VIDEO_PAPERS = [[120, 200, 138, 218], [220, 224, 238, 242],
                [356, 215, 374, 233], [439, 160, 459, 180]]
ASSET_CLASSES = ('纸箱', '废弃眼镜', '面包', '烟头')
VIDEO_CLASSES = ('菜叶', '纸箱', '面包', '废弃眼镜')


def label_lines(boxes, width, height):
    return ''.join(f'{cls} {(a+c)/(2*width):.6f} {(b+d)/(2*height):.6f} '
                   f'{(c-a)/width:.6f} {(d-b)/height:.6f}\n' for cls, (a,b,c,d) in boxes)


def generate(output):
    output = output.resolve()
    if not output.is_relative_to(EXPORTS.resolve()) or output == EXPORTS.resolve():
        raise ValueError('output must be below datasets/exports/waste_bin')
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from lab.training_common.production_guard import generation_guard
    media = [BACKGROUND, ASSET_FRAME] + [ICONS / (name + '.png') for name in sorted(set(ASSET_CLASSES + VIDEO_CLASSES))]
    procedural = [BIN_REVIEW, ASSET_REVIEW, TASKS]
    codes = [Path(__file__), ROOT / 'lab/features/waste_bin/training/scripts/compose_fire_bin_pilot.py',
             ROOT / 'lab/features/waste_bin/training/scripts/compose_real_bin_sticker_pairs.py',
             ROOT / 'lab/features/waste_bin/training/scripts/bin_render_contract.py']
    with generation_guard('waste_bin/compose_distinct_sticker_pilot.py/' + output.relative_to(ROOT).as_posix(),
                          media + procedural, [output], {}, code_paths=codes, procedural_inputs=procedural) as run:
        if run.reused:
            return json.loads((output / 'manifest.json').read_text(encoding='utf8'))
        return _generate(output)


def _generate(output):
    output = output.resolve()
    if not output.is_relative_to(EXPORTS.resolve()) or output == EXPORTS.resolve():
        raise ValueError('output must be a new directory below datasets/exports/waste_bin')
    review = json.loads(BIN_REVIEW.read_text(encoding='utf8'))
    video_row = next(r for r in review['images'] if r['id']=='frame_004951')
    asset = json.loads(ASSET_REVIEW.read_text(encoding='utf8'))
    asset_row = next(r for r in asset['samples'] if r['id']=='0004')
    for row in (video_row, asset_row):
        if (row['split'] != 'regression' or
            row['tasks']['waste_bin_state']['state'] != 'candidate_complete_positive' or
            row['tasks']['waste_sticker']['state'] != 'candidate_complete_negative'):
            raise ValueError('parent must be reviewed regression with no sticker candidate')
    classes = list(next(x for x in json.loads(TASKS.read_text(encoding='utf8'))['tasks']
                        if x['id'] == 'waste_bin')['waste_items'])
    # Place square pictures beside fixed markings, away from the curved bin
    # bottom; the positions are manually fitted and still require review.
    asset_boxes = [[x['class_id'], x['xyxy']] for x in asset_row['tasks']['waste_bin_state']['candidate_boxes']]
    jobs = [('asset', ASSET_FRAME, asset_row, asset_boxes, ASSET_PAPERS, ASSET_FIXED, ASSET_CLASSES,
             ('gray','green','red','blue')),
            ('video', BACKGROUND, video_row, video_row['tasks']['waste_bin_state']['boxes'], VIDEO_PAPERS,
             FIXED_MARKINGS, VIDEO_CLASSES, ('gray','green','blue','red'))]
    output.mkdir(parents=True)
    for folder in ('images', 'labels/waste_bin_state', 'overlays'):
        (output/folder).mkdir(parents=True)
    report = {'status': 'running', 'purpose': 'small output-pixel readability study; development regression only',
              'parent_reviews': [BIN_REVIEW.relative_to(ROOT).as_posix(), ASSET_REVIEW.relative_to(ROOT).as_posix()],
              'scale_reference': 'Archive/Assets/captures/car_real_capture_20261112/P22_TRASH.png',
              'asset_root': ICONS.relative_to(ROOT).as_posix(),
              'asset_license': 'redistribution status not established',
              'reuse': 'same 20261112 bin appearances, two exposed parents and five source icons; not independent scenes',
              'samples': []}
    try:
        for stem, background, parent, bin_boxes, papers, fixed, names, colors in jobs:
            frame = image(background)
            height, width = frame.shape[:2]
            if (width,height) not in ((640,480),(640,360)):
                raise ValueError('unexpected parent dimensions')
            if len(bin_boxes) != 4:
                raise ValueError('expected four candidate bin boxes')
            stickers=[]
            for color, name, box, forbidden, (_, bin_box) in zip(colors, names, papers, fixed, bin_boxes):
                a,b,c,d = box
                if not (bin_box[0] <= a < c <= bin_box[2] and bin_box[1] <= b < d <= bin_box[3]):
                    raise ValueError('paper must fit in a visible bin box')
                if overlap(box, forbidden):
                    raise ValueError('paper overlaps the fixed bin marking')
                source = ICONS/(name+'.png')
                patch = image(source)
                patch = cv2.resize(patch, (c-a,d-b), interpolation=cv2.INTER_AREA)
                patch = np.clip(patch.astype(np.float32)*.90, 0, 255).astype(np.uint8)
                patch = cv2.GaussianBlur(patch, (3,3), .25)
                frame[b:d,a:c] = patch
                stickers.append({'bin_color': color, 'source_class_hypothesis': name,
                    'source_class_id': classes.index(name), 'source_image': source.relative_to(ROOT).as_posix(),
                    'candidate_bbox_xyxy': box, 'output_class_id': None,
                    'transform': {'resize': 'area to destination rectangle', 'brightness_factor': .90,
                                  'gaussian_sigma': .25}, 'output_pixel_review': 'pending'})
            write_image(output/'images'/(stem+'.png'), frame)
            label = f'labels/waste_bin_state/{stem}.txt'
            (output/label).write_text(label_lines(bin_boxes,width,height), encoding='utf8')
            report['samples'].append({'id': stem, 'image': f'images/{stem}.png', 'image_size': [width,height],
                'source_image': background.relative_to(ROOT).as_posix(), 'source_sample_id': parent['id'],
                'source_video': parent.get('source_video', parent.get('parent_video')),
                'source_time_seconds': parent.get('source_time_seconds'),
                'split': 'regression', 'fixed_marking_exclusion_xyxy': fixed,
                'bin_state': {'state':'candidate_complete_positive', 'boxes':bin_boxes, 'label':label},
                'waste_sticker': {'state':'partial', 'candidate_boxes':stickers, 'label':None,
                                  'reason':'output class review pending'},
                'visual_review':'pending', 'human_accepted':False})
        report['status']='complete_candidates'
    except BaseException as exc:
        report.update(status='failed_or_interrupted', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    result=generate(ROOT/args.output)
    print(json.dumps({'status':result['status'],'count':len(result['samples'])}))
