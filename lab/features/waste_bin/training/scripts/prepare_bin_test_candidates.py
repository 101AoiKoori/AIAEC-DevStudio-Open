"""Create photo-derived sticker and bin-state test candidates from one scene.

The manual whole-bin and front-panel rectangles are approximate. The four
fixed lid states and source image must be visually reviewed before use. Every
variant shares one original photograph and is one test source group.
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import composite_bin_real_scene as core

ROOT = Path(__file__).resolve().parents[5]
PHOTO = ROOT / 'Archive/Assets/working_sources/legacy/waste_bin/照片/四色桶正面_闭盖_分类标志_001.jpg'
# Rectangles in a 960×540 resizing of the source image, estimated from the
# visible bin extents; subsequent manual review is needed for exact bounds.
BINS = [
    ('blue', 'open', (35, 53, 242, 252), (82, 170, 188, 244)),
    ('red', 'closed', (242, 41, 420, 267), (282, 183, 390, 258)),
    ('green', 'open', (449, 65, 622, 285), (481, 204, 592, 276)),
    ('gray', 'open', (639, 72, 849, 296), (690, 214, 800, 288)),
]


def normalize_box(box):
    x1, y1, x2, y2 = box
    return ((x1+x2)/1920, (y1+y2)/1080, (x2-x1)/960, (y2-y1)/540)


def _generate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--seed', type=int, default=20261120)
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    if not out.is_relative_to((ROOT / 'datasets/exports/waste_bin').resolve()) or out.exists() or args.count < 1:
        parser.error('output must be new and within datasets/exports/waste_bin')
    image = cv2.resize(core.read_image(PHOTO), (960, 540), interpolation=cv2.INTER_AREA)
    names, icons, categories, contract = core.load_icons()
    rng = random.Random(args.seed)
    counts = Counter(); scenarios = Counter(); lights = Counter()
    for task in ('state', 'sticker'):
        (out/task/'images').mkdir(parents=True)
        (out/task/'labels').mkdir()
    (out/'previews').mkdir()
    (out/'run.json').write_text(json.dumps({'status':'running','count':args.count}),encoding='utf-8')
    with (out/'manifest.jsonl').open('x', encoding='utf-8') as manifest:
        for i in range(args.count):
            result = image.copy()
            desired = core.sticker_count(i); scenarios[desired] += 1
            order = list(range(4)); rng.shuffle(order)
            sticker_lines=[]; sticker_records=[]
            for position in order[:desired]:
                color, state, box, front = BINS[position]
                x1,y1,x2,y2 = front
                mask = np.zeros((540,960), dtype=bool)
                mask[y1:y2,x1:x2] = True
                options=list(range(24)); rng.shuffle(options); options.sort(key=lambda c:counts[c])
                cls=options[0]
                paper = core.paper_on_bin(result, mask, icons[cls], rng)
                if paper is None:
                    raise RuntimeError('cannot place sticker in manual front region')
                counts[cls]+=1
                sticker_lines.append(str(cls)+' '+' '.join(f'{v:.6f}' for v in normalize_box(paper)))
                sticker_records.append({'class_id':cls,'name':names[cls],'bin':color,'bbox_xyxy':paper})
            light=('dim','normal','bright')[i%3]; lights[light]+=1
            gain={'dim':rng.uniform(.72,.88),'normal':rng.uniform(.94,1.06),'bright':rng.uniform(1.1,1.22)}[light]
            result=np.clip(result.astype(np.float32)*gain,0,255).astype(np.uint8)
            state_image=np.clip(image.astype(np.float32)*gain,0,255).astype(np.uint8)
            state_lines=[]; overlay=result.copy()
            for color,state,box,front in BINS:
                cls=0 if state=='open' else 1
                state_lines.append(str(cls)+' '+' '.join(f'{v:.6f}' for v in normalize_box(box)))
                cv2.rectangle(overlay,box[:2],box[2:],(0,255,0),2)
            for item in sticker_records:
                box=item['bbox_xyxy'];cv2.rectangle(overlay,box[:2],box[2:],(0,200,255),2)
            stem=f'bin_photo_test_{i:04d}'
            for task,lines in (('state',state_lines),('sticker',sticker_lines)):
                core.write_image(out/task/'images'/(stem+'.jpg'),state_image if task == 'state' else result)
                (out/task/'labels'/(stem+'.txt')).write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8')
            if i<12: core.write_image(out/'previews'/(stem+'.jpg'),overlay)
            record={'image':stem+'.jpg','task_splits':{'state':'test_candidate','sticker':'scene_diagnostic_shared_icons'},
                    'source_photo':PHOTO.relative_to(ROOT).as_posix(),
                    'source_group':'four_bin_photo_001','source_session':'unknown',
                    'states':{color:state for color,state,_,_ in BINS},'state_boxes_approximate':True,
                    'stickers':sticker_records,'brightness':light,'gain':gain}
            manifest.write(json.dumps(record,ensure_ascii=False)+'\n')
    (out/'classes.json').write_text(json.dumps({'state':['open','closed'],'sticker':names,
        'sticker_to_waste_category_id':categories,'canonical_waste_names':contract},ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps({'status':'complete_candidates',
        'task_splits':{'state':'test_candidate','sticker':'scene_diagnostic_shared_icons'},'count':args.count,
        'source_group':'four_bin_photo_001','source_count':1,'source_session':'unknown',
        'sticker_source_group':'waste_bin_toolkit_24icons_shared_across_splits',
        'sticker_evaluation_scope':'seen artwork and changed scene; not independent source artwork',
        'state_note':'one fixed four-bin configuration, labels are visually estimated, not verified ground truth',
        'sticker_counts':dict(scenarios),'sticker_class_counts':{names[i]:counts[i] for i in range(24)},
        'lighting':dict(lights),'description':'real photo with programmatically placed paper stickers; photo-derived synthetic test candidates'},ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'run.json').write_text(json.dumps({'status':'complete_candidates','count':args.count}),encoding='utf-8')
    print('complete:',args.count)


def _guard_inputs():
    config = core.yaml.safe_load(core.CLASSES.read_text(encoding='utf8'))
    media = [PHOTO] + [core.CACHE / (item['name'] + '.npz') for item in config['items']]
    media += [ROOT / 'Archive/Assets/working_sources/digital/waste_sticker/toolkit/SourcePic' / (item['name'] + '.png')
              for item in config['items']]
    return media, [core.CLASSES, ROOT / 'config/perception_tasks.json']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--count',type=int,required=True)
    parser.add_argument('--seed',type=int,default=20261120)
    args,_=parser.parse_known_args()
    out=(ROOT/args.out).resolve()
    media,procedural=_guard_inputs()
    sys.path.insert(0,str(ROOT))
    from lab.training_common.production_guard import generation_guard
    with generation_guard('waste_bin.prepare_test_candidates:'+str(out.relative_to(ROOT)),
                          media,[out],{'argv':sys.argv[1:]},
                          code_paths=[Path(__file__).resolve(),Path(core.__file__).resolve()],
                          procedural_inputs=procedural,root=ROOT) as run:
        if run.reused:
            return
        _generate()


if __name__=='__main__':
    main()
