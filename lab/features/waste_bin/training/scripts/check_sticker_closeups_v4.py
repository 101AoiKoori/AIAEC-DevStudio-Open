"""Check supplemental close-view sticker candidates and their source grouping."""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

import generate_sticker_closeups_v4 as gen
import generate_waste_v4_grouped as grouped
sys.path.insert(0, str(gen.ROOT))
from lab.training_common.production_guard import generation_guard


def main():
    cv2.setNumThreads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset')
    args = parser.parse_args()
    dataset = (gen.ROOT/args.dataset).resolve()
    report_path = dataset / 'manifest.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    media, procedural = screen_sources()
    procedural.add(report_path)
    for item in report['records']:
        media.add(gen.ROOT / item['image'])
        if item.get('label'):
            procedural.add(gen.ROOT / item['label'])
    destination = dataset / 'structural_check.json'
    codes = [Path(__file__), Path(gen.__file__), Path(grouped.__file__), Path(gen.base.__file__)]
    with generation_guard('check_sticker_closeups_v4:' + destination.relative_to(gen.ROOT).as_posix(),
                          sorted(media), [destination], {}, code_paths=codes,
                          procedural_inputs=sorted(procedural), root=gen.ROOT) as run:
        if run.reused:
            result = json.loads(destination.read_text(encoding='utf-8'))
            print(json.dumps({k: v for k, v in result.items() if k != 'counts'}, ensure_ascii=False))
            if result['status'] != 'structural_check_pass':
                raise SystemExit(1)
            return
        return _check(dataset, report)


def screen_sources():
    """Enumerate the same source files as load_inputs, without icon assets."""
    source_manifest = gen.SOURCE / 'manifest.json'
    procedural = {source_manifest}
    media = set()
    for index, path in enumerate(gen.QUANTITY_PATHS):
        manifest = path / 'manifest.json'
        procedural.add(manifest)
        for sample in json.loads(manifest.read_text(encoding='utf-8'))['samples']:
            group = sample['group']
            included = ((index == 0 and group < 8) or (index == 1 and 8 <= group <= 12)
                        or (index == 2 and 16 <= group <= 20))
            if sample['candidate_label'] and included:
                media.update((path / sample['image'], path / sample['mask_visible']))
    for item in json.loads(source_manifest.read_text(encoding='utf-8'))['records']:
        if item['generation_method'] != 'blender_render':
            media.add(gen.ROOT / item['background']['image'])
    return media, procedural


def _check(dataset, report):

    sources, _ = gen.load_inputs()
    screened, blocked, links = gen.screen_foregrounds(sources)
    allowed = {(split, color, render) for (split, color), values in screened.items()
               for _, render, _ in values}
    by_key = {(split, color, render):entry for (split, color), values in screened.items()
              for entry, render, _ in values}
    errors, counts, box_sizes = [], Counter(), []
    backgrounds = defaultdict(set)
    renders = defaultdict(set)
    for item in report['records']:
        key, split, color, render = item['id'],item['split'],item['bin_color'],item['source_render']
        cls, status = item['icon_class_candidate'], item['status']
        tone = 'dark' if item['tone']['value_gain'] < .9 else 'normal'
        counts[(split,str(cls),color,tone,status)] += 1
        backgrounds[split].add(item['background']['parent_image'])
        renders[split].add(render)
        if (split,color,render) not in allowed:
            errors.append(f'{key}: source was excluded by foreground grouping')
        else:
            cropped = gen.single_bin_crop(by_key[(split,color,render)],color)
            if not cropped or cropped[2] != item['source_crop_xywh']:
                errors.append(f'{key}: source crop disagrees or crosses another bin')
        image = cv2.imdecode(np.fromfile(str(gen.ROOT/item['image']),np.uint8),cv2.IMREAD_COLOR)
        if image is None or image.shape != (480,640,3):
            errors.append(f'{key}: image not 640x480')
        x1,y1,x2,y2 = item['bbox_xyxy']
        if not (0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480 and 46 <= min(x2-x1,y2-y1) <= 70):
            errors.append(f'{key}: invalid paper box or size')
        box_sizes.append(x2-x1)
        floor = np.zeros((480,640),dtype=np.uint8)
        cv2.fillPoly(floor,[np.asarray(item['background']['ground_polygon_xy'],dtype=np.int32)],1)
        floor = cv2.erode(floor,np.ones((11,11),np.uint8))
        if any(not (0<=x<640 and 0<=y<480 and floor[y,x])
               for points in item['ground_support_xy'].values() for x,y in points):
            errors.append(f'{key}: support outside reviewed ground')
        if status == 'candidate_visible':
            if cls not in gen.VISIBLE or not item['label']:
                errors.append(f'{key}: invalid complete label state')
            else:
                label = (gen.ROOT/item['label']).read_text(encoding='utf-8').strip()
                if label != grouped.box_line(cls,item['bbox_xyxy']):
                    errors.append(f'{key}: class or box does not match manifest')
        elif status == 'partial_unknown':
            if cls not in gen.UNKNOWN or item['label'] is not None:
                errors.append(f'{key}: unknown incorrectly labelled')
        else:
            errors.append(f'{key}: unsupported status')
    for name, groups in (('background',backgrounds),('render',renders)):
        for first,second in (('train','val'),('train','test'),('val','test')):
            if groups[first] & groups[second]:
                errors.append(f'{name}: exact parent reused across {first}/{second}')
    if not report['pilot']:
        for split, repeat in (('train',3),('val',1),('test',1)):
            for cls in gen.VISIBLE:
                for color in gen.COLORS:
                    for tone in ('normal','dark'):
                        if counts[(split,str(cls),color,tone,'candidate_visible')] != repeat:
                            errors.append(f'{split}: missing visible class/color/tone quota')
            for cls in gen.UNKNOWN:
                for color in gen.COLORS:
                    for tone in ('normal','dark'):
                        if counts[(split,str(cls),color,tone,'partial_unknown')] != 1:
                            errors.append(f'{split}: missing partial class/color/tone quota')
    result = {'status':'structural_check_pass' if not errors else 'structural_check_failed',
        'records':len(report['records']), 'counts':{'|'.join(k):v for k,v in counts.items()},
        'paper_width_px_range':[min(box_sizes),max(box_sizes)],
        'distinct_source_renders':{k:len(v) for k,v in renders.items()},
        'distinct_background_parents':{k:len(v) for k,v in backgrounds.items()},
        'blocked_crop_sources':len(blocked), 'cross_split_crop_links':links,
        'scope':'structural and source-screen checks; threshold is heuristic and does not establish independent scenes or semantic ground truth',
        'errors':errors}
    (dataset/'structural_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='counts'},ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
