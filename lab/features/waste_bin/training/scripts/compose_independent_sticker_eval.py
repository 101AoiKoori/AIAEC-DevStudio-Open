"""Place separate new vector artwork over clean candidate bin-scene images.

This creates a limited-class, synthetic evaluation candidate; it is not a
real-print acceptance set. Val and test use different generated artwork files
and different scene sources, but the physical object and session relationships
are not established. All labels remain candidates for visual review.
"""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/waste_bin_state or waste_sticker/. Production remains paused.")

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import composite_bin_real_scene as core
from draw_independent_sticker_eval import CANDIDATE_CLASSES

ROOT = Path(__file__).resolve().parents[5]
ART = ROOT/'datasets/exports/waste_bin/sticker_distinct_12_artwork_preview_20261110'
SOURCE = {'val':ROOT/'datasets/exports/waste_bin/bin_state_val_sticker_scene_diag_v3_120_20261110',
          'test':ROOT/'datasets/exports/waste_bin/bin_state_test_sticker_scene_diag_v3_120_20261110'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--split',choices=('val','test'),required=True);parser.add_argument('--count',type=int,default=120)
    parser.add_argument('--seed',type=int,default=20261124);args=parser.parse_args()
    # Resolve all consumed media before reserving any output.
    output = (ROOT / args.out).resolve()
    owner = (ROOT / 'datasets/exports/waste_bin').resolve()
    if not output.is_relative_to(owner) or output == owner or args.count < 1:
        parser.error('output must be under datasets/exports/waste_bin and count positive')
    source = SOURCE[args.split]
    manifest = source / 'manifest.jsonl'
    records = [json.loads(line) for line in manifest.read_text(encoding='utf8').splitlines()]
    media = [source / 'state/images' / row['image'] for row in records if not row.get('state_empty')]
    media += [ART / args.split / f'{i:02d}.png' for i in CANDIDATE_CLASSES]
    procedural = [manifest, source / 'classes.json']
    parameters = {'split': args.split, 'count': args.count, 'seed': args.seed}
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from lab.training_common.production_guard import generation_guard
    declared = list(dict.fromkeys(media + procedural))
    with generation_guard('waste_bin/compose_independent_sticker_eval.py/' + output.relative_to(ROOT).as_posix(),
                          declared, [output], parameters,
                          code_paths=[Path(__file__), ROOT / 'lab/features/waste_bin/training/scripts/composite_bin_real_scene.py', ROOT / 'lab/features/waste_bin/training/scripts/draw_independent_sticker_eval.py'], procedural_inputs=procedural) as run:
        if run.reused:
            return
        return _produce(args)


def _produce(args):
    out=(ROOT/args.out).resolve()
    if not out.is_relative_to((ROOT/'datasets/exports/waste_bin').resolve()) or args.count<1:
        raise ValueError('output must be under datasets/exports/waste_bin and count positive')
    source=SOURCE[args.split]
    records=[json.loads(line) for line in (source/'manifest.jsonl').read_text(encoding='utf8').splitlines()]
    records=[r for r in records if not r.get('state_empty')]
    pools={n:[r for r in records if len(r['stickers'])>=n] for n in range(5)}
    if any(not pool for pool in pools.values()):raise ValueError('missing eligible source scenes')
    art={i:core.read_image(ART/args.split/f'{i:02d}.png') for i in CANDIDATE_CLASSES}
    rng=random.Random(args.seed); class_counts=Counter(); scenarios=Counter()
    names=json.loads((source/'classes.json').read_text(encoding='utf8'))['sticker']
    for sub in ('images','labels','preview'): (out/sub).mkdir(parents=True)
    (out/'run.json').write_text(json.dumps({'status':'running','count':args.count}),encoding='utf8')
    with (out/'manifest.jsonl').open('x',encoding='utf8') as manifest:
        for i in range(args.count):
            desired=core.sticker_count(i);scenarios[desired]+=1
            row=rng.choice(pools[desired]);image=core.read_image(source/'state/images'/row['image'])
            h,w=image.shape[:2]; objects=[];lines=[];overlay=image.copy()
            locations=list(row['stickers']);rng.shuffle(locations)
            for previous in locations[:desired]:
                choices=list(CANDIDATE_CLASSES);rng.shuffle(choices);choices.sort(key=lambda c:class_counts[c]);cls=choices[0]
                x1,y1,x2,y2=[round(v) for v in previous['bbox_xyxy']]
                if not (0<=x1<x2<=w and 0<=y1<y2<=h):raise ValueError('invalid source placement')
                icon=cv2.resize(art[cls],(x2-x1,y2-y1),interpolation=cv2.INTER_AREA)
                icon=np.clip(icon.astype(np.float32)*row['gain'],0,255).astype(np.uint8)
                image[y1:y2,x1:x2]=icon;overlay[y1:y2,x1:x2]=icon
                xywh=((x1+x2)/(2*w),(y1+y2)/(2*h),(x2-x1)/w,(y2-y1)/h)
                lines.append(str(cls)+' '+' '.join(f'{v:.6f}' for v in xywh))
                objects.append({'class_id':cls,'name':names[cls],'bbox_xyxy':[x1,y1,x2,y2],
                                'artwork':(ART/args.split/f'{cls:02d}.png').relative_to(ROOT).as_posix(),
                                'bin':previous['bin']});class_counts[cls]+=1
                cv2.rectangle(overlay,(x1,y1),(x2,y2),(0,200,255),1)
            stem=f'waste_vector_{args.split}_{i:04d}'
            core.write_image(out/'images'/(stem+'.jpg'),image)
            (out/'labels'/(stem+'.txt')).write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf8')
            if i<12:core.write_image(out/'preview'/(stem+'.jpg'),overlay)
            record={'image':f'images/{stem}.jpg','label':f'labels/{stem}.txt',
                    'split':args.split,'scene_source':(source/'state/images'/row['image']).relative_to(ROOT).as_posix(),
                    'scene_source_group':row['source_group'] if 'source_group' in row else row.get('source_photo'),
                    'artwork_source_group':f'new_procedural_vectors_{args.split}',
                    'shared_vector_program':'draw_independent_sticker_eval.py',
                    'brightness':row['brightness'],'objects':objects}
            manifest.write(json.dumps(record,ensure_ascii=False)+'\n')
    (out/'summary.json').write_text(json.dumps({'status':'candidate_synthetic_domain_not_real_acceptance',
        'split':args.split,'count':args.count,'covered_class_ids':list(CANDIDATE_CLASSES),
        'missing_class_ids':[i for i in range(24) if i not in CANDIDATE_CLASSES],
        'class_counts':{names[i]:class_counts[i] for i in CANDIDATE_CLASSES},'scenarios':dict(scenarios),
        'artwork_source_group':f'new_procedural_vectors_{args.split}',
        'limitations':'12 of 24 classes; stylized hand-drawn vectors, different from actual printed icons; val/test share generator concept; scene or device session relationship unknown; not human-verified labels'},ensure_ascii=False,indent=2),encoding='utf8')
    (out/'run.json').write_text(json.dumps({'status':'complete_candidates','count':args.count}),encoding='utf8')
    print('complete:',args.count)


if __name__=='__main__':main()
