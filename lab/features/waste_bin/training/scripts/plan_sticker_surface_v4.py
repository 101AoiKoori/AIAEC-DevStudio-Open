"""Plan a small prior-weighted sticker-surface candidate batch; no images written."""
import argparse
import json
import random
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[5]
GEOMETRY = 'datasets/annotated/waste_bin/waste_sticker_surface_geometry_v5_20261114.jsonl'
ART = 'datasets/annotated/waste_bin/waste_sticker_artwork_source_manifest_20261114.jsonl'
TOOLKIT = 'datasets/raw/waste_bin/toolkit'
CONSTRAINTS = {'max_long_edge': 1280, 'max_short_edge': 720,
               'region_geometry': 'explicit_bin_surface_quad',
               'labels': 'inserted_visible_candidates_background_unknown'}


def make_plan(seed, budget):
    if not 2 <= budget <= 12:
        raise ValueError('budget must be within 2..12 to cover both parents')
    geometry = [json.loads(line) for line in (ROOT / GEOMETRY).read_text(encoding='utf8').splitlines()]
    names = [x['name'] for x in yaml.safe_load((ROOT / TOOLKIT / 'configs/classes.yaml').read_text(encoding='utf8'))['items']]
    artworks = [json.loads(line) for line in (ROOT / ART).read_text(encoding='utf8').splitlines()]
    available = {u['class_id']: row['image'] for row in artworks for u in row.get('task_use_status', [])
                 if u.get('task') == 'waste_sticker' and u.get('split') == 'train'
                 and u.get('role') == 'target_foreground' and u.get('status') == 'ready'}
    if set(available) != set(range(24)) or len(geometry) != 2:
        raise ValueError('expected 24 artwork relations and two surface candidates')
    rng = random.Random(seed)
    states = []
    for index, record in enumerate(geometry):
        parent = record['source_video']
        for n in range(6):
            x1, y1 = record['bins'][0]['allowed_polygon'][0]
            x2, y2 = record['bins'][0]['allowed_polygon'][2]
            upper = min(.46, (y2-y1-4)/record['estimated_front_width_px'],
                        (x2-x1-4)/record['estimated_front_width_px'])
            if upper < .39:
                raise ValueError('available face height cannot fit the reference paper scale')
            size = round(record['estimated_front_width_px'] * rng.uniform(.39, upper))
            x = rng.randint(x1+1, x2-size-1)
            y = y2-size-1
            cls = (index*6+n) % 24
            placements = [{'bin_id':'red', 'class_id':cls, 'paper_quad':[[x,y],[x+size,y+1],
                           [x+size-1,y+size],[x-1,y+size-1]], 'identifiability':'identifiable'}]
            p = {'photo':record['path'], 'record_index':index, 'placements':placements,
                 'gain':round(rng.uniform(.7,1.0),3), 'saturation':round(rng.uniform(.7,1.0),3),
                 'blur_sigma':rng.choice([0.0,.5,1.0])}
            states.append({'id':f'parent_{index}_state_{n}', 'min_count':1 if n == 0 else 0,'max_count':1,
                'prior':{'alpha':8,'beta':2} if index else {'alpha':2,'beta':8},
                'exposure_weight':1, 'failure_cost':1, 'failure_event':'candidate_surface_label_incomplete',
                'conditions':['source_parent_'+str(index)],
                'background':{'kind':'camera_photo','image':p['photo'],'geometry_manifest':GEOMETRY,'record_index':index},
                'target':{'kind':'explicit_bin_stickers','placements':placements},
                'labels':{'format':'yolo_detection','status':'inserted_visible_candidates_background_unknown'},
                'augmentation':{'gain':p['gain'],'saturation':p['saturation'],'blur_sigma':p['blur_sigma'],'placement':'explicit_surface_quads'},
                'parameters':p, 'sources':[p['photo'],available[cls]], 'parent_video':parent})
    return {'schema_version':1,'model_key':'waste_sticker','target_split':'train','budget':budget,
            'purpose':'Limited two-parent surface candidates with perspective and bounded tone/blur',
            'uncertainty_weight':0,'independence_basis':'two parent videos; capture-session independence unknown',
            'constraints':CONSTRAINTS,'strata':states,
            'planning':{'seed':seed,'geometry_status':'AI visual candidate; limited visual review required',
                        'paper_width_over_front_width_design_range':[.39,.46],
                        'scale_reference':'datasets/annotated/waste_bin/waste_sticker_p22_stationary_annotations_20261114.json',
                        'prior_scope':'design allocation assumptions, not measured failure rates'}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=20261114)
    p.add_argument('--budget',type=int,default=6);args=p.parse_args()
    out=(ROOT/args.out).resolve()
    if not out.is_relative_to(ROOT) or out.suffix!='.json' or out.exists():
        raise ValueError('output must be a new workspace JSON')
    with out.open('x',encoding='utf8') as f:
        json.dump(make_plan(args.seed,args.budget),f,ensure_ascii=False,indent=2);f.write('\n')
    print('candidate plan:',out.relative_to(ROOT).as_posix())


if __name__=='__main__':
    main()
