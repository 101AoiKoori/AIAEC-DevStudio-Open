"""Plan e-bike combinations in reviewed unobstructed areas across two source videos."""

if __name__ == "__main__":
    raise SystemExit("Task helper is import-only; use the primary entry in datasets/scripts/e_bike/. Production remains paused.")

import argparse
import copy
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
from datasets.scripts.e_bike import compose_v4_grounded_train as ground
from datasets.scripts.e_bike import plan_usage_batch as common
from datasets.scripts.common.bayesian import allocate, plan_generation  # noqa: E402
from lab.scripts.data.v4_source_gate import build_queue  # noqa: E402
from datasets.scripts.common.perspective import planar_warp  # noqa: E402

RECTS = {
    'frame_005751': [[[15,330],[180,330],[180,460],[15,460]]],
    'frame_007000': [[[10,210],[140,210],[140,465],[10,465]], [[390,210],[625,210],[625,465],[390,465]]],
    'frame_007045': [[[360,200],[620,200],[620,465],[360,465]]],
}
OBSTACLES = {
    'frame_005751': [[200, 0, 640, 480]],
    'frame_007000': [[145, 0, 385, 460]],
    'frame_007045': [[100, 0, 350, 415]],
}


def reviewed_backgrounds(legacy_manifest):
    originals = common.load(ground.BACKGROUND)['backgrounds']
    use = common.load(common.USAGE)
    allowed = {r['image'] for r in use['records'] if common.relation(r, 'background', {'ready'})}
    rows = []
    for bg in originals:
        if bg['id'] not in RECTS or bg['image'] not in allowed:
            continue
        row = copy.deepcopy(bg)
        row.update({'safe_regions': RECTS[bg['id']], 'forbidden_boxes': OBSTACLES[bg['id']],
                    'source_group': 'full_scene_video',
                    'safe_region_evidence': 'full-frame review; conservative floor rectangles exclude visible fixtures'})
        rows.append(row)
    legacy = common.load(legacy_manifest)['backgrounds']
    if len(legacy) != 2 or any(r.get('source_group') != 'legacy_scene_001_video' for r in legacy):
        raise ValueError('expected the two guarded reviewed legacy crops')
    for row in legacy:
        row['safe_regions'] = [[[20,100],[500,100],[500,465],[20,465]]]
        row['safe_region_evidence'] = 'reviewed crop floor; right rail excluded'
    rows.extend(legacy)
    if len(rows) != 5:
        raise ValueError('three reviewed full-scene backgrounds and two legacy crops are required')
    return rows


def make_plan(backgrounds, seed=20261114):
    rng = random.Random(seed)
    usage, design = common.load(common.USAGE), common.load(common.DESIGN)['v4_combination_source_plan_20261114']
    queue = build_queue(ROOT)
    selection = common.load(ground.bike.SELECTION)['vehicles']
    fg = {'upright': [], 'fallen': []}
    for record in usage['records']:
        for rel in common.relation(record, 'target_foreground', {'ready'}):
            fg[rel['posture']].extend(x for x in selection if x['path'] == record['image'] and x['class'] == rel['posture'])
    if any(len(v) != 1 for v in fg.values()):
        raise ValueError('expected exactly one reviewed foreground for each posture')
    for entries in fg.values():
        for entry in entries:
            for key in ('path', 'source_frame', 'source_video'):
                path = ROOT / entry[key]
                if not path.is_file() or ground.source_role(ROOT, path, queue)[0] == 'regression':
                    raise ValueError('missing or regression-only foreground source: ' + entry[key])
    for bg in backgrounds:
        if any(not (ROOT/bg[k]).is_file() for k in ('image','parent_image','source_video')):
            raise ValueError('missing background source')
        if ground.source_role(ROOT, ROOT/bg['parent_image'], queue)[0] == 'regression':
            raise ValueError('regression background cannot be used for train')
    groups = {group: [b for b in backgrounds if b['source_group'] == group]
              for group in ('full_scene_video', 'legacy_scene_001_video')}
    if not all(groups.values()):
        raise ValueError('both source video groups are required')
    sprites = {}
    for cls, entries in fg.items():
        entry = entries[0]
        src = ground.bike.scene.assets.load_rgba(ROOT/entry['path'])
        for yaw in (-6, 0, 6):
            warped = planar_warp(src, yaw, 0, 0)
            for size in (120, 160, 200, 220):
                sprite = ground.bike.scene.assets.resized(warped, **({'height':size} if cls=='upright' else {'width':size}))
                if int((sprite[:,:,3] > 127).sum()) < 120:
                    raise ValueError('foreground below minimum visibility')
                sprites[(cls,yaw,size)] = sprite
    states, evidence = [], []
    for branch, classes in zip(design['branches'], common.CLASSES):
        event = branch['combination_id']
        target = branch['real_background_composite_images']
        if not classes:
            target = sum(len(v)*len(ground.TONES) for v in groups.values())
        abstract = []
        for name, prior in (('full_scene_video',(2,3)), ('legacy_scene_001_video',(2,2))):
            cap = len(groups[name])*len(ground.TONES) if not classes else target-1
            abstract.append({'id':name, 'failure_event':event, 'min_count':1, 'max_count':cap,
                             'prior':{'alpha':prior[0],'beta':prior[1]}, 'exposure_weight':1,
                             'failure_cost':1, 'conditions':['parent_group:'+name]})
        request = {'schema_version':1,'model_key':'e_bike','target_split':'train','budget':target,
                   'purpose':'Allocate by parent video before sampling; priors are design assumptions.',
                   'uncertainty_weight':0,'independence_basis':'Two known source videos; independent capture sessions are unverified.',
                   'strata':abstract}
        result = allocate(request, ROOT)
        reference = copy.deepcopy(request)
        for item in reference['strata']:
            item['prior'] = {'alpha':1,'beta':1}
        equal = allocate(reference, ROOT)
        evidence.append({'branch':event,'request':request,'result':result,
                         'equal_prior_counts':{x['id']:x['planned_count'] for x in equal['strata']}})
        seen = set()
        for allocation in result['strata']:
            for _ in range(allocation['planned_count']):
                for attempt in range(5000):
                    bg = rng.choice(groups[allocation['id']])
                    # The narrow 005751 floor region cannot hold a complete
                    # vehicle at the reviewed reference scale.
                    if classes and bg['id'] == 'frame_005751':
                        continue
                    tone = rng.choice(tuple(ground.TONES))
                    actors, targets, bounds = [], [], []
                    for cls in classes:
                        yaw = rng.choice((-6,0,6))
                        poly = rng.choice(bg['safe_regions'])
                        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
                        x, y = rng.randint(min(xs)+5,max(xs)-5), rng.randint(min(ys)+5,max(ys)-5)
                        # Reference boxes occupy about one third of a 360-pixel
                        # frame; scale to a 480-pixel frame and increase nearer
                        # the bottom. Perspective bands are design estimates.
                        size = 120 if y < 340 else 160 if y < 400 else 200 if y < 440 else 220
                        sprite = sprites[(cls,yaw,size)]
                        spec = {'foot_x':x, 'foot_y':y, 'size_px':size, 'yaw':yaw, 'pitch':0, 'roll':0,
                                'relative_exposure':rng.choice((.95,1.,1.05))}
                        actor = dict(spec, sprite=sprite)
                        try:
                            ground.bike.scene.validate_actor_placement(actor, 'ground_only')
                            ground.validate_safe_placement(actor, bg)
                            if not ground.grounded(actor, ground.ground_zone(bg)):
                                raise ValueError('not grounded')
                        except ValueError:
                            break
                        h,w = sprite.shape[:2]; l,t = round(x-w/2),round(y-h); box=(l,t,l+w,t+h)
                        if l<0 or t<0 or box[2]>640 or box[3]>480 or any(
                                box[0]<b[2] and b[0]<box[2] and box[1]<b[3] and b[1]<box[3] for b in bounds):
                            break
                        bounds.append(box);actors.append(spec);targets.append({'class':cls,'source':fg[cls][0]['path']})
                    if len(actors) != len(classes):
                        continue
                    signature=json.dumps([bg['id'],targets,actors,tone],sort_keys=True)
                    if signature in seen:
                        continue
                    seen.add(signature)
                    break
                else:
                    raise ValueError('could not sample safe distinct state: '+event)
                used={bg[k] for k in ('image','parent_image','source_video')}
                for cls in classes:
                    used.update(fg[cls][0][k] for k in ('path','source_frame','source_video','source_image') if k in fg[cls][0])
                gain,sat=ground.TONES[tone]
                state=common.state_base(len(states)+1,event)
                state.update({'background':{'id':bg['id'],'image':bg['image']},'target':{'vehicles':targets},
                              'labels':ground.GROUND_LABELS,'augmentation':{'tone':tone},
                              'parameters':{'actors':actors,'value_gain':gain,'saturation_gain':sat},'sources':sorted(used)})
                states.append(state)
    plan=common.header(states,ground.SAFE_CONSTRAINTS,'Reviewed floor regions and two source videos; generated images remain candidates.')
    plan['allocation_evidence']={'method':'Beta design priors allocate parent-video quotas before sampling',
        'basis':'Design assumptions, not measured failure rates; equal-prior comparison retained.',
        'scale_reference':'datasets/annotated/e_bike/v4_low_angle_multibike_review_20261112/review_manifest.json: unverified candidate upright boxes about 97–126 px high and fallen boxes 112–125 px wide in a 360px image; 120/160/200/220px bands for 480px backgrounds increase toward near ground. Frame normalization and near-field extension are image-space design estimates, not physical calibration.',
        'branches':evidence}
    _,vehicles=ground.bike.sources(include_people=False)
    original=ground.source_role;cache={}
    def cached(root,path,known):
        if path not in cache:cache[path]=original(root,path,known)
        return cache[path]
    ground.source_role=cached
    try:ground.validate_ground_plan(plan,backgrounds,vehicles,queue)
    finally:ground.source_role=original
    plan_generation(plan,ROOT)
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-manifest',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    out=(ROOT/args.output_dir).resolve();owner=(ROOT/'datasets/scripts/e_bike/config/plans').resolve()
    if not out.is_relative_to(owner) or out==owner or out.exists():
        parser.error('output must be a new directory under datasets/scripts/e_bike/config/plans')
    backgrounds=reviewed_backgrounds((ROOT/args.legacy_manifest).resolve())
    plan=make_plan(backgrounds)
    out.mkdir(parents=True)
    for name,value in (('backgrounds',{'backgrounds':backgrounds}),('plan',plan)):
        (out/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('planned',plan['budget'],'states from',len(backgrounds),'backgrounds')


if __name__=='__main__':
    main()
