"""Compile the current quota design into exact split and foreground states."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from datasets.scripts.common.config import load_task_config, read_yaml
from datasets.scripts.meter.backgrounds import select_backgrounds
from datasets.scripts.meter.build_current_sample import direct_sources, read, write
from datasets.scripts.meter.full_variations import degradation_controls

ROOT=Path(__file__).resolve().parents[3]
SPLITS=('train','val','test')
SCENES=('Arena','Arena Indoor Soft Light','Arena Detailed')
ROUTES=('person','e_bike','fire','building_name_paper','sign','waste_bin_state','waste_sticker','meter')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def deck(weights,count,rng):
    keys=list(weights);total=sum(weights.values())
    quotas={key:int(weights[key]/total*count) for key in keys}
    order=sorted(keys,key=lambda key:weights[key]/total*count-quotas[key],reverse=True)
    for key in order[:count-sum(quotas.values())]:quotas[key]+=1
    values=[key for key in keys for _ in range(quotas[key])]
    rng.shuffle(values)
    return values


def state(sid,p,sources,*,negative=False):
    return {'id':sid,'failure_event':'meter_presence_eight_point_or_capture_condition',
            'min_count':1,'max_count':1,'prior':{'alpha':1,'beta':1},'exposure_weight':1.,'failure_cost':1.,
            'conditions':[],'background':{'kind':p['branch']},
            'target':{'kind':'empty_meter_scene' if negative else 'dual_needle_meter'},
            'labels':{'kind':'empty_pose_label' if negative else 'eight_point_pose','kpt_shape':[8,3]},
            'augmentation':{'kind':'declared_coordinate_and_rgb_controls'},'parameters':p,'sources':sources}


def plan(split,states,purpose,**kwargs):
    return {'schema_version':1,'model_key':'meter','target_split':split,'budget':len(states),
            'constraints':{'resolution':[640,480],'kpt_shape':[8,3],'label_policy':'explicit_known_geometry_or_native_visual_pose'},
            'uncertainty_weight':0,'purpose':purpose,
            'independence_basis':'Split-specific renders and source groups; shared meter and room assets; synthetic parameter generalization.',
            'strata':states,**kwargs}


def background_inventory(split,recipe,source_blend):
    if split=='train':
        return select_backgrounds({'split':'train','frames_per_scene_route':24})
    rows=[]
    for si,scene in enumerate(SCENES):
        folder=f"{recipe['work_root']}/{split}/backgrounds/chunk_{si:04d}"
        for route in ROUTES:
            for frame in recipe['background_frames'][split]:
                name=str(float(frame)).replace('.','p')
                key=f'{split}:{scene}:{route}:{frame}'
                rows.append({'record_id':None,'background_key':key,'image':f'{folder}/{route}_f{name}.png',
                             'source_kind':'blender_render','image_size':[640,480],'split':split,'scene':scene,
                             'route':route,'render_frame':frame,'parent_group':'datasets/assets/blender/aiaec_dataset.blend',
                             'source_blend':source_blend,'qualification':'pending_render_inventory_and_intake'})
    return rows


def compile_plans():
    _,source=load_task_config('meter')
    recipe=read(ROOT/source['full_recipe'])
    sample=read(ROOT/source['sample_recipe'])
    allocation=read_yaml(ROOT/source['full_design'])['tasks']['meter']['split_design']
    views=read(ROOT/sample['camera_inventory'])['views']
    work=ROOT/recipe['work_root'];inputs=work/'production_inputs'
    inputs.mkdir(parents=True,exist_ok=True)
    source_blend=(inputs/'source.blend').relative_to(ROOT).as_posix()
    if not (ROOT/source_blend).is_file():
        import shutil
        shutil.copy2(ROOT/source['station_source'],ROOT/source_blend)
    artifacts={}
    direct=dict(direct_sources(sample))
    all_pairs={}
    for split_index,split in enumerate(SPLITS):
        rng=random.Random(recipe['seed']+split_index*100003)
        n=recipe['geometry_groups'][split]
        distribution=allocation['synthetic_positive_distributions']
        readings=deck({'continuous':recipe['continuous_reading_share'],'grid_anchor':1-recipe['continuous_reading_share']},n,rng)
        levels=allocation['reading_grid'][split]
        view_weights={}
        for view in views:
            distance={1.35:'near_1p35m',1.9:'middle_1p90m',2.5:'far_2p50m'}[view['distance_m']]
            yaw={-15:'left15',0:'frontal',15:'right15'}[view['yaw_deg']]
            elevation={-8:'up8',0:'level',8:'down8'}[view['elevation_deg']]
            view_weights[view['view']]=distribution['distance'][distance]*distribution['yaw'][yaw]*distribution['elevation'][elevation]
        camera_deck=deck(view_weights,n,rng);by_view={v['view']:v for v in views}
        foreground=[];grid_index=0
        for i in range(n):
            if readings[i]=='continuous':t,h=round(rng.uniform(.001,.999),6),round(rng.uniform(.001,.999),6)
            else:
                pair=grid_index%(len(levels)**2);grid_index+=1;t,h=levels[pair//len(levels)],levels[pair%len(levels)]
            view=by_view[camera_deck[i]]
            glare_modes=['none']
            if i%3==0:glare_modes.append(('temperature_tip','humidity_tip','temperature_max')[(i//3)%3])
            for glare in glare_modes:
                for material in ('white','colored'):
                    fid=f'meter_{split}_fg_{i:04d}_{material}_{glare}'
                    p={'branch':'blender_render','view':view['view'],'face_material':material,
                       'temperature_fraction':t,'humidity_fraction':h,'reading_sampling':readings[i],
                       'distance_m':view['distance_m'],'yaw_deg':view['yaw_deg'],'elevation_deg':view['elevation_deg'],
                       'glare':glare,'geometry_group':i}
                    foreground.append(state(fid,p,[source_blend,sample['face_variants'][material]]))
        pairs={(s['parameters']['temperature_fraction'],s['parameters']['humidity_fraction']) for s in foreground}
        if any(pairs&previous for previous in all_pairs.values()):raise ValueError('Needle pairs overlap across splits')
        all_pairs[split]=pairs
        render=plan(split,sorted(foreground,key=lambda s:(s['parameters']['view'],s['id'])),
                    'Split-specific transparent meter foregrounds',source_blend=source_blend,
                    cycles_samples=recipe['cycles_samples'],render_seed=recipe['seed']+split_index*100003,
                    endpoint_config=source['endpoint_config'],face_variants=sample['face_variants'],
                    camera_lens_scale=recipe['lens_scale'][split],render_exposure_ev=recipe['render_exposure_ev'][split])
        inventory=background_inventory(split,recipe,source_blend)
        for row in inventory:row.setdefault('background_key',row['record_id'])
        inventory_by_key={r['background_key']:r for r in inventory}
        quota=allocation['splits'][split]
        states=[]
        for index in allocation['direct_record_groups'][split]['indices']:
            original=direct[index]
            assert original['pose_annotation']['complete']
            p={'branch':'direct_task_pose','source_record_index':index,'source_image':original['path']}
            states.append(state(f'meter_{split}_real_{index:04d}',p,[original['path']]))
        assert len(states)==quota['direct_real']
        for branch in ('real_background_composite','blender_background_composite'):
            count=quota[branch]
            if not count:continue
            backgrounds=[r for r in inventory if (r['source_kind']=='blender_render')==(branch=='blender_background_composite')]
            if not backgrounds:raise ValueError('No backgrounds for planned branch '+split+'/'+branch)
            pair_count=(count+1)//2
            axes={key:deck(distribution[key],pair_count,rng) for key in ('saturation','exposure','light_relation','glare','occlusion')}
            degradations=deck(recipe['degradation_distribution'],pair_count,rng)
            clean={material:[s for s in foreground if s['parameters']['face_material']==material and s['parameters']['glare']=='none'] for material in ('white','colored')}
            glare={material:[s for s in foreground if s['parameters']['face_material']==material and s['parameters']['glare']!='none'] for material in ('white','colored')}
            for j in range(pair_count):
                controls={'x_fraction':rng.random(),'y_fraction':rng.random(),
                          'scale_factor':round(rng.uniform(*recipe['scale_factor_range']),5)}
                degradation=degradation_controls(rng,degradations[j])
                occlusion=None if axes['occlusion'][j]=='none' else {'x_fraction':rng.choice([.05,.95]),'y_fraction':rng.uniform(.25,.75),'rgb':[45,48,46]}
                bg=backgrounds[j%len(backgrounds)]
                for material in ('white','colored'):
                    if j*2+(material=='colored')>=count:continue
                    pool=(clean if axes['glare'][j]=='none' else glare)[material]
                    fg=pool[(j*17)%len(pool)]
                    p={**fg['parameters'],'branch':branch,'foreground_id':fg['id'],'background_key':bg['background_key'],
                       **{key:axes[key][j] for key in ('saturation','exposure','light_relation')},
                       'placement_controls':controls,'degradation':degradation,'occlusion':occlusion}
                    sid=f'meter_{split}_{branch}_{j:04d}_{material}'
                    states.append(state(sid,p,fg['sources']+[bg['image']]))
        count=quota['empty_negative']
        axes={key:deck(distribution[key],count,rng) for key in ('saturation','exposure')}
        degradations=deck(recipe['degradation_distribution'],count,rng)
        for i in range(count):
            bg=inventory[i%len(inventory)]
            p={'branch':'empty_negative','background_key':bg['background_key'],
               'saturation':axes['saturation'][i],'exposure':axes['exposure'][i],
               'degradation':degradation_controls(rng,'sensor_noise' if i>=len(inventory) and degradations[i]=='clean' else degradations[i])}
            states.append(state(f'meter_{split}_negative_{i:04d}',p,[bg['image']],negative=True))
        assert len(states)==quota['total'],(split,len(states),quota)
        exact=plan(split,states,'Full meter pose dataset with continuous readings, randomized layout and camera degradation',
                   render_plan=f'datasets/scripts/meter/config/full_{split}_render_plan.json',
                   background_inventory=f"{recipe['work_root']}/{split}/background_inventory.json",
                   recipe=source['full_recipe'],quota=quota)
        write(ROOT/f'datasets/scripts/meter/config/full_{split}_plan.json',exact)
        write(ROOT/exact['render_plan'],render)
        write(ROOT/exact['background_inventory'],{'task':'meter','split':split,'backgrounds':inventory})
        artifacts[split]={'images':len(states),'foreground_renders':len(foreground),'backgrounds':len(inventory),
                          'branches':dict(Counter(s['parameters']['branch'] for s in states)),
                          'render_reading_modes':dict(Counter(s['parameters']['reading_sampling'] for s in foreground)),
                          'plan_sha256':digest(exact)}
    write(inputs/'compilation.json',{'splits':artifacts,'source_blend':source_blend,
          'source_blend_sha256':hashlib.sha256((ROOT/source_blend).read_bytes()).hexdigest()})
    return artifacts
