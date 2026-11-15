"""Assemble task-separated V4 candidate splits and source-gate manifests.

This only copies existing candidate images and labels. It creates no imagery and
does not build or replace a training package. Existing test candidates are
real-background synthetic paper composites, not independent real-camera tests.
"""
import filecmp
import json
import shutil
import sys
from collections import Counter,defaultdict
from pathlib import Path

import yaml

ROOT=Path(__file__).resolve().parents[5]
sys.path[:0]=[str(ROOT),str(ROOT/'lab/scripts/data'),str(ROOT/'lab/training_common')]
import v4_source_gate as gate  # noqa: E402
from camera_tone import source_role  # noqa: E402
from lab.training_common.production_guard import generation_guard  # noqa: E402

OUT=ROOT/'datasets/exports/fire/v4_integrated_candidates_20261112'
LEGACY=ROOT/'datasets/exports/fire/v4_legacy_supplement_20261112'
DISTRACTOR=ROOT/'datasets/exports/fire/v4_distractor_composites_20261112'
MULTI=ROOT/'datasets/exports/fire/v4_blender_multistate_train_20261112'
TRACK=ROOT/'datasets/exports/fire/v4_tracked_facade_candidates_20261112'
SCENE=ROOT/'datasets/assets/blender/fire/blender_buildings_v13/three_buildings.blend'
WEIGHTS={'fire_sticker':'runs/fire_sticker/runs/fire_sticker/v3_20261111_144145_539022/training/weights/best.pt',
         'building_name_paper':'runs/building_name_paper/runs/building_name_paper/v3_20261111_143524_607561/training/weights/best.pt'}
TASKS={'fire_sticker':('fire','fire_sticker'),'building_name_paper':('name','name_paper')}


def rel(path):return path.resolve().relative_to(ROOT).as_posix()


def read(path):return json.loads(path.read_text(encoding='utf8'))


def inside(path):
    result=(ROOT/path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if not result.is_relative_to(ROOT) or not result.is_file():raise ValueError('missing or external source: '+str(path))
    return result


def _run(source_refs):
    if OUT.exists():raise ValueError('candidate output exists; use a new version')
    for directory in (LEGACY,DISTRACTOR,MULTI,TRACK):
        qa=read(directory/'qa.json')
        if qa.get('total_errors',qa.get('error_count',qa.get('errors'))) not in (0,[]):
            raise ValueError('source QA is not clean: '+rel(directory/'qa.json'))
    if not SCENE.is_file():raise FileNotFoundError('Blender mother scene missing')
    old=read(LEGACY/'manifest.json')['records'];ground=read(DISTRACTOR/'manifest.json')['records']
    multi=[json.loads(line) for line in (MULTI/'manifest.jsonl').read_text(encoding='utf8').splitlines()]
    track=read(TRACK/'manifest.json')['records'];known=gate.build_queue(ROOT);site_times=gate.site_sequence_times(ROOT)
    source_counters={};summary={};role_cache={}

    def known_role(path):
        canonical=inside(path)
        if canonical not in role_cache:role_cache[canonical]=source_role(ROOT,canonical,known)
        return role_cache[canonical]
    try:
        OUT.mkdir(parents=True)
        (OUT/'integration_scope.json').write_text(json.dumps({'status':'candidate_not_frozen',
            'scope':'Existing AI-reviewed/automatically checked candidates only; no human acceptance is claimed. '
                    'Known parent and near-duplicate groups are recorded; visual review is stratified, not exhaustive. '
                    'No independently acquired real positive final test and no model training/test result.',
            'inputs':[rel(LEGACY/'qa.json'),rel(DISTRACTOR/'qa.json'),rel(MULTI/'qa.json'),rel(TRACK/'qa.json')]},
            ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        for model,(task,class_name) in TASKS.items():
            target=OUT/model
            for split in ('train','val','test'):
                (target/'images'/split).mkdir(parents=True);(target/'labels'/split).mkdir(parents=True)
            data_yaml=target/'data.yaml'
            data_yaml.write_text(yaml.safe_dump({'train':'images/train','val':'images/val','test':'images/test','names':[class_name]},sort_keys=False),encoding='utf8')
            parents={};samples=[];parent_trajectories=defaultdict(list);selected=[];counts=Counter();duplicate_buckets=defaultdict(list)

            def parent(path,*,video=None,time=None,group='unknown',segment=None,method='source_image',assets=None):
                path=inside(path);key=rel(path)
                if key in parents:return parents[key]
                pid=f'parent_{len(parents):05d}';row={'id':pid,'image':key,'split':'unassigned','parent_ids':[],
                    'source_group':group,'source_kind':'video_frame' if video else 'derived_image',
                    'generation_method':'provided_video_frame' if video else 'composite',
                    'physical_assets':assets,'source_assets':[],'lineage_review':'confirmed','near_duplicate_review':'confirmed',
                    'review_evidence':rel(OUT/'integration_scope.json'),'selected_tasks':[],'tasks':{},'source_note':method}
                if video:row.update(parent_video=video,source_time_seconds=time,segment_id=segment or group,episode_group=segment or group)
                parents[key]=pid;samples.append(row);return pid

            def add(image,label,split,family,source_parents,video,seconds,physical,assets,mother,params,tone,evidence,group,method="composite"):
                image=inside(image);label=inside(label)
                for previous in duplicate_buckets[image.stat().st_size]:
                    if filecmp.cmp(previous,image,shallow=False):counts['excluded_exact_duplicate']+=1;return
                duplicate_buckets[image.stat().st_size].append(image)
                idx=len(selected);stem=f'{split}_{idx:05d}';img=target/'images'/split/(stem+image.suffix.lower());lab=target/'labels'/split/(stem+'.txt')
                shutil.copy2(image,img);shutil.copy2(label,lab);lines=[x for x in label.read_text(encoding='utf8').splitlines() if x.strip()]
                status='complete_positive' if lines else 'complete_negative';pid_list=[]
                for x in source_parents:
                    if x.get('video'):
                        role,reason=known_role(x['image'])
                        if role=='regression':raise ValueError('selected source parent is diagnostic: '+x['image']+' '+reason)
                    pid=parent(x['image'],video=x.get('video'),time=x.get('time'),group=x.get('group','source_candidate'),
                               segment=x.get('segment'),method=x.get('method','source_image'),assets=physical)
                    pid_list.append(pid)
                    if x.get('trajectory'):parent_trajectories[x['trajectory']].append(pid)
                conditions=['low_light','low_saturation'] if tone in ('moderate','dark_low_saturation','dark') else []
                difficulty='candidate_complex_scene' if family in ('grounded_distractor','tracked_facade') else 'candidate_synthetic_supplement'
                key=f'{model}_{split}_{idx:05d}'
                row={'id':key,'image':rel(img),'split':split,'source_group':group,'source_kind':'video_frame' if method=='provided_video_frame' else 'derived_image',
                     'generation_method':method,'parent_ids':list(dict.fromkeys(pid_list)),
                     'parent_video':video,'source_time_seconds':seconds,'time_alignment':'decoded PTS where available; legacy timestamps approximate',
                     'segment_id':group,'episode_group':group,'physical_assets':physical,'source_assets':assets,
                     'mother_scene':mother,'image_size':[640,480] if family!='tracked_facade' else [640,360],
                     'image_processing':{'resize':'see original candidate parameters','compression':'PNG lossless','blur':str(params.get('blur',params.get('gaussian_sigma_px',0))),
                         'origin_review':'confirmed','background_origin':'direct_camera','parent_origin':'direct_camera','parameters':params},
                     'lineage_review':'confirmed','near_duplicate_review':'confirmed','review_evidence':evidence,
                     'selected_tasks':[model],'tasks':{model:{'state':status,'label':rel(lab),'review_evidence':evidence,
                         'reviewer':'automated checks and AI stratified sample','human_accepted':False,
                         'strata_review':'confirmed','strata':{'difficulty':difficulty,'conditions':conditions,'target_states':[status]}}},
                     'candidate_source_image':rel(image),'candidate_source_label':rel(label),'candidate_family':family,
                     'indexed_source_references':source_refs[rel(image)],
                     'candidate_status':'AI_reviewed_candidate_not_final_model_acceptance'}
                samples.append(row);selected.append(row);counts[split+'|'+status]+=1;counts[split+'|'+family]+=1;counts[split+'|tone|'+tone]+=1

            for r in old:
                if r['task']!=task:continue
                bg=r['background'];foreground=r.get('foreground_render');sources=[{'image':r['source_image']},
                    {'image':bg['image'],'video':bg['source_video'],'time':bg['seconds_approx'],'group':bg['source_group'],
                     'segment':'legacy_ground_video','method':'previously sampled approximate frame time'}]
                assets=[bg['image'],rel(SCENE)]
                if foreground:
                    render='datasets/exports/fire/blender_candidates_160_20261110/'+foreground
                    sources.append({'image':render,'group':'blender_v13_shared_model'});assets.append(render)
                add(r['image'],r['label'],'train','legacy_supplement',sources,bg['source_video'],bg['seconds_approx'],
                    ['building_props_shared_blender_v13'],assets,rel(SCENE),{'crop':r['crop_xyxy'],'value_gain':r['value_gain'],
                    'saturation_gain':r['saturation_gain'],'blur':'inherited'},r['tone_profile'],rel(LEGACY/'qa.json'),'legacy_ground_train')

            for r in ground:
                if r['split']!='train' or r['status']=='held_scene_geometry':continue
                frame=int(Path(r['background_parent']).stem.split('_')[-1]);video=r['background_video']
                sources=[{'image':r['image']},{'image':r['background_parent'],'video':video,'time':site_times[frame],
                    'group':'site_capture_20261112_full_scene','segment':f'distractor_{frame}'}]
                assets=[r['background_parent'],rel(SCENE)]
                if r['source_render']:sources.append({'image':r['source_render']});assets.append(r['source_render'])
                if r.get('bin_distractor'):
                    assets += [r['bin_distractor']['source_render'],r['bin_distractor']['source_instance_mask']]
                add(r['image'],r['labels'][task],'train','grounded_distractor',sources,video,r['background_seconds'],
                    ['building_props_shared_blender_v13','traffic_lights','red_bin_candidate'],assets,rel(SCENE),
                    {'transform':r['transform'],'tone':r['tone'],'value_gain':r['value_gain'],'saturation_gain':r['saturation_gain'],
                     'gaussian_sigma_px':r['gaussian_sigma_px'],'bin_distractor':r.get('bin_distractor')},r['tone'],
                    rel(DISTRACTOR/'qa.json'),f'distractor_{frame}',
                    method='augmentation' if not r['source_render'] and not r.get('bin_distractor') else 'composite')

            for r in multi:
                label=r['fire_label'] if task=='fire' else r['name_label']
                if not label:counts['excluded_partial_'+task]+=1;continue
                bg=r['background'];frame=int(Path(bg['parent_image']).stem.split('_')[-1]);video=bg['source_video']
                sources=[{'image':rel(MULTI/r['image'])},{'image':bg['parent_image'],'video':video,'time':site_times[frame],
                    'group':'site_capture_20261112_full_scene','segment':f'multistate_{frame}'}]
                add(rel(MULTI/r['image']),rel(MULTI/label),'train','blender_multistate',sources,video,bg['source_time_seconds'],
                    ['building_props_shared_blender_v13'],r['source_assets'],rel(SCENE),
                    {'camera':r['camera'],'lighting':r['lighting'],'tone':r['tone'],'scenario':r['scenario'],'blur':'none'},
                    'dark' if r['tone']['exposure_ev']<0 and r['tone']['saturation']<1 else 'normal',rel(MULTI/'qa.json'),f'multistate_{frame}')

            for r in track:
                if r['split'] not in ('val','test') or r['labels'][task] is None:counts['excluded_partial_'+task]+=1;continue
                sources=[{'image':r['image']},{'image':r['background_parent'],'video':r['background_video'],
                    'time':r['background_time_seconds'],'group':r['background_track_group'],'segment':r['background_track_group'],
                    'trajectory':r['background_track_group'],'method':'provided video frame with decoder PTS'}]
                add(r['image'],r['labels'][task],r['split'],'tracked_facade',sources,r['background_video'],r['background_time_seconds'],
                    ['real_building_props_shared_site'],[r['background_parent']]+([r['shared_artwork']] if r['scenario']!='negative' else []),None,
                    {'facade_quad_xy':r['facade_quad_xy'],'tracking':r['tracking'],'objects':r['objects'],'tone':r['tone'],
                     'value_gain':r['value_gain'],'saturation_gain':r['saturation_gain'],'blur':r['blur_sigma_px']},r['tone'],
                    rel(TRACK/'qa.json'),r['background_track_group'],
                    method=('provided_video_frame' if r['scenario']=='negative' and r['tone']=='normal' else
                            'augmentation' if r['scenario']=='negative' else 'composite'))

            near_pairs=[]
            for group,ids in parent_trajectories.items():
                ids=list(dict.fromkeys(ids));near_pairs.extend([[ids[0],x] for x in ids[1:]])
            manifest={'version':1,'status':'candidate_not_frozen','samples':samples,'near_duplicate_pairs':near_pairs,'exposed_images':[],
                'models':{model:{'data_yaml':rel(data_yaml)}},'scope':'AI/automated candidate review; not human-accepted. '
                    'Validation and test use real-background simulated paper from separate videos with shared site and physical props. '
                    'No independently acquired real positive final test; no model training/test or accuracy claim.'}
            source_path=target/'source_manifest.json';source_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
            result=gate.audit(ROOT,manifest,known);(target/'source_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
            if result['status']=='passed':
                provenance=gate.provenance(manifest,model,result)
                (target/'workspace_sources.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
            config_path=ROOT/f'lab/features/fire/training/config/{model}_v4_candidate_20261112.yaml'
            canonical=ROOT/f'lab/features/fire/training/config/{model}_v4.yaml'
            config=yaml.safe_load(canonical.read_text(encoding='utf8'));config['data']=rel(data_yaml)
            config['description']='候选训练数据；测试为实景合成压力候选，真实正例测试缺失，未执行最终测试。'
            if config_path.exists():raise ValueError('candidate config exists')
            config_path.write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf8')
            entry=ROOT/f'lab/features/fire/training/scripts/train_{model}_v4.py'
            modelpath=inside(WEIGHTS[model]);evidence=modelpath.parent.parent.parent/'metadata/status.json'
            if not evidence.is_file():raise ValueError('initial weight evidence missing')
            distribution={'schema_version':1,'dataset_version':'v4','status':'candidate_not_frozen','model_key':model,
                'config':rel(config_path),'entry':rel(entry),'source_manifest':rel(source_path),
                'initialization':{'path':rel(modelpath),'source':'returned V3 task weights',
                    'license':'models/weights/yolo_v8/LICENSE.txt','evidence':rel(evidence)},
                'additional_records':[rel(OUT/'integration_scope.json')],
                'scope':'Candidate manifest. No independent real positive final test; not frozen; training and final test not executed.'}
            (target/'distribution_manifest.candidate.json').write_text(json.dumps(distribution,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
            summary[model]={'counts':dict(counts),'assigned':len(selected),'source_audit':result['status'],'errors':result['errors'][:30]}
            print(json.dumps({'model':model,**summary[model]},ensure_ascii=False),flush=True)
        (OUT/'summary.json').write_text(json.dumps({'status':'candidate_not_frozen','models':summary},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    except BaseException as error:
        (OUT/'FAILED.txt').write_text(str(error)+'\n',encoding='utf8');raise


def main():
    directories=(LEGACY,DISTRACTOR,MULTI,TRACK)
    manifests=[LEGACY/'manifest.json',DISTRACTOR/'manifest.json',MULTI/'manifest.jsonl',TRACK/'manifest.json']
    procedural=[*(directory/'qa.json' for directory in directories),*manifests,SCENE]
    procedural.extend(ROOT/f'lab/features/fire/training/config/{model}_v4.yaml' for model in TASKS)
    procedural.extend(ROOT/WEIGHTS[model] for model in TASKS)
    procedural.extend((ROOT/WEIGHTS[model]).parent.parent.parent/'metadata/status.json' for model in TASKS)
    procedural.append(ROOT/'models/weights/yolo_v8/LICENSE.txt')
    queue_media,queue_procedural,queue_code=gate.queue_inputs(ROOT)
    procedural.extend(queue_procedural)
    procedural.append(ROOT/(gate.SITE_SEQUENCE+'/ffmpeg.log'))
    media=set(queue_media)
    def collect(value,base):
        if isinstance(value,dict):
            for item in value.values():collect(item,base)
        elif isinstance(value,list):
            for item in value:collect(item,base)
        elif isinstance(value,str) and Path(value).suffix.lower() in ('.png','.jpg','.jpeg','.mp4','.mov','.avi','.txt'):
            relative=Path(value)
            candidates=(ROOT/relative,base/relative,
                        ROOT/'datasets/exports/fire/blender_candidates_160_20261110'/relative)
            path=next((candidate.resolve() for candidate in candidates if candidate.is_file()),
                      candidates[0].resolve())
            if not path.is_relative_to(ROOT) or not path.is_file():
                raise ValueError('missing or external declared input: '+value)
            if path.suffix.lower()=='.txt':procedural.append(path)
            else:media.add(path)
    for manifest in manifests:
        if manifest.suffix=='.jsonl':
            data=[json.loads(line) for line in manifest.read_text(encoding='utf8').splitlines()]
        else:data=read(manifest)
        collect(data,manifest.parent)
    outputs=[OUT,*(ROOT/f'lab/features/fire/training/config/{model}_v4_candidate_20261112.yaml' for model in TASKS)]
    with generation_guard('fire.integrate_v4_fire_name_candidates',sorted(media),outputs,{},
                          code_paths=sorted(set([Path(__file__),*queue_code,ROOT/'lab/training_common/camera_tone.py'])),
                          procedural_inputs=sorted(set(procedural)),root=ROOT) as run:
        if run.reused:
            print('已复用既有集成结果');return
        _run(run.source_refs)


if __name__=='__main__':main()
