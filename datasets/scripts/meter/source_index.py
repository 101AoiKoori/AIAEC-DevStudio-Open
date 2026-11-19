"""Bridge audited meter split records to the existing V4 training source contract."""
from pathlib import Path
from collections import Counter,defaultdict
import json
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from datasets.scripts.common.config import load_task_config, read_yaml
from datasets.scripts.meter.build_current_sample import read, write
from lab.training_common.production_guard import generation_guard


def source_record(row,split,source,direct_groups):
    base=Path('datasets/exports/meter')/split
    image=(base/row['image']).as_posix();label=(base/row['pose_label']).as_posix()
    annotation=(base/row['annotation']).as_posix()
    branch=row['parameters']['branch']
    record={'image':image,'label':label,'split':split,'workspace_image':image,'workspace_label':label,
            'label_status':'reviewed','review_state':'complete_positive' if row['instance_count'] else 'complete_negative',
            'review_evidence':annotation,'source_review_evidence':annotation,
            'evaluation_scope':'controlled_shared_asset_diagnostic','strata_review':'confirmed'}
    p=row['parameters']
    conditions=[f'{key}:{p[key]}' for key in ('reading_sampling','face_material','saturation','exposure','light_relation','glare','view') if key in p]
    if 'degradation' in p:conditions.append('degradation:'+p['degradation']['kind'])
    if 'placement_controls' in p:conditions.append('randomized_position_and_scale')
    record['strata']={'conditions':conditions,'target_states':['single' if row['instance_count'] else 'absent']}
    if branch=='direct_task_pose':
        index=row['source_record_index']
        visual=Path('datasets/annotated/meter/real_pose_visual')/f'{index:04d}.json'
        record.update(source_group=direct_groups[split]['parent'],source_id=direct_groups[split]['parent'],
            source_kind='video_frame' if 12<=index<=55 else 'photo',
            generation_method='provided_video_frame' if 12<=index<=55 else 'camera_capture',
            image_processing={'resolution_review':'confirmed','source_size':row['image_size'],
                              'output_size':row['image_size'],'resampling':'none','clarity':'original'},
            source_assets=[row['source_image']],
            source_review_evidence=visual.as_posix() if (ROOT/visual).is_file() else source['direct_review_manifest'],
            development_contact=['direct_visual_pose_annotation'])
        if split=='train':record['parent_video']='Archive/Assets/captures/car_real_capture_20261112/V_20261112_152304.mp4'
    else:
        background=row['background'];rendered=background['source_kind']=='blender_render'
        parent=background['parent_group']
        record.update(source_group=parent,source_id=parent,source_assets=list(row['sources']),
                      parent_samples=[background['image']])
        if rendered:
            record.update(source_kind='synthetic_scene',generation_method='synthetic_render',
                          mother_scene='datasets/assets/blender/aiaec_dataset.blend',
                          image_processing={'origin_review':'confirmed','parent_origin':'synthetic',
                                            'background_origin':'synthetic','render_composite':bool(row['instance_count'])})
        else:
            record.update(source_kind='derived_image',generation_method='composite' if row['instance_count'] else 'augmentation',
                          parent_video=parent,
                          image_processing={'origin_review':'confirmed','background_origin':'direct_camera',
                                            'parent_origin':'direct_camera'})
        if row['instance_count']:
            record['physical_assets']=['dual_needle_meter_station']
            record['parent_samples'].append(row['foreground_id'])
            record['foreground_evidence']=row['foreground_annotation']
    return record


def build_index():
    _,source=load_task_config('meter')
    allocation=read_yaml(ROOT/source['full_design'])['tasks']['meter']['split_design']
    output=ROOT/'datasets/exports/meter';records=[];inputs=[];coverage={}
    for split in ('train','val','test'):
        manifest=output/split/'manifest.json';samples=output/split/'samples.jsonl'
        if read(manifest)['summary']['status']!='passed':raise ValueError('Split has not passed audit: '+split)
        inputs.extend([manifest,samples])
        axes=defaultdict(Counter);images=positive=0
        for line in samples.read_text(encoding='utf8').splitlines():
            row=json.loads(line)
            images+=1;positive+=row['instance_count']
            p=row['parameters']
            for key in ('branch','reading_sampling','face_material','saturation','exposure','light_relation','glare','view'):
                axes[key][str(p.get(key,'not_recorded'))]+=1
            axes['degradation'][p.get('degradation',{}).get('kind','not_recorded')]+=1
            axes['placement']['randomized' if 'placement_controls' in p else 'not_applicable']+=1
            record=source_record(row,split,source,allocation['direct_record_groups'])
            for field in ('review_evidence','source_review_evidence'):
                if not (ROOT/record[field]).is_file():raise FileNotFoundError(record[field])
            records.append(record)
        coverage[split]={'images':images,'instances':positive,'negative_images':images-positive,
                         'keypoints':positive*8,'axes':{k:dict(v) for k,v in axes.items()}}
    shared='datasets/assets/blender/aiaec_dataset.blend'
    exposures=defaultdict(set)
    for record in records:
        for relation in ('source_group','mother_scene','physical_assets','source_assets','parent_samples'):
            values=record.get(relation,[])
            for value in (values if isinstance(values,list) else [values]):
                if value:exposures[(relation,value)].add(record['split'])
    result={'model_key':'meter','selection_mode':'manifest_only','records':records,
            'source_reuse':[{'relation':relation,'source':value,'splits':sorted(splits),
                             'basis':'Shared geometric assets; images rendered separately for each split.'}
                            for (relation,value),splits in sorted(exposures.items()) if len(splits)>1],
            'shared_assets':[shared],'coverage':{'images':len(records),'keypoint_shape':[8,3],'splits':coverage},
            'scope':'Audited known synthetic geometry and native AI visual Pose. Current working dataset.'}
    configs=[Path(__file__).with_name('config')/name for name in ('dataset.yaml','source.yaml')]
    with generation_guard('meter.full.training_source_index',[],[output/'workspace_sources.json'],
            {'records':len(records)},root=ROOT,procedural_inputs=[*configs,*inputs,ROOT/source['full_design']],
            code_paths=[Path(__file__),Path(__file__).with_name('build_full_dataset.py')],
            replace_outputs=True,replace_verifier=lambda *_:True) as run:
        if not run.reused:
            write(output/'workspace_sources.json',result)
            run.set_result({'records':len(records)})
    return result


if __name__=='__main__':
    index=build_index();print('METER_TRAINING_INDEX',len(index['records']))
