"""Copy complete current samples into parent-grouped baseline train/val inputs."""
import json
from pathlib import Path
from collections import Counter
import shutil
import sys

from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from datasets.scripts.common.config import load_task_config
from datasets.scripts.common.package_direct_real import check_yolo
from datasets.scripts.common.frame_environment import query_frames
from datasets.scripts.common import recoverable_publish as publisher
from lab.training_common.production_guard import generation_guard, receipt_completed


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def relative(path):return Path(path).resolve().relative_to(ROOT).as_posix()
def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def selected_samples(manifest,policy):
    val=set(policy['validation_parent_groups']);selected=[]
    for row in manifest['records']:
        if row.get('unresolved_regions') or row.get('label_scope')=='known_postures_only_not_training':
            continue
        group=row.get('source_group') or row['background']['parent_group']
        split='val' if group in val else 'train'
        if row['source_split'] in ('train','val','test') and row['source_split']!=split:
            raise ValueError('Current source assignment disagrees with baseline group: '+row['state_id'])
        image,label=ROOT/row['image'],ROOT/row['label']
        text=label.read_text(encoding='utf-8');check_yolo(text,2,0)
        if len(text.splitlines())!=len(row['objects']):raise ValueError('Incomplete sample labels')
        selected.append((row,split,group))
    groups={s:{g for _,ss,g in selected if ss==s} for s in ('train','val')}
    if groups['train'] & groups['val']:raise ValueError('Baseline parent groups overlap')
    if any(not any(ss==s for _,ss,_ in selected) for s in ('train','val')):raise ValueError('Empty baseline split')
    return selected


def main():
    _,config=load_task_config('e_bike')
    manifest_path=ROOT/config['sample_manifest'];manifest=read(manifest_path)
    policy=read(ROOT/config['baseline_data']);selected=selected_samples(manifest,policy)
    source_kinds={row['frame_environment']['record_id']:query_frames(
        task='e_bike',role='direct_target',record_id=row['frame_environment']['record_id'],status=None)[0]['source_kind']
        for row,_,_ in selected if row['branch']=='direct_real'}
    output=ROOT/'datasets/exports/e_bike';operation='e_bike.prepare_baseline_data'
    media=sorted({ROOT/row['image'] for row,_,_ in selected})
    metadata={manifest_path,ROOT/manifest['generation_plan'],ROOT/config['baseline_data'],
              ROOT/'datasets/scripts/e_bike/config/source.yaml',ROOT/'datasets/scripts/e_bike/config/dataset.yaml',
              ROOT/config['databases']['frame_environment'],
              *(ROOT/row['label'] for row,_,_ in selected)}
    codes=[Path(__file__),Path(publisher.__file__),ROOT/'datasets/scripts/common/package_direct_real.py']
    with publisher.sample_lock(output):
        committed=lambda token:receipt_completed(ROOT,operation,token)
        publisher.recover(output,committed)
        owned={'data.yaml','manifest.json','workspace_sources.json','source_records/sample_manifest.json','source_records/generation_plan.json'}
        if (output/'manifest.json').is_file():
            old=read(output/'manifest.json')
            if old.get('producer')!='e_bike.prepare_baseline_data':raise ValueError('Export ownership differs')
            for row in old['records']:
                owned.update((Path(row[k]).relative_to(output.relative_to(ROOT)).as_posix() for k in ('image','label')))
            owned.update(f'{s}/manifest.json' for s in ('train','val'))
        preserved=[relative(p) for p in output.rglob('*') if p.is_file() and p.relative_to(output).as_posix() not in owned]
        try:
            with generation_guard(operation,media,[output],{'policy':policy},code_paths=codes,procedural_inputs=sorted(metadata),
                                  replace_outputs=True,replace_verifier=lambda *_:True,root=ROOT) as run:
                if run.reused:
                    print(read(output/'manifest.json')['summary']);return
                stage=publisher.prepare(output,run.token);publisher.copy_unowned(output,stage,owned)
                (stage/'source_records').mkdir(exist_ok=True)
                write(stage/'source_records/sample_manifest.json',manifest)
                write(stage/'source_records/generation_plan.json',read(ROOT/manifest['generation_plan']))
                evidence=relative(output/'source_records/sample_manifest.json');records=[]
                for row,split,group in selected:
                    for directory in ('images','labels'):(stage/split/directory).mkdir(exist_ok=True,parents=True)
                    image=stage/split/'images'/Path(row['image']).name;label=stage/split/'labels'/Path(row['label']).name
                    if image.exists() or label.exists():raise ValueError('Export collides with unowned file')
                    shutil.copyfile(ROOT/row['image'],image);shutil.copyfile(ROOT/row['label'],label)
                    direct=row['branch']=='direct_real';mirrored=direct and row.get('mirrored',False)
                    photo=direct and 'photo' in source_kinds[row['frame_environment']['record_id']]
                    method='augmentation' if mirrored else ('camera_capture' if photo else 'provided_video_frame') if direct else 'composite'
                    kind='derived_image' if mirrored or not direct else 'photo' if photo else 'video_frame'
                    processing={'origin_review':'confirmed'}
                    if direct:
                        processing.update(parent_origin='direct_camera',source_size=row['image_size'],output_size=row['image_size'],
                                          resolution_review='confirmed',resampling='none',clarity='original',
                                          resolution_basis='unchanged decoded review-frame pixels')
                    else:processing['background_origin']='direct_camera' if row['branch']=='real_background_composite' else 'blender_scene'
                    records.append({'image':relative(output/split/'images'/image.name),'label':relative(output/split/'labels'/label.name),
                        'split':split,'workspace_image':row['image'],'workspace_label':row['label'],'source_group':group,
                        'source_id':group,'source_kind':kind,'generation_method':method,'image_processing':processing,
                        'label_status':'reviewed','review_state':'complete_positive' if row['objects'] else 'complete_negative',
                        'source_review_evidence':evidence,'review_evidence':evidence,
                        'parent_sample':row.get('source_image') or row['background']['image'],
                        'parent_video':group,'source_split':row['source_split'],'source_record':row,
                        'development_contact':['known_scene_annotation_review']})
                summary={'images':len(records),'by_split':dict(Counter(r['split'] for r in records)),
                         'boxes_by_split':{s:dict(Counter(o['class'] for r in records if r['split']==s for o in r['source_record']['objects'])) for s in ('train','val')},
                         'scope':'First bounded baseline, known physical assets and scenes. Parent paths separated; no independent new-scene accuracy claim.'}
                for split in ('train','val'):write(stage/split/'manifest.json',{'status':'current_baseline_input','records':[r for r in records if r['split']==split]})
                write(stage/'data.yaml',{'names':['upright','fallen'],'train':'train/images','val':'val/images'})
                write(stage/'workspace_sources.json',{'model_key':'e_bike','selection_mode':'manifest_only','records':records,'source_reuse':[]})
                write(stage/'manifest.json',{'producer':operation,'status':'development_baseline_not_formal_release','summary':summary,'records':records})
                publisher.publish(output,stage,run.token,keep_root=True);run.set_result(summary)
            publisher.recover(output,committed)
        except BaseException:
            publisher.recover(output,committed);raise
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
