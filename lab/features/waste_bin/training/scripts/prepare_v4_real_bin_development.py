"""Prepare two visually reviewed site frames for bin-state development only."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[5]
sys.path[:0]=[str(ROOT),str(ROOT/'lab/training_common')]
from camera_tone import apply_tone,read_image,source_role,write_image  # noqa: E402
from lab.scripts.data.v4_source_gate import build_queue,site_sequence_times,SITE_VIDEO  # noqa: E402

REVIEW=ROOT/'datasets/annotated/waste_bin/v4_review_20261112/review_manifest.json'
NEGATIVE_REVIEW=ROOT/'runs/e_bike/v4_real_dev_source_contact.png'
OUT=ROOT/'datasets/exports/waste_bin/v4_real_bin_development_20261112'


def rel(path):return path.relative_to(ROOT).as_posix()


def _generate():
    if OUT.exists():raise ValueError('Output exists; use a new version')
    if not NEGATIVE_REVIEW.is_file():raise ValueError('Negative source review evidence missing')
    review=json.loads(REVIEW.read_text(encoding='utf8'))
    positive=next(x for x in review['images'] if x['id']=='frame_002311')
    task=positive['tasks']['waste_bin_state']
    if positive['split']!='unassigned' or task['state']!='candidate_complete_positive':
        raise ValueError('Positive source no longer approved as a candidate')
    rows=[{'source':ROOT/positive['image'],'boxes':task['boxes'],'label':ROOT/task['label'],
           'review_evidence':task['review_evidence'],'state':'positive'},
          {'source':ROOT/'datasets/annotated/site_capture_20261112/sequence_640_v1/full_scene/frame_002800.png',
           'boxes':[],'label':None,'review_evidence':rel(NEGATIVE_REVIEW),'state':'negative'}]
    queue=build_queue(ROOT);times=site_sequence_times(ROOT);records=[]
    (OUT/'images/val').mkdir(parents=True);(OUT/'labels/val').mkdir(parents=True)
    for row in rows:
        source=row['source'];role,reason=source_role(ROOT,source,queue)
        if role=='regression':raise ValueError('Source restricted to regression: '+str(reason))
        image=read_image(source)
        if image.shape[:2]!=(360,640):raise ValueError('Expected 640x360 frame')
        lines=row['label'].read_text(encoding='utf8').splitlines() if row['label'] else []
        if len(lines)!=len(row['boxes']):raise ValueError('Label count differs from reviewed boxes')
        for line,(class_id,box) in zip(lines,row['boxes']):
            cls,x,y,w,h=map(float,line.split());a,b,c,d=box;expected=((a+c)/1280,(b+d)/720,(c-a)/640,(d-b)/360)
            if int(cls)!=class_id or max(abs(i-j) for i,j in zip((x,y,w,h),expected))>1e-5:
                raise ValueError('Review/label geometry mismatch')
        for tone,value,saturation in (('original',1.,1.),('moderate',.7,.6)):
            stem=source.stem+'_'+tone;img=OUT/'images/val'/(stem+'.png');label=OUT/'labels/val'/(stem+'.txt')
            frame=image if tone=='original' else apply_tone(image,value_gain=value,saturation_gain=saturation)
            write_image(img,frame);label.write_text(('\n'.join(lines)+'\n') if lines else '',encoding='utf8')
            index=int(source.stem.split('_')[-1]);records.append({'image':rel(img),'label':rel(label),
                'split':'val_development','parent_image':rel(source),'source_video':SITE_VIDEO,
                'source_time_seconds':times[index],'state':row['state'],'candidate_boxes':row['boxes'],
                'generation_method':'provided_video_frame' if tone=='original' else 'real_image_tone_augmentation',
                'tone_profile':tone,'value_gain':value,'saturation_gain':saturation,'review_evidence':row['review_evidence']})
    manifest={'status':'development_candidates_not_deliverable','parent_frames':len(rows),'images':len(records),
              'scope':'Two distinct frames from one site video; one positive and one negative parent, with tone variants. Not an independent capture or frozen test.',
              'records':records}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'output':rel(OUT),'images':len(records),'parent_frames':len(rows)},ensure_ascii=False))


def _guard_inputs():
    from lab.scripts.data.v4_source_gate import queue_inputs, SITE_SEQUENCE
    media, procedural, code = queue_inputs(ROOT)
    review=json.loads(REVIEW.read_text(encoding='utf8'))
    positive=next(x for x in review['images'] if x['id']=='frame_002311')
    task=positive['tasks']['waste_bin_state']
    media=set(media) | {ROOT/positive['image'], ROOT/SITE_SEQUENCE/'frame_002800.png', NEGATIVE_REVIEW}
    if task.get('review_evidence'):
        evidence=ROOT/task['review_evidence']
        if evidence.suffix.lower() in ('.png','.jpg','.jpeg','.webp'):
            media.add(evidence)
        else:
            procedural.append(evidence)
    procedural=set(procedural) | {REVIEW, ROOT/task['label'], ROOT/SITE_SEQUENCE/'ffmpeg.log'}
    return sorted(media), sorted(procedural), code


def main():
    if '--help' in sys.argv[1:] or '-h' in sys.argv[1:]:
        argparse.ArgumentParser(description=__doc__).parse_args(['--help'])
    from lab.training_common.production_guard import generation_guard
    media,procedural,code=_guard_inputs()
    code=list(code)+[Path(__file__).resolve(),ROOT/'lab/training_common/camera_tone.py']
    with generation_guard('waste_bin.prepare_real_development:'+str(OUT.relative_to(ROOT)),
                          media,[OUT],{'profiles':['original','moderate']},code_paths=code,
                          procedural_inputs=procedural,root=ROOT) as run:
        if run.reused:
            return
        _generate()


if __name__=='__main__':main()
