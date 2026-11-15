"""Prepare reviewed complete site frames as small e-bike development candidates."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[5]
sys.path[:0]=[str(ROOT),str(ROOT/'lab/training_common')]
from camera_tone import apply_tone,read_image,source_role,write_image  # noqa: E402
from lab.scripts.data.v4_source_gate import build_queue  # noqa: E402
from lab.training_common.production_guard import generation_guard  # noqa: E402
from e_bike_queue_inputs import queue_inputs  # noqa: E402

REVIEW=ROOT/'datasets/annotated/e_bike/v4_vehicle_review_20261112_batch2/review_manifest.json'
OUT=ROOT/'datasets/exports/e_bike/v4_real_development_candidates_20261112'
SELECTED=('frame_002800','frame_008000')
TRAIN_MANIFEST=ROOT/'datasets/exports/e_bike/v4_grounded_training_candidates_20261112/manifest.json'


def rel(path):
    return path.relative_to(ROOT).as_posix()


def main():
    rows = [r for r in json.loads(REVIEW.read_text(encoding='utf-8'))['samples']
            if Path(r['image']).stem in SELECTED]
    checked = [ROOT / row['image'] for row in rows]
    media = checked + [ROOT / row['source_video'] for row in rows]
    procedural = [REVIEW, TRAIN_MANIFEST] + [ROOT / row['label'] for row in rows]
    queue = build_queue(ROOT)
    extra_media, extra_procedural = queue_inputs(ROOT, queue, checked)
    media = list(dict.fromkeys(media + extra_media))
    procedural = list(dict.fromkeys(procedural + extra_procedural))
    with generation_guard('e_bike.real_development:' + str(OUT), media, [OUT], {'selected': SELECTED},
                          procedural_inputs=procedural,
                          code_paths=[__file__, ROOT/'lab/training_common/camera_tone.py',
                                      ROOT/'lab/scripts/data/v4_source_gate.py', ROOT/'lab/training_common/train.py',
                                      ROOT/'lab/training_common/engine/data.py', ROOT/'lab/training_common/engine/config.py',
                                      Path(__file__).with_name('e_bike_queue_inputs.py')]) as run:
        if run.reused:
            return
        return _main()


def _main():
    if OUT.exists():raise ValueError('Output exists; use a new version')
    rows=json.loads(REVIEW.read_text(encoding='utf8'))['samples']; rows=[r for r in rows if Path(r['image']).stem in SELECTED]
    if len(rows)!=len(SELECTED):raise ValueError('Missing reviewed candidate frames')
    queue=build_queue(ROOT); records=[]
    if not TRAIN_MANIFEST.is_file():raise ValueError('Grounded training source manifest required for parent isolation')
    training=json.loads(TRAIN_MANIFEST.read_text(encoding='utf8'))['records']
    train_parents={r['background']['parent_image'] for r in training}
    (OUT/'images/val').mkdir(parents=True);(OUT/'labels/val').mkdir(parents=True)
    for row in rows:
        source=ROOT/row['image'];role,reason=source_role(ROOT,source,queue)
        if rel(source) in train_parents:raise ValueError('Development parent occurs in training: '+rel(source))
        if role=='regression' or row['split']!='unassigned' or row['state'] not in ('complete_positive','complete_negative'):
            raise ValueError('Frame cannot enter development set: '+str(reason))
        image=read_image(source)
        if image.shape[:2]!=(360,640):raise ValueError('Expected 640x360 frame')
        lines=(ROOT/row['label']).read_text(encoding='utf8').splitlines()
        if len(lines)!=len(row['candidate_boxes']):raise ValueError('Label count differs from reviewed boxes')
        for line,box in zip(lines,row['candidate_boxes']):
            cls,x,y,w,h=map(float,line.split());a,b,c,d=box['xyxy']
            expected=((a+c)/1280,(b+d)/720,(c-a)/640,(d-b)/360)
            if int(cls)!=box['class_id'] or max(abs(i-j) for i,j in zip((x,y,w,h),expected))>1e-5:
                raise ValueError('Label differs from reviewed geometry')
        for tone,value,sat in (('original',1.,1.),('moderate',.7,.6)):
            name=source.stem+'_'+tone; img=OUT/'images/val'/(name+'.png');label=OUT/'labels/val'/(name+'.txt')
            output=image if tone=='original' else apply_tone(image,value_gain=value,saturation_gain=sat)
            write_image(img,output);label.write_text(('\n'.join(lines)+'\n') if lines else '',encoding='utf8')
            records.append({'image':rel(img),'label':rel(label),'split':'val_development','parent_image':rel(source),
                'source_video':row['source_video'],'source_time_seconds':row['source_time_seconds'],
                'segment_id':row['segment_id'],'state':row['state'],'candidate_boxes':row['candidate_boxes'],
                'generation_method':'provided_video_frame' if tone=='original' else 'real_image_tone_augmentation',
                'tone_profile':tone,'value_gain':value,'saturation_gain':sat,'review_evidence':rel(REVIEW)})
    manifest={'status':'development_candidates_not_deliverable','parent_frames':len(rows),'images':len(records),
              'scope':'One site video, two previously reviewed frames and tone variants; no fallen-class positive and no independent collection claim.',
              'records':records}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'output':rel(OUT),'images':len(records),'parent_frames':len(rows)},ensure_ascii=False))


if __name__=='__main__':main()
