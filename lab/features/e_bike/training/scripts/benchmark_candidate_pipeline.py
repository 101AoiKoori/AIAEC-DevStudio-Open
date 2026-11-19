"""Primary-threshold end-to-end compute timing with frozen prediction checks."""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from datasets.scripts.common.images import read_image
from lab.training_common.engine.runtime import setup_paths,prepare_ultralytics
from lab.features.e_bike.training.scripts.input_scale_experiment import make_predictor
from lab.features.e_bike.training.scripts.two_stage_experiment import crop
from lab.features.e_bike.training.scripts.candidate_verifier_experiment import finish_pool


def infer_primary(detector,model,frame,original=False):
    """Research primary pipeline, shared by timing and per-image inspection."""
    from ultralytics.utils import ops
    if detector.predictor is not None and not hasattr(detector.predictor,'decoded_before_nms'):
        detector.predictor=None
    detector.predict(source=frame,imgsz=640,rect=True,conf=.25,iou=.3 if original else .95,
                     max_det=100 if original else 300,device='0',half=False,verbose=False,predictor=make_predictor())
    predictor=detector.predictor;ids=predictor.kept_raw_ids;boxes=predictor.raw_boxes
    input_boxes=ops.xywh2xyxy(predictor.decoded_before_nms[0,:4].T).numpy();found=[]
    for start in range(0,len(ids),32):
        chunk=[i for i in ids[start:start+32] if boxes[i][2]>boxes[i][0] and boxes[i][3]>boxes[i][1]]
        if not chunk:continue
        probabilities=model.predict(source=[crop(frame,boxes[i])[0] for i in chunk],imgsz=160,device='0',verbose=False)
        for index,result in zip(chunk,probabilities):
            score=float(predictor.decoded_before_nms[0,4,index])
            item={'raw_id':index,'xyxy':boxes[index].tolist(),'input_xyxy':input_boxes[index].tolist(),'raw_confidence':score}
            if original:item['class_id']=('upright','fallen').index(model.names[result.probs.top1])
            else:item['verifier_probs']={model.names[k]:float(v) for k,v in enumerate(result.probs.data.cpu().tolist())}
            found.append(item)
    return found if original else finish_pool(found,.25,.3,'joint')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='runs/e_bike/candidate_latency_20261118')
    args=parser.parse_args();out=(ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT/'runs/e_bike'):raise ValueError('Expected research output')
    out.mkdir(parents=True,exist_ok=False);setup_paths(out);cv2.setNumThreads(1);torch.set_num_threads(2)
    YOLO=prepare_ultralytics(out)
    from ultralytics.utils import ops
    detector=YOLO(str(ROOT/'runs/e_bike/keypoint_control_20261118/training/weights/best.pt'))
    configs={
        'original':('runs/e_bike/two_stage_20261118/classifier/weights/best.pt','runs/e_bike/candidate_verifier_20261118/evaluation/c025_i030_raw/predictions.jsonl'),
        'semantic_verifier':('runs/e_bike/candidate_verifier_20261118/classifier/weights/last.pt','runs/e_bike/candidate_verifier_20261118/evaluation/c025_i030_joint/predictions.jsonl'),
        'completeness':('runs/e_bike/candidate_completeness_20261118/classifier/weights/last.pt','runs/e_bike/candidate_completeness_20261118/evaluation/c025_i030_joint/predictions.jsonl'),
        'matched_control':('runs/e_bike/candidate_completeness_control_20261118/classifier/weights/last.pt','runs/e_bike/candidate_completeness_control_20261118/evaluation/c025_i030_joint/predictions.jsonl')}
    classifiers={name:YOLO(str(ROOT/p)) for name,(p,_) in configs.items()}
    refs={name:{r['source_image']:r for r in (json.loads(line) for line in (ROOT/p).read_text(encoding='utf-8').splitlines())} for name,(_,p) in configs.items()}
    images=list(refs['original']);frames={p:read_image(ROOT/p) for p in images}

    def infer(name,frame):
        return infer_primary(detector,classifiers[name],frame,name=='original')

    # Each model sees identical warmup inputs; no disk reads occur in timed calls.
    for name in configs:
        for path in (images[0],images[len(images)//2],images[-1]):infer(name,frames[path])
    records=[];mismatches=[]
    for repeat in range(2):
        for index,path in enumerate(images):
            names=list(configs)
            if (repeat+index)%2:names.reverse()
            for name in names:
                torch.cuda.synchronize();started=time.perf_counter();found=infer(name,frames[path]);torch.cuda.synchronize()
                elapsed=(time.perf_counter()-started)*1000
                expected=refs[name][path]['predictions']
                equivalent=len(found)==len(expected) and all(a['class_id']==b['class_id'] and np.allclose(a['xyxy'],b['xyxy'],atol=1e-4,rtol=0) for a,b in zip(found,expected))
                if not equivalent:mismatches.append({'mode':name,'image':path,'repeat':repeat})
                records.append({'mode':name,'image':path,'repeat':repeat,'milliseconds':elapsed,'predictions':len(found),'reference_equivalent':equivalent})
    summary={'scope':'Windows offline CUDA compute; no ROS/target machine claim. Includes preprocessing, detector, crop classification and NMS; excludes load and disk IO.',
             'gpu':torch.cuda.get_device_name(),'warmup_images_per_mode':3,'passes':2,'images':len(images),
             'optimization':'Primary confidence .25 applied before crop classification; equivalent predictions must be checked against broad .01 diagnostic replay.',
             'reference_mismatches':mismatches,
             'timing':{name:{'median_ms':float(np.median([r['milliseconds'] for r in records if r['mode']==name])),
                             'p95_ms':float(np.percentile([r['milliseconds'] for r in records if r['mode']==name],95)),
                             'mean_ms':float(np.mean([r['milliseconds'] for r in records if r['mode']==name]))} for name in configs}}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
