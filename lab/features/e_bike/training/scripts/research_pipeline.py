"""Inspect one image using the hash-pinned e-bike research candidate."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import torch

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from datasets.scripts.common.images import read_image,write_image
from lab.training_common.engine.runtime import setup_paths,prepare_ultralytics
from lab.features.e_bike.training.scripts.benchmark_candidate_pipeline import infer_primary


def load_models(config_path,output):
    config=json.loads(config_path.read_text(encoding='utf-8'))
    if config['pipeline']!='semantic_joint_640_c025_i030':raise ValueError('Unsupported research pipeline')
    if not torch.cuda.is_available():raise RuntimeError('This research implementation requires CUDA; CPU inference not validated')
    for role in ('detector','classifier'):
        item=config[role];path=ROOT/item['path']
        if not path.is_file():raise FileNotFoundError(path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError(f'{role} weight hash differs from selected research candidate')
    setup_paths(output);YOLO=prepare_ultralytics(output)
    detector=YOLO(str(ROOT/config['detector']['path']));classifier=YOLO(str(ROOT/config['classifier']['path']))
    if len(detector.names)!=1 or set(classifier.names.values())!={'background','fallen','upright'}:raise ValueError('Unexpected model role/class mapping')
    return detector,classifier,config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='datasets/scripts/e_bike/config/research_pipeline.json')
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=(ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT/'runs/e_bike'):raise ValueError('Research outputs must stay inside runs/e_bike')
    source=Path(args.source).resolve();frame=read_image(source);out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);torch.set_num_threads(2)
    detector,classifier,config=load_models((ROOT/args.config).resolve(),out)
    torch.cuda.synchronize();started=time.perf_counter();found=infer_primary(detector,classifier,frame);torch.cuda.synchronize()
    elapsed=(time.perf_counter()-started)*1000
    counts=Counter(p['class_id'] for p in found);marked=frame.copy()
    display_order=sorted(range(len(found)),key=lambda i:((found[i]['xyxy'][0]+found[i]['xyxy'][2])/2,(found[i]['xyxy'][1]+found[i]['xyxy'][3])/2))
    for index,i in enumerate(display_order,1):found[i]['display_id']=index
    for p in found:
        x1,y1,x2,y2=map(round,p['xyxy']);color=(40,220,40) if p['class_id']==0 else (0,160,255)
        cv2.rectangle(marked,(x1,y1),(x2,y2),color,1);label='upright' if p['class_id']==0 else 'fallen'
        cv2.putText(marked,f'{p["display_id"]} {label}',(max(0,x1),max(12,y1-3)),0,.4,color,1)
    result={'status':'research_inference_complete','source':str(source),'predicted_counts':{'total':len(found),'upright':counts[0],'fallen':counts[1]},
            'instances':found,'compute_ms_including_first_call_warmup':elapsed,
            'model_config':config,'scope':config['scope'],'note':'Prediction only; not human-confirmed counts or task-region coverage.'}
    write_image(out/'annotated.jpg',marked);(out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(out),'counts':result['predicted_counts']},ensure_ascii=False))


if __name__=='__main__':main()
