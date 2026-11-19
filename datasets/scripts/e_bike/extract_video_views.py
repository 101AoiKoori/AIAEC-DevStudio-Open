"""Sample intermediate real orbit frames and extract source-recorded SAM2 cutouts."""
import argparse
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.images import write_image
from datasets.scripts.common.config import load_task_config
from lab.training_common.production_guard import generation_guard


def extract(output, count=12, device='0'):
    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'tmp/ultralytics'))
    from ultralytics import SAM
    _, source = load_task_config('e_bike')
    old = json.loads((ROOT / source['ground_vehicle_selection']).read_text(encoding='utf-8'))
    videos = {row['class']: row['source_video'] for row in old['vehicles']}
    weights = ROOT / 'models/weights/e_bike/sam2_t.pt'
    if not output.resolve().is_relative_to(ROOT / 'datasets/annotated/e_bike'):
        raise ValueError('output must be in annotated/e_bike')
    if not 2 <= count <= 48:
        raise ValueError('count must be 2..48 per posture')
    with generation_guard('e_bike.extract_video_views:' + output.relative_to(ROOT).as_posix(),
                          [ROOT / p for p in videos.values()] + [weights], [output],
                          {'count_per_posture': count, 'device': device,
                           'sampling': 'midpoints of equal time bins; actual decoded frames, no image morphing'},
                          code_paths=[Path(__file__)], procedural_inputs=[weights,
                              ROOT/'datasets/scripts/e_bike/config/dataset.yaml',
                              ROOT/'datasets/scripts/e_bike/config/source.yaml']) as run:
        if run.reused:
            return
        if output.exists() and any(output.iterdir()):
            raise ValueError('output must be empty')
        (output / 'frames').mkdir(parents=True, exist_ok=True)
        (output / 'cutouts').mkdir(exist_ok=True)
        model = SAM(str(weights))
        records = []
        for posture, video in sorted(videos.items()):
            cap = cv2.VideoCapture(str(ROOT / video))
            fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0 or total < count:
                raise ValueError('unreadable orbit: ' + video)
            try:
                for index in range(count):
                    number = round((index + .5) * (total - 1) / count)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, number)
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError('cannot decode frame ' + str(number))
                    h, w = frame.shape[:2]
                    small = cv2.resize(frame, (round(w * 1024 / max(h,w)), round(h * 1024 / max(h,w))))
                    sh, sw = small.shape[:2]
                    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
                    yellow = ((hsv[:,:,0] >= 12) & (hsv[:,:,0] <= 40) &
                              (hsv[:,:,1] > 110) & (hsv[:,:,2] > 100))
                    yellow[:round(sh*.25)] = False
                    yellow[round(sh*.75):] = False
                    yy,xx = np.nonzero(yellow)
                    if len(xx) < 50:
                        raise ValueError('yellow vehicle prompt is missing')
                    # The old whole-chair box can segment the chair instead of the vehicle.
                    # Fit a tighter prompt to this source's yellow body and include dark wheels/handlebars.
                    x1,x2 = np.percentile(xx,[1,99]); y1,y2 = np.percentile(yy,[1,99])
                    box = [max(0,x1-sw*.12), max(0,y1-sh*.07), min(sw,x2+sw*.12), min(sh,y2+sh*.045)]
                    point = [int(xx[len(xx)//2]), int(yy[len(yy)//2])]
                    result = model.predict(small, bboxes=[box], points=[point], labels=[1],
                                           device=device, imgsz=1024, verbose=False)[0]
                    if result.masks is None or len(result.masks.data) != 1:
                        raise ValueError('no unique mask')
                    mask = result.masks.data[0].cpu().numpy().astype(np.uint8)*255
                    mask = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
                    points = cv2.findNonZero(mask)
                    if points is None:
                        raise ValueError('empty mask')
                    x,y,bw,bh = cv2.boundingRect(points)
                    stem = f'{posture}_{number:06d}'
                    fp, mp = output/'frames'/(stem+'.png'), output/'cutouts'/(stem+'.png')
                    write_image(fp, small)
                    write_image(mp, np.dstack((small, mask))[y:y+bh,x:x+bw])
                    row = {'path': mp.relative_to(ROOT).as_posix(), 'class': posture,
                           'source_frame': fp.relative_to(ROOT).as_posix(), 'source_video': video,
                           'source_frame_index': number, 'timestamp_seconds': number/fps,
                           'source_image_size': [w,h], 'decoded_image_size': [sw,sh],
                           'mask_bbox_xyxy': [x,y,x+bw,y+bh],
                           'box_prompt_xyxy': box, 'positive_point_xy': point,
                           'view': 'orbit_intermediate_unclassified',
                           'contact_basis': 'bottom of actual alpha silhouette; approximate, not physical calibration',
                           'status': 'sam2_candidate_pending_visual_review'}
                    records.append(row)
                    data = {'status': 'extracting', 'vehicles': records,
                            'scope': 'Two existing parent videos; intermediate views are derivatives, not independent captures.'}
                    (output/'selection.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
                    print(json.dumps({'posture':posture,'frame':number,'extracted':len(records)}),flush=True)
            finally:
                cap.release()
        data['status'] = 'complete_candidates_pending_visual_review'
        (output/'selection.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--count', type=int, default=12)
    p.add_argument('--device', default='0')
    args = p.parse_args()
    extract((ROOT/args.output).resolve(), args.count, args.device)
