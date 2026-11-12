"""Extract candidate person masks from separate real video/photo source groups.

The four samples are SAM2 candidates and need visual review; this tool does
not train, validate a model, or infer independent recording sessions.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM

ROOT = Path(__file__).resolve().parents[5]
VIDEO = ROOT / 'Archive/Assets/电动车与人物/视频/人物立牌正背面_电动车直立_倒伏和遮挡_场地扫描_012.mp4'
CAMERA_PERSON = ROOT / 'Archive/Assets/电动车与人物/照片/小车摄像头实拍_近似平视_人物和直立电动车_004.ppm'
CAMERA_SCENE = ROOT / 'Archive/Assets/红绿灯/照片/小车摄像头实拍_近似平视_信号灯正面与道路_001.ppm'
WEIGHTS = ROOT / 'models/weights/e_bike/sam2_t.pt'
PROMPTS = [('val', 0, 72.0, [620, 240, 875, 765]),
           ('val', 1, 40.5, [260, 120, 545, 735]),
           ('test', 0, None, [340, 265, 415, 423]),
           ('test', 1, None, [70, 270, 138, 425])]


def read_photo(path):
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('unreadable source: ' + str(path))
    return image


def read_video(seconds):
    cap = cv2.VideoCapture(str(VIDEO))
    try:
        if not cap.isOpened():
            raise ValueError('cannot open source video')
        cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, image = cap.read()
        if not ok:
            raise ValueError('cannot decode video at ' + str(seconds))
        return image, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    finally:
        cap.release()


def write_image(path, image):
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError('failed to encode ' + str(path))
    encoded.tofile(str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', '0'), default='cpu')
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to((ROOT / 'datasets/annotated/person').resolve()) or output.exists():
        parser.error('output must be a new directory in datasets/annotated/person')
    for source in (VIDEO, CAMERA_PERSON, CAMERA_SCENE, WEIGHTS):
        if not source.is_file():
            parser.error('missing source: ' + str(source))
    output.mkdir(parents=True)
    manifest = {'status': 'running', 'records': [], 'backgrounds': {},
                'limits': 'SAM2 masks are AI review candidates; material identity and recording sessions may overlap training'}
    def save():
        (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    save()
    try:
        model = SAM(str(WEIGHTS))
        for split, cls, seconds, bbox in PROMPTS:
            if seconds is None:
                image, frame = read_photo(CAMERA_PERSON), None
                source = CAMERA_PERSON
            else:
                image, frame = read_video(seconds)
                source = VIDEO
            result = model.predict(image, bboxes=[bbox], device=args.device, imgsz=1024, verbose=False)[0]
            if result.masks is None or len(result.masks.data) != 1:
                raise ValueError('missing unique segmentation')
            mask = result.masks.data[0].cpu().numpy().astype(np.uint8) * 255
            h, w = image.shape[:2]
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            points = cv2.findNonZero(mask)
            if points is None:
                raise ValueError('empty segmentation')
            x, y, bw, bh = cv2.boundingRect(points)
            path = output / f'{split}_person_{cls}.png'
            write_image(path, np.dstack((image, mask))[y:y+bh, x:x+bw])
            manifest['records'].append({'split': split, 'class_id': cls,
                'sprite': path.relative_to(ROOT).as_posix(), 'source': source.relative_to(ROOT).as_posix(),
                'frame': frame, 'requested_seconds': seconds, 'prompt_xyxy': bbox,
                'mask_status': 'SAM2_AI_review_candidate', 'source_box_xywh': [x, y, bw, bh]})
            save()
        image, frame = read_video(4.5)
        if image.shape[:2] != (1080, 1920):
            raise ValueError('unexpected video resolution')
        bg = output / 'val_ground.png'
        write_image(bg, image[540:1080, 360:1440])
        camera = read_photo(CAMERA_SCENE)
        if camera.shape[:2] != (480, 640):
            raise ValueError('unexpected camera photo resolution')
        bg_test = output / 'test_scene.png'
        write_image(bg_test, camera[160:480, 0:640])
        manifest['backgrounds'] = {'val': {'path': bg.relative_to(ROOT).as_posix(),
            'source': VIDEO.relative_to(ROOT).as_posix(), 'frame': frame, 'crop_xyxy': [360, 540, 1440, 1080]},
            'test': {'path': bg_test.relative_to(ROOT).as_posix(),
            'source': CAMERA_SCENE.relative_to(ROOT).as_posix(), 'crop_xyxy': [0, 160, 640, 480]}}
        manifest['status'] = 'complete_candidates'
    except BaseException as exc:
        manifest['status'] = 'failed_or_interrupted'
        manifest['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save()
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == '__main__':
    main()
