"""Compare a selected new-video checkpoint, with old Test used only for regression."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='0')
    parser.add_argument('--confidence', type=float, default=.25)
    args = parser.parse_args()
    if not 0 < args.confidence < 1:
        parser.error('--confidence must be between 0 and 1')
    package = args.package.resolve()
    output = (package / args.output).resolve()
    if not output.is_relative_to(package / 'runs'):
        raise ValueError('Comparison output must stay in package runs')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'cache').mkdir()
    os.environ.update(YOLO_CONFIG_DIR=str(output / 'cache'), YOLO_AUTOINSTALL='false', YOLO_OFFLINE='true')
    sys.path.insert(0, str(package / 'scripts/common'))
    from engine.reporting import match_detections
    from ultralytics import YOLO
    from PIL import Image, ImageDraw, ImageFont
    if args.device == 'cpu':
        import torch
        torch.set_num_threads(2)

    manifest = json.loads((package / 'finetune_dataset/manifest.json').read_text(encoding='utf8'))
    new = manifest['records']
    groups = {
        'new_val_same_scene': [package / 'finetune_dataset' / r['image'] for r in new if r['split'] == 'val'],
        'new_train_raw_replay': [package / 'finetune_dataset' / r['image'] for r in new
                                 if r['split'] == 'train' and r['branch'] == 'video_frame'],
        'old_test_regression_only': sorted(p for p in (package / 'dataset/images/test').iterdir()
                                            if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}),
    }
    if not groups['old_test_regression_only']:
        raise FileNotFoundError('Original Test missing; never substitute old Train')
    models = {'before': package / 'building_name_paper_best_Val_Test/best.pt',
              'after': (package / args.candidate).resolve()}
    report = {'protocol': {'imgsz': 640, 'conf': args.confidence, 'nms_iou': .7, 'match_iou': .5, 'max_det': 100,
                           'batch': 1, 'rect': True, 'half': False, 'device': args.device},
              'original_training_images': 0,
              'scope': 'Candidate already selected by new Val; old Test is post-selection regression only. '
                       'New Val shares scene/session; B paper is present only in training replay. '
                       'Per-image detection counts do not measure cross-frame unique counting.',
              'weights': {k: {'path': str(p.relative_to(package)), 'sha256': sha256(p)} for k, p in models.items()},
              'groups': {}}
    all_rows = {}
    for model_name, weights in models.items():
        model = YOLO(str(weights))
        for group, paths in groups.items():
            total = dict(images=0, tp=0, fp=0, fn=0, exact=0, count_correct=0,
                         overcount_images=0, duplicate_overlap_suspects=0)
            rows = {}
            for result in model.predict(source=[str(p) for p in paths], stream=True, batch=1, imgsz=640,
                                        rect=True, half=False, conf=args.confidence, iou=.7, max_det=100,
                                        device=args.device, verbose=False, save=False):
                path = Path(result.path); height, width = result.orig_shape
                label = path.parent.parent.parent / 'labels' / path.parent.name / path.with_suffix('.txt').name
                truth = []
                for line in label.read_text().splitlines():
                    cls, x, y, w, h = map(float, line.split())
                    truth.append({'class_id': int(cls), 'xyxy': [(x-w/2)*width, (y-h/2)*height,
                                                                 (x+w/2)*width, (y+h/2)*height]})
                predictions = [{'class_id': int(b.cls.item()), 'confidence': float(b.conf.item()),
                                'xyxy': b.xyxy[0].tolist()} for b in result.boxes]
                matched, missed, extra = match_detections(truth, predictions, .5)
                total['images'] += 1; total['tp'] += len(matched); total['fp'] += len(extra); total['fn'] += len(missed)
                total['exact'] += int(not missed and not extra)
                total['count_correct'] += int(len(truth) == len(predictions))
                total['overcount_images'] += int(len(predictions) > len(truth))
                for i in extra:
                    a = predictions[i]['xyxy']
                    for item in truth:
                        b = item['xyxy']
                        intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
                        smaller = min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1]))
                        if smaller and intersection/smaller >= .8:
                            total['duplicate_overlap_suspects'] += 1
                            break
                rows[path.name] = {'image': path.relative_to(package).as_posix(), 'truth': truth,
                                  'predictions': predictions, 'missed': missed, 'extra': extra}
            report['groups'].setdefault(group, {})[model_name] = total
            all_rows[(model_name, group)] = rows
            print(group, model_name, json.dumps(total), flush=True)
    (output / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    details = {f'{model}/{group}': rows for (model, group), rows in all_rows.items()}
    (output / 'predictions.json').write_text(json.dumps(details, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    choices = [('new_val_same_scene', f'buildingname_f{n:06d}.jpg') for n in (30, 72, 186, 198)]
    choices += [('new_train_raw_replay', 'buildingname2_f000020.jpg')]
    failures = [name for name, r in all_rows[('before', 'old_test_regression_only')].items() if r['extra']]
    choices += [('old_test_regression_only', name) for name in failures[:3]]
    sheet = Image.new('RGB', (1280, len(choices)*390), (24, 27, 31))
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
    for index, (group, name) in enumerate(choices):
        for col, model_name in enumerate(('before', 'after')):
            row = all_rows[(model_name, group)][name]
            tile = Image.open(package / row['image']).convert('RGB'); width, height = tile.size
            tile.thumbnail((638, 354)); draw = ImageDraw.Draw(tile)
            for item in row['truth']:
                a,b,c,d = item['xyxy']; draw.rectangle((a*tile.width/width,b*tile.height/height,c*tile.width/width,d*tile.height/height),outline='#20ed74',width=2)
            for i,item in enumerate(row['predictions']):
                a,b,c,d = item['xyxy']; color = '#ff5858' if i in row['extra'] else '#28baff'
                x,y = a*tile.width/width,b*tile.height/height
                draw.rectangle((x,y,c*tile.width/width,d*tile.height/height),outline=color,width=2)
                draw.text((x,y),f'{item["confidence"]:.2f}',font=font,fill=color)
            x,y = col*640,index*390
            sheet.paste(tile,(x,y+34))
            ImageDraw.Draw(sheet).text((x+3,y+2),f'{model_name} {name} FN={len(row["missed"])} FP={len(row["extra"])}',font=font,fill='white')
    sheet.save(output / 'comparison.jpg', quality=93)


if __name__ == '__main__':
    main()
