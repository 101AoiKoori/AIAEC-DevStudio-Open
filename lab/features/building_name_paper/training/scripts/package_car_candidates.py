"""Collect existing car-camera name-paper annotations as unapproved candidates."""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/building_name_paper/.")

import argparse
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard  # noqa: E402

MANIFESTS = (
    ROOT / 'datasets/annotated/car_real_capture_20261112/field_eval_v1/annotations_root.json',
    ROOT / 'datasets/annotated/car_real_capture_20261112/field_eval_v1/annotations_frames.json',
    ROOT / 'datasets/annotated/fire/building_name_paper_v4_car_sequence_review_20261113/manifest.json',
)
SITE_MANIFESTS = (
    ROOT / 'datasets/annotated/site_capture_20261112/field_eval_v1/photo_labels.json',
    ROOT / 'datasets/annotated/site_capture_20261112/field_eval_v1/video_labels.json',
)


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def collect(manifests=MANIFESTS):
    result, seen = [], set()
    for manifest in manifests:
        data = json.loads(manifest.read_text(encoding='utf8'))
        records = data if isinstance(data, list) else data['records']
        for index, row in enumerate(records):
            if isinstance(data, list):
                if (row.get('model_key') != 'building_name_paper' or
                        row.get('review_status') not in ('visual_candidate', 'visual_negative')):
                    continue
                boxes = [item['xyxy'] for item in row['boxes'] if item.get('class_id') == 0]
                if len(boxes) != len(row['boxes']):
                    raise ValueError('unsupported class in source annotation')
            else:
                if not row.get('label') or row.get('box_xyxy') is None:
                    continue
                boxes = [] if row['box_xyxy'] == [] else [row['box_xyxy']]
            image = (ROOT / row['image']).resolve()
            if not image.is_relative_to(ROOT) or not image.is_file() or image in seen:
                raise ValueError('missing, repeated or out-of-workspace source image')
            if 'image_size' in row:
                width, height = row['image_size']
            else:
                with Image.open(image) as source_image:
                    width, height = source_image.size
            lines = []
            for x1, y1, x2, y2 in boxes:
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError('source box outside image dimensions')
                lines.append(f'0 {(x1+x2)/(2*width):.6f} {(y1+y2)/(2*height):.6f} '
                             f'{(x2-x1)/width:.6f} {(y2-y1)/height:.6f}')
            seen.add(image)
            result.append((manifest, index, row, image, lines))
    if not result:
        raise ValueError('no car-camera candidates')
    return result


def package(output, source_set='car'):
    output = Path(output).resolve()
    allowed = (ROOT / 'runs/building_name_paper').resolve()
    if not output.is_relative_to(allowed):
        raise ValueError('output must be within the building-name task mainline')
    manifests = {'car': MANIFESTS, 'site': SITE_MANIFESTS}[source_set]
    selected = collect(manifests)
    images = [item[3] for item in selected]
    with generation_guard('building_name.package_car_candidates:' + relative(output),
                          images + list(manifests), [output], {'selection': source_set + '_camera_candidates'},
                          code_paths=[Path(__file__)], procedural_inputs=manifests, root=ROOT) as run:
        if run.reused:
            return json.loads((output / 'manifest.json').read_text(encoding='utf8'))
        if output.exists():
            raise ValueError('output already exists')
        (output / 'images').mkdir(parents=True)
        (output / 'labels').mkdir()
        records = []
        for number, (manifest, index, row, image, lines) in enumerate(selected):
            stem = f'car_{number:05d}'
            target_image = output / 'images' / (stem + image.suffix.lower())
            target_label = output / 'labels' / (stem + '.txt')
            shutil.copy2(image, target_image)
            target_label.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf8')
            records.append({'image': relative(target_image), 'label': relative(target_label),
                            'source_image': relative(image), 'source_manifest': relative(manifest),
                            'record_index': index, 'source_record': row,
                            'source_refs': run.source_refs.get(relative(image), []),
                            'generation_method': ('provided_car_camera_image_or_video_frame' if source_set == 'car'
                                                  else 'provided_site_camera_image_or_video_frame'),
                            'usage': 'unassigned_candidate_not_training_approved',
                            'ocr_status': 'not_verified'})
        result = {'status': 'unassigned_candidates_not_training_approved',
                  'scope': 'Existing camera visual candidates; source status is preserved.',
                  'source_set': source_set,
                  'count': len(records), 'records': records}
        (output / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-set', choices=('car', 'site'), default='car')
    args = parser.parse_args()
    result = package(ROOT / args.output, args.source_set)
    print(json.dumps({'status': result['status'], 'count': result['count']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
