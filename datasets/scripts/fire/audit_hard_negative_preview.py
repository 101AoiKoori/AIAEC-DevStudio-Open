"""Re-read color-contrast preview pixels, ID masks, labels and paired metadata."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.annotations import yolo_row
from datasets.scripts.common.frame_environment import image_digest
from datasets.scripts.fire.fire_blender_labels import decode_ids, srgb_to_linear, visible_bbox


def audit():
    cfg = json.loads((ROOT/'datasets/scripts/fire/config/hard_negative_preview.json').read_text(encoding='utf8'))
    folder = ROOT/cfg['output']
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf8'))
    rows = manifest['records']; pairs = {}; hashes = set(); counts = {'positive': 0, 'same_color_negative': 0, 'other_color_negative': 0}
    expected = {(v['kind'], color, count) for v in cfg['variants'] for color in v['colors'] for count in cfg['target_counts']}
    actual = set()
    for row in rows:
        image = folder/row['image']; digest = image_digest(image)
        if digest in hashes:
            raise ValueError('Duplicate preview image')
        hashes.add(digest)
        if row['parents'] != [cfg['source_scene']] or row['split'] != 'train':
            raise ValueError('Blender preview parent/split differs')
        if Image.open(image).size != tuple(cfg['image_size']):
            raise ValueError('Preview image size differs')
        targets = row['instances']; distractor = row['distractor']
        actual.add((distractor['kind'], distractor['color'], len(targets)))
        rgb = np.asarray(Image.open(folder/row['instance_pass']).convert('RGB'), dtype=np.float32)/255
        ids = decode_ids(srgb_to_linear(rgb), [1, *distractor['object_ids']])
        bbox, area = visible_bbox(ids==1)
        if targets:
            if len(targets) != 1 or bbox != targets[0]['xyxy'] or area != targets[0]['visible_pixels']:
                raise ValueError('Target mask differs from its record')
            counts['positive'] += 1
        else:
            if bbox is not None:
                raise ValueError('Negative preview contains a target ID')
            counts['same_color_negative' if distractor['same_color_hard_negative'] else 'other_color_negative'] += 1
        bbox, area = visible_bbox(np.isin(ids, distractor['object_ids']))
        if bbox != distractor['xyxy'] or area != distractor['visible_pixels']:
            raise ValueError('Distractor mask differs')
        expected_label = ''.join(yolo_row(t['xyxy'], *cfg['image_size'])+'\n' for t in targets)
        if (folder/row['label']).read_text(encoding='utf8') != expected_label:
            raise ValueError('YOLO labels differ from the final visible masks')
        if sum(v['visible_count'] for v in row['building_counts'].values()) != len(targets):
            raise ValueError('Building counts do not partition the target labels')
        if row['actual']['ground_error_m'] > .0001 or row['actual']['sticker_wall_plane_error_m'] > .000001:
            raise ValueError('Geometry check failed')
        pairs.setdefault(row['pair_id'], []).append(row)
    if actual != expected or len(rows) != len(expected):
        raise ValueError('Preview color/count coverage differs from the config')
    for group in pairs.values():
        if len(group) != 2 or sorted(len(r['instances']) for r in group) != [0, 1] or group[0]['actual'] != group[1]['actual']:
            raise ValueError('Positive and negative pair changed camera, prop or lighting')
    # Compare the geometry/camera/light of the two color variants of each object.
    for variant in cfg['variants']:
        controls = [r for r in rows if r['distractor']['kind'] == variant['kind'] and not r['instances']]
        a, b = [{k:v for k,v in r['actual'].items() if k != 'color'} for r in controls]
        if a != b:
            raise ValueError('Color controls changed geometry or illumination')
    report = {'task': 'fire', 'status': 'passed', 'images': len(rows), 'target_instances': counts['positive'],
        **counts, 'paired_scenes': len(pairs), 'parent_groups': 1, 'split': 'train',
        'scope': 'Development preview, not exported or model-evaluated', 'visual_review': 'pending'}
    review = ROOT/'datasets/annotated/fire/hard_negative_review'; review.mkdir(parents=True, exist_ok=True)
    visual_path = review/'visual_review.json'
    if visual_path.is_file():
        visual = json.loads(visual_path.read_text(encoding='utf8'))
        if visual.get('manifest_sha256') == image_digest(folder/'manifest.json'):
            report['visual_review'] = visual['status']
            report['visual_review_record'] = visual_path.relative_to(ROOT).as_posix()
    sheet = Image.new('RGB', (960, 390*((len(rows)+1)//2)), '#eeeeee'); draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        im = Image.open(folder/row['image']).convert('RGB'); d = ImageDraw.Draw(im)
        for instance in row['instances']:
            d.rectangle(instance['xyxy'], outline='#00dd55', width=2)
        d.rectangle(row['distractor']['xyxy'], outline='#0099ff', width=2)
        im.thumbnail((480, 360)); x, y = index%2*480, index//2*390
        sheet.paste(im, (x, y)); draw.text((x+6, y+365), Path(row['image']).stem, fill='black')
    sheet.save(review/'contact_sheet.jpg', quality=90)
    (review/'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    audit()
