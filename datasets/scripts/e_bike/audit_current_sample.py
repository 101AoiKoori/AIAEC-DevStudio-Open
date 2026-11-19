"""Verify current labels/masks/mirror pairs and refresh focused synthetic previews."""
from collections import Counter, defaultdict
import html
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.config import load_task_config
from datasets.scripts.common.images import read_image
from datasets.scripts.common.package_direct_real import check_yolo
from datasets.scripts.e_bike.plan_low_angle_batch import occlusion_relations


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    _, config = load_task_config('e_bike')
    manifest = read(ROOT / config['sample_manifest'])
    recipe = read(ROOT / config['sample_recipe'])
    current_plan = read(ROOT / config['production_plan'])
    snapshot = read(ROOT / manifest['generation_plan'])
    assert len(manifest['records']) == snapshot['budget']
    assert sum(r['branch'] != 'direct_real' for r in manifest['records']) == current_plan['budget']
    coverage = {r['state_id']: r for r in manifest['sampling']['coverage']}
    pairs = defaultdict(list)
    branches = Counter()
    masks_checked = direct_checked = 0
    selected = []
    for row in manifest['records']:
        frame = read_image(ROOT / row['image'])
        h, w = frame.shape[:2]
        assert row['image_size'] == [w, h], row['state_id']
        text = (ROOT / row['label']).read_text(encoding='utf-8')
        check_yolo(text, 2, 0)
        labels = text.splitlines()
        assert len(labels) == len(row['objects'])
        assert not row.get('unresolved_regions')
        occupied = np.zeros((h, w), bool)
        for label, obj in zip(labels, row['objects']):
            cls, cx, cy, bw, bh = map(float, label.split())
            x1, y1, x2, y2 = obj['bbox_xyxy']
            assert int(cls) == ('upright', 'fallen').index(obj['class'])
            np.testing.assert_allclose([cx, cy, bw, bh],
                [(x1+x2)/(2*w), (y1+y2)/(2*h), (x2-x1)/w, (y2-y1)/h], atol=1e-6)
            if obj.get('visible_mask'):
                mask = read_image(ROOT / obj['visible_mask'])
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                mask = mask > 0
                yy, xx = np.nonzero(mask)
                assert [int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1] == obj['bbox_xyxy']
                assert not np.any(occupied & mask)
                occupied |= mask
                assert int(mask.sum()) == obj['visible_pixels']
                assert obj['visible_fraction'] >= .7
                assert abs(obj['visible_fraction'] + sum(v['fraction'] for v in obj['occluded_by'])
                           + obj['truncated_pixels']/obj['isolated_pixels'] - 1) < 1e-6
                masks_checked += 1
        if row['branch'] == 'direct_real':
            source = read_image(ROOT / row['source_image'])
            if row['mirrored']:
                source = source[:, ::-1]
            np.testing.assert_array_equal(frame, source)
            direct_checked += 1
        meta = coverage[row['state_id']]
        branch = meta['branch']
        branches[branch] += 1
        if meta.get('mirror_pair'):
            pairs[meta['mirror_pair']].append(row)
        if '_over_' in branch:
            assert branch in occlusion_relations(row['objects'], recipe['minimum_effective_occlusion'])
        if branch in recipe.get('queue_variants', {}):
            assert Counter(o['class'] for o in row['objects']) == Counter(recipe['queue_variants'][branch]['classes'])
            selected.append(row)
    for rows in pairs.values():
        assert len(rows) == 2
        assert Counter(o['class'] for o in rows[0]['objects']) == Counter(o['class'] for o in rows[1]['objects'])
        assert [o['foreground_source'] for o in rows[0]['objects']] == [o['foreground_source'] for o in rows[1]['objects']]
    for branch, count in recipe['paired_layouts'].items():
        assert branches[branch] == 2 * count, branch
    output = ROOT / 'datasets/annotated/e_bike/current_queue_review'
    output.mkdir(exist_ok=True, parents=True)
    cards = []
    for row in selected:
        href = Path(os.path.relpath(ROOT / row['overlay'], output)).as_posix()
        counts = Counter(o['class'] for o in row['objects'])
        caption = f"{row['state_id']} U={counts['upright']} F={counts['fallen']} mirror={coverage[row['state_id']]['mirrored']}"
        cards.append(f'<figure><a href="{href}"><img loading="lazy" src="{href}"></a><figcaption>{html.escape(caption)}</figcaption></figure>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>合成数量对照小样</title>'
        '<style>body{font:16px sans-serif;margin:24px;background:#eee}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:16px}figure{background:white;padding:10px;margin:0}img{width:100%}</style>'
        '<h1>合成数量对照：1／3／5／7 辆及镜像</h1><p>使用现有裁片、实拍底图与 Blender 理想空场；绿框直立，橙框倒伏。'
        '同组地面落点保持一致，交换姿态数量；原向与车体镜像成对。二维接地、阴影与照明为近似，未证明模型改进。</p><div class="grid">'
        + ''.join(cards) + '</div>', encoding='utf-8')
    examples = []
    for branch in ('queue_3_u1_f2', 'queue_5_u2_f3', 'queue_7_u3_f4', 'queue_single_fallen'):
        for domain in ('real_background_composite', 'blender_background_composite'):
            matches = [r for r in selected if coverage[r['state_id']]['branch'] == branch
                       and r['branch'] == domain and not coverage[r['state_id']]['mirrored']]
            if matches:
                examples.append(matches[0])
    sheet = Image.new('RGB', (1280, 504*((len(examples)+1)//2)), '#eeeeee')
    for i, row in enumerate(examples):
        with Image.open(ROOT / row['overlay']) as im:
            sheet.paste(im.convert('RGB'), (i%2*640, i//2*504))
        ImageDraw.Draw(sheet).text((i%2*640+8, i//2*504+482), coverage[row['state_id']]['branch'], fill='#111111')
    sheet.save(output/'examples.jpg', quality=94)
    result = {'summary':manifest['summary'], 'all_pairs_checked':len(manifest['records']),
              'visible_mask_boxes_checked':masks_checked, 'direct_pixels_checked':direct_checked,
              'paired_synthetic_layouts':len(pairs), 'focused_synthetic_images':len(selected),
              'images_by_condition':dict(branches), 'partial_images':0,
              'scope':'Current sample verification only; prior baseline export and model are unchanged and do not evaluate these added samples.'}
    for path in (output/'check.json', ROOT/config['current_selection']['verification']):
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()
