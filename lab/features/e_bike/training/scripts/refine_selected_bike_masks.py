"""Apply two narrowly scoped visual corrections to four selected SAM2 masks."""
import argparse
import json

import cv2
import numpy as np

import preview_person_occlusions as core


def local_background_cleanup(rgba, index):
    h, w = rgba.shape[:2]
    alpha = rgba[:, :, 3].copy()
    regions = {
        0: ((425, 485, 1073, 1105), 65, '仅去除车架间镂空内的浅灰背景；边缘需人工复核'),
        2: ((330, 460, 960, 1015), 80, '仅在前叉镂空的小区域去除低饱和浅色背景；边缘需人工复核'),
    }
    if index not in regions:
        return alpha, '未修改；局部边缘及孔洞仍需人工核对'
    (left, right, top, bottom), min_value, note = regions[index]
    hsv = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_BGR2HSV)
    roi = np.zeros((h, w), np.uint8)
    x1, x2 = round(left / 1152 * w), round(right / 1152 * w)
    y1, y2 = round(top / 2048 * h), round(bottom / 2048 * h)
    roi[y1:y2, x1:x2] = 1
    background = (roi > 0) & (hsv[:, :, 1] < 50) & (hsv[:, :, 2] > min_value)
    alpha[background] = 0
    return alpha, note


def refine(selection, output):
    if output.exists():
        raise ValueError('输出目录已存在')
    plan = json.loads(selection.read_text(encoding='utf-8'))
    items = plan['vehicles']
    expected = ['sam2_01_fallen.png', 'sam2_03_fallen.png',
                'sam2_04_upright.png', 'sam2_06_upright.png']
    if [core.workspace_path(item['path']).name for item in items] != expected:
        raise ValueError('本次修补仅适用于已选定的四个具体视角')
    output.mkdir(parents=True)
    status, changes, error = 'interrupted', [], '处理被中断'
    try:
        for index, item in enumerate(items):
            source = core.workspace_path(item['path'])
            rgba = cv2.imdecode(np.fromfile(str(source), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
                raise ValueError(f'无法读取 RGBA 掩码：{source}')
            h, w = rgba.shape[:2]
            if (w, h) != (2160, 3840):
                raise ValueError('本次修补依赖 2160×3840 的源帧尺寸')
            alpha, note = local_background_cleanup(rgba, index)
            rgba[:, :, 3] = alpha
            destination = output / source.name
            core.write_image(destination, rgba)
            item['path'] = destination.relative_to(core.ROOT).as_posix()
            item['visual_notes'] = note
            changes.append({'source_mask': source.relative_to(core.ROOT).as_posix(),
                            'result_mask': item['path'], 'visual_note': note})
        plan['vehicles'] = items
        plan['vehicle_masks'] = '四张 SAM2 候选经逐图局部对照；两张做有限局部修补，均待人工复核'
        (output / 'selection.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
        status, error = 'complete_candidates', None
    except Exception as exc:
        status, error = 'failed', f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (output / 'refinement_manifest.json').write_text(json.dumps(
            {'status': status, 'changes': changes, 'error': error}, ensure_ascii=False, indent=2), encoding='utf-8')
    return changes


def main():
    parser = argparse.ArgumentParser(description='局部修补四张指定电动车掩码')
    parser.add_argument('--selection', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        changes = refine(core.workspace_path(args.selection), core.workspace_path(args.output))
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        parser.exit(2, f'失败：{exc}\n')
    print(f'完成 {len(changes)} 张候选复核与有限修补；仍需人工复核')


if __name__ == '__main__':
    main()
