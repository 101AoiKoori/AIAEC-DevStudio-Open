"""Audit completed bin-render candidates; visual and real-world review is separate."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from bin_render_contract import STATE_CONTRACT, attach_state_evidence, decode_mask, measure_visibility


def read_mask(root, relative):
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError('missing or invalid mask ' + relative)
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.shape != (1024, 1024, 4):
        raise ValueError('mask must be 1024×1024 RGBA: ' + relative)
    return image[:, :, [2, 1, 0, 3]].astype(np.float32) / 255.0


def audit_masks(root, sample, data):
    visible = read_mask(root, sample['mask_visible'])
    new_contract = data.get('label_contract_id') == STATE_CONTRACT
    state_visible = read_mask(root, sample['mask_state_visible']) if new_contract else None
    errors = []
    for item in sample['labels']:
        isolated = read_mask(root, item['isolated_mask'])
        measured = measure_visibility(decode_mask(visible, item['bin']),
            decode_mask(isolated, item['bin']), item['in_frame'],
            data['min_visibility'], data['min_visible_pixels'])
        if new_contract:
            evidence = item['state_evidence']
            state_isolated = read_mask(root, evidence['isolated_mask'])
            checked_evidence = measure_visibility(decode_mask(state_visible, item['bin']),
                decode_mask(state_isolated, item['bin']), evidence['in_frame'],
                data['min_visibility'], data['min_visible_pixels'])
            for field in ('candidate_xyxy_normalized', 'visible_pixels', 'isolated_pixels',
                          'visible_fraction', 'label_eligible', 'rejection_reasons'):
                if checked_evidence[field] != evidence[field]:
                    errors.append(f"state mask {field} mismatch for {sample['image']} {item['bin']}")
            expected = attach_state_evidence(measured, checked_evidence, item['state'])
            if evidence['kind'] != expected['state_evidence']['kind']:
                errors.append(f"state evidence kind mismatch for {sample['image']} {item['bin']}")
            for field in ('target_eligible', 'target_rejection_reasons'):
                if item[field] != expected[field]:
                    errors.append(f"target {field} mismatch for {sample['image']} {item['bin']}")
            measured = expected
        for field in ('candidate_xyxy_normalized', 'visible_pixels', 'isolated_pixels',
                      'visible_fraction', 'label_eligible', 'rejection_reasons'):
            if measured[field] != item[field]:
                errors.append(f"mask {field} mismatch for {sample['image']} {item['bin']}")
    return errors


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    root=args.directory.resolve()
    data=json.loads((root/'manifest.json').read_text(encoding='utf8'))
    samples=data['samples']
    errors=[]
    states=defaultdict(Counter)
    groups=defaultdict(list)
    labels_count=0
    for sample in samples:
        image_path=(root/sample['image']).resolve()
        if root not in image_path.parents or not image_path.is_file():
            errors.append(f"missing or invalid image {sample['image']}")
            continue
        image=cv2.imdecode(np.fromfile(str(image_path),np.uint8),cv2.IMREAD_UNCHANGED)
        if image is None or image.shape[:2] != (1024,1024):
            errors.append(f"invalid image {sample['image']}")
        labels=sample['labels']
        if len(labels)!=4 or {item['bin'] for item in labels}!={'blue','red','green','gray'}:
            errors.append(f"incorrect bin count {sample['image']}")
        for item in labels:
            states[item['bin']][item['state']]+=1
        groups[sample['group']].append(sample)
        if 'mask_visible' in sample:
            try:
                errors.extend(audit_masks(root, sample, data))
            except (ValueError, KeyError) as exc:
                errors.append(f"invalid mask evidence {sample['image']}: {exc}")
        candidate=sample['candidate_label']
        eligible = all(item.get('label_eligible', item['in_frame']) for item in labels)
        if candidate:
            labels_count+=1
            if not eligible:
                errors.append(f'ineligible image has candidate labels {candidate}')
            candidate_path=(root/candidate).resolve()
            if root not in candidate_path.parents or not candidate_path.is_file():
                errors.append(f'missing label {candidate}')
                continue
            rows=candidate_path.read_text(encoding='utf8').splitlines()
            if len(rows)!=4:
                errors.append(f'incorrect label count {candidate}')
                continue
            for row,item in zip(rows,labels):
                fields=row.split()
                if len(fields)!=5:
                    errors.append(f'malformed label {candidate}')
                    continue
                cls=int(fields[0]); coords=list(map(float,fields[1:]))
                x,y,w,h=coords
                source=item['candidate_xyxy_normalized']
                expected=((source[0]+source[2])/2,(source[1]+source[3])/2,
                          source[2]-source[0],source[3]-source[1])
                if (item['state'] not in ('open','closed') or
                    cls!=(0 if item['state']=='open' else 1) or not item['in_frame'] or
                    w<=0 or h<=0 or x-w/2 < -1e-6 or x+w/2 > 1+1e-6 or
                    y-h/2 < -1e-6 or y+h/2 > 1+1e-6 or
                    any(abs(a-b)>1e-6 for a,b in zip(coords,expected))):
                    errors.append(f'invalid label {candidate}')
        elif eligible:
            errors.append(f"missing eligible label {sample['image']}")
    if data.get('status') not in (None, 'complete_candidates'):
        errors.append(f"run status is {data.get('status')}")
    if len(samples)!=data['requested_images']:
        errors.append(f"incomplete: {len(samples)} / {data['requested_images']}")
    recorded_images={s['image'] for s in samples}
    actual_images={p.relative_to(root).as_posix() for p in (root/'images').glob('*.png')}
    if len(recorded_images)!=len(samples) or recorded_images!=actual_images:
        errors.append('image files and manifest differ')
    recorded_labels={s['candidate_label'] for s in samples if s['candidate_label']}
    actual_labels={p.relative_to(root).as_posix() for p in (root/'labels_candidate').glob('*.txt')}
    if recorded_labels!=actual_labels:
        errors.append('candidate label files and manifest differ')
    for group,entries in groups.items():
        for field in ('camera','lighting','body_colors_linear','background_gray_linear'):
            if any(item[field]!=entries[0][field] for item in entries):
                errors.append(f'group {group} has inconsistent {field}')
        if len({tuple(item['states']) for item in entries})!=len(entries):
            errors.append(f'group {group} has duplicate state combinations')
    for name in ('blue','red','green','gray'):
        if abs(states[name]['open']-states[name]['closed'])>1:
            errors.append(f'unbalanced states for {name}: {dict(states[name])}')
    print(json.dumps({'images':len(samples),'requested':data['requested_images'],
        'images_with_candidate_labels':labels_count,'images_without_labels':len(samples)-labels_count,
        'states':states,'group_count':len(groups),'errors':errors,
        'note':'Candidate boxes and visibility require visual review.'},ensure_ascii=False,indent=2))
    if errors:
        raise SystemExit(1)


if __name__=='__main__':
    main()
