"""Check the mixed plate draft and render one contact sheet by recorded strata."""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from prepare_plate_v4_mixed_candidates import OUT, ROOT, rel, valid_label

sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import generation_guard  # noqa: E402


def main():
    manifest_path = OUT/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['status'] != 'draft_not_deliverable':
        raise ValueError('Expected an unapproved draft dataset')
    images = {ROOT/row[key] for row in manifest['records'] for key in ('source_image', 'image')}
    labels = {ROOT/row[key] for row in manifest['records'] for key in ('label', 'source_label')}
    procedural = {manifest_path, *labels}
    target = ROOT/'runs/plate_detection/v4_mixed_v2_stratified_contact.png'
    report_path = OUT/'qa.json'
    with generation_guard(
        'plate_mixed_candidates_quality_check',
        sorted(images | procedural), [target, report_path],
        {'contact_tile_size': [320, 220], 'contact_columns': 4},
        code_paths=[Path(__file__), Path(__file__).with_name('prepare_plate_v4_mixed_candidates.py')],
        procedural_inputs=sorted(procedural), root=ROOT,
    ) as run:
        if run.reused:
            print(report_path.read_text(encoding='utf-8'))
            return
        _check(manifest, target, run.source_refs)


def _check(manifest, target, source_refs):
    errors, originals, parents, samples = [], defaultdict(set), defaultdict(set), {}
    dimensions = Counter()
    for row in manifest['records']:
        source, image, label = ROOT/row['source_image'], ROOT/row['image'], ROOT/row['label']
        try:
            im = cv2.imdecode(np.fromfile(str(image), dtype=np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                raise ValueError('decode failed')
            height, width = im.shape[:2]
            if [width,height] != row['image_size'] or max(width,height) > 640:
                raise ValueError('unexpected size')
            actual, expected = valid_label(label), valid_label(ROOT/row['source_label'])
            if actual != expected:
                raise ValueError('source label differs')
            originals[hashlib.sha256(source.read_bytes()).digest()].add(row['split'])
            parents[row['parent_image']].add(row['split'])
            dimensions[(width,height)] += 1
            key = (row['split'],row['family'],row['tone_profile'],bool(actual))
            if key not in samples:
                for line in actual.splitlines():
                    _,x,y,w,h = map(float,line.split())
                    cv2.rectangle(im,(int((x-w/2)*width),int((y-h/2)*height)),
                                  (int((x+w/2)*width),int((y+h/2)*height)),(0,255,0),2)
                canvas = np.zeros((220,320,3),np.uint8)
                scale = min(320/width,200/height)
                resized = cv2.resize(im,(max(1,round(width*scale)),max(1,round(height*scale))))
                canvas[:resized.shape[0],:resized.shape[1]] = resized
                cv2.putText(canvas, f"{row['split']} {row['family']}", (3,212),
                            cv2.FONT_HERSHEY_SIMPLEX,.34,(255,255,255),1)
                samples[key] = canvas
        except (OSError,ValueError,IndexError) as exc:
            errors.append({'image':row['image'],'error':str(exc)})
    exact_cross = sum(len(roles)>1 for roles in originals.values())
    parent_cross = [name for name,roles in parents.items() if len(roles)>1]
    # Interleave families so all three appear in a bounded visual sample.
    ordered = []
    for family in ('provincial_synthetic','ccpd_road_capture','site_video_augmentation'):
        ordered.extend(value for key,value in samples.items() if key[1] == family)
    while len(ordered)%4:
        ordered.append(np.zeros((220,320,3),np.uint8))
    contact = np.vstack([np.hstack(ordered[i:i+4]) for i in range(0,len(ordered),4)])
    target.parent.mkdir(parents=True,exist_ok=True)
    cv2.imencode('.png',contact)[1].tofile(str(target))
    report = {'status':'structural_passed_visual_sample_pending' if not errors and not exact_cross and not parent_cross else 'failed',
        'images':len(manifest['records']), 'error_count':len(errors), 'errors':errors[:50],
        'image_sizes':{f'{a}x{b}':n for (a,b),n in dimensions.items()},
        'exact_source_bytes_cross_split':exact_cross, 'explicit_parent_cross_split':parent_cross,
        'duplicate_scope':'Exact bytes and explicit parents only; near duplicates may remain.',
        'visual_sample_count':len(samples),'visual_evidence':rel(target), 'source_refs':source_refs}
    (OUT/'qa.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
    if report['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
