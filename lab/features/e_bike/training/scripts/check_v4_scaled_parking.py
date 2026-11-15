"""Validate masks and compare new target scale against same-frame real bikes."""
import json
from pathlib import Path
import sys

from check_v4_multibike_parking import ROOT,run_check
from compose_v4_scaled_parking import references


def main():
    if len(sys.argv)!=2:raise ValueError('Expected one e-bike dataset path')
    base=(ROOT/Path(sys.argv[1])).resolve()
    run_check(base, scale_check=True)


def _scale_check(base, manifest, review):
    refrows={x['source_frame']:x for x in review['rows']}
    issues=[];checked=0;truncated=0
    for row in manifest['records']:
        if row['background_parent'] not in refrows:
            issues.append([row['image'],'reference_background_missing']);continue
        refs=references(refrows[row['background_parent']]);height=row['image_size'][1]
        for obj in row['objects']:
            if obj['truth']!='visible_mask_candidate':continue
            checked+=1;ref=refs[obj['class']];fy=obj['foot_xy'][1]
            minimum=ref['pixels']*(1+.35*(fy-ref['foot_y'])/height)
            if (obj['scale_reference']!=ref or fy<ref['foot_y'] or
                    abs(obj['minimum_dimension_px']-minimum)>1e-4 or obj['projected_dimension_px']<minimum):
                issues.append([row['image'],'reference_scale'])
            if obj['truncated']:
                truncated+=1
            else:
                a,b,c,d=obj['bbox_xyxy'];visible=c-a if obj['class']=='fallen' else d-b
                if visible<ref['pixels']:
                    issues.append([row['image'],'visible_dimension_smaller_than_real_reference'])
    qa_path=base/'qa.json';qa=json.loads(qa_path.read_text(encoding='utf-8'))
    qa['scale_check']={'targets':checked,'truncated_targets':truncated,'errors':issues[:50],
        'error_count':len(issues),'policy':'Per-frame real bbox; closer full sprite must meet fitted nondecreasing '
        'size, and nontruncated visible dimension must not be smaller than the real reference.',
        'limit':'Projection slope is fitted and not physically measured; cropped boxes are explicitly recorded.'}
    if issues:qa['status']='failed'
    qa_path.write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(qa['scale_check'],ensure_ascii=False))
    if issues:raise SystemExit(1)


if __name__=='__main__':main()
