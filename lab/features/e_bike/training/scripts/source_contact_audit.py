"""Evidence-only source contacts and pixel/thumbnail checks for frozen e-bike data."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from datasets.scripts.common.images import read_image, write_image
from lab.features.e_bike.training.scripts.evaluation_audit import family


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def roles(row,foreground_lookup=None,background_lookup=None):
    source = row['source_record']; details = source.get('research_details', {})
    if source.get('source_image'):
        videos = sorted(p for p in source.get('sources', []) if p.lower().endswith('.mp4'))
        return {'direct_parent': videos or [row.get('source_group', 'unknown')],
                'direct_frame': [source['source_image']]}
    background = source.get('background', {})
    foreground_records = [o['foreground_source'] for o in source.get('objects', []) if o.get('foreground_source')]
    foregrounds = [p.get('path') if isinstance(p,dict) else p for p in foreground_records]
    foregrounds += [v['source'] for v in details.get('generation_state', {}).get('target', {}).get('vehicles', [])]
    background_parent = background.get('source_video') or background.get('scene') or background.get('parent_group') or details.get('parent_group')
    background_image=background.get('image') or details.get('background')
    if background_lookup and background_image in background_lookup:
        background_parent=background_lookup[background_image]
    videos = [p for p in source.get('sources', []) if p.lower().endswith('.mp4') and p != background_parent]
    videos += [p['source_video'] for p in foreground_records if isinstance(p,dict) and p.get('source_video')]
    videos += [foreground_lookup[p]['source_video'] for p in foregrounds
               if foreground_lookup and p in foreground_lookup and foreground_lookup[p].get('source_video')]
    return {'background_image': [background_image],
            'background_parent': [background_parent], 'foreground_crop': sorted(set(p for p in foregrounds if p)),
            'foreground_parent': sorted(set(videos))}


def mirror_digest(frame):
    shape = str(frame.shape).encode()
    return min(hashlib.sha256(shape+np.ascontiguousarray(f).tobytes()).hexdigest()
               for f in (frame, frame[:, ::-1]))


def thumbnail(frame):
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64,48)).astype(np.float32)/255


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/e_bike/source_contact_20261118')
    args = parser.parse_args(); out = (ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT/'runs/e_bike'): raise ValueError('Expected research output')
    out.mkdir(parents=True, exist_ok=False); cv2.setNumThreads(1)
    original = read(ROOT/'runs/e_bike/synthetic_short_inputs_20261118/manifest.json')['records']
    evaluation = read(ROOT/'runs/e_bike/candidate_verifier_20261118/evaluation_snapshot.json')['records']
    train = [r for r in original if r['split'] == 'train']
    val = [r for r in evaluation if r['split'] == 'val']
    selection_path='datasets/annotated/e_bike/intermediate_views_20261118_refined/selection.json'
    foreground_lookup={r['path']:r for r in read(ROOT/selection_path)['vehicles']}
    background_lookup={r['source_record']['background']['image']:r['source_record']['background'].get('source_video') or r['source_record']['background'].get('parent_group')
                       for r in original if r['source_record'].get('background')}
    train_roles = defaultdict(set)
    for row in train:
        for role, paths in roles(row,foreground_lookup,background_lookup).items(): train_roles[role].update(p for p in paths if p)
    contacts=[]; hashes=defaultdict(list)
    for row in train:
        hashes[mirror_digest(read_image(ROOT/row['image']))].append(row['image'])
    for row in val:
        contact={role: sorted(set(p for p in paths if p) & train_roles[role]) for role, paths in roles(row,foreground_lookup,background_lookup).items()}
        contacts.append({'image':row['image'], 'family':family(row), 'roles':roles(row,foreground_lookup,background_lookup),
                         'same_role_train_contacts':contact,
                         'train_pixel_or_mirror_duplicates':hashes[mirror_digest(read_image(ROOT/row['image']))]})
    # Explicitly audit inherited V3 pixels as well as the immediate training set.
    legacy = ROOT/'runs/e_bike/runs/e_bike/v3_20261111_142947_802052'
    legacy_rows = [r for r in read(legacy/'data/samples.json') if r['split']=='train']
    targets = [r for r in val if r['source_record'].get('source_image')]
    unique = {family(r):r for r in targets}; targets=list(unique.values())
    thumbs=np.array([thumbnail(read_image(ROOT/r['image'])) for r in targets])
    target_hashes={mirror_digest(read_image(ROOT/r['image'])):r['image'] for r in targets}
    nearest=[[] for _ in targets]; legacy_duplicates=[]; missing=[]
    for index,row in enumerate(legacy_rows):
        path=legacy/row['run_image']
        if not path.exists(): missing.append(str(path.relative_to(ROOT))); continue
        frame=read_image(path); digest=mirror_digest(frame)
        if digest in target_hashes: legacy_duplicates.append({'evaluation':target_hashes[digest], 'legacy_train':path.relative_to(ROOT).as_posix()})
        small=thumbnail(frame)
        distances=np.minimum(np.mean((thumbs-small)**2, axis=(1,2)),np.mean((thumbs-small[:,::-1])**2,axis=(1,2)))
        for j,d in enumerate(distances):
            nearest[j].append((float(d),path.relative_to(ROOT).as_posix()))
            nearest[j]=sorted(nearest[j])[:3]
        if index%1000==0: print(json.dumps({'legacy_training_scanned':index+1}),flush=True)
    neighbors=[{'evaluation':row['image'], 'nearest_legacy':[{'thumbnail_mse':d,'image':p} for d,p in items]}
               for row,items in zip(targets,nearest)]
    rows=[]
    for row,items in zip(targets,nearest):
        cells=[]
        for p in [row['image']]+[p for _,p in items]:
            frame=cv2.resize(read_image(ROOT/p),(320,240));canvas=np.full((265,320,3),255,np.uint8);canvas[25:]=frame
            cv2.putText(canvas,Path(p).stem[:40],(3,17),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,0),1);cells.append(canvas)
        rows.append(np.concatenate(cells,axis=1))
    write_image(out/'nearest_legacy_contact.jpg',np.concatenate(rows,axis=0))
    report={'immediate_train_images':len(train),'evaluation_images':len(val),'contacts':contacts,
            'foreground_lineage_selection':selection_path,
            'legacy_training_scanned':len(legacy_rows)-len(missing),'legacy_missing_inputs':missing,
            'legacy_exact_mirror_duplicates':legacy_duplicates,'nearest_legacy_thumbnails':neighbors,
            'nearest_preview':(out/'nearest_legacy_contact.jpg').relative_to(ROOT).as_posix(),
            'limits':['Thumbnail similarity is only a review lead, not proof of duplicate or same source.',
                      'Role contacts follow recorded lineage, not inferred filename equality.',
                      'Absent exact pixel overlap does not prove source isolation; inherited initialization and repeated validation contact remain.',
                      'Immediate training is original156; later focused detector ablations are separate model branches.'],
            'initialization_chain':['V3 train4000 -> V4 baseline train108/val10 (118 total) -> two-stage detector train156 -> keypoint-control detector train156.',
                                    'V4 detector backbone transfer -> two-stage classifier original156 GT crops -> three-class verifier original156 crops.',
                                    'Old26 repeatedly evaluated throughout; existing32 repeatedly evaluated after creation.']}
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'legacy_scanned':report['legacy_training_scanned'],'legacy_exact_duplicates':len(legacy_duplicates),
                      'current_pixel_duplicate_images':sum(bool(r['train_pixel_or_mirror_duplicates']) for r in contacts),
                      'same_role_contacts':{k:sum(bool(r['same_role_train_contacts'].get(k)) for r in contacts) for k in train_roles}},ensure_ascii=False))


if __name__ == '__main__': main()
