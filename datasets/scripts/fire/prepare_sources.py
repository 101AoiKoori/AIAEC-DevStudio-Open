"""Apply the task's selected parent splits and reviewed facade/photo geometry."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from datasets.scripts.common.assign_split import assign_selected
from datasets.scripts.common.config import load_task_config
from datasets.scripts.common.frame_environment import query_frames, edit_database

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    source = load_task_config('fire')[1]
    plan = json.loads((ROOT/source['full_dataset_plan']).read_text(encoding='utf8'))
    selected = {f['frame_id']: f for f in plan['frames']}
    for role in ('direct_target', 'empty_negative', 'composite_background'):
        rows = query_frames(task='fire', role=role)
        for split in ('train', 'val', 'test'):
            pending = [r['record_id'] for r in rows if r['frame_id'] in selected
                       and selected[r['frame_id']]['split'] == split and r['split'] == 'unassigned']
            if args.apply and pending:
                outcomes = assign_selected('fire', role, split, pending)
                if any(r['status'] not in ('assigned', 'unchanged') for r in outcomes):
                    raise ValueError(f'Frame split assignment failed: {outcomes}')
    if args.apply:
        rows = {r['frame_id']: r for role in ('direct_target', 'empty_negative')
                for r in query_frames(task='fire', role=role) if r['frame_id'] in selected}
        reviews = [json.loads(line) for line in (ROOT/source['wall_plane_review']).read_text(encoding='utf8').splitlines() if line.strip()]
        newest = {r['frame_id']: (n, r) for n, r in enumerate(reviews) if 'geometry' in r}
        with edit_database() as writer:
            for fid, f in selected.items():
                if not f['wall_background']:
                    continue
                current = query_frames(task='fire', role='composite_background', frame_id=fid, split=f['split'])
                if current and current[0].get('geometry', {}).get('wall_surfaces'):
                    continue
                n, review = newest[fid]
                row = rows[fid]
                if review['content_sha256'] != row['content_sha256']:
                    raise ValueError(f'Wall evidence bytes changed: {fid}')
                reference = {'manifest': source['wall_plane_review'], 'record_index': n}
                geometry_id = writer.observe(fid, task='fire', kind='geometry', scope='placement',
                    evidence_key='fire_wall_projection', reference=reference, payload=review['geometry'],
                    origin='machine', algorithm_version='fire_wall_homography_v2')
                labels = [e['evidence_id'] for e in row['evidence'] if e['kind'] == 'label']
                if len(labels) != 1:
                    raise ValueError(f'One complete source label observation required: {fid}')
                details = {k: deepcopy(row[k]) for k in ('image', 'parent_group', 'source_kind', 'scope')}
                details.update(label_evidence_id=labels[0], geometry_evidence_id=geometry_id,
                    role_basis='background_review', reason_codes=['complete_fire_labels_and_wall_plane_registration'])
                writer.decide(fid, task='fire', role='composite_background', split=f['split'], status='eligible',
                    evidence_ids=[*labels, geometry_id], details=details, origin='machine')
    parents = {}
    counts = {}
    for role in ('direct_target', 'empty_negative', 'composite_background'):
        for row in query_frames(task='fire', role=role):
            if row['frame_id'] not in selected:
                continue
            expected = selected[row['frame_id']]['split']
            if row['split'] != expected:
                raise ValueError(f'Parent split differs for {role}/{row["frame_id"]}: {row["split"]} != {expected}')
            parents.setdefault(row['parent_group'], set()).add(row['split'])
            counts[(role, row['split'])] = counts.get((role, row['split']), 0)+1
    if any(len(splits) > 1 for splits in parents.values()):
        raise ValueError('A selected capture parent occurs in more than one split')
    print(json.dumps({'parent_groups': len(parents), 'counts': {str(k): v for k, v in counts.items()},
                      'applied': args.apply}, ensure_ascii=False))


if __name__ == '__main__':
    main()
