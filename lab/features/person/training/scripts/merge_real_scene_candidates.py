"""Mix visually reviewed team images with real-scene synthetic person candidates.

Both input sources share the same video source and remain AI training candidates.
This tool makes no validation/test split and writes no vehicle supervision.
"""
import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SCENES = {'single': 875, 'multi': 1050, 'occlusion': 1050, 'no_person': 525}
REVIEWED_TEAM_IDS = (355,429,426,205,441,460,2,360,383,168,193,186,28,625,179,170,425,246,297,442,375,8,579,474,559,11,197,55,565)
REVIEWED_TEAM_NAMES = frozenset(f'images/team_train_train_{i:06}.jpg' for i in REVIEWED_TEAM_IDS)


def read_rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines()]


def select_singles(rows, retain, seed):
    """Largest-remainder proportional selection within class/background/light strata."""
    groups = defaultdict(list)
    for row in rows:
        key = (row['lighting']['stratum'][3], row['background']['kind'], row['lighting']['level'])
        groups[key].append(row)
    total = len(rows)
    if not 0 <= retain <= total or not total:
        raise ValueError('invalid single-image quota')
    counts = {k: len(v) * retain // total for k, v in groups.items()}
    missing = retain - sum(counts.values())
    ranked = sorted(groups, key=lambda k: (-(len(groups[k])*retain % total), str(k)))
    for key in ranked[:missing]:
        counts[key] += 1
    rng = random.Random(seed)
    chosen = []
    for key in sorted(groups, key=str):
        items = sorted(groups[key], key=lambda r: r['image'])
        rng.shuffle(items)
        chosen.extend(items[:counts[key]])
    if len(chosen) != retain or len({r['image'] for r in chosen}) != retain:
        raise ValueError('invalid selection')
    return chosen


def approved_team(rows):
    approved = [r for r in rows if r['image'] in REVIEWED_TEAM_NAMES]
    if len(approved) != len(REVIEWED_TEAM_NAMES) or len({r['image'] for r in approved}) != len(approved):
        raise ValueError('reviewed team images missing or duplicate')
    for row in approved:
        aug = row['source_augmentation']
        if (row['source'] != 'teammate_synthetic' or row['class_id'] not in (0,1)
                or row['source_video_group'] != 'scene_video_49s_shared'
                or row['status'] != 'AI_review_candidate_not_ground_truth'
                or aug.get('flip') is not False or aug.get('crop') is not False or aug.get('occlude') is not False
                or not 0 <= row['source_time_seconds'] <= 5.97 or not .8 <= aug['brightness'] <= 1.25):
            raise ValueError('team image no longer meets review conditions')
    return approved


def validate_source(base, image, label):
    for p in (image, label):
        resolved = (base/p).resolve()
        if not resolved.is_relative_to(base.resolve()) or not resolved.is_file():
            raise ValueError('source missing or outside input: '+str(p))


def exclusions(singles, selected, team_all):
    selected_names = {r['image'] for r in selected}
    result = [{'source_kind':'synthetic','source_image':r['image'],
               'reason':'single_scene_stratified_quota_selection'}
              for r in singles if r['image'] not in selected_names]
    result.extend({'source_kind':r['source'],'source_image':r['image'],
                   'reason':('gray_background_preview' if r['source']=='local_occlusion_candidates'
                             else 'not_in_individually_reviewed_team_subset')}
                  for r in team_all if r['image'] not in REVIEWED_TEAM_NAMES)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic', type=Path, required=True)
    parser.add_argument('--team', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=7901)
    args = parser.parse_args()
    synth, team, out = ((ROOT/p).resolve() for p in (args.synthetic,args.team,args.output))
    if not synth.is_relative_to((ROOT/'datasets/exports/person').resolve()) or not team.is_relative_to((ROOT/'datasets/annotated/person').resolve()):
        parser.error('input directory outside designated person dataset areas')
    if (not out.is_relative_to((ROOT/'datasets/exports/person').resolve()) or out.exists()
            or any(out.is_relative_to(source) or source.is_relative_to(out) for source in (synth, team))):
        parser.error('output must be a new, non-overlapping person export directory')
    summary = json.loads((synth/'summary.json').read_text(encoding='utf8'))
    synthetic = read_rows(synth/'manifest.jsonl')
    if summary['status'] != 'complete_candidates' or len(synthetic) != 3500 or Counter(r['scenario']['kind'] for r in synthetic) != SCENES:
        raise ValueError('synthetic batch incomplete or quota mismatch')
    if len({r['image'] for r in synthetic}) != len(synthetic): raise ValueError('duplicate synthetic name')
    team_all = read_rows(team/'manifest.jsonl')
    reviewed = approved_team(team_all)
    singles = [r for r in synthetic if r['scenario']['kind'] == 'single']
    selected = select_singles(singles, SCENES['single']-len(reviewed), args.seed)
    selected += [r for r in synthetic if r['scenario']['kind'] != 'single']
    names = []
    for row in selected:
        if row['split'] != 'train_candidate' or row['background']['source_group'] != 'scene_video_49s_shared' or row['status'] != 'AI_synthetic_candidate_requires_visual_review': raise ValueError('invalid synthetic source/split/status')
        validate_source(synth, row['person_image'], row['person_label'])
        for obj in row['objects']:
            validate_source(synth, obj['full_mask'], obj['visible_mask'])
        names.append('synthetic_'+row['image'])
    for row in reviewed:
        validate_source(team,row['image'],row['label'])
        names.append('team_'+Path(row['image']).name)
    if len(set(names)) != len(names) or len(names) != 3500: raise ValueError('output name collision or count')
    out.mkdir(parents=True,exist_ok=False)
    (out/'images').mkdir(); (out/'labels').mkdir()
    records=[]
    try:
        for kind, base, rows in [('synthetic',synth,selected),('team',team,reviewed)]:
            for row in rows:
                img = row['person_image'] if kind == 'synthetic' else row['image']
                label = row['person_label'] if kind == 'synthetic' else row['label']
                name = kind+'_'+Path(img).name
                dest_image = Path('images')/name
                dest_label = Path('labels')/(Path(name).stem+'.txt')
                shutil.copyfile(base/img,out/dest_image)
                shutil.copyfile(base/label,out/dest_label)
                records.append({'image':dest_image.as_posix(),'label':dest_label.as_posix(),
                                'source_kind':kind,'source_root':base.relative_to(ROOT).as_posix(),
                                'source_record':row,'scenario':'single' if kind=='team' else row['scenario']['kind'],
                                'box_semantics':'unoccluded pasted-person rectangle, approximate visible box' if kind=='team' else 'visible mask box',
                                'source_group':'scene_video_49s_shared','split':'train_candidate',
                                'status':'AI_review_candidate_not_ground_truth'})
        (out/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf8')
        excluded = exclusions(singles,selected,team_all)
        (out/'excluded.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in excluded),encoding='utf8')
        result={'status':'complete_candidates','count':len(records),'source_counts':dict(Counter(r['source_kind'] for r in records)),
                'scenario_counts':dict(Counter(r['scenario'] for r in records)), 'selection_seed':args.seed,
                'purpose':'person detection only, training candidates; not ground truth or independent evaluation',
                'source_group':'scene_video_49s_shared','team_rectangle_note':'unoccluded rectangle vs estimated alpha-visible box; parameters rounded, all 29 estimated IoU >= 0.9705',
                'exclusion_note':'only 29 individually reviewed team images admitted; other team samples and gray previews not selected'}
        (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps(result,ensure_ascii=False))
    except BaseException as exc:
        (out/'FAILED.txt').write_text(f'{type(exc).__name__}: {exc}\n',encoding='utf8')
        raise


if __name__ == '__main__': main()
