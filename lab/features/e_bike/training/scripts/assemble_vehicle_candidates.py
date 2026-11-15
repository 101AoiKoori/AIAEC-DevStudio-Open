"""Create a portable, source-audited entry point for candidate e-bike splits."""

if __name__ == "__main__":
    raise SystemExit("Legacy dataset CLI is retired; use datasets/scripts/e_bike/. Production remains paused.")

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
EXPORTS = ROOT / 'datasets/exports/e_bike'
SPLITS = {
    'train': 'real_scene_v4_candidates_4000_20261110',
    'val': 'field_validation_candidates_200_20261110',
    'test': 'camera_test_candidates_200_20261110',
}


def gather(name):
    directory = EXPORTS / name
    summary = json.loads((directory/'summary.json').read_text(encoding='utf8'))
    records = [json.loads(line) for line in (directory/'manifest.jsonl').read_text(encoding='utf8').splitlines()]
    if summary['status'] != 'complete_candidates' or len(records) != summary['completed']:
        raise ValueError(f'incomplete candidates: {name}')
    files, sources = [], set()
    for record in records:
        image = (directory/record['image']).resolve()
        label = (directory/record['label']).resolve()
        if not image.is_relative_to(EXPORTS) or not label.is_relative_to(EXPORTS) or not image.is_file() or not label.is_file():
            raise ValueError(f'missing or unsafe image/label: {record["image"]}')
        if len(label.read_text(encoding='utf8').splitlines()) != len(record['objects']):
            raise ValueError(f'label count mismatch: {record["image"]}')
        files.append(image)
        bg = record['background']
        sources.update((bg['image'], bg['source_video']))
        sources.update(bg.get('person_sources', []))
        for item in record['objects']:
            source = item['mask_source']
            sources.update((source['path'], source['source_frame'], source['source_video']))
    return files, sources, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = (ROOT/args.output).resolve()
    if not out.is_relative_to(EXPORTS):
        parser.error('output must be under datasets/exports/e_bike')
    gathered = {split: gather(name) for split, name in SPLITS.items()}
    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        overlap = gathered[left][1] & gathered[right][1]
        if overlap:
            raise ValueError(f'source paths overlap {left}/{right}: {sorted(overlap)[:3]}')
    media = [image for files, _, _ in gathered.values() for image in files]
    procedural = [EXPORTS / name / filename for name in SPLITS.values()
                  for filename in ('summary.json', 'manifest.jsonl')]
    procedural += [image.parent.parent / 'labels' / (image.stem + '.txt') for image in media]
    sys.path.insert(0, str(ROOT))
    from lab.training_common.production_guard import generation_guard
    operation = 'e_bike.assemble_vehicle_candidates:' + out.relative_to(ROOT).as_posix()
    with generation_guard(operation, list(dict.fromkeys(media)), [out], {'splits': SPLITS},
                          code_paths=[Path(__file__), ROOT / 'lab/training_common/production_guard.py'], procedural_inputs=list(dict.fromkeys(procedural)), root=ROOT) as run:
        if run.reused:
            return
        if out.exists():
            parser.error('output must be a new directory under datasets/exports/e_bike')
        out.mkdir(parents=True)
        # Omitting `path` makes Ultralytics resolve split directories against the
        # YAML's parent, provided the caller passes an absolute YAML filename.
        yaml = ''.join(f'{split}: ../{name}/images\n' for split, name in SPLITS.items())
        yaml += 'names:\n  0: upright\n  1: fallen\n'
        (out/'data.yaml').write_text(yaml, encoding='utf8')
        metadata = {'status': 'candidate_only', 'classes': {'upright': 0, 'fallen': 1},
                    'splits': {}, 'path_overlap': 'no shared source file paths',
                    'limits': 'Same model across splits; capture-session relationships and masks require review. '
                              'Validation/test are synthetic derivatives of very few source frames, not independent scenes.',
                    'source_groups': {'train': 'scene_video_49s_shared',
                                      'val': 'field_20261108_session_candidate',
                                      'test': 'vehicle_camera_pics_zip_session_uncertain'}}
        for split, name in SPLITS.items():
            files, sources, summary = gathered[split]
            # List files are relative to this index directory, independent of drive.
            rows = [f'../{name}/images/{p.name}' for p in files]
            (out/f'{split}.txt').write_text('\n'.join(rows)+'\n', encoding='utf8')
            metadata['splits'][split] = {'count': len(files), 'source_path_count': len(sources),
                                          'source_paths': sorted(sources),
                                          'underlying_summary': f'../{name}/summary.json',
                                          'sampling_limits': summary['limits']}
        (out/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
        print(json.dumps({split: len(gathered[split][0]) for split in SPLITS}, ensure_ascii=False))


if __name__ == '__main__':
    main()
