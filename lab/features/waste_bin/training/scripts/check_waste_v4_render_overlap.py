"""Report cross-split render provenance and descriptive visual similarity."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]


def _generate():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset')
    args = parser.parse_args()
    path = (ROOT / args.dataset).resolve()
    report = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    groups = defaultdict(dict)
    for record in report['records']:
        if record.get('source_render'):
            groups[record['split']][record['source_render']] = record
    arrays = {}
    for records in groups.values():
        for name in records:
            image = cv2.imdecode(np.fromfile(str(ROOT / name), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError('missing rendered frame: ' + name)
            arrays[name] = cv2.resize(image, (80, 60), interpolation=cv2.INTER_AREA).astype(np.int16)
    comparisons = {}
    for first, second in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        common = sorted(groups[first].keys() & groups[second].keys())
        matching = 0
        minimum = None
        pair = None
        for left, a in groups[first].items():
            for right, b in groups[second].items():
                if a['bin_states'] != b['bin_states']:
                    continue
                matching += 1
                difference = float(np.mean(np.abs(arrays[left] - arrays[right])))
                if minimum is None or difference < minimum:
                    minimum, pair = difference, [left, right]
        comparisons[first+'|'+second] = {'same_render_paths': common, 'matching_visible_state_pairs': matching,
            'minimum_grayscale_mean_abs_difference_80x60': minimum, 'closest_pair': pair}
    result = {'scope': '80x60 grayscale distance is descriptive, not a validated near-duplicate threshold; shared Blender scene and trajectory family remain',
              'source_groups': {s: sorted({r['render_group'] for r in rec.values()}) for s, rec in groups.items()},
              'source_trajectory_ids_per_render_run': {s: sorted({r['render_trajectory'] for r in rec.values()})
                                                       for s, rec in groups.items()},
              'distinct_source_render_frames': {s: len(rec) for s, rec in groups.items()},
              'cross_split': comparisons}
    (path / 'render_overlap_check.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset')
    args = parser.parse_args()
    path = (ROOT / args.dataset).resolve()
    manifest = path / 'manifest.json'
    records = json.loads(manifest.read_text(encoding='utf-8'))['records']
    media = sorted({ROOT / record['source_render'] for record in records if record.get('source_render')})
    output = path / 'render_overlap_check.json'
    sys.path.insert(0, str(ROOT))
    from lab.training_common.production_guard import generation_guard
    with generation_guard('waste_bin.check_render_overlap:' + str(path.relative_to(ROOT)),
                          media, [output], {}, code_paths=[Path(__file__).resolve()],
                          procedural_inputs=[manifest], root=ROOT) as run:
        if not run.reused:
            _generate()


if __name__ == '__main__':
    main()
