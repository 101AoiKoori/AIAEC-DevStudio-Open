"""List base records to isolate before combining close-view sticker candidates."""
import argparse
import json
from pathlib import Path
import sys
from collections import Counter, defaultdict

import composite_bin_real_scene as base
sys.path.insert(0, str(base.ROOT))
from lab.training_common.production_guard import generation_guard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--supplement', required=True)
    args = parser.parse_args()
    source, supplement = (base.ROOT/args.base).resolve(), (base.ROOT/args.supplement).resolve()
    procedural = [source / 'manifest.json', supplement / 'manifest.json',
                  supplement / 'foreground_isolation.json']
    destination = supplement / 'merge_isolation_proposal.json'
    with generation_guard('check_sticker_merge_v4:' + destination.relative_to(base.ROOT).as_posix(),
                          [], [destination], {}, code_paths=[Path(__file__), Path(base.__file__)],
                          procedural_inputs=procedural, root=base.ROOT) as run:
        if run.reused:
            print('Existing merge proposal reused:', destination)
            return
        return _check(source, supplement)


def _check(source, supplement):
    original = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    extra = json.loads((supplement/'manifest.json').read_text(encoding='utf-8'))
    blocked = json.loads((supplement/'foreground_isolation.json').read_text(encoding='utf-8'))
    blocked_colors = defaultdict(list)
    for node in blocked:
        blocked_colors[(node['split'],node['source_render'])].append(node)
    isolated = []
    for item in original['records']:
        matches = [node for node in blocked_colors[(item['split'],item.get('source_render'))]
                   if node['color'] in item['bin_states']]
        if matches:
            isolated.append({'id':item['id'],'split':item['split'],'source_render':item['source_render'],
                'matched_foreground_groups':[{'color':node['color'],'state':node['state'],
                    'group_kept_in':node['group_kept_in']} for node in matches]})
    seen_bg, seen_render = defaultdict(set), defaultdict(set)
    for row in original['records']:
        seen_bg[row['split']].add(row['background']['parent_image'])
        if row.get('source_render'):
            seen_render[row['split']].add(row['source_render'])
    for row in extra['records']:
        seen_bg[row['split']].add(row['background']['parent_image'])
        seen_render[row['split']].add(row['source_render'])
    exact = {}
    for first,second in (('train','val'),('train','test'),('val','test')):
        exact[first+'|'+second] = {'background':sorted(seen_bg[first]&seen_bg[second]),
                                  'render':sorted(seen_render[first]&seen_render[second])}
    output = {'status':'proposed_isolation_not_applied',
        'base':source.relative_to(base.ROOT).as_posix(), 'supplement':supplement.relative_to(base.ROOT).as_posix(),
        'reason':'base frame contains a bin crop blocked by the supplemental cross-split foreground graph',
        'scope':'conservative proposal includes blocked crops even when no linked retained crop was sampled; threshold is not calibrated',
        'isolated_base_records':isolated, 'isolated_count_by_split':dict(Counter(x['split'] for x in isolated)),
        'exact_cross_split_sources':exact}
    (supplement/'merge_isolation_proposal.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'base_records_to_isolate':len(isolated),'by_split':output['isolated_count_by_split'],
        'exact_cross_split_sources':{k:{t:len(v) for t,v in vals.items()} for k,vals in exact.items()}},ensure_ascii=False))


if __name__ == '__main__':
    main()
