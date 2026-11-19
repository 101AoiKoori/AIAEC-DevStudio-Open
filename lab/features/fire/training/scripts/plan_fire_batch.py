"""Plan fire candidate batches from task-use relations without generating images."""
import argparse
import copy
import json
import random
import sqlite3
import sys
from collections import Counter
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lab.training_common.production_guard import plan_production  # noqa: E402
from datasets.scripts.common.blender_workspace import WORKSPACE_BLEND
from fire_blender_labels import explicit_scenario  # noqa: E402

BASE = Path('datasets/annotated/fire/fire_sticker_v4_arena_handoff_20261113')
ARTWORK = 'Archive/Assets/working_sources/digital/fire/team_assets/image1.png'
RECORDED_SCENE = 'datasets/assets/blender/fire/blender_buildings_v13/three_buildings.blend'
SCENE = WORKSPACE_BLEND.relative_to(ROOT).as_posix()


def matching_relations(records, task, split, role):
    """Match all requested dimensions in the same use object."""
    found = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get('task_use_status'), list):
            raise ValueError('image use records must contain a relation list')
        for use in record['task_use_status']:
            if not isinstance(use, dict):
                raise ValueError('image use relation must be an object')
            if (use.get('task'), use.get('split'), use.get('role')) == (task, split, role):
                found.append((record, use))
    return found


def workspace_file(value, root=ROOT):
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError('expected workspace-relative POSIX path')
    lexical = Path(value)
    if lexical.is_absolute() or any(p in ('', '.', '..') for p in value.split('/')):
        raise ValueError('invalid workspace path')
    file = (root / lexical).resolve()
    if root.resolve() not in file.parents or not file.is_file():
        raise ValueError('source is missing or outside workspace: ' + value)
    return file


def registered_scene(scene, registry, root=ROOT):
    """Check the Blender asset record read-only; a scene is not an image."""
    file = workspace_file(scene, root)
    registry_file = workspace_file(registry, root)
    with closing(sqlite3.connect(registry_file.as_uri() + '?mode=ro', uri=True)) as db:
        row = db.execute('SELECT current_path,classification,lifecycle_state,truth_level '
                         'FROM assets WHERE current_path=?', (scene,)).fetchone()
    if not row or row[:3] != (scene, 'editable_blender_workspace', 'active_in_use'):
        raise ValueError('Blender scene registration is missing or not active')
    return {'path': file.relative_to(root).as_posix(), 'classification': row[1],
            'lifecycle_state': row[2], 'truth_level': row[3], 'registry': registry}


def build_batch(selection, uses, single_c, seed=431, root=ROOT):
    if type(seed) is not int or not 0 <= seed <= 2_147_483_647:
        raise ValueError('seed must be an integer from 0 to 2147483647')
    rows = uses.get('records')
    if not isinstance(rows, list):
        raise ValueError('image-use records are missing')
    matches = matching_relations(rows, 'fire_sticker', 'train', 'fixed_pattern')
    ready = [(row, use) for row, use in matches if use.get('status') == 'ready']
    if len(ready) != 1 or ready[0][0].get('path') != ARTWORK:
        raise ValueError('the unique official ready training pattern is missing or ambiguous')
    artwork = workspace_file(ready[0][0]['path'], root).relative_to(root).as_posix()
    assets = selection['current_source_selection_20261114']['background_candidates']['blender']
    libraries = [asset for asset in assets if asset.get('id') == 'building_library']
    if len(libraries) != 1 or libraries[0].get('path') not in (RECORDED_SCENE, SCENE):
        raise ValueError('the planned building model source is missing or ambiguous')
    scene = registered_scene(SCENE, 'datasets/blender_source_registry.sqlite3', root)
    original = selection['current_source_selection_20261114']['v4_combination_plan']
    declaration = copy.deepcopy(original['entry_ready_plan']['declaration'])
    states = [state for state in declaration['strata']
              if not (state['target'].get('active_buildings') == ['C'] and
                      state['target'].get('fire_counts') == {'C': 1})]
    replacements = copy.deepcopy(single_c['strata'])
    if (single_c.get('model_key') != 'fire_sticker' or single_c.get('target_split') != 'train' or
            single_c.get('budget') != 110 or len(replacements) != 6 or
            any(s['target'].get('active_buildings') != ['C'] or s['target'].get('fire_counts') != {'C': 1}
                or s['target'].get('placement') not in ('c_supported_left', 'c_supported_right')
                for s in replacements)):
        raise ValueError('bounded C single-sticker input is not the expected plan')
    states.extend(replacements)
    for state in states:
        explicit_scenario(state['target'])
        if state['parameters']['scenario_index'] is not None:
            raise ValueError('unsupported non-explicit scenario in source plan')
        state['parameters']['seed'] = seed
        state['parameters']['device'] = 'OPTIX'
    random.Random(seed).shuffle(states)
    if len({s['id'] for s in states}) != len(states):
        raise ValueError('duplicate renderer state')
    declaration.update(budget=900, strata=states,
        purpose='V4 fire Blender candidate schedule; generated labels remain candidates',
        independence_basis='one approximate building model family; fixed official artwork')
    planned = plan_production(declaration, root)
    if planned['planned_total'] != 900:
        raise ValueError('renderer allocation did not reach 900')
    actual = Counter()
    for state in planned['strata']:
        target = state['target']
        if len(target['active_buildings']) != 1:
            raise ValueError('unexpected multi-building state')
        building = target['active_buildings'][0]
        count = target['fire_counts'][building]
        kind = 'zero' if count == 0 else 'single' if count == 1 else 'multi'
        actual[f'blender_{building}_{kind}'] += state['planned_count']
    leaves = original['branch_plans']
    expected = {b['combination_id']: b['quota_images'] for b in leaves
                if b['combination_id'].startswith('blender_')}
    if dict(actual) != expected or sum(expected.values()) != 900:
        raise ValueError('renderer states do not match the existing branch quotas')
    report_leaves = []
    for b in leaves + original['additional_branches']:
        target = b['quota_images']
        arranged = actual.get(b['combination_id'], 0)
        if arranged:
            reason = None
        elif target == 0:
            reason = 'No separate image quota; overlap requires its own supported state and label handling.'
        elif b['combination_id'] == 'direct_signal_negative':
            reason = 'The selected signal image has no accepted full-frame fire-negative label and is not an allowed signal-export input.'
        elif b['combination_id'].startswith('direct_'):
            reason = 'Current selected direct images do not supply complete task labels for this train branch; regression/reserved inputs remain excluded.'
        else:
            reason = 'Current photo compositor requires whole-frame negative truth and source-compatible input; selected candidate backgrounds do not supply it.'
        report_leaves.append({'combination_id': b['combination_id'], 'target_images': target,
                              'arranged_attempts': arranged, 'gap': target - arranged,
                              'blocking_condition': reason})
    use_counts = {}
    for role in ('direct_sample', 'background', 'hard_negative', 'fixed_pattern'):
        use_counts[role] = dict(Counter(use.get('status', 'unknown') for _, use in
                                        matching_relations(rows, 'fire_sticker', 'train', role)))
    report = {'schema_version': 1, 'model_key': 'fire_sticker', 'target_split': 'train',
        'seed': seed, 'plan_status': 'allocated_attempts_not_generated_or_accepted',
        'source_selection': {'task': 'fire_sticker', 'split': 'train', 'role': 'fixed_pattern',
            'status': 'ready', 'selected_path': artwork, 'relation_evidence': ready[0][1].get('evidence'),
            'scene_parent': scene, 'scene_is_image': False, 'parent_share': {'building_library': 900},
            'reallocation': 'none; the Blender branch already specifies the single model family',
            'role_counts': use_counts},
        'randomization': {'state_order': 'seeded shuffle', 'seed': seed,
            'within_state': 'renderer uses seed + sample index; seeded slot, view and lighting variations',
            'sampling': 'with replacement across attempts; one fixed pattern and one scene parent',
            'independence': 'repeated variants are not independent parents or guaranteed distinct outputs'},
        'renderer_plan': {'states': len(states), 'attempts': 900, 'device': 'OPTIX',
            'source_scene': SCENE, 'entry': 'datasets/scripts/fire/render_blender_fire_dataset.py',
            'labels': 'candidate labels; unresolved or truncated targets suppress output'},
        'leaves': report_leaves, 'target_total': sum(x['target_images'] for x in report_leaves),
        'arranged_total': sum(x['arranged_attempts'] for x in report_leaves),
        'unfilled_total': sum(x['gap'] for x in report_leaves),
        'not_run': ['image_generation', 'frame_extraction', 'rendering', 'training'],
        'limits': ['The Blender scene and fixed artwork are shared parents, not 900 independent sources.',
                   'Actual valid-image and label counts are unknown until an authorized production run.',
                   'Unknown history is not itself used as an exclusion; reserved and regression relations are not selected for train.',
                   'The renderer explicit-state integration exists in the current working tree and is not fully committed.']}
    if (report['target_total'], report['arranged_total'], report['unfilled_total']) != (3000, 900, 2100):
        raise ValueError('branch totals differ from the existing V4 plan')
    return declaration, report


def main():
    parser = argparse.ArgumentParser(description='Build a fire batch plan without rendering')
    parser.add_argument('--uses', default=(BASE / 'fire_image_use_manifest_20261114.json').as_posix())
    parser.add_argument('--selection', default=(BASE / 'arena_source_and_generation_plan.json').as_posix())
    parser.add_argument('--single-c', default=(BASE / 'single_building_C_single_sticker_train_20261114.plan.json').as_posix())
    parser.add_argument('--seed', type=int, default=431)
    parser.add_argument('--output-plan', required=True)
    parser.add_argument('--output-report', required=True)
    args = parser.parse_args()
    sources = [workspace_file(value) for value in (args.selection, args.uses, args.single_c)]
    declaration, report = build_batch(*(json.loads(p.read_text(encoding='utf8')) for p in sources), seed=args.seed)
    destinations = []
    for value in (args.output_plan, args.output_report):
        p = Path(value)
        if p.is_absolute() or '\\' in value or any(x in ('', '.', '..') for x in value.split('/')):
            raise ValueError('output must be a canonical workspace-relative path')
        dest = (ROOT / p).resolve()
        if (ROOT / 'datasets/annotated/fire').resolve() not in dest.parents or dest.suffix != '.json':
            raise ValueError('output must be a JSON under the fire annotated directory')
        if dest.exists() or not dest.parent.is_dir():
            raise ValueError('output must be new and its existing parent must be a directory')
        destinations.append(dest)
    if len(set(destinations)) != 2:
        raise ValueError('plan and report destinations must differ')
    for dest, data in zip(destinations, (declaration, report)):
        with dest.open('x', encoding='utf8', newline='\n') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write('\n')
    print(json.dumps({'plan': args.output_plan, 'report': args.output_report,
                      'arranged_attempts': report['arranged_total'], 'gap': report['unfilled_total']}))


if __name__ == '__main__':
    main()
