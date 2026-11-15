"""Allocate camera-region quotas before sampling concrete Blender states."""
import argparse
import copy
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
from datasets.scripts.common.bayesian import allocate  # noqa: E402
from datasets.scripts.e_bike import plan_blender_batch as base  # noqa: E402

PRIORS = {'left': (2, 3), 'center': (1, 3), 'right': (2, 3)}


def region(x):
    return 'left' if x < -10 else 'right' if x > 10 else 'center'


def make_plan(scene, seed=20261114):
    rng = random.Random(seed)
    # Build a finite pool of actual parameter states, grouped before allocation.
    candidates = defaultdict(list)
    signatures = set()
    template = None
    for turn in range(5):
        template = base.make_plan(scene, seed=seed + turn)
        for state in template['strata']:
            signature = json.dumps(state['parameters'], sort_keys=True)
            if signature in signatures:
                continue
            signatures.add(signature)
            candidates[(state['failure_event'], region(state['parameters']['camera_x_mm']))].append(state)
    design = base.read(base.DESIGN)['v4_combination_source_plan_20261114']['branches']
    output, evidence = [], []
    for branch in design:
        event = branch['combination_id']
        budget = branch['blender_parent_scene_quota'][scene]
        strata = []
        for key, (alpha, beta) in PRIORS.items():
            capacity = len(candidates[(event, key)])
            strata.append({'id': key, 'failure_event': event, 'min_count': 0, 'max_count': capacity,
                           'prior': {'alpha': alpha, 'beta': beta}, 'exposure_weight': 1,
                           'failure_cost': 1, 'conditions': ['camera_region:' + key]})
        request = {'schema_version': 1, 'model_key': 'e_bike', 'target_split': 'train',
                   'purpose': 'Allocate candidate camera regions from explicit design priors; not measured failure rates.',
                   'budget': budget, 'uncertainty_weight': 0,
                   'independence_basis': 'Regions share a fitted scene and foreground parents.', 'strata': strata}
        allocated = allocate(request, ROOT)
        reference = copy.deepcopy(request)
        for state in reference['strata']:
            state['prior'] = {'alpha': 1, 'beta': 1}
        equal = allocate(reference, ROOT)
        if [s['planned_count'] for s in allocated['strata']] == [s['planned_count'] for s in equal['strata']]:
            raise ValueError('design priors did not affect the allocation')
        evidence.append({'branch': event, 'request': request, 'result': allocated,
                         'equal_prior_reference': {s['id']: s['planned_count'] for s in equal['strata']}})
        for state in allocated['strata']:
            samples = rng.sample(candidates[(event, state['id'])], state['planned_count'])
            for sample in samples:
                sample = copy.deepcopy(sample)
                sample['id'] = f'state_{len(output)+1:06d}'
                output.append(sample)
    template['strata'] = output
    template['budget'] = len(output)
    template['allocation_evidence'] = {'method': 'Beta priors allocate camera-region quotas before concrete state sampling',
        'basis': 'Priors are design assumptions, not empirical failure probabilities; equal-prior counterfactual is included.',
        'branches': evidence}
    return template


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=base.SCENES, required=True)
    args = parser.parse_args()
    plan = make_plan(args.scene)
    print(json.dumps(plan, ensure_ascii=False, indent=2))



if __name__ == '__main__':
    main()
