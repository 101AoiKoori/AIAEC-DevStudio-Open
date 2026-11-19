"""Allocate a bounded image budget over explicitly feasible strata.

This produces a production *plan*, never samples or quality claims. A Beta
posterior describes an assumed group-level failure probability, not a model
accuracy guarantee. Validation observations may adjust training quotas only;
validation and test quotas are prior-only and test observations are forbidden.
"""
import argparse
import json
import math
from pathlib import Path
import platform
import random
import sys

import scipy
from scipy.special import betaincinv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODELS = frozenset(('person', 'e_bike', 'fire_sticker', 'building_name_paper',
                    'sign', 'waste_bin_state', 'waste_sticker', 'meter', 'plate_detection'))
MAX_STATES = 100_000
MAX_BUDGET = 10_000_000
MAX_OBSERVATIONS = 100_000
MAX_JSON_BYTES = 64 * 1024 * 1024


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    return value


def _positive(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    if (value < 0 if zero else value <= 0):
        raise ValueError(f'{name} must be {"nonnegative" if zero else "positive"}')
    if value > 1_000_000 or (value and value < 0.000001):
        raise ValueError(f'{name} is outside the supported numeric range')
    return float(value)


def _workspace_path(root, value):
    value = _text(value, 'path')
    path = (root / value).resolve()
    if not path.is_relative_to(root) or path == root or path.relative_to(root).as_posix() != value:
        raise ValueError('Expected canonical workspace-relative path: ' + value)
    return path


def _evidence(root, path, cache):
    """Read a human-confirmed or traceable machine group report, never a test result."""
    resolved = _workspace_path(root, path)
    runs = (root / 'runs').resolve()
    if (not resolved.is_relative_to(runs) or resolved.suffix.lower() != '.json' or not resolved.is_file() or
            any(part.lower() == 'test' for part in resolved.relative_to(runs).parts)):
        raise ValueError('Group evidence must be an existing JSON report under runs')
    if resolved.stat().st_size > MAX_JSON_BYTES:
        raise ValueError('Group evidence exceeds the input size bound')
    if path not in cache:
        data = json.loads(resolved.read_text(encoding='utf-8-sig'))
        if not isinstance(data, dict) or data.get('schema_version') != 1 or data.get('split') != 'val':
            raise ValueError('Group evidence must be validation evidence; test is forbidden')
        origin = data.get('evidence_origin', 'human')
        if origin == 'machine':
            machine = data.get('machine_evidence')
            if not isinstance(machine, dict):
                raise ValueError('Machine evidence needs a method and result reference')
            _text(machine.get('method'), 'machine evidence method')
            _text(machine.get('result_reference'), 'machine evidence result_reference')
        elif origin != 'human' or data.get('review_status') != 'confirmed':
            raise ValueError('Human validation evidence must be confirmed')
        groups = data.get('groups')
        if not isinstance(groups, list) or not groups or len(groups) > MAX_OBSERVATIONS:
            raise ValueError('Reviewed evidence must contain groups')
        indexed = {}
        for group in groups:
            if not isinstance(group, dict):
                raise ValueError('Group evidence entry must be a mapping')
            group_id = _text(group.get('group_id'), 'group_id')
            if group_id in indexed:
                raise ValueError('Evidence group needs unique ID')
            indexed[group_id] = group
        _text(data.get('model_key'), 'evidence model_key')
        _text(data.get('state_id'), 'evidence state_id')
        _text(data.get('failure_event'), 'evidence failure_event')
        _text(data.get('grouping_basis'), 'evidence grouping_basis')
        cache[path] = (data, indexed)
    return cache[path]


def _weighted_extras(capacities, weights, count, ids):
    """Capped proportional apportionment with deterministic largest remainder."""
    if not count:
        return [0] * len(capacities)
    # A linear bisection over [0, max(capacity/weight)] loses all precision
    # when the permitted weights span many orders of magnitude.  Search in
    # logarithmic scale and retain the underestimate before apportionment.
    low = math.log(count) - math.log(math.fsum(weights)) - 2.0
    high = math.log(max(capacity / weight for capacity, weight in zip(capacities, weights) if capacity))
    for _ in range(120):
        middle = (low + high) / 2
        scale = math.exp(middle)
        if math.fsum(min(capacity, scale * weight) for capacity, weight in zip(capacities, weights)) < count:
            low = middle
        else:
            high = middle
    scale = math.exp(low)
    exact = [min(capacity, scale * weight) for capacity, weight in zip(capacities, weights)]
    extras = [min(capacity, math.floor(value)) for capacity, value in zip(capacities, exact)]
    left = count - sum(extras)
    order = sorted(range(len(ids)), key=lambda index: (-(exact[index] - extras[index]), ids[index]))
    for index in order:
        if not left:
            break
        if extras[index] < capacities[index]:
            extras[index] += 1
            left -= 1
    if left or sum(extras) != count:
        raise ValueError('Cannot allocate exact integer budget')
    return extras


def _conditional_extras(result, states, count, specification):
    """Balance declared categorical axes while sampling only feasible states.

    Symmetric Dirichlet pseudocounts smooth inverse-coverage weights. Those
    weights are a design policy, not observations about real-world frequency.
    An optional stratifying axis distributes remaining capacity to its least
    represented value first, so attainable strata differ by at most one when
    they have equal capacity and minimums.
    """
    if not isinstance(specification, dict):
        raise ValueError('conditional_balance must be an object')
    axes = specification.get('axes')
    if (not isinstance(axes, list) or not axes or
            any(not isinstance(axis, str) or not axis.strip() for axis in axes) or
            len(axes) != len(set(axes))):
        raise ValueError('conditional_balance axes must be unique nonempty names')
    group_axis = specification.get('stratify_by')
    if group_axis is not None and group_axis not in axes:
        raise ValueError('stratify_by must name a balancing axis')
    secondary_axis = specification.get('secondary_stratify_by')
    if secondary_axis is not None and (secondary_axis not in axes or secondary_axis == group_axis):
        raise ValueError('secondary_stratify_by must name a different balancing axis')
    prior = _positive(specification.get('prior', 1), 'conditional_balance prior')
    seed = specification.get('seed', 0)
    if type(seed) is not int:
        raise ValueError('conditional_balance seed must be an integer')

    state_axes = []
    totals = {axis: {} for axis in axes}
    for state, row in zip(states, result):
        values = state.get('axes')
        if not isinstance(values, dict) or any(axis not in values for axis in axes):
            raise ValueError('every balanced state requires the declared categorical axes')
        for axis in axes:
            value = values[axis]
            if (isinstance(value, (dict, list)) or value is None or
                    not isinstance(value, (str, int, float, bool)) or
                    isinstance(value, float) and not math.isfinite(value)):
                raise ValueError('balanced axis values must be finite JSON scalars')
            totals[axis][value] = totals[axis].get(value, 0) + row['min_count']
        state_axes.append(values)

    conditional = {axis: {} for axis in axes if axis != group_axis}
    if group_axis is not None:
        for values, row in zip(state_axes, result):
            group = values[group_axis]
            for axis in conditional:
                key = (group, values[axis])
                conditional[axis][key] = conditional[axis].get(key, 0) + row['min_count']

    remaining = [row['max_count'] - row['min_count'] for row in result]
    extras = [0] * len(result)
    rng = random.Random(seed)
    for _ in range(count):
        eligible = [index for index, capacity in enumerate(remaining) if capacity]
        if group_axis is not None:
            minimum = min(totals[group_axis][state_axes[index][group_axis]] for index in eligible)
            eligible = [index for index in eligible
                        if totals[group_axis][state_axes[index][group_axis]] == minimum]
        if secondary_axis is not None:
            minimum = min(totals[secondary_axis][state_axes[index][secondary_axis]] for index in eligible)
            eligible = [index for index in eligible
                        if totals[secondary_axis][state_axes[index][secondary_axis]] == minimum]
        weights = []
        for index in eligible:
            values = state_axes[index]
            weight = result[index]['allocation_weight']
            for axis in axes:
                weight /= prior + totals[axis][values[axis]]
                if group_axis is not None and axis != group_axis:
                    weight /= prior + conditional[axis].get((values[group_axis], values[axis]), 0)
            weights.append(weight)
        chosen = rng.choices(eligible, weights=weights, k=1)[0]
        remaining[chosen] -= 1
        extras[chosen] += 1
        values = state_axes[chosen]
        for axis in axes:
            totals[axis][values[axis]] += 1
            if group_axis is not None and axis != group_axis:
                key = (values[group_axis], values[axis])
                conditional[axis][key] = conditional[axis].get(key, 0) + 1
    summary = {axis: [{'value': value, 'count': number} for value, number in counts.items()]
               for axis, counts in totals.items()}
    balance_summary = {'axes': list(axes), 'stratify_by': group_axis, 'prior': prior, 'seed': seed,
                       'counts': summary}
    if group_axis is not None:
        # Report only combinations declared in the input states. Several states
        # can share one cell; include cells assigned zero and their capacity.
        cross_counts = {}
        for axis, counts in conditional.items():
            cells = {}
            for values, row in zip(state_axes, result):
                key = (values[group_axis], values[axis])
                cells[key] = cells.get(key, 0) + row['max_count']
            cross_counts[axis] = [
                {'group_value': group, 'value': value, 'count': counts[(group, value)],
                 'capacity': capacity, 'group_count': totals[group_axis][group]}
                for (group, value), capacity in cells.items()
            ]
        balance_summary['cross_counts'] = cross_counts
    if secondary_axis is not None:
        balance_summary['secondary_stratify_by'] = secondary_axis
    return extras, balance_summary


def allocate(request, root=ROOT):
    """Return bounded per-state quotas from sparse states and validation groups."""
    root = Path(root).resolve()
    if not isinstance(request, dict) or request.get('schema_version') != 1:
        raise ValueError('Expected schema_version 1')
    model = request.get('model_key')
    split = request.get('target_split')
    if model not in MODELS or split not in ('train', 'val', 'test', 'unassigned'):
        raise ValueError('Unknown model or target split')
    purpose = _text(request.get('purpose'), 'purpose')
    budget = request.get('budget')
    if type(budget) is not int or not 0 < budget <= MAX_BUDGET:
        raise ValueError('Budget must be a positive integer within the resource bound')
    uncertainty = _positive(request.get('uncertainty_weight'), 'uncertainty_weight', zero=True)
    grouping = _text(request.get('independence_basis'), 'independence_basis')
    states = request.get('strata')
    if not isinstance(states, list) or not states or len(states) > MAX_STATES:
        raise ValueError('Provide a nonempty bounded list of feasible strata')
    cache, result, ids, observation_count = {}, [], set(), 0
    for state in states:
        if not isinstance(state, dict):
            raise ValueError('Stratum must be a mapping')
        state_id = _text(state.get('id'), 'stratum id')
        if state_id in ids:
            raise ValueError('Duplicate stratum id: ' + state_id)
        ids.add(state_id)
        failure_event = _text(state.get('failure_event'), 'failure_event')
        conditions = state.get('conditions', [])
        if (not isinstance(conditions, list) or
                any(not isinstance(value, str) or not value.strip() for value in conditions) or
                len(conditions) != len(set(conditions))):
            raise ValueError('Conditions must be a list of unique nonempty strings')
        low, high = state.get('min_count'), state.get('max_count')
        if type(low) is not int or type(high) is not int or not 0 <= low <= high <= MAX_BUDGET:
            raise ValueError('Invalid stratum minimum/maximum')
        prior = state.get('prior')
        if not isinstance(prior, dict):
            raise ValueError('Each stratum requires an explicit Beta prior')
        alpha = _positive(prior.get('alpha'), 'prior alpha')
        beta = _positive(prior.get('beta'), 'prior beta')
        exposure = _positive(state.get('exposure_weight'), 'exposure_weight')
        cost = _positive(state.get('failure_cost'), 'failure_cost')
        observations = state.get('observations', [])
        if not isinstance(observations, list):
            raise ValueError('Observations must be a list')
        observation_count += len(observations)
        if observation_count > MAX_OBSERVATIONS:
            raise ValueError('Observations exceed the resource bound')
        if split != 'train' and observations:
            raise ValueError('Validation/test/unassigned allocation is prior-only; observations are forbidden')
        groups, sources, reviewed = set(), set(), []
        failures = 0
        for observation in observations:
            if not isinstance(observation, dict):
                raise ValueError('Observation must be a mapping')
            origin = observation.get('evidence_origin', 'human')
            if (observation.get('split') != 'val' or origin not in ('human', 'machine') or
                    (origin == 'human' and observation.get('review_status') != 'confirmed')):
                raise ValueError('Only eligible validation observations are allowed; test is forbidden')
            evidence = _text(observation.get('evidence'), 'observation evidence')
            group_id = _text(observation.get('group_id'), 'observation group_id')
            if group_id in groups:
                raise ValueError('Duplicate group in stratum: ' + group_id)
            report, indexed = _evidence(root, evidence, cache)
            if origin != report.get('evidence_origin', 'human'):
                raise ValueError('Observation and evidence origin differ')
            if (report['model_key'] != model or report['state_id'] != state_id or
                    report['failure_event'] != failure_event or report['grouping_basis'] != grouping):
                raise ValueError('Evidence task, state, failure definition or grouping basis differs')
            if group_id not in indexed:
                raise ValueError('Observed group missing from reviewed evidence')
            group = indexed[group_id]
            group_ready = (group.get('result_status') == 'complete' if origin == 'machine'
                           else group.get('review_status') == 'confirmed')
            group_sources = group.get('source_ids')
            if (not group_ready or type(group.get('failed')) is not bool or
                    not isinstance(group_sources, list) or not group_sources or
                    any(not isinstance(item, str) or not item.strip() for item in group_sources) or
                    len(group_sources) != len(set(group_sources))):
                raise ValueError('Observed evidence group needs a complete binary result and source IDs')
            source_ids = set(group['source_ids'])
            if sources & source_ids:
                raise ValueError('Source is reused across groups within a stratum')
            groups.add(group_id)
            sources.update(source_ids)
            failures += int(group['failed'])
            entry = {'group_id': group_id, 'source_ids': sorted(source_ids),
                     'failed': group['failed'], 'evidence': evidence}
            if origin == 'machine':
                entry.update(evidence_origin='machine', machine_evidence=dict(report['machine_evidence']))
            reviewed.append(entry)
        n = len(reviewed)
        a, b = alpha + failures, beta + n - failures
        mean = a / (a + b)
        std = math.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1)))
        risk_weight = exposure * cost * mean
        uncertainty_component = exposure * cost * uncertainty * std
        weight = risk_weight + uncertainty_component
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError('Nonfinite or zero allocation weight')
        low95, high95 = float(betaincinv(a, b, .025)), float(betaincinv(a, b, .975))
        if not (math.isfinite(low95) and math.isfinite(high95)):
            raise ValueError('Cannot calculate Beta interval')
        has_machine = any(item.get('evidence_origin') == 'machine' for item in reviewed)
        result.append({'id': state_id, 'failure_event': failure_event,
            'conditions': conditions, 'min_count': low, 'max_count': high,
            'prior': {'alpha': alpha, 'beta': beta}, 'observation_status':
                ('validation_groups' if has_machine else 'reviewed_groups') if n else 'prior_only',
            'observed_groups': n, 'failed_groups': failures, 'group_observations': reviewed,
            'posterior': {'alpha': a, 'beta': b, 'mean': mean, 'std': std, 'equal_tail_95': [low95, high95],
                          'interpretation': (('validation_groups_assuming_independence' if has_machine else
                                              'reviewed_validation_groups_assuming_independence') if n else
                                             'prior_only_not_observed_or_test_performance')},
            'exposure_weight': exposure, 'failure_cost': cost, 'risk_weight': risk_weight,
            'uncertainty_component': uncertainty_component, 'allocation_weight': weight})
    required = sum(row['min_count'] for row in result)
    capacity = sum(row['max_count'] for row in result)
    if not required <= budget <= capacity:
        raise ValueError(f'Budget {budget} outside minimum/maximum coverage [{required}, {capacity}]')
    balance = request.get('conditional_balance')
    balance_summary = None
    if balance is None:
        extras = _weighted_extras([r['max_count'] - r['min_count'] for r in result],
                                  [r['allocation_weight'] for r in result], budget - required,
                                  [r['id'] for r in result])
    else:
        extras, balance_summary = _conditional_extras(result, states, budget - required, balance)
    for row, extra in zip(result, extras):
        row['planned_count'] = row['min_count'] + extra
        row['planned_fraction'] = row['planned_count'] / budget
    output = {'schema_version': 1, 'status': 'planned_not_produced', 'purpose': purpose, 'model_key': model,
            'target_split': split, 'budget': budget, 'planned_total': sum(r['planned_count'] for r in result),
            'environment': {'python': platform.python_version(), 'scipy': scipy.__version__},
            'independence_basis': grouping, 'uncertainty_weight': uncertainty,
            'formula': 'weight = exposure_weight * failure_cost * (posterior_mean + uncertainty_weight * posterior_std); '
                       'minimums first, remainder capped-proportional with deterministic largest remainders',
            'scope': 'Quotas and planned fractions are production heuristics, not state occurrence probabilities, '
                     'data, measured coverage or a quality guarantee. '
                     'Prior-only states have no observations. Group independence is an assumption, not proved by IDs; '
                     'groups can overlap across strata. Test performance is never inferred here. '
                     'Small floating-point differences across library versions are possible.',
            'strata': result}
    if balance_summary is not None:
        output['conditional_balance'] = balance_summary
        output['formula'] = ('Beta risk weight times inverse-smoothed axis and within-stratum coverage; '
                             'least-represented feasible stratum first when specified')
        for row, state in zip(result, states):
            row['axes'] = dict(state['axes'])
    return output


def plan_generation(request, root=ROOT):
    """Allocate concrete declared generation states without model feedback.

    Constraints and state fields are declarations to be implemented by the
    caller, not proof that a renderer applied them or that pixels satisfy them.
    """
    try:
        declaration = json.loads(json.dumps(request, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError('generation declaration must be finite JSON') from error
    if not isinstance(declaration, dict):
        raise ValueError('generation declaration must be an object')
    if not isinstance(declaration.get('constraints'), dict) or not declaration['constraints']:
        raise ValueError('generation constraints must be a nonempty object')
    states = declaration.get('strata')
    if not isinstance(states, list) or not states:
        raise ValueError('generation requires explicit feasible strata')
    for state in states:
        if not isinstance(state, dict):
            raise ValueError('generation state must be an object')
        if state.get('observations', []):
            raise ValueError('generation planning does not accept model feedback observations')
        for name in ('background', 'target', 'labels', 'augmentation', 'parameters'):
            if not isinstance(state.get(name), dict) or not state[name]:
                raise ValueError(f'generation state {name} must be a nonempty object')
        sources = state.get('sources')
        if (not isinstance(sources, list) or not sources or
                any(not isinstance(item, str) or not item.strip() for item in sources) or
                len(sources) != len(set(sources))):
            raise ValueError('generation state sources must be unique nonempty strings')
    result = allocate(declaration, root)
    for planned, state in zip(result['strata'], states):
        for name in ('background', 'target', 'labels', 'augmentation', 'sources', 'parameters'):
            planned[name] = state[name]
    result['constraints'] = declaration['constraints']
    # Keep the provenance of the design the planner consulted. This does not
    # imply that the declared strata satisfy the design's contrasts.
    if 'design_context' in declaration:
        if not isinstance(declaration['design_context'], dict):
            raise ValueError('design context must be an object')
        result['design_context'] = declaration['design_context']
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv == ['--plan-stdin']:
        # For interpreters such as Blender's that cannot import SciPy. This
        # path produces no files and uses the same allocator as direct calls.
        try:
            source = sys.stdin.buffer.read(MAX_JSON_BYTES + 1)
            if len(source) > MAX_JSON_BYTES:
                raise ValueError('Allocation request exceeds the input size bound')
            request = json.loads(source.decode('utf-8-sig'))
            result = plan_generation(request)
            sys.stdout.write(json.dumps(result, ensure_ascii=True, allow_nan=False))
            return 0
        except (ValueError, TypeError, KeyError, OSError) as error:
            print(f'{type(error).__name__}: {error}', file=sys.stderr)
            return 1
    from lab.training_common.production_guard import generation_guard

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Workspace-relative JSON request')
    parser.add_argument('--output', required=True, help='Workspace-relative JSON plan under runs/')
    args = parser.parse_args(argv)
    try:
        source = _workspace_path(ROOT, args.input)
        output = _workspace_path(ROOT, args.output)
        if not output.is_relative_to((ROOT / 'runs').resolve()) or output.suffix.lower() != '.json':
            raise ValueError('Output must be a JSON file under runs/')
        if source.stat().st_size > MAX_JSON_BYTES:
            raise ValueError('Allocation request exceeds the input size bound')
        request = json.loads(source.read_text(encoding='utf-8-sig'))
        evidence = {_workspace_path(ROOT, item['evidence']) for state in request.get('strata', [])
                    for item in state.get('observations', []) if isinstance(item, dict) and item.get('evidence')}
        inputs = [source, *sorted(evidence)]
        with generation_guard('bayesian_allocation:' + args.output, inputs, [output],
                              {'request': request, 'scipy': scipy.__version__},
                              code_paths=[Path(__file__)], procedural_inputs=inputs, root=ROOT) as run:
            if run.reused:
                print('ALLOCATION_REUSED', output, flush=True)
                return 0
            output.parent.mkdir(parents=True, exist_ok=True)
            try:
                result = allocate(request, ROOT)
                result['input'] = args.input
            except BaseException as exc:
                with output.open('w', encoding='utf-8') as stream:
                    json.dump({'status': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', 'input': args.input,
                               'error': f'{type(exc).__name__}: {exc}'}, stream, ensure_ascii=False, indent=2)
                    stream.write('\n')
                raise
            with output.open('w', encoding='utf-8') as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write('\n')
        print('ALLOCATION_PLAN', output, flush=True)
        return 0
    except KeyboardInterrupt:
        print('INTERRUPTED: inspect the recorded allocation status', file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
