"""Map allocated station states to explicit renderer parameters.

This module prepares a mapping only; a renderer must consume every returned
state and record each produced image through the shared production guard.
"""

if __name__ == "__main__":
    raise SystemExit("Task helper is import-only; use the primary entry in datasets/scripts/meter/. Production remains paused.")

import math


LIMITS = {
    'temperature_fraction': (0.0, 1.0),
    'humidity_fraction': (0.0, 1.0),
    'ground_angle_deg': (60.0, 70.0),
    'camera_z': (7.5, 13.5),
    'camera_x_offset': (-1.0, 1.0),
    'camera_yaw_deg': (-2.0, 2.0),
    'exposure_ev': (-0.5, 0.0),
    'saturation': (0.65, 1.0),
    'light_side': (-1.0, 1.0),
}
OCCLUSIONS = frozenset(('none', 'temperature_tip', 'humidity_tip', 'temperature_min'))
ARENA_LIMITS = {
    'temperature_fraction': (0., 1.), 'humidity_fraction': (0., 1.),
    'ground_angle_deg': (60., 70.), 'station_x': (-.75, -.45), 'station_y': (.05, .25),
    'camera_x': (-.70, -.50), 'camera_y': (-.83, -.67), 'camera_z': (.395, .445),
    'exposure_ev': (-.5, 0.), 'saturation': (.65, 1.),
}
ARENA_OPTIONAL = {'camera_lens_mm': (22., 40.)}


def validate_arena_declaration(declaration, *, count, sources, foreground):
    """Check the arena renderer's exact supported generation contract."""
    if not isinstance(declaration, dict) or declaration.get('model_key') != 'meter':
        raise ValueError('arena declaration must target meter')
    if declaration.get('target_split') != 'train':
        raise ValueError('arena production only supports train in this round')
    if type(count) is not int or count < 1 or declaration.get('budget') != count:
        raise ValueError('arena count must equal the positive plan budget')
    if (type(foreground) is not bool or not isinstance(declaration.get('constraints'), dict) or
            type(declaration['constraints'].get('foreground')) is not bool):
        raise ValueError('arena foreground constraint must be a boolean')
    if declaration.get('constraints') != {'resolution': [640, 480], 'label_policy': 'candidate_pose',
                                          'foreground': foreground}:
        raise ValueError('unsupported arena constraints')
    states = declaration.get('strata')
    if not isinstance(states, list) or not states:
        raise ValueError('arena declaration requires strata')
    seen = set()
    for state in states:
        if not isinstance(state, dict):
            raise ValueError('arena stratum must be an object')
        expected = {'background': {'kind': 'arena_scene'}, 'target': {'kind': 'dual_needle_meter'},
                    'labels': {'kind': 'candidate_pose'},
                    'augmentation': {'kind': 'explicit_arena_parameters'}}
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError('unsupported arena background, target, labels or augmentation')
        if state.get('sources') != list(sources) or state.get('conditions') != []:
            raise ValueError('arena sources must identify both scenes and conditions must use parameters')
        if type(state.get('max_count')) is not int or not 0 <= state['max_count'] <= 1:
            raise ValueError('each exact arena state has capacity at most one image')
        params = state.get('parameters')
        if (not isinstance(params, dict) or not set(ARENA_LIMITS).issubset(params) or
                set(params) - set(ARENA_LIMITS) - set(ARENA_OPTIONAL)):
            raise ValueError('arena parameters must contain the supported fields')
        for key, (low, high) in {**ARENA_LIMITS, **{k: v for k, v in ARENA_OPTIONAL.items() if k in params}}.items():
            value = params[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError('unsupported arena parameter: ' + key)
        identity = tuple(float(params[key]) for key in sorted(ARENA_LIMITS)) + (float(params.get('camera_lens_mm', 22.)),)
        if identity in seen:
            raise ValueError('duplicate exact arena parameters are not independent states')
        seen.add(identity)


def allocated_arena_states(plan, *, count):
    """Expand allocated arena states without fabricating repeated samples."""
    if not isinstance(plan, dict) or plan.get('model_key') != 'meter' or plan.get('budget') != count:
        raise ValueError('arena allocation does not match the requested budget')
    states = plan.get('strata')
    if not isinstance(states, list) or not states:
        raise ValueError('arena allocation has no strata')
    result, seen = [], set()
    for state in states:
        if not isinstance(state, dict):
            raise ValueError('arena allocation state must be an object')
        name, quota = state.get('id'), state.get('planned_count')
        if not isinstance(name, str) or not name.strip() or name in seen:
            raise ValueError('arena state ID must be unique and nonempty')
        seen.add(name)
        if type(quota) is not int or quota not in (0, 1):
            raise ValueError('each exact arena state can produce at most one image')
        if quota:
            result.append({'id': name, 'parameters': dict(state['parameters']), 'sources': list(state['sources'])})
    if len(result) != count:
        raise ValueError('arena allocation differs from budget')
    return result


def validate_station_declaration(declaration, *, count, source, allow_occlusion):
    """Reject state semantics this renderer cannot execute before opening a guard."""
    if not isinstance(declaration, dict) or declaration.get('model_key') != 'meter':
        raise ValueError('station declaration must target meter')
    if declaration.get('target_split') != 'train':
        raise ValueError('station production only supports train in this round')
    if declaration.get('budget') != count or type(count) is not int or count < 1:
        raise ValueError('station count must equal the positive plan budget')
    if declaration.get('constraints') != {'resolution': [640, 480], 'camera_fallback': 'reject',
                                           'label_policy': 'candidate_pose'}:
        raise ValueError('unsupported station constraints')
    states = declaration.get('strata')
    if not isinstance(states, list) or not states:
        raise ValueError('station declaration requires strata')
    parameter_sets = set()
    for state in states:
        if not isinstance(state, dict):
            raise ValueError('station stratum must be an object')
        expected = {'background': {'kind': 'station_scene'},
                    'target': {'kind': 'dual_needle_meter'},
                    'labels': {'kind': 'candidate_pose'},
                    'augmentation': {'kind': 'explicit_station_parameters'}}
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError('unsupported station background, target, labels or augmentation')
        if state.get('sources') != [source]:
            raise ValueError('station sources must name exactly the loaded blend')
        if state.get('conditions') != []:
            raise ValueError('station conditions must be expressed as supported parameters')
        if type(state.get('max_count')) is not int or not 0 <= state['max_count'] <= 1:
            raise ValueError('each exact station state has capacity at most one image')
        # Validate even a zero-quota state, before any output is created.
        candidate = dict(state, planned_count=1)
        allocated_station_states({'model_key': 'meter', 'budget': 1, 'strata': [candidate]},
                                 count=1, source=source, allow_occlusion=allow_occlusion)
        parameters = tuple(float(state['parameters'][key]) for key in sorted(LIMITS)) + (state['parameters']['occlusion'],)
        if parameters in parameter_sets:
            raise ValueError('duplicate exact station parameters are not independent states')
        parameter_sets.add(parameters)


def allocated_station_states(plan, *, count, source, allow_occlusion):
    """Return the exact state sequence; reject unsupported or ambiguous plans."""
    if not isinstance(plan, dict) or plan.get('model_key') != 'meter':
        raise ValueError('station plan must target meter')
    if type(count) is not int or count < 1 or plan.get('budget') != count:
        raise ValueError('station count must equal the positive plan budget')
    strata = plan.get('strata')
    if not isinstance(strata, list) or not strata:
        raise ValueError('station plan requires allocated strata')
    result, seen = [], set()
    for state in strata:
        if not isinstance(state, dict):
            raise ValueError('station stratum must be an object')
        name, quota = state.get('id'), state.get('planned_count')
        if not isinstance(name, str) or not name.strip() or name in seen:
            raise ValueError('station stratum ID must be unique and nonempty')
        seen.add(name)
        if type(quota) is not int or quota not in (0, 1):
            raise ValueError('each exact station state can produce at most one image')
        sources = state.get('sources')
        if not isinstance(sources, list) or source not in sources:
            raise ValueError('station source must be declared in each stratum')
        params = state.get('parameters')
        if not isinstance(params, dict) or set(params) != set(LIMITS) | {'occlusion'}:
            raise ValueError('station parameters must contain exactly the supported fields')
        for field, (low, high) in LIMITS.items():
            value = params[field]
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError('unsupported station parameter: ' + field)
        if params['light_side'] not in (-1, 1):
            raise ValueError('light_side must be -1 or 1')
        if params['occlusion'] not in OCCLUSIONS:
            raise ValueError('unsupported station occlusion')
        if params['occlusion'] != 'none' and not allow_occlusion:
            raise ValueError('station occlusion requires the occluder option')
        result.extend({'id': name, 'parameters': dict(params), 'sources': list(sources)}
                      for _ in range(quota))
    if len(result) != count:
        raise ValueError('allocated station counts differ from budget')
    return result
