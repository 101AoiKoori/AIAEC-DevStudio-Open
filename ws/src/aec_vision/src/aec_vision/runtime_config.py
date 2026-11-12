"""Load task switches and prepare a route; no ROS or motion side effects."""
import json
import math
from pathlib import Path

TASK_IDS = frozenset(('traffic_light', 'person', 'sign', 'waste_bin', 'fire',
                      'thermal', 'meter', 'plate', 'e_bike'))
NAVIGATION_GATES = frozenset(('traffic_light', 'sign'))


class ConfigError(ValueError):
    """Invalid or unsafe competition configuration."""


def load_config(path):
    with Path(path).open(encoding='utf-8') as stream:
        data = json.load(stream)
    validate_config(data)
    return data


def validate_config(data):
    if not isinstance(data, dict) or type(data.get('version')) is not int or data['version'] != 1:
        raise ConfigError('unsupported configuration version')
    tasks = data.get('tasks')
    if not isinstance(tasks, dict) or set(tasks) != TASK_IDS:
        raise ConfigError('tasks must contain the nine known task IDs')
    for task_id, task in tasks.items():
        if not isinstance(task, dict) or type(task.get('enabled')) is not bool:
            raise ConfigError('{}.enabled must be a boolean'.format(task_id))
    routes = data.get('routes')
    if not isinstance(routes, dict) or not routes:
        raise ConfigError('routes must be a non-empty object')
    selected = data.get('selected_route')
    if selected is not None and (not isinstance(selected, str) or selected not in routes):
        raise ConfigError('selected_route must name a route or be null')
    for route_id, route in routes.items():
        if not isinstance(route_id, str) or not route_id or not isinstance(route, dict):
            raise ConfigError('invalid route')
        waypoints = route.get('waypoints')
        if not isinstance(waypoints, list):
            raise ConfigError('{}.waypoints must be a list'.format(route_id))
        seen = set()
        for waypoint in waypoints:
            if not isinstance(waypoint, dict):
                raise ConfigError('waypoint must be an object')
            waypoint_id = waypoint.get('id')
            if not isinstance(waypoint_id, str) or not waypoint_id or waypoint_id in seen:
                raise ConfigError('waypoint IDs must be non-empty and unique within a route')
            seen.add(waypoint_id)
            if type(waypoint.get('calibrated')) is not bool:
                raise ConfigError('{}.calibrated must be a boolean'.format(waypoint_id))
            pose = waypoint.get('pose')
            if not isinstance(pose, dict) or set(pose) != {'x', 'y', 'yaw'}:
                raise ConfigError('{}.pose must have x, y and yaw'.format(waypoint_id))
            for value in pose.values():
                if value is not None and (isinstance(value, bool) or
                        not isinstance(value, (int, float)) or not math.isfinite(value)):
                    raise ConfigError('pose values must be finite numbers or null')
            jobs = waypoint.get('tasks')
            if not isinstance(jobs, list):
                raise ConfigError('{}.tasks must be a list'.format(waypoint_id))
            job_ids = set()
            for job in jobs:
                if not isinstance(job, dict) or not isinstance(job.get('type'), str) or job['type'] not in TASK_IDS:
                    raise ConfigError('{}.tasks contains an unknown task'.format(waypoint_id))
                job_id = job.get('id')
                if not isinstance(job_id, str) or not job_id or job_id in job_ids:
                    raise ConfigError('task IDs must be non-empty and unique within a waypoint')
                if not isinstance(job.get('context', {}), dict):
                    raise ConfigError('task context must be an object')
                job_ids.add(job_id)


def enabled_tasks(data):
    validate_config(data)
    return {name for name, value in data['tasks'].items() if value['enabled']}


def prepare_route(data, route_id=None):
    """Validate prerequisites and return enabled tasks per waypoint, without driving."""
    validate_config(data)
    active = enabled_tasks(data)
    if not NAVIGATION_GATES.issubset(active):
        raise ConfigError('navigation requires both traffic_light and sign to be enabled')
    selected = data.get('selected_route') if route_id is None else route_id
    if selected is None:
        raise ConfigError('no selected route')
    if not isinstance(selected, str) or selected not in data['routes']:
        raise ConfigError('unknown route')
    waypoints = data['routes'][selected]['waypoints']
    if not waypoints:
        raise ConfigError('selected route has no waypoints')
    prepared = []
    for waypoint in waypoints:
        if not waypoint['calibrated'] or any(v is None for v in waypoint['pose'].values()):
            raise ConfigError('{} is not calibrated'.format(waypoint['id']))
        prepared.append(dict(waypoint, tasks=[t for t in waypoint['tasks'] if t['type'] in active]))
    return prepared
