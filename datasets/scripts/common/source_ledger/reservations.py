"""Match task-owned parent declarations without assigning a split."""

from ..paths import SPLITS
from ..source_history import historical_source


def matching_groups(row, groups, history):
    """Match the same source identities and historical aliases used in person packaging."""
    values = [row.get(key) for key in ('source_group', 'parent_group', 'parent_video', 'source_video',
                                       'source_image', 'original_source_image', 'workspace_image',
                                       'mother_scene', 'source_scene')]
    values = {historical_source(value.replace('\\', '/').rstrip('/'), history)
              for value in values if isinstance(value, str) and value}
    result = []
    for group in groups:
        paths = {historical_source(path.replace('\\', '/').rstrip('/'), history) for path in group.paths}
        if group.group_id in values or any(value == path or value.startswith(path + '/')
                                           for value in values for path in paths):
            result.append(group)
    return result


def reservation_conflicts(groups, split):
    """Return explicit reservations incompatible with split; proposed is not reserved."""
    result = []
    for group in groups:
        rule = group.split
        primary = rule.get('primary')
        if (rule.get('status') == 'reserved' or rule.get('action') == 'reserved') and (
                primary in SPLITS and primary != split and split not in (rule.get('same_source_allowed_in') or [])):
            result.append(group)
    return result


def candidate_reservation(row, split, groups, history):
    """Report declarations, not an authorization or a database assignment."""
    matched = matching_groups(row, groups, history)
    # Frame paths can locate a declared parent directory. They remain source
    # identifiers; no file access or change of split is performed here.
    for path in row.get('paths', []):
        for group in matching_groups({'workspace_image': path}, groups, history):
            if group not in matched:
                matched.append(group)
    conflicts = reservation_conflicts(matched, split)
    state = ('conflict' if conflicts else 'unknown' if not matched else
             'ambiguous' if len(matched) > 1 else 'no_declared_conflict')
    return {'status': state, 'matched_groups': [group.group_id for group in matched],
            'conflicts': [{'group': group.group_id, 'reserved_split': group.split.get('primary')}
                          for group in conflicts]}
