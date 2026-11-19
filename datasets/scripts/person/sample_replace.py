"""Identify files owned by a previous person generation manifest."""
import json
import os
from pathlib import Path, PurePosixPath


def _owned_path(value, workspace, output, role):
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError('invalid person manifest path')
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or '..' in parsed.parts or parsed.as_posix() != value:
        raise ValueError('person manifest path must be canonical and workspace-relative')
    path = workspace.joinpath(*parsed.parts)
    try:
        relative = path.relative_to(output)
    except ValueError as error:
        raise ValueError('person manifest path outside the output') from error
    if (len(relative.parts) not in (2, 3) or relative.parts[0] != role or
            (len(relative.parts) == 3 and relative.parts[1] not in ('synthetic', 'direct_real'))):
        raise ValueError('person manifest path outside expected image or label directory')
    return relative.as_posix()


def owned_files(output, previous, output_state, workspace):
    """Verify an earlier manifest against its receipt; do not claim other files.

    Manifest records use workspace-relative paths. File snapshots are compared
    with the receipt before an adapter may replace anything; unlisted files are
    preserved and any new-name collision must be rejected by the caller.
    """
    output, workspace = Path(output), Path(workspace)
    manifest = output / 'manifest.json'
    if not manifest.exists():
        if previous is not None:
            raise ValueError('previous person receipt exists but manifest is missing')
        return set()
    if manifest.is_symlink() or previous is None:
        raise ValueError('person manifest has no matching production receipt')
    info = json.loads(manifest.read_text(encoding='utf-8-sig'))
    if info.get('status') not in ('complete_task_yolo', 'candidate_not_semantically_verified'):
        raise ValueError('unsupported person manifest status')
    records = info.get('records')
    if not isinstance(records, list):
        raise ValueError('person manifest records missing')
    names = ['manifest.json']
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('invalid person manifest record')
        names.extend((_owned_path(record.get('image'), workspace, output, 'images'),
                      _owned_path(record.get('label'), workspace, output, 'labels')))
    if len({os.path.normcase(name) for name in names}) != len(names):
        raise ValueError('person manifest contains duplicate paths')
    root = output.relative_to(workspace).as_posix()
    before = {row['path']: row for row in previous.get('outputs', [])}
    current = {row['path']: row for row in output_state}
    for name in names:
        path = output / name
        relative = path.relative_to(output)
        traversed = [output.joinpath(*relative.parts[:index]) for index in range(1, len(relative.parts) + 1)]
        if not path.is_file() or output.is_symlink() or any(p.is_symlink() for p in traversed):
            raise ValueError('person manifest-owned file missing or unsafe: ' + name)
        key = root + '/' + name
        if not before.get(key) or before[key] != current.get(key):
            raise ValueError('person manifest-owned file does not match previous receipt: ' + name)
    return set(names)
