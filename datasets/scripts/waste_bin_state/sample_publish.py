"""Waste-bin sample manifest ownership and shared publication interface."""
import json
import os
from pathlib import PurePosixPath

from datasets.scripts.common.recoverable_publish import (
    _metadata, _write_json, copy_unowned, new_stage, prepare, publish, recover, sample_lock, signature,
)


def _sample_path(value, role):
    if not isinstance(value, str) or '\\' in value:
        raise ValueError('invalid manifest sample path')
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 2 or path.parts[0] != role or path.name in ('.', '..'):
        raise ValueError('manifest sample path outside ' + role)
    return path.as_posix()


def owned_files(output, previous, output_state, workspace):
    """Validate old manifest and previous receipt; return owned relative names."""
    manifest_path = output / 'manifest.json'
    if not manifest_path.exists():
        if previous is not None:
            raise ValueError('previous sample receipt exists but manifest is missing')
        return set()
    if manifest_path.is_symlink() or previous is None:
        raise ValueError('sample manifest has no matching production receipt')
    data = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if data.get('task') != 'waste_bin_state' or data.get('status') != 'complete_representative_sample':
        raise ValueError('sample manifest task or status is invalid')
    records = data.get('records')
    if not isinstance(records, list):
        raise ValueError('sample manifest records missing')
    names = ['manifest.json']
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('sample manifest record invalid')
        names.extend((_sample_path(record.get('image'), 'images'), _sample_path(record.get('label'), 'labels')))
    keys = [os.path.normcase(name) for name in names]
    if len(keys) != len(set(keys)):
        raise ValueError('sample manifest contains duplicate paths')
    relroot = output.relative_to(workspace).as_posix()
    before = {row['path']: row for row in previous.get('outputs', [])}
    current = {row['path']: row for row in output_state}
    for name in names:
        path = relroot + '/' + name
        if not (output / name).is_file() or (output / name).is_symlink():
            raise ValueError('manifest-owned sample file missing or unsafe: ' + name)
        if before.get(path) != current.get(path) or not before.get(path):
            raise ValueError('manifest-owned sample does not match previous receipt: ' + name)
    return set(names)
