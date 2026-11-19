"""Recoverable same-volume directory publishing for cooperating producers.

Callers hold ``sample_lock`` across recovery, reservation, installation, and
receipt commit. The journal and backups are siblings of the target directory.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid


def _metadata(output):
    parent = output.parent
    return (parent / ('.' + output.name + '.publish.lock'),
            parent / ('.' + output.name + '.publish.json'))


@contextmanager
def sample_lock(output):
    """Serialize cooperating sample publishers, including recovery."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock, _ = _metadata(output)
    if lock.is_symlink():
        raise ValueError('sample lock cannot be a symlink')
    with lock.open('a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0, os.SEEK_END)
            if not handle.tell():
                handle.write(b'0')
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def signature(path):
    """Identify files and empty directories without following symlinks."""
    if path.is_symlink():
        raise ValueError('sample directory must not be a symlink')
    if not path.exists():
        return None
    if not path.is_dir():
        raise ValueError('sample directory must be a real directory')
    rows = []
    for entry in sorted(path.rglob('*')):
        if entry.is_symlink():
            raise ValueError('sample content cannot contain symlinks')
        rel = entry.relative_to(path).as_posix()
        if entry.is_dir():
            rows.append([rel, 'dir'])
        elif entry.is_file():
            rows.append([rel, 'file', _file_digest(entry)])
        else:
            raise ValueError('sample content must be regular files or directories')
    return rows


def _write_json(path, value):
    tmp = path.with_name(path.name + '.new.' + uuid.uuid4().hex)
    if path.is_symlink():
        raise ValueError('sample journal path cannot be a symlink')
    try:
        with tmp.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def prepare(output, token):
    """Journal the intended stage before writing any staged files."""
    output = Path(output)
    _, journal = _metadata(output)
    if journal.exists() or journal.is_symlink():
        raise ValueError('unfinished sample publication must be recovered first')
    if not token or not all(c in '0123456789abcdef' for c in token):
        raise ValueError('invalid production token')
    stage = new_stage(output, token)
    backup = output.parent / ('.' + output.name + '.backup.' + token)
    if backup.exists() or backup.is_symlink():
        raise ValueError('sample stage/backup location is invalid')
    info = {'version': 1, 'output': output.name, 'stage': stage.name, 'backup': backup.name,
            'token': token, 'old': signature(output), 'new': None, 'phase': 'preparing'}
    _write_json(journal, info)
    return stage


def _sync_contents(source, destination):
    """Replace files inside an existing root, retaining identical files."""
    source_state = signature(source)
    signature(destination)
    files = {row[0]: row[2] for row in source_state if row[1] == 'file'}
    directories = {row[0] for row in source_state if row[1] == 'dir'}
    for path in destination.rglob('*'):
        if path.is_file() and path.relative_to(destination).as_posix() not in files:
            path.unlink()
    for path in sorted((p for p in destination.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        if path.relative_to(destination).as_posix() not in directories:
            path.rmdir()
    for relative in sorted(directories):
        (destination / relative).mkdir(exist_ok=True, parents=True)
    for relative, checksum in sorted(files.items(), key=lambda pair: (Path(pair[0]).name == 'manifest.json', pair[0])):
        target = destination / relative
        if not target.is_file() or _file_digest(target) != checksum:
            shutil.copy2(source / relative, target)
    if signature(destination) != source_state:
        raise ValueError('sample content changed during file publication')


def publish(output, stage, token, *, keep_root=False):
    """Install a staged result; keep old files for receipt-based recovery.

    keep_root copies the recovery backup and installs file contents without
    renaming an open destination directory. Readers use the completed receipt.
    """
    output, stage = Path(output), Path(stage)
    _, journal = _metadata(output)
    info = json.loads(journal.read_text(encoding='utf-8'))
    if (info.get('token') != token or info.get('phase') != 'preparing' or
            info.get('stage') != stage.name or stage.parent != output.parent):
        raise ValueError('sample stage does not match the publication journal')
    backup = output.parent / info['backup']
    if signature(output) != info['old'] or backup.exists() or backup.is_symlink():
        raise ValueError('sample changed while preparing replacement')
    info.update(new=signature(stage), phase='prepared')
    if info['new'] is None:
        raise ValueError('sample stage is missing')
    if keep_root and output.exists():
        info['phase'] = 'copying_backup'
        _write_json(journal, info)
        shutil.copytree(output, backup)
        if signature(backup) != info['old'] or signature(output) != info['old']:
            raise ValueError('sample changed while copying its recovery backup')
        info['phase'] = 'prepared_contents'
        _write_json(journal, info)
        _sync_contents(stage, output)
        return
    _write_json(journal, info)
    if output.exists():
        output.rename(backup)
    stage.rename(output)


def recover(output, receipt_completed):
    """Finish an installed committed run or restore its prior sample on retry.

    Does not claim power-loss durability across filesystem/database boundaries.
    Refuses unexpected changes and leaves all files in place for inspection.
    """
    output = Path(output)
    _, journal = _metadata(output)
    if not journal.exists():
        return False
    if journal.is_symlink():
        raise ValueError('sample journal cannot be a symlink')
    info = json.loads(journal.read_text(encoding='utf-8'))
    token = info.get('token', '')
    prefix = '.' + output.name
    if (info.get('version') != 1 or info.get('output') != output.name or
            not token or not all(c in '0123456789abcdef' for c in token) or
            info.get('stage') != prefix + '.stage.' + token or
            info.get('backup') != prefix + '.backup.' + token):
        raise ValueError('invalid sample publication journal')
    stage, backup = output.parent / info['stage'], output.parent / info['backup']
    old, new = info.get('old'), info.get('new')
    actual, saved, pending = signature(output), signature(backup), signature(stage)
    if info.get('phase') == 'preparing':
        if receipt_completed(token) or actual != old or saved is not None:
            raise ValueError('sample changed during interrupted preparation')
        if pending is not None:
            shutil.rmtree(stage)
        journal.unlink()
        return True
    if info.get('phase') == 'copying_backup':
        if receipt_completed(token) or actual != old or pending != new:
            raise ValueError('sample changed while preparing its content backup')
        if saved is not None:
            shutil.rmtree(backup)
        shutil.rmtree(stage)
        journal.unlink()
        return True
    if info.get('phase') == 'prepared_contents':
        if saved != old or pending != new or old is None or new is None:
            raise ValueError('sample content recovery evidence changed')
        if receipt_completed(token):
            if actual != new:
                raise ValueError('committed sample differs from its staged contents')
        else:
            before = {row[0]: row[1:] for row in old}
            after = {row[0]: row[1:] for row in new}
            current = {row[0]: row[1:] for row in actual} if actual is not None else {}
            preserved = {path: row for path, row in before.items() if after.get(path) == row}
            if (current.keys() - (before.keys() | after.keys()) or
                    any(current.get(path) != row for path, row in preserved.items())):
                raise ValueError('unowned sample content changed during publication')
            _sync_contents(backup, output)
        shutil.rmtree(backup)
        shutil.rmtree(stage)
        journal.unlink()
        return True
    if info.get('phase') != 'prepared' or new is None:
        raise ValueError('invalid sample publication phase')
    if receipt_completed(token):
        if actual != new or pending is not None or saved not in (old, None):
            raise ValueError('committed sample differs from publication journal')
        if saved is not None:
            shutil.rmtree(backup)
    elif saved is not None:
        if saved != old or actual not in (new, None) or pending not in (new, None):
            raise ValueError('sample files changed during interrupted publication')
        if actual is not None:
            shutil.rmtree(output)
        backup.rename(output)
        if pending is not None:
            shutil.rmtree(stage)
    elif old is None and actual == new and pending is None:
        shutil.rmtree(output)
    else:
        if actual != old or pending not in (new, None):
            raise ValueError('sample files changed before publication')
        if pending is not None:
            shutil.rmtree(stage)
    journal.unlink()
    return True


def new_stage(output, token):
    path = output.parent / ('.' + output.name + '.stage.' + token)
    if path.exists() or path.is_symlink():
        raise ValueError('sample stage already exists')
    return path


def copy_unowned(output, stage, owned):
    """Keep all other files, including .gitkeep; never follow symlinks."""
    stage.mkdir()
    if not output.exists():
        return
    signature(output)
    for item in sorted(output.rglob('*')):
        rel = item.relative_to(output)
        if rel.as_posix() in owned:
            continue
        target = stage / rel
        if item.is_dir():
            target.mkdir(exist_ok=True, parents=True)
        else:
            target.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy2(item, target)
