"""Inspect or explicitly update the launcher of an existing portable environment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

SOURCE = Path(__file__).resolve().with_name('run.ps1')
WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_BUNDLE = WORKSPACE / 'training_packages' / 'portable_env'


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def launcher_status(bundle: Path, *, source: Path = SOURCE) -> dict:
    """Compare only the launcher bytes; do not inspect or change dependencies."""
    bundle = Path(bundle)
    if not bundle.is_dir():
        raise FileNotFoundError(f'Portable environment directory does not exist: {bundle}')
    desired = Path(source).read_bytes()
    target = bundle / 'run.ps1'
    if target.is_symlink():
        raise ValueError(f'Launcher is a symbolic link: {target}')
    current = target.read_bytes() if target.exists() else None
    return {'bundle': str(bundle), 'status': 'missing' if current is None else
            'current' if current == desired else 'outdated',
            'source_sha256': _digest(desired),
            'launcher_sha256': _digest(current) if current is not None else None}


def update_launcher(bundle: Path, *, source: Path = SOURCE) -> dict:
    """Atomically replace only run.ps1, after detecting intervening changes."""
    bundle, source = Path(bundle), Path(source)
    before = launcher_status(bundle, source=source)
    if before['status'] == 'current':
        return {**before, 'action': 'unchanged'}
    desired = source.read_bytes()
    if _digest(desired) != before['source_sha256']:
        raise RuntimeError('Launcher source changed before update')
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=bundle, prefix='.run-', suffix='.tmp', delete=False) as output:
            temp = Path(output.name)
            output.write(desired)
        if launcher_status(bundle, source=source) != before:
            raise RuntimeError('Launcher or source changed during update')
        os.replace(temp, bundle / 'run.ps1')
        temp = None
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    return {**launcher_status(bundle, source=source), 'action': 'updated',
            'previous_launcher_sha256': before['launcher_sha256']}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE,
                        help='existing portable_env directory (default: training_packages/portable_env)')
    parser.add_argument('--update', action='store_true', help='explicitly update only run.ps1')
    args = parser.parse_args()
    try:
        report = update_launcher(args.bundle) if args.update else launcher_status(args.bundle)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'{type(error).__name__}: {error}\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
