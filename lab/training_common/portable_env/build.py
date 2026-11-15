"""Build a copyable dependency directory; no Python interpreter is bundled."""
import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parent
MINOR = sys.version_info.minor
LOCK = ROOT / ('requirements-win-cp3{}-cu128.lock'.format(MINOR))


def run(command, log, env):
    log.write(json.dumps(command, ensure_ascii=False) + '\n')
    log.flush()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding='utf-8', errors='replace', env=env)
    for line in process.stdout:
        print(line, end='')
        log.write(line)
        log.flush()
    code = process.wait()
    if code:
        raise RuntimeError('Command failed with exit code {}: {}'.format(code, command[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=None, help='New directory name under bundles/')
    parser.add_argument('--resolve', action='store_true', help='Intentionally update the pinned lock before building')
    args = parser.parse_args()
    if sys.platform != 'win32' or sys.implementation.name != 'cpython' or sys.version_info.major != 3 or MINOR not in (12, 14) or sysconfig.get_platform() != 'win-amd64' or sysconfig.get_config_var('Py_DEBUG') or sysconfig.get_config_var('Py_GIL_DISABLED'):
        parser.error('Build with standard Windows x64 CPython 3.12 or 3.14; Python 3.13 is unsupported')
    args.output = args.output or 'win-cp3{}-cu128-v3'.format(MINOR)
    if len(Path(args.output).parts) != 1 or args.output in ('.', '..'):
        parser.error('--output must be one new directory name')
    output = ROOT / 'bundles' / args.output
    if output.exists():
        parser.error('Output already exists; choose a new bundle version: ' + str(output))
    uv = shutil.which('uv')
    if not uv:
        parser.error('Source machine requires uv on PATH. Target computers do not require uv or pip.')
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    record = ROOT / 'runs' / ('build_' + stamp)
    record.mkdir(parents=True)
    (record / 'tmp').mkdir()
    environment = os.environ.copy()
    environment.update(UV_CACHE_DIR=str(ROOT / 'runs' / 'uv_cache'), UV_PYTHON_DOWNLOADS='never',
                       UV_PYTHON=str(Path(sys.executable).resolve()), PYTHONIOENCODING='utf-8',
                       TEMP=str(record / 'tmp'), TMP=str(record / 'tmp'))
    state = {'status': 'running', 'bundle': str(output.relative_to(ROOT)), 'started': stamp,
             'python': sys.version, 'platform': sysconfig.get_platform()}
    try:
        with (record / 'build.log').open('w', encoding='utf-8') as log:
            if args.resolve or not LOCK.exists():
                run([uv, 'pip', 'compile', str(ROOT / ('requirements.in' if MINOR == 12 else 'requirements-modern.in')), '--python', sys.executable,
                     '--python-version', '3.{}'.format(MINOR), '--python-platform', 'x86_64-pc-windows-msvc',
                     '--torch-backend', 'cu128', '--no-python-downloads', '--no-header', '--no-annotate',
                     '--no-build', '--output-file', str(LOCK)], log, environment)
            output.mkdir(parents=True)
            run([uv, 'pip', 'install', '--python', sys.executable, '--target', str(output / 'site-packages'),
                 '--requirements', str(LOCK), '--torch-backend', 'cu128', '--no-python-downloads',
                 '--no-build', '--link-mode', 'copy', '--strict'], log, environment)
            distributions = list(importlib.metadata.distributions(path=[str(output / 'site-packages')]))
            packages = {d.metadata['Name']: d.version for d in distributions}
            if len(packages) != len(distributions):
                raise RuntimeError('Duplicate distribution metadata in dependency directory')
            manifest = {'schema_version': 1, 'name': args.output, 'status': 'built',
                        'python_implementation': 'cpython', 'python_version': [3, MINOR],
                        'platform': 'win-amd64', 'bits': 64, 'cache_tag': 'cpython-3{}'.format(MINOR),
                        'extension_suffix': '.cp3{}-win_amd64.pyd'.format(MINOR), 'backend': 'cu128',
                        'packages': dict(sorted(packages.items(), key=lambda item: item[0].lower())),
                        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        'python_included': False,
                        'integrity': 'distribution versions, RECORD file existence and recorded sizes; no content hashes'}
            for name in ('bootstrap.py', 'run.ps1'):
                shutil.copy2(ROOT / name, output / name)
            shutil.copy2(ROOT / 'BUNDLE_README.md', output / 'README.md')
            shutil.copy2(LOCK, output / 'requirements.lock')
            (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            run([sys.executable, '-I', '-S', '-B', str(output / 'bootstrap.py'), '--check', '--offline',
                 '--report', str(record / 'environment_check.json')], log, environment)
        state['status'] = 'complete'
    except BaseException as error:
        state['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        state['error'] = '{}: {}'.format(type(error).__name__, error)
        raise
    finally:
        (record / 'build_state.json').write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Bundle ready: ' + str(output))
    print('Build evidence: ' + str(record))


if __name__ == '__main__':
    main()
