"""Copy a built environment and exercise its offline launcher and Windows workers."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', default='win-cp312-cu128-v3')
    parser.add_argument('--model', help='Trusted local .pt checkpoint for the native check')
    parser.add_argument('--require-cuda', action='store_true')
    args = parser.parse_args()
    if len(Path(args.bundle).parts) != 1:
        parser.error('--bundle must name a built directory below bundles/')
    source = ROOT / 'bundles' / args.bundle
    if not (source / 'manifest.json').is_file():
        parser.error('Bundle missing or build incomplete')
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = ROOT / 'runs' / ('relocation_' + stamp)
    output.mkdir(parents=True)
    relocated = output / 'copied environment with spaces'
    state = {'status': 'running', 'source_bundle': args.bundle, 'copied_environment': relocated.name,
             'scope': 'same Windows host, copied dependency path, existing interpreter; not a second physical computer'}
    (output / 'state.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    try:
        shutil.copytree(source, relocated)
        poison = output / 'unrelated system packages'
        poison.mkdir()
        (poison / 'numpy.py').write_text("raise RuntimeError('External PYTHONPATH was incorrectly loaded')\n", encoding='utf-8')
        env = os.environ.copy()
        env['PYTHONPATH'] = str(poison)
        report = output / 'native' / 'report.json'
        command = ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(relocated / 'run.ps1'),
                   '-Check', '-Offline', '-Report', str(report)]
        if args.require_cuda:
            command.append('-RequireCuda')
        if args.model:
            command += ['-Model', str(Path(args.model).resolve())]
        native = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8', errors='replace', env=env, timeout=300)
        (output / 'native.log').write_text(native.stdout, encoding='utf-8')
        if native.returncode:
            raise RuntimeError('Copied environment self-check failed; see native.log')
        state['native_report'] = str(report.relative_to(output))
        script = output / 'probe_spawn.py'
        script.write_text('''import json
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset

if __name__ == '__main__':
    loader = DataLoader(TensorDataset(torch.arange(12)), batch_size=3, num_workers=2)
    values = [int(value) for batch in loader for value in batch[0]]
    assert values == list(range(12)), values
    print(json.dumps({'status':'passed','workers':2,'samples':len(values),
                      'torch_origin':str(Path(torch.__file__).relative_to(os.environ['AEC_PORTABLE_ENV']))}))
''', encoding='utf-8')
        spawned = subprocess.run([sys.executable, '-I', '-S', '-B', str(relocated / 'bootstrap.py'),
                                  '--offline', '--script', str(script)], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                                 env=env, timeout=120)
        (output / 'workers.log').write_text(spawned.stdout, encoding='utf-8')
        if spawned.returncode:
            raise RuntimeError('Two-worker DataLoader failed; see workers.log')
        state['workers'] = 2
        manifest_path = relocated / 'manifest.json'
        original_manifest = manifest_path.read_text(encoding='utf-8')
        wrong = json.loads(original_manifest)
        wrong['python_version'] = [3, 11]
        try:
            manifest_path.write_text(json.dumps(wrong, indent=2), encoding='utf-8')
            mismatch = subprocess.run([sys.executable, '-I', '-S', '-B', str(relocated / 'bootstrap.py'), '--check'],
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                      encoding='utf-8', errors='replace', env=env, timeout=30)
            (output / 'incompatible_python.log').write_text(mismatch.stdout, encoding='utf-8')
            if mismatch.returncode == 0 or 'Incompatible Python' not in mismatch.stdout:
                raise RuntimeError('Python minor version mismatch was not rejected')
        finally:
            manifest_path.write_text(original_manifest, encoding='utf-8')
        state['wrong_minor_rejected'] = True
        state['pythonpath_ignored'] = True
        state['status'] = 'passed'
    except BaseException as error:
        state['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        state['error'] = '{}: {}'.format(type(error).__name__, error)
        raise
    finally:
        (output / 'state.json').write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': state['status'], 'evidence': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
