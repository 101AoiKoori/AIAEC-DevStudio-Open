"""Isolated stdlib bootstrap for a copied dependency bundle. Invoke with -I -S -B."""
import argparse
import csv
import datetime
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import runpy
import site
import socket
import struct
import subprocess
import sys
import sysconfig
import tempfile
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parent
DLL_HANDLES = []


def inside(path, root):
    try:
        Path(path).resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def interpreter_info():
    return {'python_implementation': sys.implementation.name, 'python_version': list(sys.version_info[:2]),
            'platform': sysconfig.get_platform(), 'bits': struct.calcsize('P') * 8,
            'cache_tag': sys.implementation.cache_tag, 'extension_suffix': sysconfig.get_config_var('EXT_SUFFIX'),
            'debug': bool(sysconfig.get_config_var('Py_DEBUG')),
            'free_threaded': bool(sysconfig.get_config_var('Py_GIL_DISABLED'))}


def validate_interpreter(manifest):
    observed = interpreter_info()
    mismatches = {key: {'expected': manifest[key], 'actual': observed[key]}
                  for key in ('python_implementation', 'python_version', 'platform', 'bits', 'cache_tag', 'extension_suffix')
                  if manifest.get(key) != observed[key]}
    if observed['debug'] or observed['free_threaded']:
        mismatches['build'] = 'Debug/free-threaded Python is unsupported'
    if mismatches:
        raise RuntimeError('Incompatible Python interpreter: ' + json.dumps(mismatches))
    return observed


def configure_paths(manifest):
    validate_interpreter(manifest)
    if not sys.flags.isolated or not sys.flags.no_site:
        raise RuntimeError('Launch via run.ps1 or python -I -S -B bootstrap.py; system site-packages must stay disabled')
    sys.dont_write_bytecode = True
    os.environ['AEC_PORTABLE_ENV'] = str(ROOT)
    os.environ['AEC_PORTABLE_BOOTSTRAP'] = str(Path(__file__).resolve())
    vendor = ROOT / 'site-packages'
    if not vendor.is_dir():
        raise RuntimeError('Missing site-packages; copy the whole environment bundle')
    initial = {str(Path(path).resolve()) for path in sys.path if path}
    for pth in vendor.glob('*.pth'):
        for line in pth.read_text(encoding='utf-8-sig').splitlines():
            line = line.strip()
            if not line or line.startswith('#') or line.startswith(('import ', 'import\t')):
                continue
            if Path(line).is_absolute() or not inside(vendor / line, vendor):
                raise RuntimeError('Nonportable path in .pth: ' + pth.name)
    # Import lines belong to the pinned installed wheels; capture failures instead of hiding them.
    errors = io.StringIO()
    previous = sys.stderr
    try:
        sys.stderr = errors
        site.addsitedir(str(vendor))
    finally:
        sys.stderr = previous
    if errors.getvalue():
        raise RuntimeError('Dependency .pth processing failed: ' + errors.getvalue())
    for path in sys.path:
        if str(Path(path).resolve()) not in initial and not inside(path, vendor):
            raise RuntimeError('A dependency .pth added an external import path: ' + str(path))
    # Retain handles for the process lifetime. Package loaders handle their bundled .libs directories.
    for path in (vendor / 'torch' / 'lib', vendor / 'torchvision', vendor / 'numpy.libs', vendor / 'scipy.libs'):
        if path.is_dir() and hasattr(os, 'add_dll_directory'):
            DLL_HANDLES.append(os.add_dll_directory(str(path)))
    return vendor


def inventory(vendor, manifest, full=False):
    expected = {key.lower().replace('_', '-'): value for key, value in manifest['packages'].items()}
    found = {}
    files = 0
    missing = []
    sizes = []
    for dist in importlib.metadata.distributions(path=[str(vendor)]):
        name = dist.metadata['Name'].lower().replace('_', '-')
        if name in found:
            raise RuntimeError('Duplicate package metadata: ' + name)
        found[name] = dist.version
        if full:
            record = dist.read_text('RECORD')
            if record is None:
                raise RuntimeError('Wheel RECORD missing: ' + name)
            for row in csv.reader(io.StringIO(record)):
                if not row:
                    continue
                candidate = Path(dist.locate_file(row[0])).resolve()
                if not inside(candidate, ROOT):
                    raise RuntimeError('Wheel RECORD path leaves bundle: ' + row[0])
                if candidate.suffix == '.pyc':
                    continue
                files += 1
                if not candidate.is_file():
                    missing.append(str(candidate.relative_to(ROOT)))
                elif len(row) > 2 and row[2] and candidate.stat().st_size != int(row[2]):
                    sizes.append(str(candidate.relative_to(ROOT)))
    if expected != found:
        raise RuntimeError('Dependency versions differ from manifest: ' + json.dumps({
            'missing': sorted(set(expected) - set(found)), 'extra': sorted(set(found) - set(expected)),
            'versions': {key: [expected[key], found[key]] for key in set(expected) & set(found) if expected[key] != found[key]}}))
    if missing or sizes:
        raise RuntimeError('Dependency files missing or truncated: ' + json.dumps({'missing': missing[:20], 'size_mismatch': sizes[:20]}))
    return {'packages': len(found), 'record_files_checked': files, 'check': 'versions/existence/recorded sizes; no hashes'}


def deny_network():
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def local(address):
        return isinstance(address, tuple) and address[0] in ('127.0.0.1', '::1', 'localhost')

    def connect(sock, address):
        if not local(address):
            raise OSError('Offline mode: outbound connection blocked')
        return original_connect(sock, address)

    def connect_ex(sock, address):
        if not local(address):
            raise OSError('Offline mode: outbound connection blocked')
        return original_connect_ex(sock, address)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(key, None)
    os.environ.update(ULTRALYTICS_OFFLINE='true', YOLO_OFFLINE='true', YOLO_AUTOINSTALL='false',
                      HF_HUB_OFFLINE='1', WANDB_MODE='offline', PIP_NO_INDEX='1',
                      NO_PROXY='*', no_proxy='*')


def self_check(vendor, manifest, require_cuda, report_dir, model):
    result = {'status': 'running', 'interpreter': interpreter_info(),
              'python_patch': platform.python_version(), 'inventory': inventory(vendor, manifest, full=True)}
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    from packaging.markers import default_environment
    installed = {canonicalize_name(dist.metadata['Name']): dist.version
                 for dist in importlib.metadata.distributions(path=[str(vendor)])}
    marker_environment = dict(default_environment(), extra='')
    conflicts = []
    for dist in importlib.metadata.distributions(path=[str(vendor)]):
        for raw in dist.requires or []:
            dependency = Requirement(raw)
            if dependency.marker and not dependency.marker.evaluate(marker_environment):
                continue
            version = installed.get(canonicalize_name(dependency.name))
            if version is None or version not in dependency.specifier:
                conflicts.append({'package': dist.metadata['Name'], 'requires': raw, 'installed': version})
    if conflicts:
        raise RuntimeError('Dependency requirements conflict: ' + json.dumps(conflicts))
    result['dependency_requirements'] = 'passed'
    import numpy as np
    import torch
    import torchvision
    import cv2
    from PIL import Image
    import matplotlib
    import ultralytics
    import yaml
    import tensorboard
    from torch.utils.tensorboard import SummaryWriter
    from torchvision.ops import nms
    imports = {'numpy': np, 'torch': torch, 'torchvision': torchvision, 'cv2': cv2,
               'ultralytics': ultralytics, 'yaml': yaml, 'tensorboard': tensorboard, 'matplotlib': matplotlib}
    for name, module in imports.items():
        if not inside(module.__file__, vendor):
            raise RuntimeError(name + ' was loaded outside the copied environment')
    result['imports'] = {name: str(Path(module.__file__).relative_to(ROOT)) for name, module in imports.items()}
    tensor = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    if not np.allclose((tensor @ tensor).numpy(), np.arange(9, dtype=np.float32).reshape(3, 3) @ np.arange(9, dtype=np.float32).reshape(3, 3)):
        raise RuntimeError('Tensor/NumPy conversion check failed')
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8, 0.7])
    if nms(boxes, scores, 0.5).tolist() != [0, 2]:
        raise RuntimeError('torchvision CPU NMS failed')
    if cv2.cvtColor(np.zeros((8, 8, 3), dtype=np.uint8), cv2.COLOR_BGR2GRAY).shape != (8, 8):
        raise RuntimeError('OpenCV native operation failed')
    Image.new('RGB', (8, 8)).save(report_dir / 'native_image.png')
    writer = SummaryWriter(str(report_dir / 'tensorboard'))
    writer.add_scalar('environment/native_check', 1, 0)
    writer.close()
    with socket.socket() as local_socket:
        local_socket.bind(('127.0.0.1', 0))
        port = local_socket.getsockname()[1]
    command = [sys.executable, '-I', '-S', '-B', str(ROOT / 'bootstrap.py'), '--offline',
               '--module', 'tensorboard.main', '--', '--logdir', str(report_dir / 'tensorboard'),
               '--host', '127.0.0.1', '--port', str(port), '--load_fast', 'false']
    with (report_dir / 'tensorboard_server.log').open('w', encoding='utf-8') as server_log:
        server = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT,
                                  creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            last_error = None
            for _ in range(60):
                if server.poll() is not None:
                    raise RuntimeError('TensorBoard server exited; see tensorboard_server.log')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:{}/data/plugin/scalars/tags'.format(port), timeout=1) as response:
                        tags = json.load(response)
                    if any('environment/native_check' in value for value in tags.values()):
                        result['tensorboard_http'] = 'scalar event visible through localhost server'
                        break
                except (OSError, ValueError) as error:
                    last_error = str(error)
                time.sleep(0.5)
            else:
                raise RuntimeError('TensorBoard scalar was not readable: ' + str(last_error))
        finally:
            server.terminate()
            server.wait(timeout=10)
    cuda = torch.cuda.is_available()
    result['cuda'] = {'available': cuda, 'build': torch.version.cuda, 'compiled_architectures': torch.cuda.get_arch_list() if cuda else []}
    if require_cuda and not cuda:
        raise RuntimeError('CUDA was required but unavailable; check NVIDIA driver and supported GPU; no CPU fallback')
    if cuda:
        value = tensor.to('cuda') @ tensor.to('cuda')
        if not torch.allclose(value.cpu(), tensor @ tensor):
            raise RuntimeError('CUDA matrix operation failed')
        if nms(boxes.cuda(), scores.cuda(), 0.5).cpu().tolist() != [0, 2]:
            raise RuntimeError('torchvision CUDA NMS failed')
        conv = torch.nn.Conv2d(3, 4, 3).cuda()
        conv(torch.zeros((1, 3, 16, 16), device='cuda')).sum().backward()
        torch.cuda.synchronize()
        result['cuda'].update(name=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                              operations=['matmul', 'torchvision_nms', 'convolution_forward_backward'])
    if model:
        path = Path(model).resolve()
        if not path.is_file() or path.suffix.lower() != '.pt':
            raise ValueError('--model must be a trusted existing .pt file; downloads are disabled')
        loaded = ultralytics.YOLO(str(path))
        prediction = loaded.predict(np.zeros((320, 320, 3), dtype=np.uint8), imgsz=320,
                                    device=0 if require_cuda else 'cpu', save=False, verbose=False)
        result['model'] = {'name': path.name, 'task': loaded.task, 'prediction_images': len(prediction)}
    result['status'] = 'passed'
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--script')
    mode.add_argument('--module')
    parser.add_argument('--report', help='Check report JSON; defaults to runs/environment_checks/')
    parser.add_argument('--require-cuda', action='store_true')
    parser.add_argument('--offline', action='store_true', help='Block Python outbound sockets; localhost remains available')
    parser.add_argument('--model', help='Optional trusted local YOLO checkpoint for self-check')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    vendor = configure_paths(manifest)
    inventory(vendor, manifest)
    if args.offline:
        deny_network()
    if args.require_cuda and not args.check:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was required but unavailable; no CPU fallback')
        torch.zeros(1, device='cuda').add_(1)
        torch.cuda.synchronize()
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    if args.check:
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        report = Path(args.report).resolve() if args.report else Path.cwd() / 'runs' / 'environment_checks' / stamp / 'report.json'
        report.parent.mkdir(parents=True, exist_ok=True)
        temporary = report.parent / 'tmp'
        temporary.mkdir(exist_ok=True)
        (report.parent / 'ultralytics' / 'Ultralytics').mkdir(parents=True, exist_ok=True)
        (report.parent / 'matplotlib').mkdir(exist_ok=True)
        tempfile.tempdir = str(temporary)
        os.environ.update(MPLCONFIGDIR=str(report.parent / 'matplotlib'),
                          YOLO_CONFIG_DIR=str(report.parent / 'ultralytics'),
                          TORCH_HOME=str(report.parent / 'torch_cache'),
                          XDG_CACHE_HOME=str(report.parent / 'cache'),
                          TEMP=str(temporary), TMP=str(temporary))
        try:
            result = self_check(vendor, manifest, args.require_cuda, report.parent, args.model)
        except Exception as error:
            result = {'status': 'failed', 'error': '{}: {}'.format(type(error).__name__, error), 'traceback': traceback.format_exc()}
            report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            raise
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'status': result['status'], 'report': str(report), 'cuda': result['cuda']}, ensure_ascii=False))
    elif args.script:
        target = Path(args.script).resolve()
        if not target.is_file() or target.suffix.lower() != '.py':
            parser.error('--script must name an existing Python script')
        sys.path.insert(0, str(target.parent))
        sys.argv = [str(target)] + arguments
        runpy.run_path(str(target), run_name='__main__')
    else:
        sys.argv = [args.module] + arguments
        runpy.run_module(args.module, run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('Environment launch failed: {}: {}'.format(type(error).__name__, error), file=sys.stderr)
        print('Check Python 3.12 x64, complete bundle copy, Visual C++ runtime and NVIDIA driver.', file=sys.stderr)
        raise SystemExit(2)
