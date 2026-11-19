"""Build a new portable V4 model package from reviewed YOLO data and local weights."""
import argparse
from copy import deepcopy
import filecmp
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from tempfile import TemporaryDirectory

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'lab/training_common'))
from engine.config import read_yaml, resolve, validate  # noqa: E402
from engine.data import _file_sha256, _label_counts, _require_v4_review, preflight_dataset, records  # noqa: E402
from launch_task import _positive_seconds, launcher_settings  # noqa: E402
from portable_env.sync_launcher import launcher_status  # noqa: E402


TASK_MODELS = {'person': 'person', 'e_bike': 'e_bike', 'fire': 'fire_sticker',
               'building_name_paper': 'building_name_paper', 'sign': 'sign',
               'waste_bin_state': 'waste_bin_state', 'waste_sticker': 'waste_sticker', 'meter': 'meter'}
COMMON = Path('lab/training_common')
PORTABLE = Path('training_packages/portable_env')
SCRIPTS = {'start_v3.ps1': 'scripts/start_training.ps1', 'launch_v3.py': 'scripts/launch.py',
           'monitor_training.py': 'scripts/monitor_training.py', 'watch_training.ps1': 'scripts/watch_training.ps1',
           'tensorboard_v3.py': 'scripts/view_tensorboard.py',
           'tensorboard_metrics.py': 'scripts/tensorboard_metrics.py',
           'run_training_tool.ps1': 'scripts/run_training_tool.ps1',
           'launch_task.py': 'scripts/launch_task.py',
           'train.py': 'scripts/common/train.py'}
TRAIN_ENTRY = '''"""Run the shared training CLI with this package's configuration."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/common'))
from engine.cli import main

if __name__ == '__main__':
    raise SystemExit(main(default_root=ROOT, default_config='config/train.yaml'))
'''
METER_ENTRY = '''"""Run eight-point meter training and needle diagnostics."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/training'))
from train_meter_v4 import main

if __name__ == '__main__':
    raise SystemExit(main())
'''
STICKER_ENTRY = '''"""Run waste-sticker training and its task-specific reports."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/training'))
from train_waste_sticker_v4 import main

if __name__ == '__main__':
    raise SystemExit(main())
'''
BIN_STATE_ENTRY = '''"""Run waste-bin-state training and its task-specific reports."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/training'))
from train_waste_bin_state_v4 import main

if __name__ == '__main__':
    raise SystemExit(main())
'''
SIGN_ENTRY = '''"""Run sign training and its task-specific reports."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/training'))
from train_sign_v4 import main

if __name__ == '__main__':
    raise SystemExit(main())
'''
PAPER_ENTRY = '''"""Run building-paper training and its task-specific reports."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/training'))
from train_building_name_paper_v4 import main

if __name__ == '__main__':
    raise SystemExit(main())
'''
FIRE_ENTRY = '''"""Run fire training, source/size audit and post-training reports."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/training'))
from train_fire_sticker_v4 import main

if __name__ == '__main__':
    raise SystemExit(main())
'''
PACKAGE_README = r'''# {model} V4 training package

Copy this task directory together with the adjacent `portable_env` directory as
one distribution. The destination needs a compatible Windows x64 CPython interpreter
(3.12 or 3.14), NVIDIA driver and Visual C++ runtime; the bundle does not include
Python itself. The epoch budget is stored in `config/train.yaml` and the maximum
running time is stored in `config/launcher.yaml`. Fill any unset budget before a new run.
The launcher YAML also controls TensorBoard and its browser window. Resume keeps the original
epoch budget and uses the previous wall-clock limit if the launcher YAML leaves it unset.

```powershell
.\train.ps1
.\train.ps1 -Resume
.\tensorboard.ps1
.\scripts\watch_training.ps1 -Follow
```

The package includes the audited inputs required by its training policy. Runs and checkpoints
are written to this package's `runs/`. Move the complete run directory for resume.
'''


def _read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def _launcher_report(root):
    """Report shared launcher identity without changing or blocking the environment."""
    try:
        status = launcher_status(root / PORTABLE, source=root / COMMON / 'portable_env/run.ps1')
    except (OSError, ValueError) as exc:
        return {'status': 'unavailable', 'reason': str(exc)}
    status['bundle'] = PORTABLE.as_posix()
    return status


def _destination(root, name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name) or name.casefold() == 'portable_env':
        raise ValueError('Output name must be one task folder name in training_packages')
    parent = root / 'training_packages'
    if not parent.is_dir() or parent.resolve() != parent:
        raise ValueError('training_packages must be an existing local workspace directory')
    return parent / name


def _evidence_paths(root, source_path, records, model):
    mapping_path = source_path.parent / 'source_records/evidence_map.json'
    source_map = _read_json(mapping_path) if mapping_path.is_file() else None
    refs = set()
    for row in records:
        selected_tasks = row.get('selected_tasks')
        if isinstance(selected_tasks, list) and model in selected_tasks:
            refs.update((row['review_evidence'], row['tasks'][model]['review_evidence']))
        else:
            refs.update((row['source_review_evidence'], row['review_evidence']))
    return {ref: resolve(root, source_map[ref] if source_map is not None else ref) for ref in sorted(refs)}


def _prepare(root, task, data_path, config_path, weights_path, output_name=None, *,
             include_test=False, wall_seconds=None):
    if task not in TASK_MODELS:
        raise ValueError('Unknown V4 data task: ' + task)
    model = TASK_MODELS[task]
    destination = _destination(root, output_name or model)
    data, config_file, weights = (resolve(root, value) for value in (data_path, config_path, weights_path))
    source_path = data.parent / 'workspace_sources.json'
    engine = root / COMMON / 'engine'
    script_files = {root / COMMON / source: target for source, target in SCRIPTS.items()}
    script_files[root / 'lab/train.ps1'] = 'train.ps1'
    script_files[root / 'lab/tensorboard.ps1'] = 'tensorboard.ps1'
    script_files.update({file: 'scripts/common/engine/' + file.name for file in sorted(engine.glob('*.py'))})
    if model == 'meter':
        task_scripts = root / 'lab/features/meter/training/scripts'
        script_files[task_scripts / 'train_meter_v4.py'] = 'scripts/training/train_meter_v4.py'
        if include_test:
            script_files[task_scripts / 'evaluate_meter.ps1'] = 'evaluate.ps1'
    if model == 'fire_sticker':
        task_scripts=root/'lab/features/fire/training/scripts'
        for name in ('train_fire_sticker_v4.py','fire_report.py'):
            script_files[task_scripts/name]='scripts/training/'+name
        if include_test:
            script_files[task_scripts/'evaluate_fire_sticker.ps1']='evaluate.ps1'
    if model == 'building_name_paper':
        task_scripts=root / 'lab/features/fire/training/scripts'
        for name in ('train_building_name_paper_v4.py','paper_report.py'):
            script_files[task_scripts/name]='scripts/training/'+name
        if include_test:
            script_files[task_scripts/'evaluate_building_name_paper.ps1']='evaluate.ps1'
    if model == 'sign':
        task_scripts = root / 'lab/features/sign/training/scripts'
        for name in ('train_sign_v4.py', 'sign_report.py'):
            script_files[task_scripts / name] = 'scripts/training/' + name
        if include_test:
            script_files[task_scripts / 'evaluate_sign.ps1'] = 'evaluate.ps1'
    if model == 'waste_sticker':
        task_entry = root / 'lab/features/waste_bin/training/scripts/train_waste_sticker_v4.py'
        script_files[task_entry] = 'scripts/training/train_waste_sticker_v4.py'
        if include_test:
            task_evaluate = root / 'lab/features/waste_bin/training/scripts/evaluate_waste_sticker.ps1'
            script_files[task_evaluate] = 'evaluate.ps1'
    if model == 'waste_bin_state':
        task_entry = root / 'lab/features/waste_bin/training/scripts/train_waste_bin_state_v4.py'
        script_files[task_entry] = 'scripts/training/train_waste_bin_state_v4.py'
        if include_test:
            task_evaluate = root / 'lab/features/waste_bin/training/scripts/evaluate_waste_bin_state.ps1'
            script_files[task_evaluate] = 'evaluate.ps1'
    launcher_yaml = root / COMMON / 'launcher.yaml'
    required = [data, config_file, weights, source_path, engine / 'cli.py', launcher_yaml, *script_files,
                *(root / PORTABLE / name for name in ('run.ps1', 'bootstrap.py', 'manifest.json', 'requirements.lock'))]
    missing = sorted({path.relative_to(root).as_posix() for path in required if not path.is_file()})
    if not (root / PORTABLE / 'site-packages').is_dir():
        missing.append((PORTABLE / 'site-packages').as_posix())
        missing.sort()
    report = {'task': task, 'model_key': model, 'version': 'v4', 'status': 'blocked',
              'output': destination.relative_to(root).as_posix(), 'dataset': data.relative_to(root).as_posix(),
              'audited_sources': source_path.relative_to(root).as_posix(), 'missing_assets': missing, 'issues': [],
              'portable_launcher': _launcher_report(root)}
    if destination.exists() or destination.is_symlink():
        report['issues'].append('Output already exists; existing packages are never replaced')
    if destination.with_name('.' + destination.name + '.build.lock').exists():
        report['issues'].append('Output is locked by a package build')
    if weights.suffix.lower() != '.pt':
        report['issues'].append('Initialization must be a local .pt file')
    amp_model = None
    config_source_hash = None
    if config_file.is_file():
        try:
            config_source_hash = _file_sha256(config_file)
            config = read_yaml(config_file)
            report.update(training_config=config_file.relative_to(root).as_posix(), configured_epochs=config.get('epochs'),
                          budget_status='Epoch and wall-clock budgets are recorded in the package configuration')
            config.update(data=data.relative_to(root).as_posix(), model=weights.relative_to(root).as_posix())
            # The explicitly supplied data YAML defines the package sources.
            config.pop('data_overrides', None)
            validate(config)
            if (config['version'].lower() != 'v4' or config['model_key'] != model or 'training_policy' not in config
                    or config['task'] != ('pose' if task == 'meter' else 'detect')):
                raise ValueError('Config must target this V4 model and declare a training policy')
            if config.get('amp'):
                if not config.get('amp_check_model'):
                    raise ValueError('amp requires a local amp_check_model')
                amp_model = resolve(root, config['amp_check_model'])
                if amp_model.suffix.lower() != '.pt' or not amp_model.is_file():
                    raise ValueError('Missing local AMP check .pt: ' + config['amp_check_model'])
        except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
            report['issues'].append(str(exc))
    if missing or report['issues']:
        return report, None
    try:
        launcher_source_hash = _file_sha256(launcher_yaml)
        launcher_defaults = launcher_settings(launcher_yaml, None)
        launcher_defaults.pop('train_config')
        if wall_seconds is not None:
            launcher_defaults['wall_seconds'] = _positive_seconds(wall_seconds)
        resolve(root, source_path.relative_to(root))
        header, splits, inspection = preflight_dataset(root, config)
        if model == 'meter':
            from engine.meter_report import POINTS
            if header.get('kpt_shape') != [8, 3] or tuple(config.get('keypoint_names', ())) != POINTS:
                raise ValueError('Meter package requires kpt_shape [8, 3] and the named eight-point order')
        if include_test:
            splits['test'] = records(root, config, 'test', training_mode='fixed_validation')
            if not splits['test']:
                raise ValueError('The requested packaged Test split is empty')
            _require_v4_review({'test': splits['test']})
        source_manifest = _read_json(source_path)
        by_image = {(row['split'], row['image']): row for row in source_manifest['records']
                    if row.get('split') in splits}
        selected = [(item, by_image[split, item['image']]) for split, items in splits.items() for item in items]
        evidence = _evidence_paths(root, source_path, [raw for _, raw in selected], model)
        if (_file_sha256(config_file) != config_source_hash or
                _file_sha256(launcher_yaml) != launcher_source_hash):
            raise ValueError('Training configuration changed while preparing package')
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        report['issues'].append(str(exc))
        return report, None
    report.update(status='ready', training_mode=inspection['training_mode'],
                  dataset_counts={split: len(items) for split, items in splits.items()},
                  training_dataset_counts=inspection['available_images'], label_counts=_label_counts(splits, header['names']),
                  included_test_for_evaluation=include_test, configured_wall_seconds=launcher_defaults['wall_seconds'],
                  evidence_files=len(set(evidence.values())), initialization=config['model'],
                  validation_scope='Policy-selected training inputs')
    for field in ('source_reuse', 'source_group_unknown_counts', 'development_contact_counts', 'evaluation_scope', 'exposure_warnings',
                  'class_coverage_warnings'):
        if field in inspection:
            report[field] = inspection[field]
    return report, (destination, config, header, source_manifest, selected, evidence, script_files, weights, amp_model,
                    launcher_defaults, config_file, launcher_yaml, config_source_hash, launcher_source_hash)


def _copy(src, dst, root):
    if not src.resolve().is_relative_to(root):
        raise ValueError('Source escapes workspace: ' + str(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if not filecmp.cmp(src, dst, shallow=False):
        raise RuntimeError('Source changed while copying: ' + str(src))


def _code_fingerprint(target, files):
    hashes = {name: _file_sha256(target / name) for name in sorted(files)}
    encoded = json.dumps(hashes, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return {'algorithm': 'sha256', 'digest': hashlib.sha256(encoded).hexdigest(), 'files': hashes}


def _populate(root, target, report, plan):
    (_, config, header, source_manifest, selected, evidence, scripts, weights, amp_model,
     launcher_defaults, config_file, launcher_yaml, config_source_hash, launcher_source_hash) = plan
    model = report['model_key']
    # Check the complete source set before and after packing to avoid a package
    # assembled from independently modified versions of its scripts/configs.
    source_files = set(scripts) | {config_file, launcher_yaml, weights}
    source_files.update(root / PORTABLE / name for name in ('run.ps1', 'bootstrap.py', 'manifest.json', 'requirements.lock'))
    # The optional 3.14 bundle is shared, not copied into each task package.
    # Record both available manifests and check that this set remains stable.
    environment_manifests = {'.': root / PORTABLE / 'manifest.json',
                             'cp314': root / PORTABLE / 'cp314/manifest.json'}
    available_manifests = {name: path for name, path in environment_manifests.items() if path.is_file()}
    source_files.update(available_manifests.values())
    if amp_model is not None:
        source_files.add(amp_model)
    source_hashes = {path: _file_sha256(path) for path in source_files}
    engine_sources = set((root / COMMON / 'engine').glob('*.py'))
    if not engine_sources.issubset(source_files):
        raise RuntimeError('Training engine files changed between preparation and packaging')
    if (source_hashes[config_file] != config_source_hash or
            source_hashes[launcher_yaml] != launcher_source_hash):
        raise RuntimeError('Training configuration changed between preparation and packaging')
    dataset = target / 'dataset'
    source_rows = []
    for index, (item, provenance) in enumerate(selected):
        split = item['split']
        image_source, label_source = (root / item[key] for key in ('image', 'label'))
        stem = f'{index:07d}_{image_source.stem}'
        image = dataset / 'images' / split / (stem + image_source.suffix.lower())
        label = dataset / 'labels' / split / (stem + '.txt')
        _copy(image_source, image, root)
        _copy(label_source, label, root)
        if _file_sha256(label) != item['label_sha256']:
            raise RuntimeError('Audited label changed after preflight: ' + item['label'])
        if item.get('image_sha256') and _file_sha256(image) != item['image_sha256']:
            raise RuntimeError('Audited image changed after preflight: ' + item['image'])
        row = deepcopy(provenance)
        row.update(image=image.relative_to(target).as_posix(), label=label.relative_to(target).as_posix(),
                   workspace_image=item['workspace_image'], workspace_label=item['workspace_label'])
        selected_tasks = row.get('selected_tasks')
        if isinstance(selected_tasks, list) and model in selected_tasks:
            row['selected_tasks'] = [model]
            row['tasks'] = {model: row['tasks'][model]}
        source_rows.append(row)
    copied_evidence = {}
    for index, path in enumerate(sorted(set(evidence.values()))):
        copied = dataset / 'source_records' / f'{index:04d}_{path.name}'
        _copy(path, copied, root)
        copied_evidence[path] = copied.relative_to(target).as_posix()
    _write_json(dataset / 'source_records/evidence_map.json', {ref: copied_evidence[path] for ref, path in evidence.items()})
    _write_json(dataset / 'workspace_sources.json', {**source_manifest, 'records': source_rows})
    data_spec = {**header, **{split: 'images/' + split for split in report['dataset_counts']}}
    if report['training_mode'] == 'fixed':
        data_spec['val'] = None
    (dataset / 'data.yaml').write_text(yaml.safe_dump(data_spec, allow_unicode=True, sort_keys=False), encoding='utf-8')

    for source, destination in scripts.items():
        _copy(source, target / destination, root)
    entry = {'fire_sticker': FIRE_ENTRY, 'building_name_paper': PAPER_ENTRY, 'waste_sticker': STICKER_ENTRY,
             'waste_bin_state': BIN_STATE_ENTRY, 'sign': SIGN_ENTRY, 'meter': METER_ENTRY}.get(model, TRAIN_ENTRY)
    (target / 'scripts/train.py').write_text(entry, encoding='utf-8')
    _copy(weights, target / 'weights/pretrained.pt', root)
    packaged_config = {**config, 'data': 'dataset/data.yaml', 'model': 'weights/pretrained.pt',
                       'output_root': 'runs/' + model}
    packaged_config.pop('amp_check_model', None)
    if amp_model is not None:
        _copy(amp_model, target / 'weights/amp_check.pt', root)
        packaged_config['amp_check_model'] = 'weights/amp_check.pt'
    (target / 'config').mkdir()
    (target / 'config/launcher.yaml').write_text(yaml.safe_dump(launcher_defaults, sort_keys=False), encoding='utf-8')
    (target / 'config/train.yaml').write_text(yaml.safe_dump(packaged_config, allow_unicode=True, sort_keys=False),
                                             encoding='utf-8')
    _, _, packaged = preflight_dataset(target, packaged_config)
    if packaged['available_label_counts'] != {key: report['label_counts'][key]
                                             for key in packaged['available_label_counts']}:
        raise RuntimeError('Packaged label counts differ from the reviewed source')
    if report['included_test_for_evaluation']:
        test_records = records(target, packaged_config, 'test', training_mode='fixed_validation')
        _require_v4_review({'test': test_records})
        if _label_counts({'test': test_records}, header['names'])['test'] != report['label_counts']['test']:
            raise RuntimeError('Packaged Test labels differ from the reviewed source')
    if (set((root / COMMON / 'engine').glob('*.py')) != engine_sources or
            {name for name, path in environment_manifests.items() if path.is_file()} != set(available_manifests) or
            any(_file_sha256(path) != digest for path, digest in source_hashes.items())):
        raise RuntimeError('Training source or configuration changed while packaging')
    fingerprint = _code_fingerprint(target, [*scripts.values(), 'scripts/train.py',
                                             'config/train.yaml', 'config/launcher.yaml'])
    # The source launcher can change after planning; report what was checked at build time.
    report['portable_launcher'] = _launcher_report(root)
    manifest = {**report, 'status': 'built', 'package_kind': 'aiaec_v4_training',
                'code_fingerprint': fingerprint,
                'initial_weights_sha256': _file_sha256(target / packaged_config['model']),
                'source_index_sha256': _file_sha256(dataset / 'workspace_sources.json'),
                'portable_environment': '../portable_env',
                'portable_environment_manifest_sha256': source_hashes[root / PORTABLE / 'manifest.json'],
                'portable_environment_manifests_sha256': {
                    name: source_hashes[path] if name in available_manifests else None
                    for name, path in environment_manifests.items()},
                'python_included': False,
                'runtime_requirement': 'Windows x64 CPython 3.12 or 3.14; matching GPU driver and VC runtime'}
    _write_json(target / 'package_manifest.json', manifest)
    (target / 'README.md').write_text(PACKAGE_README.format(model=model), encoding='utf-8')
    if model == 'meter':
        policy = config['training_policy']
        with (target / 'README.md').open('a', encoding='utf-8') as guide:
            guide.write('\n## 温湿度计八点 Pose\n\n'
                '单类 meter_dial；点序固定为温度轴心、温度针尖、湿度轴心、湿度针尖、'
                '温度最小刻度、温度最大刻度、湿度最小刻度、湿度最大刻度。'
                '使用既有八点 V3 权重微调，框、坐标和点存在性共同训练，所有头保持可训练。\n\n'
                f'最多 {config["epochs"]} 轮，策略 {policy["mode"]}。')
            if policy['mode'] == 'early_stopping':
                guide.write(f'前 {policy["min_epochs"]} 轮不累计耐心，之后连续 {policy["patience"]} 轮'
                    f'未达到 {policy["min_delta"]} 的 Val Pose mAP50–95 提升则停止；'
                    f'平台期最早第 {policy["min_epochs"] + policy["patience"]} 轮停止。')
            guide.write('严格最高的 Val Pose mAP50–95 保存为 best.pt，同分保留较早轮次。'
                f'单次启动硬时限 {launcher_defaults["wall_seconds"]} 秒，包含暂存、训练及训练后验证；'
                '超时记为中断，不等同完成，不自动重启。续训保留原轮数和完整检查点状态，'
                '另一次启动的时间预算需计入用户允许的总预算。\n\n'
                f'输入 {config["imgsz"]}，batch {config["batch"]}，AdamW；禁用翻转、Mosaic、MixUp、'
                'copy-paste 和额外透视。仅用轻量旋转、平移、尺度和 HSV；'
                '不追加会破坏小表盘细节的在线模糊。显存不足时显式减小 batch，输入尺寸保持不变。\n\n'
                '训练后 Val 和显式 Val/Test 评估自动生成 meter_report.json 与 meter_needle_errors.csv：'
                '八点像素及表盘短边归一化误差、两根针的独立方向角误差、短于10像素的针、'
                '表盘尺寸分组、漏检、点不可用及负图误报。真值可见性1/2均参加坐标监督，0为未知；'
                '预测点置信度至少0.5才计入专项误差。缺失保留在分母中，不当作零误差。'
                '角度是投影平面的诊断值，不是透视校正后的温湿度读数。\n\n'
                'meter-report 可由已保存的预测重建专项报告；原 keypoint-report 命令保留。'
                'Train/Val 用于训练与选模，Test 仅在冻结权重后使用。共享 Blender 物料和既有实拍接触'
                '随来源清单保留，评估属于当前数据域诊断。\n\n')
            if report['included_test_for_evaluation']:
                guide.write('训练结束后执行：\n\n```powershell\n'
                            '.\\evaluate.ps1 -Run "runs/meter/<运行目录>"\n```\n\n'
                            '顺序为完整 Val、冻结最佳权重、Test；各阶段失败即停止，已有评估不覆盖。\n')
    if model == 'fire_sticker':
        with (target/'README.md').open('a',encoding='utf8') as guide:
            guide.write('\n## Fire-sticker training\n\n'
                'One class: fire_sticker. Red lamps, bins and cones are context, never extra target classes. '
                'Ordinary negatives and reviewed hard negatives retain empty YOLO labels. Positive-with-distractor '
                'images retain every visible sticker and its known building. All metadata travels with this package.\n\n'
                'The configured ceiling is 150 epochs. The first 120 epochs never count against early-stopping '
                'patience; thereafter 15 epochs without a validation mAP50-95 improvement of at least 0.001 stop '
                'the run. The strictly best validation checkpoint is still retained. The 36000-second launcher '
                'limit includes setup and postprocessing, and takes precedence; timeout is not completion. '
                'Resume requires a compatible incomplete run and its complete optimizer/scheduler/RNG state. '
                'Each launch has its own time limit; do not treat a restart as authorization for extra total time.\n\n'
                'Input 640, batch 16, AdamW, modest HSV variation and scale 0.05 preserve small stickers and '
                'color cues. Flips, Mosaic, MixUp, copy-paste and extra perspective are disabled. On a smaller '
                'GPU explicitly reduce batch; do not silently resize inputs or switch training semantics.\n\n'
                '`scripts/train.py audit` describes source branches, ordinary/hard negatives, distractor-positive '
                'coverage, input-pixel size bins and known building labels. It is a data audit, not a model score. '
                '`fire-report` consumes saved predictions for size/branch/context/building recall, exact-image '
                'success, count error and negative false positives. Unknown metadata remains unknown. '
                'False detections are not assigned to a building without spatial evidence.\n\n'
                'Shared assets, repeated parent photos/video frames and development contact make Val/Test '
                'controlled diagnostics. Test does not select epochs, checkpoints or thresholds. '
                'This package has not started training or model evaluation.\n\n'
                '```powershell\n.\\scripts\\run_training_tool.ps1 -Entry (Join-Path $PWD \'scripts/train.py\') -ScriptArguments @(\'audit\')\n'
                '.\\train.ps1\n.\\train.ps1 -Resume\n```\n')
            if report['included_test_for_evaluation']:
                guide.write('\nAfter training: `.\\evaluate.ps1 -Run "runs/fire_sticker/<run-directory>"`. '
                    'The sequence is full Val, fire report, frozen selection, Test and fire report. '
                    'Existing evaluations are not overwritten; every failure stops the sequence.\n')
    if model == 'building_name_paper':
        with (target/'README.md').open('a',encoding='utf8') as guide:
            guide.write('\n## Building-name paper detection\n\n'
                'One class: paper; no OCR supervision. Budget, optimizer, mild augmentation and validation selection '
                'are fixed in config/train.yaml. The launcher wall_seconds is a hard limit including training setup '
                'and postprocessing; timeout does not mean completion. Resume uses complete checkpoints and the '
                'original training semantics. Batch/device may be adjusted explicitly for the destination GPU.\n\n'
                'scripts/train.py paper-report reports input-pixel sizes, generated yaw/height controls, building '
                'counts, partial-paper layouts, branches, source groups, exact-image detection and empty-scene '
                'false positives. Unknown real camera views remain unknown; false positives receive image context '
                'rather than an invented building ID. paper-sizes remains available for compatible size reports. '
                'Most generated inputs are native 640 pixels; larger model inputs do not recover lost detail. '
                'Shared building assets and prior development exposure make Val/Test controlled diagnostics.\n\n')
            if report['included_test_for_evaluation']:
                guide.write('Train/Val select the checkpoint; Test is evaluated only after selection is frozen. '
                    'After training, run the actual directory below. Existing evaluations are never overwritten.\n\n'
                    '```powershell\n.\\evaluate.ps1 -Run "runs/building_name_paper/<run-directory>"\n```\n')
    if model == 'sign':
        with (target / 'README.md').open('a', encoding='utf-8') as guide:
            guide.write('\n## Sign detection and regression evaluation\n\n'
                        'One class: sign_fixture. Train/Val select the best checkpoint; Test is excluded from selection. '
                        'The configured epoch ceiling, early stopping and hard wall-clock limit are recorded in config/. '
                        'Confirmed per-instance views, visibility, layout and original-coordinate boxes remain in '
                        'dataset/workspace_sources.json. Inherited objects with unknown views stay unknown.\n\n'
                        'scripts/train.py sign-report reads an evaluation and this metadata to report four views, '
                        'complete/partial/occluded/pole-only fixtures, short-side-under32 recall, source groups, '
                        'whole-image success and empty-scene false positives. Shared physical foreground and historical '
                        'Test contact make these controlled diagnostics, not independent field generalization.\n\n')
            if report['included_test_for_evaluation']:
                guide.write('After training, validate the selected checkpoint, freeze it and evaluate Test:\n\n'
                            '```powershell\n.\\evaluate.ps1 -Run "runs/sign/<run-directory>"\n```\n'
                            'Both evaluations write sign_report.json. Generic check/train/freeze/evaluate and '
                            'the compatible view-report command are available through scripts/train.py.\n')
    if model == 'waste_sticker':
        with (target / 'README.md').open('a', encoding='utf-8') as guide:
            guide.write('\n## Waste-sticker diagnostics\n\n'
                        'Validation and Test evaluations write `waste_sticker_class_report.json` with '
                        '24-class precision/recall, image success rates for zero through four stickers, '
                        'and false positives on empty-label images. Fixed artwork and scene reuse '
                        'are recorded as controlled source diagnostics.\n\n')
            if report['included_test_for_evaluation']:
                counts = ', '.join(f'{split} {count}' for split, count in report['dataset_counts'].items())
                guide.write(f'This package contains {counts} images. '
                            'Training selects checkpoints using Val. After training, run the following '
                            'with the actual run directory to validate the selected checkpoint, '
                            'freeze its inputs and evaluate Test.\n\n'
                            '```powershell\n.\\evaluate.ps1 -Run "runs/waste_sticker/<run-directory>"\n```\n')
    if model == 'waste_bin_state':
        with (target / 'README.md').open('a', encoding='utf-8') as guide:
            guide.write('\n## Waste-bin-state diagnostics\n\n'
                        'The dataset classes are `open` and `closed`, with visible whole-bin boxes. '
                        'Per-bin colors, states, original image sizes and generation axes are kept '
                        'in `dataset/workspace_sources.json`. Run `scripts/train.py bin-state-report` '
                        'with an evaluation directory to report box misses, state confusion, '
                        'whole-image success and false positives on empty scenes. '
                        'The older `color-state-report` command accepts an explicit reviewed color file.\n\n')
            if report['included_test_for_evaluation']:
                counts = ', '.join(f'{split} {count}' for split, count in report['dataset_counts'].items())
                guide.write(f'This package contains {counts} images. '
                            'Training uses Train and Val. After training, use the completed run directory '
                            'to select and freeze its checkpoint, then evaluate Test.\n\n'
                            '```powershell\n.\\evaluate.ps1 -Run "runs/waste_bin_state/<run-directory>"\n```\n')
    return manifest


def package(task, data, config, weights, *, output_name=None, plan_only=False, include_test=False,
            wall_seconds=None, root=ROOT):
    root = Path(root).resolve()
    report, plan = _prepare(root, task, data, config, weights, output_name,
                           include_test=include_test, wall_seconds=wall_seconds)
    if plan_only or plan is None:
        return report
    destination = plan[0]
    lock = destination.with_name('.' + destination.name + '.build.lock')
    with lock.open('xb'):
        pass
    try:
        with TemporaryDirectory(prefix='.v4-training-', dir=destination.parent) as temporary:
            staged = Path(temporary) / destination.name
            staged.mkdir()
            manifest = _populate(root, staged, report, plan)
            if destination.exists() or destination.is_symlink():
                raise FileExistsError('Output appeared during build; refusing replacement: ' + str(destination))
            staged.rename(destination)
    finally:
        lock.unlink()
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=TASK_MODELS)
    parser.add_argument('--data', required=True, help='Workspace-relative audited data.yaml with adjacent workspace_sources.json')
    parser.add_argument('--config', required=True, help='Workspace-relative V4 training config with explicit training_policy')
    parser.add_argument('--weights', required=True, help='Workspace-relative trusted local .pt initialization')
    parser.add_argument('--output-name', help='New folder name under training_packages; defaults to the model key')
    parser.add_argument('--plan-only', action='store_true', help='Check requirements without writing a package')
    parser.add_argument('--include-test', action='store_true', help='include reviewed Test inputs for post-selection evaluation')
    parser.add_argument('--wall-seconds', type=float, help='set the packaged training launcher wall-clock limit')
    args = parser.parse_args(argv)
    try:
        report = package(args.task, args.data, args.config, args.weights, output_name=args.output_name,
                         plan_only=args.plan_only, include_test=args.include_test, wall_seconds=args.wall_seconds)
    except (OSError, ValueError, RuntimeError, yaml.YAMLError) as exc:
        print('ERROR:', exc, file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report['status'] == 'blocked' else 0


if __name__ == '__main__':
    raise SystemExit(main())
