"""Copy the selected YOLO splits into independent portable training packages.

This command does not train or download anything. Shared source material is
allowed; report evaluation results as applying to the supplied splits.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'training_packages'
TEMPLATE = ROOT / 'lab' / 'training_common'
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
TASKS = {
    'person': ('datasets/exports/person/yolo_scene_candidates_4400_20261110/data.yaml', 'detect', 150),
    'e_bike': ('datasets/exports/e_bike/detection_candidates_4400_20261110/data.yaml', 'detect', 150),
    'fire_sticker': ('datasets/exports/fire/fire_detection_candidates_4240_20261110/data.yaml', 'detect', 150),
    'building_name_paper': ('datasets/exports/fire/name_paper_detection_candidates_4240_20261110/data.yaml', 'detect', 150),
    'sign': ('datasets/exports/sign/detection_mixed_candidates_3400_20261110/data.yaml', 'detect', 120),
    'waste_bin_state': ('datasets/exports/waste_bin/bin_and_sticker_candidate_splits_v2_20261110/state_candidates.yaml', 'detect', 120),
    'waste_sticker': ('datasets/exports/waste_bin/bin_and_sticker_candidate_splits_v2_20261110/sticker_candidates_12eval.yaml', 'detect', 160),
    'meter': ('datasets/exports/meter/opencv_rendered_meter_candidates_5000_20261110/data.yaml', 'pose', 180),
    'plate_detection': ('datasets/exports/plate/provincial_synthetic/round2_multi_car/dataset.yaml', 'detect', 120),
}
NOTES = {
    'person': '检测社区人员与非社区人员立牌。',
    'e_bike': '检测直立及倒伏电动车；停车区总数与倒伏数由检测结果另行汇总。',
    'fire_sticker': '检测火焰纸贴，不检测楼体。',
    'building_name_paper': '检测楼名纸条位置；文字识别需要单独的 OCR。',
    'sign': '检测牌面、杆和底座组成的整件牌具。',
    'waste_bin_state': '检测垃圾桶开盖和闭盖状态。',
    'waste_sticker': '检测 24 类垃圾贴纸；类别 ID 对应的素材名称、规则名称和分类见 dataset/class_names_zh.yaml；验证和测试复用了训练图案。',
    'meter': '检测表盘并定位 8 个关键点，顺序见 dataset/keypoints.yaml；训练、验证和测试共享渲染批次。',
    'plate_detection': '训练和验证使用省赛合成车牌，测试使用 CCPD 道路实拍；检测位置，完整车牌识别需要单独的 OCR。',
}

STICKER_EVAL = {
    'val': 'datasets/exports/waste_bin/bin_state_val_sticker_scene_diag_v3_120_20261110/sticker/images',
    'test': 'datasets/exports/waste_bin/bin_state_test_sticker_scene_diag_v3_120_20261110/sticker/images',
}
STICKER_NAMES = [
    'plastic_bottle', 'can', 'glass_bottle', 'newspaper', 'cardboard_box', 'clothes',
    'leftover_food', 'vegetable_leaves', 'fruit_core', 'tea_leaves', 'bread', 'fish_bone',
    'button_battery', 'fluorescent_lamp', 'mercury_thermometer', 'pill', 'paint', 'film',
    'cigarette_butt', 'used_tissue', 'disposable_chopsticks', 'broken_ceramics', 'shell', 'discarded_glasses',
]
METER_KEYPOINTS = [
    ('temperature_pivot', '温度轴心'), ('temperature_tip', '温度针尖'),
    ('humidity_pivot', '湿度轴心'), ('humidity_tip', '湿度针尖'),
    ('temperature_min', '温度最小刻度'), ('temperature_max', '温度最大刻度'),
    ('humidity_min', '湿度最小刻度'), ('humidity_max', '湿度最大刻度'),
]
PLATE_TEST = {'test': 'datasets/exports/plate/provincial_ccpd/detection/images/test'}
PLATE_EXTRA = {
    'train': 'datasets/exports/plate/provincial_synthetic/synthetic_train/detector/images/train',
    'val': 'datasets/exports/plate/provincial_synthetic/synthetic_train/detector/images/val',
}

TASK_OWNERS = {'fire_sticker': 'fire', 'building_name_paper': 'fire',
               'waste_bin_state': 'waste_bin', 'waste_sticker': 'waste_bin',
               'plate_detection': 'plate'}


def workspace_config(name):
    owner = TASK_OWNERS.get(name, name)
    return ROOT / 'lab/features' / owner / 'training/config' / (name + '_v3.yaml')


def same_file_bytes(left, right):
    """Compare contents directly, without a checksum or persisted fingerprint."""
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open('rb') as first, right.open('rb') as second:
        while True:
            block = first.read(1024 * 1024)
            if block != second.read(1024 * 1024):
                return False
            if not block:
                return True


def verify_mirror_data(package, config):
    """Check source selection, image bytes and label text without changing data."""
    sys.path.insert(0, str(TEMPLATE))
    from engine.data import resolve_sources, dataset_header
    expected_header = dataset_header(ROOT, config['data'])
    actual = yaml.safe_load((package / 'dataset/data.yaml').read_text(encoding='utf-8'))
    expected_names = config.get('class_names', expected_header['names'])
    actual_names = actual['names']
    if isinstance(actual_names, dict):
        actual_names = [actual_names[i] for i in range(len(actual_names))]
    if list(expected_names) != actual_names or expected_header.get('kpt_shape') != actual.get('kpt_shape'):
        raise ValueError('Mirror class/keypoint definitions differ from workspace: ' + str(package))
    counts, provenance = {}, []
    for split in ('train', 'val', 'test'):
        selected = {}
        for source in resolve_sources(ROOT, config, split):
            for image in sorted(source['images'].rglob('*')):
                if not image.is_file() or image.suffix.lower() not in EXTENSIONS:
                    continue
                if image.name in selected:
                    raise ValueError('Ambiguous flattened mirror filename: ' + image.name)
                relative = image.relative_to(source['images'])
                label = source['labels'] / relative.with_suffix('.txt')
                selected[image.name] = (image, label, source['source_group'])
        images = package / 'dataset/images' / split
        labels = package / 'dataset/labels' / split
        actual_images = {p.name for p in images.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS}
        if set(selected) != actual_images:
            raise ValueError('Mirror split differs from workspace source selection: ' + split)
        if {Path(n).stem + '.txt' for n in selected} != {p.name for p in labels.glob('*.txt')}:
            raise ValueError('Mirror image/label names differ: ' + split)
        for name, (image, label, group) in selected.items():
            mirrored_label = labels / (Path(name).stem + '.txt')
            if not same_file_bytes(image, images/name) or label.read_text(encoding='utf-8').split() != mirrored_label.read_text(encoding='utf-8').split():
                raise ValueError('Mirror image bytes or label mismatch: ' + name)
            provenance.append({'image': (images/name).relative_to(package).as_posix(),
                               'label': mirrored_label.relative_to(package).as_posix(), 'split': split,
                               'source_group': group,
                               'workspace_image': image.relative_to(ROOT).as_posix(),
                               'workspace_label': label.relative_to(ROOT).as_posix()})
        counts[split] = len(selected)
    return counts, provenance


def refresh_code(name):
    """Refresh managed V3 code/config only; preserve datasets, weights and runs."""
    package = inside_workspace(DEST / name)
    if not (package / 'dataset/data.yaml').is_file():
        raise FileNotFoundError('Existing dataset required: {}'.format(package))
    config = yaml.safe_load(workspace_config(name).read_text(encoding='utf-8'))
    engine = TEMPLATE / 'engine'
    required = [engine / 'cli.py', engine / 'runner.py', TEMPLATE / 'launch_v3.py',
                TEMPLATE / 'start_v3.ps1', TEMPLATE / 'tensorboard_v3.py']
    for path in required:
        if not path.is_file():
            raise FileNotFoundError('V3 source incomplete: {}'.format(path))
    counts, provenance = verify_mirror_data(package, config)
    # All filenames overwritten here are generator-owned entry points.
    scripts = package / 'scripts'
    common = scripts / 'common'
    (common / 'engine').mkdir(parents=True, exist_ok=True)
    for source in sorted(engine.glob('*.py')):
        shutil.copy2(source, common / 'engine' / source.name)
    shutil.copy2(TEMPLATE / 'train.py', common / 'train.py')
    obsolete_viewer = scripts / 'tensorboard.py'
    if obsolete_viewer.is_file():
        if obsolete_viewer.read_text(encoding='utf-8') != (TEMPLATE / 'tensorboard_v3.py').read_text(encoding='utf-8'):
            raise ValueError('Old managed viewer was modified; preserve it before refreshing: ' + str(obsolete_viewer))
        obsolete_viewer.unlink()
    for source, target in [('launch_v3.py', 'launch.py'), ('start_v3.ps1', 'start_training.ps1'),
                           ('tensorboard_v3.py', 'view_tensorboard.py')]:
        shutil.copy2(TEMPLATE / source, scripts / target)
    entry = ('"""V3 mirror; edit the workspace task source, then refresh this package."""\n'
             'import sys\nfrom pathlib import Path\n'
             'ROOT = Path(__file__).resolve().parents[1]\n'
             "sys.path.insert(0, str(ROOT / 'scripts/common'))\n"
             'from engine.cli import main\n'
             "if __name__ == '__main__':\n"
             "    raise SystemExit(main(default_root=ROOT, default_config='config/train.yaml'))\n")
    (scripts / 'train.py').write_text(entry, encoding='utf-8')
    # Keep old links usable while every executable implementation lives in scripts/.
    for legacy, target in [('train.py', 'train.py'), ('tensorboard_log.py', 'view_tensorboard.py')]:
        (package / legacy).write_text(
            "import runpy\nfrom pathlib import Path\nif __name__ == '__main__':\n"
            "    runpy.run_path(str(Path(__file__).resolve().parent / 'scripts' / "
            + repr(target) + "), run_name='__main__')\n", encoding='utf-8')
    (package / 'start_training.ps1').write_text(
        "& (Join-Path $PSScriptRoot 'scripts/start_training.ps1') @args\nexit $LASTEXITCODE\n", encoding='utf-8')
    config.pop('data_overrides', None)
    config['data'] = 'dataset/data.yaml'
    config['model'] = 'weights/' + Path(config['model']).name
    config['output_root'] = 'runs/' + name
    (package / 'runs').mkdir(exist_ok=True)
    yaml_write(package / 'config/train.yaml', config)
    for filename in ('requirements.in', 'requirements-win-cp312-cu128.lock'):
        source = TEMPLATE / 'portable_env' / filename
        if source.is_file():
            shutil.copy2(source, package / ('requirements.lock' if filename.endswith('.lock') else filename))
    (package / 'requirements.txt').write_text(
        '# Use the sibling portable_env bundle with scripts/start_training.ps1.\n'
        '# Source build instructions and the full lock are in portable_env/README.md.\n', encoding='utf-8')
    data = yaml.safe_load((package / 'dataset/data.yaml').read_text(encoding='utf-8'))
    if config.get('class_names'):
        data['names'] = config['class_names']
        yaml_write(package / 'dataset/data.yaml', data)
    write_package_readme(package, name, config['epochs'])
    manifest = {'version': 'v3', 'workspace_config': workspace_config(name).relative_to(ROOT).as_posix(),
                'workspace_entry': 'lab/features/{}/training/scripts/train_{}.py'.format(TASK_OWNERS.get(name, name), name),
                'common_source': 'lab/training_common/engine',
                'refresh_scope': 'managed code and V3 config; existing images, labels, weights and runs preserved'}
    (package / 'mirror_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (package / 'dataset/workspace_sources.json').write_text(json.dumps(
        {'schema_version': 1, 'counts': counts, 'records': provenance}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return manifest


def mirror_environment():
    """Copy one shared dependency bundle beside task mirrors, never into each task."""
    source = TEMPLATE / 'portable_env/bundles/win-cp312-cu128-v3'
    target = inside_workspace(DEST / 'portable_env')
    for minor in (13, 14):
        extra = TEMPLATE / ('portable_env/bundles/win-cp3{}-cu128-20261111'.format(minor))
        if not (extra / 'manifest.json').is_file():
            raise FileNotFoundError('Build the Python 3.{} dependency bundle first: {}'.format(minor, extra))
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    if target.exists():
        previous = json.loads((target / 'manifest.json').read_text(encoding='utf-8'))
        for key in ('packages', 'python_version', 'platform', 'backend'):
            if previous[key] != manifest[key]:
                raise ValueError('Existing dependency bundle differs; choose a new distribution directory before updating it')
        for name in ('bootstrap.py', 'run.ps1', 'README.md', 'requirements.lock', 'manifest.json'):
            shutil.copy2(source / name, target / name)
    else:
        shutil.copytree(source, target)
    for minor in (13, 14):
        extra = TEMPLATE / ('portable_env/bundles/win-cp3{}-cu128-20261111'.format(minor))
        destination = target / ('cp3{}'.format(minor))
        if destination.exists():
            existing = json.loads((destination / 'manifest.json').read_text(encoding='utf-8'))
            incoming = json.loads((extra / 'manifest.json').read_text(encoding='utf-8'))
            if existing['packages'] != incoming['packages'] or existing['python_version'] != incoming['python_version']:
                raise ValueError('Existing dependency bundle differs: ' + str(destination))
        else:
            shutil.copytree(extra, destination)
    return {'bundle': target.relative_to(ROOT).as_posix(), 'python_versions': ['3.12', '3.13', '3.14'],
            'backend': manifest['backend']}


def inside_workspace(path):
    path = path.resolve()
    path.relative_to(ROOT.resolve())
    return path


def yaml_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding='utf-8')


def waste_class_mapping(original_names, items):
    """Keep source labels and the project's canonical competition names."""
    return {
        'names': original_names,
        'competition_names': {index: items[name]['name'] for index, name in original_names.items()},
        'categories': {index: items[name]['category'] for index, name in original_names.items()},
    }


def labels_for(images):
    parts = list(images.parts)
    indices = [i for i, part in enumerate(parts) if part == 'images']
    if not indices:
        raise ValueError('Images directory lacks images component: {}'.format(images))
    parts[indices[-1]] = 'labels'
    return Path(*parts)


def source_sets(config_path, overrides=None):
    data = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    base = Path(data.get('path', config_path.parent))
    if not base.is_absolute():
        base = config_path.parent / base
    base = inside_workspace(base)
    for split in ('train', 'val', 'test'):
        images = inside_workspace(ROOT / overrides[split]) if overrides and split in overrides else inside_workspace(base / data[split])
        labels = inside_workspace(labels_for(images))
        if not images.is_dir() or not labels.is_dir():
            raise ValueError('Missing candidate images/labels: {}'.format(images))
        yield split, images, labels


def configured_sources(name):
    sys.path.insert(0, str(TEMPLATE))
    from engine.data import resolve_sources
    config = yaml.safe_load(workspace_config(name).read_text(encoding='utf-8'))
    return [(split, source['images'], source['labels']) for split in ('train', 'val', 'test')
            for source in resolve_sources(ROOT, config, split)]


def copy_candidates(package, config_path, overrides=None, extra=None, selected_sources=None):
    record = {}
    sources = list(source_sets(config_path, overrides)) if selected_sources is None else selected_sources
    for split, path in (extra or {}).items():
        image_dir = inside_workspace(ROOT / path)
        sources.append((split, image_dir, inside_workspace(labels_for(image_dir))))
    for old_split, image_dir, label_dir in sources:
        target = package / 'dataset'
        images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if not images:
            raise ValueError('Empty candidate source: {}'.format(image_dir))
        for image in images:
            label = label_dir / (image.stem + '.txt')
            if not label.is_file():
                raise ValueError('Missing candidate label: {}'.format(label))
            if (target / 'images' / old_split / image.name).exists() or (target / 'labels' / old_split / label.name).exists():
                raise ValueError('Duplicate output image or label: {}'.format(image.name))
            shutil.copy2(image, target / 'images' / old_split / image.name)
            shutil.copy2(label, target / 'labels' / old_split / label.name)
        entry = {'images': len(images), 'original_images': image_dir.relative_to(ROOT).as_posix(),
                 'original_labels': label_dir.relative_to(ROOT).as_posix()}
        record.setdefault(old_split, []).append(entry)
        # Keep adjacent source manifests where available, without claiming they
        # establish that the former split groups are independent.
        parents = list(image_dir.parents)[:5]
        records_dir = target / 'source_records' / old_split
        for index, parent in enumerate(parents):
            try:
                parent.relative_to(ROOT)
            except ValueError:
                continue
            for name in ('manifest.json', 'manifest.jsonl', 'summary.json', 'run.json'):
                source = parent / name
                if source.is_file():
                    records_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, records_dir / ('level{}_{}'.format(index, name)))
    return record


def write_package_readme(package, name, epochs):
    text = fr"""# {name} V3 训练分发包

本目录由工作空间 V3 训练源镜像生成；修改源配置后通过构建器刷新。源位置见 `mirror_manifest.json`。
图片和标签位于 `dataset/`，本地预训练权重位于 `weights/`；它们是额外分发副本。

## 离线启动

将完整 `portable_env/` 放在本包旁边，或放进本包内。环境要求的 CPython 次版本、Windows 架构和 GPU 驱动条件见环境自己的 README；启动器自动寻找兼容解释器，不在目标机安装依赖。

```powershell
.\scripts\start_training.ps1
# 环境放在其他位置时显式指定
.\scripts\start_training.ps1 -Environment ..\portable_env -NoWait
# 仅检查数据、配置和本地权重
& ..\portable_env\run.ps1 -Script .\scripts\train.py -Offline -ScriptArguments @('check')
```

所有实现放在 `scripts/`；根目录同名脚本仅兼容转发。已有独立 Python 训练环境也可使用 `python -B scripts/train.py check` 或 `train`。
统一首轮输入 640×640，预算 {epochs} epoch；数据、优化器、增强、设备和预算由 `config/train.yaml` 明确控制。CUDA 不可用时明确报错，需显式配置 CPU 才使用 CPU。

## 训练、评估和回传

训练默认使用验证集选模和评估；最终测试需单独执行 `evaluate --split test --weights <相对包根目录的本次best.pt路径>`。`tune` 使用验证集进行有限试验，`compare` 对已生成的运行指标做比较。完整参数可用相应命令的 `--help` 查看。

本包拥有独立的 `runs/`，位置由本包脚本确定，不受启动终端的当前目录影响。模型运行写入本包的 `runs/{name}/<运行编号>/`，启动器日志写入本包的 `runs/_launcher/`；其他任务包使用它们自己的 `runs/`。共享 `portable_env` 仅提供依赖。

所有运行产物在本包的 `runs/`：终端日志、TensorBoard、实际配置、数据与缓存、权重、训练曲线、混淆矩阵、PR/F1 曲线、逐类 CSV/图、预测预览、空标签误检和失败样本。复制本包完整的 `runs` 回工作空间，并保留任务名和运行编号。数据来源与同源复用信息见 `dataset/inventory.json` 和来源记录；同源指标不等于独立场次表现。中断与失败保留实际状态，不会写成完成。

{name}: {NOTES[name]}

预训练权重许可见 `weights/LICENSE.txt`。仪表关键点顺序见其 `dataset/keypoints.yaml`；垃圾贴纸编号和规则名称见其 `dataset/class_names_zh.yaml`。模型训练指标不代替 OCR、跨帧计数、归属、读数或实车任务验收。
"""
    (package / 'README.md').write_text(text, encoding='utf-8')


def build(name):
    configuration = yaml.safe_load(workspace_config(name).read_text(encoding='utf-8'))
    source, task, epochs = configuration['data'], configuration['task'], configuration['epochs']
    config_path = inside_workspace(ROOT / source)
    original = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    package = DEST / name
    if package.exists():
        raise FileExistsError('Package already exists, refusing overwrite: {}'.format(package))
    package.mkdir(parents=True)
    for file in ('train.py', 'tensorboard_log.py', 'start_training.ps1', 'requirements.txt'):
        shutil.copy2(TEMPLATE / file, package / file)
    model = Path(configuration['model']).name
    (package / 'weights').mkdir()
    shutil.copy2(inside_workspace(ROOT / configuration['model']), package / 'weights' / model)
    shutil.copy2(ROOT / 'models' / 'weights' / 'yolo_v8' / 'LICENSE.txt', package / 'weights' / 'LICENSE.txt')
    if name == 'plate_detection':
        shutil.copy2(TEMPLATE / 'licenses' / 'CCPD_LICENSE.txt', package / 'CCPD_LICENSE.txt')
    for split in ('train', 'val', 'test'):
        for kind in ('images', 'labels'):
            (package / 'dataset' / kind / split).mkdir(parents=True)
    names = configuration.get('class_names', original['names'])
    if name == 'waste_sticker':
        tasks = json.loads((ROOT / 'config' / 'perception_tasks.json').read_text(encoding='utf-8'))['tasks']
        items = next(item['waste_items'] for item in tasks if item['id'] == 'waste_bin')
        yaml_write(package / 'dataset' / 'class_names_zh.yaml', waste_class_mapping(original['names'], items))
    data = {'train': 'images/train', 'val': 'images/val', 'test': 'images/test', 'names': names}
    if task == 'pose':
        data['kpt_shape'] = original['kpt_shape']
    if name == 'meter':
        yaml_write(package / 'dataset' / 'keypoints.yaml', {'keypoints': [
            {'id': index, 'name': key, '中文': label}
            for index, (key, label) in enumerate(METER_KEYPOINTS)
        ]})
    yaml_write(package / 'dataset' / 'data.yaml', data)
    yaml_write(package / 'config' / 'train.yaml', {'task': task, 'data': 'dataset/data.yaml',
        'model': 'weights/' + model, 'imgsz': 640, 'epochs': epochs, 'batch': 16, 'workers': 4,
        'patience': 30, 'seed': 42, 'project': 'runs', 'name': 'train', 'test_after_training': True})
    record = copy_candidates(package, config_path, selected_sources=configured_sources(name))
    (package / 'dataset' / 'inventory.json').write_text(json.dumps({
        'source_yaml': source, 'status': 'packaged_selected_splits', 'splits': record},
        ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write_package_readme(package, name, epochs)
    return record


def write_index():
    lines = ['# V3 训练分发镜像', '',
        '本目录是工作空间任务训练工程的额外分发副本。开发源在 lab/features/<任务>/training 和 lab/training_common；本目录不作为工作空间代码或数据依赖。', '',
        '将一个任务包与同级 portable_env 一起复制到目标 Windows 电脑。兼容的本机 CPython 由启动器自动发现；目标端不安装或下载依赖。环境条件及验证范围见 portable_env/README.md。', '',
        '| 任务 | YOLO 类型 | 首轮 epoch |', '| --- | --- | ---: |']
    for name in TASKS:
        configuration = yaml.safe_load(workspace_config(name).read_text(encoding='utf-8'))
        task, epochs = configuration['task'], configuration['epochs']
        lines.append('| [{}]({}/README.md) | {} | {} |'.format(name, name, task, epochs))
    lines.extend(['', '每包使用 scripts/start_training.ps1；代码、配置和完整运行产物分别位于本包 scripts、config、runs。九包各有独立 runs，依赖环境可以共享。训练默认评估 val，最终 test 独立执行；回传对应包的整个 runs。', '',
        '红绿灯采用 HSV/ROI；热异常采用楼层热响应方案，均不额外生成 YOLO 包。火焰与楼名、桶状态与垃圾贴纸分别训练。', '',
        '所有包保持原有类别和数据组合。分发前通过工具的 --refresh-code 更新镜像；该操作保留图片、标签、权重和既有 runs。', ''])
    (DEST / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tasks', nargs='+', choices=sorted(TASKS))
    parser.add_argument('--refresh-code', action='store_true', help='Refresh managed V3 code/config in existing packages; preserve data, weights and runs')
    parser.add_argument('--include-environment', action='store_true', help='Copy the built shared portable_env beside the task packages')
    args = parser.parse_args()
    for name in args.tasks:
        if not args.refresh_code:
            build(name)
        print(name, json.dumps(refresh_code(name), ensure_ascii=False), flush=True)
    if args.include_environment:
        print('environment', json.dumps(mirror_environment(), ensure_ascii=False), flush=True)
    write_index()


if __name__ == '__main__':
    main()
