"""Build a lightweight meter inference folder: scripts, model and configuration."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[5]
SOURCE = Path(__file__).resolve().parent
PACKAGE_NAME = 'meter_inference'
RUN_BAT = '''@echo off
setlocal
python -X utf8 -B "%~dp0infer_meter.py" %*
exit /b %errorlevel%
'''
RUN_PS1 = '''[CmdletBinding()]
param(
    [Parameter(Mandatory=$true,Position=0)][string]$Source,
    [string]$Device='0',
    [int]$ImageSize=640,
    [string]$Output,
    [string]$Python='python'
)
$ErrorActionPreference='Stop'
$inferenceArgs=@('--source',$Source,'--device',$Device,'--imgsz',"$ImageSize")
if ($Output) { $inferenceArgs+=@('--output',$Output) }
& $Python -X utf8 -B (Join-Path $PSScriptRoot 'infer_meter.py') @inferenceArgs
exit $LASTEXITCODE
'''
REQUIREMENTS = '''# Dependencies only; no Python environment is shipped.
# Verified locally with Python 3.12 and Torch 2.7.1+cu128 / torchvision 0.22.1+cu128.
# Select CPU/CUDA builds of torch and torchvision for the destination computer.
ultralytics==8.3.203
torch==2.7.1
torchvision==0.22.1
opencv-python==4.10.0.84
numpy==1.26.4
Pillow==10.4.0
matplotlib==3.9.2
'''
README = '''# 温湿度计推理标注包

包含 Python 源码、训练后的 `models/best.pt` 和刻度配置；不包含 Python 解释器、
虚拟环境或依赖安装包。复制整个文件夹即可使用，不依赖原工作空间的代码或路径。
在已有 Python 环境中运行，依赖版本见 `requirements.txt`。本次实际测试环境为
Windows、Python 3.12、Torch 2.7.1+cu128、Ultralytics 8.3.203；未作 ROS/实车验收。

## 启动

在本文件夹打开终端，使用已安装相应依赖的 Python：

```powershell
python infer_meter.py --source "图片路径.png"
python infer_meter.py --source "视频路径.mp4"
python infer_meter.py --source examples/P18_GAUGE_vision.png --device cpu
.\\run.bat --source "图片路径.png"
.\\run.ps1 -Source "图片路径.png" -Python "D:\\your-env\\Scripts\\python.exe"
```

默认 CUDA 设备0；无可用 CUDA 时显式传 `--device cpu`。`run.bat` 使用当前终端的
`python`，也可直接用自己的 Python 完整路径执行 `infer_meter.py`。
模型从包内相对路径加载，不需要下载权重。

默认输出在本文件夹 `outputs/<文件名_时间>/`。图片生成 `annotated.png`、带放大视图的
`preview.jpg` 和 `result.json`；视频逐帧生成 `annotated.mp4`（无音频）、`frames.jsonl`
和 `result.json`。可用 `--output 新目录` 指定位置；已有输出目录不会覆盖。

## 读数与标注

`config/meter_readout.json` 默认温度 −30～50°C、1°C 一小格；湿度 0～100%RH、
2%RH 一小格。主刻度每10单位，可分别调整范围、间距和方向。
两组各自使用轴心、针尖、量程最小和最大端点，共八点，按图像坐标顺时针弧插值。
绿色短线为推算刻度，黄色/青色为双针及轴心，蓝色/红色点为量程两端。

低置信度点、零长度指针、无效量程或指针落在空白弧时，该项输出 unknown；
另一项独立计算。未检出表盘时两项均为 unknown，不填默认读数。
JSON 同时保存八点坐标及置信度、检测框、读数、配置和模型 SHA256。

这是等角度估算，没有透视校正或实物逐刻度标定。斜视和模型点位误差会影响读数；
绿色刻度不是检测得到的印刷刻度，显示的小数位不代表已验证的测量精度。

## 模型与文件来源

`models/best.pt` 来自120轮早停运行的第97轮最佳权重，默认640输入。
已测视频的近景大表盘在640输入下漏检较多，可显式尝试 `--imgsz 320`，
但不能由此推断其他场景准确率。`package_manifest.json` 保存模型和源码摘要。
脚本与核心是构建时源码快照，修改原维护源后需重新打包。
'''


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def publish_code(target, weights):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE / 'infer_meter_pose.py', target / 'infer_meter.py')
    shutil.copy2(ROOT / 'ws/src/aec_vision/src/aec_vision/meter.py', target / 'meter_geometry.py')
    (target / 'models').mkdir(exist_ok=True)
    shutil.copy2(weights, target / 'models/best.pt')
    cfg = json.loads((SOURCE.parent / 'config/pose_readout.json').read_text(encoding='utf8'))
    cfg['weights'] = 'models/best.pt'
    (target / 'config').mkdir(exist_ok=True)
    (target / 'config/meter_readout.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    (target / 'run.bat').write_bytes(RUN_BAT.replace('\n', '\r\n').encode('ascii'))
    for name, content in [('run.ps1', RUN_PS1), ('README.md', README), ('requirements.txt', REQUIREMENTS)]:
        (target / name).write_text(content, encoding='utf-8-sig' if name.endswith('.ps1') else 'utf8')


def write_manifest(target, weights):
    sources = [SOURCE / 'infer_meter_pose.py', SOURCE / 'package_meter_inference.py',
               SOURCE.parent / 'config/pose_readout.json', ROOT / 'ws/src/aec_vision/src/aec_vision/meter.py']
    files = ['infer_meter.py', 'meter_geometry.py', 'config/meter_readout.json', 'models/best.pt',
             'run.bat', 'run.ps1', 'README.md', 'requirements.txt']
    manifest = dict(kind='meter_script_model_inference', built_at_utc=datetime.now(timezone.utc).isoformat(),
                    python_included=False, dependencies_included=False, system_python_required=True,
                    model_source=str(Path(weights).resolve()), model_sha256=sha(target / 'models/best.pt'),
                    source_hashes={str(p.relative_to(ROOT)): sha(p) for p in sources},
                    files={p: sha(target / p) for p in files},
                    scope='Scripts, model and config; existing Python environment required. Image-plane equal-angle estimates; no physical calibration or target-machine acceptance.')
    (target / 'package_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return manifest


def build(output, weights):
    output, weights = Path(output).resolve(), Path(weights).resolve()
    parent = (ROOT / 'training_packages').resolve()
    if output.parent != parent or output.name != PACKAGE_NAME:
        raise ValueError('Inference delivery must be training_packages/meter_inference')
    if output.exists():
        raise FileExistsError('Delivery folder exists; existing deliveries are not overwritten')
    if not weights.is_file():
        raise FileNotFoundError(weights)
    tracked = [SOURCE / 'infer_meter_pose.py', SOURCE / 'package_meter_inference.py',
               SOURCE.parent / 'config/pose_readout.json', ROOT / 'ws/src/aec_vision/src/aec_vision/meter.py', weights]
    before = {str(p): sha(p) for p in tracked}
    with tempfile.TemporaryDirectory(prefix='.meter-inference-', dir=parent) as temp:
        staging = Path(temp).resolve()
        if not staging.is_relative_to(parent):
            raise ValueError('Unexpected staging path')
        target = staging / PACKAGE_NAME
        publish_code(target, weights)
        example = ROOT / 'training_packages/meter/runs/meter/v4_20261118_140109_229142/evaluation/development_P18_GAUGE_vision_20261118_225417/input_snapshot.png'
        if example.is_file():
            (target / 'examples').mkdir()
            shutil.copy2(example, target / 'examples/P18_GAUGE_vision.png')
        manifest = write_manifest(target, weights)
        if before != {str(p): sha(p) for p in tracked}:
            raise RuntimeError('Inference sources changed during build')
        if output.parent != parent or not target.resolve().is_relative_to(staging):
            raise ValueError('Unexpected final path')
        target.rename(output)
    print(json.dumps(dict(status='built', output=str(output), model_sha256=manifest['model_sha256']), ensure_ascii=False), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'training_packages/meter_inference')
    parser.add_argument('--weights', type=Path, default=ROOT / 'training_packages/meter/runs/meter/v4_20261118_140109_229142/training/weights/best.pt')
    args = parser.parse_args()
    build(args.output, args.weights)


if __name__ == '__main__':
    main()
