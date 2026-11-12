# Windows 可复制训练依赖

本目录维护 Windows 训练依赖包的构建、隔离启动和兼容性诊断。源码、依赖输入及全量版本锁进入 Git；`bundles/` 是包含第三方依赖的本地分发产物，`runs/` 保存构建缓存、日志和核对结果。目标机使用自己已安装的 **标准 CPython 3.12、3.13 或 3.14 x64**；依赖包不携带解释器，目标机无需 uv 或 pip。

Python 3.12 环境使用 PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128、NumPy 1.26.4；3.13 和 3.14 分别使用独立依赖目录，包含 PyTorch 2.9.1+cu128、torchvision 0.24.1+cu128、NumPy 2.5.4，三者均使用 Ultralytics 8.3.203 和原有 YOLOv8 模型。PyTorch 2.7 引入 Blackwell 支持；官方提供 Windows CUDA 12.8 的 2.7.1/0.22.1 配套 wheel。[PyTorch 2.7 说明](https://pytorch.org/blog/pytorch-2-7/)、[官方版本组合](https://pytorch.org/get-started/previous-versions/)。Ultralytics 固定版本的加载器显式处理 PyTorch 2.6 以后的 `weights_only` 默认值；本工程只加载已确认来源的本地检查点。[8.3.203 加载代码](https://github.com/ultralytics/ultralytics/blob/v8.3.203/ultralytics/utils/patches.py#L87)。RTX 50 系列目标机的实际训练仍需在相应设备运行，不能由本机 RTX 4060 的核对结果替代。

## 构建与更新

源电脑安装兼容解释器和 uv 后，从工作空间根目录执行：

```powershell
python lab/training_common/portable_env/build.py
# 使用各自安装的解释器构建另外两个版本
py -3.13 lab/training_common/portable_env/build.py --output win-cp313-cu128-20261111
py -3.14 lab/training_common/portable_env/build.py --output win-cp314-cu128-20261111
```

`build.py` 根据运行它的 Python 次版本选择对应全量锁，安装到新的 `bundles/<输出名称>/site-packages/`，使用 wheel 并强制复制文件，不创建解释器、激活脚本或到源缓存的软链接。目录已存在时停止。需要改变依赖时，修改 3.12 的 `requirements.in` 或 3.13/3.14 的 `requirements-modern.in`，并显式重新解析到新版本：

```powershell
python lab/training_common/portable_env/build.py --resolve --output win-cp312-cu128-v3-next
```

先核对新包，再更新分发包的环境副本。构建期使用 `uv pip compile` 和 `uv pip install --target --link-mode copy --no-build --no-python-downloads`；构建日志及失败状态位于 `runs/build_<时间>/`，下载缓存在 `runs/uv_cache/`。安装不会修改系统或项目 `.venv`。[uv CLI](https://docs.astral.sh/uv/reference/cli/#uv-pip-install)。

## 分发与运行

运行 `tools/build_training_packages.py --refresh-code --include-environment person` 将三套已构建依赖组装为分发根目录的 `portable_env/`；完整目录约 15.52 GiB。九个任务包可以与它平级，共用一份依赖。只分发一个任务时，复制该任务包和 `portable_env/` 即可。依赖中的原生库使 CUDA 包占用数 GB；省去解释器不会显著缩小 PyTorch 的 CUDA 库。

```text
分发目录/
    portable_env/{run.ps1,bootstrap.py,manifest.json,requirements.lock,site-packages/}
    person/{scripts/,config/,dataset/,weights/,runs/}
    meter/{scripts/,config/,dataset/,weights/,runs/}
```

在分发目录中核对当前机器：

```powershell
.\portable_env\run.ps1 -Check -RequireCuda -Offline -Report .\person\runs\environment\report.json
```

`run.ps1` 依次检查显式 `-Python`（或 `AIAEC_PYTHON`）、Python Launcher、PATH 及 Python 注册表；实际执行候选解释器探测实现、次版本、架构和构建类型。不兼容时列出原因并退出，不安装依赖、不降级为别的 Python。显式指定的解释器不兼容时不会改选其他解释器。正常补丁版本差异不要求重新安装依赖，实际诊断仍须通过。

执行任务脚本或模块时：

```powershell
.\portable_env\run.ps1 -Script .\person\scripts\train.py -Offline -ScriptArguments @('check')
.\portable_env\run.ps1 -Module tensorboard.main -ScriptArguments @('--logdir', '.\person\runs', '--host', '127.0.0.1')
```

使用 `-ScriptArguments @(...)` 保留带空格的参数；任务脚本具体命令由对应任务 README 维护。PowerShell 若受执行策略限制，可在当前启动命令使用系统允许的脚本执行方式，启动器不修改机器策略。

## 隔离与兼容边界

启动器使用 `python -I -S -B bootstrap.py`，保留标准库，显式加入本包 `site-packages`。`-I` 忽略外部 `PYTHONPATH` 和用户 site，`-S` 避免系统 site 自动加载；不依靠全局 `PYTHONPATH`。[Python 3.12 命令行](https://docs.python.org/3.12/using/cmdline.html)。

`site.addsitedir` 处理固定依赖 wheel 的 `.pth`，拒绝其中离开依赖目录的路径，处理后核对导入路径；`.pth` 的 Python 导入语句是依赖代码的一部分，仅允许可信的已固定依赖包。`.pyd` 必须匹配所选 CPython 次版本、Windows x64 和标准 ABI。PyTorch、NumPy、SciPy 的 DLL 搜索目录按当前包位置注册，保留各 wheel 自带的包数据及 DLL。[site.addsitedir](https://docs.python.org/3.12/library/site.html#site.addsitedir)、[os.add_dll_directory](https://docs.python.org/3.12/library/os.html#os.add_dll_directory)。

不使用 wheel 自动生成的 `Scripts/*.exe` 控制台入口，它们可能引用构建解释器；使用 `-Module` 或工作空间脚本。启动器导出 `AEC_PORTABLE_ENV` 与 `AEC_PORTABLE_BOOTSTRAP`；需要启动新 Python 进程的任务应再次通过 bootstrap，保持相同隔离环境。Windows DataLoader 使用 spawn，任务入口须保留 `if __name__ == '__main__'`。

目标机必须有支持该 GPU/CUDA 构建的 NVIDIA 驱动，以及原生 wheel 所需的 Windows 运行库；缺 DLL、驱动或不支持的 GPU 会报告失败，启动器不安装驱动或 Visual C++ 运行库。PyTorch wheel 携带运行所需 CUDA 库，普通训练不要求另装完整 CUDA Toolkit。[CUDA 12.8 驱动说明](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)。

`uv venv --relocatable` 改善入口和激活脚本的相对引用，但不保证任意二进制或包资源路径可迁移，也没有消除解释器与 ABI 的前提；本任务选择独立依赖目录与目标机解释器匹配，避免额外复制解释器。[uv relocatable 约束](https://docs.astral.sh/uv/reference/cli/#uv-venv--relocatable)。

## 核对内容

`-Check` 检查全部分发包版本、wheel RECORD 文件存在及记录的尺寸，执行 NumPy/PyTorch 转换、OpenCV、Pillow、torchvision CPU NMS、TensorBoard 事件写入；可用 CUDA 时执行矩阵乘法、CUDA NMS、卷积前后向，并报告显卡、计算能力与 wheel 编译架构。`-RequireCuda` 要求 CUDA 成功，不静默转 CPU。`-Model <本地可信.pt>` 另核对实际 YOLO 权重加载与一张空白图推理。报告和产生的图片、配置、事件文件均进入指定报告目录。

`-Offline` 设置框架离线标记、禁用自动依赖安装、移除本进程的 HTTP/HTTPS/ALL 代理变量并设置 `NO_PROXY=*`，阻止当前 Python 进程向非本机地址建立 socket 连接；本机 TensorBoard 可继续工作。这是应用进程范围的离线检查，不修改系统防火墙或断开物理网络。版本、文件存在和尺寸核对用于发现漏拷、截断及版本混用，不是内容散列或恶意篡改证明。

2026-11-11 在 Windows、CPython 3.12.8、RTX 4060 Laptop、驱动 616.92 上完成构建。`bundles/win-cp312-cu128-v3/` 含 54 个固定依赖、约 6.0 GiB 文件；构建记录为 `runs/build_20261111_101946_865253/`。复制到含空格新路径后，22,515 个 RECORD 文件、依赖约束、原生导入、CPU/CUDA 算子、TensorBoard HTTP 标量读取、YOLOv8n 本地权重加载及单图推理通过；两进程 DataLoader 读取 12 个样本通过，外部 PYTHONPATH 被忽略，Python 次版本不匹配被拒绝。证据为 `runs/relocation_20261111_103020_361101/state.json`、`native/report.json` 和 `workers.log`。另外四个兼容、缺失文件及离线代理隔离单元检查通过。

这些核对使用同一物理电脑与现有解释器，只改变依赖目录位置；RTX 50 实机、另一台电脑及不同解释器安装目录尚未实测。分发目标用户说明由 `BUNDLE_README.md` 生成到包内 `README.md`。

2026-11-11 新增 3.13.17 和 3.14.9 构建，各 42 个依赖，证据分别为 `runs/build_20261111_123146_651097/environment_check.json` 和 `runs/build_20261111_123146_500070/environment_check.json`。本机 RTX 4060 Laptop 通过 CUDA 矩阵、NMS 和卷积前后向；每个版本使用人物包各 8 张训练与验证图、CPU 320 输入、1 epoch 完成训练和独立验证。3.14 的一键启动经只提供 3.14 的 PATH 实际检查；RTX 50 系设备、其他物理电脑尚未实测。依赖分别位于分发环境根目录、`cp313/` 和 `cp314/`，由启动器匹配本机解释器；所有训练输出仍在对应任务包的 `runs/`。
