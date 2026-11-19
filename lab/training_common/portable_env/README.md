# Windows 可复制训练依赖

本目录维护 Windows 训练依赖包的构建、隔离启动和兼容性诊断。源码、依赖输入及全量版本锁进入 Git；`bundles/` 是包含第三方依赖的本地分发产物，`runs/` 保存构建缓存、日志和核对结果。目标机使用自己已安装的 **标准 CPython 3.12 或 3.14 x64**；Python 3.13 不受支持。依赖包不携带解释器，目标机无需 uv 或 pip。

Python 3.12 环境使用 PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128、NumPy 1.26.4；3.14 使用独立依赖目录，包含 PyTorch 2.9.1+cu128、torchvision 0.24.1+cu128、NumPy 2.5.4，两者均使用 Ultralytics 8.3.203 和原有 YOLOv8 模型。PyTorch 2.7 引入 Blackwell 支持；官方提供 Windows CUDA 12.8 的 2.7.1/0.22.1 配套 wheel。[PyTorch 2.7 说明](https://pytorch.org/blog/pytorch-2-7/)、[官方版本组合](https://pytorch.org/get-started/previous-versions/)。Ultralytics 固定版本的加载器显式处理 PyTorch 2.6 以后的 `weights_only` 默认值；本工程只加载已确认来源的本地检查点。[8.3.203 加载代码](https://github.com/ultralytics/ultralytics/blob/v8.3.203/ultralytics/utils/patches.py#L87)。RTX 50 系列目标机的实际训练仍需在相应设备运行，不能由本机 RTX 4060 的核对结果替代。

## 当前工作电脑

2026-11-15 在本机核对：13th Gen Intel Core i5-13500HX、15.7 GB 内存，显示设备为 Intel UHD Graphics 与 NVIDIA GeForce RTX 4060 Laptop GPU（驱动 32.0.16.1692，即 616.92）。项目 `.venv` 于 2026-11-15 用本机标准 CPython 3.12.8 执行 `python -m venv` 重建，按 `requirements-win-cp312-cu128.lock` 安装 54 个依赖，`sys.base_prefix` 为该解释器的标准安装目录，`uv pip check` 通过。NumPy 1.26.4、OpenCV 4.10.0、SciPy 1.14.1、PyYAML 6.0.2、pandas 2.2.3、PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128、Ultralytics 8.3.203 导入通过；`torch.cuda.is_available()` 为 True、CUDA 12.8、设备能力 sm_89、设备名 NVIDIA GeForce RTX 4060 Laptop GPU。`nvidia-smi` 报告驱动 616.92、显存 8188 MiB、算力 8.9；该调用在 DSH 命令进程内报 `Failed to initialize NVML: Unknown Error`，经 WMI 启动或普通终端调用输出正常。

2026-11-15 算子与推理核对（同一 `.venv`）：`torch.cuda.get_arch_list()` 为 `['sm_50', 'sm_60', 'sm_61', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']`，cuDNN 90701，bf16 可用。4096×4096 fp32 与 fp16 矩阵乘、`Conv2d` 前后向、`torchvision.ops.nms` 均在 cuda:0 上通过（NMS 保留索引 `[0, 2]`）。本地 `models/weights/yolo_v8/yolov8n.pt` 与 `models/weights/e_bike/sam2_t.pt` 对 640×640 空白图在 cuda:0 完成推理，空白图无目标，输出 0 框与 0 掩码，该次峰值显存 0.518 GB。这些输入仅用于验证环境，不是数据集质量或模型效果评估。

2026-11-14 在另一台电脑核查：Windows 系统为 NVIDIA GeForce RTX 5070（12 GB），驱动 617.42；Blender 5.2.3 检出该设备的 OptiX 并完成 16×16、1 采样渲染。该电脑的项目 `.venv` 使用 CPython 3.12.14 与本目录的 `requirements-win-cp312-cu128.lock`，并在该电脑通过 CUDA 算子与依赖自检；这些结论不适用于本机。环境不提交 Git。

现有 `training_packages/portable_env/` 在 RTX 5070 那台电脑以 CPython 3.12.14 和 3.14.7 分别执行 `-Check -RequireCuda -Offline`，均通过依赖检查、CUDA 矩阵乘法、torchvision NMS 和卷积前后向；报告位于 `runs/environment_check_rtx5070_20261114/portable_cp312.json` 和 `portable_cp314.json`。这是环境和算子检查，任务训练、数据标签和模型效果未在此阶段验证。`lab/training_common/requirements.txt` 保留旧版 V1/V2 依赖，不能用于 RTX 50 系列设备的依赖环境。

训练引擎和本次检查的数据分割入口默认选择 CUDA 设备 0，支持 OptiX 的相关数据渲染入口默认使用 OptiX；需要 CPU 时按入口通过 `device: cpu`、`--device cpu`、`--device CPU` 或 `--render-device cpu` 显式选择。设备不可用时检查依赖和驱动，不将失败报告为 GPU 成功。历史 CPU 回归检查及不支持 GPU 的操作保留各自用途。小车正式推理使用其独立配置，硬件和部署环境另行核验。

## 安装

在已安装标准 CPython 3.12 与 uv 的 Windows 电脑上，用本机解释器新建工作空间环境并安装锁定依赖：

```powershell
python -m venv .venv
uv pip install --python .venv/Scripts/python.exe --no-python-downloads --no-build --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match -r lab/training_common/portable_env/requirements-win-cp312-cu128.lock
uv pip check --python .venv/Scripts/python.exe
```

`python` 指本机标准 CPython 3.12.8（`py -3.12` 同样指向它），不使用 `uv venv` 或 `uv python install` 生成或下载解释器；本机 `py -0p` 另列上一轮 uv 安装 3.12.14 时写入的登记项，该解释器目录已删除。依赖下载约 3.2 GiB（其中 torch wheel 3.0 GiB）；本机 2026-11-15（uv 0.13.5）按上述步骤联网安装的 `.venv` 为 6.08 GiB、26,430 个文件、55 个 dist-info（54 个锁定依赖加 `python -m venv` 自带的 pip 24.3.1），文件数含核对时生成的 `__pycache__`；按同一锁重装得到逐项相同的 54 个版本。此前由 `uv venv` 建成的环境在 `Scripts\` 下报 `uv trampoline failed to canonicalize script path`；按上述步骤生成的 `yolo version`、`ultralytics version` 返回 8.3.203，`torchrun.exe`、`tensorboard.exe` 可启动。控制台入口不可用时改用 `python -m ultralytics` 等模块调用。

## 构建与更新

源电脑安装兼容解释器和 uv 后，从工作空间根目录执行：

```powershell
python lab/training_common/portable_env/build.py
# 使用已安装的 3.14 解释器构建对应依赖
py -3.14 lab/training_common/portable_env/build.py --output win-cp314-cu128-20261111
```

`build.py` 根据运行它的 Python 次版本选择对应全量锁，安装到新的 `bundles/<输出名称>/site-packages/`，使用 wheel 并强制复制文件，不创建解释器、激活脚本或到源缓存的软链接。目录已存在时停止。需要改变依赖时，修改 3.12 的 `requirements.in` 或 3.14 的 `requirements-modern.in`，并显式重新解析到新版本：

```powershell
python lab/training_common/portable_env/build.py --resolve --output win-cp312-cu128-v3-next
```

先核对新包，再更新分发包的环境副本。构建期使用 `uv pip compile` 和 `uv pip install --target --link-mode copy --no-build --no-python-downloads`；构建日志及失败状态位于 `runs/build_<时间>/`，下载缓存在 `runs/uv_cache/`。安装不会修改系统或项目 `.venv`。[uv CLI](https://docs.astral.sh/uv/reference/cli/#uv-pip-install)。

## 分发与运行

V4 任务包使用[当前构建入口](../../../docs/TRAINING_WORKFLOW.md#v4-任务包构建与启动)，并与现有 `portable_env/` 放在同一层。只分发一个任务时，复制该任务包和 `portable_env/` 即可。依赖中的原生库使 CUDA 包占用数 GB。

新构建的启动器分别检查根目录与可选 `cp314/` 的 manifest、启动脚本和依赖目录；某一环境缺失或损坏时报告原因并尝试其他匹配环境。仅有目录不证明依赖完整，仍需在目标机运行 `-Check`。已分发环境是快照，需更新启动器后才具备此行为。

构建任务包使用现有共享环境，不会自动替换其中的启动器。可先从工作空间根目录只读检查它是否与源码一致：

```powershell
python lab/training_common/portable_env/sync_launcher.py
```

报告 `current`、`outdated` 或 `missing`，并列出两边的 SHA-256；任务包的计划和构建报告也在 `portable_launcher` 字段给出该状态，无法比较时记为 `unavailable`。比较只提供提示，不自动更新环境。确认要更新时运行 `python lab/training_common/portable_env/sync_launcher.py --update`，仅替换共享环境的 `run.ps1`，保留依赖、manifest 和其他文件。检查或更新其他分发目录时使用 `--bundle <portable_env目录>`。更新前后可在目标机运行 `-Check`；启动器字节一致不代表依赖已完成检查。

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

2026-11-15 按同一锁从网络重建 `bundles/win-cp312-cu128-v3/`：54 个固定依赖、22,521 个文件、6.0 GiB，包版本与 2026-11-11 构建逐项一致（旧副本内的 554 个陈旧 `.pyc` 新构建不生成）；构建记录为 `runs/build_20261115_100916_997022/`，构建自带离线自检通过。用该副本的 `bootstrap.py --check --offline --require-cuda --model models/weights/yolo_v8/yolov8n.pt` 核对通过，报告在 `runs/rebuild_check_20261115/report.json`。复制到含空格新路径后重跑迁移检查：22,515 个 RECORD 文件、依赖约束、原生导入、CPU/CUDA 算子、TensorBoard HTTP 标量读取、YOLOv8n 本地权重加载及单图推理通过；两进程 DataLoader 读取 12 个样本通过，外部 PYTHONPATH 被忽略，Python 次版本不匹配被拒绝；证据为 `runs/relocation_20261115_102249_396571/state.json`、`native/report.json` 和 `workers.log`。2026-11-11 在 Windows、CPython 3.12.8、RTX 4060 Laptop、驱动 616.92 上首次构建该目录，当时的迁移证据为 `runs/relocation_20261111_103020_361101/`；另外四个兼容、缺失文件及离线代理隔离单元检查通过。

上述迁移核对均在同一物理电脑与已安装解释器上完成，只改变依赖目录位置。分发目标用户说明由 `BUNDLE_README.md` 生成到包内 `README.md`。

2026-11-11 的 3.14.9 构建包含 42 个依赖，证据为 `runs/build_20261111_123146_500070/environment_check.json`。本机 RTX 4060 Laptop 通过 CUDA 矩阵、NMS 和卷积前后向；使用人物包各 8 张训练与验证图、CPU 320 输入、1 epoch 完成训练和独立验证。3.14 的一键启动经只提供 3.14 的 PATH 实际检查；RTX 50 系设备、其他物理电脑尚未实测。依赖分别位于分发环境根目录和 `cp314/`，由启动器匹配本机解释器；所有训练输出仍在对应任务包的 `runs/`。

2026-11-12 在同一 Windows 主机上，3.12.8 和 3.14.9 的依赖分别复制到含空格路径，通过解释器自动发现、ABI 与架构核对、隔离及离线导入、CPU/CUDA 算子、两进程 DataLoader 和错版本拒绝检查；证据分别为 `runs/relocation_20261112_172110_562120/state.json` 和 `runs/relocation_20261112_172110_698554/state.json`。共享分发目录的自动选择 3.12 与显式指定 3.14 自检通过，报告在 `runs/dual_environment_20261112/`。真实 3.13.17 在启动器及构建器的拒绝检查通过。用户已移除旧 3.13 分发副本、三个已确认的构建目录及其锁文件；3.12 的根目录依赖和 3.14 的独立依赖目录保留。

移除后复核共享目录：3.12 自动发现和显式指定 3.14 的离线、CUDA 自检均通过，记录在工作空间 `runs/portable_post_cp313_removal_20261112/`；显式指定 3.13.17 仍返回不支持提示并以非零状态退出。上述 CUDA 实测设备为 RTX 4060 Laptop，不代表 RTX 5060/5070 实测。
