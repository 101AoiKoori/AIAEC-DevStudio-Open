# Windows 训练依赖包

整个目录与任务包一起复制使用。需要本机已安装标准 **CPython 3.12 或 3.14 x64**，Python 3.13 不受支持；版本对应依赖目录必须完整，Python 的安装盘符可以不同。本包不包含 Python 解释器，运行时不需要 pip、uv 或下载依赖。

Python 3.12 使用 PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128；3.14 使用独立的 PyTorch 2.9.1+cu128、torchvision 0.24.1+cu128。各版本完整依赖见各自的 `requirements.lock` 和 `manifest.json`。CUDA 12.8 wheel 包含 Blackwell `sm_120`，用于 RTX 5060/5070 等 RTX 50 系列；目标机仍须安装支持该 GPU 和 CUDA 构建的 NVIDIA 驱动及所需 Visual C++ 运行库。普通训练无需单独安装完整 CUDA Toolkit。启动器不安装或更新驱动和运行库。

建议布局：

```text
分发目录/
    portable_env/   本目录
    person/         人物任务包
    meter/          仪表任务包
```

在分发目录先执行 GPU 自检：

```powershell
.\portable_env\run.ps1 -Check -RequireCuda -Offline -Report .\person\runs\environment\report.json
```

启动器会通过 Python Launcher、PATH 和注册表发现解释器；也可用 `-Python <本机python.exe路径>` 明确指定。Python 次版本、架构、ABI 或依赖副本不匹配时退出并报告，不静默换版本或改成 CPU。

运行单个任务时优先使用该包的启动入口；直接调用示例：

```powershell
.\portable_env\run.ps1 -Script .\person\scripts\train.py -Offline -ScriptArguments @('check')
.\portable_env\run.ps1 -Module tensorboard.main -ScriptArguments @('--logdir', '.\person\runs', '--host', '127.0.0.1')
```

任务脚本的具体命令见任务 README。`-ScriptArguments @(...)` 保留带空格的参数。使用本启动器或模块入口，不运行 `site-packages/Scripts/` 下安装器生成的 exe。启动采用 `-I -S -B`，不加载系统或用户 site-packages，不使用外部 `PYTHONPATH`。

`-Check` 核对包版本、依赖要求、文件存在和记录尺寸，检查原生导入、TensorBoard 事件及本机 HTTP 可读性、CPU/CUDA 算子。所有自检文件写入 `-Report` 所在目录。`-Offline` 阻止当前 Python 进程的非本机 socket 连接，保留 localhost；它不改变系统网络或防火墙配置。完整目录包含两套依赖，各任务可以共用一份。

2026-11-15 按同一锁重建 3.12 依赖目录（54 个包、约 6.0 GiB），自检、含空格路径复制、两进程 DataLoader、外部 PYTHONPATH 忽略和 Python 次版本拒绝检查均通过，YOLOv8n 权重加载与单图推理通过。2026-11-11 在同一台 Windows、CPython 3.12.8、RTX 4060 Laptop 上首次完成这些核对。3.14.9 已在同一台电脑通过 CUDA 算子检查、8 图 CPU 训练和验证。2026-11-14 在另一台 Windows 电脑的 RTX 5070 上，3.12.14 和 3.14.7 均通过 CUDA 算子及依赖自检；整包训练未运行。目标电脑仍应执行上述自检。模型训练输出以任务 `runs/` 中的实际记录为准。
