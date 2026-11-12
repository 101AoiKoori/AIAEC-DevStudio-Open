# 识别任务开发与文档系统

需求的唯一编辑入口是 [`config/perception_tasks.json`](../config/perception_tasks.json)。每个任务的 `requirements_markdown` 保存详细需求，`summary` 和 `score` 同时出现在任务 README 与总览中，`summary` 也用于任务矩阵；`matrix_name` 和 `matrix_check` 控制矩阵中的名称与验证重点。`scheduling` 保存各任务的初步调度策略并生成到任务 README 和总览，方案详见 [异步识别与资源调度](ASYNC_PERCEPTION_DESIGN.md)。需求修改时应在配置中同时更新相关摘要和详细说明。修改后运行 `python tools/sync_docs.py`；九份任务 README、任务总览和任务矩阵中的标记区会同步更新。生成的任务 README 不要直接编辑，标记区外的文字可正常维护。

运行 `python tools/sync_docs.py --check` 可检查文档是否同步；`--list` 列出生成文件。Git 的 `pre-commit` 钩子会在提交时更新并暂存这些生成文档。仓库首次克隆后，在仓库根目录运行 `git config core.hooksPath .githooks` 启用；本工作空间已启用。钩子处理工作区中的当前配置，提交前应先完成对配置和代码的编辑。更改文件后立即更新可手动运行生成命令；无需等到提交。

`--stage` 要求需求配置与 Git 暂存版本一致，避免把尚未纳入本次提交的需求生成进文档；有未暂存差异时在写文件前报错。先确定本阶段配置并暂存，再提交。生成器不锁定工作树，协作时仍须串行协调 Git 与相关文件写入。

## 代码与数据的位置

```text
config/perception_tasks.json                        # 唯一需求配置
lab/features/<任务>/training/{scripts,config,tests} # 训练和数据处理
lab/features/<任务>/inference/{config,tests}        # 实验配置、评估和演示
ws/src/aec_vision/src/aec_vision/
    <任务ID>.py                                    # 实验与 ROS 共用的推理核心
ws/src/aec_vision/                      # catkin 包和未来的 ROS 适配层
datasets/{raw,annotated,exports}/<任务>/            # 数据
models/weights/<任务>/                             # 训练所得权重
ws/src/aec_vision/models/                          # 所有正式识别模型文件直接放这里
```

训练和推理配置分开。当前车牌正式 CNN 是混合数据微调权重，双 OCR 仲裁以它为主；部署和冲突处置见[比赛双 OCR 决策](research/PLATE_DUAL_OCR.md#本次比赛双-ocr-决策)。训练数据、权重和本地运行输出不提交 Git。训练按需进行，传统视觉方法不需要训练。类别、输入尺寸、预处理、依赖、模型格式和推理参数应明确记录，并在正式配置中以部署包可访问的路径引用；模型单独传到小车。

按用户确定的短路径约定，正式识别模型统一直接放 `ws/src/aec_vision/models/`，不增加 `plate/` 等任务层或其他模型子目录。单文件权重采用可区分任务的文件名，例如 `plate_detector_best.pt`、`plate_ocr_best.pt`；当前 PaddleOCR v4 必需的 `inference.json`、`inference.pdiparams`、`inference.yml` 也直接放在该目录，配置的 `model_dir` 为 `models`。以后引入同名多文件模型时，须先解决加载器及文件名冲突，不得覆盖现有文件。训练研究权重仍在根目录 `models/weights/<任务>/`。

识别训练以合成数据为主，现有照片和视频也可参与训练、验证与测试。数据记录已知来源与生成方式；按用户确定的方案，同源素材可跨任务及集合复用，评估时说明复用范围；实拍增强、实拍拟合渲染及实拍场景合成分开报告，详见[数据集设计](DATASET_DESIGN.md)。九个 YOLOv8 模型的独立训练入口与配置位于各任务 training 区域，共用引擎在 `lab/training_common/`；训练、V1/V2 试验归档、V3 分发及结果回传见[训练流程](TRAINING_WORKFLOW.md)。`training_packages/` 是额外镜像，不作为工作空间代码、配置或数据依赖。采样量是首轮计划，按实际评估结果修订。

比赛运行时的任务开关、候选路线和 waypoint 使用部署包内的独立配置，避免运行时依赖文档配置。具体见 [任务开关与路线配置](TASK_SWITCHES_AND_ROUTES.md)。

## 共用推理脚本

正式推理核心直接开发在 `ws/src/aec_vision/src/aec_vision/`。独立测试导入这份 Python 模块；ROS 适配层也导入同一个模块，无需复制。Windows 开发机、Ubuntu 小车是不同文件系统，仍需通过 NoMachine 将更新后的包同步到小车。小车上完成 `catkin_make` 并 `source devel/setup.bash` 后，ROS 节点可导入包。

Windows PowerShell 离线实验时，可在仓库根目录执行：

```powershell
$env:PYTHONPATH = (Resolve-Path ws/src/aec_vision/src).Path
python -c "import aec_vision; print(aec_vision.__file__)"
```

推理核心不导入 ROS，不访问 `lab/`、`refs/`、`datasets/`，接收图像和上下文后返回结果。ROS 适配层负责接收实际相机数据和调度请求，再调用核心；实际 topic、消息、设备、参数和调度方式在实车联调时确定。总览显示“源码已建立”仅代表任务模块文件存在，并不代表识别可用或经过验证。生成器不会理解代码或自动推断需求变化，需要由开发者更新配置里的对应说明。

当前包提供车牌双 OCR 核心、仪表几何实验、垃圾桶分类与关联候选，以及红绿灯 HSV 候选和状态器；训练入口位于实验区。车牌离线验证范围见[卡通验证回归](research/PLATE_DUAL_OCR.md#卡通合成验证集回归2026-11-07)，仪表实验见[仪表研究](research/METER_VISION.md)。各模块的完整任务联动、ROS 节点、播报和正式证据输出仍需实现与验证。
## 需求状态与提醒

逐条标签写入 `config/perception_tasks.json` 的需求正文，生成任务 README。状态为：`[已明确]`（口径或方案已定）、`[待解决]`（具体实现/数据/验证待完成）、`[待全局规划]`（列出依赖接口，可继续独立工作）、`[未定]`（真正需要决策的信息）、`[已解决]`（注明解决范围和直接证据）。素材已具备、策略已实施、代码完成、实测通过分别描述，禁止用一个状态隐含全部阶段完成。

提醒沿用稳定任务/事项 ID；解决后原位更新标签及证据，需求变化同步所有引用。提醒由后续任务检索读取，当前未创建定时通知或后台提醒服务。共性规则以文档治理 Skill 为准，研究文档链接需求源，不另建重复状态台账。

带状态的事项写成 ``- `[待解决]` PERSON-R01：说明``；允许省略列表符号和反引号。ID 以大写任务 ID 加 `-` 开头，后接大写字母、数字或由 `-` 分隔的语义片段，例如 `PERSON-COUNT`。同一 ID 保留稳定含义；部分完成需写清范围和剩余项，必要时新增关联事项。`sync_docs.py` 检查带标签行的状态、ID 格式、任务归属及唯一性，不自动判断证据充分性、无标签正文覆盖率或语义是否变化。普通维护任务沿用所属文档，不加入识别任务配置。
