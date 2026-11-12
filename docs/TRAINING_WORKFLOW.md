# 任务训练、版本试验与分发

> 本文提及的部分本地素材、数据和历史运行产物未随本仓库提供；相关名称保留用于说明来源和实验背景。

开发源位于 `lab/features/<任务>/training/` 与 `lab/training_common/`。`training_packages/` 保存供分发的可重建训练包。包内脚本和数据按当次分发需要替换、删除或重新生成。训练开发读取工作空间代码和数据，运行证据按下文回传。数据来源、候选标注和同源复用范围由[数据区说明](../datasets/README.md)维护；历史数据归档由[技术迭代归档](../Archive/technical/README.md)维护。

## 模型与入口

| 模型 | 任务目录 | 训练脚本 | 配置 | 首轮 epoch |
| --- | --- | --- | --- | ---: |
| 人物两类 | person | `train_person.py` | `person_v3.yaml` | 150 |
| 电动车姿态 | e_bike | `train_e_bike.py` | `e_bike_v3.yaml` | 150 |
| 火焰纸贴 | fire | `train_fire_sticker.py` | `fire_sticker_v3.yaml` | 150 |
| 楼名纸条 | fire | `train_building_name_paper.py` | `building_name_paper_v3.yaml` | 150 |
| 禁行牌具 | sign | `train_sign.py` | `sign_v3.yaml` | 120 |
| 桶开闭 | waste_bin | `train_waste_bin_state.py` | `waste_bin_state_v3.yaml` | 120 |
| 垃圾贴纸 | waste_bin | `train_waste_sticker.py` | `waste_sticker_v3.yaml` | 160 |
| 仪表八点 | meter | `train_meter.py` | `meter_v3.yaml` | 180 |
| 车牌检测 | plate | `train_plate_detection.py` | `plate_detection_v3.yaml` | 120 |

表中脚本位于对应任务的 `training/scripts/`，配置位于 `training/config/`。八个检测模型使用本地 YOLOv8n，仪表使用 YOLOv8n-pose；首轮输入 640、batch 16、seed 42、patience 30。优化器、学习率及增强由配置明确传入，不接受静默忽略的配置键。文字和八点模型关闭镜像增强；默认 workers 0、AMP 关闭以形成可移植的离线基线，设备默认为单张 CUDA 卡。GPU 不可用时报告失败，使用 CPU 须显式修改配置。

仪表八点顺序固定为温度轴心、温度针尖、湿度轴心、湿度针尖、温度最小刻度、温度最大刻度、湿度最小刻度、湿度最大刻度。

车牌现有 `train_cnn_ocr.py` 保持自己的多源 OCR 训练职责。红绿灯采用 HSV/ROI 标定，热异常采用楼层热响应方案，不因统一训练工具而增加 YOLO 模型。楼名与车牌检测不读取文字，仪表 Pose 不直接完成刻度读数，人物／电动车／火焰还需要关联和计数；通用训练指标与完整任务验收分别记录。

## 工作空间运行

使用已安装匹配依赖的本机环境，或通过[可复制环境](../lab/training_common/portable_env/README.md)的启动器调用。以人物为例：

```powershell
.venv/Scripts/python.exe -B lab/features/person/training/scripts/train_person.py check
.venv/Scripts/python.exe -B lab/features/person/training/scripts/train_person.py train
```

`check` 检查配置、原始 split、类别、标签与本地权重；`train` 保存独立运行记录，以 val 选模并生成验证报告。需要调整配置时使用 `--config <工作空间相对路径>`，恢复中断训练使用 `--resume <本次runs内的training/weights/last.pt>`。恢复时检查与原配置的关系；结束后已剥离优化器的权重不能冒充可恢复检查点。

最终测试单独执行，路径中的运行编号替换为真实运行目录：

```powershell
.venv/Scripts/python.exe -B lab/features/person/training/scripts/train_person.py evaluate --split test --weights runs/person/<运行编号>/training/weights/best.pt --run-dir runs/person/<运行编号>
```

`tune` 根据显式配置执行有限试验，仅用 val 选择；`compare --runs <运行目录1> <运行目录2>` 输出已有结果比较。不同样本范围、阈值、输入分辨率、数据来源或软件环境需在比较中保留，不能只报一个提升百分比。测试集不参与调参或选模。

调参时另存一份任务配置，在其中加入下列 `tune` 段，然后运行 `tune --config <该配置路径>`。此例最多三次训练、总计两小时；实际 epoch 沿用配置，需要短试验时在该副本中明确降低。总时限覆盖数据准备、训练、评估和诊断；超时保留部分产物并返回 `budget_exhausted`，进程清理可能略超过截止时刻。

```yaml
tune:
  grid:
    lr0: [0.0005, 0.001, 0.002]
  max_trials: 3
  max_hours: 2
```

训练参数搜索默认按检测 Box mAP50–95、仪表 Pose mAP50–95 选取验证候选。对已有权重运行 `tune --weights <best.pt>` 则扫描验证集置信度与 NMS IoU；其配置副本的 `tune` 段使用 `confidence: [0.1, 0.25, 0.4]`、`iou: [0.5, 0.7]`、`max_trials: 6`、`max_hours: 1`、`objective: f1`，与训练参数搜索的配置分开。

## 运行产物

九个分发包各自拥有独立 `runs/`，以脚本所在包为根定位。例如人物为 `training_packages/person/runs/`，仪表为 `training_packages/meter/runs/`。包内模型输出为 `runs/<模型>/<版本与运行编号>/`，启动器日志为本包 `runs/_launcher/`；依赖环境可共享，训练产物按包保存。工作空间源入口使用工作空间的 `runs/<模型>/`；归档试验使用各版本目录内的 `runs/`。

每次模型运行中，`config/` 保存实际生效配置和框架参数，`metadata/` 保存代码版本、环境、设备、执行阶段和成功／失败／中断状态，`logs/` 保存输出和异常，`data/` 保存本次使用的图片、标签、来源对应与数量，`cache/`、`temp/` 接收框架缓存和临时数据。数据原件保持只读；运行目录中的数据副本使原始标签目录免受框架缓存写入影响。

权重和框架训练图表位于 `training/`，TensorBoard 事件位于 `tensorboard/`，独立评估位于 `evaluation/<评估名>/`。保留 loss、Box/Pose 指标、PR/F1 曲线、混淆矩阵、逐类 CSV/图表、逐图预测、预览与失败样本、固定阈值误检漏检和分来源统计；某类别无真值时相应指标为空并注明分母，不能填为成功。仪表关键点指标与框混淆矩阵分别解读。没有 OCR、读数或跨帧真值的评估明确记为未评估完整任务链。每次评估另存图表状态清单；框架未生成的原生曲线或逐类图以独立说明图标出缺失，不重建虚构曲线。历史运行可在 `postprocessing/` 下追加这一清单，保留原始指标和图表。

从其他电脑回传对应包的整个 `runs/`，保留任务、版本和运行编号，禁止覆盖同名运行。运行清单使用相对样本引用；框架生成的绝对路径记录保留其原运行环境意义，重新评估或恢复时由工具重定位运行内的数据路径。恢复训练读取原运行的冻结配置和数据，仅允许修改 batch、workers、device；训练完成后剥离优化器的最终权重不能继续恢复。初始模型和原始训练数据继续由各任务的数据与模型目录管理。

已回传 V3 的冻结权重、同场地候选诊断结果和下一轮对照设计见[实拍诊断与 V4 训练设计](research/FIELD_EVALUATION_V3_20261112.md)。

## V1、V2 与 V3

V1、V2 是本轮实际执行的低预算试验，采用与 V3 相同的完整工程和图表契约。每任务训练使用覆盖类别的同一组 50 张、seed 42、320×320、batch 8、workers 0；V1 为 SGD、lr0 0.01、1 epoch，V2 为 SGD、lr0 0.005、2 epoch。验证和最终测试沿用任务固定集合，不把训练小样本数量误写成评估样本数量。训练预算和学习率同时改变时，比较只说明组合配置效果，不做单因素归因。所有指标来自实际执行，不预设 V1/V2 低分或 V3 高分。

`lab/training_common/smoke_versions.py` 提供 plan、prepare、run、archive。试验准备时冻结当时的引擎与任务配置，每次串行训练并按真实状态记录；失败或中断的版本不能归档成完成版本。完成后放入 `Archive/technical/revisions/<任务>/<模型>_yolo_v1/` 或 v2，其中包含脚本、配置和本地 `runs`；文档不放入技术快照。来源和实际结果在[技术版本索引](../Archive/technical/revisions/README.md)及本文件记录。

V3 是当前完整训练工程及配置，在分发电脑上执行正式预算。V1/V2 本机与 V3 分发环境的软件版本分别记录；RTX 50 系电脑采用环境包的 CUDA 12.8 依赖组合；分发环境按本机 CPython 3.12、3.13 或 3.14 自动选择独立依赖目录。V3 代码准备完成与 V3 模型训练完成是两个状态。

V1/V2 的冻结源码保持运行时原貌；冻结后 V3 补充的路径约束、运行目录迁移、调参总时限及离线代理处理属于工程修改。正式比较 V3 模型时，用同一最终评估环境、数据和阈值重新评估三版权重；现有 V1/V2 原始结果继续保留其环境记录。

## 分发镜像

已有九包更新：

```powershell
.venv/Scripts/python.exe -B tools/build_training_packages.py --refresh-code person e_bike fire_sticker building_name_paper sign waste_bin_state waste_sticker meter plate_detection
```

构建器核对工作空间选定源与镜像的文件集合、逐字节图片内容和标签内容，然后刷新受管理的脚本、V3 配置及来源清单；保留已有图片、标签、权重和 `runs`。新增任务包时省略 `--refresh-code`，目标存在则拒绝覆盖。加入 `--include-environment` 将已构建的依赖包复制为同级 `portable_env/`；各任务实现放在包内 `scripts/`，根目录旧入口仅转发。

垃圾贴纸验证／测试采用 24 类共享图案场景；车牌 train/val 组合两批省赛合成来源，test 使用 CCPD 道路照片。这些选择在工作空间 V3 配置中明确列出，镜像不会自行改变数据组合。分发环境的构建、自动解释器发现、ABI/DLL 检查及迁移验证见[Windows 依赖包说明](../lab/training_common/portable_env/README.md)。

## 工程验证

2026-11-11，九个分发包通过新便携环境的 `check`，逐包记录位于各自 `runs/_verification/preflight.json`；合计 41,277 张图片与标签完成结构检查，镜像刷新前另逐字节核对来源图片和标签。训练工具 46 项检查通过，涵盖标签与类别契约、输出路径、数据来源映射、镜像刷新、启动失败清理、运行迁移和调参超时。便携环境另有四项兼容与隔离检查，以及换目录后的原生库、CUDA 算子、DataLoader 和 TensorBoard 检查，证据见其说明。

人物分发包通过新依赖环境执行了 CPU 8 张训练／8 张验证、320 输入、1 epoch 的集成检查，保存训练图表、TensorBoard、权重与验证结果，目录为 `training_packages/person/runs/_verification/v3_environment_check_20261111_103746_284299/`。最终引擎另在同一运行内完成 `evaluation/val_final_engine/` 评估与图表清单检查。这是启动与环境检查，V3 正式预算尚未执行。

中断与迁移续训使用本机旧训练环境在 CPU 上实际验证：8 张训练、4 张验证、计划 2 epoch；第 1 epoch 保存后主动中断，将完整运行复制到新位置，再用运行内数据完成第 2 epoch。原始 checkpoint 和指标文件内容保持不变，新运行 CSV 的 epoch 为 `[1, 2]`，证据见 `runs/engine_checks/resume_migration_20261111_104833/summary.json`。

## 2026-11-11 冒烟试验结果

九个模型的 V1/V2 共 18 次训练及完整 val/test 评估已完成并归档。每对版本使用相同的 50 张训练图，训练子集保留全部类别；垃圾贴纸覆盖 24 类。执行环境为 Windows、RTX 4060 Laptop、Torch 2.2.2+cu121、Ultralytics 8.2.103。训练加评估进程累计约 30.23 分钟。

下表为各任务固定测试集上的 Box mAP，取值 0–1；不是分类准确率或完整任务成功率。点击任务查看该任务的 val/test 比较数据和图表。

| 模型 | test 图片 | V1 mAP50 | V2 mAP50 | V1 mAP50–95 | V2 mAP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 人物 | 200 | 0.006143 | 0.009515 | 0.003756 | 0.005983 |
| 电动车 | 200 | 0.010233 | 0.011764 | 0.006172 | 0.007429 |
| 火焰纸贴 | 120 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 楼名纸条 | 120 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 禁行牌 | 200 | 0.013777 | 0.018612 | 0.008574 | 0.012602 |
| 桶开闭 | 120 | 0.049426 | 0.054491 | 0.036494 | 0.046427 |
| 垃圾贴纸 | 120 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 仪表 | 500 | 0.001824 | 0.064969 | 0.000512 | 0.046380 |
| 车牌检测 | 2800 | 0.000560 | 0.001065 | 0.000238 | 0.000500 |

仪表 Pose mAP50 为 V1 0.302557、V2 0.272389；Pose mAP50–95 为 V1 0.054447、V2 0.044231。本次 V2 关键点指标下降。两版在诊断置信度 0.25 下没有匹配检测框，关键点误差缺少可计算的配对，记录为 null。

九组比较分别核对了 val/test 样本来源、类别顺序与指标库版本。原始汇总为 `runs/smoke_versions/training_iterations_20261111/test_results.json`，运行与归档对应见同目录 `suite.json`，完整性检查见 `artifact_audit.json`。训练数据与测试数据的来源关系沿用[数据区说明](../datasets/README.md)；该表只描述这些固定集合。V3 正式训练与效果评估待分发电脑完成后回传。
