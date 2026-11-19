# 任务训练、版本试验与分发

截至2026-11-19，仪表、火焰、楼名纸条、禁行牌、垃圾桶开闭和垃圾贴纸的完整训练与Val/Test结果均已回传；人物另有既定基线与窄亮边微调，电动车处于有界研究阶段。实际数据规模、选模轮次、误检漏检及微调回归见[阶段报告](research/DATASET_MODEL_REPORT_20261119.md)。下文的构建检查、启动命令及历史运行分别描述其对应阶段。

本页维护 V4 训练、续训和分发入口。上游提供指定 split 的图片、标签、数据入口和来源清单。生产进度与尚缺组合由[数据制作](DATASET_PRODUCTION.md#当前生产进度)维护，训练阶段读取已交付的数据批次。代码和配置位于 `lab/features/<任务>/training/` 与 `lab/training_common/`，便携包可按工作空间配置重建。

## V4 后续窗口的执行方向

八项 V4 任务共用训练引擎。每个任务独立确定类别、输入尺寸、初始化、增强、优化器及总轮数，默认用 `fixed_validation` 逐轮在 val 上评估和选优。数据交付后先用当前任务配置预检，再按已授权的预算启动基线训练；根据实际识别失败定向补合成或补拍实拍，并在用户指定的范围内微调。车牌沿用省赛模型。 [生产与训练交接](DATASET_PRODUCTION.md#任务分工与阶段边界) · [启动入口](#v4-任务包构建与启动)

**训练数据接线的验收。** 将任务正式图签导出到 `datasets/exports/<数据任务>/`，现有任务配置读取同目录的 `data.yaml` 和由真实批次清单派生的 `workspace_sources.json`。运行该任务原训练入口的 `check`。返回 0 且报告的 train/val 计数与本次发布计划及实际清单相符，才说明本次输入已接通。要核实某项缺口是否补齐，按任务现行计划中的类别、组合、来源或分集逐项对照实际批次清单和对应图签，报告实际数量与仍缺数量。缺类别和来源接触可从预检报告核对，识别效果按后续评估结果判断。

## 训练与微调的失败驱动闭环

以任务识别成功为目标。完整图签和可行来源具备后先训练基线，不要求先凑齐设计张数、固定实拍比例或新的独立采集场次。用可访问的实拍与合成样例检查模型的实际错误，逐例标明漏检、误检、类别混淆、尺度、视角、遮挡及下游关联失败；按错误补合成数据，或补拍真实条件下的实拍并标注，修正标签或推理阈值，在用户指定的预算内微调。每次保留基线、补充数据的来源、微调配置和同一口径的前后结果，观察到改善才沿用新权重。来源分组及测试条件用于解释结果，不作为开展基线训练和定向改进的门槛；没有独立新场景数据时不声称已测得该场景的效果即可。

人物已完成正式训练和限定微调。其余七项已有训练配置及共享引擎入口，组合与配额仍按各任务进度补齐。工作空间与便携包的路径、冻结数据和运行目录见[运行产物](#运行产物)，标签和来源规则见[生产流程](DATASET_PRODUCTION.md#从原始采集到导出)。

| 人物运行 | 实际结果 | 交接 |
| --- | --- | --- |
| 正式 `v4_formal_20261116` | 从第 9 轮的完整检查点续至第 150 轮，累计 7,100 次优化器更新。该运行采用当时的固定预算策略，留有 `last.pt` 和精简 `last_inference.pt`。该策略未产出 `best.pt`。 | [训练摘要](../training_packages/person/runs/person/v4_formal_20261116/metadata/optimization_summary.json) |
| 窄亮边专项 `v4_narrow_edge_finetune_20261116` | 完成 10 轮，留有 `best.pt`、`last.pt`、`last_inference.pt` 和训练后验证。 | 当前留用 [10 轮最佳权重](../training_packages/person/runs/person/v4_narrow_edge_finetune_20261116/training/weights/best.pt) · [验证产物](../training_packages/person/runs/person/v4_narrow_edge_finetune_20261116/metadata/post_training.json) |
| 窄亮边专项 `v4_narrow_edge_finetune_30ep_20261117` | 完成 30 轮，独立保存权重和训练后验证。 | [30 轮对照](../training_packages/person/runs/person/v4_narrow_edge_finetune_30ep_20261117/metadata/post_training.json) |

专项开发回归在同一条历史路线的 53 张图上执行，其中 18 张已进入两次专项训练。10 轮与 30 轮结果分别有 4 和 6 个误报，置信度为 0.25、匹配 IoU 为 0.50、NMS IoU 为 0.70。用户报告的现场置信度为 0.10，应与开发参数分开记录。独立路线盲测、现场验收和正式最终测试尚无已完成记录。 [10 轮开发回归](../training_packages/person/runs/person/v4_narrow_edge_finetune_20261116/evaluation/development_regression_53/summary.json) · [30 轮开发回归](../training_packages/person/runs/person/v4_narrow_edge_finetune_30ep_20261117/evaluation/development_regression_53/summary.json)

V4 暂存按任务和 split 核对逐图来源、完整标签及正负状态。验证报告可按来源、已知单条件或双条件及类别分层。`by_domain_conditions_class.csv` 的统计以整图作为分组单位，多条件组可以重叠。按遮挡或视角分组的整图数据不能当作某种实例的单独召回率。

## 模型与入口

| 模型 | 任务目录 | 训练脚本 | V3 配置 | V3 最大 epoch |
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

表为 V3 默认入口与预算，脚本位于各任务的 `training/scripts/`，配置位于 `training/config/`。V4 使用各任务自己的训练 YAML。检测任务与仪表八点 Pose 共享执行引擎，模型类型、输入尺寸、轮数和增强按实际配置传入。V4 的具体启动方式见[统一入口](#v4-任务包构建与启动)。

仪表八点顺序固定为温度轴心、温度针尖、湿度轴心、湿度针尖、温度最小刻度、温度最大刻度、湿度最小刻度、湿度最大刻度。

车牌现有 `train_cnn_ocr.py` 保持自己的多源 OCR 训练职责。红绿灯采用 HSV/ROI 标定，热异常采用楼层热响应方案，不因统一训练工具而增加 YOLO 模型。楼名与车牌检测不读取文字，仪表 Pose 不直接完成刻度读数，人物／电动车／火焰还需要关联和计数；通用训练指标与完整任务验收分别记录。

火焰既有[误报与过拟合对照方案](research/DATASET_COVERAGE_TRAINING.md#火焰误报与过拟合处理)，记录了红桶与信号灯硬负例及历史训练候选；对照结果按该历史方案的样本与参数解读。

## 人物 V4 工作空间入口

[入口脚本](../lab/features/person/training/scripts/train_person_v4.py)加载[人物 V4 配置](../lab/features/person/training/config/person_v4.yaml)，通过共享引擎读取正式人物导出。初始配置使用本地通用 YOLOv8n、640 输入、batch 8、AdamW 和轻量几何与曝光增强。已交付的工作空间导出为 train 3,000、val 300、test 300，其中直接实拍分别为 8、2、53 张。人物便携训练包当前包含 train 与 val。53 张已知路线实拍的历史接触和使用范围见[上文](#v4-后续窗口的执行方向)。

从仓库根目录运行 `check` 检查配置、权重和数据证据。确定训练预算后，将 `$personEpochs` 和 `$personWallSeconds` 设为实际轮数和秒数，再运行第二条命令。轮数也可写入配置。使用分发包时，在包根目录运行 `./train.ps1`。

```powershell
.venv/Scripts/python.exe -B lab/features/person/training/scripts/train_person_v4.py check
.venv/Scripts/python.exe -B lab/features/person/training/scripts/train_person_v4.py train --epochs $personEpochs --wall-seconds $personWallSeconds
```

- `training_policy.mode: fixed_validation` 为八项默认。按总轮数训练并逐轮验证。检测任务以 Box mAP50–95 选 `best.pt`，仪表以 Pose mAP50–95 选优。严格提高时更新，同分保留较早轮次。
- `training_policy.mode: early_stopping` 沿用相同选优指标。预先设定 `min_epochs`、`patience` 和 `min_delta`，最小轮数之后连续 `patience` 轮无足够进步才停止。历史 `fixed` 模式使用 train，可以导出末轮推理权重。
- 正常结束后，`best.pt` 和 `last_inference.pt` 为 FP16 精简推理权重。`last.pt` 和 `epochN.pt` 保留续训所需状态。`evaluate_after_training: true` 用精简后的最佳权重生成 `evaluation/val_final/summary.json`，状态与文件大小见 `metadata/post_training.json`。训练内和训练后均使用 val，整体占用同一次壁钟预算。最终 test 走明确的冻结与评估入口。
- `check` 会核对数据目录中的 `workspace_sources.json`、逐图任务标签与其证据。谱系和近重复审核状态随来源使用范围解释，未知时仍可使用具备完整监督的图签。新策略的运行按已锁定的 Ultralytics 8.3.203 实现，续训用同一运行的冻结数据和检查点。
- 预检和暂存报告逐类图数、框数及负图数；选中的训练集或验证集缺少目标类别时记录提示，供判断模型和评估的覆盖范围。来源组未知时报告各集合的未知数量，不因此阻止训练，也不将验证集称为独立；已知来源接触和相同图像仍按实际检查。标签在检查和暂存之间发生变化时，训练前退出。训练中损失或待保存的权重、优化器状态出现 NaN 或 Inf 时退出，保留上一次完整检查点。超时原因单独记录。
- 正式测试沿用同一运行的显式 val 结果。运行 `freeze --weights <候选权重> --run-dir <运行> --evaluation-name <val 名称>` 时，如果验证快照中没有 test，入口读取当前数据配置中的 test 并纳入该运行的冻结输入。之后使用返回的权重执行 `evaluate --split test`。选择说明可随记录提供。

## 断点续训

共享 YOLO 入口可从运行目录或该运行的 `training/weights/last.pt` 续训。迁移时复制整个运行目录，放入目标工作空间或任务包的 `runs/`。续训预检验证冻结数据、标签和配置对应的检查点签名。它使用运行里的配置及原总轮数，续跑的壁钟上限单独指定。

```powershell
.venv/Scripts/python.exe -B lab/features/person/training/scripts/train_person_v4.py train --resume "runs/person/$runId" --wall-seconds $resumeSeconds
.\training_packages\portable_env\run.ps1 -Python $pythonExe -Script .\lab\features\person\training\scripts\train_person_v4.py -Offline -ScriptArguments @('train', '--resume', "runs/person/$runId", '--wall-seconds', "$resumeSeconds")
```

新检查点记录原始模型、EMA、优化器、scaler、调度器、早停窗口、随机状态和优化步数。训练从最后完整轮之后回到同一个运行目录。未完成轮重新执行，数据迭代器和跨轮尚未提交的梯度累积会重新建立。每次续跑另有状态和优化日志。

已完成预算、已触发早停、缺少优化器或与冻结输入不符的检查点会被拒绝。按检查点实际格式恢复，旧 V3 保留其原生恢复路径。新训练在 [Python 3.12／3.14 的对应依赖](../lab/training_common/portable_env/README.md)中读取检查点，需要使用同版项目脚本和锁定的 Ultralytics。切换 Python 时 Torch 和 NumPy 也会变化，结果受版本及设备影响。

合成 CPU 夹具已核对 CPython 3.12.8 与 3.14.9 两个方向的固定预算续训、3.14.9→3.12.8 的早停续训，以及普通共享 V3 和 V4 的续训。普通 V4 的夹具还将整个运行复制到含空格的路径，原件权重和指标散列值一致。当前证据仅覆盖合成夹具和所选便携依赖。

独立入口 `train_cnn_ocr.py --resume <运行目录>/last.pt` 会按原标注、图片及字符表摘要核对输入，迁移时也需要这些输入文件。仪表只框预热用 `train_meter_box_prewarm_v4.py --resume <运行目录>/training/weights/last.pt --wall-seconds <秒数>`，复用原运行内已暂存的数据，继续冻结八点头。两条独立入口以合同单测核对恢复逻辑。

## V4 任务包构建与启动

### Windows PowerShell 入口

从仓库根目录运行同一个脚本。任务名可用 `person`、`e_bike`、`fire`、`building_name_paper`、`sign`、`waste_bin_state`、`waste_sticker` 或 `meter`。

```powershell
.\lab\train.ps1 person
.\lab\train.ps1 person -Resume
.\lab\train.ps1 person -Plan
```

任务训练 YAML 负责数据、初始权重、总轮数 `epochs` 和训练策略。共享 [launcher.yaml](../lab/training_common/launcher.yaml) 负责单次壁钟时限 `wall_seconds`、TensorBoard、端口和浏览器。`tasks` 下可按任务覆盖这些设置，也可通过 `train_config` 指定已审核的训练 YAML。首次启动前填好轮数和时限。`-Plan` 只显示当前入口和配置。

`-Resume` 自动选择同任务最新有 `last.pt` 且未结束的运行。也可加 `-Run <运行目录名>`。续训使用原数据和总轮数，`wall_seconds` 留空时复用上一段已记录的时限。新引擎将有效时限保存到运行目录的 `metadata/wall_budget.json`，随单次运行复制；历史运行可回查原包的启动记录，若两者都没有则需要指定时限。旧训练包需重新构建后才含此行为。TensorBoard 默认自动打开浏览器，终端进度用 `████░░` 显示。

其余七项的默认 YAML 也已启用逐轮验证与选优。启动前以该任务交付的 train、val 和确认的预算为准。

### 单独查看 TensorBoard

从仓库根目录运行对应任务。每个命令只读取一个任务，默认自动选择本机端口，可同时打开多个任务。`-Run` 可进一步指定单次运行。

```powershell
.\lab\tensorboard.ps1 person
.\lab\tensorboard.ps1 meter
.\lab\tensorboard.ps1 person -Package -Run v4_formal_20261116
```

`-Package` 读取现有便携包，查看缓存写在工作空间 `runs/_tensorboard/`。TensorBoard 的运行名按「任务 / 运行 / 阶段」显示。

- `00 Overall` 读取整次训练的逐轮记录。断点续训重跑的轮次按新记录显示，合为一条总曲线。
- `01 Train 2026-11-16 07.59.51 UTC` 为首次训练，后续按启动时间显示为 `02 Resume ...`、`03 Resume ...`。阶段曲线读取各自的事件日志并保留原轮次。旧续训 ID 在查看器中用阶段开始时间显示，新训练的原始事件目录采用 UTC 时间戳。
- `Evaluation/<名称>` 显示独立评估的汇总、逐类指标、推理耗时和诊断。实际执行验证时，逐轮曲线也显示 Box 或 Pose 的 Precision、Recall、mAP50 和 mAP50–95。固定预算模式的 mAP 来自实际运行的显式评估。

### 生成便携任务包

八项任务共用[构建器](../lab/scripts/data/package_v4_training.py)。从仓库根目录传入已审核的数据 YAML、该任务的 V4 训练配置和本地初始权重，保持 `workspace_sources.json` 与数据 YAML 同目录。先用 `--plan-only` 查缺件和输入问题。材料就绪后，移除 `--plan-only` 可生成新包。默认包名已经存在时，用 `--output-name` 指定尚未使用的新包名。

```powershell
.venv/Scripts/python.exe -B lab/scripts/data/package_v4_training.py --task $task --data $dataYaml --config $trainYaml --weights $initialWeights --output-name $newPackage --plan-only
```

火焰的默认包名为 `fire_sticker`。逐轮验证选优和验证早停包默认收录 train 和 val，纯固定预算包默认收录 train。增加 `--include-test` 可将已审核的 Test 一并交付，仍只使用训练策略指定的 split 训练和选模；`--wall-seconds` 将单次时限写入包内启动配置。`portable_env/` 放在任务包旁边。包内 `config/train.yaml` 和 `config/launcher.yaml` 保存本包配置，从新包根目录执行。

共享 `training_packages/portable_env` 已用于训练。跨机交接时，复制包含任务训练脚本、配置、数据和所需权重的任务目录及同级共享环境，在目标机从包内文件和交付环境运行。共享环境内含依赖，目标机提供匹配的 Python 与驱动。任务包是构建时快照，已有包的版本以本包清单为准。现有本机 `training_packages/person` 的清单早于代码指纹字段。

```powershell
.\train.ps1
.\train.ps1 -Resume
.\tensorboard.ps1
.\scripts\watch_training.ps1 -Follow
```

`launcher.yaml` 默认启用随训练自动启动的 TensorBoard，查看器会在训练期间周期刷新。Windows 启动器将训练和自动开启的 TensorBoard 纳入同一个 Job Object，启动器退出时系统回收其后代进程。独立的 `tensorboard.ps1` 可继续查看运行记录。强制关闭时状态文件可能停留在上次写入，终端监控可核对已关联启动器的进程。续训从最后完整检查点开始。

### 温湿度计八点 Pose 训练包

仪表维护源为 [meter_v4.yaml](../lab/features/meter/training/config/meter_v4.yaml)，使用既有八点 V3 权重微调。最多 150 轮，前 100 轮不累计早停耐心，之后连续 20 轮没有至少 0.001 的 Val Pose mAP50–95 提升则停止，平台期最早第 120 轮停止。严格最高 Val Pose mAP50–95 保存为 `best.pt`，同分保留较早轮次。仪表启动器时限为 36,000 秒（10 小时），包含暂存、训练和训练后验证；超时优先于轮数，记录为中断且不自动重启。另一次续训启动的时限应计入用户允许的总预算。

640 输入、batch 16、AdamW、学习率 0.0003；框、八点坐标与点存在性损失均启用，保持各头可训练。关闭翻转、Mosaic、MixUp、copy-paste 和额外透视，仅用轻量旋转、平移、尺度及 HSV。小表盘和短湿度针优先保留原图细节，首轮不因输入放大而声称新增信息。

[训练入口](../lab/features/meter/training/scripts/train_meter_v4.py)保留 `keypoint-report`，并提供 `meter-report`。训练后 Val 与显式 Val/Test 评估自动生成 `meter_report.json` 和 `meter_needle_errors.csv`，分别报告八点的像素及表盘尺寸归一化误差、温湿两针的投影方向角误差、针长和表盘尺寸分组、漏检、点不可用与负图误报。真值可见性 1/2 均参与坐标监督，0 为未知；专项报告只使用置信度至少 0.5 的预测点，缺失保留在分母中。投影角误差和 Pose AP 均不替代透视校正后的物理读数验收。

构建器为仪表保留任务入口和专项报告；带 `--include-test` 时一并交付 [evaluate_meter.ps1](../lab/features/meter/training/scripts/evaluate_meter.ps1)。完整包位于 `training_packages/meter`，与同级 `portable_env` 一起复制。在包根执行 `./train.ps1`；训练后执行 `./evaluate.ps1 -Run "runs/meter/<运行目录>"`，依次完整 Val、冻结最佳权重、Test。Test 不用于选模；共享素材与历史实拍接触随来源清单保留。

本轮包内图签核对为 Train 6,480／Val 1,080／Test 1,080；43 项相关检查及便携环境下 CUDA 的实际增强加载、单批 Pose 损失与反向梯度、单图推理通过。打包检查阶段未执行优化器更新，也未启动正式训练。[包内检查回执](../training_packages/meter/package_checks.json)。

2026-11-18 完整训练 `v4_20261118_140109_229142` 使用包内冻结输入与配置，CUDA异常后从第97轮检查点恢复，最终完成120轮；Val选择第97轮。Val、冻结权重及Test均已完成，最终状态见 [metadata/status.json](../training_packages/meter/runs/meter/v4_20261118_140109_229142/metadata/status.json)，指标与针角误差见[阶段报告](research/DATASET_MODEL_REPORT_20261119.md)。旧失败状态文件保留当时记录。

### 禁行牌训练包

当前完整包为 [training_packages/sign](../training_packages/sign/README.md)，携带 Train 2500、Val 300、Test 300 图签、来源及逐实例诊断属性、同类 V3 初始权重和共享训练引擎。与同级 `portable_env` 一起使用；无需原工作空间的数据或代码。包内预检、初始权重加载和一张 Train 图的 CPU 前向检查通过。随后完整训练完成115轮，Val选第95轮，Val/Test回传已核对，见[阶段报告](research/DATASET_MODEL_REPORT_20261119.md)。[检查回执](../training_packages/sign/package_checks.json)

训练上限120轮，前90轮不累计早停耐心，之后连续20轮没有至少0.001的验证mAP50–95提升则停止；仍保存严格最高Val分数的权重。单次最长36000秒（10小时），包含暂存、训练和训练后处理，硬时限优先。640输入、batch16、AdamW及现有轻量增强保留；不能从轮数保证收敛或无过拟合。

在包根运行 `./train.ps1`；续训用 `./train.ps1 -Resume`。完成后运行 `./evaluate.ps1 -Run "runs/sign/<运行目录>"`，按Val评估、专项报告、冻结、Test评估和专项报告的顺序执行；任一步失败即停止。Test不用于调参或选模。

`sign_report.json` 分别报告四视面（含斜视）、完整／截断／遮挡／杆底座、短边小于32像素的实例召回，以及来源、色调、场景配额、整图成功和空负误报。实例属性逐框核对标签与尺寸，原图继承目标的未知视面保留未知，误检不被猜分到某个视面。机器确认记录可直接消费，不补造人工签收。共享牌具及Test旧接触的范围随来源索引保留。

### 火焰贴纸训练包

[火焰包](../training_packages/fire_sticker/README.md)与同级 `portable_env` 一起交接。包内含 Train 3000、Val 300、Test 300 图签、当前来源和逐实例元数据、同类 V3 初始化权重、共享引擎及火焰专属 `audit`、`fire-report` 入口。本次只生成数据、配置及训练包；按用户要求不执行单元测试、模型推理或训练评估，也不启动训练。构建器保留打包所需的图签、文件及副本一致性检查。

训练维护源为 [fire_sticker_v4.yaml](../lab/features/fire/training/config/fire_sticker_v4.yaml)：最多 150 轮，前 120 轮不累计早停耐心，之后连续 15 轮没有至少 0.001 的 Val mAP50-95 改善则停止；严格最高验证分数仍逐轮留用。通常最早在第 135 轮触发该早停，150 轮为硬上限。单次启动最长 36,000 秒，包含数据暂存、训练及后处理，优先于轮数；超时或中断不记为完成。续训保留原轮数、优化器、调度器与随机状态，每次启动的时限单独记录；需要累计机时上限时使用剩余预算。

640 输入、batch 16、workers 0；AdamW、lr0=0.0002、lrf=0.08、weight_decay=0.0005、预热 3 轮及余弦衰减。缩放 0.05、旋转 2°、平移 0.02，保留适度 HSV 变化；关闭翻转、Mosaic、MixUp、copy-paste 和额外透视，保留小目标结构与主要颜色。AMP 关闭，种子 42，周期检查点间隔 5 轮；复用共享引擎的非有限状态处理、完整断点和外部硬时限。目标机显存不足时显式降低 batch，不默默缩小输入或更换训练语义。

当前 Train 含 1800 张普通正例、600 张贴纸与红橙干扰物同框正例、450 张硬负例及150张普通空负；Val/Test各有30张真实红桶硬负和30张普通空负。两集各90张墙面合成正例补入45个8–16像素、135个16–32像素的目标，保留既有较大目标与实拍正例；目标大小是640输入下的估计，不能解释成恢复传感器细节。总计3600图、5704个指定贴纸框，父来源分集沿用，Blender派生只在Train。共享资产、重复父照片/视频帧及既往开发接触仍限制独立场景结论。

从包根运行 `./train.ps1`，续训运行 `./train.ps1 -Resume`。`audit`描述来源、空负类型、同框干扰、目标尺寸和已知逐楼归属；不执行模型推理。训练后按需运行 `./evaluate.ps1 -Run "runs/fire_sticker/<run-directory>"`，依次完成 Val、火焰专项报告、选模冻结、Test及专项报告；任何失败即停止，已有评估不覆盖。Test不参与早停、选模或阈值调整。

`fire-report`只消费已有预测，报告目标尺寸/来源/干扰类型/楼宇的召回、整图无漏检且无额外误检的比例、计数误差及空负误报；等数量但框错位不计整图成功。误检保留整图上下文，不猜测其所属楼宇。未知来源或视角继续记为未知。

火焰实拍桶微调使用[精简专项计划](../datasets/scripts/fire/config/reshoot_compact_finetune_plan.json)、[工作空间配置](../lab/features/fire/training/config/fire_sticker_reshoot_finetune_v4.yaml)和原包内 `config/finetune.yaml`；数据入口为 `dataset_finetune_compact/data.yaml`。[核对回执](../training_packages/fire_sticker/dataset_finetune_compact/verification.json)记录Train 1224、Val/Test各300张及父来源检查，完整 `dataset_finetune/` 候选池保留，原一键启动配置不自动切换。配置固定10轮、640输入、batch 4、AdamW学习率0.00004、预热1轮且 `warmup_bias_lr: 0.0`，预热阶段各参数组由0渐增至当前调度学习率，避免使用默认0.1的偏置预热起点；最长7200秒。10轮逐轮验证选优，使用新优化器从远程best初始化，不按resume恢复。本轮已完成10轮及原best对比，见[运行结果](../training_packages/fire_sticker/finetune_result.json)。包内来源证据引用已接通，连续评估复用相同字体缓存以避免Windows映射文件占用。以下为本轮使用的复现入口：

```powershell
.\scripts\run_training_tool.ps1 -Entry (Join-Path $PWD "scripts/train.py") -ScriptArguments @('train', '--config', 'config/finetune.yaml', '--wall-seconds', '7200')
```

### 楼名纸条训练包

当前[楼名包](../training_packages/building_name_paper/README.md)与同级 `portable_env` 一起交接。包内含 Train 3000、Val 300、Test 300 的单类 paper 图签、完整来源与逐纸元数据、V3 同类 YOLOv8n 初始权重、共享引擎及楼名专项入口。监督为纸条定位，不读取 OCR 真值。共享便携依赖下的包内预检、权重加载及一张 Train 图的 CPU 有限前向通过，代码指纹和环境清单匹配；30 项包与专项回归通过。[检查回执](../training_packages/building_name_paper/package_checks.json)

训练维护源为 [building_name_paper_v4.yaml](../lab/features/fire/training/config/building_name_paper_v4.yaml)。上限 150 轮，前 120 轮不累计早停耐心，之后连续 15 轮未取得至少 0.001 的 Val mAP50–95 改善则停止；严格最高 Val 分数对应的权重仍逐轮保留。机时维护于 [launcher.yaml](../lab/training_common/launcher.yaml) 的楼名项，打包时显式写入 36,000 秒（10 小时）。这是单次启动的硬时限，涵盖暂存、训练和训练后处理，优先于轮数；到时中断不记为训练完成。续训保留原总轮数与优化器状态，每次启动的时限继续记录；需要累计机时预算时，应使用剩余预算，不把重新启动当作已获授权的额外机时。

| 参数 | 当前设置与目的 |
| --- | --- |
| 输入、batch、workers | 640、16、0；多数生成图原生 640，提高输入尺寸不能恢复丢失细节。目标显存不足时可显式减小 batch，续训允许变更 batch／workers／device。 |
| 优化器 | AdamW，lr0=0.0002，lrf=0.08，余弦衰减，weight_decay=0.0005，预热 3 轮；使用已有同类检测权重初始化。 |
| 增强 | 旋转 2°、平移 0.02、缩放 0.05，轻度 HSV；关闭翻转、Mosaic、MixUp 和额外透视，保留已生成的低机位及左前大偏航。 |
| 数值及保存 | AMP 关闭，确定性种子 42，5 轮周期检查点；复用共享引擎的非有限损失检测、完整断点状态与硬时限控制。 |
| 固定诊断 | conf=0.25、匹配 IoU=0.5、NMS IoU=0.7、max_det=100；Test 不用于阈值或模型选择。 |

包内运行 `./train.ps1`；已有未完成运行可用 `./train.ps1 -Resume`。训练后运行 `./evaluate.ps1 -Run "runs/building_name_paper/<运行目录>"`，依次执行 Val 评估、`paper-report`、兼容的 `paper-sizes`、冻结选模、Test 评估及两份诊断。任何阶段失败即停止；已有评估结果不覆盖，通用 CLI 可用新名称显式接续。

`paper_report.json` 包含输入短边 0–8／8–16／16–32／大于32像素的逐实例召回，A/B/C 已知纸条、偏航与机高生成控制、单／两／三楼、有楼无纸／部分有纸／全有纸、来源分支与父组、整图全对率、计数误差和空负误报。等数量但漏一纸又误检一处不计整图全对；误检按整图上下文分组，不猜其属于哪栋楼。元数据与最终标签顺序及坐标核对，实拍未知机位保持未知。Val/Test 本批没有≤8像素实例，相应召回为未知；共享楼体、有限底图和既往开发接触使这些结果属于同源诊断。原远程训练已完成 135 轮，第 108 轮为 best，回传 best 与 Val/Test 文件已核对；后续独立新视频微调见下节。

### 楼名新视频独立微调

按用户指定，本次数据与运行均在 `training_packages/building_name_paper`。数据入口为 [finetune_dataset/data.yaml](../training_packages/building_name_paper/finetune_dataset/data.yaml)，[清单](../training_packages/building_name_paper/finetune_dataset/manifest.json)记录逐纸四角、完整标签、父视频与派生关系；不读取或混入旧 `dataset` 的 Train。Train 只用 `buildingname1.mp4`、`buildingname2.mp4`，Val 只用 `buildingname.mp4`。裁剪及双／三纸条多视图拼图均来自新视频且只进 Train；拼图是人工布局，不能作为实拍多楼效果证据。三段共用现场与楼体，Val 仅支持同场景诊断，未额外构造 Test。

维护源为 [building_name_paper_video_finetune_v4.yaml](../lab/features/fire/training/config/building_name_paper_video_finetune_v4.yaml)，同步到包内 `config/video_finetune.yaml`。从用户回传的 `building_name_paper_best_Val_Test/best.pt` 初始化新优化器，640 输入、batch 4、AdamW `lr0=0.00005`、2 轮预热，偏置预热学习率也显式为 0.00005。上限 60 轮，最少 20 轮后启用 10 轮耐心早停，每 5 轮保存；单次 36,000 秒硬时限。复用原包共享引擎，完整 last、Val 最高分 best 与正常完成后的 FP16 精简输出沿用既有行为。

从包根调用现有入口，可指定配置而无需改写原训练入口：

```powershell
./scripts/start_training.ps1 -Config config/video_finetune.yaml -WallSeconds 36000 -NoTensorBoard
```

[new_video_finetune_20261118](../training_packages/building_name_paper/runs/video_finetune/new_video_finetune_20261118/metadata/status.json) 已在 Windows RTX 4060 Laptop 离线完成 43 轮，约 344 秒，按 Val 选择第 33 轮 best；完整 last、FP16 best、last_inference 及周期权重均保留。原模型与微调模型在 21 张 Val（13 纸条）固定 conf=0.25、NMS IoU=0.7、匹配 IoU=0.5 下，TP 4→10、FP 2→0、FN 9→3，整图全对 10→18；残余漏检为边界截断纸条。Val 不含 B 楼纸条，51 张 Train 原始帧回放的 TP 22→39、FP 5→0、FN 17→0 不代表独立验证。

选模后才回放原 Test，未将其图像用于训练或选模。本机统一 FP32 协议下 TP 333→334、FP 50→38、FN 1→0，与真纸条高度重叠的额外框 34→2；但背景额外框 16→36，整图全对 268→264，多计图 31→36。因此新权重保留为独立候选，不覆盖原 best，不宣称重复计数问题已全部解决。本机回放原模型的数值与回传远程报告不同，不能跨协议直接相减；CPU／GPU 同协议复验一致。详见[包内结果](../training_packages/building_name_paper/finetune_result.json)和[逐图比较](../training_packages/building_name_paper/runs/video_finetune/new_video_finetune_20261118/comparison_cpu/comparison.json)。这里只验证单帧框与计数，跨帧唯一楼宇计数及 ROS 实车流程未评估。

用户追加要求检查 conf=0.8，保持同一已选权重与其余推理设置。相对 conf=0.25，微调候选在本机原 Test 的 TP 334→333、FP 38→3、FN 0→1，整图全对 264→296，重叠额外框 2→0；新视频 Val 的 TP 10→8、FP 0→0、FN 3→5，额外过滤掉 f000180（约 0.79）和 f000192 的真纸条。Train 原始帧回放仍为 TP 39、FP 0、FN 0。[阈值对比图](../training_packages/building_name_paper/runs/video_finetune/new_video_finetune_20261118/comparison_conf080/threshold_comparison.jpg)与[逐图结果](../training_packages/building_name_paper/runs/video_finetune/new_video_finetune_20261118/comparison_conf080/comparison.json)单独保存；只做用户要求的阈值检查，不以原 Test 再选权重，未覆盖 0.25 训练评估配置。

### 垃圾桶开闭训练包

当前补拍微调使用[专用配置](../lab/features/waste_bin/training/config/waste_bin_state_reshoot_finetune_v4.yaml)，从用户回传的原始远程 `best.pt` 初始化新优化器，统一 640 输入，本次 batch 4、完整 10 轮；AdamW 初始学习率 0.0001、预热 1 轮，显式设置 `warmup_bias_lr: 0.0`。该键经共享引擎传入 Ultralytics，避免沿用其 0.1 默认偏置预热值。新数据按[真实桶主导配方](../datasets/scripts/waste_bin_state/config/reshoot_finetune_plan.json)重编，来源、整桶标签、实测遮挡及数量读取[当前导出](../datasets/exports/waste_bin_state/manifest.json)与[核对回执](../datasets/annotated/waste_bin_state/reshoot_finetune/verification.json)。同一补拍 session 全部归 Train，原 Val/Test 字节保留，已知回归不能代表独立新实拍泛化。单桶合成仅放大每色开闭均有证据的机位；多桶使用实拍兼容正面／背面组，缺失低机位开盖及高俯视不由缩放补造。[本次运行](../runs/waste_bin_state/v4_real_priority_finetune_10ep_batch4_20261118/metadata/status.json)已完整完成 10 轮且无训练中断，best 为第 1 轮；完整 last、精简 last 及周期检查点均已核对。原始模型与便携包保持原位，未自动替换部署权重。

工作空间[启动器](../lab/training_common/launcher.yaml)的 `waste_bin_state` 项已选择此专项配置，沿用上一轮本地运行的 7,200 秒单次硬时限。`lab/train.ps1 waste_bin_state -Plan` 只查看选择结果，实际启动仍需另行执行；原完整基线配置与便携包继续保留。

本次[统一 640 对比](../runs/waste_bin_state/v4_real_priority_comparison640_batch4_20261118/bin_state_comparison.json)中，原始／best／last 的 Val 整桶漏检为 265／257／258、开闭错 0／0／2、额外预测 2／3／23；Test 整桶漏检 27／28／49、开闭错 0／2／2、额外预测 2／43／49。按用户优先范围单列 ≥90px 桶，Val 正确数为 110／118／117（134 个），Test 为 213／214／213（218 个）。效果改善有限并伴随误检增加，不将损失下降解释为普遍改善。所有模型、类别、图签、尺寸和固定阈值同口径；Test 不参与模型或阈值选择。

[指定 f002017 图](../runs/waste_bin_state/v4_real_priority_finetune_10ep_batch4_20261118/diagnostics/f002017/comparison.json)在 640 下三者均漏红桶，best 和 last 均无绿桶开闭重复；960 下 last 检出红桶（闭盖 0.838），但将开盖绿桶、蓝桶误判闭盖。原模型、上轮候选、本轮 best/last 的预测与对照图均保留。用户进一步指出桶身自带的分类标识图案不匹配：[标识对照](../runs/waste_bin_state/v4_real_priority_finetune_10ep_batch4_20261118/diagnostics/f002017/classification_marking_comparison.jpg)确认 Blender 红、绿桶图案与实物不同；该对象与下方另贴物品卡区分。局部遮挡诊断另存，但遮掉标识不等于换成实物同款，当前未做分类标识替换的控制变量实验，也未因此修改资产或追加训练。

上一轮 [960 输入的 batch 4 运行](../runs/waste_bin_state/v4_reshoot_finetune_10ep_batch4_20261118/metadata/status.json)已完成 10 轮，Val 最优为第 5 轮，期间 CUDA 错误后从完整检查点恢复，完整 last 与精简权重均已保存。[同条件 640 比较](../runs/waste_bin_state/v4_reshoot_comparison640_batch4_20261118/bin_state_comparison.json)：Val 整桶漏检 265→258、开闭判错 0→3、额外预测 2→17；Test 整桶漏检 27→66、开闭判错 0→1、额外预测 2→13。此结果只说明统一 640 下的回归，不能把输入尺寸影响全部归为小桶能力丢失。初始训练包、回传权重及已执行运行输入保留；新配方使用原始远程权重，不续训这轮候选。

[桶开闭配置](../lab/features/waste_bin/training/config/waste_bin_state_v4.yaml)使用通用 COCO `yolov8n.pt` 初始化，640 输入、batch 16、AdamW、初始学习率 0.001、3 轮预热、120 轮逐轮验证选优。随机缩放为 0.05，减轻对小桶口特征的影响。类别保持 `open`、`closed`，以整桶框监督。

`training_packages/waste_bin_state` 携带 Train 3,500、Val 186、Test 253 张图签、来源记录、初始权重及评估入口。与同级 `portable_env` 一起交接。训练前在包内 `config/launcher.yaml` 填写实际单次运行时限；本次只完成打包检查，不启动训练。

训练完成后，从包根执行 `./evaluate.ps1 -Run "runs/waste_bin_state/<run-directory>"`，对 Val 选出的检查点执行 Val 评估、冻结后执行 Test。专项报告用不区分类别的 IoU 一对一匹配将整桶漏检、开闭判错和正确识别分开统计，额外预测计入误检；同时按颜色（含黑色）、来源、机位及目标大小分组。颜色只用于评估诊断，不参与训练损失或选模。报告读取包内来源元数据；缺少的属性记为未知，不补造人工确认。共享母场景和同源派生的结果按当前清单范围解释。

### 垃圾贴纸训练包

[垃圾贴纸配置](../lab/features/waste_bin/training/config/waste_sticker_v4.yaml)与数据入口使用同序 24 类名称，保留原类别 ID 和 V3 最优权重初始化。当前配置为 960 输入、batch 8、AdamW、初始学习率 0.0002、150 轮逐轮验证选优；工作空间任务启动设置及便携包均为单次 36,000 秒（10 小时）硬时限，包含暂存、训练和训练后精简与验证。正常完成后保留完整 `last.pt` 和每 5 轮检查点，导出精简 `last_inference.pt`，并将 Val 选出的 `best.pt` 精简为 FP16 推理权重。触发硬时限可能未完成全部轮数、精简或最终验证，以实际运行记录为准。轻量几何与曝光增强保留原图贴纸布局，小贴纸使用较高输入分辨率训练。当前输入为 Train 1,800、Val 240、Test 240，制作范围见[生产进度](DATASET_PRODUCTION.md#当前生产进度)。

`training_packages/waste_sticker` 携带三集图签、来源证据、初始权重及专属评估代码。与同级 `portable_env` 一起复制后，从包根执行 `./train.ps1`；训练结束后执行以下命令，使用实际运行目录替换占位符。

```powershell
.\evaluate.ps1 -Run "runs/waste_sticker/<run-directory>"
```

该命令对 Val 选出的 `best.pt` 执行完整 Val 评估，冻结检查点、配置和数据，再执行一次 Test。每次 Val／Test 评估自动保存 `waste_sticker_class_report.json`，含 24 类精确率与召回率、零至四贴整图无漏检且无额外误检的成功率、空负例误检率。通用 mAP、逐图预测和诊断图沿用共享引擎。Test 不参与选模和阈值调整；共享固定图案及 Blender 基础资产的结果属于当前来源的受控诊断。

### 垃圾贴纸白色背景硬负例微调

2026-11-18 用户要求弃用上一轮微调初始化，重新从远程完整训练的原始 `best.pt` 微调。当前[计划](../datasets/scripts/waste_sticker/config/recall_preserving_finetune_plan.json)和[清单](../datasets/annotated/waste_bin/waste_sticker_recall_preserving/manifest.json)完整回放原 Train，补充同一 Reshoot 会话的真实白色干扰，以及全 24 类小尺寸白卡／真图案对照；Reshoot 全部归 Train。当前[微调配置](../lab/features/waste_bin/training/config/waste_sticker_white_stripe_finetune_v4.yaml)为 10 轮、960、batch 1、AdamW lr0=0.00002、freeze=10、warmup_bias_lr=0、36,000 秒上限。禁用几何缩放增强以保留显式小目标像素分层，温和颜色增强保留。原 Test 及上一轮挑战 Test 按相同解码像素和标签保留为已接触的回归；不进入 Train，也不作独立新实拍结论。[数据核验](../datasets/annotated/waste_bin/waste_sticker_recall_preserving/recall_preservation_audit.json)通过后已启动[本轮运行](../runs/waste_sticker/v4_recall_preserving_from_remote_20261118/metadata/train_status.json)，本轮已完成 10 轮，Val 选出第 7 轮，last、周期检查点及精简权重完整保留。[对比结论](../runs/waste_sticker/v4_recall_preserving_from_remote_20261118/comparison/workflow_status.json)：原 Val 540 个目标仍全检出，但已知 Test 总漏检 5→7，条纹负例仍有 11／20 张误报，候选不作为原始 best 的替代。冻结骨干参数完全相同不等于检测召回不变。[用户指定实拍对照](../runs/waste_sticker/v4_recall_preserving_from_remote_20261118/comparison/f002017/predictions.json)中，原始 best 框出四张纸，候选漏蓝桶一张；两者均存在类别错判。该图只用于单张诊断，未用于选模或训练。

以下为上一轮已完成实验，权重不作为本轮初始化。

远程完整基线完成 150 轮，用户回传的 `best.pt` 为第 94 轮 Val 选出的 FP16 权重。[专项计划](../datasets/scripts/waste_sticker/config/hard_negative_finetune_plan.json)登记为 `fine_tune`，与全量预算分开统计；[图签清单](../datasets/annotated/waste_bin/waste_sticker_hard_negative/manifest.json)保留生成时配方、模型挖掘排序、既有完整无贴证据及正负对照身份。白线和空白纸卡不新增类别，真正白底物品贴保留原标签；原 Test 不进入 Train。

上一轮从回传 best 初始化新优化器，960 输入、batch 1、AdamW、初始学习率 0.00005、单次 36,000 秒上限，实际配置见[运行快照](../runs/waste_sticker/v4_white_stripe_finetune_20261118/config/effective.json)。原 24 类顺序及完整 `last`、Val 选优 `best`、周期检查点与精简导出沿用共享引擎。工作空间 `launcher.yaml` 的贴纸任务指向当前专项配置，正式基线配置保留。

本次原计划 20 轮，用户在运行中要求第 10 轮停止；完整 `epoch10.pt` 原子发布后，仅停止本任务训练进程。当前配方与新运行默认配置已改为 10 轮，既有运行的 20 轮配置快照与前十轮学习率轨迹保留。[停止回执](../runs/waste_sticker/v4_white_stripe_finetune_20261118/metadata/user_epoch_stop.json)记录实际 10 轮、248 个检查点内优化器步及父控制器 code 15；训练状态按用户中止解释。第 8 轮为前十轮 Val 最优；[停止后导出入口](../lab/features/waste_bin/training/scripts/finalize_waste_sticker_user_stop.py)验证完整检查点后复用共享精简与评估，不再训练，`last.pt` 字节保持不变。后续冻结和 Test 读取各运行自己的配置快照。

新 Val 的[基线评估](../runs/waste_sticker/v4_20261118_175431_633652/evaluation/val_selection/summary.json)在固定 0.25 阈值下为 639 TP、143 FP、1 FN，104 张空负中 57 张误检。[当前微调运行](../runs/waste_sticker/v4_white_stripe_finetune_20261118/metadata/train_status.json)正常执行后，用 Val 选定的权重冻结并执行 Test；[分组比较入口](../lab/features/waste_bin/training/scripts/compare_white_stripe_finetune.py)仅消费已生成的逐图评估，分别报告原回归、白线、空白卡和干扰同框正例。基线与微调使用相同样本、尺寸、类别和阈值。新增挑战图使用既有 Test 父组，结论属于受控回归，不表述为独立新实拍泛化。

中文显示使用目标机已安装的 CJK 字体（本机 Microsoft YaHei），同时配置原生标注和 Matplotlib；缺少 CJK 字体时保留原离线字体后备。字体只从本机读取用于当前运行，不将 Windows 系统字体作为可分发素材打包。

[完整 Test 对比](../runs/waste_sticker/v4_white_stripe_finetune_20261118/comparison/test.json)已完成：同阈值总 FP 162→25、TP 575→576、FN 5→4；原 Test 回归组 FP 16→3，但 FN 1→2，条纹负例仍有 12／20 张误检。mAP50–95 为 0.95697→0.94747，属于误检减少伴随局部召回及定位指标回归的部分改善。[用户指定失败图复核](../runs/waste_sticker/v4_white_stripe_finetune_20261118/comparison/user_selected_failures.json)区分同位置重复类别框、高置信白色部件误检与小药片漏检。挖到的误检硬负全部为渲染来源，Train／Val 缺少在 960 输入下短边不超过 24 像素的药片正例；部分挑战白条使用了局部接地小框，不能解释成道路尺度斑马线覆盖。这些失败是重新编制本轮真实白色部件、小目标及同尺度空白卡对照的依据。[上一轮候选权重](../runs/waste_sticker/v4_white_stripe_finetune_20261118/training/weights/best.pt)不代表上述问题已经解决。

## 旧工作空间运行入口

V4 从[统一入口](#v4-任务包构建与启动)启动。复现 V3 时使用对应版本的任务脚本、配置、样本和阈值。独立测试集留出至选模、调参和阈值选择结束。 [V3 回传与诊断](research/FIELD_EVALUATION_V3_20261112.md)

## 运行产物

重新构建的各分发包应拥有独立 `runs/`，以脚本所在包为根定位。例如人物为 `training_packages/person/runs/`，仪表为 `training_packages/meter/runs/`。包内模型输出为 `runs/<模型>/<版本与运行编号>/`，启动器日志为本包 `runs/_launcher/`；依赖环境可共享，训练产物按包保存。工作空间源入口使用工作空间的 `runs/<模型>/`；归档试验使用各版本目录内的 `runs/`。

每次模型运行中，`config/` 保存实际生效配置和框架参数，`metadata/` 保存代码版本、环境、设备、执行阶段和成功／失败／中断状态，`logs/` 保存输出和异常，`data/` 保存本次使用的图片、标签、来源对应与数量，`cache/`、`temp/` 接收框架缓存和临时数据。数据原件保持只读；运行目录中的数据副本使原始标签目录免受框架缓存写入影响。

权重和框架训练图表位于 `training/`，TensorBoard 事件位于 `tensorboard/`，独立评估位于 `evaluation/<评估名>/`。保留 loss、Box/Pose 指标、PR/F1 曲线、混淆矩阵、逐类 CSV/图表、逐图预测、预览与失败样本、固定阈值误检漏检和分来源统计；某类别无真值时相应指标为空并注明分母，不能填为成功。仪表关键点指标与框混淆矩阵分别解读。没有 OCR、读数或跨帧真值的评估明确记为未评估完整任务链。每次评估另存图表状态清单；框架未生成的原生曲线或逐类图以独立说明图标出缺失，不重建虚构曲线。历史运行可在 `postprocessing/` 下追加这一清单，保留原始指标和图表。

从其他电脑回传对应包的整个 `runs/`，保留任务、版本和运行编号，避免覆盖同名运行。运行清单使用相对样本引用。需要续训时按[断点续训](#断点续训)重定位数据。初始模型和原始训练数据继续由各任务的数据与模型目录管理。

训练交接引用冻结的数据、实际优化器更新和运行状态。评估附权重、数据范围、分辨率、置信度、NMS 和匹配口径。已接触的路线和样本报告其接触范围。 [V4 产出数据](DATASET_PRODUCTION.md#当前生产进度) · [V3 实拍诊断](research/FIELD_EVALUATION_V3_20261112.md)

## V1、V2 与 V3

V1、V2 是本轮实际执行的低预算试验，采用与 V3 相同的完整工程和图表契约。每任务训练使用覆盖类别的同一组 50 张、seed 42、320×320、batch 8、workers 0；V1 为 SGD、lr0 0.01、1 epoch，V2 为 SGD、lr0 0.005、2 epoch。验证和最终测试沿用任务固定集合，不把训练小样本数量误写成评估样本数量。训练预算和学习率同时改变时，比较只说明组合配置效果，不做单因素归因。所有指标来自实际执行，不预设 V1/V2 低分或 V3 高分。

`lab/training_common/smoke_versions.py` 提供 plan、prepare、run、archive。试验准备时冻结当时的引擎与任务配置，每次串行训练并按真实状态记录；失败或中断的版本不能归档成完成版本。完成后放入 `Archive/technical/revisions/<任务>/<模型>_yolo_v1/` 或 v2，其中包含脚本、配置和本地 `runs`；文档不放入技术快照。来源和实际结果在[技术版本索引](../Archive/technical/revisions/README.md)及本文件记录。

V3 是已有完整训练工程及配置；八个完整运行已回传，人物生产训练由用户中止。V4 按本节后续方向独立设计。V1/V2 本机与 V3 分发环境的软件版本分别记录；RTX 50 系电脑采用环境包的 CUDA 12.8 依赖组合；当前分发环境按本机 CPython 3.12 或 3.14 自动选择独立依赖目录。V3 代码准备完成与 V3 模型训练完成是两个状态。

V1/V2 的冻结源码保持运行时原貌；冻结后 V3 补充的路径约束、运行目录迁移、调参总时限及离线代理处理属于工程修改。正式比较 V3 模型时，用同一最终评估环境、数据和阈值重新评估三版权重；现有 V1/V2 原始结果继续保留其环境记录。

## 分发镜像

V4 任务包从正式数据入口、训练配置与本地权重构建。将 `portable_env/` 放在任务包同级，启动时按本机 CPython 3.12 或 3.14 加载对应依赖。移交时保留完整运行目录和所用任务包配置。 [任务包构建与启动](#v4-任务包构建与启动) · [便携环境与迁移](../lab/training_common/portable_env/README.md) · [技术迭代归档](../Archive/technical/README.md)

新构建包记录可用便携环境 manifest 的构建期摘要；运行记录对照实际选用的环境，环境变化或无法比较只记录状态，不阻止训练。摘要只标识 manifest，不能证明依赖文件完整。已构建的包是源码快照，不会随工作空间修改自动更新：本机现有人物包仍包含旧训练引擎及旧版本记录逻辑，需要新行为时应另行构建并保留已有运行内容。

## 工程验证

2026-11-11，九个分发包通过新便携环境的 `check`，逐包记录原位于各自 `runs/_verification/preflight.json`，随已删除的任务包移除；合计 41,277 张图片与标签完成结构检查，镜像刷新前另逐字节核对来源图片和标签。训练工具 46 项检查通过，涵盖标签与类别契约、输出路径、数据来源映射、镜像刷新、启动失败清理、运行迁移和调参超时。便携环境另有四项兼容与隔离检查，以及换目录后的原生库、CUDA 算子、DataLoader 和 TensorBoard 检查，证据见其说明。

人物分发包通过新依赖环境执行了 CPU 8 张训练／8 张验证、320 输入、1 epoch 的集成检查，保存训练图表、TensorBoard、权重与验证结果，原目录为 `training_packages/person/runs/_verification/v3_environment_check_20261111_103746_284299/`，已随人物包移除。最终引擎另在同一运行内完成 `evaluation/val_final_engine/` 评估与图表清单检查。这是当时的启动与环境检查；后续 V3 生产状态见[回传记录](research/FIELD_EVALUATION_V3_20261112.md#v3-训练行为)。

中断与迁移续训使用本机旧训练环境在 CPU 上实际验证：8 张训练、4 张验证、计划 2 epoch；第 1 epoch 保存后主动中断，将完整运行复制到新位置，再用运行内数据完成第 2 epoch。原始 checkpoint 和指标文件内容保持不变，新运行 CSV 的 epoch 为 `[1, 2]`，证据见 `runs/engine_checks/resume_migration_20261111_104833/summary.json`。

## 2026-11-11 冒烟试验结果

九个模型的 V1/V2 共 18 次训练及完整 val/test 评估已完成并归档。每对版本使用相同的 50 张训练图，训练子集保留全部类别；垃圾贴纸覆盖 24 类。执行环境为 Windows、RTX 4060 Laptop、Torch 2.2.2+cu121、Ultralytics 8.2.103。训练加评估进程累计约 30.23 分钟。

下表为各任务固定测试集上的 Box mAP，取值 0–1；不是分类准确率或完整任务成功率。点击任务查看该任务的 val/test 比较数据和图表。

| 模型 | test 图片 | V1 mAP50 | V2 mAP50 | V1 mAP50–95 | V2 mAP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| [人物](../lab/features/person/training/runs/person/v1_v2_comparison_20261111/comparison.json) | 200 | 0.006143 | 0.009515 | 0.003756 | 0.005983 |
| [电动车](../lab/features/e_bike/training/runs/e_bike/v1_v2_comparison_20261111/comparison.json) | 200 | 0.010233 | 0.011764 | 0.006172 | 0.007429 |
| [火焰纸贴](../lab/features/fire/training/runs/fire_sticker/v1_v2_comparison_20261111/comparison.json) | 120 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| [楼名纸条](../lab/features/fire/training/runs/building_name_paper/v1_v2_comparison_20261111/comparison.json) | 120 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| [禁行牌](../lab/features/sign/training/runs/sign/v1_v2_comparison_20261111/comparison.json) | 200 | 0.013777 | 0.018612 | 0.008574 | 0.012602 |
| [桶开闭](../lab/features/waste_bin/training/runs/waste_bin_state/v1_v2_comparison_20261111/comparison.json) | 120 | 0.049426 | 0.054491 | 0.036494 | 0.046427 |
| [垃圾贴纸](../lab/features/waste_bin/training/runs/waste_sticker/v1_v2_comparison_20261111/comparison.json) | 120 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| [仪表](../lab/features/meter/training/runs/meter/v1_v2_comparison_20261111/comparison.json) | 500 | 0.001824 | 0.064969 | 0.000512 | 0.046380 |
| [车牌检测](../lab/features/plate/training/runs/plate_detection/v1_v2_comparison_20261111/comparison.json) | 2800 | 0.000560 | 0.001065 | 0.000238 | 0.000500 |

仪表 Pose mAP50 为 V1 0.302557、V2 0.272389；Pose mAP50–95 为 V1 0.054447、V2 0.044231。本次 V2 关键点指标下降。两版在诊断置信度 0.25 下没有匹配检测框，关键点误差缺少可计算的配对，记录为 null。

九组比较分别核对了 val/test 样本来源、类别顺序与指标库版本。原始汇总为 `runs/smoke_versions/training_iterations_20261111/test_results.json`，运行与归档对应见同目录 `suite.json`，完整性检查见 `artifact_audit.json`。训练数据与测试数据的来源关系沿用[数据区说明](../datasets/README.md)；该表只描述这些固定集合。后续八个 V3 完整运行已经回传，人物生产训练中途停止，见[回传记录](research/FIELD_EVALUATION_V3_20261112.md#v3-训练行为)。


## 电动车 V4 首轮有界基线（2026-11-18）

当前完整三分支小样由 `datasets/scripts/e_bike/prepare_baseline_data.py` 导出，父组策略维护于 `datasets/scripts/e_bike/config/source.yaml#baseline_data`，实际输入为 `datasets/exports/e_bike/data.yaml`（train 108／val 10）。初始化使用既有 V3 权重，参数维护于 `lab/features/e_bike/training/config/e_bike_v4.yaml`，batch 4、imgsz 640；运行 `v4_baseline_20261118` 完成 20 轮，约 164.84 秒，无独立 test。

同一验证输入前后 mAP50 为 0.579→0.740，mAP50-95 为 0.286→0.425；固定 conf 0.25、NMS IoU 0.7 的逐图两态数量同时全对为 0/10→1/10。检测改善，计数仍不可靠；主要失败为同一倒伏车重复框及部分车漏检。验证包含镜像与邻帧，且物料与场景已知，结果仅作定向补数据诊断。

[模型](../runs/e_bike/v4_baseline_20261118/training/weights/best.pt) · [计数对比](../runs/e_bike/v4_baseline_20261118/evaluation/val_final/pose_count_comparison.json) · [结果回执](../datasets/annotated/e_bike/baseline_result.json)。同图计数对比入口为 `lab/features/e_bike/training/scripts/report_pose_counts.py --before <旧评估目录> --after <新评估目录>`，校验两次图像、真值和诊断阈值一致。
