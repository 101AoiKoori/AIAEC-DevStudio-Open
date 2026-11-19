# 电动车姿态数据任务

本任务为 `e_bike`。定位任务范围内车辆并区分直立与倒伏。 下列描述用于导航，任务目标、标注范围与组合对照以配置中的 `e_bike` 条目为维护源。

- [学习目标与标注范围](../../../config/task_targets.yaml)；[组合轴与对照](../../../config/combination_axes.yaml)：街道／停车区、视面及人与车辆遮挡。
- [本任务数据配置](config/dataset.yaml)与[素材来源配置](config/source.yaml)：具体分支读取其支持的字段。
- [制作入口](../README.md#恢复生产入口)列出可执行分支；[任务脚本](./plan_grounded_distribution.py)是其中一个入口。产量与接线状态以[当前生产进度](../../../docs/DATASET_PRODUCTION.md#当前生产进度)及实际回执为准。
- [训练入口](../../../lab/features/e_bike/training/scripts/train_e_bike_v4.py)及[训练流程](../../../docs/TRAINING_WORKFLOW.md)：按任务配置和实际预算使用；脚本存在不代表模型已训练或完成交付。
- [计数实验全景与优化路线](../../../docs/research/E_BIKE_COUNTING.md)：已试方向、17/26组合的失败归因、未试路线和文件分类。`source.yaml#experiment_inventory` 为本机可重建的实验文件索引；使用 `lab/features/e_bike/training/scripts/audit_experiments.py` 刷新，不移动或删除权重与历史证据。

当前编批由 `source.yaml#sample_recipe` 指向的配方维护。`production_plan` 是底图合成的逐状态执行计划，`sample_plan` 汇总完整实拍、实拍底图合成和 Blender 底图合成，当前 `sample_manifest` 的 `plan` 指向汇总计划，全局登记 `task_targets.yaml#tasks.e_bike.plans` 只登记该汇总小样计划，避免重复计入合成子计划；用途为 sample，不计正式 train 预算。生成时输入和实际输出由持久生产回执保存，当前 `generation_plan` 指向小样内的输入快照。

当前车体使用两段原始平视旋转视频各 12 个中间时刻的真实帧和 SAM2 裁片，原向／镜像布局成对安排。镜像不改变姿态类别，不增加独立父来源数。完整实拍通过共享 `package_direct_real.prepare` 读取当前逐图资格与原尺寸标签。首轮基线按 `source.yaml#baseline_data` 指向的父组策略明确分配 train／val，并将逐图库分配同步到小样；已有 val 保留 val，原拟议但未执行的 test 用途在来源配置中改为当前 train 用途。原始图片与镜像逐像素核对，实拍框不填写无法测得的可见率。

```powershell
# 已有裁片可直接编批并生成当前三分支小样
.venv/Scripts/python.exe -X utf8 -B datasets/scripts/e_bike/plan_low_angle_batch.py
.venv/Scripts/python.exe -X utf8 -B datasets/scripts/e_bike/build_current_sample.py
.venv/Scripts/python.exe -X utf8 -B datasets/scripts/e_bike/audit_current_sample.py
# 将完整小样导出为首轮基线输入，按父组隔离 train／val
.venv/Scripts/python.exe -X utf8 -B datasets/scripts/e_bike/prepare_baseline_data.py
# 训练前核对实际数据，再执行有界基线
.venv/Scripts/python.exe -X utf8 -B lab/features/e_bike/training/scripts/train_e_bike_v4.py check
.venv/Scripts/python.exe -X utf8 -B lab/features/e_bike/training/scripts/train_e_bike_v4.py train --epochs 20 --wall-seconds 900 --run-name v4_baseline_20261118
# 需要继续从原视频补视面时使用提取入口，输出落在任务 annotated 区
.venv/Scripts/python.exe -X utf8 -B datasets/scripts/e_bike/extract_video_views.py --output datasets/annotated/e_bike/intermediate_views_20261118_refined --count 12
```

提取器使用本地 `models/weights/e_bike/sam2_t.pt`，按黄色车身收紧提示框，避免把椅面选为前景。提取结果要看实际裁片，已有输出沿生成物替换约定处理。汇总入口复用共享可恢复发布工具，按旧清单识别生成物并保留人工内容，阶段失败可恢复。

合成编批与出图调用同一接地、遮挡和镜像实现；接地点与阴影来自实际 alpha 轮廓底部接触簇。尺度按底图地面和远近布局作像素拟合，不能视为相机标定或多接地点物理校准。本批有效车车遮挡按至少 5% 记录，最低车辆可见率沿用 70%。实拍底图按 `layout.queue` 制作八车长队，包括两态各四车、全直立、全倒伏及各自镜像对照；近远尺寸和实际遮挡需同时通过现行可见性检查。图签、掩膜与带框浏览页位于 `datasets/samples/e_bike/`；实际数量和对比边界见[当前生产进度](../../../docs/DATASET_PRODUCTION.md#当前生产进度)。

本轮优先用原始车载视频的 PTS 抽帧入口筛查低视角长队：`lab/scripts/media/extract_capture_frames.py` 消费 annotated 区的两个 `queue_*_segments.csv`，帧清单保留原始 PTS、time_base 和秒数。`source.yaml#queue_review_manifest` 是本轮逐图视觉证据，入库后配方引用实际 record_id。近景和长队均已补齐当前完整逐车标签。七车长队首帧由用户确认数量、姿态并接受框，另一帧由时序对应复核；四车近景由用户确认两态各二车。人工确认与机器时序对应分别留证，不把相邻帧算作独立采集。当前清单无未决区域；历史部分标注保留在数据库版本中。

首轮基线已完成：上一轮 118 张完整小样导出至 `datasets/exports/e_bike/`，实际 split、来源和输入快照以导出清单为准，未创建正式数据版本。训练与同图计数对比记录在 `source.yaml#baseline_result`，模型位于 `runs/e_bike/v4_baseline_20261118/training/weights/best.pt`。已知场景验证的检测指标改善，但直立／倒伏数量同时全对仍少，重复框与漏检是下一轮定向补数据依据；镜像、邻帧及已知物料的结果不代表独立新场景表现。

本轮只用现有素材继续合成。`sample_recipe.json#queue_variants` 维护长队落点子集和逐图姿态数量，`paired_layouts` 维护实际配对预算；保留既有八车、四种车车遮挡、单车和空场对照。新增一／三／五／七车数量与姿态交换，在现有实拍底图及 Blender 理想空场上沿各自地面槽位，分别生成原向／车体镜像；不是新增实拍来源。审计入口逐图核对标签、可见掩膜、遮挡分配及镜像类别数量，刷新 `current_selection.focused_review`。当前小样增补与上一轮训练导出分开，旧模型不代表已学习本轮新增样本；实际数量查当前清单。 随后以旧基线初始化完成一次 10 轮短微调，输入与配置快照保存在相应 runs 目录；新增镜像对按组留出。计数未改善，未替换原基线，分组结果见 `source.yaml#short_finetune_result`。 两轮模型的逐张推理去重对照见 `source.yaml#nms_comparison_result`；实验脚本 `lab/features/e_bike/training/scripts/compare_nms.py` 读取冻结输入清单、两份权重与新的 runs 输出目录，固定 conf 0.25 和输入640，对比同类别0.7／0.5、跨类别0.5／0.3。输出逐图预览、数量与逐车匹配审计；参数探索不修改现行推理配置。 后续两阶段微小训练结果见 `source.yaml#two_stage_result`：研究快照将框映射为单类别 e_bike、从原姿态标注生成裁片，沿用原图分集。入口为 `lab/features/e_bike/training/scripts/two_stage_experiment.py` 的 prepare／classifier／evaluate；检测训练复用 `train_e_bike_v4.py` 与生成的 detector.yaml，分类训练复用已安装 Ultralytics 分类入口，单独设10轮及600秒上限。正确框、旧框重新判姿态、新单类别检测接分类分别评估，研究用类别映射不改变当前任务两态标签或正式推理接口。 双点辅助实验由 `lab/features/e_bike/training/scripts/keypoint_experiment.py` 的 prepare／evaluate 接续；prepare生成研究用Pose快照与train.yaml，训练继续复用同一V4入口。座椅源点维护于 `source.yaml#seat_anchor_review`，未知点不标，投影同步实际缩放、镜像和遮挡；可见框中心仅为几何派生点。模型与同起点同轮数检测对照见 `source.yaml#keypoint_result`，结果未证明双点带来净计数收益。

四部件点实验继续复用 `keypoint_experiment.py`，prepare 时显式传入 `--anchors`，读取 `source.yaml#four_part_anchor_review` 指向的源点；`point_names` 固定前轮／后轮／座椅／车篮顺序，镜像不交换前后轮身份，实拍未标部件和源裁片未知点保留 v0。整车可见框保持原标注，通过同一合成变换与掩膜投影各点。研究模型为单类别四点 Pose，计数时仍接既有姿态分类器。结果维护于 `source.yaml#four_part_result`，以末轮权重作主对照，自动选优权重单列；训练预算与普通检测器对照一致。四点及部件引导抑制未证明净计数收益，未修改正式类别或推理配置。

RT-DETR 可行性摸底使用 `lab/features/e_bike/training/scripts/rtdetr_experiment.py` 的 prepare／train／evaluate，复用已有输入审计、暂存、训练参数、运行记录和外部硬时限；模型与原生无NMS预测走已安装的 `RTDETR` API。共享V4启动器仍按YOLO架构构建，不将RT-DETR权重直接交给该启动器。官方预训练权重来源及哈希随实验保存，未引入外部源码。显式AdamW预热偏置学习率设0，首轮继承0.1的记录保留作配置排查；原始COCO起点与此前已适配小车的YOLO不同，短训不构成架构优劣定论。模型和三档固定阈值结果见 `source.yaml#rtdetr_result`，未采用作正式推理。

独立点计数摸底使用 `lab/features/e_bike/training/scripts/point_counter_experiment.py` 的 prepare／train／evaluate。该实现借鉴P2PNet的一对一匹配思路，冻结已有小车YOLO骨干，训练新的密集点位置与姿态分类头，推理不做NMS；未复制上游源码，也不是官方VGG16版本复现。点头类别为background／upright／fallen，使用独立研究入口，不交给通用detect／pose训练器。十轮训练、固定阈值比较、训练集拟合及完整权重读回证据见 `source.yaml#point_counter_result`。逐图两态数量可与既有结果比较；点距离匹配指标与检测框IoU指标分别解释。后续固定部件归属及凸包实验见下段，未与本点计数头联合训练。

多部件归属与凸包小实验使用 `lab/features/e_bike/training/scripts/part_hull_experiment.py --experiment runs/e_bike/<新实验名>`，复用已有四部件模型与姿态分类器，不重训。同类预测部件按置信度固定原型关联；两个共享部件，或一个共享部件加尺度加厚凸包的重合与邻近证据，才参与候选去重；未知点保留。固定参数比较宽松候选、同模型框NMS、部件归属和归属加凸包，不用GT计数参与处理。结果、重复预测与漏检审计及104张预览见 `source.yaml#part_hull_result`；已有YOLO参照使用另一模型权重，控制变量对照应看同模型框NMS。本轮不是独立部件检测或可学习归属模型。

输入尺度与候选诊断使用 `lab/features/e_bike/training/scripts/input_scale_experiment.py --experiment runs/e_bike/<新实验名>`，固定17/26组合权重，不重训，对比640／960／1280及640全图加两块固定比例远端裁区。继承已安装DetectionPredictor，在原生NMS前保存全部解码框及分数、原生保留索引，映射回原图；裁区边界过滤和跨视图NMS分别审计。GT仅用于IoU匹配及预览，不参与选区或候选处理。结果见 `source.yaml#input_scale_result`，含640复现检查、逐GT低分/NMS证据、四组预览和两例NMS专图；未替换当前推理设置。

640置信度/NMS回放使用 `lab/features/e_bike/training/scripts/threshold_nms_experiment.py --experiment runs/e_bike/<新实验名>`：保存未裁剪解码张量，原生硬NMS回放并保留raw_id，比较5档置信度×5档IoU及3组Gaussian Soft-NMS。原参数须逐框、逐姿态复现17/26；同一raw_id的姿态分类按32个一批缓存复用，GT只参与评估。`--weights <项目内权重> --small-grid` 用于训练候选的四组温和设置，新增误检raw_id比较仅限同一模型内部，跨模型按GT匹配和完整计数比较。结果维护于 `source.yaml#threshold_nms_result`，含28组参数、40轮微调末轮与自动选优权重、原7例定位/姿态状态及936张预览；训练仍走现有V4入口与批次finetune.yaml，原17/26组合保留。

定向数据研究使用 `lab/features/e_bike/training/scripts/build_focused_inputs.py --experiment runs/e_bike/<新输入名>`，消费 `source.yaml#focused_recipe` 并生成由 `focused_research_inputs` 登记的研究图签。调用既有接地/阴影/可见掩膜合成，布局族的姿态交换与镜像整组分集；研究可见率下限由配方显式传入，既有制作默认0.7保持。完整实拍裁片扩展至包含所有相交车辆，背景裁片须与全部训练GT框（外扩2px）无交集，再由原检测器排序选取硬负例；仅继承训练来源，不使用旧验证图。姿态标签与单类检测标签分开保存；训练YAML只含原26张val，新布局holdout仅在训练结束后评估。生成后 `audit.json` 核对掩膜框、遮挡、标签、原val字节、布局分集与跨split像素重复。`--resume-partial` 只允许配方相同且尚无完整清单的批次，不能覆盖完成输入。研究快照不切换当前sample_plan或正式exports；训练继续使用V4入口及该快照的train.yaml，评估使用已有关键点实验入口的detect分支，支持独立 `evaluation_group`。

本轮定向数据、训练与旧26/新32分组比较的实际证据见 `source.yaml#focused_result`。综合预览同时显示原17/26、末轮及mAP选优权重；原参数640/conf0.25/IoU0.3为主，IoU0.5为辅助对照。候选在合成留出上的改善与实拍子集退步分开报告，不据合并分数直接替换原模型。

等预算数据消融使用 `lab/features/e_bike/training/scripts/data_ablation_experiment.py` 的 prepare／train／evaluate／summarize，配方由 `source.yaml#ablation_recipe` 维护，`ablation_research_inputs` 指向实际准备回执。四组保留同一旧train，分别补等量旧图重采样控制、训练硬负例、完整实拍裁片、完整姿态/镜像合成族；验证与既有布局留出不加入train。训练调用现有V4有界入口，固定轮数并核对实际更新及学习率/梯度累积轨迹；末轮为主，姿态分类器固定。summarize分开统计漏车/误检、匹配框内姿态混淆及GT框分类诊断，并生成五组逐图错误标色预览。GT框分类只用于诊断，不能当检测成绩；控制组的重复采样明确保留父来源，不算新增独立图像。

整帧采样补充对照使用同一脚本，prepare显式传 `--recipe-key sampling_recipe --experiment runs/e_bike/<补充实验名>`，由同一随机选择顺序生成与裁片组逐一对应的原始训练父帧；train/evaluate读取该补充实验的快照，summarize传 `--control-study runs/e_bike/data_ablation_20261118` 合并既有四组。`sampling_recipe`与`sampling_research_inputs`维护这条接线；它是在四组结果之后设计的补充实验，不能当作五组同时预注册。综合结果与所有权重见 `source.yaml#data_ablation_result`。五组均为204槽位、固定20轮、实际76次更新及完全相同的学习率/累积轨迹；图像数、正负比例及标注实例数分别记录，不用相同轮数替代相同训练量证据。

数据制作使用同一 [dataset-production-loop](../../../.agents/skills/dataset-production-loop/SKILL.md)，无需按任务复制 Skill。

<!-- TASK_DESIGN:START -->

## 学习目标与组合设计

本节由共享配置生成；修改目标与组合请编辑 [task_targets.yaml](../../../config/task_targets.yaml) 和 [combination_axes.yaml](../../../config/combination_axes.yaml)，运行 `python lab/scripts/contracts/state_coverage.py --write`。

**学习及标注**

- 类别：upright、fallen
- 数量：empty、single、multi
- 负例：empty

标出任务范围内各车的可见车体及直立或倒伏。范围内车辆姿态不可判时保留存在和可见框证据，状态记未知。

零／单／多车是画面覆盖；停车区总车数和倒伏数从合格实例按现场覆盖汇总。

**素材用途**

- `direct_target`：全帧至少一辆任务范围内的目标车，范围内各车的可见框及直立／倒伏类别齐全。
- `target_foreground`：逐车掩膜、姿态和接地点；真实视面另用于视面对照。整帧其余车辆标签缺失时仍可用已确认裁片。

**组合轴与对照**

以下是本任务的设计条件；`dynamic` 表示依据实际素材和任务计划确定。

- `background`：dynamic
- `location`：street、parking
- `view`：front、side、back
- `person_occlusion`：none、present
- `vehicle_occlusion`：none、present

**布局**：街道／停车区的可用接地区域、车辆尺寸和遮挡，保留已有车辆标签。可用的远近和尺度证据用于细化布局，缺失时记录近似依据。

**pose_context**：车的有无和姿态不能由路面或背景猜出。

- 安排：同一确定无车底图配空场、单直立、单倒伏；再在可行的街道和停车区复用两种姿态，并让两态复用相同调色范围。
- 核对：同底图或相同场地角色内用实际标签核对零／直立／倒伏，按父视频统计独立覆盖。

**occlusion_view**：有人、相邻车辆或某一视面不能成为倒伏提示。

- 安排：同姿态有／无人车有效遮挡；在可辨视面中安排两态以及少量两态同框。
- 核对：用实例类别与真实视面、可见率和遮挡者交集逐框连接；总可见率无法区分人挡和车挡。

对照描述来自设计配置；结构化记录目前为部分编码，具体制作按任务入口和可用素材安排。

<!-- TASK_DESIGN:END -->

候选验证器实验入口为 `lab/features/e_bike/training/scripts/candidate_verifier_experiment.py <prepare|train|evaluate> --experiment runs/e_bike/<新实验名>`，配方见 `source.yaml#verifier_recipe`。仅原156 train挖掘GT裁片、IoU合格预测裁片和实际带上下文裁片与全部GT无交集的背景；局部车/两态歧义忽略，镜像按族分组、像素去重。以旧两态分类器初始化三类研究头（background／fallen／upright），完整160方形裁片训练20轮；初始化接触过原训练GT，内部val不称独立验证。冻结原检测器，先宽松候选、三类过滤、再最终NMS，比较原姿态／仅过滤／新姿态／概率加权，并用无过滤候选池控制隔离流程变化。旧26和既有布局32从不生成训练裁片，只作反复使用的开发诊断。实际旧26从17到20、实拍4到6/10、布局21/32保持；去掉2个车辆FP、修正3个匹配姿态，但原7个漏检仍在，低阈值重复未解决。1160张带框预览及训练末轮权重见 `source.yaml#candidate_verifier_result`；仅研究候选，不更改正式类别、生产样本或部署。

评价复核入口 `lab/features/e_bike/training/scripts/evaluation_audit.py --experiment runs/e_bike/<候选实验> --output runs/e_bike/<新审计目录>` 读取冻结预测，报告数量全对、严格逐车框姿态全对、IoU敏感性、贪心/最大有效匹配数差异及原帧/镜像族稳定性；不更改历史指标或原生AP。结果见 `source.yaml#evaluation_audit_result`，持续研究断点见 `research_review`。

完整性研究复用 `candidate_verifier_experiment.py prepare --recipe-key completeness_recipe --experiment runs/e_bike/<质量组>`：低IoU且与车相交的候选可显式作内部拒绝类（不是正式背景标签），默认原验证器仍只收GT无交集背景。`prepare-control --quality-experiment runs/e_bike/<质量组> --experiment runs/e_bike/<对照组>` 将新增局部槽位替换为同split纯背景重复，保持文件槽位、类别数量、正例、起点及训练轮数；train/evaluate接续。两组本次各1233train、252内部val，20轮；局部拒绝旧26仅16/26、严格16/26，对照20/26、严格18/26，未采用质量组。`compare_candidate_roles.py` 固定原语义筛选并比较分类器姿态输出，GT只评估；`benchmark_candidate_pipeline.py` 预热后同步CUDA测全链计算，核对58图各两遍预测等价。`source_contact_audit.py` 检查直接156训练角色接触、镜像像素与继承V3的4000训练图，对缩略图近邻只生成检查线索。结果见 `completeness_result`／`source_contact_result`；当前生产图签和部署不变。

训练种子复核使用 `verifier_seed_study.py <prepare|train|evaluate|summarize> --experiment runs/e_bike/<新研究目录>`，配方 `seed_study_recipe` 声明原seed42及新43/44；逐文件SHA验证1262裁片、类别和split相同，仅训练seed改变，每组20轮，固定原参数评估并报告所有组，不按验证最高分发布。`candidate_pool_audit.py` 只回放缓存，以GT计算NMS前的几何匹配容量并检查冲突，不作推理；原两例分别涉及同车框质量和相邻合框，不能全部按两辆车错误抑制处理。`evaluation_audit.py` 新增多GT兼容框与成功图歧义计数；框匹配成功不等价于视觉身份已验证。入口与回执见 `seed_study_result`／`candidate_pool_result`。

连续质量排序入口 `quality_rank_experiment.py <prepare|train|evaluate|diagnose-ranking> --experiment runs/e_bike/<新实验>` 读取 `quality_rank_recipe`。仅原156 train挖自然候选，分IoU区间限额取样，原帧/镜像族分组并对裁片像素去重；语义分类器1280维全局池化特征冻结，另存6维无绝对中心坐标的框几何。两个小MLP分别使用几何或图像+几何，回归最大GT IoU并学习无歧义同GT对的排序；内部val不独立于骨干初始化。主比较保留原语义门槛、检测conf0.25及NMS0.3，仅改变排序分数。GT辅助排序明确只作诊断；`diagnose-ranking` 是主结果后追加的纯质量分数诊断，单独保存。自建 `geometry.pt`／`image_geometry.pt` 是Torch质量头状态及归一化参数，须经本实验 `model_for` 加载并配合登记的冻结语义骨干，不能当作Ultralytics整车模型或直接部署。实际输入审计、训练/推理结果及预览见 `quality_rank_result`；本轮无净收益，保留原20/26候选。

P2/P3对照入口 `p2_detector_experiment.py <prepare|train-fit|evaluate-fit|train-full|evaluate-full>` 消费 `p2_recipe`：只复制共同骨干0–9层，两组重建neck/head；先8图拟合，再同156训练、26旧val、固定32布局诊断。第一轮每轮2批置信度未充分学习，后续把同8图重复为64槽位，未增加独立样本；两组都45/45匹配、8/8训练图全对。完整40轮各1560次更新，LR/累积轨迹逐步一致；P3/P2主参数仅14/12张旧26数量正确，未采用。`p2_confidence_probe.py` 是事后0.25/0.10/0.03×NMS0.3/0.7的有限标定核查，无新训练，仍未超原候选。`pose_aware_probe.py` 按预测姿态分组NMS，三seed均18/26，未采用。结果见对应source入口。

保留候选的单图检查：`.venv/Scripts/python.exe lab/features/e_bike/training/scripts/research_pipeline.py --source <实际图片路径> --output runs/e_bike/<新输出>`。配置由 `research_pipeline` 指向；检测器与语义分类器SHA256及类别角色均核对，复用成本脚本的 `infer_primary`，58图逐框逐姿态复现。输出 `annotated.jpg`／`result.json`，编号按框中心从左到右，仅研究预测；当前实现CUDA、640、conf0.25、NMS0.3、完整160裁片。未接入正式部署、ROI或跨帧汇总；实际权重和边界查 `goal_research_result`。
