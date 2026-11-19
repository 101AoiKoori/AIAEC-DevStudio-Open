# V4 数据制作

## 当前生产进度

**2026-11-19 汇总核对。** 仪表、火焰、楼名纸条、禁行牌、垃圾桶开闭和垃圾贴纸均已有完整训练及Val/Test回传。仪表最终完成120轮（第97轮Pose最优），禁行牌完成115轮（第95轮最优）；垃圾桶开闭当前Train已切换为804张真实桶主导微调输入，完整基线的3500张Train保留于历史清单和训练包。数据规模、训练指标及各次微调得失见[数据集与模型训练阶段报告](research/DATASET_MODEL_REPORT_20261119.md)与[聚合证据](research/DATASET_MODEL_REPORT_20261119.json)。早期构建、审计和试跑记录保留其发生时状态。

**全局推进原则。** 直接用已合格的共享底图、Blender 和实拍来源编排可行组合，生成完整图签并推进基线训练。按当批容量调整来源份额，缺少某种实拍或待补判的帧只影响对应分支。入库端复用已有机器证据，补齐目标标签和本次用途所需的空间，资格齐备即可消费。产量、父来源、split 和评价范围按实际清单记录，再按漏检、误检、类间混淆、关联及计数失败定向补合成或补拍实拍并微调。[训练与微调](TRAINING_WORKFLOW.md#训练与微调的失败驱动闭环)

**人物 V4 已收官。** 数据计划已执行，train、val、test 均已正式导出，图片与 YOLO 标签逐项配对。场景底图与直接实拍的父来源按集合隔离，未发现跨集同字节图片。各集合共享固定立牌图案和部分车辆裁片。模型已完成 150 轮正式训练和白侧板窄亮边专项微调，留用 10 轮微调运行的最佳推理权重。[人物生产命令](../datasets/scripts/README.md#人物最小生产路径) · [数据入口](../datasets/exports/person/data.yaml) · [来源索引](../datasets/exports/person/workspace_sources.json) · [训练交接](TRAINING_WORKFLOW.md#人物-v4-工作空间入口)

| 集合 | 总张数 | 合成 | 直接实拍 | 社区框 | 非社区框 | 空标签图 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [train](../datasets/exports/person/train/manifest.json) | 3,000 | 2,992 | 8 | 7,217 | 1,770 | 257 |
| [val](../datasets/exports/person/val/manifest.json) | 300 | 298 | 2 | 825 | 188 | 19 |
| [test](../datasets/exports/person/test/manifest.json) | 300 | 247 | 53 | 792 | 141 | 43 |

`train` 的合成包括实拍底图 2,092 张和 Blender 底图 900 张，覆盖单人、仅白背、正背同框、多人及电动车遮挡。60 张[人物小样](../datasets/samples/person/manifest.json)中有 52 张合成布局进入该批训练计划。`test` 的 53 张实拍与 83 张合成底图来自已接触的车载路线，其余 164 张合成使用另一个旧停车视频。该集合用于已知路线回归和合成评估。

以下为当前有界批次。2026-11-18 按用户要求先跑五任务小样，逐张解码并核对标签数量、格式和框范围，抽看带框及关键点的拼图。电动车和禁行牌用现有编制器按当前逐图库重编本次小样，电动车当前计划维护于 `datasets/scripts/e_bike/config/production_plan.json`，`datasets/annotated/e_bike/current_sample_plan.json` 保留此前 40 张批次的生成输入；禁行牌当批输入保存在 `tmp/sign_sample_recheck_plan.json`。火焰随后重编当前数据库三分支计划，实际记录见下表。其中禁行牌随后完成440张配对小样和3100张三集数据，当前配额以 `source.yaml#full_dataset` 为维护源，执行计划为该键指向的 Train／Val 计划，全局统计登记于 `config/task_targets.yaml#tasks.sign.plans`；其余任务进展见各自当前清单。生产回执保留输入与实际输出。楼名当前小样中的 5 张 train 实拍也出现在同一任务的 3000 张 train 批次中，来源复用按清单计量。

| 任务与分支 | 实际输出 | 当前边界 |
| --- | ---: | --- |
| [楼名四分支小样](../datasets/samples/building_name_paper/manifest.json) | 80 张、114 个纸条框、6 张空负 | 实拍 9 张／6 框，实拍底图替字 12 张／12 框，纯 Blender 31 张／48 框，楼体＋实拍底图 28 张／48 框。新分支覆盖两张车载底图 × 七种楼数组合 × 低机位左前／平视正前，地面投影和实际 Cycles 阴影随掩膜保留；来源分集及既往评估接触按清单记录。[合成拼图](../datasets/samples/building_name_paper/hybrid_contact_sheet.jpg) |
| [仪表共享底板与光色小样](../datasets/samples/meter/manifest.json) | 433 张：54 实拍 Pose、1 部分点实拍、126 Blender 预览、8 实拍底图合成、244 Blender 底图合成；432 份 Pose 标签，126 张透明 RGBA | 共用池按 3 灯光场景×8 路线分层选取 96 张渲染底板，加 3 张已审查接地区的实拍底板，共 99 张。白盘／彩盘同几何配对，饱和度、曝光、前景相对亮度三轴独立轮换。修正实拍底板楼体重叠落点；保留 0055 八点纠偏。当前 unassigned，图签及配对检查通过。[单张预览](../datasets/samples/meter/contact_sheet.jpg) · [三集合配额草案](FULL_TRAIN_PLAN_DESIGN.md#meter)。 |
| [火焰四种生成方式小样](../datasets/samples/fire/manifest.json) | 194 张、205 个火焰框 | 直接实拍含空负 55 张／15 框，实拍墙面合成 90 张／120 框，完整 Blender 31 张／46 框，楼体＋实拍场景 18 张／24 框。15 张实拍底图已配准墙面；混合分支使用楼角 PnP 相机近似拟合、逐帧视锥与地面足迹检查、Cycles 物理阴影及线性光合成。[独立复验](../datasets/annotated/fire/current_sample_review/audit.json)重解码 49 张 ID 图并核对 70 个渲染贴纸框，37 个父来源跨分集重叠为零。整楼未完整观察时保持 null。[混合拼图](../datasets/annotated/fire/current_sample_review/hybrid_contact_sheet.jpg)。 |
| [电动车合成迭代小样](../datasets/samples/e_bike/manifest.json) | 当前 182 张：完整实拍／镜像 42、实拍底图合成 70、Blender 底图合成 70；直立 336、倒伏 322 框 | 现有素材新增 64 张一／三／五／七车与姿态数量交换、车体镜像对照，各 32 张实拍／Blender底图；[带框预览](../datasets/annotated/e_bike/current_queue_review/index.html) · [拼图](../datasets/annotated/e_bike/current_queue_review/examples.jpg)。上一轮 118 张导出与基线保留；新增48张已做10轮短微调，计数未改善。 |
| [禁行牌全量数据](../datasets/exports/sign/data.yaml) | Train 2,500 张／2,000 框／500 空负；Val 300 张／275 框／75 空负；Test 300 张／225 框／75 空负 | 完整、截断或遮挡、可辨杆底座、空场和普通杆配额已兑现。三集背景父来源分开，前景共享；Test有既往诊断接触；全量图签和训练入口预检通过。440 张已接受小样保留作对照。  [完整训练包](../training_packages/sign/README.md)已构建：120轮上限、90轮保护、20轮早停耐心、36000秒（10小时）；随后完成115轮及Val/Test回传，Val最优第95轮。 |
| [垃圾贴纸完整数据集](../datasets/exports/waste_sticker/data.yaml) | Train 1,800、Val 240、Test 240，共 5,130 个框 | 每图 1–4 桶，各集覆盖 24 类。Train 180 张无贴负例、单至四贴各 405；Val、Test 各 24 张负例、单至四贴各 54。逐图来源和实际范围见导出清单。 |
| [楼名全量检测集](../datasets/exports/building_name_paper/manifest.json) | train 3000／val 300／test 300 | 4152 个纸条框、756 张空负；四分支分配及逐集审计见当前清单，只有 paper 检测监督。 |

**仪表当前全量已完成。** Train 6480、Val 1080、Test 1080，共 8640 张；7776 张正例、864 张无表图、62208 个关键点。2448 张独立透明前景、579 张 train 底板及 val/test 各 72 张专用底板参与制作；54 张原生实拍按 45／7／2 入三集并保留像素。随机位置尺度、约 70% 连续双针采样、轻微模糊／压缩／噪声已接入；逐图框、八点与显示区四角共用变换。每 250 张合成块独立审计并保存回执；三集图签审计、跨集 RGB 重复检查及 V4 训练数据入口预检通过。当前工作集未创建发布标签或正式版本目录。[数据入口](../datasets/exports/meter/data.yaml) · [全量清单](../datasets/exports/meter/manifest.json) · [Train 预览](../datasets/exports/meter/train/contact_sheet.jpg) · [生产入口](../datasets/scripts/meter/README.md#当前全量制作入口) · [数据预检](../datasets/annotated/meter/full_production/production_inputs/training_data_preflight.json)。


2026-11-18 电动车进入首轮基线：用户确认七车长队为直立 4／倒伏 3、四车近景为两态各二；另一长队帧按时序对应补齐，证据区分人工与机器复核。首轮 118 对图签核验后完成训练，以下指标仅指该轮冻结导出。全局登记仍为 `config/sample_plan.json`、用途 sample，不将拷贝后的基线重复计入全量 train 草案。`prepare_baseline_data.py` 消费当前完整小样和 `source.yaml#baseline_data`，导出 108 张 train／10 张 val，父路径分离。以既有 V3 权重初始化，batch 4、640 像素、20 轮，约 165 秒完成；[best.pt](../runs/e_bike/v4_baseline_20261118/training/weights/best.pt)的 mAP50 为 0.740，同图两态数量全对从 0/10 提高到 1/10，仍有同车重复框和漏检。验证含镜像与邻帧、已知场景物料，不能代表独立实场泛化。二维裁片的视面、倒伏多点接地与照明仍有近似。[逐图证据](../datasets/annotated/e_bike/queue_video_review.jsonl) · [首轮结果](../datasets/annotated/e_bike/baseline_result.json) · [计数对比](../runs/e_bike/v4_baseline_20261118/evaluation/val_final/pose_count_comparison.json)。

2026-11-18 随后按用户要求只用现有素材继续合成，新增 64 张数量／姿态对照，维护于 `sample_recipe.json#queue_variants` 与 `paired_layouts`。在现有实拍底图与 Blender 理想空场沿各自长队地面槽位取一／三／五／七车，三／五／七车各交换两态数量，原向和车体镜像成对；既有八车、四种车车遮挡、单车和空场继续保留。当前小样 182 对图签、552 个可见掩膜框、68 对合成镜像及 42 张实拍／镜像像素检查通过；未新增实拍、未改动上一轮训练导出和模型；随后短微调见下方记录。全局仍登记原 `config/sample_plan.json`，用途 sample。审计与预览通过 `datasets/scripts/e_bike/audit_current_sample.py` 重建；阴影、视面与接地是二维近似，当前成图不证明模型效果。

2026-11-18 电动车新增64张数量对照小测：沿用 `v4_baseline_20261118/training/weights/best.pt`，640输入、conf 0.25、NMS IoU 0.7、匹配IoU 0.5，不训练、不调参。两态数量同时全对28/64，逐实例无漏检且无额外误检27/64；单车16/16、三车10/16、五车2/16、七车0/16。实拍底图16/32、Blender底图12/32，镜像数量一致16/32对。该批共享既有前景和场景素材，仅作合成诊断，不能与上一轮10张实拍验证直接比较。[结果](../runs/e_bike/v4_baseline_20261118/evaluation/synthetic_count_smoke_20261118/summary.json) · [逐图数量](../runs/e_bike/v4_baseline_20261118/evaluation/synthetic_count_smoke_20261118/pose_counts.json)。

2026-11-18 电动车短微调对照已完成：以首轮 best.pt 初始化，旧 train108 加新增合成48，共 train156；旧实拍 val10 加成对留出的新增合成16，共 val26。640输入、batch4、lr0=0.0001，10轮约66.61秒（硬上限600秒）。同图固定阈值下，合成留出两态数量全对6/16→6/16，原实拍1/10→1/10；26图平均两态数量绝对误差合计1.81→2.65，误检40→69、漏检17→10。mAP50 0.808→0.825，mAP50-95 0.518→0.558，检测指标改善但计数未改善。保留原基线，不采用本轮候选作部署。输入与配置为本次研究快照，当前 samples 与上一轮 exports 保持原位；同源合成留出含共享素材和底图，仅作诊断。[结果与候选模型](../datasets/annotated/e_bike/short_finetune_result.json)。

2026-11-18 电动车继续完成两轮模型×四种NMS的完整推理对照，固定conf0.25、输入640、逐张rect推理，共26张、102个GT实例。两轮模型两态数量全对依次为：原同类IoU0.7均7/26；同类0.5为基线11/26、短微调8/26；跨类0.5为12/26、9/26；跨类0.3均13/26，逐车无漏检无多报均12/26。基线误检40→7、漏检17→21，短微调误检69→16、漏检10→18；基线实拍最好3/10。去重改善重复计数，同时会损失真实框覆盖；GT几何覆盖损失不等同物理车辆被删，部分是较差的大框压掉较准的框。参数仍仅作同源探索，未修改现行推理配置。逐张原设置复现既有TP/FP/FN；早期混合尺寸batch4探索与旧框过滤估计保留追溯，以本次完整逐张推理为准。[结果](../datasets/annotated/e_bike/nms_comparison_result.json) · [逐图八组预览](../runs/e_bike/nms_comparison_single_20261118/review.html)。

2026-11-18 电动车两阶段微小训练已完成：保留156/26原图分集，研究快照将检测标签映射为单类别e_bike，裁出训练556／验证102个带8%上下文、保比例补方的姿态样本。单类别检测器与姿态分类器各10轮，分别约67.73秒和33.47秒，各有600秒硬上限。分类器由本地原检测器迁移兼容骨干参数、重建分类头，未下载额外权重。正确GT框上98/102辆姿态正确、23/26图两态数量全对（仅作分类上界诊断）；旧检测框重新判姿态最好12/26，新单类别检测接分类最好10/26（实拍2/10、合成8/16，NMS0.3），均未超过上一轮去重基线13/26。新单类别检测器在此预算内分车不足，未替换原基线或正式配置。两份.pt、输入快照、分类顺序、分组与逐框匹配结果在[两阶段结果](../datasets/annotated/e_bike/two_stage_result.json)，成图见[四组对照](../runs/e_bike/two_stage_20261118/evaluation/review.html)。

2026-11-18 电动车完成框中心／座椅双点辅助实验与同轮数普通检测对照，各追加10轮、每轮运行硬上限600秒，沿用156/26分集和同一姿态分类器。24个源裁片视面经AI视觉核对，15个可确认可见座椅表面中心，9个不标；实拍座椅未专门标注。框中心由可见框派生，不是新增语义监督。投影严格复现裁片裁边、平面变换、缩放、镜像和摆放，并用重建的完整／可见掩膜检查552个合成框；训练座椅可见300、已知遮挡2，验证可见42，其余为未标。双点模型配合框NMS0.3达到15/26两态数量全对（实拍3/10、合成12/16），直接中心／座椅点去重只有1～6/26，重复点抑制不足。同起点、同lr0.0003、同10轮的普通单类别检测器对照达到17/26（实拍4/10、合成13/16），因此不能把15/26归因为关键点净收益。保留现行推理与原基线。[源座椅点](../datasets/annotated/e_bike/seat_anchor_sources.jpg) · [完整结果与两份.pt](../datasets/annotated/e_bike/keypoint_result.json) · [五组点／框去重预览](../runs/e_bike/keypoint_inputs_20261118/evaluation/review.html)。后续四部件点训练记录见下段。

2026-11-18 四部件点加整体框实验已完成：24个既有裁片视面，前轮／后轮／座椅／车篮有可确认AI近似源点的视面数分别15／14／15／21；未知源点和实拍部件点不标。沿原156/26划分投影，训练已知部件点1298、验证146，框与原图保持不变。由与普通检测对照相同的单类别权重初始化，lr0.0003、10轮约72.16秒、600秒上限；共用同一姿态分类器。固定第10轮权重、同框NMS0.3时，四点和普通检测器均14/26两态数量全对，逐车全匹配分别13/26与14/26，四点实拍3/10、合成11/16。四点自动选优best也14/26，未超过此前普通检测器best17/26；后者选优口径不同，作为次要历史结果。用至少两个可靠对应部件区分重叠框的实验抑制仅2/26或6/26，大量重复预测，未采用。前后轮身份在镜像时保持，全部10组推理读回同一26图及GT。保留原模型、现行推理及118张导出。[源点预览](../datasets/annotated/e_bike/four_part_anchor_sources.jpg) · [四点结果与权重](../datasets/annotated/e_bike/four_part_result.json) · [逐图对照](../runs/e_bike/four_part_inputs_20261118/evaluation/review.html)。

2026-11-18 已实际试跑RT-DETR-L：使用现有Ultralytics原生实现与官方v8.3.0预训练资产，下载来源、66,511,432字节及SHA256随实验记录，未引入外部源码。沿用156/26图与同一姿态分类器，640输入、batch2、AdamW lr0.0001、AMP关闭、deterministic关闭，每次10轮及900秒上限。首轮继承偏置预热lr0.1，10轮约254.64秒；排查后第二轮仅改warmup_bias_lr=0，10轮约303.13秒。修正后末轮mAP50为0.4651；原生无NMS推理在预定conf0.1／0.25／0.5下两态数量全对分别0/26、0/26、2/26，conf0.25误检285／漏检49，conf0.5误检26／漏检56。第一轮三档均0/26，保留其配置排查证据。六组推理验证同一26图、102个GT，未替换原模型及118张导出。该模型从通用COCO权重起步，与已有多轮小车微调YOLO的17/26参照初始化不同，只说明本轮配置和短预算尚未得到可用计数，不能否定RT-DETR架构。[结果与权重](../datasets/annotated/e_bike/rtdetr_result.json) · [修正预热后的三档逐图预览](../runs/e_bike/rtdetr_biasfix_20261118/evaluation/review.html)。

2026-11-18 已完成P2PNet思路独立点计数小训练：使用156/26图，冻结既有小车YOLO骨干，仅训练新点位置与姿态头，匈牙利一对一匹配、推理无NMS；未复制上游源码，不是官方VGG16复现。10轮共390次更新，训练约13.55秒；同一26图在固定conf0.25／0.5／0.75下两态数量全对分别0/26、0/26、2/26，未超过既有YOLO参照17/26。conf0.5时训练集也仅9/156图数量全对，存在大量重复点，说明本轮轻量点头尚未拟合，不能据此否定独立点计数路线。点距离匹配与旧框IoU指标分别解释；完整权重重载后骨干特征与缓存最大误差0，5项坐标、匹配和损失测试通过。原模型及118张导出保留；评估仍为已接触来源的研究对照。用户提出的多部件凸包及厚度约束尚未测试，凸包相交本身不构成多车计数依据。[结果与权重](../datasets/annotated/e_bike/point_counter_result.json) · [三档逐图点预览](../runs/e_bike/point_counter_micro_20261118/evaluation/review.html)。

2026-11-18 已完成多部件归属加凸包固定参数小实验：复用四部件best模型及既有姿态分类器，不重训；同一26图以conf0.25、宽松NMS IoU0.95获取候选，按同类预测点的固定原型归属、两个共享部件或一个共享部件加尺度加厚凸包证据去重，不用GT参与处理。宽松候选／仅归属／归属加凸包的重复预测分别320／271／223，凸包方案相对候选减少30.3%，两态数量全对仍仅1/26；同模型框NMS0.3为14/26、重复0，已有另一YOLO模型参照17/26。相对宽松候选，归属加凸包删101个预测并损失1个GT覆盖，框NMS删345个并损失5个GT覆盖；保守删除未带来计数净收益。重复按最大IoU≥0.5归属同GT的多余预测统计，另报车辆漏检，不能把误删当改善。模型候选的部件点不一致、背景伪部件仍是限制；这不是独立部件检测或可学习归属模型。已生成104张带点、加厚凸包、删除候选的四组预览与实拍/合成拼图，10项相关测试通过；原权重哈希及118张导出保持。[结果](../datasets/annotated/e_bike/part_hull_result.json) · [逐图对照](../runs/e_bike/part_hull_probe_20261118/review.html) · [拼图](../runs/e_bike/part_hull_probe_20261118/comparison_contact.jpg)。

2026-11-18 已按视觉研发、实验治理及历史/文件治理Skill完成电动车计数全景梳理与实验文件分类，未新增训练。当前17/26参照为单类别检测、姿态分类及NMS0.3组合；同一26图逐车审计为车辆FP5／FN7，9张数量失败中7张涉及定位、2张仅姿态错误，重复指标已为0。后续建议优先输入尺度/候选诊断、定向合成与现有组合微调，再做有训练集拟合证据的独立计数头。已建立[研究维护页](research/E_BIKE_COUNTING.md)和由source.yaml引用的experiment_inventory，分类26个runs目录中的50份模型检查点、2份Pose初始化、1份骨干状态及182份特征缓存，核对9份结果引用均存在、11项结构化路径/哈希对一致。原路径和历史记录保留，无文件迁移、删除或正式版本创建；盘点入口只读现有实验并刷新证据索引。

2026-11-18 已围绕17/26组合完成固定权重输入尺度与原生NMS前候选诊断，未重训。26图、102GT，conf0.25、NMS0.3、逐张矩形推理；640精确复现17/26且逐框坐标最大差0，960为12/26、1280为10/26、640全图加两块固定远端裁区为16/26；车辆FP/FN分别5/7、10/23、16/33、5/8。640的7个漏检中5个位置吻合候选分数低于0.25，2个有高分候选被NMS删除，已验证重叠保留框并出专图。固定裁区不读GT，原图比例、2px边界过滤和跨视图去重均留证；该窗口不是物理远端标定，其他裁区与尺度训练仍未验证。已保存原生NMS前全部解码框/分数、保留索引、四组104张预览及2张NMS图；8项坐标/归因测试通过。结果及实验索引已接入source.yaml，保持640及原权重、118张导出。[结果](../datasets/annotated/e_bike/input_scale_result.json) · [逐图预览与候选追溯](../runs/e_bike/input_scale_probe_20261118/review.html)。

2026-11-18 已完成640下置信度/NMS与有界微调对照：同一原生NMS前解码张量比较5档conf×5档硬NMS及3组Gaussian Soft-NMS，原17/26逐框逐姿态复现，28组均未超越。conf0.20/IoU0.3为14/26、车辆FP8/FN7；0.25/0.5为15/26、FP15/FN6、重复3；0.20/0.5找回原7例中的3个，但数量12/26、FP19/FN4。0.01/0.7可按IoU匹配找回全部7例，却FP421、重复174、数量1/26。随后同156/26、640、batch4，从17/26检测器继续最多40轮、lr0.0001、warmup_bias_lr0、min15/patience10、墙钟900秒，实际40轮约204.56秒未早停，分类器不变；末轮原参数15/26（实拍1/10、合成14/16），mAP自动选优权重14/26，二者找回原7例中3个且姿态正确，但新增4个GT漏检。保留原640/conf0.25/IoU0.3组合及原权重、118张导出；未新增训练数据或正式发布。已生成936张参数与训练预览、原7例定位/姿态矩阵、误检raw_id与跨模型GT变化审计，12项相关测试通过。[结果](../datasets/annotated/e_bike/threshold_nms_result.json) · [综合对照](../runs/e_bike/threshold_nms_probe_20261118/review.html)。

2026-11-18 已完成定向数据与一轮有界训练：研究配方focused_recipe复用4个实拍/Blender底图及24个车体视面，生成128张训练合成、32张整布局族留出（8族）、54张完整实拍放大裁片、48张训练来源GT框外硬负例；训练386图、早停val仍为旧26，新32在训练结束后才评估。研究可见率下限显式0.55，46个实例低于0.7，原制作默认0.7保持；四种车车遮挡、掩膜框、无目标裁区、原val字节、分组及跨split像素去重均已机器核对，无新增人工确认。初始化原17/26检测器，640/conf0.25/IoU0.3及原姿态分类器保持，40轮约419.30秒。原模型／末轮／mAP选优在旧26为17／16／18，新32为21／22／23；选优实拍3/10低于原4/10，旧合成15/16高于13/16，旧26车辆FP/FN为3/7（原5/7）、实拍FN7（原4）。合成有小幅收益，实拍退步，保留原17/26参照，未发布新正式版本。15项相关测试通过，数据与58图三模型对照均已输出，当前182张sample及118张exports保留；研究输入单独由source.yaml登记，不切换现行生产计划。[结果](../datasets/annotated/e_bike/focused_result.json) · [262张新增数据小样](../runs/e_bike/focused_inputs_20261118/review.html) · [前后对照](../runs/e_bike/focused_comparison_20261118/review.html)。

2026-11-18 电动车本轮有界研究完成交接：P2/P3共享骨干、重建检测头，8图45辆在增加相同更新预算后均可完整拟合；完整156训练40轮，各1560更新且LR/累积轨迹一致，旧26主参数数量P3=14、P2=12，有限置信度复核仍未超保留20/26。姿态分组NMS三seed均退为18/26，未采用。保留17→20、框姿态匹配16→18的三seed稳定开发集增益，实际权重SHA固定，单图研究入口重新加载后58图逐框逐姿态复现。资料核对补充BPJDet/JointDet部件归属、CrowdDet集合实例、DDQ方向，尚未迁移到小车。正式数据182小样/118导出、部署和原权重未替换。[研究交接](../datasets/annotated/e_bike/goal_research_result.json) · [P2/P3对照](../runs/e_bike/p2_detector_densefit_20261118/review.html)。

2026-11-18 电动车连续质量排序实验完成：仅原156训练来源挖2224候选（train1842/内部val382），冻结语义特征，几何/图像+几何两头各50轮回归IoU并训练无歧义同车排序。主混合排序旧26均20/18，图像头布局32退为20/20；事后纯质量也无全面净收益，未采用。GT辅助模式只诊断。候选层排序比例提高未转成整图收益，关键同车框仍排序错误；原控制58图逐框逐姿态复现。后续转向P2/P3细粒度检测头的小子集受控拟合，不继续穷举旧开发集混合系数；当前数据与部署不变。[预览](../runs/e_bike/quality_rank_20261118/review.html) · [回执](../datasets/annotated/e_bike/quality_rank_result.json)。

2026-11-18 电动车三种子与候选池复核完成：同1262裁片、固定split/起点、20轮，seed42/43/44均旧26数量20、框姿态匹配18，实拍6/5、布局32为21/21，成功图集合一致；只能支持已知开发集训练稳定性。NMS前原阈值语义候选最多匹配97/102、最终95/102；两处损失分别是同车较完整框被较差框抑制、相邻合框的多GT匹配歧义，不都等价于车辆消失。新增框匹配歧义诊断和反例测试，现有原17/16及新20/18不变；下一步固定语义筛选研究连续框质量排序，goal继续，正式数据与部署不变。[三种子预览](../runs/e_bike/verifier_seed_study_20261118/review.html) · [候选池复核](../datasets/annotated/e_bike/candidate_pool_result.json)。

2026-11-18 电动车持续研究补齐来源接触与完整性小训练对照：当前58图与直接156训练无像素/镜像重复，继承V3训练4000图与5张实拍原帧也无精确重复；48张合成评估均共享训练背景与车体来源，不能称来源隔离。局部候选拒绝组与等量纯背景重复对照均1233 train／252内部val、20轮，旧26数量/严格分别16/16与20/18，实拍3/3与6/5；硬拒绝少2FP却新增6个漏GT，未采用。固定原语义筛选再用质量组判姿态也未改善，严格降至17/26。原方案与语义验证器离线计算中位18.51/33.64ms，4方案58图两遍预测均与冻结结果一致；不代表Ubuntu/ROS耗时。下一步多种子复核20/18及设计明确来源分组，当前数据/部署保持。[对照预览](../runs/e_bike/completeness_comparison_20261118/review.html) · [实验回执](../datasets/annotated/e_bike/completeness_result.json)。

2026-11-18 已建立电动车持续研究goal，并完成首轮评价复核：20组冻结预测在IoU0.3/0.5/0.7回放，IoU0.5下原参照数量17/26但严格逐车框/姿态全对16/26，新验证器数量20/26但严格18/26，实拍严格4→5/10；计数仍有错漏抵消。实拍10图实际5原帧及镜像、3已声明父组，两方向均数量正确的原帧族1/5→2/5，不能当10个独立场景；来源别名、近邻及初始化接触矩阵仍待继续核查。原参数下贪心和最大有效匹配数一致，其他条件存在5条逐图敏感记录，历史结果保持。4项新测试通过；后续依次研究来源相关性、运行成本、候选完整性/同车去重及召回，正式数据与部署未切换。[复核结果](../datasets/annotated/e_bike/evaluation_audit_result.json) · [持续研究与阶段断点](research/E_BIKE_COUNTING.md#持续研究goal与评价复核)。

2026-11-18 已完成电动车三类候选验证器实验：冻结原17/26检测器，仅原156训练来源生成1046 train／216内部val裁片，完整160输入训练20轮约75.97秒。640/conf0.25/NMS0.3下，原参照及无过滤池控制17/26，仅非车过滤18/26，联合姿态20/26；实拍4→6/10，既有布局32保持21/32。逐车FP5→3、FN仍7，修正3个匹配姿态，58图无新增GT丢失；低阈值仍有大量局部/重复预测。原26及布局32是反复使用的开发诊断，内部val受初始化接触影响。原参照与新末轮权重均保留，仅研究候选；当前182小样、118导出与部署不变。[条件与带框预览](../runs/e_bike/candidate_verifier_20261118/review.html) · [持久回执](../datasets/annotated/e_bike/candidate_verifier_result.json)。

2026-11-18 已完成等训练量数据消融及同来源整帧采样补充：四组均保留旧156 train，补48旧图重采样控制／硬负例／完整实拍裁片／12族合成，各204槽位、固定20轮、seed42、batch4、lr0.0001，从同一17/26权重开始，分类器不变。实测每组76次更新且逐更新LR/epoch/梯度累积轨迹完全一致；末轮主比较旧26为17/17/17/16，实拍3/3/2/3，既有布局32为21/20/20/22。裁片组实拍FP/FN7/10，对照5/6；据此追加与48裁片父帧逐一对应的整帧重采样组，同样76次更新及相同轨迹，实拍4/10、FP/FN3/7，但旧26仅16/26、新32为21/32。原参照旧17/26、实拍4/10、FP/FN4/4，保留原组合。五组GT实例数762/556/648/680/664，单种子配比干预不作普遍数据类型排名；整帧组是事后设计的补充，原四组回执保留。GT框姿态诊断98/102及95/96，仅用于区分定位与裁片分类，不当检测成绩。已输出六组58图错误标色及权重/训练量回执，5项新测试通过，当前sample/exports及原权重保持。[综合结果](../datasets/annotated/e_bike/data_ablation_result.json) · [原四组](../runs/e_bike/data_ablation_20261118/review.html) · [含整帧补充的逐车对照](../runs/e_bike/real_sampling_followup_20261118/review.html)。

**禁行牌全量制作与核对。** 已按接受的小样扩为 Train 2500、Val 300、Test 300。Train 的完整／截断或遮挡／可辨杆底座／空场／普通杆为 1300／500／200／300／200；Val 为 140／60／25／45／30。实际使用 Train 344、Val 67 张底图或原帧，父组 15／4；Train 相机来源 1700（直接实拍 9）、Blender 底图 800，Val 相机来源 300（直接实拍 11）。两集均覆盖四视面与远近／截断／杆底座16格，分别有80／16张实景灯体遮挡，目标短边小于32像素的合成图为318／28张；70个原有框的输出出现随图保留。普通杆硬负200／30张为各集一张原图的裁区与色调派生。逐图解码、标签范围、独立可见掩膜框、遮挡与继承框、完整色调组的亮度／饱和度检查通过，集内及跨集相同RGB重复为零，背景父组跨集重叠为零。Val 曾缺少侧视小目标，已在固定300张内补齐并重核。当前计划登记、实际统计及训练来源索引均已切到全量；Test已生成并作受控回归；完整训练及Val/Test回传已完成，未创建正式数据版本。多数场地同源且共享牌具视频，评价范围为受控诊断。Windows数据预检通过。[当前制作入口](../datasets/scripts/sign/README.md#当前全量数据) · [全量审计](../datasets/annotated/sign/full_dataset_audit.json) · [带框总览](../datasets/annotated/sign/full_dataset_contact.jpg) · [训练预检](../datasets/annotated/sign/full_training_preflight.json) Test使用20张停车视频帧、1个父组，完整140／边缘截断60／杆底座25／空负75，无已确认普通信号灯杆和直接牌具正例。19条旧待验证资格的源像素接触检查通过并写回；初次新增的8张车体遮挡造成牌面内部背景窗口，已全部换成边缘截断，原照片及原掩膜完整。最终三集均覆盖16格，封闭遮挡背景窗口提示为零；三对集合的父组及相同RGB交叉均为零。训练仍仅用Train／Val选择，Test入口已启用。[三集预检](../datasets/annotated/sign/full_three_split_preflight.json) · [修订回执](../datasets/annotated/sign/test_occlusion_repair.json)。

**禁行牌共享底图扩源预检。** 2026-11-18 逐帧访问当前 585 张 train 合格底图（Blender 564、同视频实拍 19、照片 2）。582 张在本轮近地条带及底座跨度条件下找到布局，已编成独立的 [1,164 张扩源候选计划](../datasets/scripts/sign/config/background_expansion_plan.json)，该扩源计划尚未渲染。另 3 帧保留原资格并记录本轮搜索失败；三个物理父来源不能等同于 582 个独立场景。原 `background_source_inventory.json` 的 155 条未分配背景已完成逐图复核，现行状态见下方补证结果。[操作入口](../datasets/scripts/sign/README.md#共享底图扩源)

**禁行牌底图补证与分集。** 2026-11-18 原 inventory 的 155 条 `sign/unassigned/unknown` 已逐图写回。147 条为 `eligible`（139 张实拍、8 张旧 Blender），其中 10 张保留已有牌具完整框；147 条已全部分配为 Train 70、Val 77、Test 0。另 4 条主体牌具帧为 `blocked`，4 条落点不明的合成用途为 `unknown`，其完整无牌证据可用于负例。原有 585 张 Train 合格底图继续可用。新增覆盖的是 14 个此前未在禁行牌合格池出现的实拍父路径，多数拍于同一室内模型场地。按实际父视频、照片及场次记录覆盖，原 inventory 保留为历史快照。`bins004` 父视频跨任务已有 Val 接触，禁行牌本批已归 Val。上一轮分集小样为 Train 162 张（79 框、83 空负）、Val 31 张（18 框、14 空负）；使用 99 张底图、19 个不交叉父来源引用，保留 3 个旧牌框。历史 Val 照片与 Train 同父，该轮已排除。前景共享同一牌具视频，Val 为受控场景诊断。图签解码、可见掩膜框及继承框核验通过，跨集无相同 RGB 图。旧根目录 188 张已按清单清理，当前根清单为分集汇总。[审计](../datasets/annotated/sign/current_sample_audit.json) · [Train 小样](../datasets/samples/sign/train/manifest.json) · [Val 小样](../datasets/samples/sign/val/manifest.json) · [155 帧复核回执](../datasets/annotated/common/review/sign_background_155_20261118/recheck_receipt.json)

**禁行牌已接受的440张困难组合小样。** 本轮Train305张、236框、69空负；Val135张、140框、23空负，保留36个原有牌具框的输出出现。使用Train66／Val25张底图、15／4个父来源；每集合均覆盖四视面×小目标／中景／部分截断／杆底座截断16格。85组同布局四色调经亮度与饱和度实测核对；40张目标框短边小于32像素。各集合8张实景灯体遮挡和4张普通信号灯支杆硬负；这些硬负为两张不同分集原帧的色调派生。跨集无背景父来源交叉或相同RGB图像。上一轮193张生成图签已清理；以识别任务与有效监督为目标，不把接地及阴影真实感作为普遍生产门槛。真实识别效果尚未评估，未启动训练。[本轮审计](../datasets/annotated/sign/current_sample_audit.json) · [带框总览](../datasets/annotated/sign/current_sample_contact.jpg)


**楼名全量检测数据。** 已完成 train 3000 张／3495 框／601 空负、val 300 张／323 框／82 空负、test 300 张／334 框／73 空负。13 张实拍底图来自 4 个父视频，按父来源分集，选用 182 个 Blender 有纸／无纸物理模板。全量分支为实拍 41、原纸框替字 195、楼体＋实拍底图 2164、纯渲染 1200。三集分别覆盖全部 26 种楼体／纸条开关状态，train 的多楼部分有纸 1016 张、有楼无纸 598 张；低机位 0.14／0.20 m，左前单楼 70°、多楼 50°，透视与接地阴影沿用物理层。逐图解码、图签、最后可见纸面掩膜及框核对通过，3600 张无相同 RGB 重复，选中实拍底图父组无跨集重叠。当前计划登记、训练数据入口及实际清单已同步；val/test 保留既往开发接触并共享楼体资产，用于受控诊断；原远程训练和新视频微调情况见下方，未创建正式发布版本。[数据入口](../datasets/exports/building_name_paper/data.yaml) · [审计](../datasets/exports/building_name_paper/audit.json) · [Train 拼图](../datasets/exports/building_name_paper/train_contact_sheet.jpg) · [Val 拼图](../datasets/exports/building_name_paper/val_contact_sheet.jpg) · [Test 拼图](../datasets/exports/building_name_paper/test_contact_sheet.jpg) · [复跑入口](../datasets/scripts/building_name_paper/README.md#当前全量检测数据)

**楼名训练打包。** 当前 [训练包](../training_packages/building_name_paper/README.md)已包含三集 3600 张图签、同类 V3 初始权重、共享引擎及楼名专项诊断；150 轮上限，120 轮后允许 15 轮耐心早停，单次硬时限 10 小时。3600 条最终纸框与已知生成控制已绑定来源索引，显式声明两类 Blender 分支共享楼体资产。共享便携环境下的包内预检、代码指纹、权重加载和单图 CPU 有限前向通过；原远程训练已完成 135 轮，第 108 轮为 best；回传模型已核对，新增视频微调见下方。Test 不参与当前微调选模。[检查回执](../training_packages/building_name_paper/package_checks.json) · [配置与启动](TRAINING_WORKFLOW.md#楼名纸条训练包)

**楼名新视频独立微调。** 按用户指定，已在训练包内新增 [finetune_dataset](../training_packages/building_name_paper/finetune_dataset/manifest.json)，只消费三段新拍视频及其派生图，原训练图参与数为 0。Train 231 张／177 框／90 空负，Val 21 张／13 框／8 空负；原始抽帧共 72 张，其余为 Train 裁剪及 24 张双／三纸条人工多视图拼图。纸面四角视觉核对后用光流与单应性传播，逐框检查；不采用弱模型的预测框充当真值。Train 使用 buildingname1/2，Val 使用 buildingname，视频文件隔离但共享现场与拍摄 session，属于同场景诊断。已在包内从回传 best.pt 以低学习率完成 43 轮微调，第 33 轮为 best，保留完整 last 和精简权重；新 Val 的 TP／FP／FN 为 4／2／9→10／0／3。原 Test 的重叠额外框减少，但背景误检仍有回归，候选独立保留；数据和运行均与原全量 Train 分开。[数据制作](../datasets/scripts/building_name_paper/README.md#新视频独立微调) · [微调配置与进度](TRAINING_WORKFLOW.md#楼名新视频独立微调)

**楼名 Blender 光照与低机位。** 2026-11-18 已将 `Buildings` 保存为四相机场景，A/B/C 单楼和 ABC 全景各有一条 275 帧轨迹。相机离地 0.14–0.23 m，单楼覆盖左前 70°，三楼覆盖左前 50°，参考车载帧构图作近似，实际相机标定仍未知。A、B、C Collection 分别控制楼体显示及渲染，支持全部七种非空组合；纸条与文字可随楼体或单独开关。照明改为 World 环境光和 Cycles 漫反射反弹，纸面为粗糙漫反射。已认可的低机位 Blender 分支覆盖七种楼数组合及左前、正前、右前对照，当前小样直接复用这些图签，计划和掩膜保存在本任务已有的 annotated/fire 归属下。实拍、实拍底图替字与楼体＋实拍底图通过 build_current_sample.py 接到同一小样入口。新增楼体合成由道路标线消失点拟合地平线、按射线反投影接地，并用带楼／空场漫反射接收面比值产生阴影；焦距和机高保持近似声明。前景、地面光照层和纸条掩膜保存在 annotated/fire/building_name_paper_sample/hybrid，实际产量由 source.yaml#sample_manifest 定位。四条轨迹的 1,100 个整数帧均检查纸框在画面内，每条抽查 92 帧纸面遮挡。火焰独立使用同工程的 `Fire Buildings` 场景和 241 帧轨迹，共享楼宇几何。[Blender 分支证据](../datasets/annotated/fire/building_name_paper_sample/blender/manifest.json) · [轨迹、开关与入口](../datasets/scripts/building_name_paper/README.md#blender-光照轨迹和楼体开关)

本次替换前的生成小样已归档于 `Archive/technical/datasets/e_bike/sample_before_recheck_20261118/`、`Archive/technical/datasets/sign/sample_before_recheck_20261118/`、`Archive/technical/datasets/meter/sample_before_recheck_20261118/`、`Archive/technical/datasets/fire/sample_before_recheck_20261118/`，分别保留旧 36、40、3、18 张及原清单，供本次对照。以上为相应小样运行的记录；垃圾贴纸随后已完成本页所列三个 split 的完整导出，其他任务的实际进度按各自当前清单核对。

电动车全量 train 草案为 4,000 张，禁行牌 2,500 张 Train、300 张 Val 与 300 张 Test 已执行。早期逐图库快照另有尚未分配 split 的完整直接实拍正例：电动车 16、火焰 5、楼名 4、禁行牌 6、桶状态 17、人物 3 个帧版本。同一照片可属于多个任务。推进中的任务按父来源、既往接触和 `source.yaml` 的拟议用途确定 split，再用当前资格打包。桶状态与仪表的合成入口使用对应的 Blender 前景。[各任务入口](../datasets/scripts/README.md#恢复生产入口) · [组合轴覆盖](DATASET_DESIGN.md#训练目标与组合轴覆盖生成) · [全量预算依据](FULL_TRAIN_PLAN_DESIGN.md)

**2026-11-17 垃圾桶优先批次。** 逐图库按原帧记录 `bins004` 的 20 张 val 四桶图，共 80 个开闭框；这 20 帧的四桶开闭状态组合相同，只是视角有所变化，组合数与父视频组数各为 1，不能将帧数或实例框数当作组合多样性。另将已完整审查的 2 张 1920×1080 四桶图按 `source.yaml` 对应父视频分到 train。当时的[train 直接实拍清单](../datasets/annotated/waste_bin_state/full_export_history/20261117T095913236346Z/train/manifest.json)导出 12 张（2 张正例、10 张整帧无桶负例，8 个桶框），[当时的 val 清单](../datasets/annotated/waste_bin_state/full_export_history/20261117T095913236346Z/val/manifest.json)导出 20 张正例（80 个桶框）。图签逐项存在、导出图片哈希与入库原图一致，train 与 val 无同帧或同背景父组。数据库另有 2 张与 train 同属 `full_scene` 父视频的 val 空负资格，本次 val 导出未选它们；此前 22 张 val 的生成副本保留在 `tmp/waste_bin_state_val_22_before_parent_filter_20261117/` 供追溯。旧选源中的 10 张 train 空背景已通过实际合成入口的标签、父图和接触掩膜检查；Blender 前景、掩膜与标签已按下方批次补齐。两张有桶但红桶盖不可判的旧帧仍为部分标签，不计完整正例。

**桶状态实拍底图补录。** 轨道帧入库前，逐图库有 108 个合格帧身份（train 22、val 27、test 59），合成选源入口已读取三个 split 的全部合格底图。机器先用已有完整标签和源图像素脚点为 41 张写入资格，再参照底图中已有的桶框估算相对尺寸，放大试放偏小的前景，补入 14 张（val 6、test 8）。本轮剩余 13 张中，5 张已有脚点但标签或格式未齐，5 张首次试放未找到完整脚点，3 张放大后未找到完整脚点。机器证据满足用途即可生效。借用 train 前景试放所得的空间证据可用于 val 和 test 底图入库，正式生成时核对前景、底图与输出的 split 配对。`pool_159.png` 等 22 张以桶为画面主体的帧，桶任务常规合成用途为 `blocked`，桶开闭直接正例资格仍为 `eligible`。[首批机器写入回执](../datasets/annotated/common/review/bin_probe_machine_qualification_20261117/receipt.json) · [调整尺度写入回执](../datasets/annotated/common/review/bin_probe_scaled_qualification_20261117/receipt.json) · [剩余试放清单](../datasets/annotated/waste_bin_state/pool_batch_probe_remaining_20261117/manifest.json) · [主体修订回执](../datasets/annotated/common/review/bin_primary_subject_exclusion_20261117/receipt.json)

**Arena 八路相机通用底图。** 从统一工程 `datasets/assets/blender/aiaec_dataset.blend` 的 `Arena`、`Arena Indoor Soft Light` 和 `Arena Detailed` 各沿八条轨道渲染，每路 24 帧，每种场景 192 张，共 576 张，保存在 `datasets/raw/shared/backgrounds/rendered/aiaec_eight_routes*_20261117/`。每张均按场景物件记录写入八任务的 train 整帧 `empty_negative` 资格。三种场景共用同一 Blender 父来源。利用保存的相机矩阵和约 4.2 米见方的地面投影，564 张形成可用接地候选，分别取得人物、电动车、禁行牌和桶状态的 train `composite_background` 资格。其余 12 张在这四项任务中仅有空负资格。楼名纸条和仪表均复用整帧空场证据，各有 576 张取得 train `composite_background` 资格。任务计划按实际构图核对放置。这批通用空场对火焰和垃圾贴纸当前用于空负例。表面贴放分别使用下方的可贴立面及可贴桶面批次。[Arena 帧清单](../datasets/annotated/common/sources/aiaec_eight_routes_20261117/manifest.json) · [Indoor Soft Light 帧清单](../datasets/annotated/common/sources/aiaec_eight_routes_indoor_20261117/manifest.json) · [Detailed 帧清单](../datasets/annotated/common/sources/aiaec_eight_routes_detailed_20261117/manifest.json) · [入库入口](../datasets/scripts/common/ingest_shared_route_backgrounds.py)

**火焰当前接线。** 当前 train 3,000、val 300、test 300，共 3,600 对图签、5,704 个指定贴纸框。已将 train 450、val/test 各 30 张普通空负替换为硬负例，本阶段随后补齐600张同框干扰正例，并将Val/Test各90张既有合成正例改为小目标，未涉及的文件保持原字节。train 红灯、红橙桶、红橙锥桶各 150 张；val/test 各复核 15 帧真实红桶图，每帧两档温和亮度派生，来自两个不同父视频。当前四种生成方式为实拍及亮度派生 115、实拍墙面合成 2,057、完整 Blender 1,218、楼体＋实拍场景 210 张，覆盖 39 个父来源，跨分集父来源和相同图片重叠为零。退出的 510 组旧图签及失效预览、掩膜、混合图层已按清单清理，保留原件、原生成计划和持久回执；当前完整配额、逐图记录、空标签、统计与训练入口已同步。原墙面配准、楼角 PnP 近似相机、Cycles 接地阴影及线性光合成继续沿用，整楼未完整观察时保持 null。首轮方案已落地，保持3600图和5704框；Val/Test各补45个8–16像素和135个16–32像素目标，保留较大目标与实拍。基础包制作阶段仅做生成及打包一致性处理；其后远程基线已完成135轮，下载的best与Val/Test结果保留在原训练包内，当前按下述专项接续。[当前集合](../datasets/exports/fire/manifest.json) · [空负替换阶段核对](../datasets/annotated/fire/current_full_review/audit.json) · [替换入口](../datasets/scripts/fire/README.md#空负例批量替换)。

**火焰10轮微调及对比已完成。** 使用1224张精简Train、batch 4，在包内完成固定10轮，约572秒，Val选出第6轮best；[结果入口](../training_packages/fire_sticker/finetune_result.json)关联权重、哈希与[对比报告](../training_packages/fire_sticker/runs/fire_sticker/v4_compact_finetune_10ep_batch4_20261118_2045_r1/comparison/comparison.md)。同一640输入、conf=0.25、NMS IoU=0.7、匹配IoU=0.5下，Test红桶误检框8→3、贴纸475/480→480/480、总误检9→4；Val贴纸474/480→480/480但误检0→2（窗内橙色干扰）。未参与本轮微调的2400张旧Train图，两版均3760/3760检出且零误检；该回归覆盖三种合成方式，原直接实拍Train已全部回放。原候选池和Val/Test图签保留，父源划分不变，现有集合是开发已接触的诊断，非独立实车泛化证据。已有预测按0.5过滤可令本轮Val/Test红桶误检归零，但Test会漏2个真小贴纸，且Val/Test各留1个其他误检；未重跑推理或修改默认阈值。启动时补齐包内来源证据引用，评估时修复Windows字体缓存重复写入；失败回执均保留，未额外训练。

**火焰硬负例开发小样。** 独立 `Fire Hard Negatives` 场景已生成 12 张颜色及有无贴纸对照：6 张正例、3 张红橙硬负例、3 张其他颜色控制负例；红／绿灯、红／蓝桶、橙／灰锥桶均保持成对机位和光照。颜色继续作为有效线索，目标保留指定图案，干扰物只记上下文。已重解码 ID 核对图签与逐楼归属，并视觉核对贴墙、比例和接地。另视觉复核两张实拍，确认其中一张红桶帧为 `fire/hard_negative` 的 val 素材，另一张保留普通负例身份。该开发小样保留；当前集合的空负批量替换已完成，未训练或评估误报率。[小样与参数](../datasets/scripts/fire/config/hard_negative_preview.json) · [独立核对](../datasets/annotated/fire/hard_negative_review/audit.json) · [颜色拼图](../datasets/annotated/fire/hard_negative_review/contact_sheet.jpg)。

**垃圾贴纸 Blender 桶面底图。** `Arena Waste Sticker Preview` 沿八条轨道各取 3 帧，得到 24 张无物品贴纸的四色桶面图。逐图库已写入 96 个可贴正面及 24 张 train `composite_background` 资格。现用选源和合成入口已逐帧试贴，持久生产回执登记实际产出 24 张图和 24 份配对 YOLO 标签。该批为试跑，当前可从回执核对产量。[桶面源清单](../datasets/annotated/waste_bin/waste_sticker_arena_surfaces_20261117/manifest.json) · [试跑回执库](../datasets/annotated/media_index/production_state.sqlite3) · [合成入口](../datasets/scripts/waste_sticker/generate_sticker_real_photo_v4.py)

**共享池索引。** 共享池清单的 `records` 保留 200 条基批次记录，`extensions` 纳入三份 Arena 渲染清单、C 楼立面清单和垃圾贴纸桶面清单。合并这五批后，候选发现器读到 824 条共享池图像路径、823 个帧身份，并核对 `raw/` 的 2,543 条图像路径全部已注册。机器审查关联表沿用原四份 JSONL 重建，`check` 已通过。各任务的用途资格以逐图库为准。[共享池清单](../datasets/annotated/common/sources/background_pool_manifest.json) · [候选发现器](../datasets/scripts/common/discover_raw_background_candidates.py) · [机器审查关联表](../datasets/annotated/common/sources/background_pool_machine_review_index.json)

桶状态当前共有 672 张合格底图（train 586、val 27、test 59）。实际合成入口通过 `qualified_backgrounds` 和 `eligible_ground_backgrounds` 读取了新增的全部 564 张，并用其中一张产出 train 合成试点及配对标签，清单状态为 `complete_candidates`。人物 V4 已收官，本批登记来源与用途资格。[轨道底图合成试点](../datasets/annotated/waste_bin_state/sample_history_20261117/route_background_composite_probe/manifest.json)

桶状态代表缩影共 42 张，含直接实拍正例 5、实拍空负 5、实拍底图合成 16、Blender 轨道底图合成 16。合成前景统一使用 `Waste Bins Photo` 的后垂开盖。198 个可行帧组成 96 个互补候选对，贝叶斯按底图类型和新增桶数分层选 16 对。两类底图中新增 1、2、3、4 桶各 4 张，80 个新增框中开、闭各 40 个。实拍底图另保留一个黑桶开盖框。32 张合成均为 train 来源，正视 31 张、斜侧 1 张，邻桶有重叠 15 张。直接实拍保留原 split，成套小样共有 37 张 train、2 张 val、3 张 test。小样根只保存 `images/`、`labels/`、`manifest.json`。[小样清单](../datasets/samples/waste_bin_state/manifest.json) · [分配审计](../datasets/annotated/waste_bin_state/sample_composite_allocation.json) · [图签总览](../datasets/annotated/waste_bin_state/sample_overview.png) · [旧小样迁移和清理](../datasets/annotated/waste_bin_state/sample_history_20261117/current_sample_predecessors.json)

**垃圾桶开闭当前阶段：真实桶主导的 10 轮微调及前后对比完成，指定红桶漏检尚未解决。** [本次运行](../runs/waste_bin_state/v4_real_priority_finetune_10ep_batch4_20261118/metadata/status.json)从原始远程 best 初始化，batch 4，实际参数为 640 输入、偏置预热学习率 0，已完成全部 10 轮，Val 选优为第 1 轮。[同条件比较](../runs/waste_bin_state/v4_real_priority_comparison640_batch4_20261118/bin_state_comparison.json)包含原始、best 和第 10 轮 last；[指定图诊断](../runs/waste_bin_state/v4_real_priority_finetune_10ep_batch4_20261118/diagnostics/f002017/comparison.json)显示 640 下红桶仍漏，960 下 last 检红却把绿、蓝开盖误判闭盖。当前图签由真实四色桶补拍、单／双／四桶真实前景合成与旧 Train 正负回放构成，规模、来源和逐色开闭配额读取[当前导出](../datasets/exports/waste_bin_state/manifest.json)、[微调计划](../datasets/scripts/waste_bin_state/config/reshoot_finetune_plan.json)与[核对回执](../datasets/annotated/waste_bin_state/reshoot_finetune/verification.json)。原 Val/Test 图像及标签字节保持一致；补拍同一 session 全部归 Train。单桶仅放大同色开闭均有证据的机位，多桶使用已观察的兼容正面／背面组；全部按完整可见桶及附属盖板标框。邻桶布局按实际 mask 交集另记遮挡，不能把要求交叠的布局数当成有效遮挡数。现有真实低机位仅灰桶闭盖，没有成对低机位开盖和高俯视，侧面裁边照片仍只作原图训练，不补造新视角。[视角观察](../datasets/annotated/waste_bin_state/reshoot/view_observations.json)与[代表图](../datasets/annotated/waste_bin_state/reshoot_finetune/overview.jpg)记录这一范围；贴图接地感及真实视角覆盖仍有限。

[当前微调配置](../lab/features/waste_bin/training/config/waste_bin_state_reshoot_finetune_v4.yaml)统一 640 输入，显式修正偏置预热学习率为 0，本次 batch 4、10 轮，从原始远程 best 初始化新优化器；运行状态、保存权重与实际参数均已核对。上一轮 [960 输入运行](../runs/waste_bin_state/v4_reshoot_finetune_10ep_batch4_20261118/metadata/status.json)已完成 10 轮，[统一 640 的比较](../runs/waste_bin_state/v4_reshoot_comparison640_batch4_20261118/bin_state_comparison.json)未改善整体回归表现；本次没有续训上一轮候选，也没有覆盖原始模型。旧微调图签已完整保存至[替换备份](../datasets/annotated/waste_bin_state/full_export_history/20261118T115303522390Z/publication_backup.json)，旧配方及源回执见[生成输入快照](../datasets/annotated/waste_bin_state/full_export_history/20261118T114233064429Z/finetune_inputs.json)。原完整基线及已执行输入保留在原训练包与[基线导出历史](../datasets/annotated/waste_bin_state/full_export_history/20261118T095958167975Z/manifest.json)。

**垃圾桶开闭初始全量交付（微调前基线）。** 正式导出共 3,939 张，train 3,500、val 186、test 253。Train 含实拍正例 4、实拍空负 24、Blender 轨道空负 276、实拍底图合成 800、Blender 轨道底图合成 2,396。成图 0／1／2／3／4 桶分别为 300／1,600／526／542／532 张，四色各自开闭等量，保留 10 个实拍黑桶开盖框。贝叶斯按底图类型与新增桶数在可行互补对内分配，1,598 个互补对的合成底图和外观参数分别固定。实际前景保持完全后折开盖，图签覆盖可见整桶。全图解码、图签配对、开闭框和 SHA-256 去重均通过。[基线训练包数据入口](../training_packages/waste_bin_state/dataset/data.yaml) · [基线全量清单](../datasets/annotated/waste_bin_state/full_export_history/20261118T095958167975Z/manifest.json) · [Train 分配回执](../datasets/annotated/waste_bin_state/full_plans/train_composite_allocation.json)

Val 含 33 张直接实拍正例、25 张实拍空负、128 张实拍底图合成。Val 开盖 107、闭盖 349，保留底图原桶标签且存在闭盖偏多。Test 含 9 张直接实拍正例、52 张实拍空负、192 张实拍底图合成，开盖 213、闭盖 202。两个评估集合各剔除 2 张与 train 同父的 `full_scene` 空负。实拍父源分属各 split，两套合成评估前景共用 `Waste Bins Photo` 桶资产，各自独立渲染。22 张 train 实拍底图来自两个父视频，564 张轨道候选共享一个 Blender 母场景。[分集与归档](../datasets/exports/waste_bin_state/manifest.json)

桶开闭便携包已在 `training_packages/waste_bin_state` 完成构建与离线检查，含全部三集图签、通用 COCO YOLOv8n 初始权重、120 轮配置、Test 与专项评估入口。代码和权重摘要、3939 张逐桶元数据绑定、便携环境预检均通过；本次未启动训练或模型推理。颜色用于后续分组诊断，不参与训练损失或选模。[训练包说明](TRAINING_WORKFLOW.md#垃圾桶开闭训练包)

组合轴参考表登记了 19 个源渲染状态，显示的差额限于该表登记的计划。桶开闭 train 的 3,500 张目标已由正式导出清单核实。[参考计划覆盖](DATASET_DESIGN.md#训练目标与组合轴覆盖生成)

桶状态后续批次读取现行逐图用途资格。缺少某个底图类型或桶数组合时，优先把该组未用配额转给同桶数的可行互补对，并登记实际配比与剩余缺额。正式打包按本批图签与实际来源核算总数。尺度有实测证据时用于细化落点，其余使用并记录有依据的近似。`eligible + unassigned` 按父来源和已有接触安排本批 split，已分集且可用的组合照常生成。[全量编制入口](../datasets/scripts/waste_bin_state/plan_full_train.py)

逐图库当前有 654 张已审无桶且具备接地资格的桶状态底图。584 张为 train（含 564 张 Blender 轨道帧），val 19 张，test 51 张。共享候选清单按 `frame_id`、资格记录 ID、父组和 split 提供来源；接收任务按本任务标签与实际落点使用。这 564 张轨道帧与按任务核对的 Blender 底图快照逐图一致。[跨任务接地候选](../datasets/annotated/waste_bin_state/ground_reuse_manifest.json)

本轮合成流程简化聚焦垃圾桶底图入库、计划和生产。车牌沿用省赛模型，垃圾分类贴纸的批次进度见下方。

**垃圾分类贴纸实拍补录。** 车载视频 `V_20261112_152304.mp4` 的 002011–002052 共 42 帧已按 `waste_sticker` 逐图审查：31 帧为完整的当帧贴纸标签、11 帧因边缘截断或模糊保留部分标签，0 帧确认无贴。四张实物贴纸始终是油漆、果核、剩饭、纽扣电池这一套；入画数量变化只是可见子集和视角变化，**物理图案组合与父视频各只有 1 个**。`002016` 的四框经原图及参考图案复核后获得 train `direct_target` 资格，已通过逐图库正式打包到 `datasets/samples/waste_sticker/`；其余 30 张完整视频帧和 `P22_TRASH.png` 完整回归图列为 `unassigned`，11 张部分标签帧不计直接正例配额。`P22_TRASH.png` 与 002027 的全帧 SSIM 约 0.990，现有捕获身份也关联二者，因此这些帧不能当作独立采集或新图案组合。[旧贴纸 train 生产回执](../datasets/annotated/media_index/production_state.sqlite3)的 192 张均为实拍桶面合成图；另按[打包声明](../datasets/scripts/waste_sticker/config/train_193_package_spec.json)组成 193 张训练候选 ZIP，含 196 个标签框（192 个合成、4 个直接实拍）。ZIP 完整性和全部图片解码通过；合成图 8 张总览图逐图查看，贴纸落在红桶面且框在贴纸区域，未见明显错位。类别与图案按生成清单对应，尚无独立模型识别验证；两张底图及红桶单一载体不能视为场景多样性。

**垃圾桶与贴纸的 Blender 出图复核。** 桶盖开闭从 `Waste Bins Photo` 重渲 19 张，其中 18 张含桶、1 张为空场。开盖统一采用盖板垂在桶后，12 个开盖框、12 个闭盖框及 19 份图签均通过掩膜审计。旧竖盖渲染与其派生合成已退役，旧清单、回执和失效预览清单收在历史记录。贴纸旧合成诊断批次完成了 26 个状态，104 个贴纸框与掩膜及图案编号逐项一致，覆盖 24 类。抽看时有小尺寸、低对比度的图案，识别效果按后续实际推理记录。[桶开闭前景清单](../datasets/annotated/waste_bin_state/foreground/manifest.json) · [旧前景回执](../datasets/annotated/waste_bin_state/v4_upright_render_superseded_20261117.json) · [贴纸合成诊断回执库](../datasets/annotated/media_index/production_state.sqlite3)

### 垃圾贴纸完整数据交付

当前 20 张小样按 `datasets/scripts/waste_sticker/config/source.yaml#sample_manifest` 查询，覆盖 24 类、36 个框；旧 26 状态 Blender 与 1 帧实拍计划保留为历史参考配方，退出现行覆盖统计。正式分集由三个 `full_<split>_plan.json` 和对应导出清单维护，小样不追加到正式产量。

<a id="垃圾贴纸生产计划"></a>

2026-11-18 已按[垃圾贴纸数据设计](DATASET_DESIGN.md#垃圾贴纸数据设计)生成 2,280 张，每图 1–4 个可见桶、每桶至多一贴。Train 1,800 张、4,050 框，含 180 张无贴负例和单至四贴各 405 张；Val、Test 各 240 张、540 框，分别含 24 张无贴负例和每档 54 张正例。训练各类 163–179 个实例，验证各类 22–24 个，测试各类 21–24 个。

| 来源 | Train | Val | Test |
| --- | ---: | ---: | ---: |
| `Waste Stickers` 原生场景 | 990 | 160 | 200 |
| Blender 桶面透视叠贴 | 270 | 0 | 0 |
| Blender 整桶叠实拍场地 | 450 | 0 | 40 |
| 实拍桶面透视叠贴 | 90 | 80 | 0 |

2,280 对图片与 YOLO 标签均已解码并核对实例框及计数，未发现相同 RGB 图像。训练入口按 `fixed_validation` 实读 Train／Val，通过完整监督和类别覆盖预检；Test 另行全量检查。三个 split 的实际产量已写入生产守卫回执。[数据入口](../datasets/exports/waste_sticker/data.yaml) · [来源索引](../datasets/exports/waste_sticker/workspace_sources.json) · [验收结果](../datasets/exports/waste_sticker/audit.json) · [执行命令](../datasets/scripts/README.md#垃圾贴纸完整数据集)

远程完整训练已完成 150 轮，Val 在第 94 轮选出最佳权重，用户回传了 [best 与评估结果](../training_packages/waste_sticker/waste_sticker_best_Val_Test/audit.json)。原 Test 的实拍场地混合分支出现白色道路标线误检，已按[专项微调计划](../datasets/scripts/waste_sticker/config/hard_negative_finetune_plan.json)制作正负对照、白卡和原 Train 正例回放。实际图签数量、类别与父来源隔离见[专项审计](../datasets/annotated/waste_bin/waste_sticker_hard_negative/audit.json)，当前输入见[专项数据入口](../datasets/annotated/waste_bin/waste_sticker_hard_negative/data.yaml)。原 Test 保留为已知回归，新增挑战仍共享留出实拍父组及固定资产。本次微调已按用户要求在完整第 10 轮检查点保存后停止，原 20 轮配置快照保留，详见[停止及精简回执](../runs/waste_sticker/v4_white_stripe_finetune_20261118/metadata/user_epoch_stop.json)。[Test 对比](../runs/waste_sticker/v4_white_stripe_finetune_20261118/comparison/test.json)显示总体误检下降，但仍有白色部件误检及原回归集新增漏检；[指定失败图](../runs/waste_sticker/v4_white_stripe_finetune_20261118/comparison/user_selected_failures.json)已核对，数据仍缺真实白色杂物与同尺度小药片对照。上一轮权重停止沿用。用户随后授权重新从远程原始 best 微调，当前[保召回计划](../datasets/scripts/waste_sticker/config/recall_preserving_finetune_plan.json)与[数据审计](../datasets/annotated/waste_bin/waste_sticker_recall_preserving/recall_preservation_audit.json)保留完整旧 Train，补真实白色干扰和全类小尺寸纸片对照；[新运行](../runs/waste_sticker/v4_recall_preserving_from_remote_20261118/metadata/train_status.json)已完成 10 轮并选出第 7 轮权重；[结果回执](../runs/waste_sticker/v4_recall_preserving_from_remote_20261118/comparison/workflow_status.json)记录误报减少但 Test 漏检增加，未采用为原始 best 的替代。详见[训练交接](TRAINING_WORKFLOW.md#垃圾贴纸白色背景硬负例微调)。

原生渲染使用已存车载相机，Train 中 18 张使用 Cycles 8 采样，其余原生图使用经对照核验的 EEVEE 128 采样，逐图记录引擎。整桶混合复用 32 个训练物理模板和 24 个测试物理模板的 RGBA、阴影与掩膜，按已审地面进行缩放、平移和轻微调色。实拍父视频按 split 留出，固定图案及基础 Blender 工程共享，评价范围为当前来源与共享资产的受控诊断。

Val 的 20 张桶面已修正原先部分下沿落到地面的几何，并保留原决策及新观察回执。旧 192 张合成导出、193 张候选包和过期小样已按授权清理，历史事实从原生产回执追溯。P22 同源实拍仍作为单独回归来源，不计入这 2,280 张。[桶面修正回执](../datasets/annotated/waste_bin/waste_sticker_current_surface_review/intake_receipt.json)

逐图库的 61 个合格合成底图帧身份为 Train 28、Val 20、Test 13，`surface` 角色共 46 个帧身份，整帧无贴负例资格共 644 个。此批实际使用 24 张 Blender 桶面、2 张 Train 实拍桶面、20 张 Val 实拍桶面、1 张 Train 低机位场地和 3 张 Test 宽地面场地。Test 的其他 10 张同父候选保留，当前模板没有使用。图签和来源按用途分别计数。[逐图库](../datasets/annotated/common/sources/frame_environment.sqlite3) · [分配回执](../datasets/annotated/waste_bin/whole_bin_ground_reuse_20261117/production_split_assignment_receipt.json)

## 任务分工与阶段边界

数据制作的全局、任务和批次职责见[视觉研发架构](VISION_ARCHITECTURE.md#数据制作的全局与局部)。

## 从原始采集到导出

按[数据集制作 Skill](../.agents/skills/dataset-production-loop/SKILL.md#执行当前批次)推进。具体命令查[逐图入库](../datasets/scripts/README.md#逐图入库与资格查询)、[任务脚本入口](../datasets/scripts/README.md#恢复生产入口)和[训练入口](TRAINING_WORKFLOW.md#v4-任务包构建与启动)。[项目执行约定](../AGENTS.md#数据设计与实现)维护正式版本发布权。

### 查询和分配候选

候选查询、显式分配及返回记录的用法统一见[逐图入库与资格查询](../datasets/scripts/README.md#逐图入库与资格查询)。

## 标注与准入状态层

字段定义和计数见[标注状态层](ID_AND_COUNT_POLICY.md#标注状态层)，各任务的用途判定见[标签、空间与用途资格](DATASET_DESIGN.md#标签空间与用途资格)。

### 机器判定与写回

首次录入读取并解码一次，复用既有观测和标签，判定主体任务、各任务的标签范围及适用的大致放置区域，并在同次提交写出各用途资格。机器证据满足本用途即为 `eligible`，人工署名只记录实际人工确认。选源和计划读取当前资格。合成核对文件身份、标签与坐标配对，以及本次实际脚点、贴放和遮挡，复用入库视觉判断。[审查入口](../datasets/scripts/README.md#背景审查入口)

缺项按 `(frame_id, task, role, split)` 定位，在入库端用已有标签、同源帧、原图或模型定向补判并重算资格。只有引用已被替换的观测等受影响用途需要更新。查询的 `evidence_superseded` 指出需要换用的证据版本。其他合格用途继续出样和出包。[资格查询实现](../datasets/scripts/common/frame_environment.py#L281)

空间录入只要求有依据的大致可用区域及原图坐标，不以精确相机角度、透视标定或凸包作为统一准入门槛。`candidate` 表示可尝试布局，实际目标的尺寸、脚点、遮挡和标签变换仍由生成入口逐次核对；无法确认大致区域时才保留空间 `unknown`。空间 `unknown` 不抹去已证实的整帧空负监督。

`regression` 记录用于追踪来源接触。按当前用途、split 和配对关系选源。组外评估按实际留出的父组或场次计数。逐字节相同的图片遵守同任务跨 split 的去重规则。 [来源配对与复用](DATASET_SOURCE_PAIRING.md#7-跨任务复用与相同图片规则)

## 八任务接入与未受管路径

按[任务脚本入口](../datasets/scripts/README.md#恢复生产入口)运行，批次回执记录实际输入、调用和产物。

## 现行入口限制与处理

| 实际触发位置 | 当前影响 | 最小处理 |
| --- | --- | --- |
| [生产守卫](../lab/training_common/production_guard.py) | 生产以实际输入、输出占用和持久回执为准。查询缓存可附加来源索引，缺失或过期时仍可生成并记录回执。 | 修正实际缺失的输入或输出冲突；重跑使用入口已有的指定批次替换方式。 |
| [配对 checker](../lab/scripts/contracts/source_pairing.py#L99)与[分配器](../lab/training_common/bayesian_allocation.py#L217) | checker 检查配对声明；结构有效的 `stale` 条目可通过，实际计划引用须按受影响的冲突核对。分配器拒绝违反本批预算、上下界或容量的计划。全量 train 草案的父组份额用于该草案。 | 修正冲突引用或显式登记同源用途，编译相关 split；分支容量不足时在本次可行组合内调配。 |
| 电动车[合成](../datasets/scripts/e_bike/compose_v4_grounded_train.py#L451)与禁行牌[合成](../datasets/scripts/sign/compose_sign_v4_measured.py#L135) | 当前生成入口可继承范围内已有目标的完整框，并核对新增目标的遮挡。现行禁行牌分集小样有 193 张。旧 inventory 的 155 条背景已写回为 147 合格、4 主体牌具受阻、4 落点未知。 | 在入库端复用已有标签，按当前主体、放置及变换证据定向补判旧记录。合格底图中的非主体目标随合成更新标签，空场分支按负例证据独立选材。 |
| [楼名全量生成](../datasets/scripts/building_name_paper/build_full_dataset.py)与[审计](../datasets/exports/building_name_paper/audit.json) | 当前三集计划覆盖完整实拍、原纸框替字、纯渲染和楼体实拍合成，预算读取 full_recipe.json。 | 按父组记录来源，按最后可见纸面掩膜核对框；变更资格或来源时重编受影响状态。 |
| 桶状态[合成](../datasets/scripts/waste_bin_state/generate_waste_v4_grouped.py#L220)与仪表[合成](../datasets/scripts/meter/compose_meter_real_background_v4.py#L111) | 桶状态编制器读取逐图库当前合格底图，合成输入包含可匹配同 split 的 Blender 图像、掩膜与标签。仪表 36 状态限旧清单的三张候选及固定投影，需要 RGBA 前景与八点标注。 | 桶状态已导出完整直接实拍并用合格底图通过单桶和四桶小样生成，继续按当期前景和底图重编计划。仪表按本任务当前配方继续推进。 |
| 垃圾贴纸[完整汇总入口](../datasets/scripts/waste_sticker/assemble_full_dataset.py)与[来源选取](../datasets/scripts/waste_sticker/source_inputs.py) | 四路输入已汇总为 Train 1,800、Val 240、Test 240 张，5,130 个框，审计通过。Val 使用 20 张实拍桶面，Test 整桶混合使用 3 张实拍场地。 | 训练读取现行 `data.yaml` 和逐图来源，评估按共享图案、Blender 基础资产及留出实拍父组说明范围。 |
| [来源审计](../lab/scripts/data/v4_source_gate.py#L674)与[近似帧审查](../datasets/scripts/common/review/judge.py#L361) | 调用这些审计的入口按各自配置处理诊断接触、时间邻近和相似度。 | 逐图库按标签和用途证据计算资格，相似度用于抽样、去重及复用记录。 |
| [公共直接实拍打包](../datasets/scripts/common/package_direct_real.py#L118) | 打包器从逐图库读 `eligible` 真实相机帧。当前可准备楼名 train 2 正、3 负，以及仪表 train 3 个整帧空负例。 | 随图保存捕获身份和标签范围。楼名检测用完整框，OCR 按实际转录。 |
| [训练数据暂存](../lab/training_common/engine/data.py#L329)与[训练入口](../lab/training_common/engine/runner.py#L239) | V4 暂存核对任务、split、标签和来源证据。当前标准训练入口按运行配置指定的验证集选优，历史运行的权重及报告按运行时配置理解。 | 先用正式数据入口和来源索引组装训练输入，再按[训练工作流](TRAINING_WORKFLOW.md#人物-v4-工作空间入口)核对、启动或恢复。 |

公共直接实拍打包器读取完整正例或完整空负例资格，优先使用任务工作副本，其次是共享相机副本。按捕获身份去重，保存 `frame_id`、资格修订、标签和来源出处。人物计划读取[背景分发快照](../datasets/scripts/person/config/background_dispatch_environment.json)。资格缺项在入库端补判，或按已具备资格的素材重编受影响的状态。 [直接实拍打包](../datasets/scripts/common/package_direct_real.py)

2026-11-16 电动车 train 首次回填得到 4 条 `empty_negative` 与 0 条 `composite_background`。随后用 `near_floor_local_support_v2` 定向补判，新增 12 条合成底图资格，其中人物 val 5、test 4，电动车 train 2，禁行牌 train 1。两张人物 test 图在原始 640×480 坐标中重算。人物 train 原 12 条继续读取原资格。电动车原 4 条负例仍可查询，这次补判只修订了选定帧的合成资格。[查询与补判入口](../datasets/scripts/README.md#背景审查入口)

2026-11-17 对旧共享背景批次完成一次独立的批量空间观测写库：该批 200 个池记录对应 199 个帧身份（两条 `007045` 路径是同帧别名）。96 条按线和纹理求得地面四边形，104 条使用近处地面先验；全部帧有局部无米制 XYZ 投影及高度参考。186 个池记录找到了类似地面的接触像素，14 个只有空间模型。抽查发现部分楼模和人物表面被接触算法误认为地面，因此 186 是待确认的机器线索，不是新增合成配额。独立空间审计写库时，池中有 85 个不同帧身份具备至少一项任务的 `composite_background` 资格；后续扩额试点变化见下文。扩池编制须同时读取入库空间观测和本任务资格，结合已标注对象、目标占位及遮挡约束放置。[批量入口与坐标定义](DATASET_DESIGN.md#背景图入库与复用) · [审计汇总](../datasets/annotated/common/perspective/background_pool_space_v1/summary.json)

**共享池扩额试点与空间复核。** `parking_003` 同父视频的 20 帧已在一次逐图录入中写入垃圾桶状态和禁行牌的整帧视觉结论；垃圾桶状态 20 条 test `empty_negative` 已生效，禁行牌 19 条新空负与原有 1 条共同形成 20 条 test 空负。实际四桶落点探针通过池索引 165–179、182 的 16 帧，逐图库据探针回执把这 16 条垃圾桶状态 test `composite_background` 修订为 `eligible`；索引 164、180、181、183 在 300 次布局尝试后仍未找到四桶位置，保持 `unknown`。新补的 19 条禁行牌 `composite_background` 仍为 `unknown`，其原有索引 180 的 1 条 test 背景资格保留。20 帧属于 1 个物理父视频：任务 `source.yaml` 的旧视频路径与池清单的现路径 SHA256 同为 `38be0e149cc8…`；来源路径映射尚未自动合并，计数按同一物理视频。禁行牌同源已有旧检查接触，只能用于已接触场景回归，不能称为独立新场景测试。[试点逐图证据](../datasets/annotated/common/review/pool_quota_pilot_20261117/review.jsonl) · [四桶小样探针](../datasets/annotated/waste_bin_state/pool_quota_probe_20261117/probe.json) · [资格修订回执](../datasets/annotated/common/review/pool_quota_pilot_20261117/probe_promotion_receipt.json) · [来源配对](../datasets/scripts/waste_bin_state/config/source.yaml)

另从 `full_scene.mp4` 同一 train 父组新增池索引 38、41、42、43 四帧：完整空负与垃圾桶状态 `composite_background` 各 4 条 `eligible`，四桶探针记录尺寸上限 0.46；索引 46 的探针失败，未纳入本批。两组试点合计新增垃圾桶状态合成底图资格 20 条（parking test 16、full_scene train 4），完整空负 24 条（20、4），分别保留父组和 split；这是逐图库资格及小样接触结果，正式生成产量以实际入口回执为准。[full_scene 入库回执](../datasets/annotated/common/review/pool_quota_full_scene_train_20261117/ingest_receipt.json) · [试点资格修订](../datasets/annotated/common/review/pool_quota_pilot_20261117/height_limit_revision_receipt.json) · [来源别名核对](BACKGROUND_SOURCE_LEDGER.md#已核实的物理父源别名)

新四桶前景采用蓝绿开、红灰闭。在同一 `full_scene` 父视频的 3 张已有 train 资格底图 `train_000163`、`train_000091`、`train_002188` 上重算脚点，实际出图 6 张、24 个 YOLO 框，使用弱接地阴影。旧竖盖前景的 site009、site012 和共享池合成预览已移除，旧探针坐标和底图证据继续保留。选用旧探针覆盖的帧时按新前景重做放置检查。这 6 张作为合成外观与图签的历史参考，正式 train 导出另行编排。[合成候选清单](../datasets/annotated/waste_bin_state/sample_history_20261117/composites/manifest.json) · [生成计划](../datasets/scripts/waste_bin_state/config/pool_pilot_plan.json)

抽查空间叠图发现：池索引 53–73 的部分线模型远边穿过楼模立面，74–115 的部分地面四边形仅剩窄前景条；索引 199 的空黑地面未检出接触像素，索引 11、36 的落脚候选包含楼体下部。199 个帧身份中，线／纹理模型为 96、近场先验为 103（按 200 个路径计为 96／104）。因此局部 XYZ、候选掩膜及覆盖率只用于选择与复核，不直接增加配额；生成入口使用具体目标的尺寸、脚点和图签验证候选。

**共享空间观测 v2 定向修订。** `shared_background_space_v2` 已对 27 个不同帧身份写入新版空间证据，旧 v1 证据保留：楼模序列索引 53–73 共 21 帧将宽泛的地面四边形缩到可见的近地接触片区；索引 36 的楼体误判接触像素从 17,064 降到 6,624。先前没有接触候选的索引 0、21、25、191、199 现在有机器候选，其中索引 199 的黑色渲染地面由 0 增至 100,336 像素。新版 27 条仍全部标为 `contact_candidate_unverified`，其中 22 条用局部地面支持、5 条用已有地面掩膜；图示改善了这些误判和漏判，但不证明任意任务目标都能放入。写库回执记录受影响的 current 任务决策前后均为 436 条，原有 1 条垃圾桶合成资格未变；本次空间证据导入本身没有提升用途资格。后续提升必须以对应任务的完整标签、实际目标尺寸与落点探针另行修订，不能把 27 帧计成新增配额。[v2 空间汇总](../datasets/annotated/common/perspective/background_pool_space_v2/summary.json) · [楼模序列 QA](../datasets/annotated/common/perspective/background_pool_space_v2/qa/facade_series_053_073.jpg) · [误判与漏判 QA](../datasets/annotated/common/perspective/background_pool_space_v2/qa/prompt_cases.jpg) · [写库回执](../datasets/annotated/common/review/pool_space_v2_import_20261117/import_receipt.json)

随后对 `full_scene` train 池索引 21 单独补录：全帧无桶已证实，垃圾桶状态 `empty_negative` 为 `eligible`；四桶脚点探针虽通过，当时预览中桶组透视与场景不符且阴影不足，`composite_background` 保持 `unknown`，原因码 `perspective_or_shadow_composite_unverified`。索引 36、45 的旧四桶探针各尝试 300 次仍无完整放置，未升合成资格。计入这张空负后，该轮试点新增垃圾桶状态空负共 25 条，合成底图仍为前述 20 条；`e_bike` 在索引 21 的既有资格不随垃圾桶状态改变。这是旧前景和旧探针的历史结果；后续按修正后的桶盖掩膜、实际可放组合、完整标签和遮挡重新判定。可靠的图像尺度比可改善缩放，但不是底图资格或制作的必填条件，缺失时记录所用的近似尺度。[索引 21 入库回执](../datasets/annotated/common/review/pool_quota_full_scene_21_20261117/ingest_receipt.json) · [当时的资格检查](../datasets/annotated/common/review/pool_quota_full_scene_21_20261117/post_ingest_check.json) · [旧四桶探针](../datasets/annotated/waste_bin_state/pool_quota_probe_20261117/full_scene_extra_probe.json)

**共享空间观测 v3 方向字段。** 在 v2 的 8 个选定帧（池索引 0、21、25、36、53、73、191、199）上写入 `shared_background_space_v3`：接触掩膜与 v2 逐像素相同，方向与尺度状态单独记录。索引 21 从有地面支持的图像线段估出地面**图像轴向** 36.3°、置信度 `moderate`；36、53、73 也有轴向估计，其余 4 帧方向 `unknown`，8 帧尺度均 `unknown`。不再把旧版假定的 0°／90°／180°／270° 投影方向当作测量值，`projected_headings_deg` 置空。图像轴向不是桶组或相机的物体 yaw，也不是光照方向；它不能独自校正合成透视和阴影。写库回执中受影响 current 任务决策前后均为 80 条，索引 21 垃圾桶合成资格仍 `unknown`；合成 A/B 的视觉结果仍待验收。[v3 观测汇总](../datasets/annotated/common/perspective/background_pool_space_v3/summary.json) · [索引 21 轴向 QA](../datasets/annotated/common/perspective/background_pool_space_v3/qa/005751_ground_axis.png) · [八帧 QA](../datasets/annotated/common/perspective/background_pool_space_v3/qa/representative_axes.jpg) · [写库回执](../datasets/annotated/common/review/pool_space_v3_import_20261117/import_receipt.json)

**共享空间观测 v5 批量写库。** 200 条共享池路径对应 199 个帧身份，全部已写入接地候选和局部无米制坐标；其中 8 张旧版漏检图经原图与 SAM 2 核对后补出接地范围，部分范围仍较窄。109 帧有地面图像方向，153 帧有通过质量检查的同帧远近排序，19 帧有地平线或局部等宽标线支持的相对尺度比。远近排序不能直接当尺寸比；标线尺度只适用于其锚点之间的中心轴。入库时保留旧观测和任务资格，垃圾桶合成底图资格仍为当时的 56 帧，后续可随完整标签、新前景脚点及遮挡证据动态修订。没有尺度比的底图可用记录了依据的近似尺寸合成，现有接地候选也不自动等于某任务的合成配额。[逐帧结果](../datasets/annotated/common/perspective/background_pool_space_v5_relative/full_pool/observations.jsonl) · [算法验收](../datasets/annotated/common/perspective/background_pool_space_v5_relative/verification_receipt.json) · [写库回执](../datasets/annotated/common/review/pool_space_v5_import_20261117/import_receipt.json)

**修正桶盖后的底图资格扩额。** 后折桶盖的新前景完成图签和掩膜审计后，以其实际轮廓对 28 张实拍底图做脚点、原图目标标签及遮挡复核；18 张形成可复核的用途证据，分别为 train 8、val 3、test 7，其中 9 张支持当前四桶组合、9 张支持单桶。新增资格先覆盖这 18 张；实际生成入口进一步查出 train 池索引 45、46 属于已知回归曝光，随即把它们的 train 合成资格改为 `blocked`，保留空间与标签证据。对全部原有资格再运行实际生成入口的曝光规则，另发现 1 张旧 train 底图也被限定为回归用途；一并阻止后，当前 `waste_bin_state/composite_background` 有效帧身份为 71（train 21、val 25、test 25）。按同一生成规则计算，原有 56 条中实际可供生成的是 55 条，本轮净增 16 条。资格依据本任务完整标签、新前景放置和来源使用范围，图像测得的尺度比不设为准入条件；生成入口对 16 张新增资格及其接地掩膜已完成读取核对。[逐帧候选与预览](../datasets/annotated/waste_bin_state/v5_new_foreground_placement_20261117/manifest.json) · [初次写库回执](../datasets/annotated/common/review/bin_pool_v5_new_foreground_import_20261117/import_receipt.json) · [新增帧曝光修订](../datasets/annotated/common/review/bin_pool_v5_new_foreground_import_20261117/exposure_amendment_receipt.json) · [全量曝光核对](../datasets/annotated/common/review/bin_pool_v5_new_foreground_import_20261117/full_exposure_audit_receipt.json)。实际生成入口随后从新入库的 train 帧索引 20、185 各出 1 张单桶与四桶小样；清单回读的资格记录均为 current，YOLO 图签分别 1、4 框且坐标合法，脚点与新掩膜放置记录一致。两张均记录 `scale_basis=planned_approximation`，没有图像尺度比也正常完成合成。[新增资格生成小样](../datasets/annotated/waste_bin_state/sample_history_20261117/composites_qualified_pool/manifest.json)

沿索引 21 的 36.3° 地面轴向另制作斜视四桶与柔影 A/B，记录了 4 个候选脚点和图签。它使用 `photo_orbit` 的 **test 前景**，场景尺度仍 `unknown`，产物标为 `diagnostic_only`；该预览只用于比较构图与接地，不能跨 split 进入 train，也没有把索引 21 的垃圾桶合成资格升为 `eligible`。[斜视柔影诊断](../datasets/annotated/waste_bin_state/pool_quota_probe_20261117/index21_along_ground_axis_soft_shadow.json) · [阴影 A/B](../datasets/annotated/waste_bin_state/pool_quota_probe_20261117/index21_soft_shadow_ab.png) · [历史小样原路径与校验记录](../datasets/annotated/waste_bin_state/sample_history_20261117/migration.json)

**首次录入的多任务事务验收。** 在临时逐图库中，`vehicle_0001.png` 只解码并提交一次，两个电动车实例以共享 `object_id` 同时供四个任务引用；同一事务保存 `e_bike`、`person`、`sign`、`meter` 的 val 完整标签及环境。提交后独立查询立即得到电动车 `direct_target eligible`，其余三任务 `empty_negative eligible`；仪表的 `composite_background` 仍为 `unknown`，空间不满足放置也没有挡住已证实的整帧空负资格。此为隔离临时库的入口验收，未把该样本写进正式库或计入上述试点配额。[一次录入回执](../datasets/annotated/common/review/first_intake_vehicle0001_20261117/temporary_intake_receipt.json) · [提交后查询](../datasets/annotated/common/review/first_intake_vehicle0001_20261117/temporary_intake_check.json)

**共享池八任务待补状态写库。** 按 199 个帧身份 × 8 个任务核对当前合成用途；452 个原先连当前 `composite_background` 决策都没有的组合，现已依据已有共享空间、任务标签及任务几何证据写为 `unknown`，分别指出标签未审／不完整、作用域待核或任务放置未审／未合格。未修改任何既有资格，也没有把共享接地候选升为任务配额。重跑对账新增数为 0，库的 `quick_check=ok`，1592 个组合现在都有可查询的合成用途状态；在下述快速补证前，垃圾桶有效合成配额为 71 帧。旧机器审查关联表按当前证据重建后由 175 条当前关联变成 159 条，排除项由 3 变成 7 条；这些变化只清除已被后续对象观测替代的旧关联，不改变逐图库资格。[批量回执](../datasets/annotated/common/review/pool_task_pending_20261117/receipt.json) · [对账入口](../datasets/scripts/common/reconcile_pool_pending.py) · [当前机器关联](../datasets/annotated/common/sources/background_pool_machine_review_index.json)

**新前景快速补证。** 从垃圾桶任务“标签完整、split 已定、落点待核”的 77 帧中筛出尚未用新桶前景试放的 67 帧，按共享接地掩膜优先查看空间较大的候选。池索引 95、189、190 的单桶试放预览经原图视觉核对，三个支点均通过来源像素掩膜，新增目标框不覆盖已有已标桶；近似 90 像素桶高和轻柔接触阴影记录在诊断清单。索引 95 的 test 试放借用 train 桶渲染验证背景几何，**不是**可用于 test 的合成图。把这三帧的局部任务接地证据及完整原图标签关联入库后，`waste_bin_state/composite_background` 当前有效帧身份由 71 增至 **74**。其中 189、190 的底图各有一只已标注的黑色开盖桶：生成器现可读取 `state`／`bin_color` 字段，保留其原框并继续给新增蓝桶出图签。实际生成入口的 train 两条计划输入已通过身份、原图哈希、现行资格、已有桶标签、接地与来源规则检查；test 底图资格可查，但现有前景渲染清单属于 train，生成入口现会拒绝跨 split 组合，待同 split 前景素材准备好再生成。本批尚未运行正式生成，仅保留诊断预览。`source_role` 对这三张为 `unknown/historical_exposure_incomplete`，未命中已知回归限定，后续正式批次仍记录此来源边界。[预览与脚点](../datasets/annotated/waste_bin_state/pool_fast_probe_20261117/manifest.json) · [入库和入口核对](../datasets/annotated/common/review/bin_pool_fast_probe_import_20261117/receipt.json)

实时库存按任务、用途、split 和父组查询 `state_coverage.py --source-status`。某条记录可分配还取决于计划所需的接地宽度、标签作用域及其他布局约束。[查询命令](../datasets/scripts/README.md#逐图入库与资格查询)

### 背景入库仍需接通的代码

| 现有入口 | 已接通与当前缺口 | 最小接线 |
| --- | --- | --- |
| [单图录入 `ingest_frame`](../datasets/scripts/common/ingest_frame_environment.py)与[历史回填](../datasets/scripts/common/backfill_frame_environment.py) | 单图接口用同一份原始字节计算哈希、解码并共享图像，统一提交多任务对象、观测和资格。任务事实由调用方或 `review` 回调提供。历史回填导入已有清单与标签，部分空间估计仍按任务读图。 | 新图用同一次录入写入判定结果。历史资料先复用已有证据，仅补齐受影响用途的缺项。 |
| [Arena 轨道帧入库](../datasets/scripts/common/ingest_shared_route_backgrounds.py) | 已写入接地观测及楼名纸条、仪表的通用底图资格。现用楼名替字与仪表固定站位配方分别使用各自指定的来源。 | 核对楼名纸条和仪表的生成计划如何选用这些帧及布置前景。 |
| [火焰三分支生成](../datasets/scripts/fire/compose_real_facades.py) | 当前数据库的 5 张完整实拍、3 张实拍底图及独立 Blender 场景已接通，形成 54 张图签。 | 现有 C 楼底图库存 24 张／240 区域可继续编制底图叠贴分支，独立评估和训练导出按后续目标安排。 |
| [共享池接收观测](../datasets/scripts/common/background_pool.py#L120)与[地面估计](../datasets/scripts/common/background_pool.py#L319) | 按来源路径汇总逐图和任务记录。旧池分支重建时仍会估计地面，仅部分任务计算目标尺度下的放置范围。后续观测追加到 `additional_evidence`，首条判定仍被消费。 | 复用现有来源清单和机器观测，按图像身份、坐标基准、观测版本增量刷新。任务记录明确当前有效证据，保留旧证据、人工内容和原批次引用。电动车几何应从入库事实读取，解除[依赖旧生成计划](../datasets/scripts/common/background_pool.py#L182)的反向引用。 |
| [媒体字段提取](../lab/scripts/media/media_index.py#L113)与[任务过滤](../lab/scripts/media/media_index_sqlite.py#L406) | 索引已识别亮度、饱和度和任务证据。部分池内成像属性仍在 `strata`，组合条件查询主要适配直接实拍记录；`task-inputs` 返回计划和配置声明。 | 库内按同一图像、任务、角色和 split 的记录查询 `eligible`，携带标签与空间证据。旧清单和索引提供回填与定位。 |
| [共享池排序](../datasets/scripts/common/background_pool.py#L355)与[禁行牌编计划](../datasets/scripts/sign/plan_sign_sources.py#L102) | 共享池按已声明用途和 split 轮转父组，几何参考行尚未进入合成用途。禁行牌已将选中背景交给贝叶斯分配；其他任务仍依赖各自接口。禁行牌对所有候选组给最低配额，扩池后可能超出固定预算。 | 把满足当前用途的背景纳入各任务现有计划编译。对已选可行组合和必做覆盖分配最低配额，可选组由分配器择用。 |
| 电动车[计划校验](../datasets/scripts/e_bike/compose_v4_grounded_train.py#L267)、禁行牌[共享池分支](../datasets/scripts/sign/compose_sign_v4_measured.py#L409) | 电动车与禁行牌配方现可继承完整旧目标框并拒绝新增目标遮住旧框。 | 按配方声明所需标签和可放区域。 |
| 禁行牌[共享池执行校验](../datasets/scripts/sign/compose_sign_v4_measured.py#L610)及[父组配对](../datasets/scripts/sign/config/source.yaml) | 现用计划按本批所选底图核对当前资格。`parking_003_017` 对应的同字节父视频已从 train 移出，保留有既往接触的 test 诊断用途。 | 新批按明确的父组和 split 选源，核验所选资格及引用素材。独立评估按回执确认未接触的来源。 |

### 受影响入口核对

入口改动按涉及的任务和样本核对。录入看一次提交后的用途资格和证据。选源及生成看是否读取当前资格，以及本次文件、标签和实际放置是否一致。旧帧按需要补录受影响的用途。

## 状态计划的输入与入口契约参考

计划声明任务、split、预算、约束及 `strata`。生成器消费该任务已实现的字段。组合与分配见[贝叶斯状态组合与生成契约](DATASET_DESIGN.md#贝叶斯状态组合与生成契约)。

## 配方与历史

人物批次记录见[全合成候选](../Archive/technical/datasets/person/v4_ground_person_3600_20261116/)、[首批 300 张快照](../Archive/technical/datasets/person/v4_train_300_20261116/)和[小样迭代记录](../datasets/scripts/person/config/sample_issues.yaml)。

[数据制作参考](DATASET_RECIPES.md)保留任务配方、实拍依据和制作参数。

- <a id="每个模型的标注范围"></a>[每个模型的标注范围](DATASET_RECIPES.md#每个模型的标注范围)
- <a id="多原件取样与场景混合"></a>[多原件取样与场景混合](DATASET_RECIPES.md#多原件取样与场景混合)
- <a id="实拍实景合成与-blender-的接入顺序"></a>[实拍、实景合成与 Blender 的接入顺序](DATASET_RECIPES.md#实拍实景合成与-blender-的接入顺序)

[2026-11-15 已撤出生产路径的批次记录](../refs/archive/DATASET_BACKGROUND_2026-11-15.md)
