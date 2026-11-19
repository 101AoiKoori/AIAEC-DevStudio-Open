# 禁行牌具数据任务

本任务为 `sign`。定位完整或仍可辨的牌具，普通立杆作为硬负例。 下列描述用于导航，任务目标、标注范围与组合对照以配置中的 `sign` 条目为维护源。

- [学习目标与标注范围](../../../config/task_targets.yaml)；[组合轴与对照](../../../config/combination_axes.yaml)：牌具视面、截断、背景与普通杆对照。
- [本任务数据配置](config/dataset.yaml)与[素材来源配置](config/source.yaml)：具体分支读取其支持的字段。
- [制作入口](../README.md#恢复生产入口)列出可执行分支；[任务脚本](./compose_sign_v4_measured.py)是其中一个入口。产量与接线状态以[当前生产进度](../../../docs/DATASET_PRODUCTION.md#当前生产进度)及实际回执为准。
- [训练入口](../../../lab/features/sign/training/scripts/train_sign_v4.py)及[训练流程](../../../docs/TRAINING_WORKFLOW.md)：按任务配置和实际预算使用；脚本存在不代表模型已训练或完成交付。

数据制作使用同一 [dataset-production-loop](../../../.agents/skills/dataset-production-loop/SKILL.md)，无需按任务复制 Skill。

当前计划与实际图签对账运行 `python -B lab/scripts/contracts/state_coverage.py --actual --json`。实际数量读取当前清单及配对标签，保留分集、实例数、未记录字段和缺失文件；查询只读。

编计划时，不可行的组合会留下缺额并继续其他组合。单个组合当前最多尝试 10,000 次，耗时随底图及候选放置而变。[编计划入口](plan_sign_sources.py)

## 当前全量数据

当前计划为 [Train 2500 张](config/full_train_plan.json)、[Val 300 张](config/full_val_plan.json)和 [Test 300 张](config/full_test_plan.json)，配额由 [source.yaml](config/source.yaml) 的 `full_dataset` 维护，登记于 `task_targets.yaml#tasks.sign.plans`。`dataset.yaml#actual_manifest` 指向 `datasets/exports/sign/manifest.json`，全局实际统计读取该全量清单及图签。

| 场景预算 | Train | Val | Test |
| --- | ---: | ---: | ---: |
| 完整牌具 | 1300 | 140 | 140 |
| 部分截断或遮挡 | 500 | 60 | 60 |
| 可辨杆底座截断 | 200 | 25 | 25 |
| 无牌空场 | 300 | 45 | 75 |
| 普通信号灯支杆硬负 | 200 | 30 | 0 |

Train 使用 800 张 Blender 底图派生及 1700 张相机来源图，其中 9 张为直接实拍正例；Val 全部使用相机来源，其中 11 张直接实拍正例。截断预算包含 Train 80／Val 16 张实景灯体遮挡。普通杆负例来自各集合一张已核无牌原帧的保留完整灯杆裁区与色调派生，不能按张数计作不同实物。原有牌具框随合成输出保留。 Test 实际为300张、225框、75空负，20张实拍帧来自一个预留停车视频父组；19条历史待验证资格已用已有完整无牌标签与当前源坐标接触检查补齐，回执为 `datasets/annotated/sign/test_background_qualification.json`。该来源没有已确认普通信号灯杆，空负保留电动车干扰；没有直接牌具实拍正例。原始牌具照片及掩膜完整，首次Test中新增的8张局部电动车遮挡造成封闭背景窗口，已全部用边缘截断状态替换，未保留该遮挡配方；记录见 `test_occlusion_repair.json`。

```powershell
.venv/Scripts/python.exe -B -m datasets.scripts.sign.build_full_dataset --split train --generate
.venv/Scripts/python.exe -B -m datasets.scripts.sign.build_full_dataset --split val --generate
.venv/Scripts/python.exe -B -m datasets.scripts.sign.build_full_dataset --split test --generate
.venv/Scripts/python.exe -B -m datasets.scripts.sign.audit_current_sample --full
.venv/Scripts/python.exe -B lab/features/sign/training/scripts/train_sign_v4.py check
```

省略 `--generate` 只编制计划；已有图签重跑前按当前 split 清单清理生成结果，保留原件、人工内容和生产回执。合成器直接写出 `exports/sign/<split>/images` 与 `labels`。核对入口逐图验证标签、提示封闭遮挡背景窗口、独立可见掩膜框、继承框、遮挡、四视面与远近截断联合格及跨集重复，汇总当前清单、`data.yaml` 和 `workspace_sources.json`。证据为 `datasets/annotated/sign/full_dataset_audit.json` 与 `full_dataset_contact.jpg`。

训练入口读取 `datasets/exports/sign/data.yaml`；本阶段完成数据及训练预检，未启动模型训练。背景按父来源分集，前景仍共享同一牌具视频，多数实拍来自同一室内模型场地，Val 用于受控诊断。Test停车视频已有旧禁行牌检查及人物Test接触，作为同场地回归，不作独立泛化结论；不参与训练选择。三集入口预检见 `datasets/annotated/sign/full_three_split_preflight.json`。识别效果需在训练后评估。

## 禁行牌训练包

当前完整包为 [training_packages/sign](../../../training_packages/sign/README.md)，携带 Train 2500、Val 300、Test 300 图签、来源及逐实例诊断属性、同类 V3 初始权重和共享训练引擎。与同级 `portable_env` 一起使用；无需原工作空间的数据或代码。包内预检、初始权重加载和一张 Train 图的 CPU 前向检查通过，尚未启动训练。[检查回执](../../../training_packages/sign/package_checks.json)

训练上限120轮，前90轮不累计早停耐心，之后连续20轮没有至少0.001的验证mAP50–95提升则停止；仍保存严格最高Val分数的权重。单次最长36000秒（10小时），包含暂存、训练和训练后处理，硬时限优先。640输入、batch16、AdamW及现有轻量增强保留；不能从轮数保证收敛或无过拟合。

在包根运行 `./train.ps1`；续训用 `./train.ps1 -Resume`。完成后运行 `./evaluate.ps1 -Run "runs/sign/<运行目录>"`，按Val评估、专项报告、冻结、Test评估和专项报告的顺序执行；任一步失败即停止。Test不用于调参或选模。

`sign_report.json` 分别报告四视面（含斜视）、完整／截断／遮挡／杆底座、短边小于32像素的实例召回，以及来源、色调、场景配额、整图成功和空负误报。实例属性逐框核对标签与尺寸，原图继承目标的未知视面保留未知，误检不被猜分到某个视面。机器确认记录可直接消费，不补造人工签收。共享牌具及Test旧接触的范围随来源索引保留。

## 已接受的分集小样

读取 [source.yaml](config/source.yaml) 的背景分配、抽样和几何参数，按物理父来源与 Blender 场景／相机路线选择代表帧。小样回放计划为 [Train](config/sample_train_plan.json) 与 [Val](config/sample_val_plan.json)。每个背景保留原图对照，可行布局分别生成四色调配对；有旧牌具的背景保留旧框，不能视为空负。

```powershell
.venv/Scripts/python.exe -B datasets/scripts/sign/plan_sign_sources.py --diverse-sample --split train
.venv/Scripts/python.exe -B datasets/scripts/sign/plan_sign_sources.py --diverse-sample --split val
.venv/Scripts/python.exe -B datasets/scripts/sign/compose_sign_v4_measured.py --output datasets/samples/sign/train --seed 1921 --plan datasets/scripts/sign/config/sample_train_plan.json
.venv/Scripts/python.exe -B datasets/scripts/sign/compose_sign_v4_measured.py --output datasets/samples/sign/val --seed 1921 --plan datasets/scripts/sign/config/sample_val_plan.json
```

重跑前仅按各 split 的当前清单清理生成图签与清单，保留目录、原件、人工内容和持久回执。根 `datasets/samples/sign/manifest.json` 为两集合汇总，消费单集合时读取子目录清单；旧单集合打包器不能直接消费混合集合汇总。

当前全局计划登记与实际统计已切到上方全量计划和导出。小样计划由 `source.yaml#diverse_sample_plans` 留作复现，440 张已接受图签保留在 `datasets/samples/sign`。

原 188 张根目录小样已清理，持久生产回执保留。该小样保留作对照，全量图签另写入 `datasets/exports/sign`。前景仍共享同一物理牌具视频，多数实拍为同一室内模型场地；Val 只说明受控诊断覆盖。背景对照若保留原有牌具，按标签计正例；其原有牌具视面未记录时保持未知。

## 本轮困难组合与核对

`source.yaml/sample_refinement` 维护四视面 × 小目标／中景／部分截断／杆底座截断的联合格、连续高度范围和局部修订。每个有效布局生成原色、降亮、降饱和、联合变化四张配对。16:9 原图按实际 4:3 裁区变换标签，完整前景放大后按画面边缘自然截断。

遮挡使用本集合原图内真实信号灯灯体，轮廓与摆位证据写在任务配置。普通信号灯支杆且整帧无牌的图片作硬负对照；原有牌具框继续保留。色调派生不计为新增独立实物或父来源。以识别任务和有效监督为目标，接地、尺度及阴影仍为像素合成近似，不以真实感作为普遍生产门槛。

```powershell
.venv/Scripts/python.exe -B -m datasets.scripts.sign.audit_current_sample
```

该入口逐张检查图签和独立重算的可见掩膜框、遮挡后框、继承框，核对16格覆盖、同布局四色调以及成图实测亮度／饱和度；检查硬负、跨集父来源与相同RGB图像，重建根汇总清单和 `datasets/annotated/sign/current_sample_contact.jpg`。当前计数与残余范围以 `current_sample_audit.json` 为准。上一轮193张生成图签已按清单替换，原件、目录和持久回执保留，未启动训练。

## 共享底图扩源

补证批次的 147 条合格背景已全部分配：Train 70、Val 77；原有 Train 585 张继续可用。Train 池现为 655 张。Val 另有一条历史照片资格与 Train 同父，本轮选样排除该父来源。`bins004`、`009`、`012` 和 `V_20261112_152304` 按整父来源保留在 Val，并如实记录同场地、跨任务接触范围。记录见 `datasets/annotated/sign/background_allocation_receipt.json`、共享批次的 `recheck_receipt.json` 与当前逐图库。

完整扩源规划可运行 `plan_sign_sources.py --expand-backgrounds`；它与有界小样分开。原 [background_expansion_plan.json](config/background_expansion_plan.json) 是扩源前 585 帧的预检快照，不代表本轮实际样本。更早的 [sample_recipe.json](config/sample_recipe.json) 和 [production_plan.json](config/production_plan.json) 留作旧配方追溯，当前命令及 `source.yaml/production_plan` 已指向全量 Train 计划。

<!-- TASK_DESIGN:START -->

## 学习目标与组合设计

本节由共享配置生成；修改目标与组合请编辑 [task_targets.yaml](../../../config/task_targets.yaml) 和 [combination_axes.yaml](../../../config/combination_axes.yaml)，运行 `python lab/scripts/contracts/state_coverage.py --write`。

**学习及标注**

- 类别：sign_fixture
- 有无：placed、absent
- 负例：absent、ordinary_pole

整体与仍可辨的局部牌具按可见范围标框；仅有一般形状、身份不可判的残段保留未知。确认普通立杆且无牌具的画面可做硬负。

absent 和 ordinary_pole 描述场景；唯一类别为 sign_fixture。

**素材用途**

- `direct_target`：全帧至少一件可辨牌具，完整牌或可辨残段的可见框齐全。
- `target_foreground`：有真实视面证据的牌具裁片及掩膜，保留杆、底座和可辨残段证据。
- `hard_negative`：普通立杆实物可确认，且输出范围内无牌具。

**组合轴与对照**

以下是本任务的设计条件；`dynamic` 表示依据实际素材和任务计划确定。

- `view`：front、oblique、side、back
- `truncation`：none、partial、pole_only

**布局**：路口有效支撑区和观察位、坐标尺寸及允许裁切或遮挡范围；既有牌具须带可见标签。

**fixture_background**：通用立杆或固定场地不能充当禁行牌具特征。

- 安排：同背景有牌和无牌；可依据杆与底座辨认的牌具裁切正例，对照形状接近的真实普通立杆硬负。
- 核对：按 actual.geometry.visible_kind、遮挡后框及已确认普通杆记录核对；可辨牌具保留框，普通杆独占画面时查完整空标签。

**view_tone**：不把单一正视模板或原始色调当作牌具本身。

- 安排：正视与斜视都用完整牌和可辨截断牌；保留各自的原图与降亮／降饱和版本作配对。
- 核对：用 actual.view、actual.geometry 和标签核对视面与截断联合格；tone 只说明处理，实际亮度与饱和度需另测。

对照描述来自设计配置；结构化记录目前为部分编码，具体制作按任务入口和可用素材安排。

<!-- TASK_DESIGN:END -->
