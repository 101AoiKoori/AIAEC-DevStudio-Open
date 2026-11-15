---
name: dataset-production-loop
description: 协调或执行本工作空间 V4 数据集制作时，先按 config/workspace_layout.yaml 检查目录契约，再以任务目标、逻辑设计和版本交接推进；不代替通用实验、文件或跨聊天治理。
---

# 数据集制作闭环

读取 `datasets/DATASET_COORDINATION.md` 和当前任务需求；按实际影响使用 `workspace-experiment-governance`、`workspace-file-governance`、`workspace-doc-governance`，涉及跨聊天再使用 `workspace-collaboration`。它们的路径、来源、授权和提交约束继续适用。总控启动文本在 `datasets/DATASET_COORDINATION_PROMPT.md`，该文件自身不创建聊天或授予后台运行。

## 选择任务与输入

开工前运行 `python lab/scripts/contracts/workspace_layout.py check --task <任务>`；新增或调整目录先改 `config/workspace_layout.yaml`，再运行 `python lab/scripts/contracts/sync_layout_docs.py`。训练目标与组合轴分别见 `config/task_targets.yaml` 和 `config/combination_axes.yaml`；覆盖和容量缺口用 `python lab/scripts/contracts/state_coverage.py`。全量计划草案见 `config/full_train_allocation.yaml` 和 `docs/FULL_TRAIN_PLAN_DESIGN.md`；草案是 PlanDraft，不覆盖 `production_plan.json`，编译前核对来源容量和父组 80% 上限，不把缺口直接当配额。`source.yaml` 的 `pairing` 是父子源与 split 预留权威；编译前运行 `python lab/scripts/contracts/source_pairing.py --check` 并核对 `plan_alignment`，`stale` 不得当已闭环。配对校核以 `source_pairing.py` 为单一 checker；`source_ledger` 只作数据访问/实验，不作第二套校验器。

核对当前用户意图、`config/perception_tasks.json`、`docs/DATASET_DESIGN.md` 和 `docs/DATASET_PRODUCTION.md`。先按任务目标、需要学习的特征与应打破的错误依赖，以逻辑确定合理状态和联合组合；不要求每个人为设计的组合先找到实拍对应证据。再结合 `datasets/README.md` 查询来源及入口落实计划，使用当前实拍并重建缺失标签，保留本轮已有标签的实际状态，确认实际路径、标签语义和权利边界。来源查询与入口检查用于执行，不是穷举研究的前置门槛。八个数据职责名不是同名目录或配置 ID 的保证；火焰与楼名对应 `lab/features/fire/`，桶状态与垃圾贴纸对应 `lab/features/waste_bin/`，垃圾贴纸的训练工作副本在 `lab/features/waste_bin/training/config/`。现行实际目录及缺失输入以磁盘为准。Blender 登记库、媒体缓存、生产回执各有独立职责；查询时确认路径与快照状态。

依据任务规则和可解释逻辑一次性确定生产方案；可控合成可构造实拍未逐一出现的合理组合，保持目标语义及正确标签。实际来源、实现范围、标签、测量及产出如实记录，未实测的权重和参数标明设计假设，不当作现场分布或已测效果。每模型只有一次训练机会；不安排常规开发验证、单次模型验证或最终测试，不以模型验证结果安排重训或选模；用户明确授权的有限素材、小样或代码契约测试按指定范围执行。按已有证据核对来源、结构和标签语义，记录未知和已知缺陷；不得把未测效果写成通过。已有用户授权范围内自主推进，不因本 Skill 重复请求批准；启动训练、部署或生产的授权按用户指令及工作区规则判断。

单张和批量生成的目标架构、明文状态输入、贝叶斯工具职责与现有实现边界遵循 [贝叶斯状态组合与生成契约](../../../docs/DATASET_DESIGN.md#贝叶斯状态组合与生成契约)。按指定批次或数据集定义组合与分布；预算不足或不可行如实报告，实际产物与配额计划分别记录并复用既有生产回执。

按[数据来源与生成方式的定义](../../../docs/DATASET_DESIGN.md#数据来源与生成方式的定义)确定允许的原始来源及 Blender 资产范围，区分标注和派生中间结果，记录全部父素材和已知来源组。数据库检索候选及证据，任务 JSON 表达本次来源与状态组合，分配器计算配额、已接入的生成器执行并记录实际结果；具体覆盖以代码为准。来源抽样的池、单位和随机方式与状态抽样、贝叶斯配额分别确定；现有分配工具不会自动随机选择来源。计划、实际生成、回执和允许用途分别记述。

## 记录与修正

已知逻辑矛盾或标签错误按证据定位并处理，不以反复生产、模型验证或多次训练作为默认流程。关键缺陷未解决时记录受限范围；预算到期不等于通过。

实际集合用途按任务方案及项目用户授权组织，不强制保留验证集或测试集；保留同源及跨集合复用记录，同源数据不冒充独立采集。确认负例、漏标、未知、候选和通过各不相同；每个模型的真值独立记录。视觉判断需说明覆盖范围与证据，禁止凭照片臆造相机参数或掩盖不可见目标。

标注和审查按 `source_available`、`annotated`、`candidate`、`auto_validated`、`human_confirmed`、`training_admitted` 六层独立计数，并在所有摘要中列出 pending；不得把 `human_confirmed=0` 写成没有标注，也不得因缺少人工确认清空候选或既有标注。程序化接受只进入 `auto_validated`；人工语义确认与框、关键点或掩膜确认分开记录；只有实际失败证据使用 rejected/excluded，缺少 `accepted_by`、未复核或未准入保持 pending。复核/manual 重跑必须 merge-preserve，默认校验只读；模板重建须显式请求、检测非空人工内容并先备份。

## 版本、协作与验收

用户直接与任务窗口协商并推进具体数据集；协调镜像负责跨窗口消息、问题和共享资源协调，按请求核查权威文件与实际证据并返回适用范围和未知。总控用于用户直接提出的总体方向，不负责日常消息中转。任务窗口自主处理日常制作，不增加逐步审批。参与窗口及职责见 `datasets/DATASET_COORDINATION.md`；跨聊天通信按用户授权、可用工具和 `workspace-collaboration` 执行，不假定自动后台监听。

任务路径、必需配置和标注根见 `config/workspace_layout.yaml` 与[数据集工作目录](../../../docs/DATASET_DESIGN.md#数据集工作目录)；每任务只维护一个当前小样和数据集，生成脚本必须消费本任务 `config/source.yaml`，共享能力放 `datasets/scripts/common/`。错误或过期结果按用户授权清除，不以并列版本目录代替 Git 和现有生成清单；严格保护 `runs/` 的内层 `runs/` 中的 V3 模型及记录。目录建立不代表旧生成器已接入新路径；全部生产和准备工作暂停至用户明确恢复。实验输出、素材和冻结版本按已有记录与目录职责管理，迭代不自动新增正式版本。版本变更记录输入、生成方式、代码配置、标签、实际 split、用途、检查和复用；冻结或已被消费的数据变更须保留版本关系及旧结果影响。Git 不保存原始数据或权重，也不能恢复被忽略文件。开工前运行 `python lab/scripts/contracts/workspace_layout.py check --task <任务>`。

共用文件由单一写者负责，Git 写操作及相同索引/数据库的写入按协作规则串行；不同 checkout 按实际结果核对。归档和清理以已核实的引用、用途、授权及恢复价值为依据，不把名称或日期当删除凭据。

报告实际产物及证据，阶段状态区分通过、受限可用与未通过；说明不完整部分和实际允许用途。交付前核查当前文件和结果，不能只引用对话自报；Windows 检查不冒充实车验收。阶段独立完成后遵守 AGENTS.md 的提交要求。
