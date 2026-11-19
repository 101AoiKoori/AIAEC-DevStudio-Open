---
name: workspace-doc-governance
description: 查找 AIAEC 项目文档的维护源，修订需求、当前说明及对应生成副本。
metadata:
  short-description: 编辑源文档并同步实际读者
---

# 项目文档维护

在维护源直接写当前做法和适用条件，其他入口链接到相应章节。批次结果留在对应记录，文档修订过程查 Git。生成内容的命令或规则从源配置修改。

## 源文件与生成物

| 要改什么 | 改哪里 | 怎样同步 |
| --- | --- | --- |
| 识别任务需求与稳定事项 ID | `config/perception_tasks.json` 中对应任务 | `python lab/scripts/contracts/sync_docs.py`，任务 README 整份生成，总览和任务矩阵有生成区块 |
| 工作目录约定 | `config/workspace_layout.yaml` | `python lab/scripts/contracts/sync_layout_docs.py`，保留有效的生成标记 |
| 训练目标、唯一全局计划登记与组合轴 | `config/task_targets.yaml`（含 `plans`）、`config/combination_axes.yaml`（仅组合设计） | `python lab/scripts/contracts/state_coverage.py --write` |
| 对象种类与生命周期 | `config/workspace_ontology.yaml` | `python lab/scripts/contracts/workspace_ontology.py --write` |
| 数据制作的当前起点 | [当前生产进度](../../../docs/DATASET_PRODUCTION.md#当前生产进度) | 协调提示、任务说明和脚本入口引用这一节 |
| 编号、计数单位、审核记录 | `config/id_policy.yaml` 和[数据记录对应表](references/data-authorities.md) | `python lab/scripts/contracts/id_policy.py --write` 同步[计数口径](../../../docs/ID_AND_COUNT_POLICY.md)，更新受影响的摘要 |

## 编辑

定义写清对象、适用范围、含义和可观察结果；能指向配置或实际入口的直接链接维护源。区分用户要求、方案、当前实现和运行事实，用具体条件代替“适当”“合理”等不可判定措辞。不同数据维度分别描述，不用单一状态推断来源、标签、用途或 split。格式及生成检查不能单独证明语义或运行效果；明确说明尚未实现或未知的范围。严格表达不等于新增生产门槛。

在普通查询中直接找到原文。可从仓库根用 `python .agents/skills/workspace-doc-governance/scripts/retrieve.py "查询词"` 查路径与行号。需要追生成关系时，查看文件头的生成说明、源配置和对应 writer。文件存在、机器判定、人工确认、训练产物分别使用各自的证据。

任务计划或清单变动时，先按 [dataset-config-operations](../dataset-config-operations/SKILL.md#变更时同步相关入口) 更新维护源与消费者，再原位更新受影响说明和 Skill；动态数量保留在清单或生成区块，不在使用说明重复手写。

改动生成内容时从源文件重建，然后运行同一生成器支持的检查。移动段落时修复现行入口及其锚点。写操作只覆盖这次选定的文件，原始数据和历史证据留在对应位置。报告说明变更及其实际作用范围。

需要检查活动文档的本地链接和章节锚点时，从仓库根运行 `python .agents/skills/workspace-doc-governance/scripts/documentation.py`；该脚本不访问网络，也不判断文档语义。Git 钩子仅在设置 `AIAEC_FULL_CHECKS=1` 时运行此检查。
