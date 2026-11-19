# 全局脚本

`lab/scripts/` 是工作空间级脚本区，按职责分类：

| 分类 | 职责 | 目录 |
| --- | --- | --- |
| `contracts/` | 目录契约、需求/生成文档、训练目标与组合轴覆盖、全量计划、对象分类学、编号口径和父子源配对检查 | [`contracts/`](contracts/) |
| `media/` | 媒体索引、SQLite 查询缓存、视频抽帧、合成背景收集和站点背景特征更新 | [`media/`](media/) |
| `blender/` | Blender MCP 桥接与源资产登记库查询/登记 | [`blender/`](blender/) |
| `data/` | 数据源门控、候选打包、现场采集评估和站点适配 | [`data/`](data/) |
| `audit/` | 生产入口静态审计、受管写入/转发/未分类路径检查 | [`audit/`](audit/) |
| `common/` | 跨任务人工辅助标注、来源审查、掩膜质量与 Blender 机位规划 | [`common/`](common/) |

所有脚本从工作空间根目录执行，不是正式 ROS 运行依赖，也不能被部署包导入。正式推理核心仍在 `ws/src/aec_vision/`，任务数据生成入口仍在 `datasets/scripts/<任务>/`，训练实验仍在 `lab/features/<任务>/`。各分类的具体入口以目录内脚本和 `--help` 为准。

文档一致性检查运行 `python .agents/skills/workspace-doc-governance/scripts/documentation.py`。它只读取活动 Markdown，检查工作空间内文件链接和 Markdown 章节锚点；不访问网络，也不替代生成器、需求源或数据语义审查。设置 `AIAEC_FULL_CHECKS=1` 时 pre-commit 运行此检查。

现场标注清单的派生计数用 `python lab/scripts/data/evaluate_field_capture.py audit-labels --labels <当前标签JSON...> --review-summary <汇总JSON>` 核对。默认只读；确认当前标签文件是目标输入后可加 `--write` 重算记录、图片、框和状态分布。工具会拒绝重复图像—任务记录、非法框、正例无框、负例带框和未知状态，并在汇总中保留原复核子集字段。

火焰和禁行牌的计划及当前图签对账使用 `python -B lab/scripts/contracts/state_coverage.py --actual --json`，同时返回计划分配和当前清单读取结果。实际框数来自标签文件，楼宇归属、视面与截断来自相应实际记录；未记录的值保持未知。加 `--source-status` 可同时查看逐图库存。

## common/

`common/` 保存跨任务的人工辅助标注、来源审查和场景规划脚本。当前范围包括：

- 实拍背景池的发现、语义审查、近重复分组、父来源和 split 预留辅助；
- 完整目标框选、标签重建、标签覆盖与未知状态检查；
- 掩膜边缘、孔洞、可见区域和遮挡层质量检查；
- Blender 母场景、相机机位、视角覆盖与参考画面配准辅助；
- 上述能力的清单、证据、覆盖报告和人工复核入口。

这些脚本服务于数据设计与人工辅助制作，不是正式 ROS 运行依赖，也不能被部署包导入。正式推理核心仍在 `ws/src/aec_vision/`，任务数据生成入口仍在 `datasets/scripts/<任务>/`，训练实验仍在 `lab/features/<任务>/`。

## common/ 放置和接口规则

- 新增脚本前先检查 `datasets/scripts/common/review/`、`lab/training_common/`、任务生成器和 `lab/scripts/` 现有分类，复用或局部扩展已有能力；只有语义和生命周期确实跨任务共用的新辅助能力才放这里。
- 任务特有的类别、计划、标签语义或生成参数留在任务目录；这里只保存跨任务算法、数据契约和人工辅助流程。
- 输入使用工作空间相对路径和现有清单证据，优先复用 `source.yaml`、媒体索引、审查记录和 Blender 登记库，不另建同义来源台账。
- 输出不得写入本目录。可重建观测按现有规则进入 `datasets/annotated/common/`，人工决策进入约定任务配置或清单，临时结果进入 `tmp/`；最终样本和数据集仍只进入 `datasets/samples/<任务>/`、`datasets/exports/<任务>/`。
- 辅助脚本不训练模型、不生成正式答案，也不把自动判定写成人工验收。人工确认字段保持空值，直到实际复核发生。
- 脚本应提供明确的输入、输出、失败状态和停止边界；不得静默回退到未登记素材，不得把缺失、孔洞、截断或未知结果改成成功。
- 新增或改变共享行为时，同步更新本文件；目录结构以 `config/workspace_layout.yaml` 为源，并运行 `python lab/scripts/contracts/sync_layout_docs.py` 与 `python lab/scripts/contracts/workspace_layout.py check`。

## common/ 已实现入口

从工作空间根目录使用 `.venv/Scripts/python.exe -m lab.scripts.common.cli ...`。所有生成物是可重建观测或人工决策草案，不是训练样本。

### 背景池

- `background discover` 只读媒体查询缓存，按路径片段和媒体类型发现图片。返回的 `review_state: unknown` 表示此命令未关联逐图入库判定，实际用途回查任务记录。
- `background summarize` 汇总已有 `plan.json/selection.json/metrics.jsonl` 和配对声明。该旧命令只有在 `ready`、`no_unlabelled_target` 且填有 `accepted_by` 时才输出 `confirmed_empty`，自动通过输出 `candidate_empty`。此处枚举描述旧报告的行为，当前生产从入库记录查询所需证据。[背景图入库与复用](../../docs/DATASET_DESIGN.md#背景图入库与复用)
- 输出默认位于 `datasets/annotated/common/review/<任务>/<批次>/background_pool.json`，记录来源计划、选择、度量、父组、近重复组、split 状态和代码摘要。

示例：

```powershell
.venv/Scripts/python.exe -m lab.scripts.common.cli background discover --contains backgrounds --limit 50 --task person --batch toolchain_20261115
.venv/Scripts/python.exe -m lab.scripts.common.cli background summarize --task person --batch v4_review_20261115_b5 --review-plan datasets/annotated/common/review/person/v4_review_20261115_b5/plan.json --review-selection datasets/annotated/common/review/person/v4_review_20261115_b5/selection.json --metrics datasets/annotated/common/review/person/v4_review_20261115_b5/metrics.jsonl
```

### 完整标签与掩膜

- `labels audit` 读取显式标注清单，检查类别顺序、可见框、完整框或掩膜以及来源，并按可见框重建 YOLO 标签。`confirmed_negative` 的空标签需要无任务实例、全帧检查和记录明确的审查方法。部分或未知标签仍按原范围记录。
- `masks audit` 复用 `datasets/scripts/common/review/masks.py` 的孔洞、贴边和多组件指标，再检查 visible 是否属于 full、隐藏区域是否由遮挡层解释、可见比例和地面接触点。默认叠加图同时绘制 full、visible、遮挡层、地面多边形和触点。
- 报告默认写入 `datasets/annotated/common/review/<任务>/<批次>/`；模型标签不是自动追加到正式数据集。

示例：

```powershell
.venv/Scripts/python.exe -m lab.scripts.common.cli labels audit --manifest lab/scripts/common/examples/person_label_sample.json --rebuild-labels datasets/annotated/common/review/person/toolchain_20261115/rebuilt_labels
.venv/Scripts/python.exe -m lab.scripts.common.cli masks audit --manifest lab/scripts/common/examples/person_mask_sample.json --overlay datasets/annotated/common/review/person/toolchain_20261115/mask_overlays
```

### Blender 机位与分支小样

- `cameras validate` 校验母场景文件与 Blender 登记库状态、相机目标投影、声明视角与计算视角、尺度分箱、机位覆盖标签和参考画面的地面单应重投影误差。它是离线投影校验，不打开 Blender，也不把拟合场景当成现场标定。
- `sample validate` 检查 direct camera、实拍合成、Blender 渲染等来源分支及空场、单人、多人、干扰等内容分支，核对每项父来源角色、标签状态、负例人工确认和遮挡证据。
- 默认校验只创建缺失的 `manual_review.json`；已有文件按 merge-preserve 增量补齐 schema 和缺失行，不覆盖任何人工值，发生写入前先备份非空人工内容。从零重建模板使用显式 `sample rebuild-review-template`。
- 人工按全帧框叠图核对类别、可见范围和遮挡关系；自然语言结论可归一化为 JSON 或 CSV 决策表后由 `sample normalize-review` 应用。坐标由候选预填或显式修正提供，未确认值保持 `pending`。

| 人工审查字段 | 通过时应写什么 |
| --- | --- |
| `objects[].category` | 实际类别，不使用任务外推断 |
| `objects[].visibility` | `full` / `partial` / `occluded` / `unknown` |
| `objects[].visible_bbox_xyxy` | 只覆盖图中可见像素 |
| `objects[].full_bbox_xyxy` | 有完整实体依据时填写；没有就保持空 |
| `background_state` | `confirmed_empty` / `contains_target` / `unknown` |
| `accepted_by` | 记录实际的人工语义确认人 |
| `objects[].box_accepted_by` | 单独确认可见框的人工；程序化值只进入自动验证层 |
| `review_state` | 默认 `pending`；`rejected/excluded` 只在有实际失败证据时填写 |
| `notes` | 缺标、遮挡层、截断、疑点与证据 |

全流程标注盘点报告 `source_available`、`annotated`、`candidate`、`auto_validated`、`human_confirmed`、`training_admitted` 与 `pending`。人工语义与框的确认范围分别记录，程序化结果保留机器来源。

示例：

```powershell
.venv/Scripts/python.exe -m lab.scripts.common.cli cameras validate --plan lab/scripts/common/examples/person_camera_plan.json
.venv/Scripts/python.exe -m lab.scripts.common.cli sample validate --manifest lab/scripts/common/examples/person_toolchain_sample.json
.venv/Scripts/python.exe -m lab.scripts.common.cli sample rebuild-review-template --manifest lab/scripts/common/examples/person_toolchain_sample.json
.venv/Scripts/python.exe -m lab.scripts.common.cli sample normalize-review --review datasets/annotated/common/review/person/toolchain_20261115/manual_review.json --answers datasets/annotated/common/review/person/toolchain_20261115/manual_review_answers.csv
.venv/Scripts/python.exe -m lab.scripts.common.cli sample export-label-manifest --review datasets/annotated/common/review/person/toolchain_20261115/manual_review.normalized.json --class-names community,noncommunity
```

## 本轮 person 小样验证

2026-11-15 使用 `lab/scripts/common/examples/person_toolchain_sample.json` 运行了有界小样校验。当前 6 项候选覆盖 `direct_camera_capture=3`、`real_image_composite=1`、`blender_render=2`，内容分支覆盖 `empty=1`、`single=1`、`multi=4`、`interference=1`；没有缺分支或来源角色错误，`human_accepted=5`、`programmatic_accepted=1`，整体为 `pass_candidate`。程序化项是 Blender 原始单帧预览，来源为确定性场景图与实例 ID 掩膜，不是人工验收，也不是数据集批次。

可重建报告和人工审查包位于：

- `datasets/annotated/common/review/person/toolchain_20261115/sample_validation.json`
- `datasets/annotated/common/review/person/toolchain_20261115/manual_review.json`
- `datasets/annotated/common/review/person/toolchain_20261115/label_audit.json`
- `datasets/annotated/common/review/person/toolchain_20261115/mask_quality.json`
- `datasets/annotated/common/review/person/toolchain_20261115/mask_overlays/`
- `datasets/annotated/common/review/person/person_arena_camera_candidates_20261115/camera_validation.json`

当前标签审计 `annotated=5`：4 条完整候选正例、1 条确认负例、10 个候选可见实例、0 条 unknown、0 条 partial，全部为 `pass_candidate`。人工语义接受覆盖 5 个审查项、16 个任务目标；人工框确认仍为 0。3 条正例框来自 `Codex visual review`，Blender 原始预览的 3 个框与掩膜来自 `programmatic_blender_generator`，合计 4 帧/10 框属于自动验证，框 pending 为 10，`training_admitted=0`。车辆遮挡裁片掩膜检查为 `review`：孔洞比例 0.03315 超过默认 0.02，且没有地面多边形；该证据说明现成裁片不能直接视为通过。机位计划无错误，前视/斜视/侧视为 3/1/1，相机尺度箱为 4/1；参考画面配准尚未声明，所以整体仍为 `review`。

该批人工审查表用 `background_state: confirmed_empty`、`full_frame_checked: true`、实际签署人和备注记录全图无目标结论。联系表使用 `panels`。人工框确认按实际状态记录，机器标签可按用途和证据使用。

背景池 b5 汇总为 `candidate_empty=5`、`rejected=5`、`unknown=9`、`confirmed_empty=0`；媒体发现额外返回 20 个 `unknown` 图片候选，仅用于后续全帧审查。

本轮新增辅助工具测试 16 项通过；现有 review/采样/透视/来源总账 55 项测试通过。`source_ledger check --task person` 仍报告 `BSL004` 错误 1 个（`occl_006` 证据缺 manifest），以及合格背景/遮挡父组不足、2 个 `contains_target` 缺可用目标标签状态的警告 3 个；这些是现有来源声明待决项，不是辅助脚本自动升级的问题。
