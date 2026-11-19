# 全局辅助脚本 `common`

本目录是跨任务辅助脚本的实现位置。能力范围、放置规则和输出边界以 [`../README.md`](../README.md) 为准。

脚本按稳定职责建立独立模块或子目录；不按任务复制整套代码，也不把任务专用生成逻辑放到这里。

| 模块 | 当前职责 | 复用的现有能力 |
| --- | --- | --- |
| `background_pool.py` | 媒体缓存候选发现；review 判定的保守背景状态汇总；父组/split 预留证据 | `review.evidence`、现有 `review_selection.json`、各任务 `source.yaml#pairing` |
| `labels.py` | 完整/部分/未知/确认负例的标签审计与 YOLO 重建 | `class_yolo_row` 的框语义、现有图片与标注清单；不替换任务入口 |
| `mask_quality.py` | full/visible/occlusion 图层、孔洞、贴边、触点与叠加证据 | `review.masks.alpha_quality` |
| `blender_cameras.py` | 母场景机位、投影/视角/尺度覆盖、参考画面配准 | `lab/scripts/blender/blender_source_registry.py`、`lab/training_common/ground_placement.py` |
| `branch_sample.py` | direct/real composite/Blender 与空场/单人/多人/干扰分支、来源链和人工审查包 | 现有候选清单与标签状态约定 |
| `contracts.py` | 工作空间路径、JSON 原子写入、代码摘要和共享枚举 | 新写的薄接口层，不复制 review 算法 |
| `cli.py` | `python -m lab.scripts.common.cli` 稳定入口 | 调用上述模块 |

`background discover` 不把媒体缓存命中当成背景；`background summarize` 也不把 `ready` 自动升级为确认空场。人工确认仍是独立字段，未确认状态保持 `unknown` 或 `candidate_empty`。

空标签文件同样不是负例。`labels audit` 只有收到 `confirmed_negative`、`review.full_frame_checked=true`、非空 `review.accepted_by` 和审查方法时才接受；未知、部分和未人工确认项都保持 pending。摘要分别报告 source/available、annotated、candidate、auto_validated、human_confirmed 和 training_admitted。程序化接受只进入 auto_validated；人工语义确认与框/掩膜确认分开计数。

`sample validate` 重跑时默认合并已有 `manual_review.json`：同 ID 项保留人工字段，只补模板缺项，并加入新项。`sample rebuild-review-template` 才允许替换模板，且已有内容先备份。

本轮真实 person 小样的可重建验证与人工审查包位置记录在 [`../README.md`](../README.md#本轮-person-小样验证)。机位校验不连接 Blender；核对当前打开场景或渲染单帧时使用可用的 Blender MCP，预览产物与正式数据批次分开。
