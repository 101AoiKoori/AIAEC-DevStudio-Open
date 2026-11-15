# 人工物体审查矩阵

`manual_review.json` 中每张图片对应一个 `status_matrix`。`objects` 是数组，不是单个对象：图片里有几个人、几个背面立牌或几个干扰物，就填写多少行。已知实例数和候选框时，生成器预填稳定行及候选 `visible_bbox_xyxy`；人工只需按全帧框叠图核对类别、可见范围和遮挡关系，再修正错误框。

审查状态分层且独立计数：`source/available`、`annotated`、`candidate`、`auto_validated`、`human_confirmed`、`training_admitted`。未人工确认的内容保持 `pending`；缺少 `accepted_by`、未走人工复核或尚未准入都不等于 `rejected`/`excluded`。程序化结果只进入 `auto_validated`，不能填写人工确认层。人工语义由 `accepted_by` 确认，框由 `objects[].box_accepted_by` 单独确认。

## 图片级字段

| 字段 | 类型 | 填写值 |
| --- | --- | --- |
| `presence` | enum | `none` / `single` / `multi` / `unknown` |
| `task_target_count` | integer/null | 任务目标实际数量；不确定时写 `null` |
| `background_state` | enum | `confirmed_empty` / `contains_target` / `unknown` |
| `reviewer` | string/null | 实际审查人；空值不表示通过 |
| `accepted_by` | string/null | 人工语义确认人；与 `reviewer` 兼容，但计数以此字段优先 |
| `full_frame_checked` | boolean | 确认逐处看过整图后写 `true` |
| `review_state` | enum | `pending` / `confirmed` / `rejected` / `excluded`；后两者必须有实际失败证据 |
| `notes` | string/null | 漏标、截断、遮挡、疑点或补充证据 |
| `objects` | array | 每个独立物体一行 |
| `other_objects` | array | 人物立牌以外、但仍影响画面的物体；结构可与 `objects` 相同 |

## 物体行字段

| 字段 | 类型 | 标准值 |
| --- | --- | --- |
| `row_id` | string | 本图稳定行标识，如 `obj_001` |
| `left_to_right_order` | integer/null | 从左到右顺序；不确定时写 `null` |
| `category` | enum | `community_person` / `noncommunity_person` / `person_standee_back` / `person_standee_nonfront` / `vehicle` / `other` / `unknown` |
| `view` | enum | `front` / `back` / `side` / `oblique` / `unknown` |
| `visibility` | enum | `full` / `partial` / `cropped` / `occluded` / `unknown` |
| `occluder` | enum | `none` / `person` / `vehicle` / `object` / `unknown` |
| `is_task_target` | boolean/null | 该物体是否属于任务目标；不确定写 `null` |
| `annotation_action` | enum | `label_visible_box` / `hard_negative_no_box` / `do_not_label` / `unknown` |
| `visible_bbox_xyxy` | array/null | `[x1,y1,x2,y2]`，只覆盖图中可见像素；框待确认时保持 null 并计 pending |
| `full_bbox_xyxy` | array/null | 有完整实体依据时的 `[x1,y1,x2,y2]`；必须包含可见框，没有依据保持空 |
| `box_accepted_by` | string/null | 实际确认该框的人；程序化名称只进入自动验证层 |
| `notes` | string/null | 该物体的事实说明 |

`person_standee_back` 只表示能看到完整白色背面轮廓的立牌。只看到很薄的侧边、斜侧面或近乎边缘视角时使用 `person_standee_nonfront`，并把 `view` 写为 `side` 或 `oblique`。两者都不是当前正面人物任务的目标，`annotation_action` 用 `hard_negative_no_box`，`visible_bbox_xyxy` 与 `full_bbox_xyxy` 保持 null。

## 标准行

```json
{
  "row_id": "obj_002",
  "left_to_right_order": 2,
  "category": "community_person",
  "view": "back",
  "visibility": "full",
  "occluder": "none",
  "is_task_target": false,
  "annotation_action": "hard_negative_no_box",
  "visible_bbox_xyxy": null,
  "full_bbox_xyxy": null,
  "box_accepted_by": null,
  "notes": "正面人物立牌的白色背面，不输出人物框"
}
```

## 多人图片

```json
{
  "presence": "multi",
  "task_target_count": 2,
  "background_state": "contains_target",
  "reviewer": "实际审查人",
  "accepted_by": "实际审查人",
  "full_frame_checked": true,
  "review_state": "confirmed",
  "notes": null,
  "objects": [
    {
      "row_id": "obj_001",
      "left_to_right_order": 1,
      "category": "community_person",
      "view": "front",
      "visibility": "full",
      "occluder": "none",
      "is_task_target": true,
      "annotation_action": "label_visible_box",
      "visible_bbox_xyxy": [80, 62, 132, 214],
      "full_bbox_xyxy": null,
      "box_accepted_by": "实际审查人",
      "notes": null
    },
    {
      "row_id": "obj_002",
      "left_to_right_order": 2,
      "category": "community_person",
      "view": "back",
      "visibility": "full",
      "occluder": "none",
      "is_task_target": false,
      "annotation_action": "hard_negative_no_box",
      "visible_bbox_xyxy": null,
      "full_bbox_xyxy": null,
      "box_accepted_by": null,
      "notes": "白色背面，不输出人物框"
    },
    {
      "row_id": "obj_003",
      "left_to_right_order": 3,
      "category": "noncommunity_person",
      "view": "front",
      "visibility": "partial",
      "occluder": "vehicle",
      "is_task_target": true,
      "annotation_action": "label_visible_box",
      "visible_bbox_xyxy": [330, 60, 385, 213],
      "full_bbox_xyxy": [327, 55, 388, 231],
      "box_accepted_by": "实际审查人",
      "notes": "下缘被电动车挡住，只框可见区域"
    }
  ],
  "other_objects": [
    {
      "row_id": "other_001",
      "left_to_right_order": 4,
      "category": "vehicle",
      "view": "side",
      "visibility": "full",
      "occluder": "none",
      "is_task_target": false,
      "annotation_action": "do_not_label",
      "visible_bbox_xyxy": null,
      "full_bbox_xyxy": null,
      "box_accepted_by": null,
      "notes": "遮挡第3个人物下缘"
    }
  ]
}
```

## 空场图片

```json
{
  "presence": "none",
  "task_target_count": 0,
  "background_state": "confirmed_empty",
  "reviewer": "实际审查人",
  "accepted_by": "实际审查人",
  "full_frame_checked": true,
  "review_state": "confirmed",
  "notes": "已逐处确认整图没有社区或非社区正面人物目标",
  "objects": [],
  "other_objects": []
}
```

`unknown` 不等于 `none`。只填写“没人”但没有完成全图检查时，保持 `background_state: unknown`。

## 当前小样预生成行数

| 记录 | `objects` 行数 | 说明 |
| --- | ---: | --- |
| `empty_candidate` | 0 | 等你确认是否为空场 |
| `real_composite_single` | 1 | 预生成 `obj_001` |
| `direct_multi` | 3 | 预生成 `obj_001` 至 `obj_003` |
| `direct_interference` | 3 | 预生成三行，并可把车辆加入 `other_objects` |
| `blender_overlay_candidate` | 0 | 实例数未知，按图片实际内容追加 |
| `blender_raw_preview` | 4 | 程序化生成：3 个正面目标加 1 个白色背面；只进入 `auto_validated` |

审查文件位于 `datasets/annotated/common/review/person/toolchain_20261115/manual_review.json`。

`blender_raw_preview` 的对象、可见框和掩膜由 Blender 场景图及实例 ID 掩膜确定性生成，接受字段为 `programmatic_blender_generator`，不计入人工验收。其证据为同批次 `blender_raw_preview_evidence.json`、`blender_raw_preview_mask_manifest.json` 和 `blender_raw_preview_mask_quality.json`。

## 拼图或接触表

一张图片包含多个独立面板时，不能在同一层反复写 `objects1,1`、`objects1,2`。使用 `contact_sheet`，每个面板拥有自己的 `panel_id` 和完整状态矩阵：

```json
{
  "image_role": "contact_sheet",
  "reviewer": "实际审查人",
  "accepted_by": "实际审查人",
  "review_state": "confirmed",
  "notes": null,
  "panels": [
    {
      "panel_id": "row1_col1",
      "source_name": "images/a_person_arena_00001.png original",
      "presence": "multi",
      "task_target_count": 3,
      "background_state": "contains_target",
      "reviewer": "实际审查人",
      "accepted_by": "实际审查人",
      "full_frame_checked": true,
      "review_state": "confirmed",
      "objects": [],
      "other_objects": [],
      "notes": null
    }
  ]
}
```

检查命令：

```powershell
.venv/Scripts/python.exe -m lab.scripts.common.cli sample check-review --review datasets/annotated/common/review/person/toolchain_20261115/manual_review.json
```

`sample validate` 默认只增量创建缺失模板；已有审查文件只做 merge-preserve，新增字段和预生成行不会覆盖任何人工值。需要从头重建时使用单独命令，非空人工内容会先自动备份：

```powershell
.venv/Scripts/python.exe -m lab.scripts.common.cli sample rebuild-review-template --manifest lab/scripts/common/examples/person_toolchain_sample.json
```

自然语言结论由审查负责人归一化为明确字段后，可写成 JSON 或 CSV 决策表。CSV 每个物体一行，帧级字段可重复；工具只应用显式值，不替用户猜测框或确认人。归一化产物默认另写文件，不覆盖源审查：

```powershell
.venv/Scripts/python.exe -m lab.scripts.common.cli sample normalize-review --review datasets/annotated/common/review/person/toolchain_20261115/manual_review.json --answers datasets/annotated/common/review/person/toolchain_20261115/manual_review_answers.csv
```

单帧审查可转换为显式标签清单。人工语义已确认但框未由人工确认时，只得到 `partial`；全部发布框都有非自动 `box_accepted_by` 后才可成为 `complete`。联系表需要面板几何，暂不直接导出为单帧 YOLO：

```powershell
.venv/Scripts/python.exe -m lab.scripts.common.cli sample export-label-manifest --review datasets/annotated/common/review/person/toolchain_20261115/manual_review.normalized.json --class-names community,noncommunity
```

校验通过后，把人工判断固化为任务配置中的接受记录。记录包含源审查路径、SHA-256、图片级状态和全部对象行；之后即使忽略目录重建，人工语义结论仍可从 Git 追踪：

```powershell
.venv/Scripts/python.exe -m lab.scripts.common.cli sample accept-review --review datasets/annotated/common/review/person/toolchain_20261115/manual_review.json --output datasets/scripts/person/config/toolchain_review_acceptance.json
```

接受记录的 `status_layers` 同时报告声明可用、已标注、候选、自动验证、人工确认、训练准入和 pending。`accepted_by` 只确认人工语义，`box_accepted_by` 只确认框，`training_admitted` 由后续数据打包层另记。缺少确认保持 pending；只有记录明确给出 `review_state: rejected/excluded` 且有失败证据时才计入排除。
