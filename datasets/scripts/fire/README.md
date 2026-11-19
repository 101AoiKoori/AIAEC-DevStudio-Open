# 火焰贴纸数据任务

本任务为 `fire`。检测指定火焰贴纸；楼面归属和逐楼计数另行记录。 下列描述用于导航，任务目标、标注范围与组合对照以配置中的 `fire` 条目为维护源。

当前图签已完成红橙硬负例、600张贴纸与干扰物同框正例以及Val/Test小目标对照；[首轮学习方案](config/first_round_learning.yaml)已落地。本轮保持全部分集规模，训练配方沿用现有正则化并适度增强颜色变化。后续按实际失败补数据、微调，不设置专项误报率前置验收。实际数量以[当前集合](../../../datasets/exports/fire/manifest.json)为准。

- [学习目标与标注范围](../../../config/task_targets.yaml)；[组合轴与对照](../../../config/combination_axes.yaml)：楼面、贴放位置以及零／单／多贴对照。
- [本任务数据配置](config/dataset.yaml)与[素材来源配置](config/source.yaml)：具体分支读取其支持的字段。
- [生产入口](./compose_real_facades.py)从当前逐图库编制直接实拍、实拍墙面贴图、完整 Blender 渲染及 Blender 与实拍场景混合合成。运行 `python -B datasets/scripts/fire/compose_real_facades.py --replace` 生成当前小样，加 `--plan-only` 编制小样和三个分集的执行计划，加 `--full --replace` 按[当前分配](config/full_dataset_plan.json)生成全量。实际产量见[当前生产进度](../../../docs/DATASET_PRODUCTION.md#当前生产进度)。
- [训练入口](../../../lab/features/fire/training/scripts/train_fire_sticker_v4.py)及[训练流程](../../../docs/TRAINING_WORKFLOW.md)：按任务配置和实际预算使用；脚本存在不代表模型已训练或完成交付。

数据制作使用同一 [dataset-production-loop](../../../.agents/skills/dataset-production-loop/SKILL.md)，无需按任务复制 Skill。

当前计划与实际图签对账运行 `python -B lab/scripts/contracts/state_coverage.py --actual --json`。实际数量读取当前清单及配对标签，保留分集、实例数、未记录字段和缺失文件；查询只读。

当前小样的 `manifest.json` 包含图签路径、来源帧和资格修订、实例楼宇 ID、作用域及逐楼可见数。整栋楼未完整观察时 `whole_building_count` 为 `null`。实拍裁区无贴只报告该裁区空负。Blender 的 `manifest.jsonl` 和 `instance_pass/` 保留可见像素证据，`validation.json`、`overlays/` 和 `contact_sheet.jpg` 用于图签核对。

父来源分集由当前逐图库维护，[分配入口](./prepare_sources.py)通过既有 `assign_selected` 应用任务选定记录；重复执行保留已分配身份。合成继承照片或父视频的分集，混合合成同时继承 Blender 母工程和实拍底图的父关系。当前母工程集中在 train。固定指定图案可跨分集使用，渲染视角和随机种子不作为新的独立母来源。全局执行计划仅登记于 `config/task_targets.yaml#tasks.fire.plans`，当前小样由 `source.yaml#sample_manifest` 定位。

混合合成读取 `source.yaml#hybrid_geometry_review` 中的楼角对应点、拟合相机和清晰地面区域。[相机拟合](./photo_geometry.py)结合约 0.4 米楼高及重建楼体比例搜索像素焦距并求解 PnP；手机 EXIF 缺少有效传感器尺寸，内参和无畸变假设均为近似，清单保留重投影误差。逐帧联合求取楼体尺度、朝向和地面占位，完整楼体须在视锥内，底部足迹须在已观察地面内。楼体与纸片使用同一刚体及统一尺度变换。

[混合渲染](./hybrid_render.py)使用 Cycles 实际灯光和 Shadow Catcher 地面生成 RGBA 图层，沿用照片中的弱漫射室内照明作近似；最后在线性光空间与等比例实拍底图合成。`render_layers/` 保留原始透明图层，`instance_pass/` 保留带场景深度的实例证据。渲染不保存或改写共享 `.blend`。相机拟合与光照匹配不代表已完成真实镜头和灯具标定。

运行 `python -B datasets/scripts/fire/audit_dataset.py` 独立复核小样，或加 `--full` 检查全量。该入口重新解码 ID 图、核对图签及逐楼分区，并逐像素重建混合图，检查精确重复和跨分集父来源。检查证据与拼图保存在 `datasets/annotated/fire/current_sample_review/` 或 `current_full_review/`，不作为训练图片。全量训练入口为 `datasets/exports/fire/data.yaml`，同目录 `workspace_sources.json` 保留机器复核证据、原父关系和开发接触。

全量总览按来源、贴数和视角选择代表图，所有图签及带框图继续逐图核对和生成。若图像制作完整而回执或预览收尾中断，可加 `--full --resume --replace`：入口核对原执行状态、来源修订及现存图签；渲染阶段复用还需 `source.yaml#completed_renderer_verification` 指向逐文件哈希与 ID 图独立复验证据。共享工程在运行期间更新时，先保留匹配原哈希的持久执行快照，在逐图清单登记 `scene_snapshot` 和原母工程哈希，不能把当前工程的新哈希写到既有渲染图上。既有回执保留其实际失败状态。

贴纸生成尺寸由 `source.yaml` 的 `sample_recipe.sticker_size` 维护，按各楼立面高度比例逐实例随机取样。实拍合成以逐图库立面多边形的投影高度作为墙面标定中心的尺度参照，Blender 使用对应前立面网格的实际高度。放大布局重新核对图内边界及贴纸间重叠，清单保存实际比例和相对原参考纸片的缩放倍数。

实拍纸片在逐图库登记的独立墙面坐标内取样，经该墙面的单应矩阵投影到原图，再转到输出裁区。墙面配准依据原图墙缝和竖直边线，尺度为近似；四角均须位于同一墙面边界内。C 楼转角两侧分别登记平面。Blender 纸片使用对应前立面平面和 `sample_recipe.wall_projection.blender_offset_m` 设置贴合距离，逐实例记录顶点平面误差。图签核对同时检查投影、墙面边界和纸片长宽比。

相机与渲染参数在 `source.yaml` 的 `camera_trajectory`、`sample_recipe` 维护。`Fire Buildings` 的 `Fire orange peel camera` 用 241 帧沿0.12／0.17／0.22 米三档低位／平视高度往返扫过前立面两侧 30°，生产渲染读取同一弧面位置。与楼名场景共用楼宇几何，批次按母工程和实际哈希追溯。

## 硬负例与颜色对照小样

`render_hard_negative_preview.py` 从同一母工程复制 A 楼与完整红桶，在后台内存创建 `Fire Hard Negatives` 场景，另建二灯信号灯和锥桶。楼面纸片使用原指定图案与 UV；求值后的曲线网格用于量取尺寸，纸片保持同一墙面，干扰物按地面缩放和摆放。共享工程不保存，检查用场景写入任务的 `annotated/fire/production_inputs/hard_negative_preview.blend`。

配方由 [hard_negative_preview.json](config/hard_negative_preview.json) 维护：红／绿灯、红／蓝桶、橙／灰锥桶分别配零贴和有贴，共 12 张。可调整随机种子、机位、物体位置、尺寸和转角、颜色亮度、粗糙度、曝光与光源强度。同物体的颜色对照保持几何和机位，有无贴纸对照保持干扰物和光照；颜色仍参与识别，不把指定贴纸整体去色。其他颜色控制样本不充作红橙硬负配额。

```powershell
& $DatasetBlender --background datasets/assets/blender/aiaec_dataset.blend --python-exit-code 1 --python datasets/scripts/fire/render_hard_negative_preview.py
.venv/Scripts/python.exe -B datasets/scripts/fire/audit_hard_negative_preview.py
```

复跑时在 Blender 命令末尾加 `-- --replace`，仅替换该小样清单内生成物。图签位于 `datasets/samples/fire/hard_negative_preview/`；独立掩膜、图签、逐楼归属及成对元数据核对位于 `datasets/annotated/fire/hard_negative_review/`。实拍视觉复核保存在同目录 `real_visual_review.jsonl`，确认的红桶帧通过既有逐图库写入 `fire/hard_negative`；未出现相近色干扰物的路口帧继续归普通负例。AI 视觉复核以 `machine` 来源记录，干扰物框只作上下文，不新增检测类别。

这些场景均继承 train Blender 母来源。开发小样保留作颜色对照；空负批量替换已另行接入当前集合，尚未训练或评估误报率。

## 空负例批量替换

运行 `.venv/Scripts/python.exe -B datasets/scripts/fire/compose_real_facades.py --replace-empty-negatives` 使用 [negative_replacement.json](config/negative_replacement.json)。入口复用独立渲染阶段，先核对新图、空标签与可见干扰物掩膜，再替换当前集合内选定的完整空负图签；所有保留文件逐一校验哈希。退出图、空标签、预览、失效掩膜和混合图层按原清单清理，退出记录保留在 `annotated/fire/hard_negative_replacement/`。现行三个 `full_*_plan.json` 按保留状态和新增状态重新核数，原 `generation_plan` 及历史回执保留生成时输入。

train 使用同母工程的参数化红灯、红橙桶、红橙锥桶；val/test 使用各自父视频的已视觉确认红桶帧及两档温和亮度派生。帧、亮度变体和独立父视频分别记录，不能将派生数量解释成独立采集；val/test 的真实硬负覆盖目前限于红桶。负例继续保留同名空 YOLO 文件，指定贴纸正例保持原图签。

`--full` 在现有硬负替换集合上复用当前结果；新建基础批次后自动接入已登记的替换入口。`--foundation-only` 明确选择仅重建来源与几何基础批次，不执行替换。`--plan-only` 对已替换集合只报告当前完整计划，避免把基础配额重新写回现行计划。独立核对仍用 `audit_dataset.py --full`，训练入口读取当前 `workspace_sources.json`；600张正例同框干扰及Val/Test各90张小目标替换由[training_preparation.json](config/training_preparation.json)维护，调用 `prepare_training_data.py`，总量与每图贴纸数保持不变。

## 实拍垃圾桶与小贴纸微调数据

当前入口为 `python -B datasets/scripts/fire/select_compact_finetune.py`，从完整候选池选取[精简专项计划](config/reshoot_compact_finetune_plan.json)的数据，输出 `training_packages/fire_sticker/dataset_finetune_compact/`。Train 为600张原数据回放和624张新增数据，Val/Test各300张沿用；图片不重生成，所有入选图签保留原字节。回放包含500张正例、100张负例，按来源、楼宇、贴纸数、尺寸与父源分层，保留全部直接实拍。配置固定10轮，预热1轮，显式设置 `warmup_bias_lr: 0.0`，主学习率0.00004；本轮不启动训练。

原 `dataset_finetune/` 保留为完整候选池，其[生成配方](config/reshoot_finetune_plan.json)与原回执保持不变，不计入当前微调计划。`build_reshoot_finetune.py` 继续负责候选池制作；重建候选池不会覆盖当前精简训练配置。精简入口加 `--verify` 可重查图签、父源及候选池哈希，不训练。

补拍按 `bins_reshoot_20261118` 整个父批次归 Train，新增正例只使用原Train已配准楼面。逐贴保留楼宇、墙面、四边形与可见框；实拍桶透视和接触阴影继续沿用候选池的近似声明。Val/Test已用于开发诊断，作为已知来源回归集合。

[当前精简清单](../../../training_packages/fire_sticker/dataset_finetune_compact/manifest.json)及[核对回执](../../../training_packages/fire_sticker/dataset_finetune_compact/verification.json)记录实际规模、标签与逐楼核对；包内 `config/finetune.yaml` 和工作空间对应训练配置消费同一精简入口。基础数据入口及原始权重保留。

<!-- TASK_DESIGN:START -->

## 学习目标与组合设计

本节由共享配置生成；修改目标与组合请编辑 [task_targets.yaml](../../../config/task_targets.yaml) 和 [combination_axes.yaml](../../../config/combination_axes.yaml)，运行 `python lab/scripts/contracts/state_coverage.py --write`。

**学习及标注**

- 类别：fire_sticker
- 数量：zero、single、multi
- 负例：zero

标出指定火焰贴纸的可辨实例；区域覆盖不全或已有标注被抑制，不能据空文件判断零贴。

零／单／多贴是覆盖；同框干扰物与指定贴纸分别记录，干扰物也可出现在有贴正例中。首轮同时学习硬负例与正例，后续按实际失败补数据和微调，不以未测误报率作为制作或首轮训练的普遍前置条件。按实际归属分楼计数。

**素材用途**

- `direct_target`：全帧至少一张在任务识别范围内可辨的指定贴纸，同范围内各实例标签完整。逐楼报数另需楼面覆盖及归属。
- `target_texture`：唯一指定火焰图案和物料尺寸，用于实拍立面贴图及 Blender。
- `hard_negative`：已确认范围内无指定贴纸的红灯／橙色灯、红橙垃圾桶、锥桶、红橙面板、其他相似图案或真实火焰／烟雾；颜色相似不构成正例。

**组合轴与对照**

以下是本任务的设计条件；`dynamic` 表示依据实际素材和任务计划确定。

- `facade`：dynamic
- `position`：dynamic
- `distractor`：none、red_signal_lamp、red_orange_bin、other_red_orange_object

**布局**：指定输出裁区、楼面 ID 和贴放四边形；保留已有贴纸及各自楼面归属。

**shape_color_contrast**：识别指定火焰图案的结构和边界，红橙颜色及亮区本身不能代替目标身份。

- 安排：同背景对照仅有干扰物与贴纸加干扰物；指定贴纸跨楼面位置，红灯、红橙桶及其他红橙物独立变化大小、位置和明暗。
- 核对：核对当前图签及干扰类型元数据；无指定贴纸的画面为空负，有贴画面保留全部指定贴纸框。首轮先共同学习，专项误报率验证后续再做。

**facade_position**：楼面身份、固定贴放位或红橙色物体不能代替指定火焰图案。

- 安排：同楼面配零贴与有贴，同一图案跨多个楼面及可行贴放位；另补确认无指定贴纸的相似图案或火焰／烟雾。
- 核对：用 source_image、楼面与实际贴放位置连接框；同一楼面查有无两格，硬负查真实内容及完整空标签。

**building_counts**：一张图或一栋楼不能成为固定报数的线索。

- 安排：同楼面配单贴和多贴，两楼同框时改变各楼贴数并保留含零贴楼的组合。
- 核对：每图每楼按实际有效实例与 building 归属计数；请求贴数、图片数和被挡后可标实例数分别核对。

对照描述来自设计配置；结构化记录目前为部分编码，具体制作按任务入口和可用素材安排。

<!-- TASK_DESIGN:END -->
