# 2026-11-12 比赛场地补录批次

> 本文提及的部分本地素材、数据和历史运行产物未随本仓库提供；相关名称保留用于说明来源和实验背景。

本批次收录用户补充的两段比赛场地视频和 160 张小车摄像头原图，共 3,096,193,369 字节。`onboard_camera_images/` 指拍摄设备是小车摄像头，照片跨多个识别任务；文件名中的 `vehicle_` 是保留的批次资产 ID，不代表电动车或车牌类别。批次日期为整理日期。用户确认小车摄像头的实际分辨率为 640×480，160 张车载 PNG 全部解码通过且均为 640×480。两段视频的拍摄设备和相互场次关系待核对。文件修改时间、原名里的时间和观察点编号保留作来源线索。

| 原件 | 批次内文件 | 尺寸与时长 | 抽帧结果 |
| --- | --- | --- | --- |
| `补录视频-全场景.mp4` | `videos/full_scene.mp4` | 3840×2160，379.480 秒 | 23 段，339 帧 |
| `人物立牌补拍.mp4` | `videos/person_standees.mp4` | 1280×720，50.062 秒 | 12 段，49 帧 |
| `小车实拍照片/` | `onboard_camera_images/vehicle_0001.png` 至 `vehicle_0160.png` | 640×480，160 张 PNG | 160 张照片已逐张看缩略图 |

原件归入 `raw/site_capture_20261112/`，改名后复核了文件数量、逐文件字节数和修改时间。照片 ID 按原名排序分配。原名含点位和时间时在来源清单中单列，实际采集次序和标注以核实记录为准。

## 处理记录

本地处理区位于 `annotated/site_capture_20261112/`，由 [批次清单](annotated/site_capture_20261112/batch_manifest.json) 关联以下产物。

| 文件 | 记录内容 |
| --- | --- |
| [来源清单](annotated/site_capture_20261112/source_inventory.csv) | 162 个原件的资产 ID、原名、原路径、当前路径、字节数、设备和文件名线索 |
| [照片初筛](annotated/site_capture_20261112/photo_review.csv) 和 [照片联系表](annotated/site_capture_20261112/photo_contact_sheets/) | 160 张照片的当前路径、主要画面和初筛备注，其中 31 张记录了需关注的情况。包括出画、遮挡、相近构图、软焦、背景真人及点位与主体不符；内容是缩略图观察候选，不是背景可用性或任务真值审核 |
| [背景观察候选](annotated/site_capture_20261112/background_source_manifest.jsonl) | 160 张图逐张记录观察到的任务候选、近似视角和各任务观察状态；158 张有可见地面锚点多边形，2 张没有地面，另有 2 张楼面透视参考。12 张区域参考原图，其余为缩略图估计。锚点尚未验证完整目标范围、尺度和已有目标标签，不能直接作为合成准入。 |
| [地面几何候选](annotated/site_capture_20261112/ground_geometry_manifest.jsonl) 和 [任务实例候选](annotated/site_capture_20261112/target_instances_manifest.jsonl) | 160 图中 117 张有局部地面候选，29 张可关联旧候选框；目标实例记录包含红绿灯、车牌等附属任务候选框，天花板图未确认任务目标；实例数和被排除旧框以清单为准。已知框非穷尽，区域和框不能直接作为完整训练真值。 |
| [可见地面局部观察](annotated/site_capture_20261112/ground_surface_manifest.jsonl) | 分别保存含地面的参考 ROI 与逐图观察到的局部地面多边形；完整地面边界、目标占用和放置资格以每条记录状态为准。矩形 ROI 不作为地面分割或空白掩膜。 |
| [地面接触区域候选](annotated/site_capture_20261112/placement_ground_visual_manifest.jsonl) | 160 张逐图复核，15 张有可供人物或电动车接地点判断的局部候选；其余保持未知。目标完整轮廓、尺寸和透视尚未验证，不能直接自动合成。 |
| [实例增量复核](annotated/site_capture_20261112/instance_refinement_manifest.jsonl) | 160 图主要可见候选已按原分辨率叠框核对，增量只记录原清单漏项并引用已有证据；远小、画面边缘和不可判目标仍可能遗漏，不代表所有任务实例均穷尽。 |
| [贴合区域候选](annotated/site_capture_20261112/placement_refinement_manifest.jsonl) | 28 张主楼或桶面照片记录了 70 个局部空白候选，保留坐标、适用任务、尺寸限制及观察依据；未列区域为未知。像素区域小于或接近目标需求时须重新判断可辨性与透视。 |
| [定性视角观察](annotated/site_capture_20261112/view_geometry_manifest.jsonl) | 160 张基于缩略图记录指定主体表面或道路视角：正视 73、轻斜 14、明显斜视 7、道路透视 64、未知 2。分类适用于记录的表面，不能直接推断其他目标或相机实际角度；yaw、pitch、roll 未测量。 |
| [视频分段表](annotated/site_capture_20261112/segments.csv) 和 [视频联系表](annotated/site_capture_20261112/video_contact_sheets/) | 35 个区段的起止、抽帧间隔、可见内容和角色。全场景约每 4 秒、人物视频约每 1 秒核对缩略图，分段边界为近似值 |
| [逐帧索引](annotated/site_capture_20261112/extract_v1/frames.csv) 和 [抽帧汇总](annotated/site_capture_20261112/extract_v1/summary.json) | 388 张保持源帧尺寸的 JPEG，含原视频路径、区段、原始 PTS、时间基准、相对时间和区段状态 |
| [车载模糊分析](annotated/site_capture_20261112/blur_analysis.json) | 160 张图的清晰度统计、同点位配准结果和局部边缘比较 |

`photo_review.csv` 已加入 `background_view`、`background_placement_plane`、`background_target_presence`、`background_use`、`background_review_state` 与 `mean_luma_0_255`、`mean_saturation_0_255`。后两列按整图灰度和 HSV 饱和度均值测得，量纲为 0–255；前五列已依据 10 张联系表逐张填写缩略图观察：158 张为条件可用候选，2 张因挡板或天花板画面不适合作为常规场地背景；标记为 `visual_candidate`，具体任务的目标和贴放区域仍需核对。`observed_content` 是缩略图内容初筛；库中可按内容及数值范围检索候选，不能把该初筛当成无目标背景或可贴合平面真值。更新统计用 `.venv/Scripts/python.exe lab/scripts/media/update_site_capture_background_features.py`，刷新索引用 `python lab/scripts/media/media_index.py build --source datasets/annotated/site_capture_20261112` 和 `python lab/scripts/media/media_index_sqlite.py sync`。

视频覆盖楼宇、楼名和火焰贴纸、温湿表、四色桶、人物立牌、电动车、车牌、禁行牌和灯杆。转场、高位俯视与目标段在分段表中标明。裁切、反射和遮挡按各任务复核；人物仅标近距正面可辨目标，白背面作为不标框的硬负例，侧背、远处及不可判人物不作正例。抽帧和照片初筛均待标注及真值核验，train／val／test 尚未划分。保留按视频和已知采集批次归组的关系，同一原视频的抽帧沿用该视频来源。

同场地评估初版使用 60 张原清晰度代表图，按任务记录候选标注、重复及待复核样本。已有生产权重的结果见[实拍诊断](../docs/research/FIELD_EVALUATION_V3_20261112.md)，本批素材的用途按[V4 数据制作方案](../docs/DATASET_DESIGN.md#v4-八项任务数据集设计)确定；各模型重训或微调尚未确定。

## 下一版序列与片段处理

现有 388 帧是区间抽样，35 段是粗分段。纳入的视频完整解码，按不超过 720p 保存母序列，随后细分连续事件片段，按任务和状态分配 train／val／test。同一原视频允许跨集合使用不同片段。`sequence_v2/person_standees/` 已从 1280×720 原视频解码 1,502 帧 PNG，逐帧时间戳及状态见该目录清单；尚未标注或切分。全场视频按面积滤波等比例降到 640×360，完整解码 11,378 帧 PNG，首末时间为 0 和约 379.446 秒；逐帧时间戳递增，抽查 7 帧可解码且尺寸正确，状态及时间戳记录在 `sequence_640_v1/full_scene/`。原视频保留，两份序列均未标注或切分。边界缓冲、重复关系、候选真值、主体和背景谱系统一按[制作细则](../docs/DATASET_PRODUCTION.md#多原件取样与场景混合)。另已入库的 [CarRealPicture 车载批次](CAR_REAL_CAPTURE_20261112.md)包含连续帧，作为 V4 核心来源与本批共同规划。

`fire_name_v4_review/` 保存 9 张火焰和楼名逐任务目视小样、独立标签、叠图与来源清单。其中 5 张全场视频帧尚未分配集合，另有 2 张视频帧处于旧诊断帧的 0.25 秒缓冲内、2 张为车载旧诊断图，均只用于开发回归。火焰为 4 正、4 负、1 部分，楼名为 3 正、5 负、1 部分；两次 AI 目视复核未发现明确漏标或将红桶、灯具误标为火焰，部分状态和小目标边界仍待人工核实。来源门控读取本批已有诊断选图及视频时间，排除已接触帧及其 0.25 秒时间缓冲；更远的近重复仍须逐项核查。

`person_vehicle_v4_review/` 保存 7 张人物与电动车小样及独立叠图，人物完整候选有社区 12 框、非社区 2 框，电动车完整候选有直立 4 框、倒伏 3 框；3 条电动车记录为部分状态。白背只作硬负例参考。3 张为旧诊断或缓冲内回归，其余 4 张未分配集合。[审核清单](annotated/site_capture_20261112/person_vehicle_v4_review/review_manifest.json)

`sign_plate_v4_review/` 保存 11 张禁行牌和车牌小样；禁行牌 3 正、6 负、2 部分，车牌 5 正、4 负、2 部分，分别含 3 和 7 个完整候选框。1 张为旧诊断回归，其余 10 张未分配集合。两次 AI 目视复核未发现确定的漏标或误标，远小目标和疑似反射保留部分状态。[审核清单](annotated/site_capture_20261112/sign_plate_v4_review/review_manifest.json)

`fire_name_v4_review/batch2/` 新增 5 张全场景视频帧，全部为开发回归：火焰和楼名各为 1 正、3 负、1 候选部分，共 4 个候选框；10 条图像—任务记录均为 AI 复核，`human_accepted: false`。[审核清单](annotated/site_capture_20261112/fire_name_v4_review/batch2/review_manifest.json)

`sign_plate_v4_review/batch2/` 新增 9 张全场景视频帧，其中 1 张旧诊断回归、8 张未分配：禁行牌 0 正、6 负、3 部分；车牌 1 正、6 负、2 部分，共 3 个候选框。18 条图像—任务记录均为 AI 复核，`human_accepted: false`。[审核清单](annotated/site_capture_20261112/sign_plate_v4_review/batch2/review_manifest.json)

上述五个小样清单共覆盖 41 张图、82 条图像—任务记录；`human_accepted` 均为 `false`。这些记录是 AI 目视候选，`false` 表示人工确认 pending，不表示 `rejected`／`excluded`；标签和实体、镜像、近重复关系按清单继续核实，不能把尚未分配的样本当作独立测试。

### 小样标注层

本表只统计五个小样清单；层定义见[标注与准入状态层](../docs/DATASET_PRODUCTION.md#标注与准入状态层)。

| 层 | 当前数量与边界 |
| --- | --- |
| `source_available` | 41 张图可访问；不是完整补录批次的 162 个原件总数。 |
| `annotated` | 66 条图像—任务记录已有标签文件。 |
| `candidate` | 82 条图像—任务记录、64 个候选框；其中 51 个框位于完整正例记录，13 个框位于 `partial` 记录。 |
| `auto_validated` | unknown；现有“两次 AI 目视复核”仍是候选审查，没有声明独立的自动验收层。 |
| `human_confirmed` | 语义 0；框／空场 0；掩膜 0。 |
| `training_admitted` | 0。 |
| pending | 82 条记录尚待人工语义和框／空场确认；`partial` 14 条、`candidate_partial` 2 条继续 pending，保留原记录。 |

本次没有覆盖人工复核写回。后续重跑 `review_manifest.json`、人工记录或模板时，必须 merge-preserve；默认校验只读，模板／整份手工记录重建须显式请求、检测非空人工内容并先备份。相关脚本尚未在本批次执行重写验证，不能据文档推定写回实现已经完成。

对上述小样及桶小样的 20 张未分配图片，按 96×72 灰度缩略图均差不超过 14 且相关系数不低于 0.975 筛查，分别与 60 张旧诊断图和彼此的 190 对比较，均未触发候选。筛查记录为 `v4_candidate_diagnostic_similarity.json` 和 `v4_candidate_pair_similarity.json`；该阈值筛查不能证明没有近重复或来源独立，片段和物理素材复用仍须核查。

素材盘点生成了 `multisource_review_0.jpg` 至 `multisource_review_3.jpg` 四张联系表，位于本批次处理区，内容为已有楼宇和桶候选缩略图。它们用于定位待审素材，未形成审核通过的多源清单。

## 车载模糊与增强候选

在 640×480 原图上，13 对照片通过严格的同点位配准。其中 9 对与较清晰参考的差异在搜索网格上对应附加高斯 σ 0 px，4 对对应 σ 0.35 px。另挑 6 对同点位明显偏软的画面，比较同类局部边缘，横向差异等效于附加高斯 σ 0.49–0.83 px，或长度约 1.7–2.9 px 的均匀线核。后者按核的方差换算，样本刻意挑选偏软画面，实际成因和出现频率待核。全图拉普拉斯方差随构图、纹理和曝光而变，批次最低值包括挡板与天花板画面。

训练增强以车载 640×480 为像素尺度。已有实拍模糊保留，增强从清晰底图或渲染图按概率采样。每次按一种模糊类型采样，方向覆盖横向、斜向及纵向。建议起点如下，比例和范围仍待验证。

| 增强档 | 起始占比 | 附加高斯 σ 或方向拖影长度 |
| --- | ---: | --- |
| 原始 | 70% | 保留原图清晰度 |
| 轻 | 20% | 0.35–0.7 px 或 2–3 px |
| 中 | 8% | 0.7–1.2 px 或 3–5 px |
| 强 | 2% | 1.2–1.6 px 或 5–7 px，作为本批实测范围以外的泛化对照 |

同时采样符合比赛机位的目标尺度、斜视、局部遮挡、明暗和反光变化。楼名、车牌文字和仪表细节以原始与轻档为主，模糊后读数、字符或关键点无法可靠确认的样本按任务标记不可读或不可见。效果用已留出的同场地原始来源组和冻结回归图检验，按清晰度、拍摄距离和任务分别报告漏检、误检或读数错误。以上为旧照片批次的增强起点，本批尚无增强训练对照结果。新增车载视频的运动和静止片段用于复核各任务模糊分布，不能将本表的比例直接视为新路线实测频率。

## 按区段重跑抽帧

从仓库根目录运行以下命令，PATH 须有 FFmpeg 和 FFprobe。修改分段时以原视频开始为零点，按秒填入起始、结束和间隔。区间含起点并排除终点。每个区段从第一张可用帧起按原始时间戳取样，导出目录须为新目录。保留较密区段时可以缩短该段间隔，每次运行上限为 2,500 张。

```powershell
python -B lab/scripts/media/extract_capture_frames.py --raw-batch datasets/raw/site_capture_20261112 --segments datasets/annotated/site_capture_20261112/segments.csv --output datasets/annotated/site_capture_20261112/extract_v2
```
