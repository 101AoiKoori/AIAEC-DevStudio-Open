# V4 数据集脚本入口

本目录收录任务可调用的脚本。人物当前交接点见[生产进度](../../docs/DATASET_PRODUCTION.md#当前生产进度)。

每任务维护自己的 `config/source.yaml` 和 `config/dataset.yaml`。来源配置指向原始素材、现有标注清单、父来源和明确选定的记录，生成器实际读取自己支持的字段。候选发现清单不等于全部被选中。任务代码只导入本任务及 `common/`，共享素材可跨任务复用。

- [人物](person/README.md)、[电动车](e_bike/README.md)、[火焰贴纸](fire/README.md)、[楼名纸条](building_name_paper/README.md)
- [禁行牌](sign/README.md)、[垃圾桶开闭](waste_bin_state/README.md)、[垃圾贴纸](waste_sticker/README.md)、[温湿度仪表](meter/README.md)

任务组合与来源约定查对应任务配置，批次产量和来源查实际清单与回执。[当前生产进度](../../docs/DATASET_PRODUCTION.md#当前生产进度)

## 人物最小生产路径

人物纹理、车辆遮挡素材和分发快照由 `person/config/source.yaml` 登记。当前入口读取 `reviewed_background_environment` 指向的逐图库分发快照。新输入入库或补判后需要重新选源时，运行下面的分发命令，把新 JSON 登记到这个角色，再编制目标集合的计划。[逐图入库与资格查询](#逐图入库与资格查询)

人物合成的分发和编计划使用 `reference_quad_px` 作为局部投影坐标。首次入库若已有候选接地掩膜而未测得四边形，会从掩膜内拟合一个局部参考矩形并记录 `reference_quad_basis=inscribed_contact_mask_rectangle`；它只提供坐标，规划器的远近尺寸仍使用注明依据的近似比例。若连可用的参考矩形也无法形成，该用途保持 `unknown` 并记录缺项。历史记录需要在入库端补判后更新资格。地面近似和各任务通用资格见[标签、空间与用途资格](../../docs/DATASET_DESIGN.md#标签空间与用途资格)。[人物入库](common/ingest_frame_environment.py#L103) · [人物编计划](person/build_full_person_plan.py#L534)

```powershell
$PersonConfig = 'datasets/scripts/person/config'
$BatchStamp = Get-Date -Format 'yyyyMMdd_HHmmssfff'
$NewPersonDispatch = "$PersonConfig/background_dispatch_$BatchStamp.json"
.venv/Scripts/python.exe -B -m datasets.scripts.person.compile_background_dispatch --environment-database datasets/annotated/common/sources/frame_environment.sqlite3 --environment-snapshot $NewPersonDispatch
```

在工作空间根目录运行。小样命令创建新计划并替换人物小样目录中的生成结果。看 `images/`、`labels/` 和 `manifest.json`，核对所需的背景、人数、深度、尺度、偏航、遮挡、空场和硬负例，问题和下一轮改动写入 [`sample_issues.yaml`](person/config/sample_issues.yaml)。当前已产小样和正式批次见[生产进度](../../docs/DATASET_PRODUCTION.md#当前生产进度)。

```powershell
$PersonConfig = 'datasets/scripts/person/config'
$BatchStamp = Get-Date -Format 'yyyyMMdd_HHmmssfff'
$PersonSamplePlan = "$PersonConfig/sample_plan_$BatchStamp.json"
.venv/Scripts/python.exe -B -m datasets.scripts.person.build_full_person_plan --sample --write --plan-output $PersonSamplePlan
if ($LASTEXITCODE -ne 0) { throw '人物小样编计划失败' }
.venv/Scripts/python.exe -B -m datasets.scripts.person.compose_planned_ground_person --production-plan $PersonSamplePlan --output datasets/samples/person --replace
```

正式生成按 split 使用已编好的监督计划。当前 train、val、test 分别读 `production_plan.json`、`val_plan_300.json`、`test_plan_300.json`。生成器把计划中列出的直接实拍与合成写入同一 split，随后打包入口检查三套清单并更新训练入口。选源或组合发生变化时，先用 `build_full_person_plan --split <split> --budget <总张数> --write --plan-output <新计划>` 编新计划，核对它的实际配额和来源，再把命令里的计划路径换成新计划。`--plan-output` 需要 `person/config/` 内未占用的 JSON 路径。[正式数据与回执](../../docs/DATASET_PRODUCTION.md#从原始采集到导出)

```powershell
$PersonConfig = 'datasets/scripts/person/config'
$PersonPlans = [ordered]@{
    train = "$PersonConfig/production_plan.json"
    val   = "$PersonConfig/val_plan_300.json"
    test  = "$PersonConfig/test_plan_300.json"
}
foreach ($Split in $PersonPlans.Keys) {
    .venv/Scripts/python.exe -B -m datasets.scripts.person.compose_planned_ground_person --production-plan $PersonPlans[$Split] --output "datasets/exports/person/$Split"
    if ($LASTEXITCODE -ne 0) { throw "人物 $Split 生成失败" }
}
.venv/Scripts/python.exe -B -m datasets.scripts.person.package_person_dataset --split train --split val --split test
```

确认某个 split 导出有误时，单独重跑该 split 的生成命令并加 `--replace`，然后重新打包。入口按原清单与生产回执识别生成图签，保留 `.gitkeep` 和其他人工文件；新名称与未认领文件碰撞时停止。进程中断后再次运行会先按发布记录及完成回执恢复；断电、文件系统损坏或不遵守同一锁的外部写入不在此保证范围。旧清单或回执缺失时先核实实际文件。`exports/person/data.yaml` 与 `workspace_sources.json` 是打包入口写出的数据与来源索引。训练入口、数据冻结点和本次结果见[人物 V4 工作空间入口](../../docs/TRAINING_WORKFLOW.md#人物-v4-工作空间入口)。

人物打包结果的来源索引使用 `manifest_only`：项目训练入口和训练包只选清单中的图签，保留在目录里的其他人工文件不进入训练。直接使用外部 Ultralytics 加载原始 `data.yaml` 时，它可能自行扫描整个图片目录，不能依靠该模式排除人工文件。

背景发现按媒体库父视频选新帧。`plan → analyze → apply` 用原帧路径连接观测与资格，输入或判定依据变化时补算受影响的证据。更换分发快照后，新编计划使用新选源，已有计划保留自身的输入引用。[机器判定与写回](../../docs/DATASET_PRODUCTION.md#机器判定与写回)

## 恢复生产入口

以下入口按任务接续，计划与实际输出见对应列。现行生产与历史批次分开记录。

|任务|先运行的本任务入口|当前计划（本任务 `config/`）|输出|
|---|---|---|---|
|人物|[人物最小生产路径](#人物最小生产路径)|新编小样计划或 `production_plan.json`（train）、`val_plan_300.json`、`test_plan_300.json`|`samples/person` 或 `exports/person/<split>`|
|电动车|`e_bike/compose_v4_grounded_train.py`|`production_plan.json`|`samples/e_bike`；直接实拍资格可另行只读查询|
|火焰|`fire/compose_real_facades.py`，当前逐图库驱动实拍、实拍背景贴图和 Blender 三分支|按当前资格重编 `production_plan.json`|`samples/fire/manifest.json`|
|楼名|[全量检测数据](building_name_paper/README.md#当前全量检测数据)：`building_name_paper/build_full_dataset.py --replace`|`full_recipe.json`、`full_<split>_plan.json`；有纸／无纸物理层与可见掩膜在 annotated 留存|`exports/building_name_paper/{train,val,test}`；`build_current_sample.py` 保留四分支小样入口|
|禁行牌|[全量编制与核对](sign/README.md#当前全量数据)：`sign/build_full_dataset.py`|`full_train_plan.json`、`full_val_plan.json`、`full_test_plan.json`|`exports/sign/{train,val,test}`，训练入口 `data.yaml`|
|垃圾贴纸|[四路完整生产](#垃圾贴纸完整数据集)|`native_full_plan.json`、`surface_full_plan.json`、`hybrid_full_plan.json`，汇总为 `full_<split>_plan.json`|`exports/waste_sticker/{train,val,test}`，训练入口 `data.yaml`|
|桶状态|`waste_bin_state/render_blender_waste_bin.py`|`render_plan.json`|前景写 `annotated/waste_bin_state/foreground`，合成候选写 `annotated/waste_bin_state/sample_composites`，当前缩影写 `samples/waste_bin_state`|
|温湿度表|`meter/build_current_sample.py --prepare` 后运行同入口|`sample_recipe.json` → `sample_plan.json`|`samples/meter`，包含透明前景、实拍／渲染／底图合成图签及单张 contact_sheet.jpg|

垃圾贴纸当前使用场景直出、Blender 桶面叠贴、整桶叠实拍场地和实拍桶面叠贴四路入口。每图 1–4 个可见桶，完整图签与训练入口按实际成图汇总。[垃圾贴纸数据设计](../../docs/DATASET_DESIGN.md#垃圾贴纸数据设计)

以下命令在工作空间根目录执行。任务样本放在 `datasets/samples/<任务>/`，数据集放在 `datasets/exports/<任务>/{train,val,test}/`。空目录用于指明后续产物的落点，人物小样和三个 split 的 `.gitkeep` 使这些目录在新 checkout 中保留。Blender 变量 `$DatasetBlender` 按[根目录说明](../../README.md#本机数据集入口)设置。渲染命令后台打开统一工程，入口按任务选择 Scene 和 Collection；GPU 示例采用 OPTIX。[场景映射](../README.md#单文件编辑与生产接线) · [并行渲染与工程写入](../README.md#并行渲染与工程写入) · [全量训练计划设计](../../docs/FULL_TRAIN_PLAN_DESIGN.md)

```powershell
# 电动车：当前合成子批；下一条命令只读检查直接实拍资格
.venv/Scripts/python.exe -B datasets/scripts/e_bike/compose_v4_grounded_train.py --plan datasets/scripts/e_bike/config/production_plan.json --output datasets/samples/e_bike
.venv/Scripts/python.exe -B -m datasets.scripts.common.package_direct_real --task e_bike --split train

# 火焰：当前逐图库三分支小样，原位替换生成结果
.venv/Scripts/python.exe -B datasets/scripts/fire/compose_real_facades.py --replace

# 楼名：完整检测实拍、替字、纯渲染与楼体实拍合成
.venv/Scripts/python.exe -B datasets/scripts/building_name_paper/build_full_dataset.py --replace

# 禁行牌：全量编制、生成并汇总核对
.venv/Scripts/python.exe -B -m datasets.scripts.sign.build_full_dataset --split train --generate
.venv/Scripts/python.exe -B -m datasets.scripts.sign.build_full_dataset --split val --generate
.venv/Scripts/python.exe -B -m datasets.scripts.sign.build_full_dataset --split test --generate
.venv/Scripts/python.exe -B -m datasets.scripts.sign.audit_current_sample --full

# 垃圾贴纸：完成四路中间产物后按实际图签汇总
.venv/Scripts/python.exe -B datasets/scripts/waste_sticker/assemble_full_dataset.py --write
```

桶状态从 `Waste Bins Photo` 输出图像、整桶掩膜和状态掩膜到 `datasets/annotated/waste_bin_state/foreground`。开盖采用盖板从后方垂下的姿态，按桶口可见像素检查。编制器读取前景清单和底图资格，用实际掩膜重算落点。需要重渲前景时确认对应生成位置为空。`datasets/samples/waste_bin_state` 下的当前数据缩影使用 `images/`、`labels/` 和 `manifest.json`。缩影总览和留作来源参考的诊断材料在 `datasets/annotated/waste_bin_state` 中查阅。

现行桶状态选源使用 `all_current_eligible`。编制器默认从逐图库枚举本任务 `train` 的 `eligible composite_background`，同帧有共享池路径时优先采用，也可读取来自其他任务 `raw` 目录的已合格帧。其他任务目录用于优先发现候选。以桶为画面主体的帧按桶标签资格供直接打包，从桶任务常规底图池排除。范围内非主体桶的完整标签随生成保留，范围外远物的实际占位与遮挡仍计入布局。[底图读取实现](waste_bin_state/qualified_pool.py)

刚性桶组在合成时读取同帧最新共享空间观测，在原图坐标用桶底脚点和参考点求尺寸比。可信地平线使用两点到地平线的距离比，标线比只在同一条标线中心轴的实测锚点区间使用。尺寸比乘以计划的参考高度，使用同一缩放重算桶掩膜、落点和图签。清单的 `actual.scale` 保存证据修订、参考高度、比值、锚点和实际高度，标线比另记原始误差敏感区间。尺度证据不适用或缩放后的图超出已核对的接地、标签和遮挡条件时，沿用原计划尺寸并记录原因。

已有目标的完整原图标签随合成输出保留并转换框。原图中的桶可使用 `lid_state`／`color` 或既有的 `state`／`bin_color` 字段；桶色是原图属性，不必属于新渲染前景的四色集合。开闭类别、框、对象身份和全帧完整度仍须一致，前景不得遮住原桶框。当前共享池垃圾桶合成底图资格和补证见[生产进度](../../docs/DATASET_PRODUCTION.md#当前生产进度)。

垃圾桶生成入口要求前景渲染清单的 `split` 与输出 `target_split` 一致。背景底图可先完成任务资格；借用其他 split 前景做的脚点诊断不构成该 split 的正式合成产物。

```powershell
& $DatasetBlender --background datasets/assets/blender/aiaec_dataset.blend --python-exit-code 1 --python datasets/scripts/waste_bin_state/render_blender_waste_bin.py -- --out datasets/annotated/waste_bin_state/foreground --plan datasets/scripts/waste_bin_state/config/render_plan.json --seed 1918 --samples 12 --device OPTIX
.venv/Scripts/python.exe -B lab/scripts/media/media_index.py build --scan-root datasets/annotated --source datasets/annotated/waste_bin_state --no-video-probe --no-clean
.venv/Scripts/python.exe -B lab/scripts/media/media_index_sqlite.py sync --partition datasets/annotated/waste_bin_state
```

制作缩影时，合成编制器用合格实拍和 Blender 轨道底图、标准前景和已核对的双桶与三桶前景编候选。贝叶斯分配结果写入 `sample_composite_allocation.json`。下面的命令生成合成候选，再与完整实拍正例和空负例打包。重放早期 `pool_pilot_plan.json` 等计划时，合成产物同样写入空的 `datasets/annotated/waste_bin_state/sample_composites`，或计划对应的正式 split。

```powershell
.venv/Scripts/python.exe -X utf8 -B datasets/scripts/waste_bin_state/plan_representative_sample.py --budget 32 --additional-render-manifest datasets/annotated/waste_bin_state/foreground_multi/manifest.json
.venv/Scripts/python.exe -X utf8 -B datasets/scripts/waste_bin_state/generate_waste_v4_grouped.py --plan datasets/annotated/waste_bin_state/sample_composite_plan.json --out datasets/annotated/waste_bin_state/sample_composites
.venv/Scripts/python.exe -B lab/scripts/media/media_index.py build --scan-root datasets/annotated --source datasets/annotated/waste_bin_state --no-video-probe --no-clean
.venv/Scripts/python.exe -B lab/scripts/media/media_index_sqlite.py sync --partition datasets/annotated/waste_bin_state
.venv/Scripts/python.exe -X utf8 -B datasets/scripts/waste_bin_state/build_current_sample.py
```

确认当前缩影有误并需要重做时，可在最后一条命令追加 `--replace`。入口根据原清单和生产回执识别旧生成图签，保留 `.gitkeep` 和其他人工文件；新生成名称若与未认领文件冲突会停止，避免覆盖。进程中断后再次运行该入口会先检查发布记录并恢复；断电、文件系统损坏或不遵守同一锁的外部写入不在此保证范围。旧清单或回执缺失时先核实实际文件，不按名称批量删除。

#### 垃圾桶开闭全量导出

全量 train 计划使用已有标准、多桶和可判斜视前景，val 和 test 分别渲染各自的前景。阶段合成写入 `datasets/annotated/waste_bin_state/full_composites/<split>`。每个集合的运行清单与分配审计记录可行组合、父源和实际图签数。

```powershell
.venv/Scripts/python.exe -X utf8 -B datasets/scripts/waste_bin_state/plan_full_train.py
foreach ($BinSplit in @('val', 'test')) {
    $BinEval = "datasets/annotated/waste_bin_state/full_evaluation/$BinSplit"
    $BinRenderSeed = if ($BinSplit -eq 'val') { 20261118 } else { 20261119 }
    .venv/Scripts/python.exe -X utf8 -B datasets/scripts/waste_bin_state/plan_full_evaluation.py foreground --split $BinSplit
    & $DatasetBlender --background datasets/assets/blender/aiaec_dataset.blend --python-exit-code 1 --python datasets/scripts/waste_bin_state/render_blender_waste_bin.py -- --out "$BinEval/foreground" --plan "$BinEval/render_plan.json" --seed $BinRenderSeed --samples 8 --device OPTIX
    .venv/Scripts/python.exe -X utf8 -B datasets/scripts/waste_bin_state/plan_full_evaluation.py composites --split $BinSplit
}
.venv/Scripts/python.exe -B lab/scripts/media/media_index.py build --scan-root datasets/annotated --source datasets/annotated/waste_bin_state --no-video-probe --no-clean
.venv/Scripts/python.exe -B lab/scripts/media/media_index_sqlite.py sync --partition datasets/annotated/waste_bin_state
.venv/Scripts/python.exe -X utf8 -B datasets/scripts/waste_bin_state/generate_waste_v4_grouped.py --plan datasets/annotated/waste_bin_state/full_plans/train_composite_plan.json --out datasets/annotated/waste_bin_state/full_composites/train
foreach ($BinSplit in @('val', 'test')) {
    .venv/Scripts/python.exe -X utf8 -B datasets/scripts/waste_bin_state/generate_waste_v4_grouped.py --plan "datasets/annotated/waste_bin_state/full_evaluation/$BinSplit/composite_plan.json" --out "datasets/annotated/waste_bin_state/full_composites/$BinSplit"
}
.venv/Scripts/python.exe -B lab/scripts/media/media_index.py build --scan-root datasets/annotated --source datasets/annotated/waste_bin_state --no-video-probe --no-clean
.venv/Scripts/python.exe -B lab/scripts/media/media_index_sqlite.py sync --partition datasets/annotated/waste_bin_state
.venv/Scripts/python.exe -X utf8 -B datasets/scripts/waste_bin_state/build_full_dataset.py --check
.venv/Scripts/python.exe -X utf8 -B datasets/scripts/waste_bin_state/build_full_dataset.py --publish --archive-previous
```

发布时逐图读取整桶框、类别和父源，校验三集图签和图像身份后替换当前导出。训练入口数据描述为 `datasets/exports/waste_bin_state/data.yaml`，先前发布移动到 `datasets/annotated/waste_bin_state/full_export_history`。

当前正式导出已有图签但缺训练来源索引时，从三份分集清单同步 `workspace_sources.json`，再运行现有训练预检。

```powershell
.venv/Scripts/python.exe -B datasets/scripts/waste_bin_state/build_full_dataset.py --sync-training-sources
.venv/Scripts/python.exe -B lab/features/waste_bin/training/scripts/train_waste_bin_state_v4.py check
```

仪表当前使用同一模型的白色／彩色印刷材质、27 个开放机位、独立双针与局部反光生成配对小样。当前维护源为本任务 `source.yaml#sample_recipe`、`sample_plan`、`sample_manifest`；先编制配方与技术贴图，再生成透明前景、中灰渲染预览及真实底图合成，八点一同变换。[当前小样入口](meter/README.md#当前材质与读数配对小样)

```powershell
.venv/Scripts/python.exe -B datasets/scripts/meter/build_current_sample.py --prepare
.venv/Scripts/python.exe -B datasets/scripts/meter/build_current_sample.py
```

下列 12 状态 Arena 渲染和 36 状态固定投影合成为历史配方的显式回放命令，当前全局计划登记不重复计入这些批次。

```powershell
& $DatasetBlender --background datasets/assets/blender/aiaec_dataset.blend --python-exit-code 1 --python datasets/scripts/meter/render_meter_arena_v4.py -- --output datasets/samples/meter --production-plan datasets/scripts/meter/config/render_plan.json --count 12 --pose --foreground --device OPTIX --samples 16
.venv/Scripts/python.exe -B lab/scripts/media/media_index.py build --scan-root datasets/samples --source datasets/samples/meter --no-video-probe --no-clean
.venv/Scripts/python.exe -B lab/scripts/media/media_index_sqlite.py sync --partition datasets/samples/meter
.venv/Scripts/python.exe -B datasets/scripts/meter/compose_meter_real_background_v4.py --production-plan datasets/scripts/meter/config/production_plan.json --count 36 --output datasets/exports/meter/train
```

垃圾贴纸当前小样由本任务 `source.yaml#sample_manifest` 指向实际清单。旧 `direct_real_plan.json` 与 `blender_plan.json` 分别通过 `legacy_direct_real_plan` 和 `legacy_blender_plan` 保留历史配方，当前覆盖统计使用三个 `full_<split>_plan.json`。楼名当前四分支小样及楼体实拍合成的渲染、重跑方式见[任务入口](building_name_paper/README.md#blender-楼体与实拍底图合成)。单独重渲纯 Blender 分支使用已认可的逐状态计划，随后重新汇总当前小样。

```powershell
& $DatasetBlender --background datasets/assets/blender/aiaec_dataset.blend --python-exit-code 1 --python datasets/scripts/building_name_paper/render_blender_names.py -- --plan datasets/annotated/fire/building_name_paper_sample/blender_plan.json --output datasets/annotated/fire/building_name_paper_sample/blender --device OPTIX
```

### 垃圾贴纸完整数据集

原生分支按已存车载相机生成三个集合的 1,350 张图。桶面分支生成 440 张，读取逐图库精确 split 的标签和桶面。整桶混合生成 Train 450、Test 40 张，真实底图和物理前景模板的复用均随逐图清单记录。当前 1–4 桶配额由这三个编制入口维护，最终汇总验证 Train 1,800、Val 240、Test 240 和所有图签。

```powershell
& $DatasetBlender --background datasets/assets/blender/aiaec_dataset.blend --python-exit-code 1 --python datasets/scripts/waste_sticker/produce_native_dataset.py -- --engine BLENDER_EEVEE --eevee-samples 128
.venv/Scripts/python.exe -B datasets/scripts/waste_sticker/produce_surface_dataset.py
.venv/Scripts/python.exe -B datasets/scripts/waste_sticker/produce_hybrid_dataset.py plan
foreach ($StickerSplit in @('train','test')) {
    & $DatasetBlender --background datasets/assets/blender/aiaec_dataset.blend --python-exit-code 1 --python datasets/scripts/waste_sticker/render_sticker_real_plate.py -- --plan "datasets/annotated/waste_bin/waste_sticker_full/hybrid/$StickerSplit/template_plan.json" --out "datasets/annotated/waste_bin/waste_sticker_full/hybrid/$StickerSplit/layers" --samples 12
    .venv/Scripts/python.exe -B datasets/scripts/waste_sticker/produce_hybrid_dataset.py compose --split $StickerSplit
}
.venv/Scripts/python.exe -B datasets/scripts/waste_sticker/assemble_full_dataset.py --write
```

汇总在生产守卫内登记实际输入、图签、来源及产量，并写入 `datasets/exports/waste_sticker/data.yaml`、`workspace_sources.json` 和 `audit.json`。原生图与整桶物理层的中间掩膜留在 `datasets/annotated/waste_bin/waste_sticker_full`，用于核对实际可见框。每个 split 的 `manifest.json` 保存当前清单。重跑时原生按状态签名续作，正式导出替换按当前清单和用户授权处理。

### 垃圾贴纸按可见桶数生成与合成

固定同一贴纸帧，选两只桶比较低机位的左、中、右与远、中、近。示例使用第 123 帧，输出目录须为空。替换为其他帧时同步改 `orbit_frame_NNNN` 目录名。

```powershell
& $DatasetBlender --background datasets/assets/blender/aiaec_dataset.blend --python-exit-code 1 --python datasets/scripts/waste_sticker/render_sticker_arena_v4.py -- --orbit-frame 123 --bins blue,green --out datasets/samples/waste_sticker/orbit_frame_0123 --samples 4
```

桶图案批量生成按所选批次显式传入 `--plan`。旧 `legacy_blender_plan` 对应文件仅保留结构参考用途，在新批次每个 `strata` 条目的 `parameters` 中加入 `"visible_bins": ["blue", "green"]`，可以逐状态选 1–4 个不同颜色名。省略时出四桶。使用局部遮挡时，`occluder_color` 取当前可见桶之一，或设为 `null`，并与该状态的 `augmentation` 对应。输出使用计划所声明的 split 和对应空目录。出图后用 `manifest.json` 的 `actual.visible_bins` 和 `actual.instances` 核对桶列表、贴纸类别及候选框。

实拍桶面合成沿用 `generate_sticker_real_photo_v4.py --plan … --out …`。按底图实际入画的桶数选源，在计划的 `parameters.placements` 中用 `bin_id` 逐面指定贴纸，`target.placements` 同步登记。底图上的原有完整标签随样本保留。相机匹配、整桶叠加所需掩膜和最终可见框的计算见[1–4 桶制作配方](../../docs/DATASET_RECIPES.md#垃圾贴纸按桶数生成与合成)。

## 逐图入库与资格查询

修改逐图证据、任务配置或组合轴时，按[数据与配置操作 Skill](../../.agents/skills/dataset-config-operations/SKILL.md)核对维护源及实际消费者。组合轴的 `state_coverage.py --write` 生成说明区块；实际编批仍以任务入口读取的计划和来源为准。

八项任务的 `source.yaml` 通过 `databases.frame_environment` 声明 `datasets/annotated/common/sources/frame_environment.sqlite3`。已确定的读取接口在 `datasets.scripts.common.frame_environment` 中，`query_frames` 查询逐图记录，`quota_summary` 查询资格汇总。路径声明、实际回填和任务入口接线按各自证据核对。[当前接线](../../docs/DATASET_PRODUCTION.md#背景入库仍需接通的代码)

新图录入函数 `datasets.scripts.common.ingest_frame_environment.ingest_frame` 从同一份原始字节计算哈希并解码，适用任务共享已解码图像。调用方提供主体、标签范围和放置区域，或通过 `review` 回调在此次录入内取得机器观测。函数统一写入共享实例、任务标注、空间结果与用途资格。[机器判定与写回](../../docs/DATASET_PRODUCTION.md#机器判定与写回)

多图机器审查用同一入口的 `--batch-jsonl`。每行是一张原图的 `image`、共享 `objects`、各任务 `task_annotations`、对应 `task_geometry`、`environment` 和 `lineage`；同一物体由各任务标签引用共同的 `object_id`。每项任务观测要声明完整度、整帧或局部范围、split 和带版本的机器视觉依据。已有帧可附 `frame_id` 与 `content_sha256` 防止把新证据写到错图；省略的任务保持原判定，不因批量回填而变成空负例。每张图单独事务，失败行之前已提交的行可按相同证据重试。来源清单仅作历史线索，不代替逐图完整视觉结论。

共享池的独立空间观测由 `.venv/Scripts/python.exe -X utf8 -B -m datasets.scripts.common.audit_background_space --write` 写入逐图库，按帧身份复用局部坐标和候选接触区域。该命令写共享观测。按任务确认已有目标标签、可用落点或表面、目标尺寸和遮挡后，通过下方 `--batch-jsonl` 写回受影响的任务资格。可缺少图像测得的尺度比，合成用有依据的近似尺寸并记录依据。分别查询 `empty_negative` 与 `composite_background` 的当前 `eligible` 记录，合格底图数随新证据重算。[底图资格和近似尺寸](../../docs/DATASET_DESIGN.md#标签空间与用途资格)

共享池任务状态对账运行 `.venv/Scripts/python.exe -m datasets.scripts.common.reconcile_pool_pending` 先查看缺少当前合成用途决策的组合；加 `--write --receipt datasets/annotated/common/review/<批次>/receipt.json` 只为这些组合写入有原因码的 `unknown`。它复用现有共享空间及任务证据，不解码图片、不补造完整标签、不改变已有资格。补齐实际任务视觉结论和放置证据后仍由 `ingest_frame_environment --batch-jsonl` 重算资格。重复对账预期新增数为零。

相对深度的 QA 入口可复用已核对的 `observations.jsonl`，逐图核对身份、原图哈希和坐标，输出到 `datasets/annotated/common/perspective/background_pool_space_v5_relative/`。用重复的 `--reuse-space-jsonl` 指定互不重叠的快照，用 `--index` 限定快照覆盖的池索引。可选模型在离线环境装好 PyTorch 和 Transformers 后从本地目录读取，实测使用 Transformers 4.51.3 和 Depth Anything V2 Small 相对深度 safetensors。省略 `--relative-depth-model` 时仍会检查标线宽度。把 `$SpatialSnapshot`、`$LocalDepthModel` 和 `$ModelSourceUrl` 改为本批的输入。

```powershell
.venv/Scripts/python.exe -X utf8 -B -m datasets.scripts.common.audit_background_space --qa-only --reuse-space-jsonl $SpatialSnapshot --relative-depth-model $LocalDepthModel --relative-depth-model-source $ModelSourceUrl
```

`perspective_cues.relative_depth` 提供同帧接地掩膜内的远近排序，由原图和镜像模型输出拟合，并用保留网格检查。`ground_relative_depth_index` 只在候选掩膜与拟合支持范围同时覆盖的源像素返回值。`perspective_cues.local_stripe_scale` 保存同一条近似等宽标线上的尺度比、两侧边线与像素误差敏感区间，`ground_stripe_scale_ratio` 只在两个实测锚点之间的轴线上求值。原有 `scale` 保留其地平线证据条件。单目排序指数无单位，且其零点任意。标线结果依赖两侧物理平行的假设。[共享深度接口](common/review/relative_depth.py)

对已核对的单帧地面多边形，可在 `--qa-only` 运行中结合 `--scene-evidence`、`--sam-weights` 和 `--preserve-current-obstacles` 做接地补证。原障碍继续生效，新增障碍检测只在本次地面证据范围内采用。单独的 QA 回执保留这些观测供协调写入者使用。

```powershell
python -X utf8 -B -m datasets.scripts.common.ingest_frame_environment --batch-jsonl $MachineReviewJsonl
```

旧共享背景池的 `background_pool_manifest.json` 保留当时的候选字段；根部 `machine_review_index` 指向可重建的审查关联表。逐图库才是当前资格权威。新的机器审查批次写库后，构建关联表时重复传入所有要保留的审查 JSONL；后批同帧同任务的结论取最新行，旧行以 `superseded_review_row` 留在排除列表。`check` 核对原图哈希、当前对象、任务观测与资格修订，不升级旧候选字段。旧池生成器仍引用已迁移的 Blender 历史文件，当前不通过重建旧池来刷新这项关联。

```powershell
python -X utf8 -B -m datasets.scripts.common.background_machine_review build --review-jsonl $BatchA --review-jsonl $BatchB --write
python -X utf8 -B -m datasets.scripts.common.background_machine_review check
```

在项目根目录执行。将 `$RawFrame` 设为要录入的原图路径，将 `$FrameReviewJson` 设为本次机器观测文件。JSON 顶层提供 `objects`、`task_annotations`、`environment`、`task_geometry` 和 `lineage`，完整标签带目标有无、作用域及证据，多个任务的实例可以引用同一 `object_id`。录入命令打印 `frame_id` 和用途资格修订；下面两条只读命令可独立查询现有库存和实际用途。

```powershell
python -X utf8 -B -m datasets.scripts.common.ingest_frame_environment $RawFrame --review-json $FrameReviewJson
python -X utf8 -B lab/scripts/contracts/state_coverage.py --source-status --json
python -X utf8 -B -c "from datasets.scripts.common.frame_environment import query_frames; print([(r['frame_id'], r['record_id']) for r in query_frames(task='e_bike', role='empty_negative', split='train')])"
```

库有实际记录后，`query_frames(task=…, role=…, split=…)` 默认返回 `eligible` 记录；`split=None` 查询全部归属，`split="unassigned"` 查询合格未分配现货，可按父来源分配到计划。特定 split 的结果为 0 时同时查未分配库存和可接入来源。按 `frame_id` 定位图像表示，按 `record_id` 查历史资格修订，按 `status` 查待补证或受阻原因。`python lab/scripts/contracts/state_coverage.py --source-status` 只读汇总分集库存和父组，加 `--json` 看状态明细。逐图库存按身份、任务、角色和 split 计；汇总同一任务、同一 split 的直接实拍配额时，合并正例和实拍空负例，按 `capture_identity` 跨角色去重。计划预算与实际产量另记。

用 `python -B -m datasets.scripts.common.backfill_frame_environment candidates --task person --role composite_background --split train` 查询已分配的 `assigned` 和可选的 `pending`。已选定的未分配项用同一模块的 `assign --task <任务> --role <用途> --split <集合> --record-id <记录>` 写入，可重复传入 `--record-id`。写入前事务复核当前资格与 `source.yaml#pairing` 的明确预留，来源配置不可用则回滚。`partial` 表示已有部分记录写入，核对逐条返回的 `assigned_record_id`。同源接触及未知或歧义的来源关系随结果记录。检查范围为同任务、同用途的当前合格记录，同一帧跨 split 复用仍需已有依据。`assign` 由操作者显式调用，参数只支持分配未分配记录；取消现有用途、登记复用依据与删除原件有各自的证据和授权范围。

目前任务计划器未自动调用这两个命令。查询返回的预留状态用于核对来源，分配结果以实际写入回执为准。

直接实拍入口的 `--id` 选择来源清单记录；如使用 `backfill_frame_environment assign` 分配素材，可用 `--frame-record-id <assigned_record_id>` 选择返回的逐图库资格记录，参数可重复。两种筛选同时使用时须同时匹配；入口仍检查该任务和 split 的现行完整标签与文件，输出摘要包含逐图库记录 ID。

逐图持久库沿用来源总账的角色词表。`direct_target` 用于目标标签完整的直接实拍正例。楼名检测层可独立成立，文字监督按 OCR 标签单独判断。`empty_negative` 由画面证据确认整帧无接收任务识别范围内的目标，空间待判或不可放置时仍可使用。裁剪区或表面的负标注只覆盖所声明的区域。合成背景单独查询 `composite_background`，按本任务核对主体、可用空间和完整标签。首次入库可用画面证据填写 `environment.primary_subject_task`。匹配该任务时，入库会写出 `blocked` 和 `primary_task_target_is_frame_subject`。合成后保留并更新范围内非主体目标的标签。原本范围外的物体因缩放或裁切进入输出的任务范围时补齐标签。任务目录只用于发现候选，`presence` 取自相应画面的目标证据。计划保留所选 `record_id`、证据修订与使用坐标。[标签、空间与用途资格](../../docs/DATASET_DESIGN.md#标签空间与用途资格)

## 从现有数据库取得当前输入

`lab/scripts/media/media_index_sqlite.py task-inputs` 查询现有 `query_cache.sqlite3` 中的任务声明。数据库保存八份来源/设计配置及明确引用的 `production_plan`、`render_plan`、`direct_real_plan`、`blender_plan`；生成器继续读取返回路径所指的本任务 YAML/JSON。数据库用于发现和反查，原配置仍是编辑源，不建立第二套计划。

```powershell
# 所有任务的当前计划、预算、集合、存在状态与路径
.venv/Scripts/python.exe -X utf8 -B lab/scripts/media/media_index_sqlite.py task-inputs --limit 0
# 按任务、集合和计划字段角色取得输入引用
.venv/Scripts/python.exe -X utf8 -B lab/scripts/media/media_index_sqlite.py task-inputs --task person --kind input --split train --role background --limit 0
# 当前来源配置，含选定记录、排除原因及父来源；桶的待合成背景ID也在这里
.venv/Scripts/python.exe -X utf8 -B lab/scripts/media/media_index_sqlite.py task-inputs --task waste_bin_state --kind config
# 查询数据库已索引的计划声明及其来源
.venv/Scripts/python.exe -X utf8 -B lab/scripts/media/media_index_sqlite.py task-inputs --task meter --kind state --plan-key render_plan --limit 0
# 从一张图反查哪些任务计划声明使用它
.venv/Scripts/python.exe -X utf8 -B lab/scripts/media/media_index_sqlite.py task-inputs --kind input --path datasets/annotated/site_capture_20261112/sequence_640_v1/full_scene/frame_001201.png --limit 0
```

`input` 的角色按状态字段定位为 `background`、`target`、`parent_source`、`annotation_reference` 或 `declared_input`，并保留 `reference_fields` 和状态 ID；它不是视觉适用性的新判定。未展开到状态的用途及排除项通过 `config` 读取，不能把整个候选池当作已选输入。`present=false` 表示文件当前不存在；桶状态合成计划为 `pending_plan`，仪表未来前景仍是计划引用。状态原文保留候选标签和来源说明。

查询会核对声明文件的存在状态、大小和修改时间；不一致时拒绝返回过期声明，并指出需刷新的任务分区。编辑某任务 YAML/计划后，仅构建该分区并同步。示例：

```powershell
.venv/Scripts/python.exe -X utf8 -B lab/scripts/media/media_index.py build --scan-root datasets/scripts --source datasets/scripts/person --no-video-probe --no-clean
.venv/Scripts/python.exe -X utf8 -B lab/scripts/media/media_index_sqlite.py sync --partition datasets/scripts/person --partition datasets/scripts/@root
```

构建器同时维护扫描根的 `@root` 元数据，复用未变化的分区。需要查询新计划时刷新对应任务分区；需要搜索新媒体时刷新对应媒体分区。

目录移动、失效或局部同步后若缓存为 `complete_catalog=false`，用不带 `--partition` 的 `sync` 恢复完整查询，也可用 `--allow-partial` 限范围查询。缓存缺失或过期时，生产继续读实际输入并使用持久回执。[生产守卫](../../lab/training_common/production_guard.py)

任务 YAML 里的日期等标量要加引号（`'2026-11-15'`）。未加引号时 PyYAML 会解析成 `date` 对象，构建索引写 JSON 时因不可序列化而失败（`TypeError: Object of type date is not JSON serializable`）；这是当前构建器的已知限制，尚未修改代码。

仪表全量使用 `meter/build_full_dataset.py --prepare` 编计划、`--pilot` 核对先导，再无参数执行；`--split train` 可单集续作，完成的透明前景块按回执与哈希复用。三集合图签及训练索引写入 `datasets/exports/meter`。[详细入口](meter/README.md#当前全量制作入口)。

仪表当前小样用 `meter/build_current_sample.py --prepare` 编制，再运行同一入口出图；`--reuse-render-from <已保存原批次>` 可复用匹配的透明前景重做合成。共享池按仪表当前合成资格、场景和路线分层选源，配方分别控制饱和度、曝光及前景相对亮度；全量三集合配额见 `config/full_train_allocation.yaml#tasks.meter`，当前仍为设计草案。

## 整理任务源与共享背景池

### 八路 Blender 空场扩池（2026-11-17）

统一工程内的八条三维相机轨道可在 `Arena`、`Arena Indoor Soft Light` 和 `Arena Detailed` 渲染。后台渲染器会切换当前求值 Scene；仅指定目标 Scene 的帧号会导致轨道相机保持静止。已用每路 24 帧、三个 Scene 共渲 576 张，按原帧、轨道、Scene、相机矩阵和工程哈希分别记录在 `datasets/annotated/common/sources/aiaec_eight_routes*_20261117/manifest.json`，原图在 `datasets/raw/shared/backgrounds/rendered/aiaec_eight_routes*_20261117/`。复现入口为 `common/render_shared_route_backgrounds.py`，每个 Scene 使用独立的 `--output` 和 `--manifest`；`Arena Detailed` 的轨道 Collection 只在后台进程链接，不保存回工程。完整录入入口为 `python -B -m datasets.scripts.common.ingest_shared_route_backgrounds --all-frames`。

逐图库对 576 张写入八任务完整空负标签。按工程相机矩阵和 4.2 m 地面得到 564 张有足够接地面积的源图，作为 train 的人物、电动车、禁行牌及桶状态 `composite_background` 资格；其余 12 张保留空负用途。楼名和仪表按通用空场用途取得 576 张 `composite_background` 资格。八条轨道和三个 Scene 仍属于同一 Blender 工程父来源，近邻帧不是独立场景。这批纯空场对火焰和垃圾贴纸仅有 `empty_negative` 资格。随后分别补入 24 张 C 楼无火焰可贴立面图，以及 24 张无贴纸四色桶面图，逐图库均取得 train `composite_background` 资格。火焰批通过 `python -B -m datasets.scripts.fire.ingest_blender_facade_backgrounds` 入库；贴纸批使用 `waste_sticker/ingest_sticker_surface_backgrounds.py`，并在 24 张上各完成一次实际贴放和 YOLO 图签核对。共享池主清单保留原有 200 条记录，并通过五份扩展清单连接 624 条新渲染图；逐图库是当前任务资格的权威。桶任务的动态选源入口已读取全部 564 张新增合格帧，并完成一张实际合成的图签小样。火焰现行入口已读取完整直接实拍及实拍立面贴放记录，并调用独立 Fire Buildings 场景渲染；C 楼底图四边形仍作为已入库库存保留。本轮火焰三分支图签见 samples/fire/manifest.json。其他任务按各自入口核对资格消费。

从工作空间根运行。首批验证可给前两条命令加 `--pilot`，随后运行全量命令。任务目标源和共享背景池均从既有清单与审查证据重建，重复执行校验已有副本。

```powershell
.venv/Scripts/python.exe -B -m datasets.scripts.common.materialize_target_sources
.venv/Scripts/python.exe -B -m datasets.scripts.common.background_pool --write
.venv/Scripts/python.exe -B -m datasets.scripts.common.package_direct_real --task person --id car_real_capture_20261112:P02_PS1
.venv/Scripts/python.exe -B datasets/scripts/sign/plan_sign_sources.py --budget 24 --shared-background-pool --dry-run
.venv/Scripts/python.exe -B lab/scripts/media/media_index_sqlite.py evidence --capture-identity car_real_capture_20261112:video:235
```

`source.yaml` 的 `target_working_manifest` 和 `background_pool_manifest` 分别指向任务工作清单和共享池。公共直接实拍打包按任务和 split 读取逐图库的当前合格相机帧，补入未列于旧清单的完整正例和全帧空负例。已有清单随记录追溯，工作副本按图像散列和捕获身份核对。合成按所选记录读取几何和标签。[来源目录](../README.md#原始来源定位) · [按需刷新查询](#从现有数据库取得当前输入)

## 背景审查入口

共享模块 `datasets/scripts/common/review/` 提供录入阶段的地面、可放区域和任务目标审查能力。现有 `plan`、`analyze`、`apply` 读取审查计划并保存机器证据，人物目前还通过分发器接入计划。入库端按受影响的图像、证据范围和算法版本批量补算，将新结论及资格写入逐图持久库；已有标签、人工字段和历史批次回执按原范围保留。[逐图入库与资格查询](#逐图入库与资格查询)

按原帧读取已录入的放置几何、语义判定和已有任务标签。证据满足本任务用途的帧可立即进入配额计划。常规合成底图按当前资格选用，生成时保留并变换范围内非主体目标的完整标签。漏标扫描在缺少任务模板时记录 `emptiness_unproven`。该理由影响需要全图空场证据的用途。几何估计失败影响需要相应放置区域的用途。[背景图入库与复用](../../docs/DATASET_DESIGN.md#背景图入库与复用)

旧计划的 `apply` 保存机器空场观测、实际搜索尺度和局部放置区域。人物审查计划用 `emptiness.reference_frame_height` 声明模板参考画高，低尺度命中后仍计算已启用的视面，局部窗口结果按覆盖范围使用。当前人物计划使用入库的标签与空间结论，刷新分发快照走[逐图库接口](#人物最小生产路径)。

地面净空门禁 `min_patch_free_ratio`、`max_patch_obstacle_components` 仍作为兼容预筛，默认值 0 表示不启用。放置区域杂物门禁由 `review/placement.py` 实现：`patch_generated` 几何阶段之后检查 patch 内对象占用、最大对象组件、垂直结构、最大自由矩形、可用脚点比例和最佳脚点身体框；`placement.required` 未写时保持旧行为。`placement_patch_cluttered` 判 `excluded`，`placement_patch_unsure` 判 `unknown`；薄长地面标线单独记录 `marking_ratio`，与本任务已有目标框重叠的组件记为 `task_target`，都不算杂物。具体参数由本批任务计划给出。

地面几何在 VP patch 失败时按顺序回退：先尝试水平相机 patch（`placement_patch_level_camera`），再尝试 Gabor/FFT 纹理地面顶边（`horizon_texture_fallback`、`placement_patch_texture_fallback`），并要求回退 patch 达到最小深度，避免把只剩窄地面的帧硬救回来。`texture.py` 提供 Gabor 滤波器组与一维纹理变点搜索；回退只在主路径失败时运行，reason code 保留使用过的路径。

地面区域已估出而全局可放性未通过时，`near_floor_local_support_v2` 再从原区域内提取逐列连通的近场接地候选。候选、障碍及未知像素均以原图坐标保存，后续布局按实际接地尺寸读掩膜。全局区域已通过时沿用原空间结果。对选定地面任务和 split 的待判合成记录，可在入库端按原文件名补判；需要重建旧坐标时把 `--frame` 换成 `--native-frame`。命令写入受影响的机器观测和合成资格，返回前后 `record_id` 和原因。完整空场的负例资格独立读取其标签证据。[定向补判入口](common/backfill_frame_environment.py) · [空间记录格式](common/review/placement_record.py)

```powershell
.venv/Scripts/python.exe -X utf8 -B -m datasets.scripts.common.backfill_frame_environment refresh-ground --frame e_bike train $SourceFrameName
```

审查入口将证据写入 `annotated/common/` 与任务 `config/`。

## 背景与来源总账

`datasets/scripts/common/source_ledger/` 从 `pairing`、批次清单、共享背景池和 Blender 登记库生成查询汇总，报告写入 `datasets/annotated/common/sources/ledger.json`。配对声明用 `python lab/scripts/contracts/source_pairing.py --check` 检查；总账用于查看派生诊断。

```powershell
.venv/Scripts/python.exe -m datasets.scripts.common.source_ledger.cli check
.venv/Scripts/python.exe -m datasets.scripts.common.source_ledger.cli write
.venv/Scripts/python.exe -m datasets.scripts.common.source_ledger.cli query --task person --role composite_background --split train
.venv/Scripts/python.exe -m datasets.scripts.common.source_ledger.cli matrix
```

检查码、背景可用性规则、输出字段和能力边界见[背景与来源总账](../../docs/BACKGROUND_SOURCE_LEDGER.md)。

## 公共能力和输出

`common/bayesian.py` 复用现有分配器及生产规划器，`perspective.py`、`images.py`、`composite.py`、`annotations.py`、`paper.py`、`blender_standees.py`、`blender_masks.py`、`blender_photo.py` 提供共享变换、成像和标签操作，`placement.py` 提供放置区域杂物评估，`texture.py` 提供几何回退用的 Gabor/FFT 纹理边界。Blender 缺 PyYAML/SciPy 时沿用工作空间解释器回退，不加载外部脚本。任务目标与组合读取相应任务配置条目，实际选源与布局由任务入口和批次计划决定。

人物计划支持可选状态及小于候选状态数的预算；楼名 Blender 外部计划支持弹性 `min_count/max_count/budget`，实际按 `planned_count` 执行。默认固定覆盖计划仍按固定配额执行，不能将固定配额描述为先验改变了分配。

小样写入 `datasets/samples/<task>`，正式数据写入 `datasets/exports/<task>/{train,val,test}`。按任务入口支持的集合运行，候选专用配方输出小样。已接入的生成入口通过生产守卫登记任务 YAML、实际输入、代码及输出。计划命令打印 JSON 或写入该任务配置计划区。
