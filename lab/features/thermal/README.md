> 自动生成：请修改 `config/perception_tasks.json`，然后运行 `python lab/scripts/contracts/sync_docs.py`。

**识别摘要：** D 楼热异常位置及楼层 [14]

**分值：** 10

**调度方案：** 后台处理（待接入调度）。采集并保留红外数据，后台判断热点及楼层。

[开源项目分析与方法研究](../../../docs/research/OPEN_SOURCE_METHODS.md#thermal)

# D 楼热异常楼层识别

规则第 14、23、25 页，输出异常楼层及对应框图。D 楼为三列四层，中列有一根竖直发热电阻。用一个红外设备观测，按楼层报告异常。

采样与处理按[定点正对采样](../../../docs/PROJECT_DIRECTION.md#定点正对采样)执行。

事项标签见[需求状态约定](../../../docs/PERCEPTION_DEVELOPMENT.md#需求状态与提醒)。

## 需要实现

- `[已明确]` THERMAL-R01：处理粒度为楼层，保存四层各自热状态；三列只用于确定中央热源观测带，输出不含列号。
- `[待解决]` THERMAL-R02：在红外图定位 D 楼中央热源带及四层边界，分别计算每层相对背景的热响应，输出异常楼层集合及状态。
- `[待解决]` THERMAL-R03：在红外图框选对应热点及异常楼层，终端与播报报告每个已确认异常楼层。
- `[待解决]` THERMAL-R04：中央电阻可能形成连续竖直热区，按四层区域分别判定，不将连通热区数量当楼层数，不假定只能一层异常。

## 状态与提醒

- `[已明确]` THERMAL-FACT：采集使用一个红外设备，中央竖直电阻为热源观测带。
- `[待解决]` THERMAL-THRESHOLD：按冷态与热态数据确定每层相对背景阈值和边界余量，未知时保留分项状态。
- `[待全局规划]` THERMAL-OBSERVE：接入 D 楼观察姿态、传感器时间和楼层映射配置。

## 工作区与验收

共享核心为 `ws/src/aec_vision/src/aec_vision/thermal.py`；采集原件按批次放入 `Archive/Assets/captures/thermal/`，逐层标签和审查记录归入 `datasets/annotated/thermal/`。逐层记录人工真值、冷态/热态、时间及热点图，保留连续竖直热区和跨层边界样本。方案见 [热异常设计](../../../docs/research/OPEN_SOURCE_METHODS.md#thermal)。
