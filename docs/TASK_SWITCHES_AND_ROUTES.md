# 识别任务开关与路线配置

配置入口是部署包内的 [`competition.json`](../ws/src/aec_vision/config/competition.json)。九个 `tasks.<任务名>.enabled` 使用 JSON 布尔值 `true` / `false`；例如只研究仪表时可把 `meter.enabled` 改为 `true`，其余保持 `false`。该开关设计允许识别程序读取启用任务并只初始化所需模型；任务调度器尚未实现，因此修改配置不会自动启动识别程序。

## 区分单任务调试和行驶

`aec_vision.runtime_config.load_config(path)` 读取配置，`enabled_tasks(config)` 返回启用的任务集合；可用于图片实验、离线测试或不发起移动的定点测试。

`prepare_route(config, route_id=None)` 只生成经过检查的路线数据，不连接 ROS、不发送导航目标。它在准备行驶路线时要求红绿灯和指示牌都启用；还要求选择已存在的非空路线，并且每个点已现场标定、坐标和角度齐全。其他被关闭的识别任务会从每个点的任务清单中过滤。此检查是必要的配置门，不代表已经具备红绿灯、禁行牌识别能力或可安全行驶；未来导航入口必须主动调用并处理检查结果。

## 多条路线与 waypoint

`routes` 可定义多个以 ID 区分的候选路线，`selected_route` 供测试选用；也可在决策模块确认分支后向 `prepare_route` 传入候选 `route_id`。现场指示牌决定分支，默认选择或命令行参数不能替代禁行判断；实际分支事件、路线切换和停止逻辑需与导航负责人对接。

一个配置示例（字段演示，不能用于现场）：

```json
{
  "id": "observe_a",
  "pose": {"x": null, "y": null, "yaw": null},
  "calibrated": false,
  "tasks": [
    {"id": "a_people", "type": "person", "context": {"street": "A"}},
    {"id": "a_bikes", "type": "e_bike", "context": {"street": "A"}}
  ]
}
```

将点加入 `routes.route_a.waypoints` 等数组并按实际经过顺序排列；`pose` 是 map 坐标系的米和弧度。现场实测并核对后填写有限数值、将 `calibrated` 改为 `true`。同一点可声明多个任务实例，`id` 在该点内唯一，`type` 是九种任务类型，`context` 记录街区或车位等场景信息；具体字段由后续识别接口定义。不同路线可有不同的点和顺序。允许不同点设置相同任务实例 ID，但汇总时必须同时使用路线、点位和实例身份，不能只用实例 ID 去重。

默认两条路线为空、开关全关闭、未选路线，因此不能生成可行驶计划。路线 A/B 是配置占位，尚未对应现场两种禁行摆位，也未包含路障、点位朝向切换、分支重入或识别目标朝向等完整导航语义。这些需要导航组共同确定后再扩展，不能直接用当前结构驱动真实小车。

## 检查

在开发机仓库根目录运行：

```powershell
python ws/src/aec_vision/tests/test_runtime_config.py
```

测试覆盖开关、分支选择、无标定和关键识别关闭等情况；Ubuntu/ROS 的节点启动、真实任务过滤和导航联动仍需后续实现并在小车上测试。调试一项识别时，必须实际调用 `enabled_tasks` 并由调度器遵守结果，当前仅提供配置读取和筛选组件。
