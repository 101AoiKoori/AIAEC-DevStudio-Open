# 省赛迁移参考

来源仓库与位置见[工作空间入口](../../README.md#先做什么)。

重点查看 `ws/src/bag/scripts/{auto,navigation,motion,tasks,vision}.py`、`config/waypoints.yaml`、`launch/auto.launch`。可复用架构与算法，但国赛新增任务、禁行分支、实体沙盘、5 分钟时限和传感器均需重新验证。选定源码核对来源和许可后复制到本工作空间独立维护；国赛地图与路线须现场标定。
