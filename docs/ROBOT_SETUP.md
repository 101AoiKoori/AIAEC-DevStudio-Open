# NoMachine 与实车开发流程

厂商说明书给出的平台是 Ubuntu 20.04、ROS Noetic；NoMachine 用于远程连接桌面。先让电脑与小车在同一网络，查看车身屏幕显示的 IP，在 NoMachine 建立连接。账号、密码、IP 以现场设备为准，不写入仓库。厂商手册描述了热点模式；该模式下可远程连接但可能不能联网。

在小车 Ubuntu 终端调查：

```bash
lsb_release -a
echo "$ROS_DISTRO"
python3 --version
ls /opt/ros/noetic/setup.bash
rostopic list
rosnode list
rosservice list
rostopic info /scan
rostopic info /odom
rostopic info /camera/color/image_raw
ls /dev/video* 2>/dev/null
```

相机 topic 名只是探测示例，先以 `rostopic list` 的实际结果为准。还要确认 `/move_base` action、`/amcl_pose`、TF 的 `map -> base_footprint`、雷达 `/scan`、TTS、红外设备以及开机启动项。将结果填入 `docs/HARDWARE_INVENTORY.md`，不要记录密码。

小车热点与互联网隔离时，使用 [离线交付流程](OFFLINE_DEPLOY.md)。开发机不必连接小车，车载 Ubuntu 也不必联网。首次运行先检查 topic、地图定位和低速导航，最后才进行全程一键启动。

厂商的 MonkeyCar HTTP API 文档列出 `/car/mode` 等接口，但不能据此认定 ROS 导航和这些接口可以同时接管底盘。正式控制链以实车 ROS 节点和服务调查为准，避免双重控制。
