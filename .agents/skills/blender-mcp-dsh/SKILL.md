---
name: blender-mcp-dsh
description: "需要连接正在运行的 Blender 执行 bpy 代码、渲染导出帧、查询场景对象或核对 Blender 资产时使用；本 skill 只服务 DSH（profile patch 注册 + lab/scripts/blender/blender_mcp_bridge.py 桥接 localhost:9876），与 Codex/Claude 的 blender-mcp 配置格式不通用。"
---

# Blender MCP 接入（DSH 专用）

本工作空间的 Blender 连接由 DSH 建立：DSH 的 stdio MCP 客户端启动 `lab/scripts/blender/blender_mcp_bridge.py`，桥接器再连接 Blender 官方 MCP 扩展的 TCP socket。原始的那套 blender-mcp 是给 Codex 用的，帧协议不同，两套配置不能互抄。

## 协议身份：先分清是哪一套

| | 本工作空间（DSH） | Codex 那套（ahujasid/blender-mcp 等） |
| --- | --- | --- |
| 扩展 | Blender 官方 `MCP`（Blender Lab，≥5.1） | 社区 addon |
| 帧格式 | **null 字节（`\0`）分隔** | 换行分隔 |
| 请求体 | `{"type":"execute","code":...,"strict_json":true}` | `{"type":"get_scene_info"}` 等具名指令 |
| 客户端 | `@deepseek-ai/dsh-mcp-client` + `lab/scripts/blender/blender_mcp_bridge.py` | `uvx blender-mcp` |

本机实测版本：Blender 5.2.3 LTS，扩展 `mcp` 1.0.3，默认 `localhost:9876`。

## 前置条件：按顺序核对

1. Blender 已启动，Preferences → Extensions → MCP 显示 `Server is running`，Host/Port 与 `BLENDER_MCP_HOST` / `BLENDER_MCP_PORT` 一致（默认 `localhost` / `9876`）。
2. 桥接器直连能通：
   ```
   python lab/scripts/blender/blender_mcp_bridge.py --code "import bpy; result = {'v': bpy.app.version_string}"
   ```
   返回 `{"status": "ok", ...}` 即协议层正常。
3. DSH 侧注册存在（见下节），工具名 `mcp__blender__blender_execute`、`mcp__blender__blender_status`。

一条命令检查全链路：`python .agents/skills/blender-mcp-dsh/scripts/check_connection.py`。

## 注册位置：DSH 只有这一个正确入口

写 `%USERPROFILE%\.dsh\profiles\<profile>\cordis.patch.yml`（Desktop web 环境为 `profiles\web\`）。插件 `@deepseek-ai/dsh-mcp-client` 随 DSH 安装，不需要另外装。

```yaml
- insert:
    - id: mcp-blender
      name: '@deepseek-ai/dsh-mcp-client'
      config:
        serverName: blender          # 只允许 [A-Za-z0-9_-]{1,32}
        transport: stdio
        command: '<python 解释器绝对路径>'
        args:
          - '<工作空间绝对路径>/lab/scripts/blender/blender_mcp_bridge.py'
        toolCallTimeoutMs: 600000    # 渲染会长时间占用，按最慢任务给
```

- profile patch **热重载**，保存后工具即可用，不需要重启 DSH（本机实测）。
- 改前备份 `cordis.patch.yml`；改后用 node 的 `yaml` 解析确认能 parse，再交给运行的 GUI。
- `save_capability` / `list_capabilities` 是团队 Bot 能力注册，普通会话不可用；不要手工改 `~/.dsh/bot/state.json`。
- 本 skill 不提供 `agents/openai.yaml`：该文件是 OpenAI/Codex 侧接口描述，本 skill 只服务 DSH，避免被别的运行时误加载。

## 工具用法

- `blender_execute(code, timeout_seconds?)`：在 Blender 内 `exec` 代码，代码必须给 `result` 赋**可 JSON 序列化**的 dict；`print` 的内容单独在 `stdout` 返回。渲染等长任务显式传 `timeout_seconds`。
- `blender_status()`：版本、打开的 .blend、场景、当前帧、帧范围、引擎、分辨率、相机。

## 已知行为与实测坑

- 一次连接只处理一次请求，服务端写完响应即关闭；请求必须带 `\0` 结束符。
- 只连上不返回，通常是漏了 `\0` 或 `strict_json` 字段；未完成请求约 10 s 被驱逐并返回 `Client timed out`。
- 请求上限 10 MiB；响应非阻塞分片发送，大结果（base64 截图）要有上限意识。
- 弱沙箱拦截：`sys.exit`、`bpy.ops.wm.quit_blender`、`bpy.ops.wm.read_factory_settings`、`bpy.ops.wm.read_userpref`、`bpy.ops.wm.read_factory_userpref`。其余 `bpy` 基本不限制，能改用户文件，操作前自己核对路径。
- `bpy.ops.wm.open_mainfile()` 可以在 execute 内直接调用，加载后同一次执行里继续取 `bpy.data` 正常；扩展的 timer 是运行时状态，不随 .blend 加载消失。
- 取动画帧必须 `scene.frame_set(f)` 后再 `obj.evaluated_get(depsgraph)`；直接读 `obj.data` 会拿到旧值。
- 渲染阻塞 Blender 主线程直至完成，响应在渲染结束后才回；本机该资产 Cycles 24 samples / 1024² 约 9.5–10.1 s/帧。
- 后台模式入口不同：`blender --background file.blend --command blender_mcp --host --port`；该模式不支持 `check_is_finished` 延迟响应。

## 工作空间约定：不要绕过

- 正式数据生成入口是 `datasets/scripts/<task>/`（waste_sticker 为 `render_sticker_arena_v4.py`），走 `production_guard` 写回执。用 MCP 做的单帧预览、目视检查**不是**数据集批次，不写回执、不进 `datasets/samples/` 或 `exports/`。
- 用户指定位置的检查产物放明确命名的子目录；`datasets/assets/blender/` 已被 Git 忽略，二进制不提交。
- Blender 资产按 `datasets/blender_source_registry.sqlite3` + `lab/scripts/blender/blender_source_registry.py` 查状态，目录列表不代表生命周期。
- 涉及数据素材生成的运行另按实验治理记录来源与标签口径。

## 资产内贴纸类别的查询方式（waste_sticker 资产）

- 四个桶各有一个 `<color> item sticker paper` 网格，24 个材质槽，`material_index` 按帧打关键帧。
- 材质带自定义属性 `class_id`、`source_artwork`，材质名 `waste_sticker_<id>_<中文名>`；帧 → 类别以 `datasets/annotated/...` 之外的计划文件为准：`lab/features/waste_bin/training/config/waste_sticker_source_plan_20261114.json` 的 `samples[frame-1].bins`。
- 该资产相机只有 3 个关键帧（x = -1.4 → 0 → 1.4）：帧 1 四桶全在画面，帧 120/240 只有绿、灰桶；整幅 1024² 下贴纸约 40 px，需要放大或近距相机才能判读。
- 本机已核对：240 帧 × 4 桶 = 960 条，材质求值与计划调度零不一致，24 类全覆盖。

## 失败处理

| 现象 | 判断 |
| --- | --- |
| `ConnectionRefusedError` | Blender 未启动，或扩展的 Server 未 running，或端口不符 |
| 连接成功但无响应 | 请求缺 `\0`；或 `strict_json` 缺失/非布尔 |
| `Client timed out` | 未在 10 s 内发完整个请求 |
| `status: error` + traceback | Blender 内代码异常；按 traceback 修代码，不要改协议 |
| 工具不在列表里 | profile patch 未保存成功/未热重载；用 check_connection.py 核对注册行 |
| 渲染超时 | 提高 `toolCallTimeoutMs` 与 `timeout_seconds`，或降采样/换 EEVEE |

## 完成后的记录

改变连接方式、端口、桥接器行为或注册位置时，更新本 skill 的对应小节；只做一次性检查时不必留档，把结论写在当次回复里。
