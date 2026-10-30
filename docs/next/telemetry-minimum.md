# Doclib 遥测

实现入口：`mineru/doclib/telemetry/service.py`、`store.py`、`constants.py`、
`mineru/doclib/background/telemetry_flush.py` 和 `mineru/cli/telemetry.py`。

## 范围与状态

遥测只属于 Doclib。CLI 和 DoclibClient 可传递调用上下文，聚合和上报由 Doclib 服务执行。
独立 parser SDK、`mineru-kit parse` 和 V1 API Server 不使用这套服务。

状态为 enabled 或 disabled，持久化键为 telemetry_state；state 表示生效开关，source 说明设置来源。

| 场景 | 行为 |
|---|---|
| 新安装 | enabled，source=default |
| 用户 enable/disable | 保存选择，source=user；升级和重启保持 |
| 旧 consent_state=unset | 一次性迁移为 enabled/default，删除旧键 |
| 旧 enabled/disabled | 保持开关并迁移到新键，source=user |
| 状态损坏、缺失或读取失败 | 暂停采集与上报，source=error，记录诊断；显式选择可修复 |
| 环境禁用 | disabled，source=environment；不写回用户偏好 |

```bash
mineru telemetry status --json
mineru telemetry preview --json
mineru telemetry disable --json
mineru telemetry enable --json
mineru telemetry flush --json
```

status 报告实际生效的 state/source、installation_id、pending_periods/pending_metrics、last_flush_at，
以及 notice_shown。source 可为 default/user/environment/error。只有实际 enabled 才能采集和发送。

## 启动前禁用

```bash
MINERU_TELEMETRY_DISABLED=1 mineru server start
```

开关在 Doclib 服务构造时读取；修改后需要重启已经运行的服务。
1/true/yes/on（忽略大小写与两侧空白）禁用；空值或 0/false/no/off 不覆盖持久化选择。
无法识别的非空值按禁用处理并记录诊断。该变量不属于 YAML 配置字段。
环境禁用时 enable 仍保存用户开启偏好，但实际状态保持 disabled，命令会说明环境覆盖。

## 关闭与并发

关闭在同一数据库事务中保存 disabled/user 并清空待发聚合；失败则整体回滚。
采集、开关修改和发送调度共享服务实例的并发控制。
关闭会取消并等待正在执行的 flush 退出后才返回成功，后续批次不再发送；已经发送到服务器的数据无法撤回。
手动与后台 flush 使用同一门面，关闭不会终止后台调度循环；再次 enable 后可继续工作。
关闭期间不采集，重新开启不补录关闭期间事件，也不恢复已清空的待发数据。

## 一次性说明

交互式业务命令在遥测实际开启且尚未告知时，向 stderr 展示默认策略、preview、disable 和启动前禁用方式。
不等待输入，不调用 enable。展示后通过 notice-shown action 保存标记；每个 Doclib 数据库独立记录。
CI、Agent、JSON、帮助、非交互或已关闭状态不展示。告知请求失败不影响原业务命令，之后允许重试告知。

## 数据与传输

指标名称和维度在 constants.py 的显式白名单内，耗时、页数和文件大小使用聚合或分桶。
payload 包含 installation_id、版本、粗粒度环境和聚合指标，不发送文档正文、文件名、路径、搜索词、API Key 或 traceback。
preview 读取最早一个 period，没有聚合时给出空预览，不发送网络请求。
后台启动后尝试 flush，之后默认每 7200 秒运行；每轮最多处理 10 个 period。
成功或明确丢弃的批次删除，可重试失败保留；取消、异常与正常完成都会释放发送锁。
