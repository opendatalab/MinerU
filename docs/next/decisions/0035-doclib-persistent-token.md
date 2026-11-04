# ADR-0035: Doclib 持久本机凭证

状态: Accepted，已实现
日期: 2026-11-04

## 背景

Doclib 是本机单用户文档库，拥有服务账户的文件访问权限。UDS 的文件权限可以隔离其他账户，
TCP loopback 仅限制连接来源机器。读取正文、导出到指定路径和停止服务都需要统一的访问边界。
二次开发程序需要稳定凭证，不能要求每次服务重启后重新配置。

## 决策

- 标准启动首次生成 32 字节随机 token，并在配置锁内写入实际 `config.yaml` 的 `doclib.auth.token`。
  已有值保持不变，普通配置读取和 SDK 构造不生成文件。环境变量 `MINERU_DOCLIB_AUTH_TOKEN`
  按既有配置优先级覆盖文件值，不回写环境注入的凭证。
- token 属于启动配置，使用 `SecretStr` 隐藏 repr/JSON 正文；不进入 SQLite 运行时配置接口。
  轮换采用 stop → 修改/删除配置值 → start，不热更新正在服务的实例。
- 全部 Doclib HTTP 接口使用 Bearer 认证，UDS/TCP 一致；状态、shutdown、OpenAPI 不例外。
  在业务路由执行前拒绝缺失、重复或错误的 Authorization，不提供匿名降级。
- 配置正文写入前限制临时文件权限，再原子替换。POSIX 使用 `0600`，Windows 使用仅授权当前账户
  与 SYSTEM 的受保护 DACL；macOS 同时清除扩展 ACL，并在创建时禁止继承 ACL。设置后读回验证，
  父目录的权限位及平台 ACL 不能让其他普通账户替换文件。无法落实权限时失败。
- CLI/SDK 默认发现重新读取当前配置中的 token；显式地址必须同时提供显式 token。
  endpoint 升为 version 3，不存 token；`server_id` 继续每次启动变化。旧 version 1/2 仅保留诊断读取，
  SDK 返回 `doclib_upgrade_required`，升级前须先用旧 CLI 停止旧服务。
- 显式 `create_app(cfg)` 和 `DoclibServer` 要求宿主提供 token，保持实例配置独立。
- 保留认证调用者的扫描、路径读取、导出和 shutdown 能力。不增加用户角色、目录白名单或导出覆盖策略。
  Router 的可信内网共享模式不在本次变更范围内。

## 替代方案

- 每次启动轮换：会打断显式 TCP 集成，未采用。
- 仅 TCP 认证：会让两个 transport 的应用契约不同，未采用。
- token 复制到 endpoint：增加凭证副本及权限维护，未采用。
- 完整多用户权限系统：超出本机单用户产品边界，未采用。

## 影响

替代 [ADR-0021](0021-doclib-local-transport-discovery.md) 的“不实现 token”和旧 endpoint 连接决策。
现行构造参数、错误及二次开发示例见 [Doclib SDK](../sdk/doclib-client.md)。
验收覆盖持久化/并发初始化、无认证请求无副作用、SDK 自动携带/显式地址隔离、重启/轮换、文件权限和旧版本拒绝。
