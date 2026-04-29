## ADDED Requirements

### Requirement: OpenAI 兼容文本补全接口

系统 SHALL 暴露与 OpenAI 兼容的文本补全接口，以便现有客户端可以直接接入。

#### Scenario: completions 请求
- **GIVEN** 服务已启动
- **WHEN** 客户端向 `POST /v1/completions` 提交 prompt 与生成参数
- **THEN** 服务 SHALL 返回包含 `id`、`object`、`created`、`model` 和 `choices` 的 JSON 响应

#### Scenario: completions streaming
- **GIVEN** 客户端向 `POST /v1/completions` 提交 `stream=true`
- **WHEN** 服务生成文本
- **THEN** 服务 SHALL 以 `text/event-stream` 返回增量事件并在结束时发送 `[DONE]`

### Requirement: OpenAI 兼容聊天补全接口

系统 SHALL 暴露与 OpenAI 兼容的聊天补全接口，以便聊天型客户端可以直接接入。

#### Scenario: chat completions 请求
- **GIVEN** 服务已启动
- **WHEN** 客户端向 `POST /v1/chat/completions` 提交消息列表与生成参数
- **THEN** 服务 SHALL 返回 assistant 消息格式的 JSON 响应

#### Scenario: chat completions streaming
- **GIVEN** 客户端向 `POST /v1/chat/completions` 提交 `stream=true`
- **WHEN** 服务生成回复
- **THEN** 服务 SHALL 以 SSE 增量返回 assistant delta 并在结束时发送 `[DONE]`

### Requirement: 健康检查与指标暴露

系统 SHALL 暴露最小服务运维接口，以支持健康探测和监控接入。

#### Scenario: liveness 检查
- **GIVEN** 服务进程正常运行
- **WHEN** 客户端请求 `GET /healthz`
- **THEN** 服务 SHALL 返回成功状态

#### Scenario: readiness 检查
- **GIVEN** 服务已完成配置与运行时初始化
- **WHEN** 客户端请求 `GET /readyz`
- **THEN** 服务 SHALL 返回成功状态

#### Scenario: metrics 暴露
- **GIVEN** 服务正在处理请求
- **WHEN** 监控系统请求 `GET /metrics`
- **THEN** 服务 SHALL 返回 Prometheus 文本格式的最小请求指标
