## MODIFIED Requirements

### Requirement: 配置与初始化

系统 SHALL 支持服务模式与 tokenizer/runtime 选择配置，以便同一代码库既可运行本地 mock 引擎，也可运行可桥接真实后端的服务模式。

#### Scenario: 服务运行模式配置
- **GIVEN** 部署人员需要以 HTTP 服务方式启动系统
- **WHEN** 系统加载配置
- **THEN** 推理引擎 SHALL 支持服务监听地址、端口和后端运行时配置

#### Scenario: 运行时后端选择
- **GIVEN** 系统以服务模式启动
- **WHEN** 加载服务后端配置
- **THEN** 系统 SHALL 支持在本地引擎模式和命令桥接模式之间进行选择

### Requirement: 分词

系统 SHALL 同时支持测试用 simple tokenizer 与可装载的真实 tokenizer，以确保控制面验证与真实部署都可运行。

#### Scenario: 真实 tokenizer 装载
- **GIVEN** 配置指定 HuggingFace tokenizer JSON 文件
- **WHEN** 系统初始化 tokenizer
- **THEN** 系统 SHALL 从该文件装载 tokenizer 并用于请求编码/解码

#### Scenario: tokenizer 装载失败
- **GIVEN** 配置指定的 tokenizer 文件不存在或格式无效
- **WHEN** 系统初始化 tokenizer
- **THEN** 系统 SHALL 返回显式错误并拒绝启动服务

## ADDED Requirements

### Requirement: 服务运行时桥接

系统 SHALL 提供服务层运行时适配，使其既可调用本地 `InferenceEngine`，也可桥接到外部命令式后端。

#### Scenario: 本地引擎后端
- **GIVEN** 服务后端配置为本地引擎模式
- **WHEN** 服务收到文本生成请求
- **THEN** 服务 SHALL 通过本地 `InferenceEngine` 处理请求并返回生成结果

#### Scenario: 命令桥接后端
- **GIVEN** 服务后端配置为命令桥接模式
- **WHEN** 服务收到文本生成请求
- **THEN** 服务 SHALL 调用外部命令并将其标准输出映射为生成结果

#### Scenario: 命令桥接失败
- **GIVEN** 命令桥接进程启动失败或返回非零退出码
- **WHEN** 服务处理请求
- **THEN** 服务 SHALL 返回描述性错误并记录失败信息
