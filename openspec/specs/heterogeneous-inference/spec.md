# 异构推理系统规格说明

## Purpose

本文档定义异构推理引擎的行为规格——一个利用 CPU-GPU 协同执行的高性能 LLM 推理微服务。系统实现 PagedAttention 进行高效的 KV Cache 管理，以及 Continuous Batching 实现最优吞吐量。

## Requirements

### Requirement: 请求管理

系统 SHALL 接受推理请求并返回生成的文本响应。用户故事：作为客户端，我希望向系统提交推理请求并接收生成的文本响应。

#### Scenario: 提交推理请求
- **GIVEN** 推理引擎已初始化
- **WHEN** 客户端提交包含输入文本和生成参数的请求
- **THEN** 推理引擎 SHALL 对输入进行分词并创建待处理请求条目

#### Scenario: 请求 ID 分配
- **GIVEN** 一个新请求被创建
- **WHEN** 调度器处理该请求
- **THEN** 调度器 SHALL 分配唯一的序列 ID 并初始化请求状态

#### Scenario: 参数验证
- **GIVEN** 生成参数包括 max_tokens、temperature 和 top_p
- **WHEN** 推理引擎验证参数
- **THEN** 推理引擎 SHALL 验证参数在可接受范围内

#### Scenario: 无效参数处理
- **GIVEN** 请求包含无效参数
- **WHEN** 推理引擎处理该请求
- **THEN** 推理引擎 SHALL 返回描述性错误而不处理该请求

#### Scenario: 请求完成
- **GIVEN** 请求完成生成
- **WHEN** 推理引擎处理完成
- **THEN** 推理引擎 SHALL 解码输出并将响应返回给客户端

---

### Requirement: PagedAttention KV Cache 管理

系统 SHALL 通过分页内存管理实现高效的 GPU 内存利用。用户故事：作为系统运维人员，我希望高效的 GPU 内存利用以服务更多并发请求。

#### Scenario: 内存分区
- **GIVEN** 系统启动
- **WHEN** KV Cache 管理器初始化
- **THEN** KV Cache 管理器 SHALL 将 GPU 内存划分为固定大小的物理块（例如每块 16 个 token）

#### Scenario: 新序列分配
- **GIVEN** 一个新序列开始
- **WHEN** KV Cache 管理器处理该序列
- **THEN** KV Cache 管理器 SHALL 通过页表分配逻辑块并映射到可用物理块

#### Scenario: 块容量扩展
- **GIVEN** 序列生成的 token 超过当前块容量
- **WHEN** KV Cache 管理器检测到容量不足
- **THEN** KV Cache 管理器 SHALL 按需分配额外的物理块

#### Scenario: 序列完成释放
- **GIVEN** 一个序列完成
- **WHEN** KV Cache 管理器处理完成事件
- **THEN** KV Cache 管理器 SHALL 将所有物理块释放回空闲池

#### Scenario: 块使用跟踪
- **GIVEN** 系统运行中
- **WHEN** KV Cache 管理器管理内存
- **THEN** KV Cache 管理器 SHALL 维护空闲块列表并跟踪每个序列的块使用情况

#### Scenario: 内存压力信号
- **GIVEN** 没有可用物理块
- **WHEN** KV Cache 管理器收到分配请求
- **THEN** KV Cache 管理器 SHALL 向调度器发出内存压力信号

#### Scenario: 块查找效率
- **GIVEN** 任意序列
- **WHEN** 需要访问 KV Cache
- **THEN** KV Cache 管理器 SHALL 提供从逻辑块索引到物理块指针的 O(1) 查找

---

### Requirement: Continuous Batching 调度器

系统 SHALL 通过连续批处理最大化 GPU 利用率。用户故事：作为系统运维人员，我希望通过连续批处理实现高吞吐量。

#### Scenario: 独立队列维护
- **GIVEN** 调度器运行中
- **WHEN** 管理请求
- **THEN** 调度器 SHALL 为 prefill 和 decode 请求维护独立队列

#### Scenario: 混合批次调度
- **GIVEN** 调度器有 prefill 和 decode 请求
- **WHEN** 调度批次
- **THEN** 调度器 SHALL 将 prefill 和 decode 请求组合到单个 GPU 执行中

#### Scenario: Prefill 完成转换
- **GIVEN** prefill 请求完成
- **WHEN** 调度器处理完成
- **THEN** 调度器 SHALL 立即将其转换为 decode 阶段，无需等待批次完成

#### Scenario: Decode 完成
- **GIVEN** decode 请求生成 EOS token 或达到 max_tokens
- **WHEN** 调度器检测到完成条件
- **THEN** 调度器 SHALL 将其标记为完成并从活动集中移除

#### Scenario: 批次约束
- **GIVEN** 调度器调度批次
- **WHEN** 组装批次
- **THEN** 调度器 SHALL 强制执行最大批次大小和最大总 token 约束

#### Scenario: 连续插入
- **GIVEN** 新请求到达
- **WHEN** 有可用批次槽位
- **THEN** 调度器 SHALL 将新请求插入下一个可用批次槽位

#### Scenario: Decode 优先级
- **GIVEN** decode 请求和 prefill 请求同时待处理
- **WHEN** 调度器调度批次
- **THEN** 调度器 SHALL 优先调度 decode 请求以最小化进行中请求的延迟

---

### Requirement: GPU 内核执行

系统 SHALL 提供优化的 GPU 内核进行注意力计算。用户故事：作为开发者，我希望优化的 GPU 内核以实现快速推理。

#### Scenario: Paged Attention 内核
- **GIVEN** GPU 执行器运行中
- **WHEN** 执行注意力计算
- **THEN** GPU 执行器 SHALL 实现通过块表间接读取 KV Cache 的分页注意力内核

#### Scenario: 变长序列支持
- **GIVEN** 执行注意力计算
- **WHEN** 批次包含不同长度的序列
- **THEN** GPU 执行器 SHALL 支持批次内的变长序列

#### Scenario: 融合操作
- **GIVEN** GPU 执行器运行中
- **WHEN** 执行计算
- **THEN** GPU 执行器 SHALL 实现融合操作以最小化 GPU 内存带宽

#### Scenario: 混合模式处理
- **GIVEN** 批次包含混合 prefill 和 decode 请求
- **WHEN** GPU 执行器执行批次
- **THEN** GPU 执行器 SHALL 适当处理不同的注意力模式

#### Scenario: CUDA Graphs
- **GIVEN** GPU 执行器在 decode 阶段
- **WHEN** 执行内核
- **THEN** GPU 执行器 SHALL 使用 CUDA Graphs 减少内核启动开销

#### Scenario: 数值精度
- **GIVEN** GPU 执行器运行中
- **WHEN** 执行计算
- **THEN** GPU 执行器 SHALL 支持 FP16/BF16 计算并使用 FP32 累加以保证数值稳定性

---

### Requirement: CPU-GPU 流水线协调

系统 SHALL 实现高效的 CPU-GPU 协调以避免瓶颈。用户故事：作为系统架构师，我希望高效的 CPU-GPU 协调以避免任一方成为瓶颈。

#### Scenario: 异步 CUDA 流
- **GIVEN** 推理引擎运行中
- **WHEN** 执行推理
- **THEN** 推理引擎 SHALL 使用异步 CUDA 流重叠 CPU 和 GPU 工作

#### Scenario: 并行执行
- **GIVEN** CPU 准备下一批次
- **WHEN** GPU 执行当前批次
- **THEN** GPU 执行器 SHALL 并发执行

#### Scenario: 锁页内存
- **GIVEN** CPU-GPU 数据传输
- **WHEN** 传输数据
- **THEN** 推理引擎 SHALL 使用锁页主机内存加速传输

#### Scenario: 增量传输
- **GIVEN** 传输批次元数据
- **WHEN** 发送数据到 GPU
- **THEN** 推理引擎 SHALL 仅发送块表更新以最小化传输大小

#### Scenario: 双缓冲
- **GIVEN** 推理引擎运行中
- **WHEN** 准备批次
- **THEN** 推理引擎 SHALL 实现双缓冲批次准备以隐藏延迟

#### Scenario: GPU 停滞处理
- **GIVEN** GPU 执行停滞
- **WHEN** 检测到停滞
- **THEN** 推理引擎 SHALL 记录警告并继续处理

---

### Requirement: 内存池管理

系统 SHALL 提供可预测的内存使用以保持负载稳定。用户故事：作为系统运维人员，我希望可预测的内存使用以保持系统负载稳定。

#### Scenario: 预分配内存
- **GIVEN** 系统启动
- **WHEN** KV Cache 管理器初始化
- **THEN** KV Cache 管理器 SHALL 预分配可配置百分比的 GPU 内存用于 KV Cache 块

#### Scenario: 内存统计跟踪
- **GIVEN** KV Cache 管理器运行中
- **WHEN** 跟踪内存使用
- **THEN** KV Cache 管理器 SHALL 跟踪包括总块数、已用块数和碎片率的内存统计

#### Scenario: 内存阈值控制
- **GIVEN** 内存利用率超过阈值
- **WHEN** 调度器检查内存状态
- **THEN** 调度器 SHALL 停止接受新的 prefill 请求

#### Scenario: 内存指标暴露
- **GIVEN** 系统运行中
- **WHEN** 监控系统查询状态
- **THEN** 推理引擎 SHALL 暴露内存使用指标用于监控

#### Scenario: 内存分配失败处理
- **GIVEN** 内存分配失败
- **WHEN** 处理失败
- **THEN** 推理引擎 SHALL 优雅地拒绝新请求而不是崩溃

---

### Requirement: 配置与初始化

系统 SHALL 支持可配置的系统参数以适应不同硬件和工作负载。用户故事：作为部署人员，我希望可配置的系统参数以针对不同硬件和工作负载进行调优。

#### Scenario: 配置参数
- **GIVEN** 推理引擎初始化
- **WHEN** 加载配置
- **THEN** 推理引擎 SHALL 接受以下配置：block_size、max_num_blocks、max_batch_size、max_num_seqs

#### Scenario: 配置验证
- **GIVEN** 加载配置
- **WHEN** 验证配置参数
- **THEN** 推理引擎 SHALL 验证所有参数并报告错误

#### Scenario: 配置来源
- **GIVEN** 推理引擎初始化
- **WHEN** 加载配置
- **THEN** 推理引擎 SHALL 支持通过文件或命令行参数进行配置

#### Scenario: 初始化日志
- **GIVEN** 系统初始化
- **WHEN** 初始化完成
- **THEN** 推理引擎 SHALL 记录系统配置和检测到的 GPU 能力

#### Scenario: GPU 能力检测
- **GIVEN** 系统初始化
- **WHEN** 检测硬件
- **THEN** 推理引擎 SHALL 检测并报告可用 GPU 内存和计算能力

---

### Requirement: 分词

系统 SHALL 提供准确的文本分词以确保输入正确处理。用户故事：作为客户端，我希望准确的文本分词以确保输入正确处理。

#### Scenario: 文本编码
- **GIVEN** 分词器初始化
- **WHEN** 编码文本
- **THEN** 分词器 SHALL 使用可配置的词汇表将输入文本编码为 token ID

#### Scenario: Token 解码
- **GIVEN** 分词器初始化
- **WHEN** 解码 token
- **THEN** 分词器 SHALL 准确地将 token ID 解码回文本

#### Scenario: 特殊 token 处理
- **GIVEN** 编码文本
- **WHEN** 遇到特殊 token
- **THEN** 分词器 SHALL 正确处理特殊 token（BOS、EOS、PAD）

#### Scenario: 往返属性
- **GIVEN** 任意有效文本输入
- **WHEN** 执行编码后解码
- **THEN** decode(encode(text)) SHALL 产生等效文本

#### Scenario: CPU 执行
- **GIVEN** 分词操作
- **WHEN** 执行分词
- **THEN** 分词器 SHALL 在 CPU 上运行以避免 GPU 内存开销

---

## 正确性属性

以下属性通过属性测试验证：

| 属性 ID | 描述 | 验证需求 |
|---------|------|----------|
| PROP-1 | 请求 ID 唯一性 | REQ-1 |
| PROP-2 | 参数验证正确性 | REQ-1 |
| PROP-3 | 序列开始时的块分配 | REQ-2 |
| PROP-4 | 增长时的块分配 | REQ-2 |
| PROP-5 | 块计数不变量 | REQ-2 |
| PROP-6 | 调度器队列状态一致性 | REQ-3 |
| PROP-7 | 批次大小约束 | REQ-3 |
| PROP-8 | Decode 优先于 Prefill | REQ-3 |
| PROP-9 | Prefill 到 Decode 转换 | REQ-3 |
| PROP-10 | 完成条件 | REQ-3 |
| PROP-11 | 变长序列处理 | REQ-4 |
| PROP-12 | 内存统计不变量 | REQ-6 |
| PROP-13 | 内存压力响应 | REQ-6 |
| PROP-14 | 配置验证 | REQ-7 |
| PROP-15 | 分词器往返 | REQ-8 |
