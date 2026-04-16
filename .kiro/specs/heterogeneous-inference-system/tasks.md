# 实现计划：异构推理系统

## 概述

本计划实现一个 Rust 版本的异构推理微服务，支持 CPU-GPU 协同执行。实现采用自底向上的方法：核心数据结构 → KV Cache 管理器 → 调度器 → GPU 执行器 → 推理引擎集成。

## 任务

- [x] 1. 项目设置和核心数据结构
  - [x] 1.1 初始化 Rust 项目
    - 创建 `Cargo.toml`，依赖：`thiserror`, `serde`, `clap`, `proptest`
    - 设置项目结构：`src/lib.rs`, `src/main.rs`
    - 需求：7.1, 7.3

  - [x] 1.2 实现核心类型和枚举
    - 实现 `RequestState` 枚举（Pending, Prefill, Decode, Completed, Failed）
    - 实现 `Request` 结构体，包含 id, tokens, parameters, state
    - 实现 `GenerationParams` 结构体，包含 max_tokens, temperature, top_p
    - 实现 `Sequence` 结构体，包含 seq_id, request, logical_blocks
    - 需求：1.1, 1.2

  - [x] 1.3 实现配置类型
    - 实现 `EngineConfig` 结构体，包含 block_size, max_num_blocks 等
    - 实现配置验证逻辑
    - 实现从文件和命令行加载配置
    - 需求：7.1, 7.2

  - [x] 1.4 编写配置验证属性测试
    - 属性 14：配置验证
    - 验证：需求 7.2

  - [x] 1.5 编写参数验证属性测试
    - 属性 2：参数验证正确性
    - 验证：需求 1.3

- [x] 2. 里程碑 - 核心类型完成 ✓

- [x] 3. KV Cache 管理器实现
  - [x] 3.1 实现 PhysicalBlock 和内存池
    - 实现 `PhysicalBlock` 结构体表示 GPU 内存区域
    - 实现 `BlockPool` 空闲列表管理
    - 实现块分配和释放
    - 需求：2.1, 2.5

  - [x] 3.2 实现 LogicalBlock 和页表
    - 实现 `LogicalBlock` 结构体，包含物理块映射
    - 实现 `PageTable` 逻辑到物理的映射
    - 实现 O(1) 查找
    - 需求：2.2, 2.7

  - [x] 3.3 实现 KVCacheManager trait 和结构体
    - 实现 `allocate_sequence()` 分配新序列
    - 实现 `allocate_block()` 序列增长
    - 实现 `free_sequence()` 清理
    - 实现 `get_block_table()` GPU 执行
    - 实现 `get_memory_stats()` 监控
    - 需求：2.2, 2.3, 2.4, 2.5, 2.6

  - [x] 3.4-3.7 属性测试
    - 属性 5：块计数不变量
    - 属性 3：序列启动时的块分配
    - 属性 4：增长时的块分配
    - 属性 12：内存统计不变量

- [x] 4. 里程碑 - KV Cache 管理器完成 ✓

- [x] 5. 调度器实现
  - [x] 5.1 实现请求队列
  - [x] 5.2 实现调度逻辑
  - [x] 5.3 实现状态转换
  - [x] 5.4 实现内存压力处理
  - [x] 5.5-5.11 属性测试

- [x] 6. 里程碑 - 调度器完成 ✓

- [x] 7. 分词器实现
  - [x] 7.1 实现 Tokenizer trait 和基本实现
  - [x] 7.2 编写往返测试（属性 15）

- [x] 8. 里程碑 - 分词器完成 ✓

- [x] 9. GPU 执行器实现
  - [x] 9.1 实现 GPU 内存管理
  - [x] 9.2 实现 ExecutionBatch 和 GPUBatchData
  - [x] 9.3 实现 paged attention kernel 接口
  - [x] 9.4 实现 GPUExecutor trait
  - [x] 9.5 属性测试（属性 11）

- [x] 10. 里程碑 - GPU 执行器完成 ✓

- [x] 11. 推理引擎集成
  - [x] 11.1 实现 InferenceEngine 结构体
  - [x] 11.2 实现主推理循环
  - [x] 11.3 实现错误处理和恢复
  - [x] 11.4 实现监控和指标

- [x] 12. 里程碑 - 集成完成 ✓

- [x] 13. 集成测试
  - [x] 13.1 端到端请求流测试
  - [x] 13.2 连续批处理测试
  - [x] 13.3 内存压力测试

- [x] 14. 最终里程碑 - 所有测试通过 ✓

## 当前状态

**所有任务已完成！**

### 测试覆盖

- 78 个单元测试
- 15 个属性测试
- 13 个集成测试

### 已实现功能

- ✅ PagedAttention KV Cache 管理
- ✅ Continuous Batching 调度器
- ✅ 内存压力感知
- ✅ 模块化 trait 抽象
- ✅ 完整的错误处理
- ✅ Mock GPU 执行器（用于测试）

### 未实现功能

- ❌ 真实 CUDA kernel
- ❌ 真实 pinned memory
- ❌ Copy-on-write KV 共享
- ❌ 异步 CPU/GPU overlap

## 备注

- GPU kernel 实现需要 CUDA toolkit
- 当前使用 MockGPUExecutor 测试 CPU 组件
- 所有测试均通过
