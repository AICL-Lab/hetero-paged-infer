# 架构概览

## 设计理念

Hetero-Paged-Infer 实现了一种**异构计算架构**，将控制流（CPU）与计算密集型操作（GPU）分离。

### 核心原则

1. **CPU 协调** — 调度、内存管理、批处理准备
2. **GPU 计算** — Attention 内核、矩阵运算、Token 生成
3. **内存效率** — PagedAttention 消除内存浪费
4. **吞吐优化** — 连续批处理最大化 GPU 利用率

## 高层架构

```mermaid
flowchart TB
    subgraph Client["客户端层"]
        Req[HTTP/gRPC 请求]
    end

    subgraph Engine["推理引擎"]
        API[API 处理]
        ORCH[编排器]
    end

    subgraph CPU["CPU 控制面"]
        T[Tokenizer]
        S[调度器]
        KVM[KV Cache 管理器]
        BB[批处理构建器]
    end

    subgraph GPU["GPU 计算面"]
        GE[GPU 执行器]
        KC[(KV Cache 内存)]
    end

    Req --> API
    API --> ORCH
    ORCH --> T
    ORCH --> S
    ORCH --> KVM
    T --> BB
    S --> BB
    KVM --> BB
    BB --> GE
    GE <--> KC
    KVM -.-> KC
```

## 组件详解

### 1. 推理引擎

协调所有组件的主编排器：

```rust
pub struct InferenceEngine {
    config: EngineConfig,
    tokenizer: Box<dyn TokenizerTrait>,
    scheduler: Box<dyn SchedulerTrait>,
    kv_cache_manager: Box<dyn KVCacheManagerTrait>,
    gpu_executor: Box<dyn GPUExecutorTrait>,
}
```

**职责：**
- 请求生命周期管理
- 逐步执行循环
- 错误恢复策略
- 指标数据收集

### 2. 调度器

实现带有 Decode 优先的**连续批处理**：

```mermaid
stateDiagram-v2
    [*] --> Pending: 提交
    Pending --> Prefill: 调度
    Prefill --> Decode: Token 就绪
    Decode --> Decode: 生成下一个
    Decode --> Completed: EOS/达到最大 Token
    Prefill --> Failed: 错误
    Decode --> Failed: 错误
    Completed --> [*]: 返回结果
    Failed --> [*]: 错误响应
```

**调度算法：**

```
1. 收集 Decode 请求（最高优先级）
2. 用 Prefill 请求填充剩余批处理槽位
3. 遵守内存和大小约束
4. 更新请求状态
```

### 3. KV Cache 管理器

实现 **PagedAttention** 内存管理：

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU 内存池                                │
├─────────────────────────────────────────────────────────────┤
│ Block 0 │ Block 1 │ Block 2 │ ... │ Block N                  │
│ [K,V]   │ [K,V]   │ [K,V]   │     │ [K,V]                    │
└─────────────────────────────────────────────────────────────┘
      ↑
页表映射：
  Sequence 0: [Block 3] → [Block 7] → [Block 12]
  Sequence 1: [Block 1] → [Block 5] → [Block 9]
```

### 4. GPU 执行器

抽象 GPU 计算：

```rust
pub trait GPUExecutorTrait {
    fn execute(&mut self, batch: &ExecutionBatch)
        -> ExecutionOutput;
    fn capture_decode_graph(&mut self, batch_size: u32);
    fn execute_graph(&mut self, batch: &ExecutionBatch)
        -> ExecutionOutput;
}
```

## 数据流

### 请求处理流水线

```mermaid
sequenceDiagram
    participant C as Client
    participant E as Engine
    participant T as Tokenizer
    participant S as Scheduler
    participant KVM as KV Cache
    participant GPU as GPU

    C->>E: Submit Request
    E->>T: Encode Text
    T-->>E: Token IDs
    E->>S: Add Request
    S->>S: Queue Request

    loop Schedule Loop
        S->>S: Build Batch
        S->>KVM: Allocate Blocks
        KVM-->>S: Block Tables
        S->>GPU: Execute Batch
        GPU-->>S: Next Tokens
        S->>S: Update States
    end

    S->>T: Decode Tokens
    T-->>S: Text Output
    S-->>E: Completed
    E-->>C: Response
```

## 内存模型

### 块结构

```rust
pub struct PhysicalBlock {
    block_id: u32,
    refcount: u32,
    data: *mut c_void,  // GPU memory pointer
}

pub struct LogicalBlock {
    logical_idx: u32,
    physical: Option<PhysicalBlockRef>,
}
```

### 内存布局

```
Token Positions:
┌─────────────────────────────────────────────────────┐
│ Block 0 │ Block 1 │ Block 2 │ Block 3 │ Block 4     │
│ 0-15    │ 16-31   │ 32-47   │ 48-63   │ 64-79       │
└─────────────────────────────────────────────────────┘

Attention Mask (Causal):
┌───┬───┬───┬───┬───┐
│ 1 │ 0 │ 0 │ 0 │ 0 │  Position 0
├───┼───┼───┼───┼───┤
│ 1 │ 1 │ 0 │ 0 │ 0 │  Position 1
├───┼───┼───┼───┼───┤
│ 1 │ 1 │ 1 │ 0 │ 0 │  Position 2
├───┼───┼───┼───┼───┤
│ 1 │ 1 │ 1 │ 1 │ 0 │  Position 3
├───┼───┼───┼───┼───┤
│ 1 │ 1 │ 1 │ 1 │ 1 │  Position 4
└───┴───┴───┴───┴───┘
```

## 性能特征

### 吞吐 vs 延迟

```mermaid
xychart-beta
    title "吞吐量 vs 批处理大小"
    x-axis [1, 8, 16, 32, 64, 128]
    y-axis "吞吐量 (tokens/s)" 0 --> 10000
    bar [500, 2000, 4000, 7000, 9500, 9800]
    line [800, 1500, 2500, 4000, 6000, 7000]
```

### 内存效率

| 方法 | 内部浪费 | 外部碎片 | 总计 |
|------|---------|---------|------|
| 静态 | 45% | 10% | 55% |
| 动态 | 20% | 8% | 28% |
| **Paged** | **<5%** | **<2%** | **<7%** |

## 可扩展性

### 水平扩展

```mermaid
flowchart LR
    subgraph LB[负载均衡器]
        nginx[Nginx/Envoy]
    end

    subgraph Workers[推理工作节点]
        W1[Worker 1]
        W2[Worker 2]
        W3[Worker 3]
        WN[Worker N]
    end

    Client --> LB
    LB --> W1
    LB --> W2
    LB --> W3
    LB --> WN
```

### 垂直扩展

- 更多 GPU 内存 → 更多并发序列
- 更多 CPU 核心 → 更快的批处理准备
- 更大的批处理大小 → 更好的 GPU 利用率

## 安全考量

1. **资源隔离** — 每个请求的内存限制
2. **输入验证** — Token 数量限制
3. **超时处理** — 防止请求挂起
4. **错误边界** — 隔离失败的请求

---

下一步：[组件详情](components.md)
