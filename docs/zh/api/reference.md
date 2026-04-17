# API 参考

## 概述

Hetero-Paged-Infer 提供 Rust API 用于将推理引擎集成到您的应用程序中。本指南涵盖核心类型、trait 和使用模式。

## 核心类型

### InferenceEngine（推理引擎）

推理操作的主入口。

```rust
use hetero_infer::{EngineConfig, GenerationParams, InferenceEngine};

// 使用默认配置创建引擎
let config = EngineConfig::default();
let mut engine = InferenceEngine::new(config)?;

// 提交请求
let params = GenerationParams {
    max_tokens: 100,
    temperature: 1.0,
    top_p: 0.9,
};
let request_id = engine.submit_request("你好，世界！", params)?;

// 运行推理
let completed = engine.run();

// 处理结果
for result in completed {
    println!("输出: {}", result.output_text);
}
```

### EngineConfig（引擎配置）

推理引擎的配置。

```rust
pub struct EngineConfig {
    pub block_size: u32,          // 每物理块 token 数（默认: 16）
    pub max_num_blocks: u32,      // 物理块总数（默认: 1024）
    pub max_batch_size: u32,      // 每批次最大序列数（默认: 32）
    pub max_num_seqs: u32,        // 最大并发序列数（默认: 256）
    pub max_model_len: u32,       // 最大序列长度（默认: 2048）
    pub max_total_tokens: u32,    // 每批次最大 token 数（默认: 4096）
    pub memory_threshold: f32,    // 内存压力阈值（默认: 0.9）
}
```

**使用示例**：

```rust
// 默认配置
let config = EngineConfig::default();

// 自定义配置
let config = EngineConfig {
    block_size: 32,
    max_num_blocks: 2048,
    max_batch_size: 64,
    ..Default::default()
};

// 验证配置
config.validate()?;

// 从文件加载
let config = EngineConfig::from_file("config.json")?;
```

### GenerationParams（生成参数）

控制文本生成的参数。

```rust
pub struct GenerationParams {
    pub max_tokens: u32,      // 最大生成 token 数
    pub temperature: f32,     // 采样温度（0.0 - 2.0）
    pub top_p: f32,          // 核采样阈值（0.0 - 1.0）
}
```

**默认值**：

```rust
impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 100,
            temperature: 1.0,
            top_p: 0.9,
        }
    }
}
```

### Request（请求）

表示一个推理请求。

```rust
pub struct Request {
    pub id: u64,
    pub input_tokens: Vec<u32>,
    pub output_tokens: Vec<u32>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub state: RequestState,
    pub created_at: Instant,
}

pub enum RequestState {
    Pending,      // 等待中
    Prefill,      // 预填充阶段
    Decode,       // 解码阶段
    Completed,    // 已完成
    Failed(String), // 失败
}
```

### Sequence（序列）

带有 KV Cache 分配的活跃请求。

```rust
pub struct Sequence {
    pub seq_id: u64,
    pub request: Request,
    pub logical_blocks: Vec<LogicalBlock>,
    pub num_computed_tokens: u32,
    pub num_generated_tokens: u32,
}
```

## Trait 接口

### TokenizerTrait（分词器接口）

```rust
pub trait TokenizerTrait: Send + Sync {
    /// 将文本编码为 token ID
    fn encode(&self, text: &str) -> Vec<u32>;
    
    /// 将 token ID 解码为文本
    fn decode(&self, tokens: &[u32]) -> String;
    
    /// 词表大小
    fn vocab_size(&self) -> u32;
    
    /// 特殊 token ID
    fn bos_token_id(&self) -> u32;
    fn eos_token_id(&self) -> u32;
    fn pad_token_id(&self) -> u32;
}
```

**SimpleTokenizer**（内置实现）：

```rust
use hetero_infer::SimpleTokenizer;

let tokenizer = SimpleTokenizer::new();
let tokens = tokenizer.encode("你好");
let text = tokenizer.decode(&tokens);
```

### SchedulerTrait（调度器接口）

```rust
pub trait SchedulerTrait: Send + Sync {
    /// 添加新请求到待处理队列
    fn add_request(&mut self, request: Request) -> Result<u64, SchedulerError>;
    
    /// 调度下一批执行
    fn schedule(&mut self) -> SchedulerOutput;
    
    /// GPU 执行后更新序列状态
    fn update_sequences(&mut self, outputs: &ExecutionOutput);
    
    /// 获取已完成的请求
    fn get_completed(&mut self) -> Vec<Request>;
    
    /// 检查是否有待处理的工作
    fn has_pending_work(&self) -> bool;
}
```

### KVCacheManagerTrait（KV Cache 管理器接口）

```rust
pub trait KVCacheManagerTrait: Send + Sync {
    /// 为新序列分配块
    fn allocate_sequence(&mut self, seq_id: u64, num_tokens: u32) -> Result<(), MemoryError>;
    
    /// 为增长的序列分配额外块
    fn allocate_block(&mut self, seq_id: u64) -> Result<PhysicalBlockRef, MemoryError>;
    
    /// 释放序列的所有块
    fn free_sequence(&mut self, seq_id: u64);
    
    /// 获取 GPU 执行用的块表
    fn get_block_table(&self, seq_id: u64) -> Option<Vec<u32>>;
    
    /// 获取内存统计
    fn get_memory_stats(&self) -> MemoryStats;
    
    /// 检查是否能分配 n 个块
    fn can_allocate(&self, num_blocks: u32) -> bool;
}

pub struct MemoryStats {
    pub total_blocks: u32,      // 总块数
    pub used_blocks: u32,       // 已用块数
    pub free_blocks: u32,       // 空闲块数
    pub num_sequences: u32,     // 序列数
}
```

### GPUExecutorTrait（GPU 执行器接口）

```rust
pub trait GPUExecutorTrait: Send + Sync {
    /// 执行一批序列
    fn execute(&mut self, batch: &ExecutionBatch) -> ExecutionOutput;
    
    /// 捕获 Decode 阶段的 CUDA Graph
    fn capture_decode_graph(&mut self, batch_size: u32);
    
    /// 使用捕获的 graph 执行
    fn execute_graph(&mut self, batch: &ExecutionBatch) -> ExecutionOutput;
}

pub struct ExecutionBatch {
    pub input_tokens: Vec<u32>,       // 所有序列的 token（扁平化）
    pub positions: Vec<u32>,          // 每个 token 的位置
    pub seq_lens: Vec<u32>,           // 各序列长度
    pub block_tables: Vec<Vec<u32>>,  // 分页式注意力块表
    pub is_prefill: Vec<bool>,        // Prefill/Decode 标志
}

pub struct ExecutionOutput {
    pub next_tokens: Vec<u32>,        // 各序列的下一个 token
    pub logits: Option<Vec<f32>>,     // Logits（可选）
}
```

## 错误处理

API 使用结构化错误类型：

```rust
use hetero_infer::EngineError;

match result {
    Ok(output) => println!("成功: {}", output),
    Err(EngineError::Config(e)) => eprintln!("配置错误: {}", e),
    Err(EngineError::Memory(e)) => eprintln!("内存错误: {}", e),
    Err(EngineError::Validation(e)) => eprintln!("验证错误: {}", e),
    Err(EngineError::Execution(e)) => eprintln!("执行错误: {}", e),
    Err(EngineError::Scheduler(e)) => eprintln!("调度器错误: {}", e),
}
```

### 错误类型

| 错误 | 说明 | 典型原因 |
|------|------|----------|
| `ConfigError` | 配置无效 | 负值、block_size 为零 |
| `MemoryError` | 内存分配失败 | 块不足、GPU OOM |
| `ValidationError` | 请求参数无效 | temperature、top_p 无效 |
| `ExecutionError` | GPU 执行失败 | CUDA 错误、超时 |
| `SchedulerError` | 调度失败 | 无效状态转换 |

## 使用示例

### 基本推理

```rust
use hetero_infer::{EngineConfig, GenerationParams, InferenceEngine};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 创建引擎
    let config = EngineConfig::default();
    let mut engine = InferenceEngine::new(config)?;
    
    // 提交请求
    let params = GenerationParams {
        max_tokens: 50,
        temperature: 0.8,
        top_p: 0.95,
    };
    let request_id = engine.submit_request("你好，世界！", params)?;
    println!("请求 {} 已提交", request_id);
    
    // 运行推理
    let completed = engine.run();
    
    // 获取结果
    for result in completed {
        println!("请求 {} 完成:", result.request_id);
        println!("  输出: {}", result.output_text);
        println!("  生成 token 数: {}", result.generated_tokens);
    }
    
    Ok(())
}
```

### 逐步执行

更好地控制推理循环：

```rust
// 提交请求
let id1 = engine.submit_request("第一个请求", params.clone())?;
let id2 = engine.submit_request("第二个请求", params)?;

// 逐步执行
while engine.has_pending_work() {
    let completed = engine.step();
    
    for result in &completed {
        println!("请求 {} 完成", result.request_id);
    }
}
```

### 自定义分词器

```rust
use hetero_infer::TokenizerTrait;

struct MyTokenizer {
    vocab: HashMap<String, u32>,
}

impl TokenizerTrait for MyTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        // 自定义编码逻辑
        vec![]
    }
    
    fn decode(&self, tokens: &[u32]) -> String {
        // 自定义解码逻辑
        String::new()
    }
    
    fn vocab_size(&self) -> u32 {
        self.vocab.len() as u32
    }
    
    fn bos_token_id(&self) -> u32 { 0 }
    fn eos_token_id(&self) -> u32 { 1 }
    fn pad_token_id(&self) -> u32 { 2 }
}
```

### 内存监控

```rust
// 获取内存统计
let stats = engine.get_memory_stats();
println!("内存使用: {}/{} 块 ({}%)", 
    stats.used_blocks, 
    stats.total_blocks,
    (stats.used_blocks as f32 / stats.total_blocks as f32) * 100.0
);
```

## 类型导出

`lib.rs` 中的主要导出：

```rust
pub use crate::config::EngineConfig;
pub use crate::engine::{InferenceEngine, EngineMetrics, CompletedRequest};
pub use crate::error::EngineError;
pub use crate::types::{Request, Sequence, GenerationParams, RequestState};
pub use crate::kv_cache::{KVCacheManager, KVCacheManagerTrait, MemoryStats};
pub use crate::scheduler::{Scheduler, SchedulerTrait, SchedulerOutput};
pub use crate::tokenizer::{SimpleTokenizer, TokenizerTrait};
pub use crate::gpu_executor::{GPUExecutorTrait, ExecutionBatch, ExecutionOutput};
```

---

*配置详情见 [CONFIGURATION.md](./CONFIGURATION.md)。*
