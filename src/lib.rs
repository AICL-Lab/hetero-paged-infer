//! # Hetero-Paged-Infer
//!
//! 异构推理系统 — 基于 `PagedAttention` 分页内存与 Continuous Batching 的推理引擎脚手架；当前计算后端为 mock。
//!
//! ## 概述
//!
//! 本库提供了一个推理引擎脚手架，实现了以下核心技术（计算后端当前为 mock）：
//!
//! - **`PagedAttention`**: 分页式 KV Cache 管理，按需分配/释放显存块
//! - **Continuous Batching**: 连续批处理调度，prefill/decode 分阶段管理
//! - **内存压力感知**: 可配置阈值，自动拒绝新请求防止 OOM
//! - **模块化设计**: 所有核心组件通过 trait 抽象，便于替换实现
//!
//! ## 架构
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │            InferenceEngine              │
//! │  ┌──────────┐  ┌──────────┐  ┌───────┐  │
//! │  │Tokenizer │  │Scheduler │  │  GPU  │  │
//! │  │          │  │          │  │Executor│ │
//! │  └──────────┘  └────┬─────┘  └───────┘  │
//! │                     │                    │
//! │              ┌──────▼──────┐            │
//! │              │ KV Cache    │            │
//! │              │ Manager     │            │
//! │              └─────────────┘            │
//! └─────────────────────────────────────────┘
//! ```
//!
//! ## 快速开始
//!
//! ```rust
//! use hetero_infer::{EngineConfig, GenerationParams, InferenceEngine};
//!
//! // 创建引擎
//! let config = EngineConfig::default();
//! let mut engine = InferenceEngine::new(config)?;
//!
//! // 提交请求
//! let params = GenerationParams {
//!     max_tokens: 50,
//!     temperature: 1.0,
//!     top_p: 0.9,
//! };
//! let (request_id, _prompt_tokens) = engine.submit_request("你好，世界！", params)?;
//!
//! // 运行推理
//! let completed = engine.run();
//!
//! for result in completed {
//!     println!("输出: {}", result.output_text);
//! }
//! # Ok::<(), hetero_infer::EngineError>(())
//! ```
//!
//! ## 核心组件
//!
//! ### 配置
//!
//! - [`EngineConfig`] - 引擎配置参数，包括块大小、批次大小、内存阈值等
//!
//! ### 推理引擎
//!
//! - [`InferenceEngine`] - 主编排器，协调所有组件
//! - [`EngineMetrics`] - 运行时指标收集
//!
//! ### 调度器
//!
//! - [`Scheduler`] - Continuous Batching 调度器
//!
//! ### KV Cache 管理
//!
//! - [`KVCacheManager`] - `PagedAttention` KV Cache 管理器
//!
//! ### GPU 执行器
//!
//! - [`MockGPUExecutor`] - Mock GPU 执行器（测试用）
//! - CudaExecutor - nvcc 编译的 CUDA 后端桥接执行器（启用 `cuda` feature；门控类型，文档链接仅在启用该 feature 时可解析）
//! - [`GPUExecutorTrait`] - GPU 执行器 trait 接口
//! - [`build_execution_batch`] - 构建执行批次
//!
//! ### 分词器
//!
//! - [`SimpleTokenizer`] - 简单字符级分词器（测试用，`without_special_tokens()` 可精确往返）
//! - [`TokenizerTrait`] - 分词器 trait 接口
//!
//! ### 类型
//!
//! - [`Request`] - 推理请求
//! - [`Sequence`] - 活跃序列（含 KV Cache）
//! - [`GenerationParams`] - 生成参数
//! - [`CompletedRequest`] - 完成的请求
//! - [`ExecutionBatch`] - GPU 执行批次
//! - [`ExecutionOutput`] - GPU 执行输出
//!
//! ### 错误处理
//!
//! - [`EngineError`] - 顶层引擎错误
//! - [`ConfigError`] - 配置错误
//! - [`ValidationError`] - 验证错误
//! - [`MemoryError`] - 内存错误
//! - [`ExecutionError`] - 执行错误
//! - [`SchedulerError`] - 调度错误

pub mod config;
pub mod engine;
pub mod error;
pub mod execution_pipeline;
pub mod gpu_executor;
pub mod kv_cache;
pub mod scheduler;
pub mod server;
pub mod tokenizer;
pub mod types;

// 测试辅助模块：仅在单元测试（cfg(test)）或 `test-utils` feature 下编译，
// 不会进入发布构建。集成测试与 bench 经由 dev-dependencies 启用该 feature。
#[cfg(any(test, feature = "test-utils"))]
#[doc(hidden)]
pub mod test_utils;

// 选择性导出，避免命名空间污染（如 error::Result 遮蔽 std::Result）
pub use config::{EngineConfig, ServingConfig, SpecialTokenIds, TokenizerConfig, TokenizerKind};
pub use engine::{EngineMetrics, InferenceEngine, StepEvents};
pub use error::{
    ConfigError, EngineError, ExecutionError, MemoryError, SchedulerError, ValidationError,
};
pub use execution_pipeline::{build_execution_batch, BatchExecutionPipeline};
#[cfg(feature = "cuda")]
pub use gpu_executor::CudaExecutor;
pub use gpu_executor::{create_default_gpu_executor, GPUExecutorTrait, MockGPUExecutor};
pub use kv_cache::KVCacheManager;
pub use scheduler::Scheduler;
pub use server::{create_router, create_router_with_engine};
pub use tokenizer::{build_tokenizer, HuggingFaceTokenizer, SimpleTokenizer, TokenizerTrait};
pub use types::{
    BlockIdx, CompletedRequest, ExecutionBatch, ExecutionOutput, GenerationParams, MemoryStats,
    PhysicalBlockRef, Request, RequestId, RequestState, SchedulerOutput, SeqId, Sequence, TokenId,
};
