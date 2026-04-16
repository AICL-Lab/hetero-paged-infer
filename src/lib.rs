//! # Hetero-Paged-Infer
//!
//! 异构推理系统 — 基于 PagedAttention 和 Continuous Batching 的 CPU-GPU 协同推理引擎。
//!
//! ## 概述
//!
//! 本库提供了一个高性能推理引擎，实现了以下核心技术：
//!
//! - **PagedAttention**: 分页式 KV Cache 管理，按需分配/释放显存块
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
//! let request_id = engine.submit_request("你好，世界！", params)?;
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
//! - [`RecoveryAction`] - 错误恢复策略
//!
//! ### 调度器
//!
//! - [`Scheduler`] - Continuous Batching 调度器
//! - [`SchedulerTrait`] - 调度器 trait 接口
//!
//! ### KV Cache 管理
//!
//! - [`KVCacheManager`] - PagedAttention KV Cache 管理器
//! - [`KVCacheManagerTrait`] - KV Cache 管理器 trait 接口
//!
//! ### GPU 执行器
//!
//! - [`MockGPUExecutor`] - Mock GPU 执行器（测试用）
//! - [`GPUExecutorTrait`] - GPU 执行器 trait 接口
//! - [`build_execution_batch`] - 构建执行批次
//!
//! ### 分词器
//!
//! - [`SimpleTokenizer`] - 简单字符级分词器（测试用）
//! - [`RoundTripTokenizer`] - 精确往返分词器
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
pub mod gpu_executor;
pub mod kv_cache;
pub mod scheduler;
pub mod tokenizer;
pub mod types;

#[cfg(test)]
pub mod test_utils;

// 选择性导出，避免命名空间污染（如 error::Result 遮蔽 std::Result）
pub use config::{EngineConfig, SpecialTokenIds};
pub use engine::{EngineMetrics, InferenceEngine, RecoveryAction};
pub use error::{
    ConfigError, EngineError, ExecutionError, MemoryError, SchedulerError, ValidationError,
};
pub use gpu_executor::{build_execution_batch, GPUExecutorTrait, MockGPUExecutor};
pub use kv_cache::{KVCacheManager, KVCacheManagerTrait};
pub use scheduler::{Scheduler, SchedulerTrait};
pub use tokenizer::{RoundTripTokenizer, SimpleTokenizer, TokenizerTrait};
pub use types::{
    BlockIdx, CompletedRequest, ExecutionBatch, ExecutionOutput, GenerationParams, LogicalBlock,
    MemoryStats, PhysicalBlockRef, Request, RequestId, RequestState, SchedulerOutput, SeqId,
    Sequence, TokenId,
};
