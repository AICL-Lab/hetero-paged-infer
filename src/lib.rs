//! Heterogeneous Inference System
//! 
//! A high-performance inference engine leveraging CPU-GPU co-execution
//! with PagedAttention and Continuous Batching.

pub mod types;
pub mod config;
pub mod error;
pub mod kv_cache;
pub mod scheduler;
pub mod tokenizer;
pub mod gpu_executor;
pub mod engine;

#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;

// 选择性导出，避免命名空间污染（如 error::Result 遮蔽 std::Result）
pub use config::EngineConfig;
pub use engine::{InferenceEngine, EngineMetrics, RecoveryAction};
pub use error::{ConfigError, EngineError, ExecutionError, MemoryError, SchedulerError, ValidationError};
pub use gpu_executor::{GPUExecutorTrait, MockGPUExecutor, build_execution_batch};
pub use kv_cache::{KVCacheManager, KVCacheManagerTrait};
pub use scheduler::{Scheduler, SchedulerTrait};
pub use tokenizer::{SimpleTokenizer, RoundTripTokenizer, TokenizerTrait, BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID, UNK_TOKEN_ID};
pub use types::{
    BlockIdx, CompletedRequest, ExecutionBatch, ExecutionOutput, GenerationParams,
    LogicalBlock, MemoryStats, PhysicalBlockRef, Request, RequestId, RequestState,
    SchedulerOutput, SeqId, Sequence, TokenId,
};
