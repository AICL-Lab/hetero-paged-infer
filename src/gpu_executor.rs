//! 执行后端抽象（EngineBackend 契约）
//!
//! [`GPUExecutorTrait`] 是引擎与"计算后端"之间的唯一契约：引擎每步调度出
//! 一个 [`ExecutionBatch`]，交给后端执行，后端返回每序列的下一 token 及
//! 可选的 logprob 信息。调度器、分页 KV 内存管理与服务控制面都在引擎侧，
//! 后端只负责"对给定 batch 做一次前向 + 采样"。
//!
//! # 契约语义（实现方必须满足）
//!
//! - **每步一个 batch**：`execute` 收到的是本步可并行计算的序列集合，
//!   prefill 与 decode 可混合（`is_prefill` 逐序列标注）。prefill 序列的
//!   `input_tokens` 是其完整 prompt，decode 序列只有 1 个新 token。
//! - **KV 由引擎侧分页管理**：`block_tables` 给出每序列的物理块表，
//!   后端据此间接访问 KV。连续 KV 后端可忽略块表（自行管理），但必须
//!   在文档中声明该边界。
//! - **输出**：`next_tokens` 与 `seq_ids` 逐序列对齐；`logprobs` 为每序列
//!   该步 token 的 top-k 信息（请求未启用或后端不支持时为 `None`）。
//!   引擎会校验输出契约，不满足的后端会被视为坏后端并快速失败：
//!   - `next_tokens.len() == seq_ids.len() == batch.seq_ids.len()`；
//!   - `seq_ids` 无重复，且与 `batch.seq_ids` 的集合完全一致（顺序可以不同）；
//!   - `logprobs` 为空或长度等于 `seq_ids.len()`。
//! - **能力声明**：[`capabilities`](GPUExecutorTrait::capabilities) 告知引擎
//!   后端实现了哪些采样语义，引擎在准入阶段据此拒绝不支持的参数。
//!
//! # 内置后端
//!
//! - [`CpuReferenceExecutor`]：
//!   默认后端，CPU 上执行真实前向（随机权重小模型），greedy 采样，提供 logprobs。
//! - [`MockGPUExecutor`]：测试用确定性占位后端。
//! - tiny-llm（C++/CUDA）对接：见 [`crate::tiny_llm_ffi`]（FFI 骨架）。
//!
//! # 接口
//!
//! ```text
//! trait GPUExecutorTrait: Send {
//!     fn execute(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput, EngineError>;
//!     fn capabilities(&self) -> ExecutorCapabilities;
//! }
//! ```

use crate::config::EngineConfig;
use crate::cpu_executor::CpuReferenceExecutor;
use crate::error::EngineError;
use crate::types::{ExecutionBatch, ExecutionOutput, SeqId, TokenId};

fn validate_execution_batch(
    config: &EngineConfig,
    batch: &ExecutionBatch,
) -> Result<(), EngineError> {
    if batch.is_empty() {
        return Ok(());
    }

    if batch.num_sequences() > config.max_batch_size as usize {
        return Err(EngineError::KernelLaunchFailed(format!(
            "Batch size {} exceeds max {}",
            batch.num_sequences(),
            config.max_batch_size
        )));
    }

    if batch.total_tokens() > config.max_total_tokens as usize {
        return Err(EngineError::KernelLaunchFailed(format!(
            "Total tokens {} exceeds max {}",
            batch.total_tokens(),
            config.max_total_tokens
        )));
    }

    Ok(())
}

/// 执行后端能力声明
///
/// 引擎在 submit 阶段据此校验生成参数，让不支持的采样语义尽早失败，
/// 而不是在执行时被后端静默忽略；同时声明 GpuTimeout 后重放是否安全。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutorCapabilities {
    /// 是否真正实现非贪心采样（temperature / top_p 生效）。
    /// 为 `false` 时仅接受 [`GenerationParams::is_greedy`] 的参数组合。
    ///
    /// [`GenerationParams::is_greedy`]: crate::types::GenerationParams::is_greedy
    pub sampling: bool,
    /// 后端执行是否幂等：`GpuTimeout` 后重放同一 batch 是否安全。
    ///
    /// 真实 CUDA kernel 超时时 KV 可能已部分写入，重放不安全，必须为 `false`；
    /// 只有能保证重放幂等的后端（如确定性 CPU/Mock）才可设为 `true`。
    pub retry_safe: bool,
}

impl ExecutorCapabilities {
    /// 仅支持 greedy 解码且 **不保证** 超时重放安全（内置后端的保守默认）。
    pub const GREEDY_ONLY: Self = Self {
        sampling: false,
        retry_safe: false,
    };
}

/// 执行后端契约（EngineBackend）
///
/// 引擎与计算后端之间的唯一接口。语义见模块文档；实现方需保证
/// `execute` 对同构输入是确定性的（便于差分测试与回归）。
pub trait GPUExecutorTrait: Send {
    /// 对给定 batch 执行一次前向 + 采样，返回每序列下一 token 与 logprob。
    ///
    /// 契约（引擎会在 [`crate::execution_pipeline::BatchExecutionPipeline::execute`]
    /// 中校验，不满足视为坏后端）：
    /// - `next_tokens` 与 `seq_ids` 逐序列对齐，长度等于 `batch.seq_ids.len()`；
    /// - `seq_ids` 无重复，且与 `batch.seq_ids` 的集合完全一致（顺序可以不同）；
    /// - `logprobs` 与 `seq_ids` 对齐，每项为该步 token 的 top-k 信息
    ///   （请求未启用或后端不支持时为 `None`）；为空或长度等于 `seq_ids.len()`；
    /// - 不能对 batch 内序列的 KV 状态做跨步假设（引擎负责持久化）。
    fn execute(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput, EngineError>;

    /// 引擎在序列到达终态（完成/失败/取消）并释放逻辑 KV 块后调用。
    ///
    /// 后端应在此时释放该序列占用的物理 KV 资源。默认空实现适用于
    /// 物理块按索引复用、写入时覆盖的后端（CPU/Mock）；持有真实物理 KV
    /// 槽位的后端（如 tiny-llm 连续 KV）必须实现，否则槽位会耗尽。
    fn sequences_finished(&mut self, _seq_ids: &[SeqId]) {}

    /// 声明后端能力；默认仅支持 greedy 解码。
    fn capabilities(&self) -> ExecutorCapabilities {
        ExecutorCapabilities::GREEDY_ONLY
    }
}

pub fn create_default_gpu_executor(
    config: EngineConfig,
    vocab_size: u32,
) -> Result<Box<dyn GPUExecutorTrait>, EngineError> {
    Ok(Box::new(CpuReferenceExecutor::new(config, vocab_size)))
}

/// Mock GPU Executor for testing without actual GPU
///
/// This executor simulates GPU execution by generating deterministic tokens.
/// Replace with real CUDA implementation for production use.
#[derive(Debug)]
pub struct MockGPUExecutor {
    config: EngineConfig,
    /// Vocabulary size for token generation
    vocab_size: u32,
    /// Counter for deterministic token generation in tests
    token_counter: u32,
}

impl MockGPUExecutor {
    pub fn new(config: EngineConfig, vocab_size: u32) -> Self {
        Self {
            config,
            vocab_size,
            token_counter: 100,
        }
    }

    /// Generate next token (mock implementation)
    fn generate_token(&mut self) -> TokenId {
        let token = self.token_counter % self.vocab_size;
        self.token_counter = self.token_counter.wrapping_add(1);
        token
    }
}

impl GPUExecutorTrait for MockGPUExecutor {
    fn execute(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput, EngineError> {
        validate_execution_batch(&self.config, batch)?;

        if batch.is_empty() {
            return Ok(ExecutionOutput::default());
        }

        // 空词表无法生成 token，不检查会导致下方 `generate_token` 取模时除零 panic。
        if self.vocab_size == 0 {
            return Err(EngineError::BackendError(
                "GPU executor received an empty vocabulary".to_string(),
            ));
        }

        // Generate one token per sequence
        let mut next_tokens = Vec::with_capacity(batch.num_sequences());
        for _ in &batch.seq_ids {
            next_tokens.push(self.generate_token());
        }

        Ok(ExecutionOutput {
            next_tokens,
            seq_ids: batch.seq_ids.clone(),
            logprobs: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::create_test_config;

    #[test]
    fn test_mock_executor_rejects_empty_vocabulary() {
        let config = create_test_config();
        let mut executor = MockGPUExecutor::new(config, 0);

        let batch = ExecutionBatch {
            input_tokens: vec![1],
            positions: vec![0],
            seq_lens: vec![1],
            block_tables: vec![vec![0]],
            is_prefill: vec![true],
            seq_ids: vec![1],
            context_lens: vec![1],
        };

        // vocab_size == 0 必须返回错误而非除零 panic
        let result = executor.execute(&batch);
        assert!(matches!(result, Err(EngineError::BackendError(_))));
    }

    #[test]
    fn test_mock_executor_execute_empty() {
        let config = create_test_config();
        let mut executor = MockGPUExecutor::new(config, 32000);

        let batch = ExecutionBatch::default();
        let result = executor.execute(&batch);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.next_tokens.is_empty());
    }

    #[test]
    fn test_mock_executor_execute_batch() {
        let config = create_test_config();
        let mut executor = MockGPUExecutor::new(config, 32000);

        let batch = ExecutionBatch {
            input_tokens: vec![1, 2, 3, 4, 5],
            positions: vec![0, 1, 2, 3, 4],
            seq_lens: vec![3, 2],
            block_tables: vec![vec![0, 1], vec![2]],
            is_prefill: vec![true, true],
            seq_ids: vec![1, 2],
            context_lens: vec![3, 2],
        };

        let result = executor.execute(&batch);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.next_tokens.len(), 2);
        assert_eq!(output.seq_ids.len(), 2);
        assert_eq!(output.next_tokens, vec![100, 101]);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use crate::test_utils::create_test_config_with_limits;
    use crate::types::{BlockIdx, SeqId};
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// **Feature: paged-inference-system, Property 11: Variable Sequence Length Handling**
        /// *For any* batch containing sequences of different lengths, the GPU_Executor shall
        /// produce correct attention outputs for each sequence independently.
        /// **Validates: Requirements 4.2**
        #[test]
        fn prop_variable_sequence_length_handling(
            num_sequences in 1usize..8,
            seq_lengths in prop::collection::vec(1u32..64, 1..8),
        ) {
            let max_total = seq_lengths.iter().sum::<u32>().max(100);
            let config = create_test_config_with_limits(8, max_total, 200);
            let mut executor = MockGPUExecutor::new(config, 32000);

            // Build batch with variable sequence lengths
            let mut batch = ExecutionBatch::default();
            let actual_num_seqs = num_sequences.min(seq_lengths.len());

            let mut token_offset = 0u32;
            for (i, &seq_len) in seq_lengths.iter().take(actual_num_seqs).enumerate() {
                let seq_id = (i + 1) as SeqId;

                // Add tokens for this sequence
                for j in 0..seq_len {
                    batch.input_tokens.push((token_offset + j) % 32000);
                    batch.positions.push(j);
                }
                token_offset += seq_len;

                batch.seq_lens.push(seq_len);
                batch.is_prefill.push(true);
                batch.seq_ids.push(seq_id);
                batch.context_lens.push(seq_len);
                batch.block_tables.push(vec![i as BlockIdx]);
            }

            // Execute batch
            let result = executor.execute(&batch);
            prop_assert!(result.is_ok(), "Execution should succeed");

            let output = result.unwrap();

            // Verify output has correct number of tokens (one per sequence)
            prop_assert_eq!(
                output.next_tokens.len(),
                actual_num_seqs,
                "Should produce one token per sequence"
            );

            // Verify sequence IDs match
            prop_assert_eq!(
                output.seq_ids.len(),
                actual_num_seqs,
                "Should have correct number of sequence IDs"
            );

            // Verify each sequence got a valid token
            for token in &output.next_tokens {
                prop_assert!(
                    *token < 32000,
                    "Generated token should be within vocabulary"
                );
            }
        }

        /// Property test for batch size validation
        #[test]
        fn prop_batch_size_validation(
            num_sequences in 1usize..20,
            max_batch_size in 1u32..10,
        ) {
            let config = create_test_config_with_limits(max_batch_size, 1000, 200);
            let mut executor = MockGPUExecutor::new(config, 32000);

            // Build batch
            let mut batch = ExecutionBatch::default();
            for i in 0..num_sequences {
                batch.input_tokens.push(i as TokenId);
                batch.positions.push(0);
                batch.seq_lens.push(1);
                batch.is_prefill.push(true);
                batch.seq_ids.push(i as SeqId);
                batch.context_lens.push(1);
                batch.block_tables.push(vec![i as BlockIdx]);
            }

            let result = executor.execute(&batch);

            if num_sequences <= max_batch_size as usize {
                prop_assert!(result.is_ok(), "Should succeed within batch limit");
            } else {
                prop_assert!(result.is_err(), "Should fail exceeding batch limit");
            }
        }

        /// Property test for deterministic output per sequence
        #[test]
        fn prop_output_per_sequence(
            num_sequences in 1usize..5,
        ) {
            let config = create_test_config_with_limits(8, 500, 200);
            let mut executor = MockGPUExecutor::new(config, 32000);

            // Build batch
            let mut batch = ExecutionBatch::default();
            for i in 0..num_sequences {
                batch.input_tokens.push(i as TokenId);
                batch.positions.push(0);
                batch.seq_lens.push(1);
                batch.is_prefill.push(false);
                batch.seq_ids.push((i + 1) as SeqId);
                batch.context_lens.push(10);
                batch.block_tables.push(vec![i as BlockIdx]);
            }

            let result = executor.execute(&batch);
            prop_assert!(result.is_ok());

            let output = result.unwrap();

            // Each sequence should get exactly one output token
            prop_assert_eq!(
                output.next_tokens.len(),
                num_sequences,
                "Each sequence should get one output token"
            );

            // Sequence IDs should match input
            for (i, &seq_id) in output.seq_ids.iter().enumerate() {
                prop_assert_eq!(
                    seq_id,
                    (i + 1) as SeqId,
                    "Sequence IDs should match"
                );
            }
        }
    }
}
