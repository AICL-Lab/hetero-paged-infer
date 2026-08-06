//! GPU 执行器 - 推理计算
//!
//! 提供 GPU 执行的抽象接口，支持：
//! - Paged Attention 块表间接访问
//! - 批次内可变序列长度
//!
//! # 当前实现
//!
//! 默认实现为 **Mock 执行器**，生成确定性占位 token，尚未接入真实模型计算。
//! 真实 CUDA kernel 为后续工作，届时再引入 build 脚本与 FFI 桥接。
//!
//! # 接口
//!
//! ```text
//! trait GPUExecutorTrait: Send {
//!     fn execute(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput, EngineError>;
//! }
//! ```

use crate::config::EngineConfig;
use crate::cpu_executor::CpuReferenceExecutor;
use crate::error::EngineError;
use crate::types::{ExecutionBatch, ExecutionOutput, TokenId};

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

/// GPU Executor trait defining the interface
pub trait GPUExecutorTrait: Send {
    /// Execute a batch of sequences
    fn execute(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput, EngineError>;
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

        /// **Feature: heterogeneous-inference-system, Property 11: Variable Sequence Length Handling**
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
