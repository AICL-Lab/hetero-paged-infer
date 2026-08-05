//! GPU 执行器 - 推理计算
//!
//! 提供 GPU 执行的抽象接口，支持：
//! - Paged Attention 块表间接访问
//! - 批次内可变序列长度
//!
//! # 当前实现
//!
//! 默认实现为 **Mock 执行器**；启用 `cuda` feature 时，会切换到
//! 由 `nvcc` 编译的 CUDA 后端桥接执行器。两种实现目前都生成确定性
//! 占位 token，尚未接入真实模型计算。
//!
//! # 接口
//!
//! ```text
//! trait GPUExecutorTrait: Send {
//!     fn execute(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput, ExecutionError>;
//! }
//! ```

use crate::config::EngineConfig;
use crate::error::ExecutionError;
use crate::types::{ExecutionBatch, ExecutionOutput, TokenId};

#[cfg(feature = "cuda")]
mod cuda_backend {
    use crate::error::ExecutionError;
    use crate::types::{SeqId, TokenId};
    use std::ffi::CStr;
    use std::os::raw::c_char;

    unsafe extern "C" {
        fn hetero_cuda_compiled_with_nvcc() -> i32;
        fn hetero_cuda_backend_name() -> *const c_char;
        fn hetero_cuda_device_available() -> i32;
        fn hetero_cuda_generate_next_tokens(
            seq_ids: *const u64,
            // 必须与 C/C++ 侧的 `unsigned long long` 严格一致；
            // 使用 `usize` 会在 32 位目标上造成 ABI 不匹配（参数错位 → 越界写）。
            num_sequences: u64,
            seed: u32,
            vocab_size: u32,
            out_tokens: *mut u32,
            used_device: *mut i32,
        ) -> i32;
    }

    pub fn compiled_with_nvcc() -> bool {
        // SAFETY: The symbol is provided by the nvcc-compiled static library when the
        // `cuda` feature is enabled. It takes no arguments and returns a plain integer.
        unsafe { hetero_cuda_compiled_with_nvcc() != 0 }
    }

    pub fn backend_name() -> Result<String, ExecutionError> {
        // SAFETY: The backend name is a null-terminated static string from the nvcc-compiled
        // library. We validate against null before converting.
        let ptr = unsafe { hetero_cuda_backend_name() };
        if ptr.is_null() {
            return Err(ExecutionError::CudaError(
                "CUDA backend returned a null name pointer".to_string(),
            ));
        }

        // SAFETY: `ptr` was checked for null and points to a valid NUL-terminated static string.
        let name = unsafe { CStr::from_ptr(ptr) }
            .to_str()
            .map_err(|error| ExecutionError::CudaError(error.to_string()))?;
        Ok(name.to_string())
    }

    pub fn device_available() -> bool {
        // SAFETY: The symbol is provided by the nvcc-built backend and takes no arguments.
        unsafe { hetero_cuda_device_available() != 0 }
    }

    pub struct GeneratedTokens {
        pub tokens: Vec<TokenId>,
        pub used_device: bool,
    }

    pub fn generate_next_tokens(
        seq_ids: &[SeqId],
        seed: u32,
        vocab_size: u32,
    ) -> Result<GeneratedTokens, ExecutionError> {
        let mut out_tokens = vec![0; seq_ids.len()];
        let mut used_device = 0;
        // SAFETY: All pointers are derived from valid slices owned by Rust for the duration of
        // the call, lengths match the provided slice lengths, and `out_tokens` is writable.
        // `num_sequences` is passed as `u64` to match the C++ `unsigned long long` parameter
        // on every target (a `usize` would mismatch the ABI on 32-bit platforms).
        let status = unsafe {
            hetero_cuda_generate_next_tokens(
                seq_ids.as_ptr(),
                seq_ids.len() as u64,
                seed,
                vocab_size,
                out_tokens.as_mut_ptr(),
                &mut used_device,
            )
        };

        match status {
            0 => Ok(GeneratedTokens {
                tokens: out_tokens,
                used_device: used_device != 0,
            }),
            -1 => Err(ExecutionError::CudaError(
                "CUDA backend received a null output buffer".to_string(),
            )),
            -2 => Err(ExecutionError::CudaError(
                "CUDA backend received sequence metadata without sequence IDs".to_string(),
            )),
            -3 => Err(ExecutionError::CudaError(
                "CUDA backend received an empty vocabulary".to_string(),
            )),
            -10 => Err(ExecutionError::CudaError(
                "CUDA backend failed to allocate device output memory".to_string(),
            )),
            -11 => Err(ExecutionError::CudaError(
                "CUDA backend kernel launch failed".to_string(),
            )),
            -12 => Err(ExecutionError::CudaError(
                "CUDA backend kernel synchronization failed".to_string(),
            )),
            -13 => Err(ExecutionError::CudaError(
                "CUDA backend failed to copy results from device".to_string(),
            )),
            other => Err(ExecutionError::CudaError(format!(
                "CUDA backend failed with status {other}"
            ))),
        }
    }
}

fn validate_execution_batch(
    config: &EngineConfig,
    batch: &ExecutionBatch,
) -> Result<(), ExecutionError> {
    if batch.is_empty() {
        return Ok(());
    }

    if batch.num_sequences() > usize_from_u32(config.max_batch_size) {
        return Err(ExecutionError::KernelLaunchFailed(format!(
            "Batch size {} exceeds max {}",
            batch.num_sequences(),
            config.max_batch_size
        )));
    }

    if batch.total_tokens() > usize_from_u32(config.max_total_tokens) {
        return Err(ExecutionError::KernelLaunchFailed(format!(
            "Total tokens {} exceeds max {}",
            batch.total_tokens(),
            config.max_total_tokens
        )));
    }

    Ok(())
}

const fn usize_from_u32(value: u32) -> usize {
    value as usize
}

/// GPU Executor trait defining the interface
pub trait GPUExecutorTrait: Send {
    /// Execute a batch of sequences
    fn execute(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput, ExecutionError>;
}

pub fn create_default_gpu_executor(
    config: EngineConfig,
    vocab_size: u32,
) -> Result<Box<dyn GPUExecutorTrait>, ExecutionError> {
    #[cfg(feature = "cuda")]
    {
        Ok(Box::new(CudaExecutor::new(config, vocab_size)?))
    }

    #[cfg(not(feature = "cuda"))]
    {
        Ok(Box::new(MockGPUExecutor::new(config, vocab_size)))
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct CudaExecutor {
    config: EngineConfig,
    vocab_size: u32,
    token_counter: u32,
    backend_name: String,
    compiled_with_nvcc: bool,
    device_available: bool,
    last_launch_used_device: bool,
}

#[cfg(feature = "cuda")]
impl CudaExecutor {
    pub fn new(config: EngineConfig, vocab_size: u32) -> Result<Self, ExecutionError> {
        let compiled_with_nvcc = cuda_backend::compiled_with_nvcc();
        let backend_name = cuda_backend::backend_name()?;
        let device_available = cuda_backend::device_available();

        Ok(Self {
            config,
            vocab_size,
            token_counter: 100,
            backend_name,
            compiled_with_nvcc,
            device_available,
            last_launch_used_device: false,
        })
    }

    pub fn backend_name(&self) -> &str {
        &self.backend_name
    }

    pub fn compiled_with_nvcc(&self) -> bool {
        self.compiled_with_nvcc
    }

    pub fn device_available(&self) -> bool {
        self.device_available
    }

    pub fn last_launch_used_device(&self) -> bool {
        self.last_launch_used_device
    }
}

#[cfg(feature = "cuda")]
impl GPUExecutorTrait for CudaExecutor {
    fn execute(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput, ExecutionError> {
        validate_execution_batch(&self.config, batch)?;

        if batch.is_empty() {
            return Ok(ExecutionOutput::default());
        }

        let generated = cuda_backend::generate_next_tokens(
            &batch.seq_ids,
            self.token_counter,
            self.vocab_size,
        )?;
        self.last_launch_used_device = generated.used_device;
        self.token_counter = self
            .token_counter
            .wrapping_add(batch.num_sequences() as u32);

        Ok(ExecutionOutput {
            next_tokens: generated.tokens,
            logits: None,
            seq_ids: batch.seq_ids.clone(),
        })
    }
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
    fn execute(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput, ExecutionError> {
        validate_execution_batch(&self.config, batch)?;

        if batch.is_empty() {
            return Ok(ExecutionOutput::default());
        }

        // 与 CUDA 后端（C++ 侧返回 -3）保持一致：空词表无法生成 token。
        // 不检查会导致下方 `generate_token` 对 vocab_size 取模时除零 panic。
        if self.vocab_size == 0 {
            return Err(ExecutionError::CudaError(
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
            logits: None,
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
        assert!(matches!(result, Err(ExecutionError::CudaError(_))));
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

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_executor_creation() {
        let config = create_test_config();
        let executor = CudaExecutor::new(config, 32000).unwrap();

        assert!(
            executor.backend_name() == "nvcc-compiled-cuda-backend"
                || executor.backend_name() == "host-fallback-cuda-backend"
        );
        assert_eq!(
            executor.compiled_with_nvcc(),
            executor.backend_name() == "nvcc-compiled-cuda-backend"
        );
        assert!(!executor.last_launch_used_device());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_executor_execute_batch() {
        let config = create_test_config();
        let mut executor = CudaExecutor::new(config, 32000).unwrap();
        let batch = ExecutionBatch {
            input_tokens: vec![1, 2, 3, 4],
            positions: vec![0, 1, 2, 3],
            seq_lens: vec![2, 2],
            block_tables: vec![vec![0], vec![1]],
            is_prefill: vec![true, false],
            seq_ids: vec![11, 22],
            context_lens: vec![2, 2],
        };

        let output = executor.execute(&batch).unwrap();

        assert_eq!(output.seq_ids, vec![11, 22]);
        assert_eq!(output.next_tokens, vec![100, 101]);
        assert_eq!(
            executor.last_launch_used_device(),
            executor.device_available()
        );
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
