//! 批次执行流水线
//!
//! 将调度器输出转换为 GPU 执行批次，执行计算并处理重试。
//! 隐藏 `ExecutionBatch` 的构造细节，为推理引擎提供高层次的执行接口。

use crate::config::EngineConfig;
use crate::error::{EngineError, ExecutionError};
use crate::gpu_executor::GPUExecutorTrait;
use crate::types::{ExecutionBatch, ExecutionOutput, SchedulerOutput};

/// 批次执行流水线
///
/// 封装从调度器输出到 GPU 执行输出的完整流程，包括：
/// - 构建 `ExecutionBatch`
/// - 调用 GPU 执行器
/// - 超时错误重试
///
/// 这是引擎与 GPU 执行之间的深层模块，隐藏了批次数据构造和重试策略。
pub struct BatchExecutionPipeline {
    /// GPU 执行器
    gpu_executor: Box<dyn GPUExecutorTrait>,
    /// 最大重试次数
    max_retry_attempts: u32,
}

impl std::fmt::Debug for BatchExecutionPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchExecutionPipeline")
            .field("max_retry_attempts", &self.max_retry_attempts)
            .finish_non_exhaustive()
    }
}

impl BatchExecutionPipeline {
    /// 创建新的批次执行流水线
    pub fn new(gpu_executor: Box<dyn GPUExecutorTrait>, config: &EngineConfig) -> Self {
        Self {
            gpu_executor,
            max_retry_attempts: config.max_retry_attempts,
        }
    }

    /// 执行调度器输出对应的批次
    ///
    /// 内部完成 `ExecutionBatch` 构建、GPU 执行、超时重试。
    ///
    /// # 返回
    ///
    /// - `Ok(ExecutionOutput)` — 执行成功
    /// - `Err(EngineError::Execution)` — 执行失败且重试耗尽
    pub fn execute(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> Result<ExecutionOutput, EngineError> {
        if scheduler_output.is_empty() {
            return Ok(ExecutionOutput::default());
        }

        let execution_batch = build_execution_batch(scheduler_output);

        let mut retries = 0;
        loop {
            match self.gpu_executor.execute(&execution_batch) {
                Ok(output) => return Ok(output),
                Err(ExecutionError::GpuTimeout) if retries < self.max_retry_attempts => {
                    retries += 1;
                    log::warn!(
                        "重试批次执行 (尝试 {}/{}): GPU 超时",
                        retries,
                        self.max_retry_attempts
                    );
                }
                Err(other_error) => return Err(EngineError::Execution(other_error)),
            }
        }
    }
}

/// 从调度器输出构建执行批次
///
/// 将 `SchedulerOutput` 中的 prefill 和 decode 序列转换为 `ExecutionBatch`，
/// 包含 GPU kernel 所需的扁平化 token、位置、块表等数据。
pub fn build_execution_batch(scheduler_output: &SchedulerOutput) -> ExecutionBatch {
    let mut batch = ExecutionBatch::default();

    // Process prefill sequences
    for seq in &scheduler_output.prefill_sequences {
        let seq_id = seq.seq_id;
        let input_tokens = &seq.request.input_tokens;
        let context_len = seq.context_len();

        // Add tokens
        batch.input_tokens.extend(input_tokens.iter().copied());

        // Add positions (0 to len-1 for prefill)
        for i in 0..input_tokens.len() {
            batch.positions.push(i as u32);
        }

        batch.seq_lens.push(input_tokens.len() as u32);
        batch.is_prefill.push(true);
        batch.seq_ids.push(seq_id);
        batch.context_lens.push(context_len);

        batch.block_tables.push(seq.get_block_table());
    }

    // Process decode sequences
    for seq in &scheduler_output.decode_sequences {
        let seq_id = seq.seq_id;
        let context_len = seq.context_len();
        let Some(input_token) = seq.decode_input_token() else {
            continue;
        };
        let Some(position) = seq.decode_position() else {
            continue;
        };

        // For decode, we process the last token already present in the context.
        batch.input_tokens.push(input_token);

        // Position points to that last context token.
        batch.positions.push(position);

        batch.seq_lens.push(1);
        batch.is_prefill.push(false);
        batch.seq_ids.push(seq_id);
        batch.context_lens.push(context_len);

        batch.block_tables.push(seq.get_block_table());
    }

    batch
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Request, RequestState, Sequence};
    use std::sync::Arc;

    #[test]
    fn test_decode_batch_uses_last_generated_token_and_position() {
        let mut request = Request::new(
            1,
            vec![10, 11, 12],
            crate::types::GenerationParams::default(),
        );
        request.output_tokens = vec![20, 21];
        request.state = RequestState::Decode;

        let sequence = Sequence {
            seq_id: 7,
            request,
            logical_blocks: Vec::new(),
            num_computed_tokens: 3,
            num_generated_tokens: 2,
        };

        let scheduler_output = SchedulerOutput {
            prefill_sequences: Vec::new(),
            decode_sequences: vec![Arc::new(sequence)],
            total_tokens: 1,
        };

        let batch = build_execution_batch(&scheduler_output);

        assert_eq!(batch.input_tokens, vec![21]);
        assert_eq!(batch.positions, vec![4]);
        assert_eq!(batch.context_lens, vec![5]);
        assert_eq!(batch.seq_ids, vec![7]);
    }

    #[test]
    fn test_decode_batch_falls_back_to_last_prompt_token() {
        let mut request = Request::new(
            1,
            vec![30, 31, 32],
            crate::types::GenerationParams::default(),
        );
        request.state = RequestState::Decode;

        let sequence = Sequence {
            seq_id: 9,
            request,
            logical_blocks: Vec::new(),
            num_computed_tokens: 3,
            num_generated_tokens: 0,
        };

        let scheduler_output = SchedulerOutput {
            prefill_sequences: Vec::new(),
            decode_sequences: vec![Arc::new(sequence)],
            total_tokens: 1,
        };

        let batch = build_execution_batch(&scheduler_output);

        assert_eq!(batch.input_tokens, vec![32]);
        assert_eq!(batch.positions, vec![2]);
        assert_eq!(batch.context_lens, vec![3]);
        assert_eq!(batch.seq_ids, vec![9]);
    }
}
