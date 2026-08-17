//! 批次执行流水线
//!
//! 将调度器输出转换为 GPU 执行批次，执行计算并处理重试。
//! 隐藏 `ExecutionBatch` 的构造细节，为推理引擎提供高层次的执行接口。

use crate::config::EngineConfig;
use crate::error::EngineError;
use crate::gpu_executor::{ExecutorCapabilities, GPUExecutorTrait};
use crate::types::{ExecutionBatch, ExecutionOutput, SchedulerOutput, SeqId};
use std::collections::HashSet;

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

    /// 底层执行后端的能力声明（供引擎在 submit 阶段校验生成参数）。
    pub fn capabilities(&self) -> ExecutorCapabilities {
        self.gpu_executor.capabilities()
    }

    /// 通知后端这些序列已到达终态（完成/失败/取消），可以释放物理 KV 资源。
    pub fn sequences_finished(&mut self, seq_ids: &[SeqId]) {
        self.gpu_executor.sequences_finished(seq_ids);
    }

    /// 执行调度器输出对应的批次
    ///
    /// 内部完成 `ExecutionBatch` 构建、GPU 执行、超时重试。
    ///
    /// # 返回
    ///
    /// - `Ok(ExecutionOutput)` — 执行成功
    /// - `Err(EngineError::InvalidStateTransition(_))` — 批次构建失败（调度器状态不一致）
    /// - `Err(EngineError::BackendError(_))` / `Err(EngineError::GpuTimeout)` — 执行失败且重试耗尽
    pub fn execute(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> Result<ExecutionOutput, EngineError> {
        if scheduler_output.is_empty() {
            return Ok(ExecutionOutput::default());
        }

        let execution_batch = build_execution_batch(scheduler_output)?;
        validate_batch_shape(&execution_batch)?;

        let mut retries = 0;
        loop {
            match self.gpu_executor.execute(&execution_batch) {
                Ok(output) => {
                    validate_execution_output(&execution_batch, &output)?;
                    return Ok(output);
                }
                Err(EngineError::GpuTimeout) if retries < self.max_retry_attempts => {
                    retries += 1;
                    log::warn!(
                        "重试批次执行 (尝试 {}/{}): GPU 超时",
                        retries,
                        self.max_retry_attempts
                    );
                }
                Err(other_error) => return Err(other_error),
            }
        }
    }
}

/// 校验执行批次内部形状一致。
///
/// 后端对批次数据结构的正确性不应做任何假设；形状不一致（例如 token 数
/// 与长度字段对不上、序列数为 0、空块表）意味着调度器状态损坏或构造 bug，
/// 必须在此快速失败，而不是把损坏数据交给后端。
fn validate_batch_shape(batch: &ExecutionBatch) -> Result<(), EngineError> {
    if batch.is_empty() {
        return Ok(());
    }

    if batch.input_tokens.len() != batch.positions.len() {
        return Err(EngineError::BackendError(format!(
            "invalid execution batch: input_tokens.len()={} != positions.len()={}",
            batch.input_tokens.len(),
            batch.positions.len()
        )));
    }

    let n = batch.seq_ids.len();
    if batch.seq_lens.len() != n
        || batch.is_prefill.len() != n
        || batch.context_lens.len() != n
        || batch.block_tables.len() != n
    {
        return Err(EngineError::BackendError(format!(
            "invalid execution batch: seq_ids.len()={} but seq_lens.len()={}, \
             is_prefill.len()={}, context_lens.len()={}, block_tables.len()={}",
            n,
            batch.seq_lens.len(),
            batch.is_prefill.len(),
            batch.context_lens.len(),
            batch.block_tables.len()
        )));
    }

    let total_len: u32 = batch.seq_lens.iter().sum();
    if total_len as usize != batch.input_tokens.len() {
        return Err(EngineError::BackendError(format!(
            "invalid execution batch: sum(seq_lens)={total_len} != input_tokens.len()={}",
            batch.input_tokens.len()
        )));
    }

    for (i, (&seq_len, block_table)) in batch
        .seq_lens
        .iter()
        .zip(batch.block_tables.iter())
        .enumerate()
    {
        if seq_len == 0 {
            return Err(EngineError::BackendError(format!(
                "invalid execution batch: sequence {} has zero length",
                batch.seq_ids[i]
            )));
        }
        if block_table.is_empty() {
            return Err(EngineError::BackendError(format!(
                "invalid execution batch: sequence {} has an empty block table",
                batch.seq_ids[i]
            )));
        }
    }

    Ok(())
}

/// 校验后端返回的执行输出是否符合契约。
///
/// 一个损坏的后端（返回空输出、缺失 token、重复/错误的 seq id 或长度错配
/// 的 logprobs）不能让它静默通过——否则请求会被卡在调度器中永不完成。
fn validate_execution_output(
    batch: &ExecutionBatch,
    output: &ExecutionOutput,
) -> Result<(), EngineError> {
    if output.next_tokens.len() != output.seq_ids.len() {
        return Err(EngineError::BackendError(format!(
            "malformed execution output: next_tokens.len()={} != seq_ids.len()={}",
            output.next_tokens.len(),
            output.seq_ids.len()
        )));
    }

    if output.seq_ids.len() != batch.seq_ids.len() {
        return Err(EngineError::BackendError(format!(
            "malformed execution output: output has {} sequences but batch has {}",
            output.seq_ids.len(),
            batch.seq_ids.len()
        )));
    }

    let seen: HashSet<SeqId> = output.seq_ids.iter().copied().collect();
    if seen.len() != output.seq_ids.len() {
        return Err(EngineError::BackendError(
            "malformed execution output: seq_ids contains duplicates".to_string(),
        ));
    }

    let expected: HashSet<SeqId> = batch.seq_ids.iter().copied().collect();
    if seen != expected {
        return Err(EngineError::BackendError(format!(
            "malformed execution output: seq_ids {:?} does not match batch seq_ids {:?}",
            seen, expected
        )));
    }

    if !output.logprobs.is_empty() && output.logprobs.len() != output.seq_ids.len() {
        return Err(EngineError::BackendError(format!(
            "malformed execution output: logprobs.len()={} != seq_ids.len()={}",
            output.logprobs.len(),
            output.seq_ids.len()
        )));
    }

    Ok(())
}

/// 从调度器输出构建执行批次
///
/// 将 `SchedulerOutput` 中的 prefill 和 decode 序列转换为 `ExecutionBatch`，
/// 包含 GPU kernel 所需的扁平化 token、位置、块表等数据。
///
/// # 错误
///
/// 如果某个 decode 序列缺少输入 token 或无法计算位置（调度器状态不一致），
/// 返回 [`EngineError::InvalidStateTransition`] 而非静默跳过——被跳过的序列既不会被推进
/// 也不会被标记失败，将导致请求永久停滞。
pub fn build_execution_batch(
    scheduler_output: &SchedulerOutput,
) -> Result<ExecutionBatch, EngineError> {
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
        let input_token = seq.decode_input_token().ok_or_else(|| {
            EngineError::InvalidStateTransition(format!(
                "decode sequence {seq_id} has no input token for batch construction"
            ))
        })?;
        let position = seq.decode_position().ok_or_else(|| {
            EngineError::InvalidStateTransition(format!(
                "decode sequence {seq_id} has zero context length"
            ))
        })?;

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

    Ok(batch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Request, RequestState, Sequence};

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
            num_generated_tokens: 2,
        };

        let scheduler_output = SchedulerOutput {
            prefill_sequences: Vec::new(),
            decode_sequences: vec![sequence],
            total_tokens: 1,
        };

        let batch = build_execution_batch(&scheduler_output).unwrap();

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
            num_generated_tokens: 0,
        };

        let scheduler_output = SchedulerOutput {
            prefill_sequences: Vec::new(),
            decode_sequences: vec![sequence],
            total_tokens: 1,
        };

        let batch = build_execution_batch(&scheduler_output).unwrap();

        assert_eq!(batch.input_tokens, vec![32]);
        assert_eq!(batch.positions, vec![2]);
        assert_eq!(batch.context_lens, vec![3]);
        assert_eq!(batch.seq_ids, vec![9]);
    }

    #[test]
    fn test_decode_batch_fails_fast_on_empty_sequence() {
        // 一个既无输入 token 又无输出 token 的 decode 序列属于调度器状态不一致，
        // 必须报错而非静默跳过（否则请求会永久滞留队列）。
        let request = Request::new(1, Vec::new(), crate::types::GenerationParams::default());

        let sequence = Sequence {
            seq_id: 13,
            request,
            logical_blocks: Vec::new(),
            num_generated_tokens: 0,
        };

        let scheduler_output = SchedulerOutput {
            prefill_sequences: Vec::new(),
            decode_sequences: vec![sequence],
            total_tokens: 1,
        };

        let result = build_execution_batch(&scheduler_output);
        assert!(matches!(
            result,
            Err(EngineError::InvalidStateTransition(_))
        ));
    }

    fn sample_batch() -> ExecutionBatch {
        ExecutionBatch {
            input_tokens: vec![1, 2, 3, 4, 5],
            positions: vec![0, 1, 2, 0, 1],
            seq_lens: vec![3, 2],
            block_tables: vec![vec![0, 1], vec![2]],
            is_prefill: vec![true, true],
            seq_ids: vec![1, 2],
            context_lens: vec![3, 2],
        }
    }

    #[test]
    fn test_validate_batch_shape_accepts_valid_batch() {
        assert!(validate_batch_shape(&sample_batch()).is_ok());
    }

    #[test]
    fn test_validate_batch_shape_rejects_token_position_mismatch() {
        let mut batch = sample_batch();
        batch.positions.pop();
        assert!(matches!(
            validate_batch_shape(&batch),
            Err(EngineError::BackendError(msg)) if msg.contains("invalid execution batch")
        ));
    }

    #[test]
    fn test_validate_batch_shape_rejects_seq_len_mismatch() {
        let mut batch = sample_batch();
        batch.seq_lens.push(99); // 多一个序列字段，与 seq_ids 不齐
        assert!(matches!(
            validate_batch_shape(&batch),
            Err(EngineError::BackendError(msg)) if msg.contains("invalid execution batch")
        ));
    }

    #[test]
    fn test_validate_batch_shape_rejects_token_total_mismatch() {
        let mut batch = sample_batch();
        batch.seq_lens[0] = 4; // sum(seq_lens)=6 != 5
        assert!(matches!(
            validate_batch_shape(&batch),
            Err(EngineError::BackendError(msg)) if msg.contains("invalid execution batch")
        ));
    }

    #[test]
    fn test_validate_batch_shape_rejects_zero_len_sequence() {
        // 保持 token 总数一致，确保命中"零长度序列"检查而非长度求和检查。
        let batch = ExecutionBatch {
            input_tokens: vec![1, 2, 3],
            positions: vec![0, 1, 2],
            seq_lens: vec![3, 0],
            block_tables: vec![vec![0], vec![1]],
            is_prefill: vec![true, true],
            seq_ids: vec![1, 2],
            context_lens: vec![3, 3],
        };
        assert!(matches!(
            validate_batch_shape(&batch),
            Err(EngineError::BackendError(msg)) if msg.contains("zero length")
        ));
    }

    #[test]
    fn test_validate_batch_shape_rejects_empty_block_table() {
        let mut batch = sample_batch();
        batch.block_tables[1] = Vec::new();
        assert!(matches!(
            validate_batch_shape(&batch),
            Err(EngineError::BackendError(msg)) if msg.contains("empty block table")
        ));
    }

    #[test]
    fn test_validate_execution_output_accepts_valid_output() {
        let batch = sample_batch();
        let output = ExecutionOutput {
            next_tokens: vec![10, 11],
            seq_ids: vec![2, 1], // 顺序可不同
            logprobs: Vec::new(),
        };
        assert!(validate_execution_output(&batch, &output).is_ok());
    }

    #[test]
    fn test_validate_execution_output_rejects_empty_output() {
        let batch = sample_batch();
        let output = ExecutionOutput::default();
        assert!(matches!(
            validate_execution_output(&batch, &output),
            Err(EngineError::BackendError(msg)) if msg.contains("malformed execution output")
        ));
    }

    #[test]
    fn test_validate_execution_output_rejects_missing_token() {
        let batch = sample_batch();
        let output = ExecutionOutput {
            next_tokens: vec![10], // 少一个 token
            seq_ids: vec![1, 2],
            logprobs: Vec::new(),
        };
        assert!(matches!(
            validate_execution_output(&batch, &output),
            Err(EngineError::BackendError(msg)) if msg.contains("malformed execution output")
        ));
    }

    #[test]
    fn test_validate_execution_output_rejects_duplicate_seq_id() {
        let batch = sample_batch();
        let output = ExecutionOutput {
            next_tokens: vec![10, 11],
            seq_ids: vec![1, 1],
            logprobs: Vec::new(),
        };
        assert!(matches!(
            validate_execution_output(&batch, &output),
            Err(EngineError::BackendError(msg)) if msg.contains("duplicates")
        ));
    }

    #[test]
    fn test_validate_execution_output_rejects_wrong_seq_id_set() {
        let batch = sample_batch();
        let output = ExecutionOutput {
            next_tokens: vec![10, 11],
            seq_ids: vec![1, 3], // 3 不在 batch 中
            logprobs: Vec::new(),
        };
        assert!(matches!(
            validate_execution_output(&batch, &output),
            Err(EngineError::BackendError(msg)) if msg.contains("does not match")
        ));
    }

    #[test]
    fn test_validate_execution_output_rejects_wrong_logprobs_length() {
        let batch = sample_batch();
        let output = ExecutionOutput {
            next_tokens: vec![10, 11],
            seq_ids: vec![1, 2],
            logprobs: vec![Some(crate::types::TokenLogprobs {
                token: 10,
                logprob: -1.0,
                top_logprobs: Vec::new(),
            })], // 长度 1 != 2
        };
        assert!(matches!(
            validate_execution_output(&batch, &output),
            Err(EngineError::BackendError(msg)) if msg.contains("malformed execution output")
        ));
    }
}
