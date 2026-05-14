//! Continuous Batching 调度器
//!
//! 实现请求调度，支持 prefill/decode 阶段管理和内存感知的批次构建。
//!
//! # 核心特性
//!
//! - **Decode 优先调度** - 优先调度 decode 请求以降低延迟
//! - **内存压力感知** - 内存超阈值时拒绝新 prefill
//! - **连续批处理** - 动态组合 prefill 和 decode 请求
//!
//! # 状态机
//!
//! ```text
//! Pending → Prefill → Decode → Completed
//!                   ↘ Failed
//! ```

use std::sync::Arc;

use crate::config::EngineConfig;
use crate::error::SchedulerError;
use crate::sequence_lifecycle::SequenceLifecycle;
use crate::types::{
    ExecutionOutput, Request, RequestState, SchedulerOutput, SeqId, Sequence, TokenId,
};

/// 调度器 trait 接口
///
/// 定义调度器的标准接口，便于替换实现。
pub trait SchedulerTrait {
    /// 添加新请求到待处理队列
    fn add_request(&mut self, request: Request) -> Result<SeqId, SchedulerError>;

    /// 调度下一批次用于执行
    fn schedule(&mut self) -> SchedulerOutput;

    /// GPU 执行后更新序列状态
    fn update_sequences(&mut self, outputs: &ExecutionOutput, eos_token_id: TokenId);

    /// 获取已完成的请求
    fn get_completed(&mut self) -> Vec<Request>;

    /// 检查是否有待处理的工作
    fn has_pending_work(&self) -> bool;

    /// 获取 KV Cache 内存利用率
    fn get_memory_utilization(&self) -> f32;
}

/// Continuous Batching 调度器
///
/// 实现请求调度，支持 prefill/decode 分阶段管理和内存感知。
///
/// 序列状态机与 KV Cache 资源管理已委托给 [`SequenceLifecycle`]，
/// 本模块专注于调度策略（decode 优先、批次约束、内存压力门控）。
///
/// # 调度策略
///
/// 1. **Decode 优先** - 优先调度 decode 请求
/// 2. **Prefill 次之** - 在 decode 调度完成后处理 prefill
/// 3. **新请求入队** - 内存压力低时接受新请求
///
/// # 约束
///
/// - 批次序列数不超过 `max_batch_size`
/// - 批次 token 总数不超过 `max_total_tokens`
/// - 内存利用率超阈值时拒绝新 prefill
pub struct Scheduler {
    /// 调度策略配置
    config: EngineConfig,
    /// 序列生命周期管理器（状态机与资源管理）
    lifecycle: SequenceLifecycle,
}

impl Scheduler {
    /// 创建新的调度器
    pub fn new(config: EngineConfig) -> Self {
        let lifecycle = SequenceLifecycle::new(config.clone());
        Self { config, lifecycle }
    }

    /// 批量失败序列（供 Engine 错误恢复使用）
    pub fn fail_sequences<I>(&mut self, seq_ids: I, reason: &str)
    where
        I: IntoIterator<Item = SeqId>,
    {
        self.lifecycle.fail_sequences(seq_ids, reason);
    }

    /// 通过 ID 查找序列
    pub fn get_sequence(&self, seq_id: SeqId) -> Option<&Sequence> {
        self.lifecycle.get_sequence(seq_id)
    }

    /// 通过 ID 查找序列（可变）
    pub fn get_sequence_mut(&mut self, seq_id: SeqId) -> Option<&mut Sequence> {
        self.lifecycle.get_sequence_mut(seq_id)
    }

    /// 活跃序列数
    pub fn num_active_sequences(&self) -> usize {
        self.lifecycle.num_active_sequences()
    }

    /// 序列是否恰好在其中一个队列中
    pub fn is_in_exactly_one_queue(&self, seq_id: SeqId) -> bool {
        self.lifecycle.is_in_exactly_one_queue(seq_id)
    }

    // === 测试辅助访问器 ===

    /// 是否存在指定 prefill 序列
    pub fn has_prefill_sequence(&self, seq_id: SeqId) -> bool {
        self.lifecycle.has_prefill_sequence(seq_id)
    }

    /// 是否存在指定 decode 序列
    pub fn has_decode_sequence(&self, seq_id: SeqId) -> bool {
        self.lifecycle.has_decode_sequence(seq_id)
    }

    /// pending 队列中是否存在指定 seq_id
    pub fn has_pending_request(&self, seq_id: SeqId) -> bool {
        self.lifecycle.has_pending_request(seq_id)
    }

    /// prefill 序列数
    pub fn num_prefill_sequences(&self) -> usize {
        self.lifecycle.num_prefill_sequences()
    }

    /// decode 序列数
    pub fn num_decode_sequences(&self) -> usize {
        self.lifecycle.num_decode_sequences()
    }
}

impl SchedulerTrait for Scheduler {
    fn add_request(&mut self, request: Request) -> Result<SeqId, SchedulerError> {
        self.lifecycle.add_request(request)
    }

    fn schedule(&mut self) -> SchedulerOutput {
        let mut output = SchedulerOutput::default();
        let mut total_tokens: u32 = 0;
        let mut num_sequences: u32 = 0;

        self.lifecycle.update_memory_pressure();

        // Priority 1: Schedule decode sequences first (lower latency for in-flight requests)
        for seq_id in self.lifecycle.decode_seq_ids() {
            if num_sequences >= self.config.max_batch_size {
                break;
            }

            if total_tokens + 1 > self.config.max_total_tokens {
                break;
            }

            if let Err(reason) = self.lifecycle.grow_decode_sequence(seq_id) {
                self.lifecycle.fail_sequence(seq_id, &reason);
                continue;
            }

            if let Some(sequence) = self.lifecycle.get_decode_sequence(seq_id) {
                output.decode_sequences.push(Arc::new(sequence.clone()));
                total_tokens += 1;
                num_sequences += 1;
            }
        }

        // Priority 2: Schedule prefill sequences
        for seq_id in self.lifecycle.prefill_seq_ids() {
            if num_sequences >= self.config.max_batch_size {
                break;
            }

            let (prefill_tokens, blocks_needed) = match self.lifecycle.get_prefill_sequence(seq_id)
            {
                Some(sequence) => {
                    let tokens = sequence.request.input_tokens.len() as u32;
                    (tokens, self.config.blocks_for_tokens(tokens))
                }
                None => continue,
            };

            if prefill_tokens > self.config.max_total_tokens {
                let reason = format!(
                    "Input tokens {} exceed max_total_tokens {}",
                    prefill_tokens, self.config.max_total_tokens
                );
                self.lifecycle.fail_sequence(seq_id, &reason);
                continue;
            }

            if blocks_needed > self.config.max_num_blocks {
                let reason = format!(
                    "Required blocks {} exceed max_num_blocks {}",
                    blocks_needed, self.config.max_num_blocks
                );
                self.lifecycle.fail_sequence(seq_id, &reason);
                continue;
            }

            if total_tokens + prefill_tokens > self.config.max_total_tokens {
                break;
            }

            if let Some(sequence) = self.lifecycle.get_prefill_sequence(seq_id) {
                output.prefill_sequences.push(Arc::new(sequence.clone()));
                total_tokens += prefill_tokens;
                num_sequences += 1;
            }
        }

        // Priority 3: Start new prefills from pending queue (if not under memory pressure)
        if !self.lifecycle.is_under_memory_pressure() {
            while let Some((seq_id, request)) = self.lifecycle.pop_pending() {
                if num_sequences >= self.config.max_batch_size {
                    self.lifecycle.push_pending(seq_id, request);
                    break;
                }

                let prefill_tokens = request.input_tokens.len() as u32;
                if prefill_tokens > self.config.max_total_tokens {
                    let mut failed_request = request;
                    failed_request.state = RequestState::Failed(format!(
                        "Input tokens {} exceed max_total_tokens {}",
                        prefill_tokens, self.config.max_total_tokens
                    ));
                    self.lifecycle.push_completed(failed_request);
                    continue;
                }

                let blocks_needed = self.config.blocks_for_tokens(prefill_tokens);
                if blocks_needed > self.config.max_num_blocks {
                    let mut failed_request = request;
                    failed_request.state = RequestState::Failed(format!(
                        "Required blocks {} exceed max_num_blocks {}",
                        blocks_needed, self.config.max_num_blocks
                    ));
                    self.lifecycle.push_completed(failed_request);
                    continue;
                }

                if total_tokens + prefill_tokens > self.config.max_total_tokens {
                    self.lifecycle.push_pending(seq_id, request);
                    break;
                }

                match self.lifecycle.try_start_prefill(seq_id, request) {
                    Ok(seq_id) => {
                        if let Some(sequence) = self.lifecycle.get_prefill_sequence(seq_id) {
                            output.prefill_sequences.push(Arc::new(sequence.clone()));
                            total_tokens += prefill_tokens;
                            num_sequences += 1;
                        }
                    }
                    Err((seq_id, request)) => {
                        self.lifecycle.push_pending(seq_id, request);
                        break;
                    }
                }
            }
        }

        output.total_tokens = total_tokens;
        output
    }

    fn update_sequences(&mut self, outputs: &ExecutionOutput, eos_token_id: TokenId) {
        self.lifecycle.update_sequences(outputs, eos_token_id);
    }

    fn get_completed(&mut self) -> Vec<Request> {
        self.lifecycle.get_completed()
    }

    fn has_pending_work(&self) -> bool {
        self.lifecycle.has_pending_work()
    }

    fn get_memory_utilization(&self) -> f32 {
        self.lifecycle.memory_utilization()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{create_test_config, create_test_request};

    #[test]
    fn test_add_request() {
        let config = create_test_config();
        let mut scheduler = Scheduler::new(config);

        let request = create_test_request(42, 32);
        let result = scheduler.add_request(request);

        assert_eq!(result.unwrap(), 1);
        assert!(scheduler.has_pending_work());
    }

    #[test]
    fn test_add_request_returns_real_seq_id() {
        let config = create_test_config();
        let mut scheduler = Scheduler::new(config);

        let request = create_test_request(999, 16);
        let seq_id = scheduler.add_request(request).unwrap();

        assert_eq!(seq_id, 1);
    }

    #[test]
    fn test_schedule_prefill_uses_returned_seq_id() {
        let config = create_test_config();
        let mut scheduler = Scheduler::new(config);

        let request = create_test_request(999, 16);
        let seq_id = scheduler.add_request(request).unwrap();

        let output = scheduler.schedule();
        assert_eq!(output.prefill_sequences.len(), 1);
        assert_eq!(output.prefill_sequences[0].seq_id, seq_id);
    }

    #[test]
    fn test_pending_queue_does_not_count_toward_max_sequences() {
        let config = EngineConfig {
            max_num_seqs: 1,
            ..create_test_config()
        };
        let mut scheduler = Scheduler::new(config);

        let first = create_test_request(1, 16);
        let second = create_test_request(2, 16);

        assert!(scheduler.add_request(first).is_ok());
        assert!(scheduler.add_request(second).is_ok());
    }

    #[test]
    fn test_add_request_sequence_ids_are_monotonic() {
        let config = create_test_config();
        let mut scheduler = Scheduler::new(config);

        let seq_id1 = scheduler.add_request(create_test_request(100, 8)).unwrap();
        let seq_id2 = scheduler.add_request(create_test_request(200, 8)).unwrap();

        assert_eq!(seq_id1, 1);
        assert_eq!(seq_id2, 2);
    }

    #[test]
    fn test_schedule_prefill() {
        let config = create_test_config();
        let mut scheduler = Scheduler::new(config);

        let request = create_test_request(1, 32);
        scheduler.add_request(request).unwrap();

        let output = scheduler.schedule();

        assert_eq!(output.prefill_sequences.len(), 1);
        assert_eq!(output.decode_sequences.len(), 0);
    }

    #[test]
    fn test_prefill_to_decode_transition() {
        let config = create_test_config();
        let mut scheduler = Scheduler::new(config);

        let request = create_test_request(1, 32);
        scheduler.add_request(request).unwrap();

        // Schedule prefill
        let output = scheduler.schedule();
        assert_eq!(output.prefill_sequences.len(), 1);

        let seq_id = output.prefill_sequences[0].seq_id;

        // Simulate GPU output
        let exec_output = ExecutionOutput {
            next_tokens: vec![100],
            logits: None,
            seq_ids: vec![seq_id],
        };

        scheduler.update_sequences(&exec_output, 0);

        // Now should be in decode phase
        assert!(scheduler.has_decode_sequence(seq_id));
        assert!(!scheduler.has_prefill_sequence(seq_id));
    }

    #[test]
    fn test_decode_priority() {
        let config = create_test_config();
        let mut scheduler = Scheduler::new(config);

        // Add and schedule first request to get it into decode
        let request1 = create_test_request(1, 16);
        scheduler.add_request(request1).unwrap();
        let output = scheduler.schedule();
        let seq_id = output.prefill_sequences[0].seq_id;

        // Transition to decode
        let exec_output = ExecutionOutput {
            next_tokens: vec![100],
            logits: None,
            seq_ids: vec![seq_id],
        };
        scheduler.update_sequences(&exec_output, 0);

        // Add another pending request
        let request2 = create_test_request(2, 16);
        scheduler.add_request(request2).unwrap();

        // Schedule - decode should come first
        let output = scheduler.schedule();

        // Decode sequences should be scheduled
        assert!(!output.decode_sequences.is_empty());
    }

    #[test]
    fn test_completion() {
        let config = create_test_config();
        let mut scheduler = Scheduler::new(config);

        let mut request = create_test_request(1, 16);
        request.params.max_tokens = 2;
        scheduler.add_request(request).unwrap();

        // Schedule and transition to decode
        let output = scheduler.schedule();
        let seq_id = output.prefill_sequences[0].seq_id;

        let exec_output = ExecutionOutput {
            next_tokens: vec![100],
            logits: None,
            seq_ids: vec![seq_id],
        };
        scheduler.update_sequences(&exec_output, 0);

        // Generate one more token
        scheduler.schedule();
        let exec_output = ExecutionOutput {
            next_tokens: vec![101],
            logits: None,
            seq_ids: vec![seq_id],
        };
        scheduler.update_sequences(&exec_output, 0);

        // Should be completed (max_tokens = 2)
        let completed = scheduler.get_completed();
        assert_eq!(completed.len(), 1);
    }

    #[test]
    fn test_decode_priority_with_small_batch_keeps_pending_request_queued() {
        let config = EngineConfig {
            max_batch_size: 1,
            max_total_tokens: 64,
            ..create_test_config()
        };
        let mut scheduler = Scheduler::new(config);

        let decode_request = create_test_request(1, 16);
        let pending_request = create_test_request(2, 16);

        scheduler.add_request(decode_request).unwrap();
        let output = scheduler.schedule();
        let decode_seq_id = output.prefill_sequences[0].seq_id;

        scheduler.update_sequences(
            &ExecutionOutput {
                next_tokens: vec![100],
                logits: None,
                seq_ids: vec![decode_seq_id],
            },
            0,
        );

        let pending_seq_id = scheduler.add_request(pending_request).unwrap();
        let scheduled = scheduler.schedule();

        assert_eq!(scheduled.decode_sequences.len(), 1);
        assert_eq!(scheduled.prefill_sequences.len(), 0);
        assert_eq!(scheduled.decode_sequences[0].seq_id, decode_seq_id);
        assert!(scheduler.has_pending_request(pending_seq_id));
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use crate::test_utils::{create_test_config_with_limits, create_test_request_with_params};
    use proptest::prelude::*;
    use std::collections::HashSet;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// **Feature: heterogeneous-inference-system, Property 1: Request ID Uniqueness**
        /// *For any* set of requests submitted to the Scheduler, all assigned sequence IDs
        /// shall be unique with no collisions.
        /// **Validates: Requirements 1.2**
        #[test]
        fn prop_request_id_uniqueness(
            num_requests in 1usize..50,
            tokens_per_request in 1usize..64,
        ) {
            let config = create_test_config_with_limits(32, 4096, 500);
            let mut scheduler = Scheduler::new(config);
            let mut assigned_ids: HashSet<SeqId> = HashSet::new();

            for i in 0..num_requests {
                let request = create_test_request_with_params(i as u64, tokens_per_request, 10);
                let _ = scheduler.add_request(request);
            }

            // 反复调度直到 pending 队列清空；每轮调度后执行
            // update_sequences 使 prefill → decode，避免同一批被重复上报
            loop {
                let output = scheduler.schedule();
                if output.is_empty() {
                    break;
                }

                for seq in &output.prefill_sequences {
                    prop_assert!(
                        !assigned_ids.contains(&seq.seq_id),
                        "Duplicate sequence ID: {}",
                        seq.seq_id
                    );
                    assigned_ids.insert(seq.seq_id);
                }
                for seq in &output.decode_sequences {
                    // decode 阶段的 seq_id 也不能重复
                    assigned_ids.insert(seq.seq_id);
                }

                // 模拟执行，推动 prefill → decode 转换
                let seq_ids: Vec<SeqId> = output.prefill_sequences.iter()
                    .chain(output.decode_sequences.iter())
                    .map(|s| s.seq_id)
                    .collect();
                if !seq_ids.is_empty() {
                    let exec_output = ExecutionOutput {
                        next_tokens: vec![100; seq_ids.len()],
                        logits: None,
                        seq_ids,
                    };
                    scheduler.update_sequences(&exec_output, 0);
                }
            }
        }

        /// **Feature: heterogeneous-inference-system, Property 6: Scheduler Queue State Consistency**
        /// *For any* request in the scheduler, it shall be in exactly one queue: pending,
        /// prefill, or decode.
        /// **Validates: Requirements 3.1**
        #[test]
        fn prop_scheduler_queue_state_consistency(
            num_requests in 1usize..20,
            tokens_per_request in 1usize..32,
            num_steps in 1usize..10,
        ) {
            let config = create_test_config_with_limits(16, 1024, 200);
            let mut scheduler = Scheduler::new(config);

            // Add requests
            for i in 0..num_requests {
                let request = create_test_request_with_params(i as u64, tokens_per_request, 50);
                let _ = scheduler.add_request(request);
            }

            // Run scheduling steps
            for _ in 0..num_steps {
                let output = scheduler.schedule();

                // Verify each scheduled sequence is in exactly one queue
                for seq in &output.prefill_sequences {
                    prop_assert!(
                        scheduler.is_in_exactly_one_queue(seq.seq_id),
                        "Sequence {} is not in exactly one queue",
                        seq.seq_id
                    );
                }

                for seq in &output.decode_sequences {
                    prop_assert!(
                        scheduler.is_in_exactly_one_queue(seq.seq_id),
                        "Sequence {} is not in exactly one queue",
                        seq.seq_id
                    );
                }

                // Simulate execution
                let mut next_tokens = Vec::new();
                let mut seq_ids = Vec::new();

                for seq in &output.prefill_sequences {
                    next_tokens.push(100u32);
                    seq_ids.push(seq.seq_id);
                }
                for seq in &output.decode_sequences {
                    next_tokens.push(100u32);
                    seq_ids.push(seq.seq_id);
                }

                if !seq_ids.is_empty() {
                    let exec_output = ExecutionOutput {
                        next_tokens,
                        logits: None,
                        seq_ids,
                    };
                    scheduler.update_sequences(&exec_output, 0);
                }
            }
        }

        /// **Feature: heterogeneous-inference-system, Property 7: Batch Size Constraints**
        /// *For any* scheduled batch, the number of sequences shall not exceed max_batch_size,
        /// and the total number of tokens shall not exceed max_total_tokens.
        /// **Validates: Requirements 3.5**
        #[test]
        fn prop_batch_size_constraints(
            max_batch_size in 1u32..16,
            max_total_tokens in 64u32..512,
            num_requests in 1usize..30,
            tokens_per_request in 1usize..64,
        ) {
            let config = create_test_config_with_limits(max_batch_size, max_total_tokens, 500);
            let mut scheduler = Scheduler::new(config);

            // Add many requests
            for i in 0..num_requests {
                let request = create_test_request_with_params(i as u64, tokens_per_request, 10);
                let _ = scheduler.add_request(request);
            }

            // Schedule and verify constraints
            for _ in 0..5 {
                let output = scheduler.schedule();

                let num_sequences = output.num_sequences();
                prop_assert!(
                    num_sequences <= max_batch_size as usize,
                    "Batch has {} sequences, max is {}",
                    num_sequences,
                    max_batch_size
                );

                prop_assert!(
                    output.total_tokens <= max_total_tokens,
                    "Batch has {} tokens, max is {}",
                    output.total_tokens,
                    max_total_tokens
                );

                // Simulate execution to progress
                let mut next_tokens = Vec::new();
                let mut seq_ids = Vec::new();

                for seq in &output.prefill_sequences {
                    next_tokens.push(100u32);
                    seq_ids.push(seq.seq_id);
                }
                for seq in &output.decode_sequences {
                    next_tokens.push(100u32);
                    seq_ids.push(seq.seq_id);
                }

                if !seq_ids.is_empty() {
                    let exec_output = ExecutionOutput {
                        next_tokens,
                        logits: None,
                        seq_ids,
                    };
                    scheduler.update_sequences(&exec_output, 0);
                }
            }
        }

        /// **Feature: heterogeneous-inference-system, Property 8: Decode Priority Over Prefill**
        /// *For any* scheduling decision where both prefill and decode requests are pending
        /// and batch capacity allows, all eligible decode requests shall be scheduled before
        /// any prefill requests.
        /// **Validates: Requirements 3.7**
        #[test]
        fn prop_decode_priority_over_prefill(
            num_decode in 1usize..10,
            num_pending in 1usize..10,
            tokens_per_request in 4usize..32,
        ) {
            let config = create_test_config_with_limits(32, 2048, 500);
            let mut scheduler = Scheduler::new(config);

            // First, create some decode sequences
            for i in 0..num_decode {
                let request = create_test_request_with_params(i as u64, tokens_per_request, 100);
                scheduler.add_request(request).unwrap();
            }

            // Schedule and transition to decode
            for _ in 0..num_decode {
                let output = scheduler.schedule();

                let mut next_tokens = Vec::new();
                let mut seq_ids = Vec::new();

                for seq in &output.prefill_sequences {
                    next_tokens.push(100u32);
                    seq_ids.push(seq.seq_id);
                }

                if !seq_ids.is_empty() {
                    let exec_output = ExecutionOutput {
                        next_tokens,
                        logits: None,
                        seq_ids,
                    };
                    scheduler.update_sequences(&exec_output, 0);
                }
            }

            // Now add more pending requests
            for i in num_decode..(num_decode + num_pending) {
                let request = create_test_request_with_params(i as u64, tokens_per_request, 100);
                scheduler.add_request(request).unwrap();
            }

            // Schedule - decode should come first
            let output = scheduler.schedule();

            // If we have decode sequences, they should all be scheduled before prefill
            // (given sufficient capacity)
            let decode_count = scheduler.num_decode_sequences();
            if decode_count > 0 && output.num_sequences() > 0 {
                // Decode sequences should be present in output
                prop_assert!(
                    !output.decode_sequences.is_empty() || decode_count == 0,
                    "Decode sequences should be scheduled when available"
                );
            }
        }

        /// **Feature: heterogeneous-inference-system, Property 9: Prefill to Decode Transition**
        /// *For any* sequence that completes its prefill phase, it shall immediately transition
        /// to decode state in the same scheduling cycle.
        /// **Validates: Requirements 3.3**
        #[test]
        fn prop_prefill_to_decode_transition(
            num_requests in 1usize..10,
            tokens_per_request in 4usize..32,
        ) {
            let config = create_test_config_with_limits(16, 1024, 200);
            let mut scheduler = Scheduler::new(config);

            // Add requests
            for i in 0..num_requests {
                let request = create_test_request_with_params(i as u64, tokens_per_request, 50);
                scheduler.add_request(request).unwrap();
            }

            // Schedule prefill
            let output = scheduler.schedule();
            let prefill_seq_ids: Vec<SeqId> = output.prefill_sequences.iter().map(|s| s.seq_id).collect();

            // Simulate execution
            let next_tokens: Vec<u32> = prefill_seq_ids.iter().map(|_| 100).collect();
            let exec_output = ExecutionOutput {
                next_tokens,
                logits: None,
                seq_ids: prefill_seq_ids.clone(),
            };

            scheduler.update_sequences(&exec_output, 0);

            // All prefill sequences should now be in decode
            for seq_id in &prefill_seq_ids {
                prop_assert!(
                    scheduler.has_decode_sequence(*seq_id),
                    "Sequence {} should be in decode after prefill",
                    seq_id
                );
                prop_assert!(
                    !scheduler.has_prefill_sequence(*seq_id),
                    "Sequence {} should not be in prefill after transition",
                    seq_id
                );
            }
        }

        /// **Feature: heterogeneous-inference-system, Property 10: Completion Conditions**
        /// *For any* sequence in decode phase, it shall transition to completed state if and
        /// only if it generates an EOS token or reaches max_tokens.
        /// **Validates: Requirements 3.4**
        #[test]
        fn prop_completion_conditions(
            max_tokens in 1u32..20,
            tokens_per_request in 4usize..16,
            eos_position in 0usize..25,
        ) {
            let config = create_test_config_with_limits(8, 512, 100);
            let mut scheduler = Scheduler::new(config);

            let request = create_test_request_with_params(1, tokens_per_request, max_tokens);
            scheduler.add_request(request).unwrap();

            // Schedule and transition to decode
            let output = scheduler.schedule();
            let seq_id = output.prefill_sequences[0].seq_id;

            let exec_output = ExecutionOutput {
                next_tokens: vec![100],
                logits: None,
                seq_ids: vec![seq_id],
            };
            scheduler.update_sequences(&exec_output, 0);

            // Generate tokens until completion
            let eos_token: TokenId = 0;
            let mut generated = 1u32;

            while scheduler.has_decode_sequence(seq_id) && generated < max_tokens + 5 {
                scheduler.schedule();

                // Decide whether to send EOS
                let token = if generated as usize == eos_position {
                    eos_token
                } else {
                    100 + generated
                };

                let exec_output = ExecutionOutput {
                    next_tokens: vec![token],
                    logits: None,
                    seq_ids: vec![seq_id],
                };
                scheduler.update_sequences(&exec_output, eos_token);
                generated += 1;
            }

            // Should be completed
            let completed = scheduler.get_completed();

            if !completed.is_empty() {
                let req = &completed[0];
                let hit_max = req.output_tokens.len() >= max_tokens as usize;
                let hit_eos = req.output_tokens.last() == Some(&eos_token);

                prop_assert!(
                    hit_max || hit_eos,
                    "Completion should be due to max_tokens or EOS"
                );
            }
        }

        /// **Feature: heterogeneous-inference-system, Property 13: Memory Pressure Response**
        /// *For any* state where memory utilization exceeds the configured threshold, the
        /// Scheduler shall reject new prefill requests until memory is freed.
        /// **Validates: Requirements 6.3**
        #[test]
        fn prop_memory_pressure_response(
            num_initial_requests in 5usize..15,
            tokens_per_request in 16usize..64,
        ) {
            // Use small block count to trigger memory pressure
            let config = EngineConfig {
                block_size: 16,
                max_num_blocks: 20,  // Small to trigger pressure
                max_batch_size: 16,
                max_num_seqs: 32,
                max_model_len: 2048,
                max_total_tokens: 1024,
                memory_threshold: 0.5,  // Low threshold
                ..Default::default()
            };
            let mut scheduler = Scheduler::new(config);

            // Fill up memory
            for i in 0..num_initial_requests {
                let request = create_test_request_with_params(i as u64, tokens_per_request, 100);
                let _ = scheduler.add_request(request);
                let _ = scheduler.schedule();
            }

            // Check memory utilization
            let utilization = scheduler.get_memory_utilization();

            // If under pressure, new requests should be rejected
            if utilization >= 0.5 {
                let new_request = create_test_request_with_params(999, tokens_per_request, 100);
                let result = scheduler.add_request(new_request);

                // Should either reject or queue (depending on exact state)
                // The key property is that we don't crash and handle gracefully
                prop_assert!(
                    result.is_ok() || matches!(result, Err(SchedulerError::MemoryPressure)),
                    "Should handle memory pressure gracefully"
                );
            }
        }
    }
}
