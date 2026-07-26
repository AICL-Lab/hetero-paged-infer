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

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use crate::config::EngineConfig;
use crate::error::SchedulerError;
use crate::kv_cache::KVCacheManager;
use crate::types::{
    ExecutionOutput, LogicalBlock, PhysicalBlockRef, Request, RequestState, SchedulerOutput, SeqId,
    Sequence, TokenId,
};

#[derive(Debug, Clone)]
struct PendingRequest {
    seq_id: SeqId,
    request: Request,
}

/// Continuous Batching 调度器
///
/// 管理序列生命周期（Pending → Prefill → Decode → Completed/Failed）、
/// KV Cache 资源、以及调度策略（decode 优先、批次约束、内存压力门控）。
pub struct Scheduler {
    config: EngineConfig,
    kv_cache: KVCacheManager,
    pending_queue: VecDeque<PendingRequest>,
    prefill_sequences: HashMap<SeqId, Sequence>,
    decode_sequences: HashMap<SeqId, Sequence>,
    completed_requests: Vec<Request>,
    next_seq_id: SeqId,
    under_memory_pressure: bool,
}

impl Scheduler {
    pub fn new(config: EngineConfig) -> Self {
        let kv_cache = KVCacheManager::new(config.max_num_blocks, config.block_size);
        Self {
            config,
            kv_cache,
            pending_queue: VecDeque::new(),
            prefill_sequences: HashMap::new(),
            decode_sequences: HashMap::new(),
            completed_requests: Vec::new(),
            next_seq_id: 1,
            under_memory_pressure: false,
        }
    }

    // === 公共接口 ===

    pub fn add_request(&mut self, request: Request) -> Result<SeqId, SchedulerError> {
        self.update_memory_pressure();

        if self.under_memory_pressure {
            return Err(SchedulerError::MemoryPressure);
        }

        let total_sequences = self.pending_queue.len() + self.num_active_sequences();
        if total_sequences >= self.config.max_num_seqs as usize {
            return Err(SchedulerError::MaxConcurrentSequencesReached(
                self.config.max_num_seqs,
            ));
        }

        let seq_id = self.next_seq_id;
        self.next_seq_id += 1;
        self.pending_queue
            .push_back(PendingRequest { seq_id, request });
        Ok(seq_id)
    }

    pub fn schedule(&mut self) -> SchedulerOutput {
        let mut output = SchedulerOutput::default();
        let mut total_tokens: u32 = 0;
        let mut num_sequences: u32 = 0;

        self.update_memory_pressure();

        // Priority 1: Schedule decode sequences first (lower latency for in-flight requests)
        for seq_id in self.decode_seq_ids() {
            if num_sequences >= self.config.max_batch_size {
                break;
            }

            if total_tokens + 1 > self.config.max_total_tokens {
                break;
            }

            if let Err(reason) = self.grow_decode_sequence(seq_id) {
                self.fail_sequence(seq_id, &reason);
                continue;
            }

            if let Some(sequence) = self.decode_sequences.get(&seq_id) {
                output.decode_sequences.push(Arc::new(sequence.clone()));
                total_tokens += 1;
                num_sequences += 1;
            }
        }

        // Priority 2: Schedule prefill sequences
        for seq_id in self.prefill_seq_ids() {
            if num_sequences >= self.config.max_batch_size {
                break;
            }

            let (prefill_tokens, blocks_needed) = match self.prefill_sequences.get(&seq_id) {
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
                self.fail_sequence(seq_id, &reason);
                continue;
            }

            if blocks_needed > self.config.max_num_blocks {
                let reason = format!(
                    "Required blocks {} exceed max_num_blocks {}",
                    blocks_needed, self.config.max_num_blocks
                );
                self.fail_sequence(seq_id, &reason);
                continue;
            }

            if total_tokens + prefill_tokens > self.config.max_total_tokens {
                break;
            }

            if let Some(sequence) = self.prefill_sequences.get(&seq_id) {
                output.prefill_sequences.push(Arc::new(sequence.clone()));
                total_tokens += prefill_tokens;
                num_sequences += 1;
            }
        }

        // Priority 3: Start new prefills from pending queue (if not under memory pressure)
        if !self.under_memory_pressure {
            while let Some(pending) = self.pending_queue.pop_front() {
                let (seq_id, request) = (pending.seq_id, pending.request);

                if num_sequences >= self.config.max_batch_size {
                    self.pending_queue
                        .push_front(PendingRequest { seq_id, request });
                    break;
                }

                let prefill_tokens = request.input_tokens.len() as u32;
                if prefill_tokens > self.config.max_total_tokens {
                    let mut failed_request = request;
                    failed_request.state = RequestState::Failed(format!(
                        "Input tokens {} exceed max_total_tokens {}",
                        prefill_tokens, self.config.max_total_tokens
                    ));
                    self.completed_requests.push(failed_request);
                    continue;
                }

                let blocks_needed = self.config.blocks_for_tokens(prefill_tokens);
                if blocks_needed > self.config.max_num_blocks {
                    let mut failed_request = request;
                    failed_request.state = RequestState::Failed(format!(
                        "Required blocks {} exceed max_num_blocks {}",
                        blocks_needed, self.config.max_num_blocks
                    ));
                    self.completed_requests.push(failed_request);
                    continue;
                }

                if total_tokens + prefill_tokens > self.config.max_total_tokens {
                    self.pending_queue
                        .push_front(PendingRequest { seq_id, request });
                    break;
                }

                match self.try_start_prefill(seq_id, request) {
                    Ok(seq_id) => {
                        if let Some(sequence) = self.prefill_sequences.get(&seq_id) {
                            output.prefill_sequences.push(Arc::new(sequence.clone()));
                            total_tokens += prefill_tokens;
                            num_sequences += 1;
                        }
                    }
                    Err((seq_id, request)) => {
                        self.pending_queue
                            .push_front(PendingRequest { seq_id, request });
                        break;
                    }
                }
            }
        }

        output.total_tokens = total_tokens;
        output
    }

    pub fn update_sequences(&mut self, outputs: &ExecutionOutput, eos_token_id: TokenId) {
        let mut to_complete = Vec::new();
        let mut to_fail = Vec::new();
        let max_model_len = self.config.max_model_len as usize;

        for (i, &seq_id) in outputs.seq_ids.iter().enumerate() {
            let Some(next_token) = outputs.next_tokens.get(i).copied() else {
                to_fail.push((
                    seq_id,
                    format!("Malformed execution output: missing next token for seq_id {seq_id}"),
                ));
                continue;
            };

            if self.prefill_sequences.contains_key(&seq_id) {
                self.transition_prefill_to_decode(seq_id);
            }

            if let Some(sequence) = self.decode_sequences.get_mut(&seq_id) {
                sequence.request.output_tokens.push(next_token);
                sequence.num_generated_tokens += 1;

                if sequence.request.total_tokens() >= max_model_len
                    || sequence.request.is_complete(eos_token_id)
                {
                    to_complete.push(seq_id);
                }
            } else {
                to_fail.push((
                    seq_id,
                    format!("Malformed execution output: unknown seq_id {seq_id}"),
                ));
            }
        }

        for (seq_id, reason) in to_fail {
            self.fail_sequence(seq_id, &reason);
        }
        for seq_id in to_complete {
            self.complete_sequence(seq_id);
        }
    }

    pub fn get_completed(&mut self) -> Vec<Request> {
        std::mem::take(&mut self.completed_requests)
    }

    pub fn has_pending_work(&self) -> bool {
        !self.pending_queue.is_empty()
            || !self.prefill_sequences.is_empty()
            || !self.decode_sequences.is_empty()
    }

    pub fn get_memory_utilization(&self) -> f32 {
        self.kv_cache.get_memory_stats().utilization()
    }

    pub fn fail_sequences<I>(&mut self, seq_ids: I, reason: &str)
    where
        I: IntoIterator<Item = SeqId>,
    {
        for seq_id in seq_ids {
            self.fail_sequence(seq_id, reason);
        }
    }

    pub fn get_sequence(&self, seq_id: SeqId) -> Option<&Sequence> {
        self.prefill_sequences
            .get(&seq_id)
            .or_else(|| self.decode_sequences.get(&seq_id))
    }

    pub fn get_sequence_mut(&mut self, seq_id: SeqId) -> Option<&mut Sequence> {
        self.prefill_sequences
            .get_mut(&seq_id)
            .or_else(|| self.decode_sequences.get_mut(&seq_id))
    }

    pub fn num_active_sequences(&self) -> usize {
        self.prefill_sequences.len() + self.decode_sequences.len()
    }

    pub fn is_in_exactly_one_queue(&self, seq_id: SeqId) -> bool {
        let in_pending = self.pending_queue.iter().any(|p| p.seq_id == seq_id);
        let in_prefill = self.prefill_sequences.contains_key(&seq_id);
        let in_decode = self.decode_sequences.contains_key(&seq_id);
        let count = [in_pending, in_prefill, in_decode]
            .iter()
            .filter(|&&flag| flag)
            .count();
        count == 1
    }

    pub fn has_prefill_sequence(&self, seq_id: SeqId) -> bool {
        self.prefill_sequences.contains_key(&seq_id)
    }

    pub fn has_decode_sequence(&self, seq_id: SeqId) -> bool {
        self.decode_sequences.contains_key(&seq_id)
    }

    pub fn has_pending_request(&self, seq_id: SeqId) -> bool {
        self.pending_queue.iter().any(|p| p.seq_id == seq_id)
    }

    pub fn num_prefill_sequences(&self) -> usize {
        self.prefill_sequences.len()
    }

    pub fn num_decode_sequences(&self) -> usize {
        self.decode_sequences.len()
    }

    // === 内部方法 ===

    fn update_memory_pressure(&mut self) {
        let stats = self.kv_cache.get_memory_stats();
        self.under_memory_pressure = stats.utilization() >= self.config.memory_threshold;
    }

    fn try_start_prefill(
        &mut self,
        seq_id: SeqId,
        request: Request,
    ) -> Result<SeqId, (SeqId, Request)> {
        let num_tokens = request.input_tokens.len() as u32;
        let blocks_needed = self.config.blocks_for_tokens(num_tokens);

        if !self.kv_cache.can_allocate(blocks_needed) {
            return Err((seq_id, request));
        }

        if self.kv_cache.allocate_sequence(seq_id, num_tokens).is_err() {
            return Err((seq_id, request));
        }

        let mut sequence = Sequence::new(seq_id, request);
        sequence.request.state = RequestState::Prefill;

        if let Some(block_table) = self.kv_cache.get_block_table(seq_id) {
            sequence.logical_blocks = block_table
                .iter()
                .enumerate()
                .map(|(i, &block_idx)| {
                    LogicalBlock::with_physical(i as u32, PhysicalBlockRef { block_idx })
                })
                .collect();
        }

        self.prefill_sequences.insert(seq_id, sequence);
        Ok(seq_id)
    }

    fn grow_decode_sequence(&mut self, seq_id: SeqId) -> Result<(), String> {
        let Some(seq) = self.decode_sequences.get(&seq_id) else {
            return Err(format!("Decode sequence not found: {seq_id}"));
        };

        let current_tokens = seq.context_len();
        let current_blocks = seq.logical_blocks.len() as u32;
        let blocks_needed = self.config.blocks_for_tokens(current_tokens + 1);

        if blocks_needed > current_blocks {
            match self.kv_cache.allocate_block(seq_id) {
                Ok(physical_ref) => {
                    let Some(sequence) = self.decode_sequences.get_mut(&seq_id) else {
                        return Err(format!("Decode sequence not found: {seq_id}"));
                    };
                    let logical_idx = sequence.logical_blocks.len() as u32;
                    sequence
                        .logical_blocks
                        .push(LogicalBlock::with_physical(logical_idx, physical_ref));
                }
                Err(err) => {
                    return Err(format!("Failed to allocate KV block: {}", err));
                }
            }
        }
        Ok(())
    }

    fn transition_prefill_to_decode(&mut self, seq_id: SeqId) {
        if let Some(mut sequence) = self.prefill_sequences.remove(&seq_id) {
            sequence.request.state = RequestState::Decode;
            sequence.num_computed_tokens = sequence.request.input_tokens.len() as u32;
            self.decode_sequences.insert(seq_id, sequence);
        }
    }

    fn complete_sequence(&mut self, seq_id: SeqId) {
        if let Some(mut sequence) = self.decode_sequences.remove(&seq_id) {
            sequence.request.state = RequestState::Completed;
            self.kv_cache.free_sequence(seq_id);
            self.completed_requests.push(sequence.request);
            return;
        }

        if let Some(mut sequence) = self.prefill_sequences.remove(&seq_id) {
            sequence.request.state = RequestState::Completed;
            self.kv_cache.free_sequence(seq_id);
            self.completed_requests.push(sequence.request);
        }
    }

    fn fail_sequence(&mut self, seq_id: SeqId, reason: &str) {
        if let Some(mut sequence) = self.decode_sequences.remove(&seq_id) {
            sequence.request.state = RequestState::Failed(reason.to_string());
            self.kv_cache.free_sequence(seq_id);
            self.completed_requests.push(sequence.request);
            return;
        }

        if let Some(mut sequence) = self.prefill_sequences.remove(&seq_id) {
            sequence.request.state = RequestState::Failed(reason.to_string());
            self.kv_cache.free_sequence(seq_id);
            self.completed_requests.push(sequence.request);
            return;
        }

        if let Some(index) = self.pending_queue.iter().position(|p| p.seq_id == seq_id) {
            if let Some(mut pending) = self.pending_queue.remove(index) {
                pending.request.state = RequestState::Failed(reason.to_string());
                self.completed_requests.push(pending.request);
            }
        }
    }

    fn decode_seq_ids(&self) -> Vec<SeqId> {
        self.decode_sequences.keys().copied().collect()
    }

    fn prefill_seq_ids(&self) -> Vec<SeqId> {
        self.prefill_sequences.keys().copied().collect()
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
    fn test_pending_queue_counts_toward_max_sequences() {
        let config = EngineConfig {
            max_num_seqs: 1,
            ..create_test_config()
        };
        let mut scheduler = Scheduler::new(config);

        let first = create_test_request(1, 16);
        let second = create_test_request(2, 16);

        assert!(scheduler.add_request(first).is_ok());
        assert!(matches!(
            scheduler.add_request(second),
            Err(SchedulerError::MaxConcurrentSequencesReached(1))
        ));
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

        let output = scheduler.schedule();
        assert_eq!(output.prefill_sequences.len(), 1);

        let seq_id = output.prefill_sequences[0].seq_id;

        let exec_output = ExecutionOutput {
            next_tokens: vec![100],
            logits: None,
            seq_ids: vec![seq_id],
        };

        scheduler.update_sequences(&exec_output, 0);

        assert!(scheduler.has_decode_sequence(seq_id));
        assert!(!scheduler.has_prefill_sequence(seq_id));
    }

    #[test]
    fn test_decode_priority() {
        let config = create_test_config();
        let mut scheduler = Scheduler::new(config);

        let request1 = create_test_request(1, 16);
        scheduler.add_request(request1).unwrap();
        let output = scheduler.schedule();
        let seq_id = output.prefill_sequences[0].seq_id;

        let exec_output = ExecutionOutput {
            next_tokens: vec![100],
            logits: None,
            seq_ids: vec![seq_id],
        };
        scheduler.update_sequences(&exec_output, 0);

        let request2 = create_test_request(2, 16);
        scheduler.add_request(request2).unwrap();

        let output = scheduler.schedule();

        assert!(!output.decode_sequences.is_empty());
    }

    #[test]
    fn test_completion() {
        let config = create_test_config();
        let mut scheduler = Scheduler::new(config);

        let mut request = create_test_request(1, 16);
        request.params.max_tokens = 2;
        scheduler.add_request(request).unwrap();

        let output = scheduler.schedule();
        let seq_id = output.prefill_sequences[0].seq_id;

        let exec_output = ExecutionOutput {
            next_tokens: vec![100],
            logits: None,
            seq_ids: vec![seq_id],
        };
        scheduler.update_sequences(&exec_output, 0);

        scheduler.schedule();
        let exec_output = ExecutionOutput {
            next_tokens: vec![101],
            logits: None,
            seq_ids: vec![seq_id],
        };
        scheduler.update_sequences(&exec_output, 0);

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

    #[test]
    fn test_add_request_counts_pending_toward_max_num_seqs() {
        let mut scheduler = Scheduler::new(EngineConfig {
            max_num_seqs: 1,
            block_size: 4,
            max_num_blocks: 32,
            max_batch_size: 8,
            max_model_len: 32,
            max_total_tokens: 32,
            memory_threshold: 0.9,
            ..Default::default()
        });

        let req = |id: u64| {
            Request::new(
                id,
                vec![10, 11, 12],
                crate::types::GenerationParams {
                    max_tokens: 2,
                    ..Default::default()
                },
            )
        };

        assert!(scheduler.add_request(req(1)).is_ok());
        let result = scheduler.add_request(req(2));

        assert!(matches!(
            result,
            Err(SchedulerError::MaxConcurrentSequencesReached(1))
        ));
    }

    #[test]
    fn test_grow_decode_sequence_fails_for_missing_sequence() {
        let mut scheduler = Scheduler::new(create_test_config());

        let result = scheduler.grow_decode_sequence(999);

        assert!(result.is_err(), "missing decode sequence must return Err");
    }

    #[test]
    fn test_update_sequences_fails_request_when_output_missing_token() {
        let mut scheduler = Scheduler::new(EngineConfig {
            block_size: 4,
            max_num_blocks: 32,
            max_batch_size: 8,
            max_num_seqs: 4,
            max_model_len: 32,
            max_total_tokens: 32,
            memory_threshold: 0.9,
            ..Default::default()
        });

        let request = Request::new(
            1,
            vec![10, 11, 12],
            crate::types::GenerationParams {
                max_tokens: 1,
                ..Default::default()
            },
        );
        let seq_id = scheduler.add_request(request).unwrap();
        let pending = scheduler.pending_queue.pop_front().unwrap();
        scheduler.try_start_prefill(pending.seq_id, pending.request).unwrap();
        scheduler.transition_prefill_to_decode(seq_id);

        scheduler.update_sequences(
            &ExecutionOutput {
                next_tokens: Vec::new(),
                logits: None,
                seq_ids: vec![seq_id],
            },
            2,
        );

        assert!(scheduler.get_sequence(seq_id).is_none());
        let completed = scheduler.get_completed();
        assert_eq!(completed.len(), 1);
        assert!(
            matches!(&completed[0].state, RequestState::Failed(message) if message.contains("missing next token")),
            "malformed output should fail request instead of fabricating token"
        );
        assert!(
            completed[0].output_tokens.is_empty(),
            "malformed output must not append synthetic tokens"
        );
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
                    assigned_ids.insert(seq.seq_id);
                }

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

        #[test]
        fn prop_scheduler_queue_state_consistency(
            num_requests in 1usize..20,
            tokens_per_request in 1usize..32,
            num_steps in 1usize..10,
        ) {
            let config = create_test_config_with_limits(16, 1024, 200);
            let mut scheduler = Scheduler::new(config);

            for i in 0..num_requests {
                let request = create_test_request_with_params(i as u64, tokens_per_request, 50);
                let _ = scheduler.add_request(request);
            }

            for _ in 0..num_steps {
                let output = scheduler.schedule();

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

        #[test]
        fn prop_batch_size_constraints(
            max_batch_size in 1u32..16,
            max_total_tokens in 64u32..512,
            num_requests in 1usize..30,
            tokens_per_request in 1usize..64,
        ) {
            let config = create_test_config_with_limits(max_batch_size, max_total_tokens, 500);
            let mut scheduler = Scheduler::new(config);

            for i in 0..num_requests {
                let request = create_test_request_with_params(i as u64, tokens_per_request, 10);
                let _ = scheduler.add_request(request);
            }

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

        #[test]
        fn prop_decode_priority_over_prefill(
            num_decode in 1usize..10,
            num_pending in 1usize..10,
            tokens_per_request in 4usize..32,
        ) {
            let config = create_test_config_with_limits(32, 2048, 500);
            let mut scheduler = Scheduler::new(config);

            for i in 0..num_decode {
                let request = create_test_request_with_params(i as u64, tokens_per_request, 100);
                scheduler.add_request(request).unwrap();
            }

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

            for i in num_decode..(num_decode + num_pending) {
                let request = create_test_request_with_params(i as u64, tokens_per_request, 100);
                scheduler.add_request(request).unwrap();
            }

            let output = scheduler.schedule();

            let decode_count = scheduler.num_decode_sequences();
            if decode_count > 0 && output.num_sequences() > 0 {
                prop_assert!(
                    !output.decode_sequences.is_empty() || decode_count == 0,
                    "Decode sequences should be scheduled when available"
                );
            }
        }

        #[test]
        fn prop_prefill_to_decode_transition(
            num_requests in 1usize..10,
            tokens_per_request in 4usize..32,
        ) {
            let config = create_test_config_with_limits(16, 1024, 200);
            let mut scheduler = Scheduler::new(config);

            for i in 0..num_requests {
                let request = create_test_request_with_params(i as u64, tokens_per_request, 50);
                scheduler.add_request(request).unwrap();
            }

            let output = scheduler.schedule();
            let prefill_seq_ids: Vec<SeqId> = output.prefill_sequences.iter().map(|s| s.seq_id).collect();

            let next_tokens: Vec<u32> = prefill_seq_ids.iter().map(|_| 100).collect();
            let exec_output = ExecutionOutput {
                next_tokens,
                logits: None,
                seq_ids: prefill_seq_ids.clone(),
            };

            scheduler.update_sequences(&exec_output, 0);

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

            let output = scheduler.schedule();
            let seq_id = output.prefill_sequences[0].seq_id;

            let exec_output = ExecutionOutput {
                next_tokens: vec![100],
                logits: None,
                seq_ids: vec![seq_id],
            };
            scheduler.update_sequences(&exec_output, 0);

            let eos_token: TokenId = 0;
            let mut generated = 1u32;

            while scheduler.has_decode_sequence(seq_id) && generated < max_tokens + 5 {
                scheduler.schedule();

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

        #[test]
        fn prop_memory_pressure_response(
            num_initial_requests in 5usize..15,
            tokens_per_request in 16usize..64,
        ) {
            let config = EngineConfig {
                block_size: 16,
                max_num_blocks: 20,
                max_batch_size: 16,
                max_num_seqs: 32,
                max_model_len: 2048,
                max_total_tokens: 1024,
                memory_threshold: 0.5,
                ..Default::default()
            };
            let mut scheduler = Scheduler::new(config);

            for i in 0..num_initial_requests {
                let request = create_test_request_with_params(i as u64, tokens_per_request, 100);
                let _ = scheduler.add_request(request);
                let _ = scheduler.schedule();
            }

            let utilization = scheduler.get_memory_utilization();

            if utilization >= 0.5 {
                let new_request = create_test_request_with_params(999, tokens_per_request, 100);
                let result = scheduler.add_request(new_request);

                prop_assert!(
                    result.is_ok() || matches!(result, Err(SchedulerError::MemoryPressure)),
                    "Should handle memory pressure gracefully"
                );
            }
        }
    }
}
