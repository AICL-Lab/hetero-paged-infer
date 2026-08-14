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

use std::collections::{BTreeMap, VecDeque};

use crate::config::EngineConfig;
use crate::error::EngineError;
use crate::kv_cache::KVCacheManager;
use crate::types::{
    ExecutionOutput, PhysicalBlockRef, Request, RequestId, RequestState, SchedulerOutput, SeqId,
    Sequence, TokenId,
};

/// 取消请求的失败原因前缀；引擎据此把终态归类为"客户端取消"而非后端错误。
pub const CANCEL_REASON_PREFIX: &str = "request cancelled";

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
    // BTreeMap（而非 HashMap）使调度迭代顺序按 seq_id 单调递增，
    // 即先提交先调度（FCFS），保证公平性并让调度行为可复现。
    prefill_sequences: BTreeMap<SeqId, Sequence>,
    decode_sequences: BTreeMap<SeqId, Sequence>,
    completed_requests: Vec<Request>,
    next_seq_id: SeqId,
    under_memory_pressure: bool,
}

impl Scheduler {
    /// 创建调度器
    ///
    /// # Panics
    ///
    /// 要求 `config.block_size > 0`（否则块数计算会除零 panic）。
    /// 经由 `InferenceEngine` 构造时 `EngineConfig::validate` 已保证这一点；
    /// 直接构造底层调度器的调用方需自行确保配置有效。
    pub fn new(config: EngineConfig) -> Self {
        debug_assert!(
            config.block_size > 0,
            "config.block_size must be greater than 0"
        );
        let kv_cache = KVCacheManager::new(config.max_num_blocks, config.block_size);
        Self {
            config,
            kv_cache,
            pending_queue: VecDeque::new(),
            prefill_sequences: BTreeMap::new(),
            decode_sequences: BTreeMap::new(),
            completed_requests: Vec::new(),
            next_seq_id: 1,
            under_memory_pressure: false,
        }
    }

    // === 公共接口 ===

    pub fn add_request(&mut self, request: Request) -> Result<SeqId, EngineError> {
        self.update_memory_pressure();

        if self.under_memory_pressure {
            return Err(EngineError::MemoryPressure);
        }

        let total_sequences = self.pending_queue.len() + self.num_active_sequences();
        if total_sequences >= self.config.max_num_seqs as usize {
            return Err(EngineError::MaxConcurrentSequencesReached(
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

            if total_tokens.saturating_add(1) > self.config.max_total_tokens {
                break;
            }

            if let Err(reason) = self.grow_decode_sequence(seq_id) {
                self.fail_sequence(seq_id, &reason);
                continue;
            }

            if let Some(sequence) = self.decode_sequences.get(&seq_id) {
                output.decode_sequences.push(sequence.clone());
                total_tokens = total_tokens.saturating_add(1);
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

            // continue 而非 break：装不下本步的大 prefill 不应阻塞后面的小 prefill
            if total_tokens.saturating_add(prefill_tokens) > self.config.max_total_tokens {
                continue;
            }

            if let Some(sequence) = self.prefill_sequences.get(&seq_id) {
                output.prefill_sequences.push(sequence.clone());
                total_tokens = total_tokens.saturating_add(prefill_tokens);
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

                if total_tokens.saturating_add(prefill_tokens) > self.config.max_total_tokens {
                    self.pending_queue
                        .push_front(PendingRequest { seq_id, request });
                    break;
                }

                match self.try_start_prefill(seq_id, request) {
                    Ok(seq_id) => {
                        if let Some(sequence) = self.prefill_sequences.get(&seq_id) {
                            output.prefill_sequences.push(sequence.clone());
                            total_tokens = total_tokens.saturating_add(prefill_tokens);
                            num_sequences += 1;
                        }
                    }
                    Err((seq_id, request)) => {
                        self.pending_queue.push_front(PendingRequest {
                            seq_id,
                            request: *request,
                        });
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

    /// 按 request_id 只读查询请求（pending / prefill / decode）。
    pub fn get_request_by_id(&self, request_id: RequestId) -> Option<&Request> {
        self.pending_queue
            .iter()
            .find(|p| p.request.id == request_id)
            .map(|p| &p.request)
            .or_else(|| {
                self.prefill_sequences
                    .values()
                    .chain(self.decode_sequences.values())
                    .find(|seq| seq.request.id == request_id)
                    .map(|seq| &seq.request)
            })
    }

    /// 因命中 stop 序列而终止请求（PINF-105）：
    /// 把输出 token 截断到 stop 序列之前、标记 `stop_sequence_hit` 并完成。
    /// 仅可能命中已开始生成的 prefill/decode 请求（pending 无输出 token）。
    /// 返回是否找到了对应请求。
    pub fn complete_by_stop_sequence(&mut self, request_id: RequestId, keep_tokens: usize) -> bool {
        let seq_id = self
            .prefill_sequences
            .values()
            .chain(self.decode_sequences.values())
            .find(|seq| seq.request.id == request_id)
            .map(|seq| seq.seq_id);
        match seq_id {
            Some(seq_id) => {
                if let Some(seq) = self
                    .prefill_sequences
                    .get_mut(&seq_id)
                    .or_else(|| self.decode_sequences.get_mut(&seq_id))
                {
                    seq.request.output_tokens.truncate(keep_tokens);
                    seq.request.stop_sequence_hit = true;
                }
                self.complete_sequence(seq_id);
                true
            }
            None => false,
        }
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

    #[cfg(test)]
    fn get_sequence(&self, seq_id: SeqId) -> Option<&Sequence> {
        self.prefill_sequences
            .get(&seq_id)
            .or_else(|| self.decode_sequences.get(&seq_id))
    }

    pub fn num_active_sequences(&self) -> usize {
        self.prefill_sequences.len() + self.decode_sequences.len()
    }

    /// 按 request_id 取消请求：无论它处于 pending、prefill 还是 decode，
    /// 都将其标记失败并释放 KV 资源。返回是否真的取消到了东西。
    pub fn cancel_by_request_id(&mut self, request_id: RequestId) -> bool {
        self.fail_by_request_id(
            request_id,
            &format!("{CANCEL_REASON_PREFIX}: client disconnected"),
        )
    }

    /// 按 request_id 将请求标记为失败（释放 KV 资源），失败原因随请求排出。
    /// 返回是否找到了对应请求。
    pub fn fail_by_request_id(&mut self, request_id: RequestId, reason: &str) -> bool {
        let seq_id = self
            .pending_queue
            .iter()
            .find(|p| p.request.id == request_id)
            .map(|p| p.seq_id)
            .or_else(|| {
                self.prefill_sequences
                    .values()
                    .chain(self.decode_sequences.values())
                    .find(|seq| seq.request.id == request_id)
                    .map(|seq| seq.seq_id)
            });
        match seq_id {
            Some(seq_id) => {
                self.fail_sequence(seq_id, reason);
                true
            }
            None => false,
        }
    }

    #[cfg(test)]
    fn is_in_exactly_one_queue(&self, seq_id: SeqId) -> bool {
        let in_pending = self.pending_queue.iter().any(|p| p.seq_id == seq_id);
        let in_prefill = self.prefill_sequences.contains_key(&seq_id);
        let in_decode = self.decode_sequences.contains_key(&seq_id);
        let count = [in_pending, in_prefill, in_decode]
            .iter()
            .filter(|&&flag| flag)
            .count();
        count == 1
    }

    #[cfg(test)]
    fn has_prefill_sequence(&self, seq_id: SeqId) -> bool {
        self.prefill_sequences.contains_key(&seq_id)
    }

    #[cfg(test)]
    fn has_decode_sequence(&self, seq_id: SeqId) -> bool {
        self.decode_sequences.contains_key(&seq_id)
    }

    #[cfg(test)]
    fn has_pending_request(&self, seq_id: SeqId) -> bool {
        self.pending_queue.iter().any(|p| p.seq_id == seq_id)
    }

    #[cfg(test)]
    fn num_decode_sequences(&self) -> usize {
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
    ) -> Result<SeqId, (SeqId, Box<Request>)> {
        let num_tokens = request.input_tokens.len() as u32;
        let blocks_needed = self.config.blocks_for_tokens(num_tokens);

        if !self.kv_cache.can_allocate(blocks_needed) {
            return Err((seq_id, Box::new(request)));
        }

        if self.kv_cache.allocate_sequence(seq_id, num_tokens).is_err() {
            return Err((seq_id, Box::new(request)));
        }

        let mut sequence = Sequence::new(seq_id, request);
        sequence.request.state = RequestState::Prefill;

        if let Some(block_table) = self.kv_cache.get_block_table(seq_id) {
            sequence.logical_blocks = block_table
                .iter()
                .map(|&block_idx| PhysicalBlockRef { block_idx })
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
        let blocks_needed = self
            .config
            .blocks_for_tokens(current_tokens.saturating_add(1));

        if blocks_needed > current_blocks {
            match self.kv_cache.allocate_block(seq_id) {
                Ok(physical_ref) => {
                    let Some(sequence) = self.decode_sequences.get_mut(&seq_id) else {
                        return Err(format!("Decode sequence not found: {seq_id}"));
                    };
                    sequence.logical_blocks.push(physical_ref);
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
    use crate::test_utils::{
        create_test_config, create_test_request, create_test_request_with_params,
    };

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
            Err(EngineError::MaxConcurrentSequencesReached(1))
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
            seq_ids: vec![seq_id],
        };
        scheduler.update_sequences(&exec_output, 0);

        scheduler.schedule();
        let exec_output = ExecutionOutput {
            next_tokens: vec![101],
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
            Err(EngineError::MaxConcurrentSequencesReached(1))
        ));
    }

    #[test]
    fn test_cancel_by_request_id_covers_all_stages_and_frees_kv() {
        let mut scheduler = Scheduler::new(create_test_config());

        // request 1 留在 pending；request 2 推进到 decode
        let pending_seq = scheduler
            .add_request(create_test_request_with_params(1, 8, 10))
            .unwrap();
        let active_seq = scheduler
            .add_request(create_test_request_with_params(2, 8, 10))
            .unwrap();

        let output = scheduler.schedule();
        assert!(output
            .prefill_sequences
            .iter()
            .any(|s| s.seq_id == active_seq));
        scheduler.update_sequences(
            &ExecutionOutput {
                next_tokens: vec![100],
                seq_ids: vec![active_seq],
            },
            0,
        );
        assert!(scheduler.has_decode_sequence(active_seq));
        assert!(scheduler.get_memory_utilization() > 0.0);

        // 取消 pending 请求
        assert!(scheduler.cancel_by_request_id(1));
        assert!(!scheduler.has_pending_request(pending_seq));

        // 取消 decode 请求：KV 必须释放
        assert!(scheduler.cancel_by_request_id(2));
        assert!(!scheduler.has_decode_sequence(active_seq));
        assert_eq!(scheduler.get_memory_utilization(), 0.0);

        // 两个被取消的请求都应经正常完成通道以失败终态排出
        let completed = scheduler.get_completed();
        assert_eq!(completed.len(), 2);
        assert!(completed.iter().all(|r| matches!(
            &r.state,
            RequestState::Failed(msg) if msg.contains("cancelled")
        )));

        // 未知 request_id 返回 false
        assert!(!scheduler.cancel_by_request_id(999));
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
        scheduler
            .try_start_prefill(pending.seq_id, pending.request)
            .unwrap();
        scheduler.transition_prefill_to_decode(seq_id);

        scheduler.update_sequences(
            &ExecutionOutput {
                next_tokens: Vec::new(),
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

            // decode 优先且批次上限（32）远大于 decode 数量（<10）：
            // 每个在途 decode 序列都必须被调度，一个都不能落下。
            let decode_count = scheduler.num_decode_sequences();
            prop_assert_eq!(
                output.decode_sequences.len(),
                decode_count,
                "every in-flight decode sequence must be scheduled each step"
            );
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

                // 利用率达到阈值后，add_request 必须确定性地拒绝（MemoryPressure），
                // 而不是"可能拒绝也可能接受"。
                prop_assert!(
                    matches!(result, Err(EngineError::MemoryPressure)),
                    "utilization {} >= threshold 0.5 must reject with MemoryPressure, got {:?}",
                    utilization,
                    result
                );
            }
        }
    }
}
