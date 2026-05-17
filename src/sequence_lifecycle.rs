//! 序列生命周期管理器
//!
//! 管理序列从提交到完成的完整状态机，以及每个阶段对应的 KV Cache 资源分配/释放。
//!
//! # 职责
//!
//! - 序列状态转换（Pending → Prefill → Decode → Completed/Failed）
//! - KV Cache 块分配与释放
//! - 序列 ID 生成
//! - 内存压力监测
//! - GPU 执行后的序列更新（追加 token、检测完成条件）

use std::collections::{HashMap, VecDeque};

use crate::config::EngineConfig;
use crate::error::SchedulerError;
use crate::kv_cache::{KVCacheManager, KVCacheManagerTrait};
use crate::types::{
    ExecutionOutput, LogicalBlock, PhysicalBlockRef, Request, RequestState, SeqId, Sequence,
    TokenId,
};

#[derive(Debug, Clone)]
struct PendingRequest {
    seq_id: SeqId,
    request: Request,
}

/// 序列生命周期管理器
///
/// 封装序列状态机和资源管理，为调度器提供高层次的序列操作接口。
/// 所有对 `HashMap` 的序列存取、KV Cache 的分配/释放都集中在此模块。
#[derive(Debug)]
pub struct SequenceLifecycle {
    config: EngineConfig,
    kv_cache: KVCacheManager,
    pending_queue: VecDeque<PendingRequest>,
    prefill_sequences: HashMap<SeqId, Sequence>,
    decode_sequences: HashMap<SeqId, Sequence>,
    completed_requests: Vec<Request>,
    next_seq_id: SeqId,
    under_memory_pressure: bool,
}

impl SequenceLifecycle {
    /// 创建新的序列生命周期管理器
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

    // === ID 生成 ===

    fn generate_seq_id(&mut self) -> SeqId {
        let id = self.next_seq_id;
        self.next_seq_id += 1;
        id
    }

    // === 内存管理 ===

    /// 更新内存压力状态
    pub fn update_memory_pressure(&mut self) {
        let stats = self.kv_cache.get_memory_stats();
        self.under_memory_pressure = stats.utilization() >= self.config.memory_threshold;
    }

    /// 是否处于内存压力下
    pub fn is_under_memory_pressure(&self) -> bool {
        self.under_memory_pressure
    }

    /// 获取内存利用率
    pub fn memory_utilization(&self) -> f32 {
        self.kv_cache.get_memory_stats().utilization()
    }

    // === 请求接收 ===

    /// 添加新请求到待处理队列
    pub fn add_request(&mut self, request: Request) -> Result<SeqId, SchedulerError> {
        self.update_memory_pressure();

        if self.under_memory_pressure {
            return Err(SchedulerError::MemoryPressure);
        }

        if self.num_active_sequences() >= self.config.max_num_seqs as usize {
            return Err(SchedulerError::MemoryPressure);
        }

        let seq_id = self.generate_seq_id();
        self.pending_queue
            .push_back(PendingRequest { seq_id, request });
        Ok(seq_id)
    }

    // === 状态转换 ===

    /// 尝试启动 pending 请求的 prefill 阶段
    ///
    /// 成功后返回 seq_id；失败时返回原始 `(seq_id, request)` 以便重新入队。
    pub fn try_start_prefill(
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

    /// 为 decode 序列按需增长 KV Cache 块
    ///
    /// 成功返回 `Ok(())`；失败返回包含错误信息的 `Err`。
    pub fn grow_decode_sequence(&mut self, seq_id: SeqId) -> Result<(), String> {
        let Some(seq) = self.decode_sequences.get(&seq_id) else {
            debug_assert!(false, "seq_id {} expected in decode but missing", seq_id);
            return Ok(());
        };

        let current_tokens = seq.context_len();
        let current_blocks = seq.logical_blocks.len() as u32;
        let blocks_needed = self.config.blocks_for_tokens(current_tokens + 1);

        if blocks_needed > current_blocks {
            match self.kv_cache.allocate_block(seq_id) {
                Ok(physical_ref) => {
                    let Some(sequence) = self.decode_sequences.get_mut(&seq_id) else {
                        return Ok(());
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

    /// 将序列标记为失败并释放资源
    pub fn fail_sequence(&mut self, seq_id: SeqId, reason: &str) {
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

    /// 批量失败序列
    pub fn fail_sequences<I>(&mut self, seq_ids: I, reason: &str)
    where
        I: IntoIterator<Item = SeqId>,
    {
        for seq_id in seq_ids {
            self.fail_sequence(seq_id, reason);
        }
    }

    // === GPU 输出处理 ===

    /// GPU 执行后更新序列状态（追加 token、prefill→decode 转换、检测完成）
    pub fn update_sequences(&mut self, outputs: &ExecutionOutput, eos_token_id: TokenId) {
        let mut to_complete = Vec::new();
        let max_model_len = self.config.max_model_len as usize;

        for (i, &seq_id) in outputs.seq_ids.iter().enumerate() {
            let next_token = outputs.next_tokens.get(i).copied().unwrap_or(0);

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
            }
        }

        for seq_id in to_complete {
            self.complete_sequence(seq_id);
        }
    }

    // === 队列访问器（供调度策略使用） ===

    /// 获取所有 decode 序列 ID
    pub fn decode_seq_ids(&self) -> Vec<SeqId> {
        self.decode_sequences.keys().copied().collect()
    }

    /// 获取所有 prefill 序列 ID
    pub fn prefill_seq_ids(&self) -> Vec<SeqId> {
        self.prefill_sequences.keys().copied().collect()
    }

    /// 获取 decode 序列（只读）
    pub fn get_decode_sequence(&self, seq_id: SeqId) -> Option<&Sequence> {
        self.decode_sequences.get(&seq_id)
    }

    /// 获取 prefill 序列（只读）
    pub fn get_prefill_sequence(&self, seq_id: SeqId) -> Option<&Sequence> {
        self.prefill_sequences.get(&seq_id)
    }

    /// 获取 decode 序列（可变）
    pub fn get_decode_sequence_mut(&mut self, seq_id: SeqId) -> Option<&mut Sequence> {
        self.decode_sequences.get_mut(&seq_id)
    }

    /// 获取 prefill 序列（可变）
    pub fn get_prefill_sequence_mut(&mut self, seq_id: SeqId) -> Option<&mut Sequence> {
        self.prefill_sequences.get_mut(&seq_id)
    }

    /// 弹出队首 pending 请求
    pub fn pop_pending(&mut self) -> Option<(SeqId, Request)> {
        self.pending_queue
            .pop_front()
            .map(|p| (p.seq_id, p.request))
    }

    /// 将请求压回 pending 队列头部
    pub fn push_pending(&mut self, seq_id: SeqId, request: Request) {
        self.pending_queue
            .push_front(PendingRequest { seq_id, request });
    }

    /// 将已完成请求加入 completed 列表（用于调度策略中的前置校验失败）
    pub fn push_completed(&mut self, request: Request) {
        self.completed_requests.push(request);
    }

    // === 通用访问器 ===

    /// 通过 ID 查找序列（任意活跃阶段）
    pub fn get_sequence(&self, seq_id: SeqId) -> Option<&Sequence> {
        self.prefill_sequences
            .get(&seq_id)
            .or_else(|| self.decode_sequences.get(&seq_id))
    }

    /// 通过 ID 查找序列（可变）
    pub fn get_sequence_mut(&mut self, seq_id: SeqId) -> Option<&mut Sequence> {
        self.prefill_sequences
            .get_mut(&seq_id)
            .or_else(|| self.decode_sequences.get_mut(&seq_id))
    }

    /// 活跃序列数（prefill + decode）
    pub fn num_active_sequences(&self) -> usize {
        self.prefill_sequences.len() + self.decode_sequences.len()
    }

    /// decode 序列数
    pub fn num_decode_sequences(&self) -> usize {
        self.decode_sequences.len()
    }

    /// prefill 序列数
    pub fn num_prefill_sequences(&self) -> usize {
        self.prefill_sequences.len()
    }

    /// 是否存在指定 prefill 序列
    pub fn has_prefill_sequence(&self, seq_id: SeqId) -> bool {
        self.prefill_sequences.contains_key(&seq_id)
    }

    /// 是否存在指定 decode 序列
    pub fn has_decode_sequence(&self, seq_id: SeqId) -> bool {
        self.decode_sequences.contains_key(&seq_id)
    }

    /// pending 队列中是否存在指定 seq_id
    pub fn has_pending_request(&self, seq_id: SeqId) -> bool {
        self.pending_queue.iter().any(|p| p.seq_id == seq_id)
    }

    /// 序列是否恰好在其中一个队列中
    pub fn is_in_exactly_one_queue(&self, seq_id: SeqId) -> bool {
        let in_prefill = self.prefill_sequences.contains_key(&seq_id);
        let in_decode = self.decode_sequences.contains_key(&seq_id);
        (in_prefill && !in_decode) || (!in_prefill && in_decode)
    }

    /// 是否还有待处理的工作
    pub fn has_pending_work(&self) -> bool {
        !self.pending_queue.is_empty()
            || !self.prefill_sequences.is_empty()
            || !self.decode_sequences.is_empty()
    }

    /// 取走所有已完成请求
    pub fn get_completed(&mut self) -> Vec<Request> {
        std::mem::take(&mut self.completed_requests)
    }
}
