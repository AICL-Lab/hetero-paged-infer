//! 调度器类型

use super::memory::PhysicalBlockRef;
use super::request::Request;
use super::{BlockIdx, SeqId, TokenId};

/// 序列
///
/// 活跃请求及其 KV Cache 块的集合。
#[derive(Debug, Clone)]
pub struct Sequence {
    /// 序列唯一标识符
    pub seq_id: SeqId,

    /// 关联的请求
    pub request: Request,

    /// KV Cache 块（按逻辑顺序排列的物理块引用）
    pub logical_blocks: Vec<PhysicalBlockRef>,

    /// 已生成的 token 数
    pub num_generated_tokens: u32,
}

impl Sequence {
    /// 从请求创建序列
    pub fn new(seq_id: SeqId, request: Request) -> Self {
        Self {
            seq_id,
            request,
            logical_blocks: Vec::new(),
            num_generated_tokens: 0,
        }
    }

    /// 获取块表（物理块索引列表）用于 GPU 执行
    pub fn get_block_table(&self) -> Vec<BlockIdx> {
        self.logical_blocks.iter().map(|b| b.block_idx).collect()
    }

    /// 计算上下文长度（输入 + 已生成）
    ///
    /// 使用饱和算术，任何输入下都不会溢出。
    pub fn context_len(&self) -> u32 {
        let input_len = u32::try_from(self.request.input_tokens.len()).unwrap_or(u32::MAX);
        input_len.saturating_add(self.num_generated_tokens)
    }

    /// 获取 decode 阶段的输入 token
    pub fn decode_input_token(&self) -> Option<TokenId> {
        self.request
            .output_tokens
            .last()
            .copied()
            .or_else(|| self.request.input_tokens.last().copied())
    }

    /// 获取 decode 阶段的位置
    pub fn decode_position(&self) -> Option<u32> {
        self.context_len().checked_sub(1)
    }
}

/// 调度器输出
///
/// 包含一个调度周期内待执行的序列。序列以克隆快照形式持有：
/// 输出仅在单线程内被执行流水线读取一次后即丢弃，无需共享所有权。
#[derive(Debug, Clone, Default)]
pub struct SchedulerOutput {
    /// Prefill 阶段的序列
    pub prefill_sequences: Vec<Sequence>,

    /// Decode 阶段的序列
    pub decode_sequences: Vec<Sequence>,

    /// 批次总 token 数
    pub total_tokens: u32,
}

impl SchedulerOutput {
    /// 检查输出是否为空
    pub fn is_empty(&self) -> bool {
        self.prefill_sequences.is_empty() && self.decode_sequences.is_empty()
    }

    /// 计算序列总数
    pub fn num_sequences(&self) -> usize {
        self.prefill_sequences.len() + self.decode_sequences.len()
    }

    /// 获取所有序列 ID（prefill 和 decode 合并）
    pub fn seq_ids(&self) -> Vec<SeqId> {
        self.prefill_sequences
            .iter()
            .chain(self.decode_sequences.iter())
            .map(|s| s.seq_id)
            .collect()
    }
}
