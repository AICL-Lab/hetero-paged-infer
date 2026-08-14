//! 执行类型

use super::{BlockIdx, SeqId, TokenId};

/// GPU 执行批次
///
/// 包含一次 GPU 执行所需的所有数据。
#[derive(Debug, Clone, Default)]
pub struct ExecutionBatch {
    /// 所有序列的 token ID（扁平化）
    pub input_tokens: Vec<TokenId>,

    /// 每个 token 的位置 ID
    pub positions: Vec<u32>,

    /// 各序列的长度（用于 attention mask）
    pub seq_lens: Vec<u32>,

    /// Paged Attention 的块表
    pub block_tables: Vec<Vec<BlockIdx>>,

    /// Prefill/Decode 标志
    pub is_prefill: Vec<bool>,

    /// 序列 ID（用于结果映射）
    pub seq_ids: Vec<SeqId>,

    /// 各序列的上下文长度
    pub context_lens: Vec<u32>,
}

impl ExecutionBatch {
    /// 检查批次是否为空
    pub fn is_empty(&self) -> bool {
        self.seq_ids.is_empty()
    }

    /// 计算序列数
    pub fn num_sequences(&self) -> usize {
        self.seq_ids.len()
    }

    /// 计算 token 总数
    pub fn total_tokens(&self) -> usize {
        self.input_tokens.len()
    }
}

/// 单个 token 位置的 logprob 信息（OpenAI `logprobs` 语义）。
#[derive(Debug, Clone)]
pub struct TokenLogprobs {
    /// 该位置生成的 token
    pub token: TokenId,
    /// 该 token 的对数概率
    pub logprob: f32,
    /// 前 k 个候选（含选中 token）的 (token_id, logprob)，按概率降序
    pub top_logprobs: Vec<(TokenId, f32)>,
}

/// GPU 执行输出
///
/// 包含 GPU 执行的结果。
#[derive(Debug, Clone, Default)]
pub struct ExecutionOutput {
    /// 各序列的下一个 token
    pub next_tokens: Vec<TokenId>,

    /// 对应的序列 ID
    pub seq_ids: Vec<SeqId>,

    /// 各序列本步生成 token 的 logprob 信息（与 `seq_ids` 对齐）。
    /// 后端未提供时为 `None`；整个字段为空表示未启用/未计算。
    pub logprobs: Vec<Option<TokenLogprobs>>,
}
