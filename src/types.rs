//! 核心类型和数据结构
//!
//! 本模块定义了推理系统的核心数据结构，包括：
//!
//! - 请求和序列表示
//! - 生成参数
//! - 执行批次
//! - 内存统计
//!
//! # 类型概览
//!
//! | 类型 | 说明 |
//! |------|------|
//! | [`Request`] | 推理请求 |
//! | [`Sequence`] | 活跃序列（含 KV Cache） |
//! | [`GenerationParams`] | 生成参数 |
//! | [`ExecutionBatch`] | GPU 执行批次 |
//! | [`ExecutionOutput`] | GPU 执行输出 |
//! | [`CompletedRequest`] | 完成的请求 |
//! | [`MemoryStats`] | 内存统计 |

use std::sync::Arc;
use std::time::Instant;

/// 请求唯一标识符
pub type RequestId = u64;

/// 序列唯一标识符
pub type SeqId = u64;

/// Token ID 类型
pub type TokenId = u32;

/// 物理块索引
pub type BlockIdx = u32;

/// 请求状态
///
/// 表示请求在推理流水线中的当前状态。
///
/// # 状态转换
///
/// ```text
/// Pending → Prefill → Decode → Completed
///                     ↘ Failed
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum RequestState {
    /// 等待调度
    Pending,

    /// Prefill 阶段（处理输入 tokens）
    Prefill,

    /// Decode 阶段（生成 tokens）
    Decode,

    /// 成功完成
    Completed,

    /// 失败，包含错误信息
    Failed(String),
}

impl RequestState {
    /// 检查请求是否处于活跃状态（Prefill 或 Decode）
    ///
    /// # 示例
    ///
    /// ```rust
    /// use hetero_infer::RequestState;
    ///
    /// assert!(!RequestState::Pending.is_active());
    /// assert!(RequestState::Prefill.is_active());
    /// assert!(RequestState::Decode.is_active());
    /// assert!(!RequestState::Completed.is_active());
    /// ```
    pub fn is_active(&self) -> bool {
        matches!(self, RequestState::Prefill | RequestState::Decode)
    }

    /// 检查请求是否处于终态（Completed 或 Failed）
    ///
    /// # 示例
    ///
    /// ```rust
    /// use hetero_infer::RequestState;
    ///
    /// assert!(!RequestState::Decode.is_terminal());
    /// assert!(RequestState::Completed.is_terminal());
    /// assert!(RequestState::Failed("error".to_string()).is_terminal());
    /// ```
    pub fn is_terminal(&self) -> bool {
        matches!(self, RequestState::Completed | RequestState::Failed(_))
    }
}

/// 生成参数
///
/// 控制文本生成的采样参数。
///
/// # 参数范围
///
/// | 参数 | 有效范围 |
/// |------|----------|
/// | `max_tokens` | > 0 |
/// | `temperature` | (0.0, 2.0] |
/// | `top_p` | (0.0, 1.0] |
///
/// # 示例
///
/// ```rust
/// use hetero_infer::GenerationParams;
///
/// // 使用默认参数
/// let params = GenerationParams::default();
///
/// // 自定义参数
/// let params = GenerationParams {
///     max_tokens: 100,
///     temperature: 0.8,
///     top_p: 0.95,
/// };
///
/// assert!(params.validate().is_ok());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GenerationParams {
    /// 最大生成 token 数
    pub max_tokens: u32,

    /// 采样温度 (0.0, 2.0]
    ///
    /// - 较低的值（如 0.1）产生更确定的输出
    /// - 较高的值（如 1.5）产生更多样化的输出
    pub temperature: f32,

    /// Top-p（核采样）参数 (0.0, 1.0]
    ///
    /// 从累积概率达到 top_p 的最小 token 集合中采样
    pub top_p: f32,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 100,
            temperature: 1.0,
            top_p: 1.0,
        }
    }
}

impl GenerationParams {
    /// 验证生成参数
    ///
    /// # 错误
    ///
    /// - [`crate::ValidationError::InvalidMaxTokens`] `max_tokens` 为 0
    /// - [`crate::ValidationError::InvalidTemperature`] `temperature` 不在有效范围内
    /// - [`crate::ValidationError::InvalidTopP`] `top_p` 不在有效范围内
    ///
    /// # 示例
    ///
    /// ```rust
    /// use hetero_infer::GenerationParams;
    ///
    /// let valid = GenerationParams {
    ///     max_tokens: 100,
    ///     temperature: 1.0,
    ///     top_p: 0.9,
    /// };
    /// assert!(valid.validate().is_ok());
    ///
    /// let invalid = GenerationParams {
    ///     max_tokens: 0,
    ///     temperature: 1.0,
    ///     top_p: 0.9,
    /// };
    /// assert!(invalid.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), crate::ValidationError> {
        if self.max_tokens == 0 {
            return Err(crate::ValidationError::InvalidMaxTokens(self.max_tokens));
        }
        if self.temperature <= 0.0 || self.temperature > 2.0 {
            return Err(crate::ValidationError::InvalidTemperature(self.temperature));
        }
        if self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(crate::ValidationError::InvalidTopP(self.top_p));
        }
        Ok(())
    }

    /// 检查参数是否有效
    ///
    /// # 示例
    ///
    /// ```rust
    /// use hetero_infer::GenerationParams;
    ///
    /// let params = GenerationParams {
    ///     max_tokens: 100,
    ///     temperature: 1.0,
    ///     top_p: 0.9,
    /// };
    /// assert!(params.is_valid());
    /// ```
    pub fn is_valid(&self) -> bool {
        self.max_tokens > 0
            && self.temperature > 0.0
            && self.temperature <= 2.0
            && self.top_p > 0.0
            && self.top_p <= 1.0
    }
}

/// 推理请求
///
/// 表示单个推理请求，包含输入 tokens 和生成参数。
///
/// # 生命周期
///
/// 1. 通过 [`Request::new`] 创建，状态为 `Pending`
/// 2. 调度器处理后进入 `Prefill` 状态
/// 3. Prefill 完成后进入 `Decode` 状态
/// 4. 生成完成后进入 `Completed` 状态
///
/// # 示例
///
/// ```rust
/// use hetero_infer::{Request, GenerationParams};
///
/// let request = Request::new(
///     1,                          // request_id
///     vec![1, 2, 3, 4, 5],        // input_tokens
///     GenerationParams::default(),
/// );
///
/// assert_eq!(request.id, 1);
/// assert_eq!(request.input_tokens.len(), 5);
/// assert!(request.output_tokens.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct Request {
    /// 请求唯一标识符
    pub id: RequestId,

    /// 输入 tokens（分词后）
    pub input_tokens: Vec<TokenId>,

    /// 生成的输出 tokens
    pub output_tokens: Vec<TokenId>,

    /// 生成参数
    pub params: GenerationParams,

    /// 当前状态
    pub state: RequestState,

    /// 创建时间戳
    pub created_at: Instant,
}

impl Request {
    /// 创建新请求
    ///
    /// # 参数
    ///
    /// * `id` - 请求唯一标识符
    /// * `input_tokens` - 输入 token 序列
    /// * `params` - 生成参数
    ///
    /// # 示例
    ///
    /// ```rust
    /// use hetero_infer::{Request, GenerationParams};
    ///
    /// let request = Request::new(42, vec![1, 2, 3], GenerationParams::default());
    /// assert_eq!(request.id, 42);
    /// ```
    pub fn new(id: RequestId, input_tokens: Vec<TokenId>, params: GenerationParams) -> Self {
        Self {
            id,
            input_tokens,
            output_tokens: Vec::new(),
            params,
            state: RequestState::Pending,
            created_at: Instant::now(),
        }
    }

    /// 计算总 token 数（输入 + 输出）
    ///
    /// # 示例
    ///
    /// ```rust
    /// use hetero_infer::{Request, GenerationParams};
    ///
    /// let mut request = Request::new(1, vec![1, 2, 3], GenerationParams::default());
    /// assert_eq!(request.total_tokens(), 3);
    ///
    /// request.output_tokens.push(4);
    /// assert_eq!(request.total_tokens(), 4);
    /// ```
    pub fn total_tokens(&self) -> usize {
        self.input_tokens.len() + self.output_tokens.len()
    }

    /// 检查生成是否完成
    ///
    /// 完成条件：
    /// - 输出 token 数达到 `max_tokens`
    /// - 生成了 EOS token
    ///
    /// # 参数
    ///
    /// * `eos_token_id` - EOS token 的 ID
    ///
    /// # 示例
    ///
    /// ```rust
    /// use hetero_infer::{Request, GenerationParams};
    ///
    /// let mut request = Request::new(
    ///     1,
    ///     vec![1, 2, 3],
    ///     GenerationParams { max_tokens: 5, ..Default::default() },
    /// );
    ///
    /// let eos_token = 0;
    /// assert!(!request.is_complete(eos_token));
    ///
    /// // 达到 max_tokens
    /// request.output_tokens = vec![10, 11, 12, 13, 14];
    /// assert!(request.is_complete(eos_token));
    /// ```
    pub fn is_complete(&self, eos_token_id: TokenId) -> bool {
        // 达到 max_tokens
        if self.output_tokens.len() >= self.params.max_tokens as usize {
            return true;
        }
        // 生成了 EOS token
        if let Some(&last_token) = self.output_tokens.last() {
            if last_token == eos_token_id {
                return true;
            }
        }
        false
    }
}

/// 物理块引用
///
/// 表示对 GPU 显存中物理块的引用。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysicalBlockRef {
    /// 物理块索引
    pub block_idx: BlockIdx,
}

/// 逻辑块
///
/// 表示映射到物理块的逻辑块。
#[derive(Debug, Clone)]
pub struct LogicalBlock {
    /// 序列内的逻辑块索引
    pub block_idx: u32,

    /// 映射的物理块（未分配时为 None）
    pub physical_block: Option<PhysicalBlockRef>,
}

impl LogicalBlock {
    /// 创建未映射的逻辑块
    pub fn new(block_idx: u32) -> Self {
        Self {
            block_idx,
            physical_block: None,
        }
    }

    /// 创建已映射的逻辑块
    pub fn with_physical(block_idx: u32, physical: PhysicalBlockRef) -> Self {
        Self {
            block_idx,
            physical_block: Some(physical),
        }
    }
}

/// 序列
///
/// 活跃请求及其 KV Cache 块的集合。
///
/// 序列是调度和执行的基本单位，包含：
/// - 原始请求
/// - 逻辑块到物理块的映射
/// - 计算进度追踪
#[derive(Debug, Clone)]
pub struct Sequence {
    /// 序列唯一标识符
    pub seq_id: SeqId,

    /// 关联的请求
    pub request: Request,

    /// KV Cache 的逻辑块
    pub logical_blocks: Vec<LogicalBlock>,

    /// 已计算的 token 数（已缓存在 KV Cache 中）
    pub num_computed_tokens: u32,

    /// 已生成的 token 数
    pub num_generated_tokens: u32,
}

impl Sequence {
    /// 从请求创建序列
    ///
    /// # 参数
    ///
    /// * `seq_id` - 序列唯一标识符
    /// * `request` - 关联的请求
    pub fn new(seq_id: SeqId, request: Request) -> Self {
        Self {
            seq_id,
            request,
            logical_blocks: Vec::new(),
            num_computed_tokens: 0,
            num_generated_tokens: 0,
        }
    }

    /// 获取块表（物理块索引列表）用于 GPU 执行
    pub fn get_block_table(&self) -> Vec<BlockIdx> {
        self.logical_blocks
            .iter()
            .filter_map(|lb| lb.physical_block.map(|pb| pb.block_idx))
            .collect()
    }

    /// 计算上下文长度（输入 + 已生成）
    pub fn context_len(&self) -> u32 {
        self.request.input_tokens.len() as u32 + self.num_generated_tokens
    }

    /// 计算当前步骤需要处理的 token 数
    pub fn num_tokens_to_process(&self) -> u32 {
        match self.request.state {
            RequestState::Prefill => self.request.input_tokens.len() as u32,
            RequestState::Decode => 1,
            _ => 0,
        }
    }

    /// 获取 decode 阶段的输入 token
    ///
    /// 返回最后一个生成的 token，如果没有则返回输入的最后一个 token。
    pub fn decode_input_token(&self) -> Option<TokenId> {
        self.request
            .output_tokens
            .last()
            .copied()
            .or_else(|| self.request.input_tokens.last().copied())
    }

    /// 获取 decode 阶段的位置
    ///
    /// 返回下一个 token 的位置索引。
    pub fn decode_position(&self) -> Option<u32> {
        self.context_len().checked_sub(1)
    }
}

/// KV Cache 内存统计
///
/// 提供内存使用情况的快照。
#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryStats {
    /// 物理块总数
    pub total_blocks: u32,

    /// 已使用的物理块数
    pub used_blocks: u32,

    /// 空闲物理块数
    pub free_blocks: u32,

    /// 活跃序列数
    pub num_sequences: u32,
}

impl MemoryStats {
    /// 计算内存利用率
    ///
    /// # 示例
    ///
    /// ```rust
    /// use hetero_infer::MemoryStats;
    ///
    /// let stats = MemoryStats {
    ///     total_blocks: 100,
    ///     used_blocks: 25,
    ///     free_blocks: 75,
    ///     num_sequences: 5,
    /// };
    /// assert!((stats.utilization() - 0.25).abs() < 0.001);
    /// ```
    pub fn utilization(&self) -> f32 {
        if self.total_blocks == 0 {
            0.0
        } else {
            self.used_blocks as f32 / self.total_blocks as f32
        }
    }
}

/// 调度器输出
///
/// 包含一个调度周期内待执行的序列。
#[derive(Debug, Clone, Default)]
pub struct SchedulerOutput {
    /// Prefill 阶段的序列
    pub prefill_sequences: Vec<Arc<Sequence>>,

    /// Decode 阶段的序列
    pub decode_sequences: Vec<Arc<Sequence>>,

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
}

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

/// GPU 执行输出
///
/// 包含 GPU 执行的结果。
#[derive(Debug, Clone, Default)]
pub struct ExecutionOutput {
    /// 各序列的下一个 token
    pub next_tokens: Vec<TokenId>,

    /// Logits（可选，用于采样）
    pub logits: Option<Vec<f32>>,

    /// 对应的序列 ID
    pub seq_ids: Vec<SeqId>,
}

/// 完成的请求
///
/// 包含已完成请求的输出结果。
#[derive(Debug, Clone)]
pub struct CompletedRequest {
    /// 原始请求 ID
    pub request_id: RequestId,

    /// 输入文本（可选）
    pub input_text: Option<String>,

    /// 生成的输出文本
    pub output_text: String,

    /// 生成的 tokens
    pub output_tokens: Vec<TokenId>,

    /// 是否成功完成
    pub success: bool,

    /// 错误信息（失败时）
    pub error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_state_is_active() {
        assert!(!RequestState::Pending.is_active());
        assert!(RequestState::Prefill.is_active());
        assert!(RequestState::Decode.is_active());
        assert!(!RequestState::Completed.is_active());
        assert!(!RequestState::Failed("error".to_string()).is_active());
    }

    #[test]
    fn test_generation_params_validation() {
        let valid = GenerationParams {
            max_tokens: 100,
            temperature: 1.0,
            top_p: 0.9,
        };
        assert!(valid.validate().is_ok());
        assert!(valid.is_valid());

        let invalid_max_tokens = GenerationParams {
            max_tokens: 0,
            temperature: 1.0,
            top_p: 0.9,
        };
        assert!(invalid_max_tokens.validate().is_err());
        assert!(!invalid_max_tokens.is_valid());

        let invalid_temp = GenerationParams {
            max_tokens: 100,
            temperature: 0.0,
            top_p: 0.9,
        };
        assert!(invalid_temp.validate().is_err());

        let invalid_top_p = GenerationParams {
            max_tokens: 100,
            temperature: 1.0,
            top_p: 1.5,
        };
        assert!(invalid_top_p.validate().is_err());
    }

    #[test]
    fn test_request_is_complete() {
        let mut request = Request::new(
            1,
            vec![1, 2, 3],
            GenerationParams {
                max_tokens: 5,
                temperature: 1.0,
                top_p: 1.0,
            },
        );

        let eos_token = 0;

        // 初始未完成
        assert!(!request.is_complete(eos_token));

        // 添加一些 tokens
        request.output_tokens = vec![10, 11, 12];
        assert!(!request.is_complete(eos_token));

        // 添加 EOS token
        request.output_tokens.push(eos_token);
        assert!(request.is_complete(eos_token));

        // 达到 max_tokens
        request.output_tokens = vec![10, 11, 12, 13, 14];
        assert!(request.is_complete(eos_token));
    }

    #[test]
    fn test_memory_stats_utilization() {
        let stats = MemoryStats {
            total_blocks: 100,
            used_blocks: 25,
            free_blocks: 75,
            num_sequences: 5,
        };
        assert!((stats.utilization() - 0.25).abs() < 0.001);

        let empty = MemoryStats::default();
        assert_eq!(empty.utilization(), 0.0);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// **Feature: heterogeneous-inference-system, Property 2: Parameter Validation Correctness**
        /// *对于任意* 生成参数 (max_tokens, temperature, top_p)，验证函数返回 true 当且仅当
        /// 所有参数都在有效范围内 (max_tokens > 0, 0 < temperature <= 2.0, 0 < top_p <= 1.0)。
        /// **验证: Requirements 1.3**
        #[test]
        fn prop_parameter_validation(
            max_tokens in 0u32..1000,
            temperature in -1.0f32..3.0,
            top_p in -0.5f32..1.5,
        ) {
            let params = GenerationParams {
                max_tokens,
                temperature,
                top_p,
            };

            let validation_result = params.validate();
            let is_valid = params.is_valid();

            // 基于参数范围的预期有效性
            let expected_valid = max_tokens > 0
                && temperature > 0.0
                && temperature <= 2.0
                && top_p > 0.0
                && top_p <= 1.0;

            // 属性: 验证结果与预期有效性一致
            prop_assert_eq!(
                validation_result.is_ok(),
                expected_valid,
                "验证不匹配，参数: max_tokens={}, temp={}, top_p={}",
                max_tokens, temperature, top_p
            );

            // 属性: is_valid() 与 validate() 一致
            prop_assert_eq!(
                is_valid,
                expected_valid,
                "is_valid() 与 validate() 不一致，参数: {:?}",
                params
            );
        }

        /// 边界条件属性测试
        #[test]
        fn prop_parameter_boundaries(
            valid_max_tokens in 1u32..1000,
        ) {
            // 边界测试: temperature = 2.0 应有效
            let params_temp_boundary = GenerationParams {
                max_tokens: valid_max_tokens,
                temperature: 2.0,
                top_p: 0.5,
            };
            prop_assert!(params_temp_boundary.is_valid());

            // 边界测试: top_p = 1.0 应有效
            let params_top_p_boundary = GenerationParams {
                max_tokens: valid_max_tokens,
                temperature: 1.0,
                top_p: 1.0,
            };
            prop_assert!(params_top_p_boundary.is_valid());

            // 边界测试: temperature > 2.0 应无效
            let params_temp_over = GenerationParams {
                max_tokens: valid_max_tokens,
                temperature: 2.001,
                top_p: 0.5,
            };
            prop_assert!(!params_temp_over.is_valid());

            // 边界测试: top_p > 1.0 应无效
            let params_top_p_over = GenerationParams {
                max_tokens: valid_max_tokens,
                temperature: 1.0,
                top_p: 1.001,
            };
            prop_assert!(!params_top_p_over.is_valid());

            // 零值边界测试
            let params_zero_temp = GenerationParams {
                max_tokens: valid_max_tokens,
                temperature: 0.0,
                top_p: 0.5,
            };
            prop_assert!(!params_zero_temp.is_valid());

            let params_zero_top_p = GenerationParams {
                max_tokens: valid_max_tokens,
                temperature: 1.0,
                top_p: 0.0,
            };
            prop_assert!(!params_zero_top_p.is_valid());

            let params_zero_max_tokens = GenerationParams {
                max_tokens: 0,
                temperature: 1.0,
                top_p: 0.5,
            };
            prop_assert!(!params_zero_max_tokens.is_valid());
        }
    }
}
