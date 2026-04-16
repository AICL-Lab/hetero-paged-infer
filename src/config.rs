//! 引擎配置类型与验证
//!
//! 本模块提供推理引擎的配置管理，包括：
//! - 配置参数定义
//! - 参数验证
//! - JSON 序列化/反序列化
//! - 文件加载/保存
//!
//! # 示例
//!
//! ```rust
//! use hetero_infer::EngineConfig;
//! use std::path::Path;
//!
//! // 使用默认配置
//! let config = EngineConfig::default();
//!
//! // 创建自定义配置
//! let config = EngineConfig::new(
//!     16,    // block_size
//!     1024,  // max_num_blocks
//!     32,    // max_batch_size
//!     256,   // max_num_seqs
//!     2048,  // max_model_len
//!     4096,  // max_total_tokens
//!     0.9,   // memory_threshold
//! )?;
//!
//! // 验证配置
//! config.validate()?;
//! # Ok::<(), hetero_infer::ConfigError>(())
//! ```

use crate::error::ConfigError;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// 特殊 Token ID 配置
///
/// 定义模型使用的特殊 token ID。
/// 不同模型可能有不同的特殊 token ID，需要根据实际模型配置。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpecialTokenIds {
    /// 句首 token ID (Beginning of Sequence)
    pub bos: u32,
    /// 句尾 token ID (End of Sequence)
    pub eos: u32,
    /// 填充 token ID (Padding)
    pub pad: u32,
    /// 未知 token ID (Unknown)
    pub unk: u32,
}

impl Default for SpecialTokenIds {
    fn default() -> Self {
        Self {
            bos: 1,
            eos: 2,
            pad: 0,
            unk: 3,
        }
    }
}

impl SpecialTokenIds {
    /// 创建新的特殊 Token ID 配置
    pub fn new(bos: u32, eos: u32, pad: u32, unk: u32) -> Self {
        Self { bos, eos, pad, unk }
    }
}

/// 引擎配置
///
/// 包含推理引擎的所有可配置参数。
///
/// # 参数说明
///
/// | 参数 | 说明 | 建议值 |
/// |------|------|--------|
/// | `block_size` | 每个物理块容纳的 token 数 | 16 |
/// | `max_num_blocks` | 最大物理块数量 | 根据 GPU 显存调整 |
/// | `max_batch_size` | 单次调度最大序列数 | 32 |
/// | `max_num_seqs` | 系统最大并发序列数 | 256 |
/// | `max_model_len` | 模型最大上下文长度 | 2048 |
/// | `max_total_tokens` | 单批次最大 token 总数 | 4096 |
/// | `memory_threshold` | 内存压力阈值 | 0.9 |
/// | `max_retry_attempts` | GPU 执行重试次数 | 2 |
/// | `special_tokens` | 特殊 Token ID 配置 | 默认值 |
///
/// # 示例
///
/// ```rust
/// use hetero_infer::EngineConfig;
///
/// let config = EngineConfig {
///     block_size: 16,
///     max_num_blocks: 1024,
///     max_batch_size: 32,
///     max_num_seqs: 256,
///     max_model_len: 2048,
///     max_total_tokens: 4096,
///     memory_threshold: 0.9,
///     ..Default::default()
/// };
///
/// assert!(config.is_valid());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// 每个 KV Cache 块容纳的 token 数
    ///
    /// 较小的块大小提供更细粒度的内存管理，但增加页表开销。
    /// 常见值为 16 或 32。
    pub block_size: u32,

    /// 物理块最大数量
    ///
    /// 决定 KV Cache 总容量。总 token 容量 = `max_num_blocks * block_size`。
    pub max_num_blocks: u32,

    /// 单次调度最大序列数
    ///
    /// 限制单次 GPU 执行的序列数量，影响 GPU 利用率和延迟。
    pub max_batch_size: u32,

    /// 系统最大并发序列数
    ///
    /// 包括 pending、prefill、decode 各阶段的序列总数上限。
    pub max_num_seqs: u32,

    /// 最大序列长度（输入 + 输出）
    ///
    /// 单个请求的 token 数上限。
    pub max_model_len: u32,

    /// 单批次最大 token 总数
    ///
    /// 限制单次 GPU 执行的 token 总量，防止显存溢出。
    pub max_total_tokens: u32,

    /// 内存压力阈值 (0.0 - 1.0)
    ///
    /// 当内存利用率超过此阈值时，调度器将拒绝新的 prefill 请求。
    /// 建议设置在 0.8-0.95 之间，留出安全余量。
    pub memory_threshold: f32,

    /// GPU 执行超时的最大重试次数
    ///
    /// 当 GPU 执行超时时，引擎将重试的最大次数。
    /// 默认值为 2。
    pub max_retry_attempts: u32,

    /// 特殊 Token ID 配置
    ///
    /// 包含 BOS、EOS、PAD、UNK 等特殊 token 的 ID。
    pub special_tokens: SpecialTokenIds,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            max_num_blocks: 1024,
            max_batch_size: 32,
            max_num_seqs: 256,
            max_model_len: 2048,
            max_total_tokens: 4096,
            memory_threshold: 0.9,
            max_retry_attempts: 2,
            special_tokens: SpecialTokenIds::default(),
        }
    }
}

impl EngineConfig {
    /// 创建新配置并验证
    ///
    /// # 参数
    ///
    /// * `block_size` - 每块 token 数
    /// * `max_num_blocks` - 最大块数
    /// * `max_batch_size` - 最大批次大小
    /// * `max_num_seqs` - 最大并发序列数
    /// * `max_model_len` - 最大序列长度
    /// * `max_total_tokens` - 单批次最大 token 数
    /// * `memory_threshold` - 内存阈值
    ///
    /// # 错误
    ///
    /// 如果任何参数无效，返回 [`ConfigError`]。
    ///
    /// # 示例
    ///
    /// ```rust
    /// use hetero_infer::EngineConfig;
    ///
    /// let config = EngineConfig::new(16, 1024, 32, 256, 2048, 4096, 0.9)?;
    /// assert!(config.is_valid());
    /// # Ok::<(), hetero_infer::ConfigError>(())
    /// ```
    pub fn new(
        block_size: u32,
        max_num_blocks: u32,
        max_batch_size: u32,
        max_num_seqs: u32,
        max_model_len: u32,
        max_total_tokens: u32,
        memory_threshold: f32,
    ) -> Result<Self, ConfigError> {
        let config = Self {
            block_size,
            max_num_blocks,
            max_batch_size,
            max_num_seqs,
            max_model_len,
            max_total_tokens,
            memory_threshold,
            max_retry_attempts: 2,
            special_tokens: SpecialTokenIds::default(),
        };
        config.validate()?;
        Ok(config)
    }

    /// 验证配置参数
    ///
    /// 检查所有参数是否在有效范围内。
    ///
    /// # 错误
    ///
    /// - [`ConfigError::InvalidBlockSize`] `block_size` 为 0
    /// - [`ConfigError::InvalidMaxNumBlocks`] `max_num_blocks` 为 0
    /// - [`ConfigError::InvalidMaxBatchSize`] `max_batch_size` 为 0
    /// - [`ConfigError::InvalidMaxNumSeqs`] `max_num_seqs` 为 0
    /// - [`ConfigError::InvalidMaxModelLen`] `max_model_len` 为 0
    /// - [`ConfigError::InvalidMaxTotalTokens`] `max_total_tokens` 为 0
    /// - [`ConfigError::InvalidMemoryThreshold`] `memory_threshold` 不在 (0.0, 1.0] 范围内
    ///
    /// # 示例
    ///
    /// ```rust
    /// use hetero_infer::EngineConfig;
    ///
    /// let config = EngineConfig::default();
    /// assert!(config.validate().is_ok());
    ///
    /// let invalid_config = EngineConfig { block_size: 0, ..Default::default() };
    /// assert!(invalid_config.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.block_size == 0 {
            return Err(ConfigError::InvalidBlockSize(self.block_size));
        }
        if self.max_num_blocks == 0 {
            return Err(ConfigError::InvalidMaxNumBlocks(self.max_num_blocks));
        }
        if self.max_batch_size == 0 {
            return Err(ConfigError::InvalidMaxBatchSize(self.max_batch_size));
        }
        if self.max_num_seqs == 0 {
            return Err(ConfigError::InvalidMaxNumSeqs(self.max_num_seqs));
        }
        if self.max_model_len == 0 {
            return Err(ConfigError::InvalidMaxModelLen(self.max_model_len));
        }
        if self.max_total_tokens == 0 {
            return Err(ConfigError::InvalidMaxTotalTokens(self.max_total_tokens));
        }
        if self.memory_threshold <= 0.0 || self.memory_threshold > 1.0 {
            return Err(ConfigError::InvalidMemoryThreshold(self.memory_threshold));
        }
        // Note: max_retry_attempts can be 0 (no retries) or any positive value
        Ok(())
    }

    /// 检查配置是否有效
    ///
    /// 返回 `true` 表示所有参数都在有效范围内。
    ///
    /// # 示例
    ///
    /// ```rust
    /// use hetero_infer::EngineConfig;
    ///
    /// assert!(EngineConfig::default().is_valid());
    ///
    /// let invalid = EngineConfig { block_size: 0, ..Default::default() };
    /// assert!(!invalid.is_valid());
    /// ```
    pub fn is_valid(&self) -> bool {
        self.block_size > 0
            && self.max_num_blocks > 0
            && self.max_batch_size > 0
            && self.max_num_seqs > 0
            && self.max_model_len > 0
            && self.max_total_tokens > 0
            && self.memory_threshold > 0.0
            && self.memory_threshold <= 1.0
    }

    /// 从 JSON 文件加载配置
    ///
    /// # 参数
    ///
    /// * `path` - 配置文件路径
    ///
    /// # 错误
    ///
    /// - [`ConfigError::FileLoadError`] 文件读取失败
    /// - [`ConfigError::ParseError`] JSON 解析失败
    /// - [`ConfigError`] 参数验证失败
    ///
    /// # 示例
    ///
    /// ```rust,no_run
    /// use hetero_infer::EngineConfig;
    /// use std::path::Path;
    ///
    /// let config = EngineConfig::from_file(Path::new("config.json"))?;
    /// # Ok::<(), hetero_infer::ConfigError>(())
    /// ```
    pub fn from_file(path: &Path) -> Result<Self, ConfigError> {
        let content =
            std::fs::read_to_string(path).map_err(|e| ConfigError::FileLoadError(e.to_string()))?;
        let config: Self =
            serde_json::from_str(&content).map_err(|e| ConfigError::ParseError(e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    /// 保存配置到 JSON 文件
    ///
    /// # 参数
    ///
    /// * `path` - 目标文件路径
    ///
    /// # 错误
    ///
    /// - [`ConfigError::ParseError`] JSON 序列化失败
    /// - [`ConfigError::FileSaveError`] 文件写入失败
    ///
    /// # 示例
    ///
    /// ```rust,no_run
    /// use hetero_infer::EngineConfig;
    /// use std::path::Path;
    ///
    /// let config = EngineConfig::default();
    /// config.to_file(Path::new("config.json"))?;
    /// # Ok::<(), hetero_infer::ConfigError>(())
    /// ```
    pub fn to_file(&self, path: &Path) -> Result<(), ConfigError> {
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| ConfigError::ParseError(e.to_string()))?;
        std::fs::write(path, content).map_err(|e| ConfigError::FileSaveError(e.to_string()))?;
        Ok(())
    }

    /// 计算指定 token 数量需要的块数
    ///
    /// 返回 `ceil(num_tokens / block_size)`。
    ///
    /// # 示例
    ///
    /// ```rust
    /// use hetero_infer::EngineConfig;
    ///
    /// let config = EngineConfig { block_size: 16, ..Default::default() };
    ///
    /// assert_eq!(config.blocks_for_tokens(0), 0);
    /// assert_eq!(config.blocks_for_tokens(1), 1);
    /// assert_eq!(config.blocks_for_tokens(16), 1);
    /// assert_eq!(config.blocks_for_tokens(17), 2);
    /// ```
    pub fn blocks_for_tokens(&self, num_tokens: u32) -> u32 {
        num_tokens.div_ceil(self.block_size)
    }

    /// 计算指定块数可容纳的 token 数
    ///
    /// 返回 `num_blocks * block_size`。
    ///
    /// # 示例
    ///
    /// ```rust
    /// use hetero_infer::EngineConfig;
    ///
    /// let config = EngineConfig { block_size: 16, ..Default::default() };
    ///
    /// assert_eq!(config.tokens_in_blocks(0), 0);
    /// assert_eq!(config.tokens_in_blocks(1), 16);
    /// assert_eq!(config.tokens_in_blocks(2), 32);
    /// ```
    pub fn tokens_in_blocks(&self, num_blocks: u32) -> u32 {
        num_blocks * self.block_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let config = EngineConfig::default();
        assert!(config.validate().is_ok());
        assert!(config.is_valid());
    }

    #[test]
    fn test_invalid_block_size() {
        let config = EngineConfig {
            block_size: 0,
            ..Default::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidBlockSize(0))
        ));
        assert!(!config.is_valid());
    }

    #[test]
    fn test_invalid_max_num_blocks() {
        let config = EngineConfig {
            max_num_blocks: 0,
            ..Default::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidMaxNumBlocks(0))
        ));
    }

    #[test]
    fn test_invalid_max_batch_size() {
        let config = EngineConfig {
            max_batch_size: 0,
            ..Default::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidMaxBatchSize(0))
        ));
    }

    #[test]
    fn test_invalid_max_num_seqs() {
        let config = EngineConfig {
            max_num_seqs: 0,
            ..Default::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidMaxNumSeqs(0))
        ));
    }

    #[test]
    fn test_invalid_memory_threshold() {
        let config = EngineConfig {
            memory_threshold: 0.0,
            ..Default::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidMemoryThreshold(_))
        ));

        let config2 = EngineConfig {
            memory_threshold: 1.5,
            ..Default::default()
        };
        assert!(matches!(
            config2.validate(),
            Err(ConfigError::InvalidMemoryThreshold(_))
        ));
    }

    #[test]
    fn test_blocks_for_tokens() {
        let config = EngineConfig {
            block_size: 16,
            ..Default::default()
        };

        assert_eq!(config.blocks_for_tokens(0), 0);
        assert_eq!(config.blocks_for_tokens(1), 1);
        assert_eq!(config.blocks_for_tokens(16), 1);
        assert_eq!(config.blocks_for_tokens(17), 2);
        assert_eq!(config.blocks_for_tokens(32), 2);
        assert_eq!(config.blocks_for_tokens(33), 3);
    }

    #[test]
    fn test_tokens_in_blocks() {
        let config = EngineConfig {
            block_size: 16,
            ..Default::default()
        };

        assert_eq!(config.tokens_in_blocks(0), 0);
        assert_eq!(config.tokens_in_blocks(1), 16);
        assert_eq!(config.tokens_in_blocks(2), 32);
        assert_eq!(config.tokens_in_blocks(3), 48);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// **Feature: heterogeneous-inference-system, Property 14: Configuration Validation**
        /// *对于任意* 配置输入，验证应拒绝 block_size <= 0, max_num_blocks <= 0, max_batch_size <= 0, 或 max_num_seqs <= 0 的配置。
        /// **验证: Requirements 7.2**
        #[test]
        fn prop_config_validation(
            block_size in 0u32..100,
            max_num_blocks in 0u32..2000,
            max_batch_size in 0u32..100,
            max_num_seqs in 0u32..500,
            max_model_len in 0u32..10000,
            max_total_tokens in 0u32..10000,
            memory_threshold in -0.5f32..1.5,
        ) {
            let config = EngineConfig {
                block_size,
                max_num_blocks,
                max_batch_size,
                max_num_seqs,
                max_model_len,
                max_total_tokens,
                memory_threshold,
                ..Default::default()
            };

            let validation_result = config.validate();
            let is_valid = config.is_valid();

            // 基于参数范围的预期有效性
            let expected_valid = block_size > 0
                && max_num_blocks > 0
                && max_batch_size > 0
                && max_num_seqs > 0
                && max_model_len > 0
                && max_total_tokens > 0
                && memory_threshold > 0.0
                && memory_threshold <= 1.0;

            // 属性: 验证结果与预期有效性一致
            prop_assert_eq!(
                validation_result.is_ok(),
                expected_valid,
                "验证不匹配，配置: {:?}",
                config
            );

            // 属性: is_valid() 与 validate() 一致
            prop_assert_eq!(
                is_valid,
                expected_valid,
                "is_valid() 与 validate() 不一致，配置: {:?}",
                config
            );
        }

        /// 特定无效配置的属性测试
        #[test]
        fn prop_invalid_configs_rejected(
            valid_block_size in 1u32..100,
            valid_max_num_blocks in 1u32..2000,
            valid_max_batch_size in 1u32..100,
            valid_max_num_seqs in 1u32..500,
            valid_max_model_len in 1u32..10000,
            valid_max_total_tokens in 1u32..10000,
            valid_memory_threshold in 0.01f32..1.0,
        ) {
            // 测试 block_size = 0
            let config_zero_block = EngineConfig {
                block_size: 0,
                max_num_blocks: valid_max_num_blocks,
                max_batch_size: valid_max_batch_size,
                max_num_seqs: valid_max_num_seqs,
                max_model_len: valid_max_model_len,
                max_total_tokens: valid_max_total_tokens,
                memory_threshold: valid_memory_threshold,
                ..Default::default()
            };
            prop_assert!(config_zero_block.validate().is_err());

            // 测试 max_num_blocks = 0
            let config_zero_blocks = EngineConfig {
                block_size: valid_block_size,
                max_num_blocks: 0,
                max_batch_size: valid_max_batch_size,
                max_num_seqs: valid_max_num_seqs,
                max_model_len: valid_max_model_len,
                max_total_tokens: valid_max_total_tokens,
                memory_threshold: valid_memory_threshold,
                ..Default::default()
            };
            prop_assert!(config_zero_blocks.validate().is_err());

            // 测试 max_batch_size = 0
            let config_zero_batch = EngineConfig {
                block_size: valid_block_size,
                max_num_blocks: valid_max_num_blocks,
                max_batch_size: 0,
                max_num_seqs: valid_max_num_seqs,
                max_model_len: valid_max_model_len,
                max_total_tokens: valid_max_total_tokens,
                memory_threshold: valid_memory_threshold,
                ..Default::default()
            };
            prop_assert!(config_zero_batch.validate().is_err());

            // 测试 max_num_seqs = 0
            let config_zero_seqs = EngineConfig {
                block_size: valid_block_size,
                max_num_blocks: valid_max_num_blocks,
                max_batch_size: valid_max_batch_size,
                max_num_seqs: 0,
                max_model_len: valid_max_model_len,
                max_total_tokens: valid_max_total_tokens,
                memory_threshold: valid_memory_threshold,
                ..Default::default()
            };
            prop_assert!(config_zero_seqs.validate().is_err());

            // 测试所有有效参数
            let valid_config = EngineConfig {
                block_size: valid_block_size,
                max_num_blocks: valid_max_num_blocks,
                max_batch_size: valid_max_batch_size,
                max_num_seqs: valid_max_num_seqs,
                max_model_len: valid_max_model_len,
                max_total_tokens: valid_max_total_tokens,
                memory_threshold: valid_memory_threshold,
                ..Default::default()
            };
            prop_assert!(valid_config.validate().is_ok());
        }
    }
}
