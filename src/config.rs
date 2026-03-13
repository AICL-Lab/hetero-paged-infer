//! Configuration types and validation

use crate::error::ConfigError;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Number of tokens per KV cache block
    pub block_size: u32,
    /// Maximum number of physical blocks for KV cache
    pub max_num_blocks: u32,
    /// Maximum number of sequences per batch
    pub max_batch_size: u32,
    /// Maximum number of concurrent sequences
    pub max_num_seqs: u32,
    /// Maximum sequence length (input + output)
    pub max_model_len: u32,
    /// Maximum total tokens per batch
    pub max_total_tokens: u32,
    /// Memory pressure threshold (0.0 - 1.0)
    pub memory_threshold: f32,
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
        }
    }
}

impl EngineConfig {
    /// Create a new configuration with validation
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
        };
        config.validate()?;
        Ok(config)
    }

    /// Validate configuration parameters
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
        Ok(())
    }

    /// Check if configuration is valid
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

    /// Load configuration from a JSON file
    pub fn from_file(path: &Path) -> Result<Self, ConfigError> {
        let content =
            std::fs::read_to_string(path).map_err(|e| ConfigError::FileLoadError(e.to_string()))?;
        let config: Self =
            serde_json::from_str(&content).map_err(|e| ConfigError::ParseError(e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    /// Save configuration to a JSON file
    pub fn to_file(&self, path: &Path) -> Result<(), ConfigError> {
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| ConfigError::ParseError(e.to_string()))?;
        std::fs::write(path, content).map_err(|e| ConfigError::FileSaveError(e.to_string()))?;
        Ok(())
    }

    /// Calculate maximum blocks needed for a sequence of given length
    pub fn blocks_for_tokens(&self, num_tokens: u32) -> u32 {
        num_tokens.div_ceil(self.block_size)
    }

    /// Calculate maximum tokens that can fit in given blocks
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
        /// *For any* configuration input, the validation shall reject configurations where
        /// block_size <= 0, max_num_blocks <= 0, max_batch_size <= 0, or max_num_seqs <= 0.
        /// **Validates: Requirements 7.2**
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
            };

            let validation_result = config.validate();
            let is_valid = config.is_valid();

            // Expected validity based on parameter ranges
            let expected_valid = block_size > 0
                && max_num_blocks > 0
                && max_batch_size > 0
                && max_num_seqs > 0
                && max_model_len > 0
                && max_total_tokens > 0
                && memory_threshold > 0.0
                && memory_threshold <= 1.0;

            // Property: validation result matches expected validity
            prop_assert_eq!(
                validation_result.is_ok(),
                expected_valid,
                "Validation mismatch for config: {:?}",
                config
            );

            // Property: is_valid() is consistent with validate()
            prop_assert_eq!(
                is_valid,
                expected_valid,
                "is_valid() inconsistent with validate() for config: {:?}",
                config
            );
        }

        /// Property test for specific invalid configurations
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
            // Test with zero block_size
            let config_zero_block = EngineConfig {
                block_size: 0,
                max_num_blocks: valid_max_num_blocks,
                max_batch_size: valid_max_batch_size,
                max_num_seqs: valid_max_num_seqs,
                max_model_len: valid_max_model_len,
                max_total_tokens: valid_max_total_tokens,
                memory_threshold: valid_memory_threshold,
            };
            prop_assert!(config_zero_block.validate().is_err());

            // Test with zero max_num_blocks
            let config_zero_blocks = EngineConfig {
                block_size: valid_block_size,
                max_num_blocks: 0,
                max_batch_size: valid_max_batch_size,
                max_num_seqs: valid_max_num_seqs,
                max_model_len: valid_max_model_len,
                max_total_tokens: valid_max_total_tokens,
                memory_threshold: valid_memory_threshold,
            };
            prop_assert!(config_zero_blocks.validate().is_err());

            // Test with zero max_batch_size
            let config_zero_batch = EngineConfig {
                block_size: valid_block_size,
                max_num_blocks: valid_max_num_blocks,
                max_batch_size: 0,
                max_num_seqs: valid_max_num_seqs,
                max_model_len: valid_max_model_len,
                max_total_tokens: valid_max_total_tokens,
                memory_threshold: valid_memory_threshold,
            };
            prop_assert!(config_zero_batch.validate().is_err());

            // Test with zero max_num_seqs
            let config_zero_seqs = EngineConfig {
                block_size: valid_block_size,
                max_num_blocks: valid_max_num_blocks,
                max_batch_size: valid_max_batch_size,
                max_num_seqs: 0,
                max_model_len: valid_max_model_len,
                max_total_tokens: valid_max_total_tokens,
                memory_threshold: valid_memory_threshold,
            };
            prop_assert!(config_zero_seqs.validate().is_err());

            // Test with all valid parameters
            let valid_config = EngineConfig {
                block_size: valid_block_size,
                max_num_blocks: valid_max_num_blocks,
                max_batch_size: valid_max_batch_size,
                max_num_seqs: valid_max_num_seqs,
                max_model_len: valid_max_model_len,
                max_total_tokens: valid_max_total_tokens,
                memory_threshold: valid_memory_threshold,
            };
            prop_assert!(valid_config.validate().is_ok());
        }
    }
}
