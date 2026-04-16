//! Shared test utilities
//!
//! Provides common helpers for unit, property, and integration tests.

use crate::config::EngineConfig;
use crate::types::{GenerationParams, Request, RequestId};

/// Create a standard test configuration
pub fn create_test_config() -> EngineConfig {
    EngineConfig {
        block_size: 16,
        max_num_blocks: 100,
        max_batch_size: 8,
        max_num_seqs: 32,
        max_model_len: 2048,
        max_total_tokens: 512,
        memory_threshold: 0.9,
        max_retry_attempts: 2,
        special_tokens: Default::default(),
    }
}

/// Create a test configuration with custom batch/token/block limits
pub fn create_test_config_with_limits(
    max_batch_size: u32,
    max_total_tokens: u32,
    max_num_blocks: u32,
) -> EngineConfig {
    EngineConfig {
        block_size: 16,
        max_num_blocks,
        max_batch_size,
        max_num_seqs: 64,
        max_model_len: 2048,
        max_total_tokens,
        memory_threshold: 0.9,
        max_retry_attempts: 2,
        special_tokens: Default::default(),
    }
}

/// Create a test request with the given number of dummy tokens
pub fn create_test_request(id: RequestId, num_tokens: usize) -> Request {
    Request::new(id, vec![1; num_tokens], GenerationParams::default())
}

/// Create a test request with custom generation params
pub fn create_test_request_with_params(id: RequestId, num_tokens: usize, max_gen: u32) -> Request {
    Request::new(
        id,
        vec![1; num_tokens],
        GenerationParams {
            max_tokens: max_gen,
            temperature: 1.0,
            top_p: 1.0,
        },
    )
}
