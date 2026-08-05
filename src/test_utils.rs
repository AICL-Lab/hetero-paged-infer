//! Shared test utilities
//!
//! Provides common helpers for unit, property, and integration tests.
//!
//! This module is only compiled under `cfg(test)` or the `test-utils`
//! feature (enabled by the crate's own dev-dependencies for integration
//! tests and benches), so test helpers never ship in release builds.

use crate::config::EngineConfig;
use crate::error::EngineError;
use crate::gpu_executor::GPUExecutorTrait;
use crate::types::{ExecutionBatch, ExecutionOutput, GenerationParams, Request, RequestId};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

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
        ..Default::default()
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
        ..Default::default()
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

/// Standard generation parameters for tests
pub fn test_params(max_tokens: u32) -> GenerationParams {
    GenerationParams {
        max_tokens,
        temperature: 1.0,
        top_p: 0.9,
    }
}

/// A GPU executor that always fails with `KernelLaunchFailed`.
///
/// Shared by unit and integration tests that exercise engine error paths.
pub struct AlwaysFailExecutor;

impl GPUExecutorTrait for AlwaysFailExecutor {
    fn execute(&mut self, _batch: &ExecutionBatch) -> Result<ExecutionOutput, EngineError> {
        Err(EngineError::KernelLaunchFailed(
            "test executor failure".to_string(),
        ))
    }
}

/// Minimal HuggingFace `WordLevel` tokenizer definition used by tokenizer tests.
const TEST_TOKENIZER_JSON: &str = r###"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": { "type": "Whitespace" },
  "post_processor": null,
  "decoder": { "type": "WordPiece", "prefix": "##", "cleanup": false },
  "model": {
    "type": "WordLevel",
    "vocab": {
      "[UNK]": 0,
      "hello": 1,
      "world": 2
    },
    "unk_token": "[UNK]"
  }
}"###;

/// Write a minimal HuggingFace tokenizer JSON to a unique temp file.
///
/// Callers own cleanup (`std::fs::remove_file`), which is best-effort:
/// a failing test may leave the file behind.
pub fn write_test_tokenizer_json() -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("hetero-test-tokenizer-{unique}.json"));
    fs::write(&path, TEST_TOKENIZER_JSON).unwrap();
    path
}
