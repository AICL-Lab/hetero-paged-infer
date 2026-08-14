//! Integration tests for the Heterogeneous Inference System
//!
//! These tests verify end-to-end functionality across all components.

use paged_infer::{
    test_utils::{create_test_config, AlwaysFailExecutor},
    EngineConfig, EngineError, GenerationParams, InferenceEngine, Scheduler, SimpleTokenizer,
};

fn create_failure_test_engine(config: EngineConfig) -> InferenceEngine {
    InferenceEngine::with_components(
        config.clone(),
        Box::new(SimpleTokenizer::new()),
        Scheduler::new(config),
        Box::new(AlwaysFailExecutor),
    )
    .unwrap()
}

/// **Integration Test: End-to-End Request Flow**
///
/// Tests the complete flow:
/// 1. Submit request
/// 2. Run until completion
/// 3. Verify output tokens generated
/// 4. Verify KV cache freed after completion
///
/// _Requirements: 1.1, 1.5, 2.4_
#[test]
fn test_end_to_end_request_flow() {
    let config = create_test_config();
    let mut engine = InferenceEngine::new(config).unwrap();

    let params = GenerationParams {
        max_tokens: 5,
        ..GenerationParams::default()
    };

    // Submit request
    let (request_id, prompt_tokens) = engine.submit_request("Hello world", params).unwrap();
    assert!(request_id > 0);
    assert!(prompt_tokens > 0);

    // Verify pending work
    assert!(engine.has_pending_work());

    // Run to completion
    let completed = engine.run();

    // Verify completion
    assert!(!completed.is_empty(), "Should have completed requests");

    let result = &completed[0];
    assert!(result.success, "Request should succeed");
    assert!(
        !result.output_tokens.is_empty(),
        "Should have generated tokens"
    );

    // Verify no pending work (KV cache should be freed)
    assert!(
        !engine.has_pending_work(),
        "Should have no pending work after completion"
    );

    // Memory should be mostly free
    let utilization = engine.memory_utilization();
    assert!(utilization < 0.5, "Memory should be freed after completion");
}

/// **Integration Test: Multiple Request Completion**
///
/// Tests that multiple requests can be processed correctly.
///
/// _Requirements: 1.1, 1.5_
#[test]
fn test_multiple_requests_completion() {
    let config = create_test_config();
    let mut engine = InferenceEngine::new(config).unwrap();

    let params = GenerationParams {
        max_tokens: 3,
        ..GenerationParams::default()
    };

    // Submit multiple requests
    let num_requests = 5;
    for i in 0..num_requests {
        engine
            .submit_request(&format!("Request number {}", i), params)
            .unwrap();
    }

    // Run to completion
    let completed = engine.run();

    // All requests should complete
    assert_eq!(
        completed.len(),
        num_requests,
        "All requests should complete"
    );

    for result in &completed {
        assert!(result.success, "Each request should succeed");
        assert!(!result.output_tokens.is_empty(), "Each should have output");
    }
}

/// **Integration Test: Request with EOS Token**
///
/// Tests that requests complete when EOS token is generated.
///
/// _Requirements: 3.4_
#[test]
fn test_request_completion_on_max_tokens() {
    let config = create_test_config();
    let mut engine = InferenceEngine::new(config).unwrap();

    let max_tokens = 5;
    let params = GenerationParams {
        max_tokens,
        ..GenerationParams::default()
    };

    engine.submit_request("Test input", params).unwrap();

    let completed = engine.run();

    assert!(!completed.is_empty());
    let result = &completed[0];

    // Should have at most max_tokens output tokens
    assert!(
        result.output_tokens.len() <= max_tokens as usize,
        "Should not exceed max_tokens"
    );
}

/// **Integration Test: Memory Utilization Tracking**
///
/// Tests that memory utilization is tracked correctly.
///
/// _Requirements: 6.2, 6.4_
#[test]
fn test_memory_utilization_tracking() {
    let config = create_test_config();
    let mut engine = InferenceEngine::new(config).unwrap();

    // Initial utilization should be low
    let initial_util = engine.memory_utilization();
    assert!(initial_util < 0.1, "Initial utilization should be low");

    // Submit requests
    let params = GenerationParams {
        max_tokens: 10,
        ..GenerationParams::default()
    };

    for i in 0..5 {
        engine
            .submit_request(&format!("Request {}", i), params)
            .unwrap();
    }

    // Schedule to allocate memory
    let _ = engine.step();

    // Utilization must strictly increase: five in-flight sequences now hold KV blocks
    let after_util = engine.memory_utilization();
    assert!(
        after_util > initial_util,
        "Utilization should increase after allocation (initial={initial_util}, after={after_util})"
    );
    assert!(
        after_util > 0.0,
        "Active sequences must occupy at least one block"
    );
}

/// **Integration Test: Invalid Request Handling**
///
/// Tests that invalid requests are rejected properly.
///
/// _Requirements: 1.4_
#[test]
fn test_invalid_request_handling() {
    let config = create_test_config();
    let mut engine = InferenceEngine::new(config).unwrap();

    // Empty input
    let params = GenerationParams::default();
    let result = engine.submit_request("", params);
    assert!(result.is_err(), "Empty input should be rejected");

    // Invalid max_tokens
    let invalid_params = GenerationParams {
        max_tokens: 0,
        ..GenerationParams::default()
    };
    let result = engine.submit_request("Hello", invalid_params);
    assert!(result.is_err(), "Invalid params should be rejected");

    // Invalid temperature (negative)
    let invalid_params = GenerationParams {
        max_tokens: 10,
        temperature: -0.5,
        top_p: 0.9,
    };
    let result = engine.submit_request("Hello", invalid_params);
    assert!(result.is_err(), "Invalid temperature should be rejected");

    // Greedy decoding (temperature == 0.0, top_p == 1.0) is valid
    let greedy_params = GenerationParams {
        max_tokens: 10,
        temperature: 0.0,
        top_p: 1.0,
    };
    let result = engine.submit_request("Hello", greedy_params);
    assert!(result.is_ok(), "Greedy temperature 0.0 should be accepted");

    // 范围内但后端不支持的采样参数：submit 阶段诚实失败，而非静默降级
    let sampled_params = GenerationParams {
        max_tokens: 10,
        temperature: 1.0,
        top_p: 1.0,
    };
    let result = engine.submit_request("Hello", sampled_params);
    assert!(
        matches!(result, Err(EngineError::UnsupportedGenerationMode(_))),
        "non-greedy temperature must be rejected with UnsupportedGenerationMode"
    );

    let top_p_params = GenerationParams {
        max_tokens: 10,
        temperature: 0.0,
        top_p: 0.9,
    };
    let result = engine.submit_request("Hello", top_p_params);
    assert!(
        matches!(result, Err(EngineError::UnsupportedGenerationMode(_))),
        "top_p != 1.0 must be rejected with UnsupportedGenerationMode"
    );

    // Invalid top_p
    let invalid_params = GenerationParams {
        max_tokens: 10,
        temperature: 1.0,
        top_p: 1.5,
    };
    let result = engine.submit_request("Hello", invalid_params);
    assert!(result.is_err(), "Invalid top_p should be rejected");

    // Total requested length beyond max_model_len
    let tiny_config = EngineConfig {
        max_model_len: 8,
        ..create_test_config()
    };
    let mut tiny_engine = InferenceEngine::new(tiny_config).unwrap();
    let too_long_params = GenerationParams {
        max_tokens: 4,
        ..GenerationParams::default()
    };
    let result = tiny_engine.submit_request("Hello", too_long_params);
    assert!(result.is_err(), "Total requested length should be rejected");
}

/// **Integration Test: Continuous Batching**
///
/// Tests that continuous batching works correctly with mixed prefill/decode.
///
/// _Requirements: 3.2, 3.3, 3.4_
#[test]
fn test_continuous_batching() {
    let config = create_test_config();
    let mut engine = InferenceEngine::new(config).unwrap();

    let params = GenerationParams {
        max_tokens: 5,
        ..GenerationParams::default()
    };

    // Submit first request
    engine.submit_request("First request", params).unwrap();

    // Run a few steps to get it into decode phase
    for _ in 0..3 {
        let _ = engine.step();
    }

    // Submit second request while first is in decode
    engine.submit_request("Second request", params).unwrap();

    // Continue running
    let completed = engine.run();

    // Both requests must complete successfully
    assert_eq!(completed.len(), 2, "both requests should complete");
    assert!(completed.iter().all(|c| c.success));
    assert!(!engine.has_pending_work());
}

/// **Integration Test: Configuration Validation**
///
/// Tests that invalid configurations are rejected.
///
/// _Requirements: 7.2_
#[test]
fn test_configuration_validation() {
    // Invalid block_size
    let invalid_config = EngineConfig {
        block_size: 0,
        ..create_test_config()
    };
    let result = InferenceEngine::new(invalid_config);
    assert!(result.is_err(), "Invalid block_size should be rejected");

    // Invalid max_num_blocks
    let invalid_config = EngineConfig {
        max_num_blocks: 0,
        ..create_test_config()
    };
    let result = InferenceEngine::new(invalid_config);
    assert!(result.is_err(), "Invalid max_num_blocks should be rejected");

    // Invalid max_batch_size
    let invalid_config = EngineConfig {
        max_batch_size: 0,
        ..create_test_config()
    };
    let result = InferenceEngine::new(invalid_config);
    assert!(result.is_err(), "Invalid max_batch_size should be rejected");
}

/// **Integration Test: Memory Pressure Handling**
///
/// Tests that the system handles memory pressure correctly:
/// 1. Fill memory to threshold
/// 2. Verify new prefills are rejected
/// 3. Complete some requests
/// 4. Verify new prefills are accepted again
///
/// _Requirements: 6.3, 6.5_
#[test]
fn test_memory_pressure_handling() {
    // Use small block count to trigger memory pressure
    let config = EngineConfig {
        block_size: 16,
        max_num_blocks: 10, // Very small to trigger pressure
        max_batch_size: 8,
        max_num_seqs: 32,
        max_model_len: 256,
        max_total_tokens: 256,
        memory_threshold: 0.5, // Low threshold
        ..Default::default()
    };

    let mut engine = InferenceEngine::new(config).unwrap();

    let params = GenerationParams {
        max_tokens: 50, // Long generation to hold memory
        ..GenerationParams::default()
    };

    // Submit requests until memory pressure, collecting everything that
    // terminates during the loop (some requests may complete or fail early).
    let mut submitted = 0;
    let mut saw_memory_pressure = false;
    let mut finished = Vec::new();
    for i in 0..20 {
        match engine.submit_request(&format!("Request {}", i), params) {
            Ok(_) => {
                submitted += 1;
                // Schedule to allocate memory
                finished.extend(engine.step().unwrap());
            }
            Err(EngineError::MemoryPressure) => {
                saw_memory_pressure = true;
                break;
            }
            Err(other) => panic!("unexpected submit error: {other}"),
        }
    }

    // With 10 blocks and a 0.5 threshold, pressure must actually trigger
    assert!(submitted > 0, "Should submit at least one request");
    assert!(
        saw_memory_pressure,
        "submitting against a nearly-full block pool must be rejected with MemoryPressure"
    );

    // Run to completion: every accepted request must reach a terminal state.
    // Under extreme pressure a decode may fail to grow its KV allocation and
    // surface as an explicit failure — what matters is that nothing gets stuck.
    finished.extend(engine.run());
    assert_eq!(
        finished.len(),
        submitted,
        "all accepted requests must terminate"
    );
    assert!(
        finished.iter().any(|c| c.success),
        "at least some requests should succeed"
    );
    assert!(!engine.has_pending_work());
}

/// **Integration Test: Large Batch Processing**
///
/// Tests processing of larger batches.
///
/// _Requirements: 3.5_
#[test]
fn test_large_batch_processing() {
    let config = EngineConfig {
        block_size: 16,
        max_num_blocks: 500,
        max_batch_size: 16,
        max_num_seqs: 64,
        max_model_len: 2048,
        max_total_tokens: 2048,
        memory_threshold: 0.9,
        ..Default::default()
    };

    let mut engine = InferenceEngine::new(config).unwrap();

    let params = GenerationParams {
        max_tokens: 3,
        ..GenerationParams::default()
    };

    // Submit many requests
    let num_requests = 20;
    for i in 0..num_requests {
        engine
            .submit_request(&format!("Batch request {}", i), params)
            .unwrap();
    }

    // Run to completion: all requests fit within limits and must complete
    let completed = engine.run();
    assert_eq!(
        completed.len(),
        num_requests,
        "all submitted requests should complete"
    );
    assert!(completed.iter().all(|c| c.success));
}

/// **Integration Test: Sequential Request Processing**
///
/// Tests that requests are processed in submission order (FCFS).
#[test]
fn test_sequential_request_processing() {
    let config = create_test_config();
    let mut engine = InferenceEngine::new(config).unwrap();

    let params = GenerationParams {
        max_tokens: 2,
        ..GenerationParams::default()
    };

    // Submit requests one at a time and process each to completion
    let mut completed_ids = Vec::new();
    for i in 0..3 {
        engine
            .submit_request(&format!("Sequential {}", i), params)
            .unwrap();

        // Run until this request completes
        let mut done = Vec::new();
        for _ in 0..50 {
            done.extend(engine.step().unwrap());
            if !done.is_empty() {
                break;
            }
        }

        assert_eq!(
            done.len(),
            1,
            "exactly one request should complete per round"
        );
        completed_ids.push(done[0].request_id);
    }

    assert_eq!(
        completed_ids,
        vec![1, 2, 3],
        "requests must complete in submission order"
    );
    assert!(!engine.has_pending_work());
}

#[test]
fn test_execution_failure_surfaces_as_completed_error() {
    let config = create_test_config();
    let mut engine = create_failure_test_engine(config);

    let params = GenerationParams {
        max_tokens: 5,
        ..GenerationParams::default()
    };

    engine.submit_request("Failure case", params).unwrap();
    let completed = engine.run();

    assert_eq!(completed.len(), 1);
    assert!(!completed[0].success);
    assert!(completed[0].error.is_some());
    assert!(!engine.has_pending_work());
}

/// **Integration Test: Metrics Collection**
///
/// Tests that metrics are collected correctly.
///
/// _Requirements: 6.4_
#[test]
fn test_metrics_collection() {
    let config = create_test_config();
    let mut engine = InferenceEngine::new(config).unwrap();

    // Get initial metrics
    let metrics = engine.get_metrics();
    assert_eq!(metrics.total_requests, 0);
    assert_eq!(metrics.completed_requests, 0);
    assert!(metrics.memory_utilization >= 0.0);
    assert!(metrics.memory_utilization <= 1.0);

    // Submit request and check metrics update
    let params = GenerationParams {
        max_tokens: 5,
        ..GenerationParams::default()
    };

    engine.submit_request("Test", params).unwrap();
    let after_submit = engine.get_metrics();
    assert_eq!(after_submit.total_requests, 1, "submission must be counted");

    let _ = engine.step();
    let metrics_after = engine.get_metrics();
    assert_eq!(
        metrics_after.active_sequences, 1,
        "the in-flight request must be tracked as an active sequence"
    );

    // Run to completion and verify terminal metrics
    let _ = engine.run();
    let final_metrics = engine.get_metrics();
    assert_eq!(final_metrics.completed_requests, 1);
    assert_eq!(final_metrics.failed_requests, 0);
    assert_eq!(final_metrics.active_sequences, 0);
    assert!(
        final_metrics.total_tokens_generated > 0,
        "generated tokens must be counted"
    );
}
