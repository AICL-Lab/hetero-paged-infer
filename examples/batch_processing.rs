//! Batch processing example for Hetero-Paged-Infer
//!
//! This example demonstrates processing multiple requests concurrently.
//!
//! Run with: cargo run --example batch_processing

use hetero_infer::{EngineConfig, GenerationParams, InferenceEngine};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("=== Hetero-Paged-Infer Batch Processing Example ===\n");

    // Create engine with larger capacity for batch processing
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

    let mut engine = InferenceEngine::new(config)?;

    // Generation parameters
    let params = GenerationParams {
        max_tokens: 10,
        temperature: 1.0,
        top_p: 0.9,
    };

    // Submit multiple requests
    let inputs = [
        "Hello",
        "World",
        "Rust",
        "Machine Learning",
        "Inference Engine",
    ];

    println!("Submitting {} requests...\n", inputs.len());

    for (i, input) in inputs.iter().enumerate() {
        let request_id = engine.submit_request(input, params)?;
        println!("  Request {}: '{}' -> ID {}", i + 1, input, request_id);
    }

    // Set max steps
    engine.set_max_steps(200);

    // Run inference
    println!("\nRunning batch inference...");
    let completed = engine.run();

    // Print results
    println!("\n=== Results ===");
    let mut success_count = 0;
    let mut fail_count = 0;

    for result in completed {
        if result.success {
            success_count += 1;
            println!(
                "Request {}: {} tokens",
                result.request_id,
                result.output_tokens.len()
            );
        } else {
            fail_count += 1;
            println!("Request {}: FAILED - {:?}", result.request_id, result.error);
        }
    }

    println!("\n=== Summary ===");
    println!("Successful: {}", success_count);
    println!("Failed: {}", fail_count);

    // Get metrics
    let metrics = engine.get_metrics();
    println!("\n=== Metrics ===");
    println!("Total tokens generated: {}", metrics.total_tokens_generated);
    println!(
        "Memory utilization: {:.2}%",
        metrics.memory_utilization * 100.0
    );
    println!("Active sequences: {}", metrics.active_sequences);

    Ok(())
}
