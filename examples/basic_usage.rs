//! Basic usage example for Hetero-Paged-Infer
//!
//! This example demonstrates the basic usage of the inference engine.
//!
//! Run with: cargo run --example basic_usage

use hetero_infer::{EngineConfig, GenerationParams, InferenceEngine};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger
    env_logger::init();

    println!("=== Hetero-Paged-Infer Basic Usage Example ===\n");

    // Create engine with default configuration
    let config = EngineConfig::default();
    println!("Creating engine with config:");
    println!("  Block size: {}", config.block_size);
    println!("  Max blocks: {}", config.max_num_blocks);
    println!("  Max batch size: {}", config.max_batch_size);
    println!("  Max model length: {}", config.max_model_len);
    println!();

    let mut engine = InferenceEngine::new(config)?;

    // Create generation parameters
    let params = GenerationParams {
        max_tokens: 20,
        temperature: 1.0,
        top_p: 0.9,
    };

    // Submit a single request
    let input_text = "Hello, world!";
    println!("Input: {}", input_text);

    let request_id = engine.submit_request(input_text, params)?;
    println!("Submitted request with ID: {}", request_id);

    // Set maximum steps to prevent infinite loop
    engine.set_max_steps(100);

    // Run inference
    println!("\nRunning inference...");
    let completed = engine.run();

    // Print results
    println!("\n=== Results ===");
    for result in completed {
        if result.success {
            println!("Request {}: SUCCESS", result.request_id);
            println!("  Output: {}", result.output_text);
            println!("  Tokens generated: {}", result.output_tokens.len());
        } else {
            println!("Request {}: FAILED", result.request_id);
            println!("  Error: {:?}", result.error);
        }
    }

    // Get final metrics
    let metrics = engine.get_metrics();
    println!("\n=== Metrics ===");
    println!("Total requests: {}", metrics.total_requests);
    println!("Completed: {}", metrics.completed_requests);
    println!("Failed: {}", metrics.failed_requests);
    println!("Total tokens generated: {}", metrics.total_tokens_generated);
    println!(
        "Memory utilization: {:.2}%",
        metrics.memory_utilization * 100.0
    );

    Ok(())
}
