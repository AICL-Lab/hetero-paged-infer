//! Custom configuration example for Hetero-Paged-Infer
//!
//! This example demonstrates using custom configuration.
//!
//! Run with: cargo run --example custom_config

use hetero_infer::{EngineConfig, GenerationParams, InferenceEngine, SpecialTokenIds};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("=== Hetero-Paged-Infer Custom Configuration Example ===\n");

    // Example 1: Create config with custom parameters
    println!("1. Creating config with custom parameters:");
    let custom_config = EngineConfig {
        block_size: 32,       // Larger blocks for better throughput
        max_num_blocks: 2048, // More memory for larger models
        max_batch_size: 64,   // Larger batches
        max_num_seqs: 512,    // More concurrent sequences
        max_model_len: 4096,  // Longer context
        max_total_tokens: 8192,
        memory_threshold: 0.85,
        max_retry_attempts: 3, // More retries on GPU timeout
        special_tokens: SpecialTokenIds::default(),
    };

    println!("  Block size: {}", custom_config.block_size);
    println!("  Max batch size: {}", custom_config.max_batch_size);
    println!("  Max retry attempts: {}", custom_config.max_retry_attempts);

    // Validate configuration
    custom_config.validate()?;
    println!("  Configuration is valid!\n");

    // Example 2: Create config with custom special tokens
    println!("2. Creating config with custom special tokens:");
    let special_tokens = SpecialTokenIds::new(
        1, // BOS token ID
        2, // EOS token ID
        0, // PAD token ID
        3, // UNK token ID
    );

    let _config_with_tokens = EngineConfig {
        special_tokens: special_tokens.clone(),
        ..EngineConfig::default()
    };

    println!("  BOS token ID: {}", special_tokens.bos);
    println!("  EOS token ID: {}", special_tokens.eos);
    println!("  PAD token ID: {}", special_tokens.pad);
    println!("  UNK token ID: {}", special_tokens.unk);

    // Example 3: Save and load config from file
    println!("\n3. Saving and loading config from file:");
    let config_path = Path::new("/tmp/hetero_infer_config.json");

    // Save config
    custom_config.to_file(config_path)?;
    println!("  Config saved to: {}", config_path.display());

    // Load config
    let loaded_config = EngineConfig::from_file(config_path)?;
    println!("  Config loaded successfully!");
    println!("  Loaded block size: {}", loaded_config.block_size);

    // Example 4: Use config with engine
    println!("\n4. Creating engine with custom config:");
    let mut engine = InferenceEngine::new(loaded_config)?;

    let params = GenerationParams {
        max_tokens: 5,
        temperature: 0.8,
        top_p: 0.95,
    };

    engine.submit_request("Test input", params)?;
    engine.set_max_steps(50);

    let completed = engine.run();
    println!("  Completed {} requests", completed.len());

    // Cleanup
    std::fs::remove_file(config_path).ok();

    Ok(())
}
