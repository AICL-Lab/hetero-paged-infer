//! Heterogeneous Inference System - Main Entry Point

use clap::Parser;
use hetero_infer::{EngineConfig, GenerationParams, InferenceEngine};
use log::info;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "hetero-infer")]
#[command(about = "Heterogeneous Inference System with CPU-GPU co-execution")]
struct Args {
    /// Path to configuration file
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Block size (tokens per block)
    #[arg(long, default_value = "16")]
    block_size: u32,

    /// Maximum number of blocks
    #[arg(long, default_value = "1024")]
    max_num_blocks: u32,

    /// Maximum batch size
    #[arg(long, default_value = "32")]
    max_batch_size: u32,

    /// Maximum number of sequences
    #[arg(long, default_value = "256")]
    max_num_seqs: u32,

    /// Maximum model length
    #[arg(long, default_value = "2048")]
    max_model_len: u32,

    /// Maximum total tokens per batch
    #[arg(long, default_value = "4096")]
    max_total_tokens: u32,

    /// Memory pressure threshold (0.0 - 1.0)
    #[arg(long, default_value = "0.9")]
    memory_threshold: f32,

    /// Input text to process
    #[arg(short, long)]
    input: Option<String>,

    /// Maximum tokens to generate
    #[arg(long, default_value = "100")]
    max_tokens: u32,

    /// Sampling temperature
    #[arg(long, default_value = "1.0")]
    temperature: f32,

    /// Top-p sampling parameter
    #[arg(long, default_value = "0.9")]
    top_p: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    let args = Args::parse();
    
    let config = if let Some(config_path) = args.config {
        EngineConfig::from_file(&config_path)?
    } else {
        EngineConfig {
            block_size: args.block_size,
            max_num_blocks: args.max_num_blocks,
            max_batch_size: args.max_batch_size,
            max_num_seqs: args.max_num_seqs,
            max_model_len: args.max_model_len,
            max_total_tokens: args.max_total_tokens,
            memory_threshold: args.memory_threshold,
        }
    };
    
    config.validate()?;
    
    info!("Starting Heterogeneous Inference System");
    info!("Configuration: {:?}", config);
    
    println!("Heterogeneous Inference System");
    println!("==============================");
    println!("Configuration:");
    println!("  Block size: {}", config.block_size);
    println!("  Max blocks: {}", config.max_num_blocks);
    println!("  Max batch size: {}", config.max_batch_size);
    println!("  Max sequences: {}", config.max_num_seqs);
    println!();
    
    // Create inference engine
    let mut engine = InferenceEngine::new(config)?;
    
    // Process input if provided
    if let Some(input_text) = args.input {
        let params = GenerationParams {
            max_tokens: args.max_tokens,
            temperature: args.temperature,
            top_p: args.top_p,
        };
        
        println!("Input: {}", input_text);
        println!("Generating up to {} tokens...", args.max_tokens);
        println!();
        
        // Submit request
        let request_id = engine.submit_request(&input_text, params)?;
        info!("Submitted request: {}", request_id);
        
        // Run inference
        engine.set_max_steps(1000);
        let completed = engine.run();
        
        // Print results
        for result in completed {
            if result.success {
                println!("Output: {}", result.output_text);
                println!("Tokens generated: {}", result.output_tokens.len());
            } else {
                println!("Error: {:?}", result.error);
            }
        }
    } else {
        println!("No input provided. Use --input to specify text to process.");
        println!();
        println!("Example:");
        println!("  hetero-infer --input \"Hello, world!\" --max-tokens 50");
    }
    
    Ok(())
}
