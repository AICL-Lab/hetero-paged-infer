//! Heterogeneous Inference System - Main Entry Point

use clap::Parser;
use hetero_infer::{create_router, EngineConfig, GenerationParams, InferenceEngine};
use log::info;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "hetero-infer")]
#[command(
    about = "Paged-memory, continuously-batched inference engine scaffold with a mock compute backend"
)]
struct Args {
    /// Path to configuration file
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Block size (tokens per block)
    #[arg(long)]
    block_size: Option<u32>,

    /// Maximum number of blocks
    #[arg(long)]
    max_num_blocks: Option<u32>,

    /// Maximum batch size
    #[arg(long)]
    max_batch_size: Option<u32>,

    /// Maximum number of sequences
    #[arg(long)]
    max_num_seqs: Option<u32>,

    /// Maximum model length
    #[arg(long)]
    max_model_len: Option<u32>,

    /// Maximum total tokens per batch
    #[arg(long)]
    max_total_tokens: Option<u32>,

    /// Memory pressure threshold (0.0 - 1.0)
    #[arg(long)]
    memory_threshold: Option<f32>,

    /// Serve listen host (overrides config file / default 127.0.0.1)
    #[arg(long)]
    host: Option<String>,

    /// Serve listen port (overrides config file / default 3000)
    #[arg(long)]
    port: Option<u16>,

    /// Input text to process
    #[arg(short, long)]
    input: Option<String>,

    /// Start OpenAI-compatible HTTP server
    #[arg(long)]
    serve: bool,

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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args = Args::parse();

    let mut config = if let Some(config_path) = args.config {
        // 之前 --config 会静默忽略所有单项 CLI 参数，极易误配；现在显式报错。
        let has_overrides = [
            args.block_size.is_some(),
            args.max_num_blocks.is_some(),
            args.max_batch_size.is_some(),
            args.max_num_seqs.is_some(),
            args.max_model_len.is_some(),
            args.max_total_tokens.is_some(),
            args.memory_threshold.is_some(),
        ]
        .contains(&true);
        if has_overrides {
            return Err(
                "--config cannot be combined with individual engine-config flags \
                        (--block-size, --max-num-blocks, --max-batch-size, --max-num-seqs, \
                         --max-model-len, --max-total-tokens, --memory-threshold)"
                    .into(),
            );
        }
        EngineConfig::from_file(&config_path)?
    } else {
        let mut config = EngineConfig::default();
        if let Some(v) = args.block_size {
            config.block_size = v;
        }
        if let Some(v) = args.max_num_blocks {
            config.max_num_blocks = v;
        }
        if let Some(v) = args.max_batch_size {
            config.max_batch_size = v;
        }
        if let Some(v) = args.max_num_seqs {
            config.max_num_seqs = v;
        }
        if let Some(v) = args.max_model_len {
            config.max_model_len = v;
        }
        if let Some(v) = args.max_total_tokens {
            config.max_total_tokens = v;
        }
        if let Some(v) = args.memory_threshold {
            config.memory_threshold = v;
        }
        config
    };

    // serving 覆盖对两种配置来源都生效（--host/--port 可与 --config 组合）
    if let Some(host) = args.host {
        config.serving.host = host;
    }
    if let Some(port) = args.port {
        config.serving.port = port;
    }

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

    if args.serve {
        let bind_addr = format!("{}:{}", config.serving.host, config.serving.port);
        info!("Starting OpenAI-compatible server on {}", bind_addr);
        println!("Server mode: {}", bind_addr);
        println!("Model name: {}", config.serving.model_name);

        let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
        let app = create_router(config)?;
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await?;
        info!("Server shut down gracefully");
        return Ok(());
    }

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
        let (request_id, prompt_tokens) = engine.submit_request(&input_text, params)?;
        info!(
            "Submitted request: {} ({} prompt tokens)",
            request_id, prompt_tokens
        );

        // Run inference
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

/// 监听 Ctrl+C（以及 Unix 平台的 SIGTERM），触发后让服务器优雅关闭：
/// 停止接受新连接，排空在途请求。
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    info!("Shutdown signal received, draining in-flight requests");
}
