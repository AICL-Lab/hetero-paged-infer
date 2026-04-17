# Quick Start Guide

This guide will get you started with Hetero-Paged-Infer in under 5 minutes.

## Prerequisites

- Rust 1.70+ (2021 edition)
- Linux environment (Ubuntu 20.04+ recommended)
- NVIDIA GPU with CUDA 11.x+ (optional)

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/LessUp/hetero-paged-infer.git
cd hetero-paged-infer
```

### 2. Build

```bash
# Build release version
cargo build --release

# Or build with all features
cargo build --release --all-features
```

### 3. Run Tests

```bash
# Run all tests
cargo test

# Run with detailed output
cargo test -- --nocapture
```

## Your First Inference

### CLI Usage

```bash
# Simple inference
./target/release/hetero-infer \
  --input "Hello, world!" \
  --max-tokens 50

# With custom parameters
./target/release/hetero-infer \
  --input "Explain machine learning" \
  --max-tokens 200 \
  --temperature 0.8 \
  --top-p 0.95
```

### Expected Output

```
Heterogeneous Inference System
==============================
Configuration:
  Block size: 16
  Max blocks: 1024
  Max batch size: 32
  Memory threshold: 0.9

Input: Hello, world!
Generating up to 50 tokens...

Output: Hello, world! This is a demonstration output from the
heterogeneous inference system showcasing PagedAttention memory
management and continuous batching capabilities...

Tokens generated: 23
```

## Library Usage

### Basic Example

```rust
use hetero_infer::{EngineConfig, GenerationParams, InferenceEngine};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create engine with default configuration
    let config = EngineConfig::default();
    let mut engine = InferenceEngine::new(config)?;
    
    // Configure generation parameters
    let params = GenerationParams {
        max_tokens: 100,
        temperature: 0.8,
        top_p: 0.95,
    };
    
    // Submit request
    let request_id = engine.submit_request(
        "Explain quantum computing in simple terms",
        params
    )?;
    
    println!("Request {} submitted", request_id);
    
    // Run inference
    let completed = engine.run();
    
    // Process results
    for result in completed {
        println!("Output: {}", result.output_text);
        println!("Tokens generated: {}", result.generated_tokens);
    }
    
    Ok(())
}
```

### Save as `example.rs`

Create the example and run it:

```bash
# Create example directory
mkdir -p examples
cat > examples/basic.rs << 'EOF'
use hetero_infer::{EngineConfig, GenerationParams, InferenceEngine};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = EngineConfig::default();
    let mut engine = InferenceEngine::new(config)?;
    
    let params = GenerationParams {
        max_tokens: 50,
        temperature: 1.0,
        top_p: 0.9,
    };
    
    let _id = engine.submit_request("Hello, world!", params)?;
    let results = engine.run();
    
    for r in results {
        println!("{}", r.output_text);
    }
    
    Ok(())
}
EOF

# Run example
cargo run --example basic
```

## Configuration

### Configuration File

Create `config.json`:

```json
{
  "block_size": 16,
  "max_num_blocks": 1024,
  "max_batch_size": 32,
  "max_num_seqs": 256,
  "max_model_len": 2048,
  "max_total_tokens": 4096,
  "memory_threshold": 0.9
}
```

Use it:

```bash
./hetero-infer --config config.json --input "Hello"
```

### Environment Variables

```bash
# Log level
export RUST_LOG=info

# Backtrace on error
export RUST_BACKTRACE=1

# Run with settings
cargo run --release -- --input "Test"
```

## Common Tasks

### View Help

```bash
# General help
./hetero-infer --help

# Help for specific options
./hetero-infer -h
```

### Multiple Requests

The engine supports continuous batching automatically:

```rust
// Submit multiple requests
let id1 = engine.submit_request("First prompt", params.clone())?;
let id2 = engine.submit_request("Second prompt", params.clone())?;
let id3 = engine.submit_request("Third prompt", params)?;

// Process all
let results = engine.run();
```

### Memory Monitoring

```rust
// Check memory stats
let stats = engine.get_memory_stats();
println!("Memory usage: {}/{} blocks", 
    stats.used_blocks, stats.total_blocks);
```

## Troubleshooting

### Build Issues

```
error: linker not found
```
**Solution:** Install build-essential
```bash
sudo apt-get install build-essential
```

### Runtime Issues

```
OutOfBlocks error
```
**Solution:** Reduce `--max-tokens` or increase `--max-num-blocks`

### Performance Issues

Enable debug logging:
```bash
RUST_LOG=debug ./hetero-infer --input "Test"
```

## Next Steps

- [Installation Guide](installation.md) - Detailed setup instructions
- [Configuration](configuration.md) - All configuration options
- [API Reference](../api/core-types.md) - Complete API documentation

---

Need help? Check the [FAQ](../development/faq.md) or [open an issue](https://github.com/LessUp/hetero-paged-infer/issues).
