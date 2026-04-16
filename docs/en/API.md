# API Reference

## Overview

Hetero-Paged-Infer provides a Rust API for integrating the inference engine into your applications. This guide covers the core types, traits, and usage patterns.

## Core Types

### InferenceEngine

The main entry point for inference operations.

```rust
use hetero_infer::{EngineConfig, GenerationParams, InferenceEngine};

// Create engine with default configuration
let config = EngineConfig::default();
let mut engine = InferenceEngine::new(config)?;

// Submit a request
let params = GenerationParams {
    max_tokens: 100,
    temperature: 1.0,
    top_p: 0.9,
};
let request_id = engine.submit_request("Hello, world!", params)?;

// Run inference
let completed = engine.run();

// Process results
for result in completed {
    println!("Output: {}", result.output_text);
}
```

### EngineConfig

Configuration for the inference engine.

```rust
pub struct EngineConfig {
    pub block_size: u32,          // Tokens per physical block (default: 16)
    pub max_num_blocks: u32,      // Total physical blocks (default: 1024)
    pub max_batch_size: u32,      // Max sequences per batch (default: 32)
    pub max_num_seqs: u32,        // Max concurrent sequences (default: 256)
    pub max_model_len: u32,       // Max sequence length (default: 2048)
    pub max_total_tokens: u32,    // Max tokens per batch (default: 4096)
    pub memory_threshold: f32,    // Memory pressure threshold (default: 0.9)
}
```

**Usage**:

```rust
// Default configuration
let config = EngineConfig::default();

// Custom configuration
let config = EngineConfig {
    block_size: 32,
    max_num_blocks: 2048,
    max_batch_size: 64,
    ..Default::default()
};

// Validate configuration
config.validate()?;

// Load from file
let config = EngineConfig::from_file("config.json")?;
```

### GenerationParams

Parameters controlling text generation.

```rust
pub struct GenerationParams {
    pub max_tokens: u32,      // Maximum tokens to generate
    pub temperature: f32,     // Sampling temperature (0.0 - 2.0)
    pub top_p: f32,          // Nucleus sampling threshold (0.0 - 1.0)
}
```

**Defaults**:

```rust
impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 100,
            temperature: 1.0,
            top_p: 0.9,
        }
    }
}
```

### Request

Represents an inference request.

```rust
pub struct Request {
    pub id: u64,
    pub input_tokens: Vec<u32>,
    pub output_tokens: Vec<u32>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub state: RequestState,
    pub created_at: Instant,
}

pub enum RequestState {
    Pending,
    Prefill,
    Decode,
    Completed,
    Failed(String),
}
```

### Sequence

Active request with KV Cache allocation.

```rust
pub struct Sequence {
    pub seq_id: u64,
    pub request: Request,
    pub logical_blocks: Vec<LogicalBlock>,
    pub num_computed_tokens: u32,
    pub num_generated_tokens: u32,
}
```

## Trait Interfaces

### TokenizerTrait

```rust
pub trait TokenizerTrait: Send + Sync {
    /// Encode text to token IDs
    fn encode(&self, text: &str) -> Vec<u32>;
    
    /// Decode token IDs to text
    fn decode(&self, tokens: &[u32]) -> String;
    
    /// Vocabulary size
    fn vocab_size(&self) -> u32;
    
    /// Special token IDs
    fn bos_token_id(&self) -> u32;
    fn eos_token_id(&self) -> u32;
    fn pad_token_id(&self) -> u32;
}
```

**SimpleTokenizer** (built-in implementation):

```rust
use hetero_infer::SimpleTokenizer;

let tokenizer = SimpleTokenizer::new();
let tokens = tokenizer.encode("Hello");
let text = tokenizer.decode(&tokens);
```

### SchedulerTrait

```rust
pub trait SchedulerTrait: Send + Sync {
    /// Add a new request to the pending queue
    fn add_request(&mut self, request: Request) -> Result<u64, SchedulerError>;
    
    /// Schedule the next batch for execution
    fn schedule(&mut self) -> SchedulerOutput;
    
    /// Update sequence states after GPU execution
    fn update_sequences(&mut self, outputs: &ExecutionOutput);
    
    /// Get completed requests
    fn get_completed(&mut self) -> Vec<Request>;
    
    /// Check if there's pending work
    fn has_pending_work(&self) -> bool;
}
```

### KVCacheManagerTrait

```rust
pub trait KVCacheManagerTrait: Send + Sync {
    /// Allocate blocks for a new sequence
    fn allocate_sequence(&mut self, seq_id: u64, num_tokens: u32) -> Result<(), MemoryError>;
    
    /// Allocate an additional block for a growing sequence
    fn allocate_block(&mut self, seq_id: u64) -> Result<PhysicalBlockRef, MemoryError>;
    
    /// Release all blocks for a sequence
    fn free_sequence(&mut self, seq_id: u64);
    
    /// Get block table for GPU execution
    fn get_block_table(&self, seq_id: u64) -> Option<Vec<u32>>;
    
    /// Get memory statistics
    fn get_memory_stats(&self) -> MemoryStats;
    
    /// Check if n blocks can be allocated
    fn can_allocate(&self, num_blocks: u32) -> bool;
}

pub struct MemoryStats {
    pub total_blocks: u32,
    pub used_blocks: u32,
    pub free_blocks: u32,
    pub num_sequences: u32,
}
```

### GPUExecutorTrait

```rust
pub trait GPUExecutorTrait: Send + Sync {
    /// Execute a batch of sequences
    fn execute(&mut self, batch: &ExecutionBatch) -> ExecutionOutput;
    
    /// Capture CUDA Graph for decode phase
    fn capture_decode_graph(&mut self, batch_size: u32);
    
    /// Execute using captured graph
    fn execute_graph(&mut self, batch: &ExecutionBatch) -> ExecutionOutput;
}

pub struct ExecutionBatch {
    pub input_tokens: Vec<u32>,
    pub positions: Vec<u32>,
    pub seq_lens: Vec<u32>,
    pub block_tables: Vec<Vec<u32>>,
    pub is_prefill: Vec<bool>,
}

pub struct ExecutionOutput {
    pub next_tokens: Vec<u32>,
    pub logits: Option<Vec<f32>>,
}
```

## Error Handling

The API uses a structured error type:

```rust
use hetero_infer::EngineError;

match result {
    Ok(output) => println!("Success: {}", output),
    Err(EngineError::Config(e)) => eprintln!("Config error: {}", e),
    Err(EngineError::Memory(e)) => eprintln!("Memory error: {}", e),
    Err(EngineError::Validation(e)) => eprintln!("Validation error: {}", e),
    Err(EngineError::Execution(e)) => eprintln!("Execution error: {}", e),
    Err(EngineError::Scheduler(e)) => eprintln!("Scheduler error: {}", e),
}
```

### Error Types

| Error | Description | Typical Cause |
|-------|-------------|---------------|
| `ConfigError` | Invalid configuration | Negative values, zero block_size |
| `MemoryError` | Memory allocation failure | Out of blocks, GPU OOM |
| `ValidationError` | Invalid request parameters | Invalid temperature, top_p |
| `ExecutionError` | GPU execution failure | CUDA error, timeout |
| `SchedulerError` | Scheduling failure | Invalid state transition |

## Usage Examples

### Basic Inference

```rust
use hetero_infer::{EngineConfig, GenerationParams, InferenceEngine};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create engine
    let config = EngineConfig::default();
    let mut engine = InferenceEngine::new(config)?;
    
    // Submit request
    let params = GenerationParams {
        max_tokens: 50,
        temperature: 0.8,
        top_p: 0.95,
    };
    let request_id = engine.submit_request("Hello, world!", params)?;
    println!("Request {} submitted", request_id);
    
    // Run inference
    let completed = engine.run();
    
    // Get results
    for result in completed {
        println!("Request {} completed:", result.request_id);
        println!("  Output: {}", result.output_text);
        println!("  Tokens generated: {}", result.generated_tokens);
    }
    
    Ok(())
}
```

### Step-by-Step Execution

For more control over the inference loop:

```rust
// Submit requests
let id1 = engine.submit_request("First request", params.clone())?;
let id2 = engine.submit_request("Second request", params)?;

// Execute step by step
while engine.has_pending_work() {
    let completed = engine.step();
    
    for result in &completed {
        println!("Request {} completed", result.request_id);
    }
}
```

### Custom Tokenizer

```rust
use hetero_infer::TokenizerTrait;

struct MyTokenizer {
    vocab: HashMap<String, u32>,
}

impl TokenizerTrait for MyTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        // Custom encoding logic
        vec![]
    }
    
    fn decode(&self, tokens: &[u32]) -> String {
        // Custom decoding logic
        String::new()
    }
    
    fn vocab_size(&self) -> u32 {
        self.vocab.len() as u32
    }
    
    fn bos_token_id(&self) -> u32 { 0 }
    fn eos_token_id(&self) -> u32 { 1 }
    fn pad_token_id(&self) -> u32 { 2 }
}
```

### Memory Monitoring

```rust
// Get memory statistics
let stats = engine.get_memory_stats();
println!("Memory usage: {}/{} blocks ({}%)", 
    stats.used_blocks, 
    stats.total_blocks,
    (stats.used_blocks as f32 / stats.total_blocks as f32) * 100.0
);
```

## Type Exports

Main exports from `lib.rs`:

```rust
pub use crate::config::EngineConfig;
pub use crate::engine::{InferenceEngine, EngineMetrics, CompletedRequest};
pub use crate::error::EngineError;
pub use crate::types::{Request, Sequence, GenerationParams, RequestState};
pub use crate::kv_cache::{KVCacheManager, KVCacheManagerTrait, MemoryStats};
pub use crate::scheduler::{Scheduler, SchedulerTrait, SchedulerOutput};
pub use crate::tokenizer::{SimpleTokenizer, TokenizerTrait};
pub use crate::gpu_executor::{GPUExecutorTrait, ExecutionBatch, ExecutionOutput};
```

---

*For configuration details, see [CONFIGURATION.md](./CONFIGURATION.md).*
