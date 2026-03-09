//! Error types for the inference system

use thiserror::Error;

/// Memory-related errors
#[derive(Error, Debug, Clone, PartialEq)]
pub enum MemoryError {
    #[error("Out of blocks: no free physical blocks available")]
    OutOfBlocks,
    
    #[error("Sequence not found: {0}")]
    SequenceNotFound(u64),
    
    #[error("Block allocation failed: {0}")]
    AllocationFailed(String),
    
    #[error("Invalid block index: {0}")]
    InvalidBlockIndex(u32),
}

/// Configuration errors
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ConfigError {
    #[error("Invalid block_size: must be > 0, got {0}")]
    InvalidBlockSize(u32),
    
    #[error("Invalid max_num_blocks: must be > 0, got {0}")]
    InvalidMaxNumBlocks(u32),
    
    #[error("Invalid max_batch_size: must be > 0, got {0}")]
    InvalidMaxBatchSize(u32),
    
    #[error("Invalid max_num_seqs: must be > 0, got {0}")]
    InvalidMaxNumSeqs(u32),
    
    #[error("Invalid max_model_len: must be > 0, got {0}")]
    InvalidMaxModelLen(u32),
    
    #[error("Invalid max_total_tokens: must be > 0, got {0}")]
    InvalidMaxTotalTokens(u32),
    
    #[error("Invalid memory_threshold: must be in (0.0, 1.0], got {0}")]
    InvalidMemoryThreshold(f32),
    
    #[error("Failed to load config file: {0}")]
    FileLoadError(String),
    
    #[error("Failed to save config file: {0}")]
    FileSaveError(String),
    
    #[error("Failed to parse config: {0}")]
    ParseError(String),
}

/// Request validation errors
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ValidationError {
    #[error("Invalid max_tokens: must be > 0, got {0}")]
    InvalidMaxTokens(u32),
    
    #[error("Invalid temperature: must be in (0.0, 2.0], got {0}")]
    InvalidTemperature(f32),
    
    #[error("Invalid top_p: must be in (0.0, 1.0], got {0}")]
    InvalidTopP(f32),
    
    #[error("Empty input text")]
    EmptyInput,
    
    #[error("Input exceeds max model length: {0} > {1}")]
    InputTooLong(usize, u32),
}

/// Execution errors
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ExecutionError {
    #[error("CUDA error: {0}")]
    CudaError(String),
    
    #[error("GPU timeout")]
    GpuTimeout,
    
    #[error("Invalid output: NaN or Inf detected")]
    InvalidOutput,
    
    #[error("Kernel launch failed: {0}")]
    KernelLaunchFailed(String),
}

/// Scheduler errors
#[derive(Error, Debug, Clone, PartialEq)]
pub enum SchedulerError {
    #[error("Memory pressure: cannot accept new prefill requests")]
    MemoryPressure,
    
    #[error("Request not found: {0}")]
    RequestNotFound(u64),
    
    #[error("Invalid state transition: {0}")]
    InvalidStateTransition(String),
}

/// Top-level engine error
#[derive(Error, Debug)]
pub enum EngineError {
    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),
    
    #[error("Validation error: {0}")]
    Validation(#[from] ValidationError),
    
    #[error("Memory error: {0}")]
    Memory(#[from] MemoryError),
    
    #[error("Execution error: {0}")]
    Execution(#[from] ExecutionError),
    
    #[error("Scheduler error: {0}")]
    Scheduler(#[from] SchedulerError),
    
    #[error("Tokenization error: {0}")]
    Tokenization(String),
}

pub type Result<T> = std::result::Result<T, EngineError>;
