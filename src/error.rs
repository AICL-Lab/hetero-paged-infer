//! 错误类型定义
//!
//! 本模块定义了推理系统的分层错误类型体系：
//!
//! ```text
//! EngineError (顶层)
//!   ├── ConfigError      (配置错误)
//!   ├── ValidationError  (验证错误)
//!   ├── MemoryError      (内存错误)
//!   ├── ExecutionError   (执行错误)
//!   └── SchedulerError   (调度错误)
//! ```
//!
//! # 错误处理策略
//!
//! | 错误类型 | 恢复策略 |
//! |----------|----------|
//! | `MemoryError::OutOfBlocks` | 等待序列完成释放内存 |
//! | `ExecutionError::GpuTimeout` | 重试最多 2 次 |
//! | `ExecutionError::CudaError` | 跳过当前序列 |
//! | `ValidationError` | 直接返回错误 |
//!
//! # 示例
//!
//! ```rust
//! use hetero_infer::{EngineError, ValidationError, GenerationParams};
//!
//! let params = GenerationParams {
//!     max_tokens: 0,
//!     temperature: 1.0,
//!     top_p: 1.0,
//! };
//!
//! match params.validate() {
//!     Ok(()) => println!("参数有效"),
//!     Err(ValidationError::InvalidMaxTokens(0)) => println!("max_tokens 无效"),
//!     Err(e) => println!("其他错误: {}", e),
//! }
//! ```

use thiserror::Error;

/// 内存相关错误
///
/// 表示 KV Cache 内存管理过程中发生的错误。
#[derive(Error, Debug, Clone, PartialEq)]
pub enum MemoryError {
    /// 物理块耗尽：没有可用的空闲块
    #[error("物理块耗尽：没有可用的空闲块")]
    OutOfBlocks,

    /// 序列不存在：{0}
    #[error("序列不存在: {0}")]
    SequenceNotFound(u64),

    /// 块分配失败：{0}
    #[error("块分配失败: {0}")]
    AllocationFailed(String),

    /// 无效的块索引：{0}
    #[error("无效的块索引: {0}")]
    InvalidBlockIndex(u32),
}

/// 配置相关错误
///
/// 表示配置参数验证或加载过程中发生的错误。
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ConfigError {
    /// 无效的 block_size：必须 > 0，实际值为 {0}
    #[error("无效的 block_size: 必须大于 0，实际值为 {0}")]
    InvalidBlockSize(u32),

    /// 无效的 max_num_blocks：必须 > 0，实际值为 {0}
    #[error("无效的 max_num_blocks: 必须大于 0，实际值为 {0}")]
    InvalidMaxNumBlocks(u32),

    /// 无效的 max_batch_size：必须 > 0，实际值为 {0}
    #[error("无效的 max_batch_size: 必须大于 0，实际值为 {0}")]
    InvalidMaxBatchSize(u32),

    /// 无效的 max_num_seqs：必须 > 0，实际值为 {0}
    #[error("无效的 max_num_seqs: 必须大于 0，实际值为 {0}")]
    InvalidMaxNumSeqs(u32),

    /// 无效的 max_model_len：必须 > 0，实际值为 {0}
    #[error("无效的 max_model_len: 必须大于 0，实际值为 {0}")]
    InvalidMaxModelLen(u32),

    /// 无效的 max_total_tokens：必须 > 0，实际值为 {0}
    #[error("无效的 max_total_tokens: 必须大于 0，实际值为 {0}")]
    InvalidMaxTotalTokens(u32),

    /// 无效的 memory_threshold：必须在 (0.0, 1.0] 范围内，实际值为 {0}
    #[error("无效的 memory_threshold: 必须在 (0.0, 1.0] 范围内，实际值为 {0}")]
    InvalidMemoryThreshold(f32),

    /// 加载配置文件失败：{0}
    #[error("加载配置文件失败: {0}")]
    FileLoadError(String),

    /// 保存配置文件失败：{0}
    #[error("保存配置文件失败: {0}")]
    FileSaveError(String),

    /// 解析配置失败：{0}
    #[error("解析配置失败: {0}")]
    ParseError(String),

    /// 缺少 HuggingFace tokenizer 路径
    #[error("缺少 HuggingFace tokenizer 路径")]
    MissingTokenizerPath,

    /// 无效的命令桥接程序
    #[error("无效的命令桥接程序: program 不能为空")]
    InvalidCommandProgram,

    /// 无效的服务端口：{0}
    #[error("无效的服务端口: 必须大于 0，实际值为 {0}")]
    InvalidServerPort(u16),

    /// 无效的模型名称
    #[error("无效的模型名称: 不能为空")]
    InvalidModelName,
}

/// 请求验证错误
///
/// 表示请求参数验证过程中发生的错误。
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ValidationError {
    /// 无效的 max_tokens：必须 > 0，实际值为 {0}
    #[error("无效的 max_tokens: 必须大于 0，实际值为 {0}")]
    InvalidMaxTokens(u32),

    /// 无效的 temperature：必须在 (0.0, 2.0] 范围内，实际值为 {0}
    #[error("无效的 temperature: 必须在 (0.0, 2.0] 范围内，实际值为 {0}")]
    InvalidTemperature(f32),

    /// 无效的 top_p：必须在 (0.0, 1.0] 范围内，实际值为 {0}
    #[error("无效的 top_p: 必须在 (0.0, 1.0] 范围内，实际值为 {0}")]
    InvalidTopP(f32),

    /// 输入文本为空
    #[error("输入文本为空")]
    EmptyInput,

    /// 输入超出最大模型长度：{0} > {1}
    #[error("输入超出最大模型长度: {0} > {1}")]
    InputTooLong(usize, u32),

    /// 请求总长度超出最大模型长度：{0} > {1}
    #[error("请求总长度超出最大模型长度: {0} > {1}")]
    TotalLengthTooLong(usize, u32),
}

/// GPU 执行错误
///
/// 表示 GPU 执行过程中发生的错误。
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ExecutionError {
    /// CUDA 错误：{0}
    #[error("CUDA 错误: {0}")]
    CudaError(String),

    /// GPU 超时
    #[error("GPU 超时")]
    GpuTimeout,

    /// 无效输出：检测到 NaN 或 Inf
    #[error("无效输出: 检测到 NaN 或 Inf")]
    InvalidOutput,

    /// Kernel 启动失败：{0}
    #[error("Kernel 启动失败: {0}")]
    KernelLaunchFailed(String),
}

/// 调度器错误
///
/// 表示调度过程中发生的错误。
#[derive(Error, Debug, Clone, PartialEq)]
pub enum SchedulerError {
    /// 内存压力：无法接受新的 prefill 请求
    #[error("内存压力: 无法接受新的 prefill 请求")]
    MemoryPressure,

    /// 请求不存在：{0}
    #[error("请求不存在: {0}")]
    RequestNotFound(u64),

    /// 无效的状态转换：{0}
    #[error("无效的状态转换: {0}")]
    InvalidStateTransition(String),
}

/// 顶层引擎错误
///
/// 包含所有可能的错误类型，作为库的主要错误返回类型。
///
/// # 示例
///
/// ```rust
/// use hetero_infer::{EngineError, ValidationError, ConfigError};
///
/// fn handle_error(error: EngineError) {
///     match error {
///         EngineError::Config(e) => eprintln!("配置错误: {}", e),
///         EngineError::Validation(e) => eprintln!("验证错误: {}", e),
///         EngineError::Memory(e) => eprintln!("内存错误: {}", e),
///         EngineError::Execution(e) => eprintln!("执行错误: {}", e),
///         EngineError::Scheduler(e) => eprintln!("调度错误: {}", e),
///         EngineError::Tokenization(msg) => eprintln!("分词错误: {}", msg),
///     }
/// }
/// ```
#[derive(Error, Debug)]
pub enum EngineError {
    /// 配置错误
    #[error("配置错误: {0}")]
    Config(#[from] ConfigError),

    /// 验证错误
    #[error("验证错误: {0}")]
    Validation(#[from] ValidationError),

    /// 内存错误
    #[error("内存错误: {0}")]
    Memory(#[from] MemoryError),

    /// 执行错误
    #[error("执行错误: {0}")]
    Execution(#[from] ExecutionError),

    /// 调度错误
    #[error("调度错误: {0}")]
    Scheduler(#[from] SchedulerError),

    /// 分词错误
    #[error("分词错误: {0}")]
    Tokenization(String),
}

/// 库级别的 Result 类型别名
pub type Result<T> = std::result::Result<T, EngineError>;
