//! 核心类型和数据结构
//!
//! 按子系统组织：
//! - [`request`] — 请求与生成参数
//! - [`scheduler`] — 序列与调度输出
//! - [`execution`] — GPU 执行批次与输出
//! - [`memory`] — 内存统计与块引用

/// 请求唯一标识符
pub type RequestId = u64;

/// 序列唯一标识符
pub type SeqId = u64;

/// Token ID 类型
pub type TokenId = u32;

/// 物理块索引
pub type BlockIdx = u32;

/// 请求状态
///
/// 表示请求在推理流水线中的当前状态。
///
/// # 状态转换
///
/// ```text
/// Pending → Prefill → Decode → Completed
///                     ↘ Failed
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum RequestState {
    /// 等待调度
    Pending,

    /// Prefill 阶段（处理输入 tokens）
    Prefill,

    /// Decode 阶段（生成 tokens）
    Decode,

    /// 成功完成
    Completed,

    /// 失败，包含错误信息
    Failed(String),
}

pub mod execution;
pub mod memory;
pub mod request;
pub mod scheduler;

pub use execution::*;
pub use memory::*;
pub use request::*;
pub use scheduler::*;
