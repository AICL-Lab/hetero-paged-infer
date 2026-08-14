//! tiny-llm 执行后端 FFI 桥接（对接骨架）
//!
//! 定义 paged-infer（Rust）与 [tiny-llm](C++/CUDA 引擎) 之间的 C ABI 契约。
//! 本模块是**骨架**：只声明接口形态与数据布局，不链接真实 C 库。
//!
//! # 为什么需要 FFI
//!
//! tiny-llm 是 C++/CUDA 项目，其推理入口是整段 `InferenceEngine::generate()`，
//! 而 paged-infer 的引擎需要"每步执行一个 batch"（[`GPUExecutorTrait`]）。
//! 因此 tiny-llm 侧必须导出步进式 C ABI，paged-infer 侧经本模块调用。
//!
//! # C ABI 契约（tiny-llm 侧待实现）
//!
//! ```c
//! typedef struct TinyLlmHandle TinyLlmHandle;
//!
//! // 加载 GGUF 模型；成功返回句柄，失败返回 NULL 并写入 err_buf。
//! TinyLlmHandle* tinyllm_load(const char* model_path,
//!                             const TinyLlmConfig* config,
//!                             char* err_buf, int err_buf_len);
//!
//! // 单步执行一个 batch（prefill/decode 混合），逐序列输出下一 token 与 logprobs。
//! // 返回 0 成功，非 0 错误码。
//! int tinyllm_step(TinyLlmHandle* handle,
//!                  const int* input_tokens, const int* positions,
//!                  const int* seq_lens, const int* block_tables,
//!                  const unsigned char* is_prefill, int num_sequences,
//!                  int* next_tokens, float* logprobs, int logprobs_k);
//!
//! // KV 生命周期（分页适配后由后端管理；连续 KV 后端可为空实现）。
//! int tinyllm_allocate_sequence(TinyLlmHandle* handle, int seq_id, int num_tokens);
//! int tinyllm_free_sequence(TinyLlmHandle* handle, int seq_id);
//!
//! void tinyllm_free(TinyLlmHandle* handle);
//! ```
//!
//! # 数据布局约定
//!
//! - `input_tokens` / `positions` 是扁平化数组（`seq_lens` 描述每序列切分）；
//! - `block_tables` 是 `num_sequences × 每序列块数` 的扁平化物理块索引，
//!   对齐 paged-infer 的 [`ExecutionBatch::block_tables`]；
//! - `logprobs_k == 0` 表示不输出 logprobs；否则 `logprobs` 为
//!   `num_sequences × logprobs_k` 的 `(token_id, logprob)` 交错数组。
//!
//! # KV 适配策略（二选一，接入里程碑时确定）
//!
//! - **策略 1（推荐）**：tiny-llm 侧实现分页 KV，按 `block_tables` 间接访问，
//!   与 paged-infer 的 BlockPool 完全对齐，形成完整 PagedAttention 故事。
//! - **策略 2**：tiny-llm 保留连续 KV，`block_tables` 参数忽略；paged-infer
//!   侧用一个"连续 KV"后端包装（块表全 0），代价是失去分页共享能力。
//!
//! # 接入前置条件（里程碑）
//!
//! 1. tiny-llm 完成 GPU 端到端生成验证并与 llama.cpp 对齐（其 ROADMAP 阶段 1）；
//! 2. tiny-llm 导出本模块声明的 C ABI 符号并构建 `libtiny_llm.a`；
//! 3. 本模块启用 `tiny-llm` cargo feature，并用 build 脚本链接静态库；
//! 4. 实现 [`crate::gpu_executor::GPUExecutorTrait`] 的 tiny-llm 适配器。

/// 与 tiny-llm `TinyLlmConfig` 对齐的配置布局（`repr(C)`，字段为 C `int`）。
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TinyLlmConfig {
    pub hidden_dim: i32,
    pub num_layers: i32,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub vocab_size: i32,
    /// 分页块大小（paged-infer 侧，策略 1 时后端按此对齐）。
    pub block_size: i32,
    pub max_batch_size: i32,
}

/// 不透明句柄：tiny-llm 侧分配的模型实例。
///
/// Rust 侧只持有指针，生命周期由 `tinyllm_load` / `tinyllm_free` 管理。
#[repr(C)]
pub struct TinyLlmHandle {
    _private: [u8; 0],
}

/// 未来在 tiny-llm 侧实现的 C 符号（`#[cfg(feature = "tiny-llm")]` 时编译）。
///
/// 骨架阶段不启用该 feature，因此这些声明不会被编译。启用后仍需在
/// build.rs 中加入 `-l tiny_llm` 链接参数（真实接入里程碑）。
#[cfg(feature = "tiny-llm")]
pub mod symbols {
    use super::{TinyLlmConfig, TinyLlmHandle};
    use std::os::raw::{c_char, c_int};

    extern "C" {
        pub fn tinyllm_load(
            model_path: *const c_char,
            config: *const TinyLlmConfig,
            err_buf: *mut c_char,
            err_buf_len: c_int,
        ) -> *mut TinyLlmHandle;

        pub fn tinyllm_step(
            handle: *mut TinyLlmHandle,
            input_tokens: *const c_int,
            positions: *const c_int,
            seq_lens: *const c_int,
            block_tables: *const c_int,
            is_prefill: *const u8,
            num_sequences: c_int,
            next_tokens: *mut c_int,
            logprobs: *mut f32,
            logprobs_k: c_int,
        ) -> c_int;

        pub fn tinyllm_allocate_sequence(
            handle: *mut TinyLlmHandle,
            seq_id: c_int,
            num_tokens: c_int,
        ) -> c_int;

        pub fn tinyllm_free_sequence(handle: *mut TinyLlmHandle, seq_id: c_int) -> c_int;

        pub fn tinyllm_free(handle: *mut TinyLlmHandle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TokenId;
    use std::mem::{align_of, size_of};

    /// C ABI 布局守卫：Rust 侧 `repr(C)` 布局不得漂移，否则与 tiny-llm 侧
    /// 的 `TinyLlmConfig` 错位。字段全为 `i32`，期望 8 字段 × 4 字节。
    #[test]
    fn tiny_llm_config_layout_is_stable() {
        assert_eq!(size_of::<TinyLlmConfig>(), 8 * 4);
        assert_eq!(align_of::<TinyLlmConfig>(), 4);
    }

    /// 配置字段与注释一一对应（防重构时删改字段导致 C ABI 失配）。
    #[test]
    fn tiny_llm_config_fields_map_to_c_ints() {
        let config = TinyLlmConfig {
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 16,
            vocab_size: 128,
            block_size: 16,
            max_batch_size: 8,
        };
        assert_eq!(config.vocab_size, 128);
        assert_eq!(config.block_size, 16);
    }

    /// TokenId 与 C int 的对应关系：FFI 数组以 `i32` 传递 token id。
    #[test]
    fn token_id_is_ffi_compatible() {
        let token: TokenId = 42;
        assert_eq!(token as i32, 42);
    }
}
