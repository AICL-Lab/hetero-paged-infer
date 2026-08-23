//! tiny-llm 执行后端 FFI 桥接
//!
//! 定义 paged-infer（Rust）与 [tiny-llm](C++/CUDA 引擎) 之间的 C ABI 契约。
//! 本模块声明接口形态与数据布局；启用 `tiny-llm` feature 后由 `build.rs`
//! 链接真实静态库，`TinyLlmExecutor` 负责调用。
//!
//! # 为什么需要 FFI
//!
//! tiny-llm 是 C++/CUDA 项目，其推理入口是整段 `InferenceEngine::generate()`，
//! 而 paged-infer 的引擎需要"每步执行一个 batch"
//! （[`crate::gpu_executor::GPUExecutorTrait`]）。
//! 因此 tiny-llm 侧必须导出步进式 C ABI，paged-infer 侧经本模块调用。
//!
//! # C ABI 契约（ABI v2，tiny-llm 侧已实现）
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
//! // num_blocks[i] = 第 i 个序列的 block_tables 长度；扁平化 block_tables 的
//! // 总长为 sum(num_blocks)。策略 2 下二者忽略（传 nullptr）。
//! int tinyllm_step(TinyLlmHandle* handle,
//!                  const int* seq_ids, const int* input_tokens, const int* positions,
//!                  const int* seq_lens, const int* block_tables, const int* num_blocks,
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
//! - `seq_ids` 显式给出每个序列的 id（由 `tinyllm_allocate_sequence` 分配），
//!   支持任意 id 的序列混批；
//! - `input_tokens` / `positions` 是扁平化数组（`seq_lens` 描述每序列切分，
//!   与 `seq_ids` 对齐）；
//! - `block_tables` 是扁平化物理块索引；`num_blocks` 给出每序列块数，
//!   对齐 paged-infer 的 [`crate::types::ExecutionBatch::block_tables`]
//!   （策略 2 下忽略）；
//! - `logprobs_k == 0` 表示不输出 logprobs；否则 `logprobs` 至少容纳
//!   `num_sequences × logprobs_k × 2` 个 `f32`，第 0/1 个值分别是以 `f32`
//!   表示的 `token_id` 与 `logprob`。`logprobs_k` 必须位于 `[0, vocab_size]`，
//!   请求输出时指针不得为空。
//!
//! # KV 适配策略
//!
//! - **策略 1（默认）**：tiny-llm 侧使用分页 KV，按 `block_tables` 间接访问，
//!   与 paged-infer 的 BlockPool 对齐。
//! - **策略 2（回退）**：tiny-llm 使用连续 KV，忽略 `block_tables` /
//!   `num_blocks`；设置 `PAGED_INFER_TINY_LLM_STRATEGY=2` 启用。
//!
//! # 构建与运行前置条件
//!
//! 1. 在 tiny-llm 仓库构建 `libtiny_llm.a`；
//! 2. 设置 `TINY_LLM_DIR` 指向包含该静态库的构建目录；
//! 3. 使用 `--features tiny-llm` 构建 paged-infer，并配置真实 GGUF 模型。

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
    /// 分页 KV 池的物理块总数；0 = 策略 2（连续 KV）。
    /// 该字段使 `TinyLlmConfig` 变为 9 个 int 的 repr(C) 布局（ABI v2）。
    pub max_num_blocks: i32,
}

/// 不透明句柄：tiny-llm 侧分配的模型实例。
///
/// Rust 侧只持有指针，生命周期由 `tinyllm_load` / `tinyllm_free` 管理。
#[repr(C)]
pub struct TinyLlmHandle {
    _private: [u8; 0],
}

/// tiny-llm 导出的 C 符号（`#[cfg(feature = "tiny-llm")]` 时编译并链接）。
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
            seq_ids: *const c_int,
            input_tokens: *const c_int,
            positions: *const c_int,
            seq_lens: *const c_int,
            block_tables: *const c_int,
            num_blocks: *const c_int,
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
    /// 的 `TinyLlmConfig` 错位。字段全为 `i32`，期望 9 字段 × 4 字节（ABI v2）。
    #[test]
    fn tiny_llm_config_layout_is_stable() {
        assert_eq!(size_of::<TinyLlmConfig>(), 9 * 4);
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
            max_num_blocks: 0,
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
