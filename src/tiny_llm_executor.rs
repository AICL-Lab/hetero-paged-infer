//! tiny-llm（C++/CUDA）真实执行后端适配器（里程碑 4）。
//!
//! 实现 [`GPUExecutorTrait`]：把引擎调度的 [`ExecutionBatch`] 经 C ABI
//! （[`crate::tiny_llm_ffi`]）交给 tiny-llm 步进执行。
//!
//! # 边界（策略 2：连续 KV）
//! - `block_tables` 忽略，KV 位置由 tiny-llm 内部跟踪；
//! - 每序列首次出现时分配 KV（context_len + 512 预留 token），
//!   decode 超出预留会失败（调用方应限制 `max_tokens`）；
//! - 仅支持 greedy 采样（[`ExecutorCapabilities::GREEDY_ONLY`]）；
//! - tokenizer 必须与 tiny-llm 加载的模型词表一致。
//!
//! 仅在 `tiny-llm` cargo feature 下编译，且需 `TINY_LLM_DIR` 指向
//! 已构建的 tiny-llm 静态库（见 build.rs）。

use crate::config::EngineConfig;
use crate::error::EngineError;
use crate::gpu_executor::{ExecutorCapabilities, GPUExecutorTrait};
use crate::tiny_llm_ffi::symbols;
use crate::tiny_llm_ffi::{TinyLlmConfig, TinyLlmHandle};
use crate::types::{ExecutionBatch, ExecutionOutput, TokenId};
use std::collections::HashSet;
use std::ffi::CString;
use std::os::raw::c_int;

/// decode 预留 token 数（allocate 时 context_len + 该值）。
const DECODE_RESERVE: i32 = 512;

/// tiny-llm 单 slot KV 的显存开销（0.5B 模型 max_seq_len=32768 时 ~400MB）。
/// 适配器把最大并发序列 clamp 到此值，避免 KV pool 超出显存。
const MAX_CONCURRENT_SEQS: i32 = 4;

/// tiny-llm 真实执行后端。
pub struct TinyLlmExecutor {
    handle: *mut TinyLlmHandle,
    allocated: HashSet<c_int>,
}

impl TinyLlmExecutor {
    /// 加载模型并构建后端（失败返回 [`EngineError::BackendError`]）。
    pub fn new(model_path: &str, config: EngineConfig) -> Result<Self, EngineError> {
        // 维度字段由 GGUF 提取，仅 block_size / max_batch_size 生效；
        // max_batch_size clamp 到显存可承受范围。
        let ccfg = TinyLlmConfig {
            hidden_dim: 0,
            num_layers: 0,
            num_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            vocab_size: 0,
            block_size: config.block_size as c_int,
            max_batch_size: (config.max_num_seqs as c_int).min(MAX_CONCURRENT_SEQS),
        };
        let path = CString::new(model_path)
            .map_err(|_| EngineError::BackendError("model path contains NUL".into()))?;
        let mut err_buf = [0i8; 512];
        let handle = unsafe {
            symbols::tinyllm_load(
                path.as_ptr(),
                &ccfg,
                err_buf.as_mut_ptr(),
                err_buf.len() as c_int,
            )
        };
        if handle.is_null() {
            let msg = err_buf
                .iter()
                .take_while(|&&c| c != 0)
                .map(|&c| c as u8 as char)
                .collect::<String>();
            return Err(EngineError::BackendError(format!(
                "tinyllm_load failed: {msg}"
            )));
        }
        Ok(Self {
            handle,
            allocated: HashSet::new(),
        })
    }

    fn ensure_allocated(&mut self, seq_id: c_int, context_len: usize) -> Result<(), EngineError> {
        if self.allocated.contains(&seq_id) {
            return Ok(());
        }
        let alloc_tokens = (context_len as i32 + DECODE_RESERVE).max(64);
        let rc = unsafe { symbols::tinyllm_allocate_sequence(self.handle, seq_id, alloc_tokens) };
        if rc != 0 {
            return Err(EngineError::BackendError(format!(
                "tinyllm_allocate_sequence({seq_id}) failed"
            )));
        }
        self.allocated.insert(seq_id);
        Ok(())
    }
}

// 裸指针只在本后端内使用（引擎单线程驱动 step），跨线程不共享。
unsafe impl Send for TinyLlmExecutor {}

impl Drop for TinyLlmExecutor {
    fn drop(&mut self) {
        unsafe { symbols::tinyllm_free(self.handle) }
    }
}

impl GPUExecutorTrait for TinyLlmExecutor {
    fn execute(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput, EngineError> {
        if batch.is_empty() {
            return Ok(ExecutionOutput::default());
        }
        let n = batch.num_sequences();

        // 首次出现的序列分配 KV（用 context_len 预估）
        let mut seq_ids: Vec<c_int> = Vec::with_capacity(n);
        for (i, &sid) in batch.seq_ids.iter().enumerate() {
            let sid_i = sid as c_int;
            let ctx = batch.context_lens.get(i).copied().unwrap_or(0) as usize;
            self.ensure_allocated(sid_i, ctx)?;
            seq_ids.push(sid_i);
        }

        let input_tokens: Vec<c_int> =
            batch.input_tokens.iter().map(|&t| t as c_int).collect();
        let positions: Vec<c_int> = batch.positions.iter().map(|&p| p as c_int).collect();
        let seq_lens: Vec<c_int> = batch.seq_lens.iter().map(|&l| l as c_int).collect();
        let is_prefill: Vec<u8> = batch.is_prefill.iter().map(|&b| b as u8).collect();
        let mut next_tokens = vec![0i32; n];

        let rc = unsafe {
            symbols::tinyllm_step(
                self.handle,
                seq_ids.as_ptr(),
                input_tokens.as_ptr(),
                positions.as_ptr(),
                seq_lens.as_ptr(),
                std::ptr::null(), // 策略 2：连续 KV，忽略 block_tables
                is_prefill.as_ptr(),
                n as c_int,
                next_tokens.as_mut_ptr(),
                std::ptr::null_mut(),
                0,
            )
        };
        if rc != 0 {
            return Err(EngineError::BackendError(format!(
                "tinyllm_step failed rc={rc}"
            )));
        }

        Ok(ExecutionOutput {
            next_tokens: next_tokens.into_iter().map(|t| t as TokenId).collect(),
            seq_ids: batch.seq_ids.clone(),
            logprobs: Vec::new(),
        })
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        ExecutorCapabilities::GREEDY_ONLY
    }
}
