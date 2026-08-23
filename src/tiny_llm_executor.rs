//! tiny-llm（C++/CUDA）真实执行后端适配器（里程碑 4）。
//!
//! 实现 [`GPUExecutorTrait`]：把引擎调度的 [`ExecutionBatch`] 经 C ABI
//! （[`crate::tiny_llm_ffi`]）交给 tiny-llm 步进执行。
//!
//! # 边界（策略 1：分页 KV，默认）
//! - `block_tables` / `num_blocks` 真实上传给 tiny-llm，tiny-llm 侧按块表
//!   scatter/gather 写入分页 KV 池（max_num_blocks 由 [`EngineConfig`] 配置，
//!   默认 1024，即默认启用策略 1）；
//! - 每序列首次出现时分配 KV（context_len + decode 预留 token，预留默认 512，
//!   `PAGED_INFER_TINY_LLM_DECODE_RESERVE` 可调），decode 超出预留会失败
//!   （调用方应限制 `max_tokens`）；
//! - 仅支持 greedy 采样（[`ExecutorCapabilities::GREEDY_ONLY`]）；
//! - tokenizer 必须与 tiny-llm 加载的模型词表一致。
//!
//! # 策略 2（连续 KV）fallback
//! 设置环境变量 `PAGED_INFER_TINY_LLM_STRATEGY=2` 时强制 `max_num_blocks=0`，
//! tiny-llm 走连续 KV，`block_tables` / `num_blocks` 传 null（行为同 D1b）。
//!
//! # 容量调节（环境变量）
//! - `PAGED_INFER_TINY_LLM_MAX_SEQS`：最大并发序列数（默认 4）。单序列 KV
//!   按 max_seq_len 预分配，该值决定 KV 池显存占用；显存充裕时可上调以做
//!   更高并发的压测（如 6GB 卡跑 0.5B 模型时并发 8 可容纳）。
//! - `PAGED_INFER_TINY_LLM_DECODE_RESERVE`：每序列 decode 预留 token 数
//!   （默认 512）。长生成场景应不低于 `max_tokens`。
//!   非法值（非整数）记录警告并回退默认值。
//!
//! 仅在 `tiny-llm` cargo feature 下编译，且需 `TINY_LLM_DIR` 指向
//! 已构建的 tiny-llm 静态库（见 build.rs）。

use crate::config::EngineConfig;
use crate::error::EngineError;
use crate::gpu_executor::{ExecutorCapabilities, GPUExecutorTrait};
use crate::tiny_llm_ffi::symbols;
use crate::tiny_llm_ffi::{TinyLlmConfig, TinyLlmHandle};
use crate::types::{ExecutionBatch, ExecutionOutput, SeqId, TokenId};
use std::collections::HashMap;
use std::ffi::CString;
use std::os::raw::c_int;

/// decode 预留 token 数默认值（allocate 时 context_len + 该值）。
const DEFAULT_DECODE_RESERVE: i32 = 512;

/// 最大并发序列默认值。tiny-llm 单 slot KV 按 max_seq_len 预分配
///（0.5B 模型 max_seq_len=32768 时 ~400MB/序列），适配器把最大并发序列
/// clamp 到此值，避免 KV pool 超出显存；显存充裕时可经环境变量上调。
const DEFAULT_MAX_CONCURRENT_SEQS: i32 = 4;

/// 连续 KV（策略 2）fallback 开关：`PAGED_INFER_TINY_LLM_STRATEGY=2`。
const STRATEGY2_ENV: &str = "PAGED_INFER_TINY_LLM_STRATEGY";

/// 最大并发序列调节：`PAGED_INFER_TINY_LLM_MAX_SEQS=<n>`（默认 4，下限 1）。
const MAX_SEQS_ENV: &str = "PAGED_INFER_TINY_LLM_MAX_SEQS";

/// decode 预留调节：`PAGED_INFER_TINY_LLM_DECODE_RESERVE=<n>`（默认 512，下限 0）。
const DECODE_RESERVE_ENV: &str = "PAGED_INFER_TINY_LLM_DECODE_RESERVE";

/// 读取 i32 环境变量；缺失/非法值回退默认（非法值记警告，避免静默误配置）。
fn env_i32_or(name: &str, default: i32) -> i32 {
    match std::env::var(name) {
        Ok(v) => match v.trim().parse::<i32>() {
            Ok(n) => n,
            Err(_) => {
                log::warn!("{name}={v:?} 不是合法整数，回退默认 {default}");
                default
            }
        },
        Err(_) => default,
    }
}

/// tiny-llm 真实执行后端。
pub struct TinyLlmExecutor {
    handle: *mut TinyLlmHandle,
    /// seq_id -> 已分配的 KV token 容量（用于 decode 越界前置检查）。
    allocated: HashMap<c_int, i32>,
    /// 是否走策略 1（分页 KV）。false = 策略 2 fallback（块表传 null）。
    paged: bool,
    /// 后端 batch 上限（config.max_batch_size clamp 到显存可承受范围），
    /// 用于 `execute` 的 batch 超限前置检查（B1）。
    max_batch_size: i32,
    /// decode 预留 token 数（`PAGED_INFER_TINY_LLM_DECODE_RESERVE`，默认 512）。
    decode_reserve: i32,
}

impl TinyLlmExecutor {
    /// 加载模型并构建后端（失败返回 [`EngineError::BackendError`]）。
    pub fn new(model_path: &str, config: EngineConfig) -> Result<Self, EngineError> {
        // 维度字段由 GGUF 提取，仅 block_size / max_batch_size / max_num_blocks
        // 生效；max_batch_size clamp 到显存可承受范围。
        // 策略 1（分页 KV）为默认：max_num_blocks = config（默认 1024）。
        // `PAGED_INFER_TINY_LLM_STRATEGY=2` 强制策略 2（连续 KV，max_num_blocks=0）。
        let force_strategy2 = std::env::var(STRATEGY2_ENV)
            .map(|v| v == "2")
            .unwrap_or(false);
        let paged = !force_strategy2;
        // 容量调节：显存充裕时可上调并发序列数；长生成场景可上调 decode 预留。
        let max_concurrent_seqs = env_i32_or(MAX_SEQS_ENV, DEFAULT_MAX_CONCURRENT_SEQS).max(1);
        let decode_reserve = env_i32_or(DECODE_RESERVE_ENV, DEFAULT_DECODE_RESERVE).max(0);
        // 后端 batch 上限应来自调度器侧的单次调度上限（config.max_batch_size），
        // 而非全局并发上限 max_num_seqs；再 clamp 到显存可承受范围。
        // 同时保存到结构体，供 execute 前置检查使用。
        let max_batch_size = (config.max_batch_size as c_int).min(max_concurrent_seqs);
        let ccfg = TinyLlmConfig {
            hidden_dim: 0,
            num_layers: 0,
            num_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            vocab_size: 0,
            block_size: config.block_size as c_int,
            max_batch_size,
            max_num_blocks: if paged {
                config.max_num_blocks as c_int
            } else {
                0
            },
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
            allocated: HashMap::new(),
            paged,
            max_batch_size,
            decode_reserve,
        })
    }

    /// 单序列首次分配的 KV token 容量：context_len + decode_reserve（下限 64）。
    /// 独立成纯函数以便单元测试锁定该契约（B15）。
    fn alloc_tokens_for(context_len: usize, decode_reserve: i32) -> i32 {
        (context_len as i32 + decode_reserve).max(64)
    }

    fn ensure_allocated(&mut self, seq_id: c_int, context_len: usize) -> Result<(), EngineError> {
        if self.allocated.contains_key(&seq_id) {
            return Ok(());
        }
        let alloc_tokens = Self::alloc_tokens_for(context_len, self.decode_reserve);
        let rc = unsafe { symbols::tinyllm_allocate_sequence(self.handle, seq_id, alloc_tokens) };
        if rc != 0 {
            return Err(EngineError::BackendError(format!(
                "tinyllm_allocate_sequence({seq_id}) failed"
            )));
        }
        self.allocated.insert(seq_id, alloc_tokens);
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

        // batch 超限在调度前就明确报错，避免 tinyllm_step 以 rc!=0 返回时
        // 无法与后端错误/OOM 区分（语义与 CpuReferenceExecutor 一致）。
        if n > self.max_batch_size as usize {
            return Err(EngineError::KernelLaunchFailed(format!(
                "Batch size {} exceeds max {}",
                n, self.max_batch_size
            )));
        }

        // 首次出现的序列分配 KV（用 context_len 预估）
        let mut seq_ids: Vec<c_int> = Vec::with_capacity(n);
        for (i, &sid) in batch.seq_ids.iter().enumerate() {
            let sid_i = sid as c_int;
            let ctx = batch.context_lens.get(i).copied().unwrap_or(0) as usize;
            self.ensure_allocated(sid_i, ctx)?;
            // 前置检查：decode 将超出 tiny-llm 预留 KV 容量（context + decode_reserve）
            // 时给出明确错误，而不是等 tinyllm_step 以 rc!=0 返回后无从归因。
            if let Some(&capacity) = self.allocated.get(&sid_i) {
                if ctx as i32 + 1 > capacity {
                    let reserve = self.decode_reserve;
                    return Err(EngineError::BackendError(format!(
                        "sequence {sid} 生成将超出 tiny-llm 预留 KV 容量 \
                         (context+{reserve}，容量 {capacity})，请调小 max_tokens \
                         或调大 PAGED_INFER_TINY_LLM_DECODE_RESERVE"
                    )));
                }
            }
            seq_ids.push(sid_i);
        }

        let input_tokens: Vec<c_int> = batch.input_tokens.iter().map(|&t| t as c_int).collect();
        let positions: Vec<c_int> = batch.positions.iter().map(|&p| p as c_int).collect();
        let seq_lens: Vec<c_int> = batch.seq_lens.iter().map(|&l| l as c_int).collect();
        let is_prefill: Vec<u8> = batch.is_prefill.iter().map(|&b| b as u8).collect();
        let mut next_tokens = vec![0i32; n];

        // 策略 1：把每序列的块表扁平化，num_blocks 给出每序列块数。
        // 策略 2 fallback：两者都传 null。
        let mut block_tables_flat: Vec<c_int> = Vec::new();
        let mut num_blocks: Vec<c_int> = Vec::with_capacity(n);
        if self.paged {
            for bt in &batch.block_tables {
                if bt.is_empty() {
                    return Err(EngineError::BackendError("empty block table".into()));
                }
                num_blocks.push(bt.len() as c_int);
                block_tables_flat.extend(bt.iter().map(|&b| b as c_int));
            }
        }
        let bt_ptr = if self.paged {
            block_tables_flat.as_ptr()
        } else {
            std::ptr::null()
        };
        let nb_ptr = if self.paged {
            num_blocks.as_ptr()
        } else {
            std::ptr::null()
        };

        let rc = unsafe {
            symbols::tinyllm_step(
                self.handle,
                seq_ids.as_ptr(),
                input_tokens.as_ptr(),
                positions.as_ptr(),
                seq_lens.as_ptr(),
                bt_ptr,
                nb_ptr,
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

    fn sequences_finished(&mut self, seq_ids: &[SeqId]) {
        for &sid in seq_ids {
            let sid_i = sid as c_int;
            if self.allocated.remove(&sid_i).is_some() {
                let rc = unsafe { symbols::tinyllm_free_sequence(self.handle, sid_i) };
                if rc != 0 {
                    log::warn!("tinyllm_free_sequence({sid}) failed rc={rc}");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc_tokens_for_capacity_formula() {
        // B15 回归：分配容量 = (context_len + decode_reserve).max(64)。
        // 这是 decode 越界检查的容量来源——锁定契约，防止公式再次漂移。
        // 注：下限 64 仅在 context+reserve < 64 时生效（旧版测试曾误断言
        // 小 context 恒等于 64，与公式矛盾——该门控测试长期未在 CI 执行）。
        let reserve = DEFAULT_DECODE_RESERVE;
        assert_eq!(TinyLlmExecutor::alloc_tokens_for(0, reserve), reserve);
        assert_eq!(TinyLlmExecutor::alloc_tokens_for(10, reserve), 10 + reserve);
        assert_eq!(
            TinyLlmExecutor::alloc_tokens_for(100, reserve),
            100 + reserve
        );
        assert_eq!(
            TinyLlmExecutor::alloc_tokens_for(2048, reserve),
            2048 + reserve
        );
        // 预留可配置：不同 reserve 值按同一公式生效
        assert_eq!(TinyLlmExecutor::alloc_tokens_for(100, 1024), 1124);
        assert_eq!(TinyLlmExecutor::alloc_tokens_for(100, 0), 100);
        // 下限 64 生效的边界
        assert_eq!(TinyLlmExecutor::alloc_tokens_for(10, 0), 64);
        assert_eq!(TinyLlmExecutor::alloc_tokens_for(0, 0), 64);
    }

    #[test]
    fn test_decode_budget_overrun_condition() {
        // execute 的越界判定：context_len + 1 > capacity 即返回明确错误。
        // 用分配公式构造容量，验证边界（capacity-1 不越界，capacity 处越界）。
        let reserve = DEFAULT_DECODE_RESERVE;
        let context_len = 100usize;
        let capacity = TinyLlmExecutor::alloc_tokens_for(context_len, reserve);

        // 恰好留在预留内：context 增长到 context+reserve-1 仍合法
        let ctx_at_edge = context_len + reserve as usize - 1;
        assert!((ctx_at_edge as i32) < capacity, "预留边界内不应越界");

        // 下一 token 即越界：context = context+reserve
        let ctx_over = context_len + reserve as usize;
        assert!(ctx_over as i32 + 1 > capacity, "超出预留时应判定越界");
    }

    #[test]
    fn test_env_i32_or_fallback() {
        // 缺失键回退默认；不设置非法值避免污染其他测试的环境。
        assert_eq!(env_i32_or("PAGED_INFER_TEST_UNSET_KEY_XYZ", 42), 42);
    }
}
