//! CPU 参考执行器
//!
//! 使用随机初始化的小型 Transformer 模型在 CPU 上执行真实前向计算，
//! 使端到端推理路径（embedding → attention with paged KV cache → FFN → 采样）
//! 真正跑通，为 Serving 控制面提供有意义的测试基础。
//!
//! 模型使用固定种子随机初始化权重，输出具有确定性。计算结果不追求精度，
//! 旨在验证 PagedAttention 块表访问、continuous batching 调度和采样路径的正确性。
//!
//! # 模型结构
//!
//! ```text
//! hidden_dim=64, num_layers=2, num_heads=4, head_dim=16, intermediate_dim=128
//! ```

use crate::config::EngineConfig;
use crate::error::EngineError;
use crate::gpu_executor::GPUExecutorTrait;
use crate::types::{BlockIdx, ExecutionBatch, ExecutionOutput, SeqId, TokenId, TokenLogprobs};
use std::collections::HashMap;

// --- 模型超参数（固定小型配置） ---
const HIDDEN_DIM: usize = 64;
const NUM_LAYERS: usize = 2;
const NUM_HEADS: usize = 4;
const NUM_KV_HEADS: usize = 4;
const HEAD_DIM: usize = 16;
const INTER_DIM: usize = 128;
const RMS_EPS: f32 = 1e-5;
const ROPE_THETA: f32 = 10000.0;

const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM;

/// 每个 token 位置返回的 top logprob 候选数（OpenAI top_logprobs 上限为 5）。
const TOP_LOGPROBS_K: usize = 5;

// --- 简单 PRNG（Xorshift32，固定种子保证确定性） ---
struct Rng(u32);

impl Rng {
    fn new(seed: u32) -> Self {
        Self(if seed == 0 { 0x9E37_79B9 } else { seed })
    }

    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }

    /// 生成 [-0.1, 0.1) 范围的伪随机浮点数
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32 - 0.5) * 0.2
    }

    fn fill(&mut self, len: usize) -> Vec<f32> {
        (0..len).map(|_| self.next_f32()).collect()
    }
}

// --- 模型权重 ---
struct CpuLayer {
    rms_att: Vec<f32>, // [hidden]
    wq: Vec<f32>,      // [hidden, hidden]
    wk: Vec<f32>,      // [hidden, kv_dim]
    wv: Vec<f32>,      // [hidden, kv_dim]
    wo: Vec<f32>,      // [hidden, hidden]
    rms_ffn: Vec<f32>, // [hidden]
    w1: Vec<f32>,      // [hidden, inter]  (gate)
    w2: Vec<f32>,      // [inter, hidden]  (down)
    w3: Vec<f32>,      // [hidden, inter]  (up)
}

struct CpuModel {
    token_embedding: Vec<f32>, // [vocab_size, hidden]
    layers: Vec<CpuLayer>,
    final_norm: Vec<f32>, // [hidden]
    lm_head: Vec<f32>,    // [hidden, vocab_size]
}

impl CpuModel {
    fn new(vocab_size: usize) -> Self {
        let mut rng = Rng::new(42);
        let layers = (0..NUM_LAYERS)
            .map(|_| CpuLayer {
                rms_att: rng.fill(HIDDEN_DIM),
                wq: rng.fill(HIDDEN_DIM * HIDDEN_DIM),
                wk: rng.fill(HIDDEN_DIM * KV_DIM),
                wv: rng.fill(HIDDEN_DIM * KV_DIM),
                wo: rng.fill(HIDDEN_DIM * HIDDEN_DIM),
                rms_ffn: rng.fill(HIDDEN_DIM),
                w1: rng.fill(HIDDEN_DIM * INTER_DIM),
                w2: rng.fill(INTER_DIM * HIDDEN_DIM),
                w3: rng.fill(HIDDEN_DIM * INTER_DIM),
            })
            .collect();

        Self {
            token_embedding: rng.fill(vocab_size * HIDDEN_DIM),
            layers,
            final_norm: rng.fill(HIDDEN_DIM),
            lm_head: rng.fill(HIDDEN_DIM * vocab_size),
        }
    }
}

// --- Paged KV Cache 数据块 ---
struct KvBlock {
    k: Vec<f32>, // [block_size, kv_dim]
    v: Vec<f32>, // [block_size, kv_dim]
}

// --- 数学函数 ---

/// 行主序矩阵 × 向量: output[r] = sum_c weight[r*cols+c] * input[c]
fn matmul_vec(weight: &[f32], input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0; rows];
    for r in 0..rows {
        let mut sum = 0.0;
        for c in 0..cols {
            sum += weight[r * cols + c] * input[c];
        }
        output[r] = sum;
    }
    output
}

fn rmsnorm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    let mean_sq: f32 = input.iter().map(|x| x * x).sum::<f32>() / n as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    input
        .iter()
        .zip(weight)
        .map(|(x, w)| x * inv_rms * w)
        .collect()
}

/// 对向量中每个 head 的维度对应用 RoPE 旋转
fn apply_rope(vec: &mut [f32], pos: usize, num_heads: usize, head_dim: usize) {
    for h in 0..num_heads {
        let off = h * head_dim;
        for i in 0..head_dim / 2 {
            let theta = pos as f32 * ROPE_THETA.powf(-((2 * i) as f32) / head_dim as f32);
            let (sin, cos) = theta.sin_cos();
            let a = vec[off + 2 * i];
            let b = vec[off + 2 * i + 1];
            vec[off + 2 * i] = a * cos - b * sin;
            vec[off + 2 * i + 1] = a * sin + b * cos;
        }
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// --- CPU 参考执行器 ---
pub struct CpuReferenceExecutor {
    config: EngineConfig,
    vocab_size: u32,
    model: CpuModel,
    kv_cache: HashMap<(usize, BlockIdx), KvBlock>,
    /// seq_id -> 最近一次 execute 的 block_table，用于序列结束时回收其 KV 块。
    seq_block_tables: HashMap<SeqId, Vec<BlockIdx>>,
}

impl std::fmt::Debug for CpuReferenceExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuReferenceExecutor")
            .field("vocab_size", &self.vocab_size)
            .field("hidden_dim", &HIDDEN_DIM)
            .field("num_layers", &NUM_LAYERS)
            .field("kv_blocks", &self.kv_cache.len())
            .finish_non_exhaustive()
    }
}

impl CpuReferenceExecutor {
    pub fn new(config: EngineConfig, vocab_size: u32) -> Self {
        let model = CpuModel::new(vocab_size as usize);
        Self {
            config,
            vocab_size,
            model,
            kv_cache: HashMap::new(),
            seq_block_tables: HashMap::new(),
        }
    }

    fn block_size(&self) -> usize {
        self.config.block_size as usize
    }

    /// 对单个序列执行前向传播，返回下一个 token 及其 logprob 信息
    fn forward_seq(
        &mut self,
        tokens: &[TokenId],
        positions: &[u32],
        block_table: &[BlockIdx],
    ) -> Result<(TokenId, TokenLogprobs), EngineError> {
        let bs = self.block_size();
        let mut hidden = vec![0.0f32; HIDDEN_DIM];

        for (i, &token) in tokens.iter().enumerate() {
            let pos = positions[i] as usize;

            // Embedding lookup
            let embed_idx = (token as usize).min(self.vocab_size as usize - 1);
            hidden.copy_from_slice(
                &self.model.token_embedding[embed_idx * HIDDEN_DIM..(embed_idx + 1) * HIDDEN_DIM],
            );

            for (layer_idx, layer) in self.model.layers.iter().enumerate() {
                // --- Attention block ---
                let normed = rmsnorm(&hidden, &layer.rms_att, RMS_EPS);
                let mut q = matmul_vec(&layer.wq, &normed, HIDDEN_DIM, HIDDEN_DIM);
                let mut k = matmul_vec(&layer.wk, &normed, KV_DIM, HIDDEN_DIM);
                let v = matmul_vec(&layer.wv, &normed, KV_DIM, HIDDEN_DIM);

                apply_rope(&mut q, pos, NUM_HEADS, HEAD_DIM);
                apply_rope(&mut k, pos, NUM_KV_HEADS, HEAD_DIM);

                // 写入 K/V 到 paged cache (layer-aware key)
                let block_idx = *block_table.get(pos / bs).ok_or_else(|| {
                    EngineError::BackendError(format!(
                        "block_table too short: need index {} for pos {}, len {}",
                        pos / bs,
                        pos,
                        block_table.len()
                    ))
                })?;
                let offset = pos % bs;
                let start = offset * KV_DIM;
                let block = self
                    .kv_cache
                    .entry((layer_idx, block_idx))
                    .or_insert_with(|| KvBlock {
                        k: vec![0.0; bs * KV_DIM],
                        v: vec![0.0; bs * KV_DIM],
                    });
                block.k[start..start + KV_DIM].copy_from_slice(&k);
                block.v[start..start + KV_DIM].copy_from_slice(&v);

                // Attention: 读取位置 0..=pos 的 K/V
                let attn_out = self.attention(&q, block_table, pos, layer_idx)?;

                // Output 投影 + 残差
                let proj = matmul_vec(&layer.wo, &attn_out, HIDDEN_DIM, HIDDEN_DIM);
                for j in 0..HIDDEN_DIM {
                    hidden[j] += proj[j];
                }

                // --- FFN block (SwiGLU) ---
                let normed2 = rmsnorm(&hidden, &layer.rms_ffn, RMS_EPS);
                let gate = matmul_vec(&layer.w1, &normed2, INTER_DIM, HIDDEN_DIM);
                let up = matmul_vec(&layer.w3, &normed2, INTER_DIM, HIDDEN_DIM);
                let mut inter = vec![0.0; INTER_DIM];
                for j in 0..INTER_DIM {
                    inter[j] = silu(gate[j]) * up[j];
                }
                let ffn_out = matmul_vec(&layer.w2, &inter, HIDDEN_DIM, INTER_DIM);
                for j in 0..HIDDEN_DIM {
                    hidden[j] += ffn_out[j];
                }
            }
        }

        // Final RMSNorm + LM Head
        let normed = rmsnorm(&hidden, &self.model.final_norm, RMS_EPS);
        let logits = matmul_vec(
            &self.model.lm_head,
            &normed,
            self.vocab_size as usize,
            HIDDEN_DIM,
        );

        // Greedy 采样 (argmax) + top-k logprobs。
        // softmax 使用 max-logit 数值稳定；logprob 即 ln(softmax)。
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|l| (l - max_logit).exp()).sum();
        let mut ranked: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (i, ((l - max_logit).exp() / exp_sum).ln()))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top = &ranked[..TOP_LOGPROBS_K.min(ranked.len())];
        let (token_idx, logprob) = top[0];
        let top_logprobs: Vec<(TokenId, f32)> =
            top.iter().map(|&(i, lp)| (i as TokenId, lp)).collect();

        Ok((
            token_idx as TokenId,
            TokenLogprobs {
                token: token_idx as TokenId,
                logprob,
                top_logprobs,
            },
        ))
    }

    /// 多头注意力：通过 block_table 读取 paged KV cache，计算 causal attention
    fn attention(
        &self,
        q: &[f32],
        block_table: &[BlockIdx],
        pos: usize,
        layer_idx: usize,
    ) -> Result<Vec<f32>, EngineError> {
        let bs = self.block_size();
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let mut output = vec![0.0; HIDDEN_DIM];

        for h in 0..NUM_HEADS {
            // GQA: 每个 query head 映射到对应的 kv head
            let kv_h = h * NUM_KV_HEADS / NUM_HEADS;
            let q_h = &q[h * HEAD_DIM..(h + 1) * HEAD_DIM];

            // 计算注意力分数（causal: 只看 0..=pos）
            let mut scores = Vec::with_capacity(pos + 1);
            for i in 0..=pos {
                let block_idx = *block_table.get(i / bs).ok_or_else(|| {
                    EngineError::BackendError(format!(
                        "block_table too short: need index {} for pos {}, len {}",
                        i / bs,
                        i,
                        block_table.len()
                    ))
                })?;
                let offset = i % bs;
                let k_start = offset * KV_DIM + kv_h * HEAD_DIM;
                let block = self.kv_cache.get(&(layer_idx, block_idx)).ok_or_else(|| {
                    EngineError::BackendError(format!(
                        "KV cache miss for layer {layer_idx}, block {block_idx}"
                    ))
                })?;
                let k_h = &block.k[k_start..k_start + HEAD_DIM];
                let score = q_h.iter().zip(k_h).map(|(a, b)| a * b).sum::<f32>() * scale;
                scores.push(score);
            }

            // Softmax
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = scores.iter().map(|s| (s - max_score).exp()).sum();
            if !exp_sum.is_finite() || exp_sum <= 0.0 {
                return Err(EngineError::InvalidOutput);
            }
            for s in &mut scores {
                *s = (*s - max_score).exp() / exp_sum;
            }

            // 加权求和 V
            for (i, &score) in scores.iter().enumerate() {
                let block_idx = *block_table.get(i / bs).ok_or_else(|| {
                    EngineError::BackendError(format!(
                        "block_table too short: need index {} for pos {}, len {}",
                        i / bs,
                        i,
                        block_table.len()
                    ))
                })?;
                let offset = i % bs;
                let v_start = offset * KV_DIM + kv_h * HEAD_DIM;
                let block = self.kv_cache.get(&(layer_idx, block_idx)).ok_or_else(|| {
                    EngineError::BackendError(format!(
                        "KV cache miss for layer {layer_idx}, block {block_idx}"
                    ))
                })?;
                let v_h = &block.v[v_start..v_start + HEAD_DIM];
                for d in 0..HEAD_DIM {
                    output[h * HEAD_DIM + d] += score * v_h[d];
                }
            }
        }

        Ok(output)
    }
}

impl GPUExecutorTrait for CpuReferenceExecutor {
    fn execute(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput, EngineError> {
        if batch.is_empty() {
            return Ok(ExecutionOutput::default());
        }

        if self.vocab_size == 0 {
            return Err(EngineError::BackendError(
                "CPU executor received an empty vocabulary".to_string(),
            ));
        }

        if batch.num_sequences() > self.config.max_batch_size as usize {
            return Err(EngineError::KernelLaunchFailed(format!(
                "Batch size {} exceeds max {}",
                batch.num_sequences(),
                self.config.max_batch_size
            )));
        }

        if batch.total_tokens() > self.config.max_total_tokens as usize {
            return Err(EngineError::KernelLaunchFailed(format!(
                "Total tokens {} exceeds max {}",
                batch.total_tokens(),
                self.config.max_total_tokens
            )));
        }

        let mut next_tokens = Vec::with_capacity(batch.num_sequences());
        let mut logprobs = Vec::with_capacity(batch.num_sequences());
        let mut token_offset = 0;

        for (seq_idx, _) in batch.seq_ids.iter().enumerate() {
            let seq_len = batch.seq_lens[seq_idx] as usize;
            let tokens = &batch.input_tokens[token_offset..token_offset + seq_len];
            let positions = &batch.positions[token_offset..token_offset + seq_len];
            let block_table = &batch.block_tables[seq_idx];
            token_offset += seq_len;

            // 记录该序列的块表，供序列结束时回收其 KV 块。
            self.seq_block_tables
                .insert(batch.seq_ids[seq_idx], block_table.to_vec());

            let (next_token, token_logprobs) = self.forward_seq(tokens, positions, block_table)?;
            next_tokens.push(next_token);
            logprobs.push(Some(token_logprobs));
        }

        Ok(ExecutionOutput {
            next_tokens,
            seq_ids: batch.seq_ids.clone(),
            logprobs,
        })
    }

    fn sequences_finished(&mut self, seq_ids: &[SeqId]) {
        for &sid in seq_ids {
            // 回收该序列占用的全部 KV 块（物理块在调度器侧是排他的：
            // 同一时刻只属于一个序列，故删除是安全的）。
            if let Some(block_table) = self.seq_block_tables.remove(&sid) {
                for block_idx in block_table {
                    for layer_idx in 0..NUM_LAYERS {
                        self.kv_cache.remove(&(layer_idx, block_idx));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::create_test_config;

    fn make_batch(tokens: &[TokenId], block_table: Vec<BlockIdx>) -> ExecutionBatch {
        ExecutionBatch {
            input_tokens: tokens.to_vec(),
            positions: (0..tokens.len() as u32).collect(),
            seq_lens: vec![tokens.len() as u32],
            block_tables: vec![block_table],
            is_prefill: vec![true],
            seq_ids: vec![1],
            context_lens: vec![tokens.len() as u32],
        }
    }

    #[test]
    fn test_cpu_executor_prefill() {
        let config = create_test_config();
        let mut executor = CpuReferenceExecutor::new(config, 128);

        let batch = make_batch(&[10, 20, 30], vec![0]);
        let result = executor.execute(&batch);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.next_tokens.len(), 1);
        assert!(output.next_tokens[0] < 128, "token must be in vocab range");
    }

    #[test]
    fn test_cpu_executor_deterministic() {
        let config = create_test_config();

        // 两个独立执行器，相同输入应产生相同输出（固定种子权重）
        let mut exec1 = CpuReferenceExecutor::new(config.clone(), 128);
        let mut exec2 = CpuReferenceExecutor::new(config, 128);

        let batch = make_batch(&[5, 10, 15], vec![0]);
        let out1 = exec1.execute(&batch).unwrap();
        let out2 = exec2.execute(&batch).unwrap();

        assert_eq!(out1.next_tokens, out2.next_tokens);
    }

    #[test]
    fn test_cpu_executor_decode_uses_paged_cache() {
        let config = create_test_config();
        let mut executor = CpuReferenceExecutor::new(config, 128);

        // Prefill: 写入 3 个 token 的 K/V 到 block 0
        let prefill = make_batch(&[10, 20, 30], vec![0]);
        let prefill_out = executor.execute(&prefill).unwrap();
        let next_token = prefill_out.next_tokens[0];

        // Decode: 1 个 token，需要读取 block 0 中位置 0-2 的 K/V
        let decode = ExecutionBatch {
            input_tokens: vec![next_token],
            positions: vec![3],
            seq_lens: vec![1],
            block_tables: vec![vec![0]],
            is_prefill: vec![false],
            seq_ids: vec![1],
            context_lens: vec![4],
        };
        let result = executor.execute(&decode);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.next_tokens.len(), 1);
        assert!(output.next_tokens[0] < 128);
    }

    #[test]
    fn test_cpu_executor_multi_block_paged_access() {
        // block_size=16，用 20 个 token 触发跨块访问
        let config = create_test_config(); // block_size=16
        let mut executor = CpuReferenceExecutor::new(config, 128);

        let tokens: Vec<TokenId> = (0..20).collect();
        // 20 个 token 需要 2 个块
        let batch = make_batch(&tokens, vec![0, 1]);
        let result = executor.execute(&batch);

        assert!(result.is_ok());
        assert!(result.unwrap().next_tokens[0] < 128);
    }

    #[test]
    fn test_cpu_executor_empty_batch() {
        let config = create_test_config();
        let mut executor = CpuReferenceExecutor::new(config, 128);

        let result = executor.execute(&ExecutionBatch::default());
        assert!(result.is_ok());
        assert!(result.unwrap().next_tokens.is_empty());
    }

    #[test]
    fn test_cpu_executor_multi_sequence() {
        let config = create_test_config();
        let mut executor = CpuReferenceExecutor::new(config, 128);

        let batch = ExecutionBatch {
            input_tokens: vec![1, 2, 3, 10, 20],
            positions: vec![0, 1, 2, 0, 1],
            seq_lens: vec![3, 2],
            block_tables: vec![vec![0], vec![1]],
            is_prefill: vec![true, true],
            seq_ids: vec![1, 2],
            context_lens: vec![3, 2],
        };

        let result = executor.execute(&batch);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.next_tokens.len(), 2);
        assert!(output.next_tokens.iter().all(|&t| t < 128));
    }

    /// B2 回归：序列结束后 `sequences_finished` 必须回收其占用的全部 KV 块
    /// （跨所有层），且清除 block table 记录，避免 KV 泄漏。
    #[test]
    fn test_sequences_finished_reclaims_kv_blocks() {
        let config = create_test_config();
        let mut executor = CpuReferenceExecutor::new(config, 128);

        // 用 20 个 token 触发跨块访问（block_size=16 → 2 个物理块）
        let tokens: Vec<TokenId> = (0..20).collect();
        let batch = make_batch(&tokens, vec![0, 1]);
        executor.execute(&batch).unwrap();

        // prefill 后：KV 缓存非空，且该序列的块表已记录
        assert!(!executor.kv_cache.is_empty(), "prefill 后应有 KV 缓存");
        assert_eq!(
            executor.seq_block_tables.get(&1),
            Some(&vec![0, 1]),
            "应记录序列 1 的块表"
        );

        // 序列结束：回收全部 KV 块
        executor.sequences_finished(&[1]);

        assert!(
            executor.kv_cache.is_empty(),
            "sequences_finished 后 KV 缓存应清空（当前 {} 块）",
            executor.kv_cache.len()
        );
        assert!(
            !executor.seq_block_tables.contains_key(&1),
            "块表记录也应清除"
        );
    }

    /// B2 回归：多波次执行下，每波结束后 KV 必须归还，
    /// 否则小 KV 池会被泄漏的块耗尽。
    #[test]
    fn test_kv_cache_bounded_across_waves() {
        let config = create_test_config();
        let mut executor = CpuReferenceExecutor::new(config, 128);

        for wave in 0..5u64 {
            // 每波用不同 seq_id，模拟新请求复用同一 executor
            let batch = ExecutionBatch {
                input_tokens: vec![1, 2, 3, 4],
                positions: vec![0, 1, 2, 3],
                seq_lens: vec![4],
                block_tables: vec![vec![wave as BlockIdx]],
                is_prefill: vec![true],
                seq_ids: vec![wave],
                context_lens: vec![4],
            };
            executor.execute(&batch).unwrap();
            assert!(!executor.kv_cache.is_empty(), "第 {wave} 波应有 KV 缓存");

            executor.sequences_finished(&[wave]);
            assert!(
                executor.kv_cache.is_empty(),
                "第 {wave} 波结束后 KV 应全部归还（剩余 {} 块）",
                executor.kv_cache.len()
            );
        }
    }
}

#[cfg(test)]
mod layer_isolation_tests {
    use super::*;
    use crate::test_utils::create_test_config;

    /// PSERV-001 regression test: Two layers writing to the same physical block
    /// at the same position must not overwrite each other.
    ///
    /// With the old HashMap<BlockIdx, KvBlock> key, layer 1 would overwrite
    /// layer 0's K/V at the same position. This test verifies that the
    /// layer-aware key prevents cross-layer contamination.
    #[test]
    fn test_layer_kv_isolation() {
        let config = create_test_config();
        let mut executor = CpuReferenceExecutor::new(config, 128);

        // Run a prefill with 3 tokens - this exercises both layers
        let batch = ExecutionBatch {
            input_tokens: vec![10, 20, 30],
            positions: vec![0, 1, 2],
            seq_lens: vec![3],
            block_tables: vec![vec![0]],
            is_prefill: vec![true],
            seq_ids: vec![1],
            context_lens: vec![3],
        };

        let result = executor.execute(&batch);
        assert!(result.is_ok(), "prefill should succeed");

        // After prefill, both layers should have independent K/V data in block 0.
        // With the old single-key HashMap, only the last layer's data would remain.
        // Verify that layer 0's cache is distinct from layer 1's cache.
        let layer0 = executor.kv_cache.get(&(0, 0));
        let layer1 = executor.kv_cache.get(&(1, 0));

        assert!(layer0.is_some(), "layer 0 cache should exist");
        assert!(layer1.is_some(), "layer 1 cache should exist");

        let l0 = layer0.unwrap();
        let l1 = layer1.unwrap();

        // K/V data for layer 0 and layer 1 should differ (different random weights)
        let k_differs: bool =
            l0.k.iter()
                .zip(l1.k.iter())
                .any(|(a, b)| (a - b).abs() > 1e-10);
        let v_differs: bool =
            l0.v.iter()
                .zip(l1.v.iter())
                .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(k_differs, "layer 0 and layer 1 K data must differ");
        assert!(v_differs, "layer 0 and layer 1 V data must differ");
    }

    /// PSERV-001: Multi-layer incremental decode should produce the same result
    /// as a full recompute when given the same input tokens.
    ///
    /// Full prefill [5,10,15,20] should produce the same next token as
    /// prefill [5,10,15] + decode [20] at position 3.
    #[test]
    fn test_multilayer_incremental_vs_full_recompute() {
        let config = create_test_config();

        // Full prefill: process all 4 tokens at once
        let mut exec_full = CpuReferenceExecutor::new(config.clone(), 128);
        let batch_full = ExecutionBatch {
            input_tokens: vec![5, 10, 15, 20],
            positions: vec![0, 1, 2, 3],
            seq_lens: vec![4],
            block_tables: vec![vec![0]],
            is_prefill: vec![true],
            seq_ids: vec![1],
            context_lens: vec![4],
        };
        let out_full = exec_full.execute(&batch_full).unwrap();

        // Incremental: prefill 3 tokens, then decode with token 20 at position 3
        let mut exec_incr = CpuReferenceExecutor::new(config, 128);
        let batch_prefill = ExecutionBatch {
            input_tokens: vec![5, 10, 15],
            positions: vec![0, 1, 2],
            seq_lens: vec![3],
            block_tables: vec![vec![0]],
            is_prefill: vec![true],
            seq_ids: vec![1],
            context_lens: vec![3],
        };
        exec_incr.execute(&batch_prefill).unwrap();

        // Decode with the SAME 4th token (20) at position 3
        let batch_decode = ExecutionBatch {
            input_tokens: vec![20],
            positions: vec![3],
            seq_lens: vec![1],
            block_tables: vec![vec![0]],
            is_prefill: vec![false],
            seq_ids: vec![1],
            context_lens: vec![4],
        };
        let out_decode = exec_incr.execute(&batch_decode).unwrap();

        // Both paths compute the same model on the same 4 tokens, so the
        // next token prediction should be identical. This verifies that
        // the layer-aware cache correctly preserves per-layer K/V across
        // prefill + decode.
        assert_eq!(
            out_decode.next_tokens[0], out_full.next_tokens[0],
            "incremental decode should match full recompute for same tokens"
        );
    }
}
