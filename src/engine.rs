//! 推理引擎 - 主编排器
//!
//! 协调所有组件实现端到端推理：
//! - Tokenizer 用于文本处理
//! - Scheduler 用于请求管理
//! - GPU Executor 用于计算
//! - KV Cache Manager 用于内存管理
//!
//! # 架构
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │            InferenceEngine              │
//! │  ┌──────────┐  ┌──────────┐  ┌───────┐  │
//! │  │Tokenizer │  │Scheduler │  │  GPU  │  │
//! │  │          │  │          │  │Executor│ │
//! │  └──────────┘  └────┬─────┘  └───────┘  │
//! │                     │                    │
//! │              ┌──────▼──────┐            │
//! │              │ KV Cache    │            │
//! │              │ Manager     │            │
//! │              └─────────────┘            │
//! └─────────────────────────────────────────┘
//! ```
//!
//! # 示例
//!
//! ```rust
//! use paged_infer::{EngineConfig, GenerationParams, InferenceEngine};
//!
//! // 创建引擎
//! let config = EngineConfig::default();
//! let mut engine = InferenceEngine::new(config)?;
//!
//! // 提交请求（CPU 参考后端当前仅支持 greedy：temperature = 0.0, top_p = 1.0）
//! let params = GenerationParams {
//!     max_tokens: 50,
//!     ..GenerationParams::default()
//! };
//! let (request_id, _prompt_tokens) = engine.submit_request("你好，世界！", params)?;
//!
//! // 运行推理
//! let completed = engine.run();
//!
//! for result in completed {
//!     println!("输出: {}", result.output_text);
//! }
//! # Ok::<(), paged_infer::EngineError>(())
//! ```

use crate::config::EngineConfig;
use crate::error::EngineError;
use crate::execution_pipeline::BatchExecutionPipeline;
use crate::gpu_executor::{create_default_gpu_executor, GPUExecutorTrait};
use crate::scheduler::Scheduler;
use crate::tokenizer::{build_tokenizer, IncrementalDecoder, TokenizerTrait};
use crate::types::{
    CompletedRequest, FinishReason, GenerationParams, Request, RequestId, RequestState, SeqId,
    TokenId,
};
use std::collections::HashMap;

/// 推理引擎
///
/// 主编排器，协调所有组件实现端到端推理。
///
/// # 组件
///
/// - **Tokenizer** - 文本与 token 之间的转换
/// - **Scheduler** - 请求调度和批次管理
/// - **GPU Executor** - GPU 计算执行
/// - **KV Cache Manager** - KV Cache 内存管理
///
/// # 示例
///
/// ```rust
/// use paged_infer::{EngineConfig, InferenceEngine};
///
/// let config = EngineConfig::default();
/// let engine = InferenceEngine::new(config)?;
/// # Ok::<(), paged_infer::EngineError>(())
/// ```
pub struct InferenceEngine {
    config: EngineConfig,
    tokenizer: Box<dyn TokenizerTrait>,
    scheduler: Scheduler,
    execution_pipeline: BatchExecutionPipeline,
    /// 每请求增量解码器：生命周期与请求一致（完成/取消/失败时销毁），
    /// 保证流式片段拼接 == 最终一次性 decode 文本。
    decoders: HashMap<RequestId, Box<dyn IncrementalDecoder>>,
    eos_token_id: u32,
    total_requests: u64,
    completed_requests_count: u64,
    failed_requests_count: u64,
    total_tokens_generated: u64,
    next_request_id: RequestId,
}

impl InferenceEngine {
    /// 创建新的推理引擎（使用默认组件）
    ///
    /// # 参数
    ///
    /// * `config` - 引擎配置
    ///
    /// # 错误
    ///
    /// 如果配置无效，返回 [`EngineError::Config`]。
    ///
    /// # 示例
    ///
    /// ```rust
    /// use paged_infer::{EngineConfig, InferenceEngine};
    ///
    /// let config = EngineConfig::default();
    /// let engine = InferenceEngine::new(config)?;
    /// # Ok::<(), paged_infer::EngineError>(())
    /// ```
    pub fn new(config: EngineConfig) -> Result<Self, EngineError> {
        config.validate()?;

        let tokenizer = build_tokenizer(&config)?;
        let vocab_size = tokenizer.vocab_size();
        let eos_token_id = tokenizer.eos_token_id();

        let scheduler = Scheduler::new(config.clone());
        let gpu_executor = create_default_gpu_executor(config.clone(), vocab_size)?;
        let execution_pipeline = BatchExecutionPipeline::new(gpu_executor, &config);

        Ok(Self {
            config,
            tokenizer,
            scheduler,
            execution_pipeline,
            decoders: HashMap::new(),
            eos_token_id,
            total_requests: 0,
            completed_requests_count: 0,
            failed_requests_count: 0,
            total_tokens_generated: 0,
            next_request_id: 1,
        })
    }

    /// 使用自定义组件创建引擎（用于测试）
    ///
    /// # 参数
    ///
    /// * `config` - 引擎配置
    /// * `tokenizer` - 自定义分词器
    /// * `scheduler` - 自定义调度器
    /// * `gpu_executor` - 自定义 GPU 执行器
    pub fn with_components(
        config: EngineConfig,
        tokenizer: Box<dyn TokenizerTrait>,
        scheduler: Scheduler,
        gpu_executor: Box<dyn GPUExecutorTrait>,
    ) -> Result<Self, EngineError> {
        config.validate()?;

        let eos_token_id = tokenizer.eos_token_id();
        let execution_pipeline = BatchExecutionPipeline::new(gpu_executor, &config);

        Ok(Self {
            config,
            tokenizer,
            scheduler,
            execution_pipeline,
            decoders: HashMap::new(),
            eos_token_id,
            total_requests: 0,
            completed_requests_count: 0,
            failed_requests_count: 0,
            total_tokens_generated: 0,
            next_request_id: 1,
        })
    }

    /// 提交新的推理请求
    ///
    /// # 参数
    ///
    /// * `text` - 输入文本
    /// * `params` - 生成参数
    ///
    /// # 返回
    ///
    /// `(请求唯一标识符, prompt token 数)`。prompt token 数来自提交时的
    /// 同一次分词，供服务层报告精确 usage，无需二次分词。
    ///
    /// # 错误
    ///
    /// - [`EngineError::EmptyInput`] / [`EngineError::InputTooLong`] / [`EngineError::TotalLengthTooLong`] - 参数无效或输入为空
    /// - [`EngineError::MemoryPressure`] / [`EngineError::MaxConcurrentSequencesReached`] - 内存压力或达到序列上限
    ///
    /// # 示例
    ///
    /// ```rust
    /// use paged_infer::{EngineConfig, GenerationParams, InferenceEngine};
    ///
    /// let config = EngineConfig::default();
    /// let mut engine = InferenceEngine::new(config)?;
    ///
    /// let params = GenerationParams {
    ///     max_tokens: 50,
    ///     ..GenerationParams::default()
    /// };
    ///
    /// let (request_id, prompt_tokens) = engine.submit_request("你好", params)?;
    /// assert!(request_id > 0 && prompt_tokens > 0);
    /// # Ok::<(), paged_infer::EngineError>(())
    /// ```
    pub fn submit_request(
        &mut self,
        text: &str,
        params: GenerationParams,
    ) -> Result<(RequestId, usize), EngineError> {
        // 验证参数
        params.validate()?;

        // 后端能力校验：不支持的采样语义在 submit 阶段诚实失败，
        // 而不是在执行时被后端静默降级为 greedy。
        if !self.execution_pipeline.capabilities().sampling && !params.is_greedy() {
            return Err(EngineError::UnsupportedGenerationMode(format!(
                "backend only supports greedy decoding (temperature == 0.0, top_p == 1.0), \
                 got temperature = {}, top_p = {}",
                params.temperature, params.top_p
            )));
        }

        // 验证输入
        if text.is_empty() {
            return Err(EngineError::EmptyInput);
        }

        // 分词
        let input_tokens = self
            .tokenizer
            .try_encode(text)
            .map_err(EngineError::Tokenization)?;
        let prompt_tokens = input_tokens.len();

        // 检查 prompt 长度
        if input_tokens.len() > self.config.max_model_len as usize {
            return Err(EngineError::InputTooLong(
                input_tokens.len(),
                self.config.max_model_len,
            ));
        }

        let total_requested_tokens = input_tokens.len() + params.max_tokens as usize;
        if total_requested_tokens > self.config.max_model_len as usize {
            return Err(EngineError::TotalLengthTooLong(
                total_requested_tokens,
                self.config.max_model_len,
            ));
        }

        // 创建请求（使用实例级 ID）
        let request_id = self.next_request_id;
        self.next_request_id += 1;
        let request = Request::new(request_id, input_tokens, params);

        // 添加到调度器
        self.scheduler.add_request(request)?;
        self.total_requests += 1;
        self.decoders
            .insert(request_id, self.tokenizer.create_decoder());

        Ok((request_id, prompt_tokens))
    }

    /// 执行一步推理
    ///
    /// 调度下一批次并通过执行流水线完成 GPU 计算。
    ///
    /// # 返回
    ///
    /// 本次步骤完成的请求列表。
    pub fn step(&mut self) -> Result<Vec<CompletedRequest>, EngineError> {
        self.step_events().map(|events| events.completed)
    }

    /// 执行一步推理，并返回细粒度事件
    ///
    /// 除完成请求外，还报告本步为每个请求新生成的 token（已解码为文本片段），
    /// 供服务层驱动 token 级流式响应。
    pub fn step_events(&mut self) -> Result<StepEvents, EngineError> {
        // 调度下一批次
        let scheduler_output = self.scheduler.schedule();
        let mut generated: Vec<(RequestId, TokenId)> = Vec::new();

        if !scheduler_output.is_empty() {
            // seq_id → request_id 映射，用于把生成的 token 归属到请求
            let request_id_by_seq: HashMap<SeqId, RequestId> = scheduler_output
                .prefill_sequences
                .iter()
                .chain(scheduler_output.decode_sequences.iter())
                .map(|seq| (seq.seq_id, seq.request.id))
                .collect();
            let seq_ids = scheduler_output.seq_ids();

            // 通过批次执行流水线完成 GPU 计算（含重试）
            match self.execution_pipeline.execute(&scheduler_output) {
                Ok(execution_output) => {
                    for (seq_id, token) in execution_output
                        .seq_ids
                        .iter()
                        .zip(execution_output.next_tokens.iter())
                    {
                        if let Some(&request_id) = request_id_by_seq.get(seq_id) {
                            generated.push((request_id, *token));
                        }
                    }
                    self.scheduler
                        .update_sequences(&execution_output, self.eos_token_id);
                }
                Err(engine_error) => {
                    let reason = engine_error.to_string();
                    self.scheduler.fail_sequences(seq_ids, &reason);
                    // 失败路径同样排出终态并销毁对应 decoder（不回放缓冲文本）
                    let (completed, tail_chunks) = self.collect_completed_requests();
                    return if completed.is_empty() {
                        Err(engine_error)
                    } else {
                        Ok(StepEvents {
                            completed,
                            chunks: tail_chunks,
                        })
                    };
                }
            }
        }

        // 用每请求增量解码器把新 token 解码为文本片段。
        // 解码失败时让请求诚实失败（tokenizer 错误），而不是静默丢失文本。
        let mut chunks: Vec<(RequestId, String)> = Vec::new();
        for (request_id, token) in generated {
            let Some(decoder) = self.decoders.get_mut(&request_id) else {
                continue;
            };
            match decoder.push(token) {
                Ok(Some(text)) if !text.is_empty() => chunks.push((request_id, text)),
                Ok(_) => {}
                Err(msg) => {
                    self.scheduler
                        .fail_by_request_id(request_id, &format!("tokenizer decode failed: {msg}"));
                }
            }
        }

        let (completed, tail_chunks) = self.collect_completed_requests();
        chunks.extend(tail_chunks);

        Ok(StepEvents { completed, chunks })
    }

    /// 排出所有到达终态的请求；同时销毁其增量解码器。
    ///
    /// 对成功完成的请求调用 `finish()` 冲刷末尾文本，作为本步附加片段返回
    /// （保证在 Done 事件之前送达）；失败/取消的请求直接丢弃解码器状态。
    fn collect_completed_requests(&mut self) -> (Vec<CompletedRequest>, Vec<(RequestId, String)>) {
        let completed_requests = self.scheduler.get_completed();
        if completed_requests.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let mut tail_chunks: Vec<(RequestId, String)> = Vec::new();
        let results = completed_requests
            .into_iter()
            .map(|req| {
                let decoder = self.decoders.remove(&req.id);
                let input_text = self.tokenizer.try_decode(&req.input_tokens).ok();
                let decoded_output = self.tokenizer.try_decode(&req.output_tokens);
                let tokenization_error = decoded_output.as_ref().err().cloned();
                let output_text = decoded_output.unwrap_or_default();
                let success =
                    matches!(req.state, RequestState::Completed) && tokenization_error.is_none();
                // 成功时区分停止原因：末 token 为 EOS → 自然停止；否则是达到
                // max_tokens 被截断。失败/取消的请求没有 finish_reason。
                let finish_reason = if success {
                    let stopped_on_eos = req.output_tokens.last() == Some(&self.eos_token_id);
                    Some(if stopped_on_eos {
                        FinishReason::Stop
                    } else {
                        FinishReason::Length
                    })
                } else {
                    None
                };
                let error = match (&req.state, tokenization_error) {
                    (RequestState::Failed(msg), _) => Some(msg.clone()),
                    (_, Some(msg)) => Some(format!("tokenizer decode failed: {msg}")),
                    _ => None,
                };

                if success {
                    if let Some(mut decoder) = decoder {
                        match decoder.finish() {
                            Ok(Some(tail)) if !tail.is_empty() => {
                                tail_chunks.push((req.id, tail));
                            }
                            Ok(_) => {}
                            Err(msg) => {
                                log::warn!("decoder finish failed for request {}: {msg}", req.id);
                            }
                        }
                    }
                }

                self.total_tokens_generated += req.output_tokens.len() as u64;
                if success {
                    self.completed_requests_count += 1;
                } else {
                    self.failed_requests_count += 1;
                }

                CompletedRequest {
                    request_id: req.id,
                    input_text,
                    output_text,
                    output_tokens: req.output_tokens,
                    success,
                    error,
                    finish_reason,
                }
            })
            .collect();

        (results, tail_chunks)
    }

    /// 运行推理循环直到所有请求完成
    ///
    /// # 返回
    ///
    /// 所有完成的请求列表。
    ///
    /// # 示例
    ///
    /// ```rust
    /// use paged_infer::{EngineConfig, GenerationParams, InferenceEngine};
    ///
    /// let config = EngineConfig::default();
    /// let mut engine = InferenceEngine::new(config)?;
    ///
    /// let params = GenerationParams::default();
    /// engine.submit_request("测试", params)?;
    ///
    /// let completed = engine.run();
    /// for result in completed {
    ///     println!("输出: {}", result.output_text);
    /// }
    /// # Ok::<(), paged_infer::EngineError>(())
    /// ```
    pub fn run(&mut self) -> Vec<CompletedRequest> {
        // 先排出循环前已缓冲的终态（如 cancel_request 产生的失败请求）；
        // 非流式调用方不需要增量片段，丢弃 tail chunks（完整文本在 output_text）。
        let (mut all_completed, _) = self.collect_completed_requests();

        while self.scheduler.has_pending_work() {
            match self.step() {
                Ok(completed) => {
                    all_completed.extend(completed);
                }
                Err(e) => {
                    log::error!("推理步骤失败: {e}");
                }
            }
        }

        all_completed
    }

    /// 按 request_id 取消请求（客户端断连时由服务层调用）。
    ///
    /// 序列无论处于哪个阶段都会被标记失败并释放 KV 资源；
    /// 其终态会在下一步经常规完成通道排出。返回是否取消成功。
    pub fn cancel_request(&mut self, request_id: RequestId) -> bool {
        self.scheduler.cancel_by_request_id(request_id)
    }

    /// 当前存活的每请求增量解码器数量（测试用：验证状态随终态清理）。
    #[cfg(test)]
    fn num_live_decoders(&self) -> usize {
        self.decoders.len()
    }

    /// 检查是否有待处理的工作
    pub fn has_pending_work(&self) -> bool {
        self.scheduler.has_pending_work()
    }

    /// 获取内存利用率
    pub fn memory_utilization(&self) -> f32 {
        self.scheduler.get_memory_utilization()
    }

    /// 获取配置
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }
}

/// 单步推理事件
///
/// 由 [`InferenceEngine::step_events`] 产生，供服务层驱动流式响应。
#[derive(Debug, Clone, Default)]
pub struct StepEvents {
    /// 本步到达终态（成功或失败）的请求
    pub completed: Vec<CompletedRequest>,
    /// 本步为各请求新生成的文本片段：`(request_id, 解码后的片段)`
    pub chunks: Vec<(RequestId, String)>,
}

/// 引擎指标
///
/// 运行时统计信息。
#[derive(Debug, Clone, Default)]
pub struct EngineMetrics {
    /// 已提交请求总数
    pub total_requests: u64,
    /// 成功完成请求总数
    pub completed_requests: u64,
    /// 失败请求总数
    pub failed_requests: u64,
    /// 已生成 token 总数
    pub total_tokens_generated: u64,
    /// 当前内存利用率
    pub memory_utilization: f32,
    /// 当前活跃序列数
    pub active_sequences: u32,
}

impl InferenceEngine {
    /// 获取当前指标
    ///
    /// # 示例
    ///
    /// ```rust
    /// use paged_infer::{EngineConfig, InferenceEngine};
    ///
    /// let config = EngineConfig::default();
    /// let engine = InferenceEngine::new(config)?;
    ///
    /// let metrics = engine.get_metrics();
    /// println!("完成请求: {}", metrics.completed_requests);
    /// # Ok::<(), paged_infer::EngineError>(())
    /// ```
    pub fn get_metrics(&self) -> EngineMetrics {
        EngineMetrics {
            total_requests: self.total_requests,
            completed_requests: self.completed_requests_count,
            failed_requests: self.failed_requests_count,
            total_tokens_generated: self.total_tokens_generated,
            memory_utilization: self.memory_utilization(),
            active_sequences: self.scheduler.num_active_sequences() as u32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{TokenizerConfig, TokenizerKind};
    use crate::error::EngineError;
    use crate::test_utils::{
        create_test_config, test_params, write_test_tokenizer_json, AlwaysFailExecutor,
        ConstantTokenExecutor,
    };
    use crate::tokenizer::SimpleTokenizer;
    use crate::types::{ExecutionBatch, ExecutionOutput};
    use std::fs;

    struct TimeoutThenSuccessExecutor {
        attempts: u32,
    }

    impl GPUExecutorTrait for TimeoutThenSuccessExecutor {
        fn execute(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput, EngineError> {
            if self.attempts == 0 {
                self.attempts += 1;
                Err(EngineError::GpuTimeout)
            } else {
                Ok(ExecutionOutput {
                    next_tokens: vec![123; batch.seq_ids.len()],
                    seq_ids: batch.seq_ids.clone(),
                })
            }
        }
    }

    struct AlwaysTimeoutExecutor;

    impl GPUExecutorTrait for AlwaysTimeoutExecutor {
        fn execute(&mut self, _batch: &ExecutionBatch) -> Result<ExecutionOutput, EngineError> {
            Err(EngineError::GpuTimeout)
        }
    }

    #[test]
    fn test_engine_creation() {
        let config = create_test_config();
        let engine = InferenceEngine::new(config);

        assert!(engine.is_ok());
    }

    #[test]
    fn test_engine_creation_fails_when_huggingface_tokenizer_file_is_missing() {
        let config = EngineConfig {
            tokenizer: TokenizerConfig {
                kind: TokenizerKind::HuggingFace,
                path: Some("/tmp/does-not-exist-tokenizer.json".into()),
            },
            ..create_test_config()
        };

        let engine = InferenceEngine::new(config);
        assert!(matches!(engine, Err(EngineError::Tokenization(_))));
    }

    #[test]
    fn test_submit_request_uses_configured_huggingface_tokenizer() {
        let path = write_test_tokenizer_json();
        let config = EngineConfig {
            max_model_len: 6,
            tokenizer: TokenizerConfig {
                kind: TokenizerKind::HuggingFace,
                path: Some(path.clone()),
            },
            ..create_test_config()
        };
        let mut engine = InferenceEngine::new(config).unwrap();

        let result = engine.submit_request(
            "hello world",
            GenerationParams {
                max_tokens: 1,
                ..GenerationParams::default()
            },
        );

        assert!(
            result.is_ok(),
            "configured HuggingFace tokenizer should be used"
        );

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_cancel_request_stops_generation_and_surfaces_failure() {
        let mut engine = InferenceEngine::new(create_test_config()).unwrap();
        let (request_id, _) = engine
            .submit_request(
                "Hello world",
                GenerationParams {
                    max_tokens: 50,
                    ..GenerationParams::default()
                },
            )
            .unwrap();

        // 推进一步让请求进入 decode
        engine.step().unwrap();
        assert!(engine.has_pending_work());

        assert!(engine.cancel_request(request_id));

        // 即使没有其他工作，run 也必须排出取消产生的失败终态
        let completed = engine.run();
        assert_eq!(completed.len(), 1);
        assert!(!completed[0].success);
        assert!(completed[0]
            .error
            .as_deref()
            .unwrap_or("")
            .contains("cancelled"));
        assert!(!engine.has_pending_work());

        assert!(!engine.cancel_request(999));
    }

    #[test]
    fn test_submit_request() {
        let config = create_test_config();
        let mut engine = InferenceEngine::new(config).unwrap();

        let params = GenerationParams {
            max_tokens: 10,
            ..GenerationParams::default()
        };

        let result = engine.submit_request("Hello", params);
        assert!(result.is_ok());
        assert!(engine.has_pending_work());
    }

    #[test]
    fn test_submit_request_rejects_non_greedy_params_on_greedy_only_backend() {
        let config = create_test_config();
        let mut engine = InferenceEngine::new(config).unwrap();

        // temperature != 0.0：CPU 参考后端没有实现采样，必须诚实失败
        let sampled = GenerationParams {
            max_tokens: 10,
            temperature: 1.0,
            top_p: 1.0,
        };
        let result = engine.submit_request("Hello", sampled);
        assert!(
            matches!(result, Err(EngineError::UnsupportedGenerationMode(_))),
            "non-zero temperature must be rejected, got {result:?}"
        );

        // top_p != 1.0：同样拒绝（即使 temperature == 0.0）
        let top_p = GenerationParams {
            max_tokens: 10,
            temperature: 0.0,
            top_p: 0.9,
        };
        let result = engine.submit_request("Hello", top_p);
        assert!(
            matches!(result, Err(EngineError::UnsupportedGenerationMode(_))),
            "top_p != 1.0 must be rejected, got {result:?}"
        );

        // greedy 组合仍然接受
        let greedy = GenerationParams {
            max_tokens: 10,
            temperature: 0.0,
            top_p: 1.0,
        };
        assert!(engine.submit_request("Hello", greedy).is_ok());
    }

    #[test]
    fn test_submit_empty_request() {
        let config = create_test_config();
        let mut engine = InferenceEngine::new(config).unwrap();

        let params = GenerationParams::default();
        let result = engine.submit_request("", params);

        assert!(result.is_err());
    }

    #[test]
    fn test_submit_invalid_params() {
        let config = create_test_config();
        let mut engine = InferenceEngine::new(config).unwrap();

        let params = GenerationParams {
            max_tokens: 0,
            ..GenerationParams::default()
        };

        let result = engine.submit_request("Hello", params);
        assert!(result.is_err());
    }

    #[test]
    fn test_submit_request_rejects_total_length_over_limit() {
        let config = EngineConfig {
            max_model_len: 8,
            ..create_test_config()
        };
        let mut engine = InferenceEngine::new(config).unwrap();

        let params = GenerationParams {
            max_tokens: 4,
            ..GenerationParams::default()
        };

        let result = engine.submit_request("Hello", params);
        assert!(matches!(result, Err(EngineError::TotalLengthTooLong(_, 8))));
    }

    #[test]
    fn test_submit_request_allows_total_length_at_limit() {
        let config = EngineConfig {
            max_model_len: 11,
            ..create_test_config()
        };
        let mut engine = InferenceEngine::new(config).unwrap();

        let params = GenerationParams {
            max_tokens: 4,
            ..GenerationParams::default()
        };

        let result = engine.submit_request("Hello", params);
        assert!(result.is_ok());
    }

    #[test]
    fn test_submit_request_rejects_prompt_too_long_before_generation_check() {
        let config = EngineConfig {
            max_model_len: 3,
            ..create_test_config()
        };
        let mut engine = InferenceEngine::new(config).unwrap();

        let result = engine.submit_request("Hello", GenerationParams::default());
        assert!(matches!(result, Err(EngineError::InputTooLong(_, 3))));
    }

    #[test]
    fn test_step() {
        let config = create_test_config();
        let mut engine = InferenceEngine::new(config).unwrap();

        let params = GenerationParams {
            max_tokens: 5,
            ..GenerationParams::default()
        };

        engine.submit_request("Hi", params).unwrap();

        let mut completed = Vec::new();
        for _ in 0..10 {
            completed.extend(engine.step().unwrap());
        }

        assert_eq!(
            completed.len(),
            1,
            "request should complete within 10 steps"
        );
        assert!(completed[0].success);
        assert!(!engine.has_pending_work());
    }

    #[test]
    fn test_completed_request_preserves_input_text() {
        let config = create_test_config();
        let mut engine = InferenceEngine::new(config).unwrap();

        engine
            .submit_request("Hello", GenerationParams::default())
            .unwrap();
        let completed = engine.run();

        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].input_text.as_deref(), Some("Hello"));
    }

    #[test]
    fn test_run_to_completion() {
        let config = create_test_config();
        let mut engine = InferenceEngine::new(config).unwrap();

        let params = GenerationParams {
            max_tokens: 3,
            ..GenerationParams::default()
        };

        engine.submit_request("Test", params).unwrap();

        let completed = engine.run();
        assert_eq!(completed.len(), 1, "the submitted request must complete");
        assert!(completed[0].success);
        assert!(!engine.has_pending_work());
    }

    #[test]
    fn test_multiple_requests() {
        let config = create_test_config();
        let mut engine = InferenceEngine::new(config).unwrap();

        let params = GenerationParams {
            max_tokens: 2,
            ..GenerationParams::default()
        };

        for i in 0..3 {
            engine
                .submit_request(&format!("Request {}", i), params)
                .unwrap();
        }

        let completed = engine.run();
        assert_eq!(completed.len(), 3, "all three requests must complete");
        assert!(completed.iter().all(|c| c.success));
        assert!(!engine.has_pending_work());
    }

    #[test]
    fn test_executor_failure_marks_request_failed_and_clears_pending_work() {
        let config = create_test_config();
        let scheduler = Scheduler::new(config.clone());
        let mut engine = InferenceEngine::with_components(
            config,
            Box::new(SimpleTokenizer::new()),
            scheduler,
            Box::new(AlwaysFailExecutor),
        )
        .unwrap();

        engine
            .submit_request("Hello", GenerationParams::default())
            .unwrap();

        let completed = engine.run();

        assert_eq!(completed.len(), 1);
        assert!(!completed[0].success);
        assert!(completed[0].error.is_some());
        assert!(!engine.has_pending_work());
    }

    #[test]
    fn test_executor_failure_updates_failure_metrics() {
        let config = create_test_config();
        let scheduler = Scheduler::new(config.clone());
        let mut engine = InferenceEngine::with_components(
            config,
            Box::new(SimpleTokenizer::new()),
            scheduler,
            Box::new(AlwaysFailExecutor),
        )
        .unwrap();

        engine
            .submit_request("Hello", GenerationParams::default())
            .unwrap();
        let _ = engine.run();

        let metrics = engine.get_metrics();
        assert_eq!(metrics.failed_requests, 1);
        assert_eq!(metrics.completed_requests, 0);
    }

    #[test]
    fn test_gpu_timeout_retries_once_then_succeeds() {
        let config = create_test_config();
        let scheduler = Scheduler::new(config.clone());
        let mut engine = InferenceEngine::with_components(
            config,
            Box::new(SimpleTokenizer::new()),
            scheduler,
            Box::new(TimeoutThenSuccessExecutor { attempts: 0 }),
        )
        .unwrap();

        engine
            .submit_request("Hello", GenerationParams::default())
            .unwrap();
        let completed = engine.run();

        assert_eq!(completed.len(), 1);
        assert!(completed[0].success);
        assert!(completed[0].error.is_none());

        let metrics = engine.get_metrics();
        assert_eq!(metrics.failed_requests, 0);
        assert_eq!(metrics.completed_requests, 1);
    }

    #[test]
    fn test_streaming_chunks_concat_equals_final_output() {
        // PINF-102 核心性质：所有流式片段拼接 == 最终一次性 decode 文本
        let config = create_test_config();
        let mut engine = InferenceEngine::new(config).unwrap();

        let (request_id, _) = engine
            .submit_request(
                "Hello stream",
                GenerationParams {
                    max_tokens: 8,
                    ..GenerationParams::default()
                },
            )
            .unwrap();

        let mut streamed = String::new();
        let mut final_text = None;
        while engine.has_pending_work() {
            let events = engine.step_events().unwrap();
            for (rid, chunk) in &events.chunks {
                assert_eq!(*rid, request_id);
                streamed.push_str(chunk);
            }
            for completed in events.completed {
                assert_eq!(completed.request_id, request_id);
                assert!(completed.success);
                final_text = Some(completed.output_text);
            }
        }

        assert_eq!(streamed, final_text.expect("request must complete"));
        assert!(!streamed.is_empty());
        assert_eq!(
            engine.num_live_decoders(),
            0,
            "decoder state must be dropped on completion"
        );
    }

    #[test]
    fn test_decoder_state_cleaned_up_on_cancel_and_failure() {
        // 取消路径：decoder 与 KV 资源一并清理
        let mut engine = InferenceEngine::new(create_test_config()).unwrap();
        let (request_id, _) = engine
            .submit_request(
                "cancel me",
                GenerationParams {
                    max_tokens: 50,
                    ..GenerationParams::default()
                },
            )
            .unwrap();
        engine.step().unwrap();
        assert_eq!(engine.num_live_decoders(), 1);

        assert!(engine.cancel_request(request_id));
        let completed = engine.run();
        assert_eq!(completed.len(), 1);
        assert!(!completed[0].success);
        assert_eq!(engine.num_live_decoders(), 0);

        // 后端失败路径：decoder 同样被清理
        let config = create_test_config();
        let scheduler = Scheduler::new(config.clone());
        let mut engine = InferenceEngine::with_components(
            config,
            Box::new(SimpleTokenizer::new()),
            scheduler,
            Box::new(AlwaysFailExecutor),
        )
        .unwrap();
        engine
            .submit_request("fail me", GenerationParams::default())
            .unwrap();
        let completed = engine.run();
        assert_eq!(completed.len(), 1);
        assert!(!completed[0].success);
        assert_eq!(engine.num_live_decoders(), 0);
    }

    /// push 阶段解码失败的 tokenizer：引擎必须让请求失败，而不是静默丢文本。
    struct FailingDecodeTokenizer(SimpleTokenizer);

    impl TokenizerTrait for FailingDecodeTokenizer {
        fn try_encode(&self, text: &str) -> Result<Vec<TokenId>, String> {
            self.0.try_encode(text)
        }
        fn try_decode(&self, tokens: &[TokenId]) -> Result<String, String> {
            self.0.try_decode(tokens)
        }
        fn vocab_size(&self) -> u32 {
            self.0.vocab_size()
        }
        fn bos_token_id(&self) -> TokenId {
            self.0.bos_token_id()
        }
        fn eos_token_id(&self) -> TokenId {
            self.0.eos_token_id()
        }
        fn pad_token_id(&self) -> TokenId {
            self.0.pad_token_id()
        }
        fn create_decoder(&self) -> Box<dyn crate::tokenizer::IncrementalDecoder> {
            struct AlwaysFailDecoder;
            impl crate::tokenizer::IncrementalDecoder for AlwaysFailDecoder {
                fn push(&mut self, _token: TokenId) -> Result<Option<String>, String> {
                    Err("simulated incremental decode failure".to_string())
                }
                fn finish(&mut self) -> Result<Option<String>, String> {
                    Ok(None)
                }
            }
            Box::new(AlwaysFailDecoder)
        }
    }

    #[test]
    fn test_decoder_push_failure_fails_request_honestly() {
        let config = create_test_config();
        let scheduler = Scheduler::new(config.clone());
        let mut engine = InferenceEngine::with_components(
            config,
            Box::new(FailingDecodeTokenizer(SimpleTokenizer::new())),
            scheduler,
            Box::new(crate::test_utils::ConstantTokenExecutor { token: 65 }),
        )
        .unwrap();

        engine
            .submit_request("decode failure", GenerationParams::default())
            .unwrap();
        let completed = engine.run();

        assert_eq!(completed.len(), 1);
        assert!(!completed[0].success);
        assert!(
            completed[0]
                .error
                .as_deref()
                .unwrap_or("")
                .contains("tokenizer decode failed"),
            "push failure must surface as tokenizer error, got {:?}",
            completed[0].error
        );
        assert_eq!(engine.num_live_decoders(), 0);
    }

    /// HuggingFace tokenizer 走缓冲式增量解码：生成过程中无中间片段，
    /// 终态时 finish 冲刷出完整文本，拼接性质仍成立。
    #[test]
    fn test_streaming_equivalence_with_buffered_hf_decoder() {
        let path = write_test_tokenizer_json();
        let config = EngineConfig {
            max_model_len: 64,
            tokenizer: TokenizerConfig {
                kind: TokenizerKind::HuggingFace,
                path: Some(path.clone()),
            },
            ..create_test_config()
        };
        let mut engine = InferenceEngine::new(config).unwrap();

        let (request_id, _) = engine
            .submit_request(
                "hello world",
                GenerationParams {
                    max_tokens: 4,
                    ..GenerationParams::default()
                },
            )
            .unwrap();

        let mut streamed = String::new();
        let mut final_text = None;
        while engine.has_pending_work() {
            let events = engine.step_events().unwrap();
            for (rid, chunk) in &events.chunks {
                assert_eq!(*rid, request_id);
                streamed.push_str(chunk);
            }
            for completed in events.completed {
                final_text = Some(completed.output_text);
            }
        }

        assert_eq!(streamed, final_text.expect("request must complete"));
        assert_eq!(engine.num_live_decoders(), 0);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_gpu_timeout_exhausts_retries_then_fails_request() {
        // 执行器持续超时：重试 max_retry_attempts 次后必须放弃，
        // 将请求标记为失败并释放资源，而不是无限重试或静默挂起。
        let config = create_test_config(); // max_retry_attempts: 2
        let scheduler = Scheduler::new(config.clone());
        let mut engine = InferenceEngine::with_components(
            config,
            Box::new(SimpleTokenizer::new()),
            scheduler,
            Box::new(AlwaysTimeoutExecutor),
        )
        .unwrap();

        engine
            .submit_request("Hello", GenerationParams::default())
            .unwrap();
        let completed = engine.run();

        assert_eq!(completed.len(), 1);
        assert!(!completed[0].success);
        assert!(
            completed[0].error.as_deref().unwrap_or("").contains("GPU"),
            "error should surface the GPU timeout, got {:?}",
            completed[0].error
        );
        assert!(!engine.has_pending_work());

        let metrics = engine.get_metrics();
        assert_eq!(metrics.failed_requests, 1);
        assert_eq!(metrics.completed_requests, 0);
    }

    #[test]
    fn test_completed_request_reports_finish_reason() {
        // PINF-103：成功完成的请求必须区分停止原因——
        // 达到 max_tokens 截断 → Length，末 token 为 EOS → Stop。
        let config = create_test_config();
        // EOS 动态取自 tokenizer 而非硬编码，避免依赖默认 special token 配置。
        let eos = SimpleTokenizer::new().eos_token_id();
        // 恒生成一个确定 != EOS 的合法字符 id（65 是 'A'；若冲突则换 66）。
        let non_eos = if eos == 65 { 66 } else { 65 };

        // Length：执行器生成非 EOS token，请求被 max_tokens 截断。
        let scheduler = Scheduler::new(config.clone());
        let mut engine = InferenceEngine::with_components(
            config.clone(),
            Box::new(SimpleTokenizer::new()),
            scheduler,
            Box::new(ConstantTokenExecutor { token: non_eos }),
        )
        .unwrap();
        engine.submit_request("hi", test_params(3)).unwrap();
        let completed = engine.run();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].finish_reason, Some(FinishReason::Length));

        // Stop：执行器每一步都生成 EOS，请求在首个 token 后自然停止。
        let scheduler = Scheduler::new(config.clone());
        let mut engine = InferenceEngine::with_components(
            config.clone(),
            Box::new(SimpleTokenizer::new()),
            scheduler,
            Box::new(ConstantTokenExecutor { token: eos }),
        )
        .unwrap();
        engine.submit_request("hi", test_params(3)).unwrap();
        let completed = engine.run();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].finish_reason, Some(FinishReason::Stop));
    }
}
