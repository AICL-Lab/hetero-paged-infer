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
//! use hetero_infer::{EngineConfig, GenerationParams, InferenceEngine};
//!
//! // 创建引擎
//! let config = EngineConfig::default();
//! let mut engine = InferenceEngine::new(config)?;
//!
//! // 提交请求
//! let params = GenerationParams {
//!     max_tokens: 50,
//!     temperature: 1.0,
//!     top_p: 0.9,
//! };
//! let request_id = engine.submit_request("你好，世界！", params)?;
//!
//! // 运行推理
//! let completed = engine.run();
//!
//! for result in completed {
//!     println!("输出: {}", result.output_text);
//! }
//! # Ok::<(), hetero_infer::EngineError>(())
//! ```

use crate::config::EngineConfig;
use crate::error::{EngineError, ValidationError};
use crate::gpu_executor::{build_execution_batch, GPUExecutorTrait, MockGPUExecutor};
use crate::scheduler::{Scheduler, SchedulerTrait};
use crate::tokenizer::{build_tokenizer, TokenizerTrait};
use crate::types::{CompletedRequest, GenerationParams, Request, RequestId, RequestState};

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
/// use hetero_infer::{EngineConfig, InferenceEngine};
///
/// let config = EngineConfig::default();
/// let engine = InferenceEngine::new(config)?;
/// # Ok::<(), hetero_infer::EngineError>(())
/// ```
pub struct InferenceEngine {
    /// 引擎配置
    config: EngineConfig,
    /// 文本处理器
    tokenizer: Box<dyn TokenizerTrait>,
    /// 请求调度器
    scheduler: Scheduler,
    /// GPU 执行器
    gpu_executor: Box<dyn GPUExecutorTrait>,
    /// EOS token ID（用于检测完成）
    eos_token_id: u32,
    /// 运行标志
    running: bool,
    /// 最大步数（用于测试，0 = 无限制）
    max_steps: usize,
    /// 已提交请求总数
    total_requests: u64,
    /// 成功完成请求总数
    completed_requests_count: u64,
    /// 失败请求总数
    failed_requests_count: u64,
    /// 已生成 token 总数
    total_tokens_generated: u64,
    /// 请求 ID 计数器（实例级，避免全局状态在测试间泄漏）
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
    /// use hetero_infer::{EngineConfig, InferenceEngine};
    ///
    /// let config = EngineConfig::default();
    /// let engine = InferenceEngine::new(config)?;
    /// # Ok::<(), hetero_infer::EngineError>(())
    /// ```
    pub fn new(config: EngineConfig) -> Result<Self, EngineError> {
        config.validate()?;

        let tokenizer = build_tokenizer(&config)?;
        let vocab_size = tokenizer.vocab_size();
        let eos_token_id = tokenizer.eos_token_id();

        let scheduler = Scheduler::new(config.clone());
        let gpu_executor = Box::new(MockGPUExecutor::new(config.clone(), vocab_size));

        Ok(Self {
            config,
            tokenizer,
            scheduler,
            gpu_executor,
            eos_token_id,
            running: false,
            max_steps: 0,
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

        Ok(Self {
            config,
            tokenizer,
            scheduler,
            gpu_executor,
            eos_token_id,
            running: false,
            max_steps: 0,
            total_requests: 0,
            completed_requests_count: 0,
            failed_requests_count: 0,
            total_tokens_generated: 0,
            next_request_id: 1,
        })
    }

    /// 设置最大步数（用于测试）
    ///
    /// # 参数
    ///
    /// * `max_steps` - 最大执行步数，0 表示无限制
    pub fn set_max_steps(&mut self, max_steps: usize) {
        self.max_steps = max_steps;
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
    /// 请求的唯一标识符。
    ///
    /// # 错误
    ///
    /// - [`EngineError::Validation`] - 参数无效或输入为空
    /// - [`EngineError::Scheduler`] - 内存压力或达到序列上限
    ///
    /// # 示例
    ///
    /// ```rust
    /// use hetero_infer::{EngineConfig, GenerationParams, InferenceEngine};
    ///
    /// let config = EngineConfig::default();
    /// let mut engine = InferenceEngine::new(config)?;
    ///
    /// let params = GenerationParams {
    ///     max_tokens: 50,
    ///     temperature: 1.0,
    ///     top_p: 0.9,
    /// };
    ///
    /// let request_id = engine.submit_request("你好", params)?;
    /// # Ok::<(), hetero_infer::EngineError>(())
    /// ```
    pub fn submit_request(
        &mut self,
        text: &str,
        params: GenerationParams,
    ) -> Result<RequestId, EngineError> {
        // 验证参数
        params.validate()?;

        // 验证输入
        if text.is_empty() {
            return Err(ValidationError::EmptyInput.into());
        }

        // 分词
        let input_tokens = self
            .tokenizer
            .try_encode(text)
            .map_err(EngineError::Tokenization)?;

        // 检查 prompt 长度
        if input_tokens.len() > self.config.max_model_len as usize {
            return Err(ValidationError::InputTooLong(
                input_tokens.len(),
                self.config.max_model_len,
            )
            .into());
        }

        let total_requested_tokens = input_tokens.len() + params.max_tokens as usize;
        if total_requested_tokens > self.config.max_model_len as usize {
            return Err(ValidationError::TotalLengthTooLong(
                total_requested_tokens,
                self.config.max_model_len,
            )
            .into());
        }

        // 创建请求（使用实例级 ID）
        let request_id = self.next_request_id;
        self.next_request_id += 1;
        let request = Request::new(request_id, input_tokens, params);

        // 添加到调度器
        self.scheduler.add_request(request)?;
        self.total_requests += 1;

        Ok(request_id)
    }

    /// 执行一步推理
    ///
    /// 调度下一批次并执行 GPU 计算。
    ///
    /// # 返回
    ///
    /// 本次步骤完成的请求列表。
    pub fn step(&mut self) -> Result<Vec<CompletedRequest>, EngineError> {
        // 调度下一批次
        let scheduler_output = self.scheduler.schedule();

        if !scheduler_output.is_empty() {
            // 构建执行批次
            let execution_batch = build_execution_batch(&scheduler_output);

            // 执行 GPU 计算
            match self.execute_batch(&execution_batch) {
                Ok(execution_output) => {
                    self.scheduler
                        .update_sequences(&execution_output, self.eos_token_id);
                }
                Err(engine_error) => {
                    let reason = engine_error.to_string();
                    self.scheduler
                        .fail_sequences(execution_batch.seq_ids.iter().copied(), &reason);
                    let completed = self.collect_completed_requests();
                    return if completed.is_empty() {
                        Err(engine_error)
                    } else {
                        Ok(completed)
                    };
                }
            }
        }

        Ok(self.collect_completed_requests())
    }

    fn execute_batch(
        &mut self,
        execution_batch: &crate::types::ExecutionBatch,
    ) -> Result<crate::types::ExecutionOutput, EngineError> {
        let mut retries = 0;

        loop {
            match self.gpu_executor.execute(execution_batch) {
                Ok(output) => return Ok(output),
                Err(exec_error) => {
                    let engine_error = EngineError::Execution(exec_error);
                    match self.handle_error(&engine_error) {
                        RecoveryAction::Retry { max_attempts } if retries < max_attempts => {
                            retries += 1;
                            log::warn!(
                                "重试批次执行 (尝试 {}/{}): {}",
                                retries,
                                max_attempts,
                                engine_error
                            );
                        }
                        _ => return Err(engine_error),
                    }
                }
            }
        }
    }

    fn collect_completed_requests(&mut self) -> Vec<CompletedRequest> {
        let completed_requests = self.scheduler.get_completed();
        if completed_requests.is_empty() {
            return Vec::new();
        }

        completed_requests
            .into_iter()
            .map(|req| {
                let decoded_output = self.tokenizer.try_decode(&req.output_tokens);
                let tokenization_error = decoded_output.as_ref().err().cloned();
                let output_text = decoded_output.unwrap_or_default();
                let success =
                    matches!(req.state, RequestState::Completed) && tokenization_error.is_none();
                let error = match (&req.state, tokenization_error) {
                    (RequestState::Failed(msg), _) => Some(msg.clone()),
                    (_, Some(msg)) => Some(format!("tokenizer decode failed: {msg}")),
                    _ => None,
                };

                self.total_tokens_generated += req.output_tokens.len() as u64;
                if success {
                    self.completed_requests_count += 1;
                } else {
                    self.failed_requests_count += 1;
                }

                CompletedRequest {
                    request_id: req.id,
                    input_text: None,
                    output_text,
                    output_tokens: req.output_tokens,
                    success,
                    error,
                }
            })
            .collect()
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
    /// use hetero_infer::{EngineConfig, GenerationParams, InferenceEngine};
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
    /// # Ok::<(), hetero_infer::EngineError>(())
    /// ```
    pub fn run(&mut self) -> Vec<CompletedRequest> {
        self.running = true;
        let mut all_completed = Vec::new();
        let mut steps = 0;

        while self.running && self.scheduler.has_pending_work() {
            match self.step() {
                Ok(completed) => {
                    all_completed.extend(completed);
                }
                Err(e) => {
                    log::error!("推理步骤失败: {}", e);
                    match self.handle_error(&e) {
                        RecoveryAction::Shutdown => {
                            self.running = false;
                        }
                        RecoveryAction::Retry { .. }
                        | RecoveryAction::SkipSequence
                        | RecoveryAction::ResetBatch => {}
                    }
                }
            }

            if !self.scheduler.has_pending_work() {
                all_completed.extend(self.collect_completed_requests());
            }

            steps += 1;
            if self.max_steps > 0 && steps >= self.max_steps {
                break;
            }
        }

        self.running = false;
        all_completed
    }

    /// 停止推理循环
    pub fn stop(&mut self) {
        self.running = false;
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

/// 错误恢复策略
///
/// 定义执行错误发生时的恢复行为。
#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryAction {
    /// 重试操作
    Retry {
        /// 最大重试次数
        max_attempts: u32,
    },
    /// 跳过问题序列
    SkipSequence,
    /// 重置当前批次
    ResetBatch,
    /// 关闭引擎
    Shutdown,
}

impl InferenceEngine {
    /// 处理执行错误并确定恢复策略
    ///
    /// # 参数
    ///
    /// * `error` - 发生的错误
    ///
    /// # 返回
    ///
    /// 推荐的恢复策略。
    pub fn handle_error(&self, error: &EngineError) -> RecoveryAction {
        match error {
            EngineError::Execution(exec_err) => match exec_err {
                crate::error::ExecutionError::CudaError(_) => RecoveryAction::SkipSequence,
                crate::error::ExecutionError::GpuTimeout => RecoveryAction::Retry {
                    max_attempts: self.config.max_retry_attempts,
                },
                crate::error::ExecutionError::InvalidOutput => RecoveryAction::SkipSequence,
                crate::error::ExecutionError::KernelLaunchFailed(_) => RecoveryAction::ResetBatch,
            },
            EngineError::Memory(_) => RecoveryAction::ResetBatch,
            EngineError::Config(_) => RecoveryAction::Shutdown,
            EngineError::Validation(_) => RecoveryAction::SkipSequence,
            EngineError::Scheduler(_) => RecoveryAction::Retry { max_attempts: 1 },
            EngineError::Tokenization(_) => RecoveryAction::SkipSequence,
        }
    }
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
    /// use hetero_infer::{EngineConfig, InferenceEngine};
    ///
    /// let config = EngineConfig::default();
    /// let engine = InferenceEngine::new(config)?;
    ///
    /// let metrics = engine.get_metrics();
    /// println!("完成请求: {}", metrics.completed_requests);
    /// # Ok::<(), hetero_infer::EngineError>(())
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
    use crate::error::ExecutionError;
    use crate::test_utils::create_test_config;
    use crate::tokenizer::SimpleTokenizer;
    use crate::types::{ExecutionBatch, ExecutionOutput};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn write_test_tokenizer_json() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("hetero-engine-tokenizer-{unique}.json"));
        fs::write(
            &path,
            r###"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": { "type": "Whitespace" },
  "post_processor": null,
  "decoder": { "type": "WordPiece", "prefix": "##", "cleanup": false },
  "model": {
    "type": "WordLevel",
    "vocab": {
      "[UNK]": 0,
      "hello": 1,
      "world": 2
    },
    "unk_token": "[UNK]"
  }
}"###,
        )
        .unwrap();
        path
    }

    struct AlwaysFailExecutor;

    impl GPUExecutorTrait for AlwaysFailExecutor {
        fn execute(&mut self, _batch: &ExecutionBatch) -> Result<ExecutionOutput, ExecutionError> {
            Err(ExecutionError::KernelLaunchFailed("boom".to_string()))
        }

        fn capture_decode_graph(&mut self, _batch_size: u32) -> Result<(), ExecutionError> {
            Ok(())
        }

        fn execute_graph(
            &mut self,
            _batch: &ExecutionBatch,
        ) -> Result<ExecutionOutput, ExecutionError> {
            Err(ExecutionError::KernelLaunchFailed("boom".to_string()))
        }

        fn has_captured_graph(&self) -> bool {
            false
        }
    }

    struct TimeoutThenSuccessExecutor {
        attempts: u32,
    }

    impl GPUExecutorTrait for TimeoutThenSuccessExecutor {
        fn execute(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput, ExecutionError> {
            if self.attempts == 0 {
                self.attempts += 1;
                Err(ExecutionError::GpuTimeout)
            } else {
                Ok(ExecutionOutput {
                    next_tokens: vec![123; batch.seq_ids.len()],
                    logits: None,
                    seq_ids: batch.seq_ids.clone(),
                })
            }
        }

        fn capture_decode_graph(&mut self, _batch_size: u32) -> Result<(), ExecutionError> {
            Ok(())
        }

        fn execute_graph(
            &mut self,
            batch: &ExecutionBatch,
        ) -> Result<ExecutionOutput, ExecutionError> {
            self.execute(batch)
        }

        fn has_captured_graph(&self) -> bool {
            false
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
                temperature: 1.0,
                top_p: 0.9,
            },
        );

        assert!(
            result.is_ok(),
            "configured HuggingFace tokenizer should be used"
        );

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_submit_request() {
        let config = create_test_config();
        let mut engine = InferenceEngine::new(config).unwrap();

        let params = GenerationParams {
            max_tokens: 10,
            temperature: 1.0,
            top_p: 0.9,
        };

        let result = engine.submit_request("Hello", params);
        assert!(result.is_ok());
        assert!(engine.has_pending_work());
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
            temperature: 1.0,
            top_p: 0.9,
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
            temperature: 1.0,
            top_p: 0.9,
        };

        let result = engine.submit_request("Hello", params);
        assert!(matches!(
            result,
            Err(EngineError::Validation(
                ValidationError::TotalLengthTooLong(_, 8)
            ))
        ));
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
            temperature: 1.0,
            top_p: 0.9,
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
        assert!(matches!(
            result,
            Err(EngineError::Validation(ValidationError::InputTooLong(_, 3)))
        ));
    }

    #[test]
    fn test_step() {
        let config = create_test_config();
        let mut engine = InferenceEngine::new(config).unwrap();

        let params = GenerationParams {
            max_tokens: 5,
            temperature: 1.0,
            top_p: 0.9,
        };

        engine.submit_request("Hi", params).unwrap();

        for _ in 0..10 {
            let _ = engine.step();
        }
    }

    #[test]
    fn test_run_to_completion() {
        let config = create_test_config();
        let mut engine = InferenceEngine::new(config).unwrap();
        engine.set_max_steps(100);

        let params = GenerationParams {
            max_tokens: 3,
            temperature: 1.0,
            top_p: 0.9,
        };

        engine.submit_request("Test", params).unwrap();

        let completed = engine.run();
        assert!(!completed.is_empty() || !engine.has_pending_work());
    }

    #[test]
    fn test_recovery_action() {
        let config = create_test_config();
        let engine = InferenceEngine::new(config).unwrap();

        let cuda_error =
            EngineError::Execution(crate::error::ExecutionError::CudaError("test".to_string()));
        assert_eq!(
            engine.handle_error(&cuda_error),
            RecoveryAction::SkipSequence
        );

        let timeout_error = EngineError::Execution(crate::error::ExecutionError::GpuTimeout);
        assert_eq!(
            engine.handle_error(&timeout_error),
            RecoveryAction::Retry { max_attempts: 2 }
        );
    }

    #[test]
    fn test_multiple_requests() {
        let config = create_test_config();
        let mut engine = InferenceEngine::new(config).unwrap();
        engine.set_max_steps(50);

        let params = GenerationParams {
            max_tokens: 2,
            temperature: 1.0,
            top_p: 0.9,
        };

        for i in 0..3 {
            engine
                .submit_request(&format!("Request {}", i), params)
                .unwrap();
        }

        let completed = engine.run();
        assert!(completed.len() <= 3);
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
        engine.set_max_steps(200);

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
}
