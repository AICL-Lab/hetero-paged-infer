//! Inference Engine - Main orchestrator
//!
//! Coordinates all components for end-to-end inference:
//! - Tokenizer for text processing
//! - Scheduler for request management
//! - GPU Executor for computation
//! - KV Cache Manager for memory

use crate::config::EngineConfig;
use crate::error::{EngineError, ValidationError};
use crate::gpu_executor::{build_execution_batch, GPUExecutorTrait, MockGPUExecutor};
use crate::scheduler::{Scheduler, SchedulerTrait};
use crate::tokenizer::{SimpleTokenizer, TokenizerTrait};
use crate::types::{CompletedRequest, GenerationParams, Request, RequestId, RequestState};

/// Main inference engine orchestrating all components
pub struct InferenceEngine {
    /// Engine configuration
    config: EngineConfig,
    /// Tokenizer for text processing
    tokenizer: Box<dyn TokenizerTrait>,
    /// Scheduler for request management
    scheduler: Scheduler,
    /// GPU executor for computation
    gpu_executor: Box<dyn GPUExecutorTrait>,
    /// EOS token ID for completion detection
    eos_token_id: u32,
    /// Running flag
    running: bool,
    /// Maximum steps (for testing, 0 = unlimited)
    max_steps: usize,
    /// Total requests submitted
    total_requests: u64,
    /// Total requests completed successfully
    completed_requests_count: u64,
    /// Total requests failed
    failed_requests_count: u64,
    /// Total tokens generated
    total_tokens_generated: u64,
    /// Request ID counter (instance-level, avoids global state leaking across tests)
    next_request_id: RequestId,
}

impl InferenceEngine {
    /// Create a new inference engine with default components
    pub fn new(config: EngineConfig) -> Result<Self, EngineError> {
        config.validate()?;

        let tokenizer = Box::new(SimpleTokenizer::new());
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

    /// Create engine with custom components (for testing)
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

    /// Set maximum steps for testing
    pub fn set_max_steps(&mut self, max_steps: usize) {
        self.max_steps = max_steps;
    }

    /// Submit a new inference request
    pub fn submit_request(
        &mut self,
        text: &str,
        params: GenerationParams,
    ) -> Result<RequestId, EngineError> {
        // Validate parameters
        params.validate()?;

        // Validate input
        if text.is_empty() {
            return Err(ValidationError::EmptyInput.into());
        }

        // Tokenize input
        let input_tokens = self.tokenizer.encode(text);

        // Check prompt length
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

        // Create request with instance-level ID
        let request_id = self.next_request_id;
        self.next_request_id += 1;
        let request = Request::new(request_id, input_tokens, params);

        // Add to scheduler
        self.scheduler.add_request(request)?;
        self.total_requests += 1;

        Ok(request_id)
    }

    /// Execute one inference step
    pub fn step(&mut self) -> Result<Vec<CompletedRequest>, EngineError> {
        // Schedule next batch
        let scheduler_output = self.scheduler.schedule();

        if !scheduler_output.is_empty() {
            // Build execution batch
            let execution_batch = build_execution_batch(&scheduler_output);

            // Execute on GPU
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
                                "Retrying batch execution after error (attempt {}/{}): {}",
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
                let output_text = self.tokenizer.decode(&req.output_tokens);
                let success = matches!(req.state, RequestState::Completed);
                let error = match &req.state {
                    RequestState::Failed(msg) => Some(msg.clone()),
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

    /// Run the inference loop until all requests complete
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
                    log::error!("Inference step failed: {}", e);
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

    /// Stop the inference loop
    pub fn stop(&mut self) {
        self.running = false;
    }

    /// Check if engine has pending work
    pub fn has_pending_work(&self) -> bool {
        self.scheduler.has_pending_work()
    }

    /// Get memory utilization
    pub fn memory_utilization(&self) -> f32 {
        self.scheduler.get_memory_utilization()
    }

    /// Get configuration
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }
}

/// Recovery action for error handling
#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryAction {
    /// Retry the operation
    Retry { max_attempts: u32 },
    /// Skip the problematic sequence
    SkipSequence,
    /// Reset the current batch
    ResetBatch,
    /// Shutdown the engine
    Shutdown,
}

impl InferenceEngine {
    /// Handle execution error and determine recovery action
    pub fn handle_error(&self, error: &EngineError) -> RecoveryAction {
        match error {
            EngineError::Execution(exec_err) => match exec_err {
                crate::error::ExecutionError::CudaError(_) => RecoveryAction::SkipSequence,
                crate::error::ExecutionError::GpuTimeout => {
                    RecoveryAction::Retry { max_attempts: 2 }
                }
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

/// Metrics for monitoring
#[derive(Debug, Clone, Default)]
pub struct EngineMetrics {
    /// Total requests submitted
    pub total_requests: u64,
    /// Total requests completed
    pub completed_requests: u64,
    /// Total requests failed
    pub failed_requests: u64,
    /// Total tokens generated
    pub total_tokens_generated: u64,
    /// Current memory utilization
    pub memory_utilization: f32,
    /// Current active sequences
    pub active_sequences: u32,
}

impl InferenceEngine {
    /// Get current metrics
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
    use crate::error::ExecutionError;
    use crate::test_utils::create_test_config;
    use crate::types::{ExecutionBatch, ExecutionOutput};

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
            Err(EngineError::Validation(ValidationError::TotalLengthTooLong(_, 8)))
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
        assert_eq!(engine.handle_error(&cuda_error), RecoveryAction::SkipSequence);

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

        engine.submit_request("Hello", GenerationParams::default()).unwrap();

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

        engine.submit_request("Hello", GenerationParams::default()).unwrap();
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

        engine.submit_request("Hello", GenerationParams::default()).unwrap();
        let completed = engine.run();

        assert_eq!(completed.len(), 1);
        assert!(completed[0].success);
        assert!(completed[0].error.is_none());

        let metrics = engine.get_metrics();
        assert_eq!(metrics.failed_requests, 0);
        assert_eq!(metrics.completed_requests, 1);
    }
}
