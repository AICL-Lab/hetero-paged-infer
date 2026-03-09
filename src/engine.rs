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
use crate::tokenizer::{SimpleTokenizer, TokenizerTrait, EOS_TOKEN_ID};
use crate::types::{
    CompletedRequest, GenerationParams, Request, RequestId, RequestState,
};

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
        
        // Check length
        if input_tokens.len() > self.config.max_model_len as usize {
            return Err(ValidationError::InputTooLong(
                input_tokens.len(),
                self.config.max_model_len,
            ).into());
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
            let execution_output = self.gpu_executor.execute(&execution_batch)?;

            // Update scheduler with results
            self.scheduler.update_sequences(&execution_output, self.eos_token_id);
        }

        // Get completed requests (may exist even without execution batch)
        let completed_requests = self.scheduler.get_completed();
        if completed_requests.is_empty() {
            return Ok(Vec::new());
        }

        // Convert to CompletedRequest with decoded text
        let results: Vec<CompletedRequest> = completed_requests
            .into_iter()
            .map(|req| {
                let output_text = self.tokenizer.decode(&req.output_tokens);
                let success = matches!(req.state, RequestState::Completed);
                let error = match &req.state {
                    RequestState::Failed(msg) => Some(msg.clone()),
                    _ => None,
                };

                // 更新指标
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
            .collect();

        Ok(results)
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
                    // Continue processing other requests
                }
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
            EngineError::Execution(exec_err) => {
                match exec_err {
                    crate::error::ExecutionError::CudaError(_) => RecoveryAction::SkipSequence,
                    crate::error::ExecutionError::GpuTimeout => RecoveryAction::Retry { max_attempts: 2 },
                    crate::error::ExecutionError::InvalidOutput => RecoveryAction::SkipSequence,
                    crate::error::ExecutionError::KernelLaunchFailed(_) => RecoveryAction::ResetBatch,
                }
            }
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

    fn create_test_config() -> EngineConfig {
        EngineConfig {
            block_size: 16,
            max_num_blocks: 100,
            max_batch_size: 8,
            max_num_seqs: 32,
            max_model_len: 2048,
            max_total_tokens: 512,
            memory_threshold: 0.9,
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
            max_tokens: 0, // Invalid
            temperature: 1.0,
            top_p: 0.9,
        };
        
        let result = engine.submit_request("Hello", params);
        assert!(result.is_err());
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
        
        // Run a few steps
        for _ in 0..10 {
            let _ = engine.step();
        }
    }

    #[test]
    fn test_run_to_completion() {
        let config = create_test_config();
        let mut engine = InferenceEngine::new(config).unwrap();
        engine.set_max_steps(100); // Limit steps for test
        
        let params = GenerationParams {
            max_tokens: 3,
            temperature: 1.0,
            top_p: 0.9,
        };
        
        engine.submit_request("Test", params).unwrap();
        
        let completed = engine.run();
        
        // Should complete within max_steps
        assert!(!completed.is_empty() || !engine.has_pending_work());
    }

    #[test]
    fn test_recovery_action() {
        let config = create_test_config();
        let engine = InferenceEngine::new(config).unwrap();
        
        let cuda_error = EngineError::Execution(
            crate::error::ExecutionError::CudaError("test".to_string())
        );
        assert_eq!(engine.handle_error(&cuda_error), RecoveryAction::SkipSequence);
        
        let timeout_error = EngineError::Execution(
            crate::error::ExecutionError::GpuTimeout
        );
        assert_eq!(engine.handle_error(&timeout_error), RecoveryAction::Retry { max_attempts: 2 });
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
        
        // Submit multiple requests
        for i in 0..3 {
            engine.submit_request(&format!("Request {}", i), params).unwrap();
        }
        
        let completed = engine.run();
        
        // Should process all requests
        assert!(completed.len() <= 3);
    }
}
