//! Core types and data structures for the inference system

use std::sync::Arc;
use std::time::Instant;

/// Unique identifier for requests
pub type RequestId = u64;

/// Unique identifier for sequences
pub type SeqId = u64;

/// Token ID type
pub type TokenId = u32;

/// Physical block index
pub type BlockIdx = u32;

/// Request state in the inference pipeline
#[derive(Debug, Clone, PartialEq)]
pub enum RequestState {
    /// Request is queued, waiting to be scheduled
    Pending,
    /// Request is in prefill phase (processing input tokens)
    Prefill,
    /// Request is in decode phase (generating tokens)
    Decode,
    /// Request has completed successfully
    Completed,
    /// Request failed with an error message
    Failed(String),
}

impl RequestState {
    pub fn is_active(&self) -> bool {
        matches!(self, RequestState::Prefill | RequestState::Decode)
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, RequestState::Completed | RequestState::Failed(_))
    }
}

/// Generation parameters for a request
#[derive(Debug, Clone, Copy)]
pub struct GenerationParams {
    /// Maximum number of tokens to generate
    pub max_tokens: u32,
    /// Sampling temperature (0.0, 2.0]
    pub temperature: f32,
    /// Top-p (nucleus) sampling parameter (0.0, 1.0]
    pub top_p: f32,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 100,
            temperature: 1.0,
            top_p: 1.0,
        }
    }
}

impl GenerationParams {
    /// Validate generation parameters
    pub fn validate(&self) -> Result<(), crate::ValidationError> {
        if self.max_tokens == 0 {
            return Err(crate::ValidationError::InvalidMaxTokens(self.max_tokens));
        }
        if self.temperature <= 0.0 || self.temperature > 2.0 {
            return Err(crate::ValidationError::InvalidTemperature(self.temperature));
        }
        if self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(crate::ValidationError::InvalidTopP(self.top_p));
        }
        Ok(())
    }

    /// Check if parameters are valid
    pub fn is_valid(&self) -> bool {
        self.max_tokens > 0
            && self.temperature > 0.0
            && self.temperature <= 2.0
            && self.top_p > 0.0
            && self.top_p <= 1.0
    }
}

/// A single inference request
#[derive(Debug, Clone)]
pub struct Request {
    /// Unique request identifier
    pub id: RequestId,
    /// Input tokens after tokenization
    pub input_tokens: Vec<TokenId>,
    /// Generated output tokens
    pub output_tokens: Vec<TokenId>,
    /// Generation parameters
    pub params: GenerationParams,
    /// Current state
    pub state: RequestState,
    /// Creation timestamp
    pub created_at: Instant,
}

impl Request {
    /// Create a new request
    pub fn new(id: RequestId, input_tokens: Vec<TokenId>, params: GenerationParams) -> Self {
        Self {
            id,
            input_tokens,
            output_tokens: Vec::new(),
            params,
            state: RequestState::Pending,
            created_at: Instant::now(),
        }
    }

    /// Total number of tokens (input + output)
    pub fn total_tokens(&self) -> usize {
        self.input_tokens.len() + self.output_tokens.len()
    }

    /// Check if generation is complete
    pub fn is_complete(&self, eos_token_id: TokenId) -> bool {
        // Complete if reached max tokens
        if self.output_tokens.len() >= self.params.max_tokens as usize {
            return true;
        }
        // Complete if generated EOS token
        if let Some(&last_token) = self.output_tokens.last() {
            if last_token == eos_token_id {
                return true;
            }
        }
        false
    }
}

/// Reference to a physical block in GPU memory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysicalBlockRef {
    /// Index of the physical block
    pub block_idx: BlockIdx,
}

/// Logical block mapped to a physical block
#[derive(Debug, Clone)]
pub struct LogicalBlock {
    /// Logical block index within the sequence
    pub block_idx: u32,
    /// Mapped physical block (None if not yet allocated)
    pub physical_block: Option<PhysicalBlockRef>,
}

impl LogicalBlock {
    pub fn new(block_idx: u32) -> Self {
        Self {
            block_idx,
            physical_block: None,
        }
    }

    pub fn with_physical(block_idx: u32, physical: PhysicalBlockRef) -> Self {
        Self {
            block_idx,
            physical_block: Some(physical),
        }
    }
}

/// A sequence is an active request with allocated KV cache
#[derive(Debug, Clone)]
pub struct Sequence {
    /// Unique sequence identifier
    pub seq_id: SeqId,
    /// The underlying request
    pub request: Request,
    /// Logical blocks for KV cache
    pub logical_blocks: Vec<LogicalBlock>,
    /// Number of tokens that have been computed (KV cached)
    pub num_computed_tokens: u32,
    /// Number of tokens generated so far
    pub num_generated_tokens: u32,
}

impl Sequence {
    /// Create a new sequence from a request
    pub fn new(seq_id: SeqId, request: Request) -> Self {
        Self {
            seq_id,
            request,
            logical_blocks: Vec::new(),
            num_computed_tokens: 0,
            num_generated_tokens: 0,
        }
    }

    /// Get the block table (physical block indices) for GPU execution
    pub fn get_block_table(&self) -> Vec<BlockIdx> {
        self.logical_blocks
            .iter()
            .filter_map(|lb| lb.physical_block.map(|pb| pb.block_idx))
            .collect()
    }

    /// Total context length (input + generated)
    pub fn context_len(&self) -> u32 {
        self.request.input_tokens.len() as u32 + self.num_generated_tokens
    }

    /// Number of tokens to process in current step
    pub fn num_tokens_to_process(&self) -> u32 {
        match self.request.state {
            RequestState::Prefill => self.request.input_tokens.len() as u32,
            RequestState::Decode => 1,
            _ => 0,
        }
    }

    /// Input token for the next decode step.
    pub fn decode_input_token(&self) -> Option<TokenId> {
        self.request
            .output_tokens
            .last()
            .copied()
            .or_else(|| self.request.input_tokens.last().copied())
    }

    /// Position for the next decode step.
    pub fn decode_position(&self) -> Option<u32> {
        self.context_len().checked_sub(1)
    }
}

/// Memory statistics for KV cache
#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryStats {
    /// Total number of physical blocks
    pub total_blocks: u32,
    /// Number of blocks currently in use
    pub used_blocks: u32,
    /// Number of free blocks
    pub free_blocks: u32,
    /// Number of active sequences
    pub num_sequences: u32,
}

impl MemoryStats {
    /// Memory utilization ratio
    pub fn utilization(&self) -> f32 {
        if self.total_blocks == 0 {
            0.0
        } else {
            self.used_blocks as f32 / self.total_blocks as f32
        }
    }
}

/// Output from the scheduler
#[derive(Debug, Clone, Default)]
pub struct SchedulerOutput {
    /// Sequences in prefill phase
    pub prefill_sequences: Vec<Arc<Sequence>>,
    /// Sequences in decode phase
    pub decode_sequences: Vec<Arc<Sequence>>,
    /// Total number of tokens in this batch
    pub total_tokens: u32,
}

impl SchedulerOutput {
    pub fn is_empty(&self) -> bool {
        self.prefill_sequences.is_empty() && self.decode_sequences.is_empty()
    }

    pub fn num_sequences(&self) -> usize {
        self.prefill_sequences.len() + self.decode_sequences.len()
    }
}

/// Batch data for GPU execution
#[derive(Debug, Clone, Default)]
pub struct ExecutionBatch {
    /// Token IDs for all sequences (flattened)
    pub input_tokens: Vec<TokenId>,
    /// Position IDs for each token
    pub positions: Vec<u32>,
    /// Sequence lengths for attention masking
    pub seq_lens: Vec<u32>,
    /// Block tables for paged attention
    pub block_tables: Vec<Vec<BlockIdx>>,
    /// Flags indicating prefill vs decode
    pub is_prefill: Vec<bool>,
    /// Sequence IDs for result mapping
    pub seq_ids: Vec<SeqId>,
    /// Context lengths for each sequence
    pub context_lens: Vec<u32>,
}

impl ExecutionBatch {
    pub fn is_empty(&self) -> bool {
        self.seq_ids.is_empty()
    }

    pub fn num_sequences(&self) -> usize {
        self.seq_ids.len()
    }

    pub fn total_tokens(&self) -> usize {
        self.input_tokens.len()
    }
}

/// Output from GPU execution
#[derive(Debug, Clone, Default)]
pub struct ExecutionOutput {
    /// Next token for each sequence
    pub next_tokens: Vec<TokenId>,
    /// Logits if needed for sampling (optional)
    pub logits: Option<Vec<f32>>,
    /// Sequence IDs corresponding to outputs
    pub seq_ids: Vec<SeqId>,
}

/// Completed request with output
#[derive(Debug, Clone)]
pub struct CompletedRequest {
    /// Original request ID
    pub request_id: RequestId,
    /// Input text (if available)
    pub input_text: Option<String>,
    /// Generated output text
    pub output_text: String,
    /// Generated tokens
    pub output_tokens: Vec<TokenId>,
    /// Whether completed successfully
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_state_is_active() {
        assert!(!RequestState::Pending.is_active());
        assert!(RequestState::Prefill.is_active());
        assert!(RequestState::Decode.is_active());
        assert!(!RequestState::Completed.is_active());
        assert!(!RequestState::Failed("error".to_string()).is_active());
    }

    #[test]
    fn test_generation_params_validation() {
        let valid = GenerationParams {
            max_tokens: 100,
            temperature: 1.0,
            top_p: 0.9,
        };
        assert!(valid.validate().is_ok());
        assert!(valid.is_valid());

        let invalid_max_tokens = GenerationParams {
            max_tokens: 0,
            temperature: 1.0,
            top_p: 0.9,
        };
        assert!(invalid_max_tokens.validate().is_err());
        assert!(!invalid_max_tokens.is_valid());

        let invalid_temp = GenerationParams {
            max_tokens: 100,
            temperature: 0.0,
            top_p: 0.9,
        };
        assert!(invalid_temp.validate().is_err());

        let invalid_top_p = GenerationParams {
            max_tokens: 100,
            temperature: 1.0,
            top_p: 1.5,
        };
        assert!(invalid_top_p.validate().is_err());
    }

    #[test]
    fn test_request_is_complete() {
        let mut request = Request::new(
            1,
            vec![1, 2, 3],
            GenerationParams {
                max_tokens: 5,
                temperature: 1.0,
                top_p: 1.0,
            },
        );

        let eos_token = 0;

        // Not complete initially
        assert!(!request.is_complete(eos_token));

        // Add some tokens
        request.output_tokens = vec![10, 11, 12];
        assert!(!request.is_complete(eos_token));

        // Add EOS token
        request.output_tokens.push(eos_token);
        assert!(request.is_complete(eos_token));

        // Or reach max tokens
        request.output_tokens = vec![10, 11, 12, 13, 14];
        assert!(request.is_complete(eos_token));
    }

    #[test]
    fn test_memory_stats_utilization() {
        let stats = MemoryStats {
            total_blocks: 100,
            used_blocks: 25,
            free_blocks: 75,
            num_sequences: 5,
        };
        assert!((stats.utilization() - 0.25).abs() < 0.001);

        let empty = MemoryStats::default();
        assert_eq!(empty.utilization(), 0.0);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// **Feature: heterogeneous-inference-system, Property 2: Parameter Validation Correctness**
        /// *For any* generation parameters (max_tokens, temperature, top_p), the validation
        /// function shall return true if and only if all parameters are within their acceptable
        /// ranges (max_tokens > 0, 0 < temperature <= 2.0, 0 < top_p <= 1.0).
        /// **Validates: Requirements 1.3**
        #[test]
        fn prop_parameter_validation(
            max_tokens in 0u32..1000,
            temperature in -1.0f32..3.0,
            top_p in -0.5f32..1.5,
        ) {
            let params = GenerationParams {
                max_tokens,
                temperature,
                top_p,
            };

            let validation_result = params.validate();
            let is_valid = params.is_valid();

            // Expected validity based on parameter ranges
            let expected_valid = max_tokens > 0
                && temperature > 0.0
                && temperature <= 2.0
                && top_p > 0.0
                && top_p <= 1.0;

            // Property: validation result matches expected validity
            prop_assert_eq!(
                validation_result.is_ok(),
                expected_valid,
                "Validation mismatch for params: max_tokens={}, temp={}, top_p={}",
                max_tokens, temperature, top_p
            );

            // Property: is_valid() is consistent with validate()
            prop_assert_eq!(
                is_valid,
                expected_valid,
                "is_valid() inconsistent with validate() for params: {:?}",
                params
            );
        }

        /// Property test for boundary conditions
        #[test]
        fn prop_parameter_boundaries(
            valid_max_tokens in 1u32..1000,
        ) {
            // Test exact boundary: temperature = 2.0 should be valid
            let params_temp_boundary = GenerationParams {
                max_tokens: valid_max_tokens,
                temperature: 2.0,
                top_p: 0.5,
            };
            prop_assert!(params_temp_boundary.is_valid());

            // Test exact boundary: top_p = 1.0 should be valid
            let params_top_p_boundary = GenerationParams {
                max_tokens: valid_max_tokens,
                temperature: 1.0,
                top_p: 1.0,
            };
            prop_assert!(params_top_p_boundary.is_valid());

            // Test just above boundary: temperature > 2.0 should be invalid
            let params_temp_over = GenerationParams {
                max_tokens: valid_max_tokens,
                temperature: 2.001,
                top_p: 0.5,
            };
            prop_assert!(!params_temp_over.is_valid());

            // Test just above boundary: top_p > 1.0 should be invalid
            let params_top_p_over = GenerationParams {
                max_tokens: valid_max_tokens,
                temperature: 1.0,
                top_p: 1.001,
            };
            prop_assert!(!params_top_p_over.is_valid());

            // Test zero boundaries
            let params_zero_temp = GenerationParams {
                max_tokens: valid_max_tokens,
                temperature: 0.0,
                top_p: 0.5,
            };
            prop_assert!(!params_zero_temp.is_valid());

            let params_zero_top_p = GenerationParams {
                max_tokens: valid_max_tokens,
                temperature: 1.0,
                top_p: 0.0,
            };
            prop_assert!(!params_zero_top_p.is_valid());

            let params_zero_max_tokens = GenerationParams {
                max_tokens: 0,
                temperature: 1.0,
                top_p: 0.5,
            };
            prop_assert!(!params_zero_max_tokens.is_valid());
        }
    }
}
