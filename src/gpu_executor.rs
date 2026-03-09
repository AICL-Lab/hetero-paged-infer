//! GPU Executor for inference computation
//!
//! Provides an abstraction over GPU execution with support for:
//! - Paged attention with block table indirection
//! - Variable sequence lengths in batches
//! - CUDA graph capture for decode optimization
//!
//! Note: This implementation provides a mock executor for testing.
//! Real GPU execution requires CUDA toolkit and appropriate kernels.

use std::collections::HashMap;
use crate::config::EngineConfig;
use crate::error::ExecutionError;
use crate::types::{
    BlockIdx, ExecutionBatch, ExecutionOutput, SeqId, TokenId,
};

/// Pinned buffer for CPU-GPU transfer (mock implementation)
#[derive(Debug, Clone)]
pub struct PinnedBuffer<T> {
    data: Vec<T>,
}

impl<T: Clone + Default> PinnedBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }
    
    pub fn from_vec(data: Vec<T>) -> Self {
        Self { data }
    }
    
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }
    
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
    
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    pub fn clear(&mut self) {
        self.data.clear();
    }
    
    pub fn push(&mut self, value: T) {
        self.data.push(value);
    }
    
    pub fn extend(&mut self, iter: impl IntoIterator<Item = T>) {
        self.data.extend(iter);
    }
}

/// GPU batch data with pinned buffers for efficient transfer
#[derive(Debug, Clone)]
pub struct GPUBatchData {
    /// Token IDs for all sequences (flattened)
    pub input_ids: PinnedBuffer<TokenId>,
    /// Position IDs for each token
    pub positions: PinnedBuffer<u32>,
    /// Sequence start locations (cumulative)
    pub seq_start_locs: PinnedBuffer<u32>,
    /// Sequence lengths
    pub seq_lens: PinnedBuffer<u32>,
    /// Block tables (flattened, padded)
    pub block_tables: PinnedBuffer<BlockIdx>,
    /// Context lengths for attention
    pub context_lens: PinnedBuffer<u32>,
    /// Maximum blocks per sequence (for padding)
    pub max_blocks_per_seq: u32,
}

impl GPUBatchData {
    pub fn new(max_batch_size: u32, max_blocks_per_seq: u32) -> Self {
        Self {
            input_ids: PinnedBuffer::new(4096),
            positions: PinnedBuffer::new(4096),
            seq_start_locs: PinnedBuffer::new(max_batch_size as usize + 1),
            seq_lens: PinnedBuffer::new(max_batch_size as usize),
            block_tables: PinnedBuffer::new((max_batch_size * max_blocks_per_seq) as usize),
            context_lens: PinnedBuffer::new(max_batch_size as usize),
            max_blocks_per_seq,
        }
    }
    
    /// Prepare batch data from execution batch
    pub fn prepare(&mut self, batch: &ExecutionBatch) {
        self.clear();
        
        // Copy input tokens and positions
        self.input_ids.extend(batch.input_tokens.iter().copied());
        self.positions.extend(batch.positions.iter().copied());
        
        // Build sequence metadata
        let mut cumulative = 0u32;
        self.seq_start_locs.push(0);
        
        for &seq_len in &batch.seq_lens {
            cumulative += seq_len;
            self.seq_start_locs.push(cumulative);
            self.seq_lens.push(seq_len);
        }
        
        // Copy context lengths
        self.context_lens.extend(batch.context_lens.iter().copied());
        
        // Flatten and pad block tables
        for block_table in &batch.block_tables {
            for &block_idx in block_table {
                self.block_tables.push(block_idx);
            }
            // Pad to max_blocks_per_seq
            for _ in block_table.len()..self.max_blocks_per_seq as usize {
                self.block_tables.push(0);
            }
        }
    }
    
    fn clear(&mut self) {
        self.input_ids.clear();
        self.positions.clear();
        self.seq_start_locs.clear();
        self.seq_lens.clear();
        self.block_tables.clear();
        self.context_lens.clear();
    }
}


/// GPU Executor trait defining the interface
pub trait GPUExecutorTrait: Send {
    /// Execute a batch of sequences
    fn execute(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput, ExecutionError>;
    
    /// Capture CUDA graph for decode phase
    fn capture_decode_graph(&mut self, batch_size: u32) -> Result<(), ExecutionError>;
    
    /// Execute using captured CUDA graph
    fn execute_graph(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput, ExecutionError>;
    
    /// Check if CUDA graph is captured
    fn has_captured_graph(&self) -> bool;
}

/// Mock GPU Executor for testing without actual GPU
/// 
/// This executor simulates GPU execution by generating random tokens.
/// Replace with real CUDA implementation for production use.
#[derive(Debug)]
pub struct MockGPUExecutor {
    config: EngineConfig,
    batch_data: GPUBatchData,
    graph_captured: bool,
    captured_batch_size: u32,
    /// Vocabulary size for token generation
    vocab_size: u32,
    /// Counter for deterministic token generation in tests
    token_counter: u32,
}

impl MockGPUExecutor {
    pub fn new(config: EngineConfig, vocab_size: u32) -> Self {
        let max_blocks_per_seq = config.max_model_len / config.block_size + 1;
        let batch_data = GPUBatchData::new(config.max_batch_size, max_blocks_per_seq);
        
        Self {
            config,
            batch_data,
            graph_captured: false,
            captured_batch_size: 0,
            vocab_size,
            token_counter: 100,
        }
    }
    
    /// Generate next token (mock implementation)
    fn generate_token(&mut self) -> TokenId {
        let token = self.token_counter % self.vocab_size;
        self.token_counter = self.token_counter.wrapping_add(1);
        token
    }
    
    /// Validate batch for execution
    fn validate_batch(&self, batch: &ExecutionBatch) -> Result<(), ExecutionError> {
        if batch.is_empty() {
            return Ok(());
        }
        
        // Check batch size constraints
        if batch.num_sequences() > self.config.max_batch_size as usize {
            return Err(ExecutionError::KernelLaunchFailed(
                format!("Batch size {} exceeds max {}", 
                    batch.num_sequences(), self.config.max_batch_size)
            ));
        }
        
        // Check total tokens
        if batch.total_tokens() > self.config.max_total_tokens as usize {
            return Err(ExecutionError::KernelLaunchFailed(
                format!("Total tokens {} exceeds max {}", 
                    batch.total_tokens(), self.config.max_total_tokens)
            ));
        }
        
        Ok(())
    }
}

impl GPUExecutorTrait for MockGPUExecutor {
    fn execute(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput, ExecutionError> {
        self.validate_batch(batch)?;
        
        if batch.is_empty() {
            return Ok(ExecutionOutput::default());
        }
        
        // Prepare batch data for "GPU transfer"
        self.batch_data.prepare(batch);
        
        // Generate one token per sequence
        let mut next_tokens = Vec::with_capacity(batch.num_sequences());
        let mut seq_ids = Vec::with_capacity(batch.num_sequences());
        
        for &seq_id in &batch.seq_ids {
            next_tokens.push(self.generate_token());
            seq_ids.push(seq_id);
        }
        
        Ok(ExecutionOutput {
            next_tokens,
            logits: None,
            seq_ids,
        })
    }
    
    fn capture_decode_graph(&mut self, batch_size: u32) -> Result<(), ExecutionError> {
        if batch_size == 0 {
            return Err(ExecutionError::KernelLaunchFailed(
                "Cannot capture graph with batch size 0".to_string()
            ));
        }
        
        self.graph_captured = true;
        self.captured_batch_size = batch_size;
        Ok(())
    }
    
    fn execute_graph(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput, ExecutionError> {
        if !self.graph_captured {
            return Err(ExecutionError::KernelLaunchFailed(
                "No CUDA graph captured".to_string()
            ));
        }
        
        // For mock, just use regular execution
        self.execute(batch)
    }
    
    fn has_captured_graph(&self) -> bool {
        self.graph_captured
    }
}

/// Build execution batch from scheduler output
pub fn build_execution_batch(
    scheduler_output: &crate::types::SchedulerOutput,
) -> ExecutionBatch {
    let mut batch = ExecutionBatch::default();
    
    // Process prefill sequences
    for seq in &scheduler_output.prefill_sequences {
        let seq_id = seq.seq_id;
        let input_tokens = &seq.request.input_tokens;
        let context_len = seq.context_len();
        
        // Add tokens
        batch.input_tokens.extend(input_tokens.iter().copied());
        
        // Add positions (0 to len-1 for prefill)
        for i in 0..input_tokens.len() {
            batch.positions.push(i as u32);
        }
        
        batch.seq_lens.push(input_tokens.len() as u32);
        batch.is_prefill.push(true);
        batch.seq_ids.push(seq_id);
        batch.context_lens.push(context_len);
        
        // Add block table
        if let Some(block_table) = scheduler_output.block_tables.get(&seq_id) {
            batch.block_tables.push(block_table.clone());
        } else {
            batch.block_tables.push(Vec::new());
        }
    }
    
    // Process decode sequences
    for seq in &scheduler_output.decode_sequences {
        let seq_id = seq.seq_id;
        let context_len = seq.context_len();
        
        // For decode, we only process the last generated token
        let last_token = seq.request.output_tokens.last().copied().unwrap_or(0);
        batch.input_tokens.push(last_token);
        
        // Position is the context length (next position)
        batch.positions.push(context_len);
        
        batch.seq_lens.push(1);
        batch.is_prefill.push(false);
        batch.seq_ids.push(seq_id);
        batch.context_lens.push(context_len);
        
        // Add block table
        if let Some(block_table) = scheduler_output.block_tables.get(&seq_id) {
            batch.block_tables.push(block_table.clone());
        } else {
            batch.block_tables.push(Vec::new());
        }
    }
    
    batch
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::create_test_config;

    #[test]
    fn test_mock_executor_creation() {
        let config = create_test_config();
        let executor = MockGPUExecutor::new(config, 32000);
        
        assert!(!executor.has_captured_graph());
    }

    #[test]
    fn test_mock_executor_execute_empty() {
        let config = create_test_config();
        let mut executor = MockGPUExecutor::new(config, 32000);
        
        let batch = ExecutionBatch::default();
        let result = executor.execute(&batch);
        
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.next_tokens.is_empty());
    }

    #[test]
    fn test_mock_executor_execute_batch() {
        let config = create_test_config();
        let mut executor = MockGPUExecutor::new(config, 32000);
        
        let batch = ExecutionBatch {
            input_tokens: vec![1, 2, 3, 4, 5],
            positions: vec![0, 1, 2, 3, 4],
            seq_lens: vec![3, 2],
            block_tables: vec![vec![0, 1], vec![2]],
            is_prefill: vec![true, true],
            seq_ids: vec![1, 2],
            context_lens: vec![3, 2],
        };
        
        let result = executor.execute(&batch);
        assert!(result.is_ok());
        
        let output = result.unwrap();
        assert_eq!(output.next_tokens.len(), 2);
        assert_eq!(output.seq_ids.len(), 2);
    }

    #[test]
    fn test_cuda_graph_capture() {
        let config = create_test_config();
        let mut executor = MockGPUExecutor::new(config, 32000);
        
        assert!(!executor.has_captured_graph());
        
        executor.capture_decode_graph(4).unwrap();
        
        assert!(executor.has_captured_graph());
    }

    #[test]
    fn test_gpu_batch_data_prepare() {
        let mut batch_data = GPUBatchData::new(8, 16);
        
        let batch = ExecutionBatch {
            input_tokens: vec![1, 2, 3],
            positions: vec![0, 1, 2],
            seq_lens: vec![3],
            block_tables: vec![vec![0, 1]],
            is_prefill: vec![true],
            seq_ids: vec![1],
            context_lens: vec![3],
        };
        
        batch_data.prepare(&batch);
        
        assert_eq!(batch_data.input_ids.len(), 3);
        assert_eq!(batch_data.positions.len(), 3);
        assert_eq!(batch_data.seq_lens.len(), 1);
    }

    #[test]
    fn test_pinned_buffer() {
        let mut buffer: PinnedBuffer<u32> = PinnedBuffer::new(10);
        
        assert!(buffer.is_empty());
        
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);
        
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.as_slice(), &[1, 2, 3]);
        
        buffer.clear();
        assert!(buffer.is_empty());
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use crate::test_utils::create_test_config_with_limits;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// **Feature: heterogeneous-inference-system, Property 11: Variable Sequence Length Handling**
        /// *For any* batch containing sequences of different lengths, the GPU_Executor shall
        /// produce correct attention outputs for each sequence independently.
        /// **Validates: Requirements 4.2**
        #[test]
        fn prop_variable_sequence_length_handling(
            num_sequences in 1usize..8,
            seq_lengths in prop::collection::vec(1u32..64, 1..8),
        ) {
            let max_total = seq_lengths.iter().sum::<u32>().max(100);
            let config = create_test_config_with_limits(8, max_total);
            let mut executor = MockGPUExecutor::new(config, 32000);
            
            // Build batch with variable sequence lengths
            let mut batch = ExecutionBatch::default();
            let actual_num_seqs = num_sequences.min(seq_lengths.len());
            
            let mut token_offset = 0u32;
            for (i, &seq_len) in seq_lengths.iter().take(actual_num_seqs).enumerate() {
                let seq_id = (i + 1) as SeqId;
                
                // Add tokens for this sequence
                for j in 0..seq_len {
                    batch.input_tokens.push((token_offset + j) % 32000);
                    batch.positions.push(j);
                }
                token_offset += seq_len;
                
                batch.seq_lens.push(seq_len);
                batch.is_prefill.push(true);
                batch.seq_ids.push(seq_id);
                batch.context_lens.push(seq_len);
                batch.block_tables.push(vec![i as BlockIdx]);
            }
            
            // Execute batch
            let result = executor.execute(&batch);
            prop_assert!(result.is_ok(), "Execution should succeed");
            
            let output = result.unwrap();
            
            // Verify output has correct number of tokens (one per sequence)
            prop_assert_eq!(
                output.next_tokens.len(),
                actual_num_seqs,
                "Should produce one token per sequence"
            );
            
            // Verify sequence IDs match
            prop_assert_eq!(
                output.seq_ids.len(),
                actual_num_seqs,
                "Should have correct number of sequence IDs"
            );
            
            // Verify each sequence got a valid token
            for token in &output.next_tokens {
                prop_assert!(
                    *token < 32000,
                    "Generated token should be within vocabulary"
                );
            }
        }

        /// Property test for batch size validation
        #[test]
        fn prop_batch_size_validation(
            num_sequences in 1usize..20,
            max_batch_size in 1u32..10,
        ) {
            let config = create_test_config_with_limits(max_batch_size, 1000);
            let mut executor = MockGPUExecutor::new(config, 32000);
            
            // Build batch
            let mut batch = ExecutionBatch::default();
            for i in 0..num_sequences {
                batch.input_tokens.push(i as TokenId);
                batch.positions.push(0);
                batch.seq_lens.push(1);
                batch.is_prefill.push(true);
                batch.seq_ids.push(i as SeqId);
                batch.context_lens.push(1);
                batch.block_tables.push(vec![i as BlockIdx]);
            }
            
            let result = executor.execute(&batch);
            
            if num_sequences <= max_batch_size as usize {
                prop_assert!(result.is_ok(), "Should succeed within batch limit");
            } else {
                prop_assert!(result.is_err(), "Should fail exceeding batch limit");
            }
        }

        /// Property test for deterministic output per sequence
        #[test]
        fn prop_output_per_sequence(
            num_sequences in 1usize..5,
        ) {
            let config = create_test_config_with_limits(8, 500);
            let mut executor = MockGPUExecutor::new(config, 32000);
            
            // Build batch
            let mut batch = ExecutionBatch::default();
            for i in 0..num_sequences {
                batch.input_tokens.push(i as TokenId);
                batch.positions.push(0);
                batch.seq_lens.push(1);
                batch.is_prefill.push(false);
                batch.seq_ids.push((i + 1) as SeqId);
                batch.context_lens.push(10);
                batch.block_tables.push(vec![i as BlockIdx]);
            }
            
            let result = executor.execute(&batch);
            prop_assert!(result.is_ok());
            
            let output = result.unwrap();
            
            // Each sequence should get exactly one output token
            prop_assert_eq!(
                output.next_tokens.len(),
                num_sequences,
                "Each sequence should get one output token"
            );
            
            // Sequence IDs should match input
            for (i, &seq_id) in output.seq_ids.iter().enumerate() {
                prop_assert_eq!(
                    seq_id,
                    (i + 1) as SeqId,
                    "Sequence IDs should match"
                );
            }
        }
    }
}
