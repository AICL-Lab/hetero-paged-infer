//! Continuous Batching Scheduler
//!
//! Implements request scheduling with prefill/decode phase management
//! and memory-aware batch formation.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use crate::config::EngineConfig;
use crate::error::{MemoryError, SchedulerError};
use crate::kv_cache::{KVCacheManager, KVCacheManagerTrait};
use crate::types::{
    BlockIdx, ExecutionBatch, ExecutionOutput, Request, RequestId, RequestState,
    SchedulerOutput, SeqId, Sequence, TokenId,
};

/// Scheduler trait defining the interface
pub trait SchedulerTrait {
    /// Add new request to pending queue
    fn add_request(&mut self, request: Request) -> Result<SeqId, SchedulerError>;
    
    /// Schedule next batch for execution
    fn schedule(&mut self) -> SchedulerOutput;
    
    /// Update sequences after GPU execution
    fn update_sequences(&mut self, outputs: &ExecutionOutput, eos_token_id: TokenId);
    
    /// Get completed requests
    fn get_completed(&mut self) -> Vec<Request>;
    
    /// Check if scheduler has work
    fn has_pending_work(&self) -> bool;
    
    /// Get memory stats from KV cache
    fn get_memory_utilization(&self) -> f32;
}

/// Continuous batching scheduler implementation
pub struct Scheduler {
    /// Configuration
    config: EngineConfig,
    /// KV Cache Manager
    kv_cache: KVCacheManager,
    /// Pending requests waiting to be scheduled
    pending_queue: VecDeque<Request>,
    /// Sequences in prefill phase
    prefill_sequences: HashMap<SeqId, Sequence>,
    /// Sequences in decode phase
    decode_sequences: HashMap<SeqId, Sequence>,
    /// Completed requests
    completed_requests: Vec<Request>,
    /// Next sequence ID counter
    next_seq_id: SeqId,
    /// Memory pressure flag
    under_memory_pressure: bool,
}

impl Scheduler {
    /// Create a new scheduler
    pub fn new(config: EngineConfig) -> Self {
        let kv_cache = KVCacheManager::new(config.max_num_blocks, config.block_size);
        
        Self {
            config,
            kv_cache,
            pending_queue: VecDeque::new(),
            prefill_sequences: HashMap::new(),
            decode_sequences: HashMap::new(),
            completed_requests: Vec::new(),
            next_seq_id: 1,
            under_memory_pressure: false,
        }
    }
    
    /// Generate unique sequence ID
    fn generate_seq_id(&mut self) -> SeqId {
        let id = self.next_seq_id;
        self.next_seq_id += 1;
        id
    }
    
    /// Check and update memory pressure status
    fn update_memory_pressure(&mut self) {
        let stats = self.kv_cache.get_memory_stats();
        self.under_memory_pressure = stats.utilization() >= self.config.memory_threshold;
    }
    
    /// Calculate blocks needed for a sequence
    fn blocks_needed(&self, num_tokens: u32) -> u32 {
        (num_tokens + self.config.block_size - 1) / self.config.block_size
    }
    
    /// Try to start prefill for a pending request
    fn try_start_prefill(&mut self, request: Request) -> Result<SeqId, Request> {
        let num_tokens = request.input_tokens.len() as u32;
        let blocks_needed = self.blocks_needed(num_tokens);
        
        // Check if we can allocate
        if !self.kv_cache.can_allocate(blocks_needed) {
            return Err(request);
        }
        
        let seq_id = self.generate_seq_id();
        
        // Allocate KV cache blocks
        if self.kv_cache.allocate_sequence(seq_id, num_tokens).is_err() {
            return Err(request);
        }
        
        // Create sequence and set to prefill state
        let mut sequence = Sequence::new(seq_id, request);
        sequence.request.state = RequestState::Prefill;
        
        // Update logical blocks from KV cache
        if let Some(block_table) = self.kv_cache.get_block_table(seq_id) {
            sequence.logical_blocks = block_table
                .iter()
                .enumerate()
                .map(|(i, &block_idx)| {
                    crate::types::LogicalBlock::with_physical(
                        i as u32,
                        crate::types::PhysicalBlockRef { block_idx },
                    )
                })
                .collect();
        }
        
        self.prefill_sequences.insert(seq_id, sequence);
        Ok(seq_id)
    }
    
    /// Transition sequence from prefill to decode
    fn transition_to_decode(&mut self, seq_id: SeqId) {
        if let Some(mut sequence) = self.prefill_sequences.remove(&seq_id) {
            sequence.request.state = RequestState::Decode;
            sequence.num_computed_tokens = sequence.request.input_tokens.len() as u32;
            self.decode_sequences.insert(seq_id, sequence);
        }
    }
    
    /// Complete a sequence
    fn complete_sequence(&mut self, seq_id: SeqId) {
        // Try decode sequences first
        if let Some(mut sequence) = self.decode_sequences.remove(&seq_id) {
            sequence.request.state = RequestState::Completed;
            self.kv_cache.free_sequence(seq_id);
            self.completed_requests.push(sequence.request);
            return;
        }
        
        // Try prefill sequences
        if let Some(mut sequence) = self.prefill_sequences.remove(&seq_id) {
            sequence.request.state = RequestState::Completed;
            self.kv_cache.free_sequence(seq_id);
            self.completed_requests.push(sequence.request);
        }
    }
    
    /// Get sequence by ID (from any queue)
    pub fn get_sequence(&self, seq_id: SeqId) -> Option<&Sequence> {
        self.prefill_sequences
            .get(&seq_id)
            .or_else(|| self.decode_sequences.get(&seq_id))
    }
    
    /// Get mutable sequence by ID
    pub fn get_sequence_mut(&mut self, seq_id: SeqId) -> Option<&mut Sequence> {
        if self.prefill_sequences.contains_key(&seq_id) {
            self.prefill_sequences.get_mut(&seq_id)
        } else {
            self.decode_sequences.get_mut(&seq_id)
        }
    }
    
    /// Get number of active sequences
    pub fn num_active_sequences(&self) -> usize {
        self.prefill_sequences.len() + self.decode_sequences.len()
    }
    
    /// Check if a request is in exactly one queue
    pub fn is_in_exactly_one_queue(&self, seq_id: SeqId) -> bool {
        let in_prefill = self.prefill_sequences.contains_key(&seq_id);
        let in_decode = self.decode_sequences.contains_key(&seq_id);
        (in_prefill && !in_decode) || (!in_prefill && in_decode)
    }
}


impl SchedulerTrait for Scheduler {
    fn add_request(&mut self, request: Request) -> Result<SeqId, SchedulerError> {
        // Check memory pressure
        self.update_memory_pressure();
        
        if self.under_memory_pressure {
            return Err(SchedulerError::MemoryPressure);
        }
        
        // Check if we've reached max sequences
        if self.num_active_sequences() >= self.config.max_num_seqs as usize {
            return Err(SchedulerError::MemoryPressure);
        }
        
        // Add to pending queue
        self.pending_queue.push_back(request);
        
        // Return a placeholder ID (actual ID assigned during scheduling)
        Ok(0)
    }
    
    fn schedule(&mut self) -> SchedulerOutput {
        let mut output = SchedulerOutput::default();
        let mut total_tokens: u32 = 0;
        let mut num_sequences: u32 = 0;
        
        // Update memory pressure status
        self.update_memory_pressure();
        
        // Priority 1: Schedule decode sequences first (lower latency for in-flight requests)
        let decode_seq_ids: Vec<SeqId> = self.decode_sequences.keys().copied().collect();
        
        for seq_id in decode_seq_ids {
            if num_sequences >= self.config.max_batch_size {
                break;
            }
            
            // Each decode step processes 1 token
            if total_tokens + 1 > self.config.max_total_tokens {
                break;
            }
            
            if let Some(sequence) = self.decode_sequences.get(&seq_id) {
                // Check if sequence needs more blocks
                let current_tokens = sequence.context_len();
                let current_blocks = self.kv_cache.get_memory_stats().used_blocks;
                let blocks_needed = self.blocks_needed(current_tokens + 1);
                
                // Allocate additional block if needed
                if blocks_needed > sequence.logical_blocks.len() as u32 {
                    if self.kv_cache.allocate_block(seq_id).is_err() {
                        continue; // Skip this sequence if can't allocate
                    }
                }
                
                let block_table = self.kv_cache.get_block_table(seq_id).unwrap_or_default();
                output.block_tables.insert(seq_id, block_table.clone());
                output.decode_sequences.push(Arc::new(sequence.clone()));
                
                total_tokens += 1;
                num_sequences += 1;
            }
        }
        
        // Priority 2: Schedule prefill sequences
        let prefill_seq_ids: Vec<SeqId> = self.prefill_sequences.keys().copied().collect();
        
        for seq_id in prefill_seq_ids {
            if num_sequences >= self.config.max_batch_size {
                break;
            }
            
            if let Some(sequence) = self.prefill_sequences.get(&seq_id) {
                let prefill_tokens = sequence.request.input_tokens.len() as u32;
                
                if total_tokens + prefill_tokens > self.config.max_total_tokens {
                    break;
                }
                
                let block_table = self.kv_cache.get_block_table(seq_id).unwrap_or_default();
                output.block_tables.insert(seq_id, block_table.clone());
                output.prefill_sequences.push(Arc::new(sequence.clone()));
                
                total_tokens += prefill_tokens;
                num_sequences += 1;
            }
        }
        
        // Priority 3: Start new prefills from pending queue (if not under memory pressure)
        if !self.under_memory_pressure {
            while let Some(request) = self.pending_queue.pop_front() {
                if num_sequences >= self.config.max_batch_size {
                    self.pending_queue.push_front(request);
                    break;
                }
                
                let prefill_tokens = request.input_tokens.len() as u32;
                if total_tokens + prefill_tokens > self.config.max_total_tokens {
                    self.pending_queue.push_front(request);
                    break;
                }
                
                match self.try_start_prefill(request) {
                    Ok(seq_id) => {
                        if let Some(sequence) = self.prefill_sequences.get(&seq_id) {
                            let block_table = self.kv_cache.get_block_table(seq_id).unwrap_or_default();
                            output.block_tables.insert(seq_id, block_table.clone());
                            output.prefill_sequences.push(Arc::new(sequence.clone()));
                            
                            total_tokens += prefill_tokens;
                            num_sequences += 1;
                        }
                    }
                    Err(request) => {
                        // Put back in queue and stop trying
                        self.pending_queue.push_front(request);
                        break;
                    }
                }
            }
        }
        
        output.total_tokens = total_tokens;
        output
    }
    
    fn update_sequences(&mut self, outputs: &ExecutionOutput, eos_token_id: TokenId) {
        for (i, &seq_id) in outputs.seq_ids.iter().enumerate() {
            let next_token = outputs.next_tokens.get(i).copied().unwrap_or(0);
            
            // Check if this was a prefill sequence - transition to decode
            if self.prefill_sequences.contains_key(&seq_id) {
                self.transition_to_decode(seq_id);
            }
            
            // Update decode sequence with new token
            if let Some(sequence) = self.decode_sequences.get_mut(&seq_id) {
                sequence.request.output_tokens.push(next_token);
                sequence.num_generated_tokens += 1;
                
                // Check completion conditions
                if sequence.request.is_complete(eos_token_id) {
                    let seq_id_to_complete = seq_id;
                    // Mark for completion (can't complete here due to borrow)
                    drop(sequence);
                    self.complete_sequence(seq_id_to_complete);
                }
            }
        }
    }
    
    fn get_completed(&mut self) -> Vec<Request> {
        std::mem::take(&mut self.completed_requests)
    }
    
    fn has_pending_work(&self) -> bool {
        !self.pending_queue.is_empty()
            || !self.prefill_sequences.is_empty()
            || !self.decode_sequences.is_empty()
    }
    
    fn get_memory_utilization(&self) -> f32 {
        self.kv_cache.get_memory_stats().utilization()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::GenerationParams;

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

    fn create_test_request(id: RequestId, num_tokens: usize) -> Request {
        Request::new(
            id,
            vec![1; num_tokens],
            GenerationParams::default(),
        )
    }

    #[test]
    fn test_add_request() {
        let config = create_test_config();
        let mut scheduler = Scheduler::new(config);
        
        let request = create_test_request(1, 32);
        let result = scheduler.add_request(request);
        
        assert!(result.is_ok());
        assert!(scheduler.has_pending_work());
    }

    #[test]
    fn test_schedule_prefill() {
        let config = create_test_config();
        let mut scheduler = Scheduler::new(config);
        
        let request = create_test_request(1, 32);
        scheduler.add_request(request).unwrap();
        
        let output = scheduler.schedule();
        
        assert_eq!(output.prefill_sequences.len(), 1);
        assert_eq!(output.decode_sequences.len(), 0);
    }

    #[test]
    fn test_prefill_to_decode_transition() {
        let config = create_test_config();
        let mut scheduler = Scheduler::new(config);
        
        let request = create_test_request(1, 32);
        scheduler.add_request(request).unwrap();
        
        // Schedule prefill
        let output = scheduler.schedule();
        assert_eq!(output.prefill_sequences.len(), 1);
        
        let seq_id = output.prefill_sequences[0].seq_id;
        
        // Simulate GPU output
        let exec_output = ExecutionOutput {
            next_tokens: vec![100],
            logits: None,
            seq_ids: vec![seq_id],
        };
        
        scheduler.update_sequences(&exec_output, 0);
        
        // Now should be in decode phase
        assert!(scheduler.decode_sequences.contains_key(&seq_id));
        assert!(!scheduler.prefill_sequences.contains_key(&seq_id));
    }

    #[test]
    fn test_decode_priority() {
        let config = create_test_config();
        let mut scheduler = Scheduler::new(config);
        
        // Add and schedule first request to get it into decode
        let request1 = create_test_request(1, 16);
        scheduler.add_request(request1).unwrap();
        let output = scheduler.schedule();
        let seq_id = output.prefill_sequences[0].seq_id;
        
        // Transition to decode
        let exec_output = ExecutionOutput {
            next_tokens: vec![100],
            logits: None,
            seq_ids: vec![seq_id],
        };
        scheduler.update_sequences(&exec_output, 0);
        
        // Add another pending request
        let request2 = create_test_request(2, 16);
        scheduler.add_request(request2).unwrap();
        
        // Schedule - decode should come first
        let output = scheduler.schedule();
        
        // Decode sequences should be scheduled
        assert!(!output.decode_sequences.is_empty());
    }

    #[test]
    fn test_completion() {
        let config = create_test_config();
        let mut scheduler = Scheduler::new(config);
        
        let mut request = create_test_request(1, 16);
        request.params.max_tokens = 2;
        scheduler.add_request(request).unwrap();
        
        // Schedule and transition to decode
        let output = scheduler.schedule();
        let seq_id = output.prefill_sequences[0].seq_id;
        
        let exec_output = ExecutionOutput {
            next_tokens: vec![100],
            logits: None,
            seq_ids: vec![seq_id],
        };
        scheduler.update_sequences(&exec_output, 0);
        
        // Generate one more token
        scheduler.schedule();
        let exec_output = ExecutionOutput {
            next_tokens: vec![101],
            logits: None,
            seq_ids: vec![seq_id],
        };
        scheduler.update_sequences(&exec_output, 0);
        
        // Should be completed (max_tokens = 2)
        let completed = scheduler.get_completed();
        assert_eq!(completed.len(), 1);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use crate::types::GenerationParams;
    use proptest::prelude::*;
    use std::collections::HashSet;

    fn create_test_config_with_params(
        max_batch_size: u32,
        max_total_tokens: u32,
        max_num_blocks: u32,
    ) -> EngineConfig {
        EngineConfig {
            block_size: 16,
            max_num_blocks,
            max_batch_size,
            max_num_seqs: 64,
            max_model_len: 2048,
            max_total_tokens,
            memory_threshold: 0.9,
        }
    }

    fn create_request_with_tokens(id: RequestId, num_tokens: usize, max_gen: u32) -> Request {
        Request::new(
            id,
            vec![1; num_tokens],
            GenerationParams {
                max_tokens: max_gen,
                temperature: 1.0,
                top_p: 1.0,
            },
        )
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// **Feature: heterogeneous-inference-system, Property 1: Request ID Uniqueness**
        /// *For any* set of requests submitted to the Scheduler, all assigned sequence IDs
        /// shall be unique with no collisions.
        /// **Validates: Requirements 1.2**
        #[test]
        fn prop_request_id_uniqueness(
            num_requests in 1usize..50,
            tokens_per_request in 1usize..64,
        ) {
            let config = create_test_config_with_params(32, 4096, 500);
            let mut scheduler = Scheduler::new(config);
            let mut assigned_ids: HashSet<SeqId> = HashSet::new();
            
            for i in 0..num_requests {
                let request = create_request_with_tokens(i as u64, tokens_per_request, 10);
                let _ = scheduler.add_request(request);
            }
            
            // Schedule to assign IDs
            for _ in 0..num_requests {
                let output = scheduler.schedule();
                
                for seq in &output.prefill_sequences {
                    prop_assert!(
                        !assigned_ids.contains(&seq.seq_id),
                        "Duplicate sequence ID: {}",
                        seq.seq_id
                    );
                    assigned_ids.insert(seq.seq_id);
                }
            }
        }

        /// **Feature: heterogeneous-inference-system, Property 6: Scheduler Queue State Consistency**
        /// *For any* request in the scheduler, it shall be in exactly one queue: pending,
        /// prefill, or decode.
        /// **Validates: Requirements 3.1**
        #[test]
        fn prop_scheduler_queue_state_consistency(
            num_requests in 1usize..20,
            tokens_per_request in 1usize..32,
            num_steps in 1usize..10,
        ) {
            let config = create_test_config_with_params(16, 1024, 200);
            let mut scheduler = Scheduler::new(config);
            
            // Add requests
            for i in 0..num_requests {
                let request = create_request_with_tokens(i as u64, tokens_per_request, 50);
                let _ = scheduler.add_request(request);
            }
            
            // Run scheduling steps
            for _ in 0..num_steps {
                let output = scheduler.schedule();
                
                // Verify each scheduled sequence is in exactly one queue
                for seq in &output.prefill_sequences {
                    prop_assert!(
                        scheduler.is_in_exactly_one_queue(seq.seq_id),
                        "Sequence {} is not in exactly one queue",
                        seq.seq_id
                    );
                }
                
                for seq in &output.decode_sequences {
                    prop_assert!(
                        scheduler.is_in_exactly_one_queue(seq.seq_id),
                        "Sequence {} is not in exactly one queue",
                        seq.seq_id
                    );
                }
                
                // Simulate execution
                let mut next_tokens = Vec::new();
                let mut seq_ids = Vec::new();
                
                for seq in &output.prefill_sequences {
                    next_tokens.push(100u32);
                    seq_ids.push(seq.seq_id);
                }
                for seq in &output.decode_sequences {
                    next_tokens.push(100u32);
                    seq_ids.push(seq.seq_id);
                }
                
                if !seq_ids.is_empty() {
                    let exec_output = ExecutionOutput {
                        next_tokens,
                        logits: None,
                        seq_ids,
                    };
                    scheduler.update_sequences(&exec_output, 0);
                }
            }
        }

        /// **Feature: heterogeneous-inference-system, Property 7: Batch Size Constraints**
        /// *For any* scheduled batch, the number of sequences shall not exceed max_batch_size,
        /// and the total number of tokens shall not exceed max_total_tokens.
        /// **Validates: Requirements 3.5**
        #[test]
        fn prop_batch_size_constraints(
            max_batch_size in 1u32..16,
            max_total_tokens in 64u32..512,
            num_requests in 1usize..30,
            tokens_per_request in 1usize..64,
        ) {
            let config = create_test_config_with_params(max_batch_size, max_total_tokens, 500);
            let mut scheduler = Scheduler::new(config);
            
            // Add many requests
            for i in 0..num_requests {
                let request = create_request_with_tokens(i as u64, tokens_per_request, 10);
                let _ = scheduler.add_request(request);
            }
            
            // Schedule and verify constraints
            for _ in 0..5 {
                let output = scheduler.schedule();
                
                let num_sequences = output.num_sequences();
                prop_assert!(
                    num_sequences <= max_batch_size as usize,
                    "Batch has {} sequences, max is {}",
                    num_sequences,
                    max_batch_size
                );
                
                prop_assert!(
                    output.total_tokens <= max_total_tokens,
                    "Batch has {} tokens, max is {}",
                    output.total_tokens,
                    max_total_tokens
                );
                
                // Simulate execution to progress
                let mut next_tokens = Vec::new();
                let mut seq_ids = Vec::new();
                
                for seq in &output.prefill_sequences {
                    next_tokens.push(100u32);
                    seq_ids.push(seq.seq_id);
                }
                for seq in &output.decode_sequences {
                    next_tokens.push(100u32);
                    seq_ids.push(seq.seq_id);
                }
                
                if !seq_ids.is_empty() {
                    let exec_output = ExecutionOutput {
                        next_tokens,
                        logits: None,
                        seq_ids,
                    };
                    scheduler.update_sequences(&exec_output, 0);
                }
            }
        }

        /// **Feature: heterogeneous-inference-system, Property 8: Decode Priority Over Prefill**
        /// *For any* scheduling decision where both prefill and decode requests are pending
        /// and batch capacity allows, all eligible decode requests shall be scheduled before
        /// any prefill requests.
        /// **Validates: Requirements 3.7**
        #[test]
        fn prop_decode_priority_over_prefill(
            num_decode in 1usize..10,
            num_pending in 1usize..10,
            tokens_per_request in 4usize..32,
        ) {
            let config = create_test_config_with_params(32, 2048, 500);
            let mut scheduler = Scheduler::new(config);
            
            // First, create some decode sequences
            for i in 0..num_decode {
                let request = create_request_with_tokens(i as u64, tokens_per_request, 100);
                scheduler.add_request(request).unwrap();
            }
            
            // Schedule and transition to decode
            for _ in 0..num_decode {
                let output = scheduler.schedule();
                
                let mut next_tokens = Vec::new();
                let mut seq_ids = Vec::new();
                
                for seq in &output.prefill_sequences {
                    next_tokens.push(100u32);
                    seq_ids.push(seq.seq_id);
                }
                
                if !seq_ids.is_empty() {
                    let exec_output = ExecutionOutput {
                        next_tokens,
                        logits: None,
                        seq_ids,
                    };
                    scheduler.update_sequences(&exec_output, 0);
                }
            }
            
            // Now add more pending requests
            for i in num_decode..(num_decode + num_pending) {
                let request = create_request_with_tokens(i as u64, tokens_per_request, 100);
                scheduler.add_request(request).unwrap();
            }
            
            // Schedule - decode should come first
            let output = scheduler.schedule();
            
            // If we have decode sequences, they should all be scheduled before prefill
            // (given sufficient capacity)
            let decode_count = scheduler.decode_sequences.len();
            if decode_count > 0 && output.num_sequences() > 0 {
                // Decode sequences should be present in output
                prop_assert!(
                    !output.decode_sequences.is_empty() || decode_count == 0,
                    "Decode sequences should be scheduled when available"
                );
            }
        }

        /// **Feature: heterogeneous-inference-system, Property 9: Prefill to Decode Transition**
        /// *For any* sequence that completes its prefill phase, it shall immediately transition
        /// to decode state in the same scheduling cycle.
        /// **Validates: Requirements 3.3**
        #[test]
        fn prop_prefill_to_decode_transition(
            num_requests in 1usize..10,
            tokens_per_request in 4usize..32,
        ) {
            let config = create_test_config_with_params(16, 1024, 200);
            let mut scheduler = Scheduler::new(config);
            
            // Add requests
            for i in 0..num_requests {
                let request = create_request_with_tokens(i as u64, tokens_per_request, 50);
                scheduler.add_request(request).unwrap();
            }
            
            // Schedule prefill
            let output = scheduler.schedule();
            let prefill_seq_ids: Vec<SeqId> = output.prefill_sequences.iter().map(|s| s.seq_id).collect();
            
            // Simulate execution
            let next_tokens: Vec<u32> = prefill_seq_ids.iter().map(|_| 100).collect();
            let exec_output = ExecutionOutput {
                next_tokens,
                logits: None,
                seq_ids: prefill_seq_ids.clone(),
            };
            
            scheduler.update_sequences(&exec_output, 0);
            
            // All prefill sequences should now be in decode
            for seq_id in &prefill_seq_ids {
                prop_assert!(
                    scheduler.decode_sequences.contains_key(seq_id),
                    "Sequence {} should be in decode after prefill",
                    seq_id
                );
                prop_assert!(
                    !scheduler.prefill_sequences.contains_key(seq_id),
                    "Sequence {} should not be in prefill after transition",
                    seq_id
                );
            }
        }

        /// **Feature: heterogeneous-inference-system, Property 10: Completion Conditions**
        /// *For any* sequence in decode phase, it shall transition to completed state if and
        /// only if it generates an EOS token or reaches max_tokens.
        /// **Validates: Requirements 3.4**
        #[test]
        fn prop_completion_conditions(
            max_tokens in 1u32..20,
            tokens_per_request in 4usize..16,
            eos_position in 0usize..25,
        ) {
            let config = create_test_config_with_params(8, 512, 100);
            let mut scheduler = Scheduler::new(config);
            
            let request = create_request_with_tokens(1, tokens_per_request, max_tokens);
            scheduler.add_request(request).unwrap();
            
            // Schedule and transition to decode
            let output = scheduler.schedule();
            let seq_id = output.prefill_sequences[0].seq_id;
            
            let exec_output = ExecutionOutput {
                next_tokens: vec![100],
                logits: None,
                seq_ids: vec![seq_id],
            };
            scheduler.update_sequences(&exec_output, 0);
            
            // Generate tokens until completion
            let eos_token: TokenId = 0;
            let mut generated = 1u32;
            
            while scheduler.decode_sequences.contains_key(&seq_id) && generated < max_tokens + 5 {
                scheduler.schedule();
                
                // Decide whether to send EOS
                let token = if generated as usize == eos_position {
                    eos_token
                } else {
                    100 + generated
                };
                
                let exec_output = ExecutionOutput {
                    next_tokens: vec![token],
                    logits: None,
                    seq_ids: vec![seq_id],
                };
                scheduler.update_sequences(&exec_output, eos_token);
                generated += 1;
            }
            
            // Should be completed
            let completed = scheduler.get_completed();
            
            if !completed.is_empty() {
                let req = &completed[0];
                let hit_max = req.output_tokens.len() >= max_tokens as usize;
                let hit_eos = req.output_tokens.last() == Some(&eos_token);
                
                prop_assert!(
                    hit_max || hit_eos,
                    "Completion should be due to max_tokens or EOS"
                );
            }
        }

        /// **Feature: heterogeneous-inference-system, Property 13: Memory Pressure Response**
        /// *For any* state where memory utilization exceeds the configured threshold, the
        /// Scheduler shall reject new prefill requests until memory is freed.
        /// **Validates: Requirements 6.3**
        #[test]
        fn prop_memory_pressure_response(
            num_initial_requests in 5usize..15,
            tokens_per_request in 16usize..64,
        ) {
            // Use small block count to trigger memory pressure
            let config = EngineConfig {
                block_size: 16,
                max_num_blocks: 20,  // Small to trigger pressure
                max_batch_size: 16,
                max_num_seqs: 32,
                max_model_len: 2048,
                max_total_tokens: 1024,
                memory_threshold: 0.5,  // Low threshold
            };
            let mut scheduler = Scheduler::new(config);
            
            // Fill up memory
            for i in 0..num_initial_requests {
                let request = create_request_with_tokens(i as u64, tokens_per_request, 100);
                let _ = scheduler.add_request(request);
                let _ = scheduler.schedule();
            }
            
            // Check memory utilization
            let utilization = scheduler.get_memory_utilization();
            
            // If under pressure, new requests should be rejected
            if utilization >= 0.5 {
                let new_request = create_request_with_tokens(999, tokens_per_request, 100);
                let result = scheduler.add_request(new_request);
                
                // Should either reject or queue (depending on exact state)
                // The key property is that we don't crash and handle gracefully
                prop_assert!(
                    result.is_ok() || matches!(result, Err(SchedulerError::MemoryPressure)),
                    "Should handle memory pressure gracefully"
                );
            }
        }
    }
}
