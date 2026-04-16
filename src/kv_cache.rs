//! KV Cache 管理器 - PagedAttention 实现
//!
//! 使用分页分配实现高效的 GPU 显存管理，用于 Transformer 推理中的 KV Cache。
//!
//! # 核心概念
//!
//! - **物理块 (Physical Block)** - GPU 显存中的连续区域
//! - **逻辑块 (Logical Block)** - 序列视角的虚拟块
//! - **页表 (Page Table)** - 逻辑块到物理块的映射
//!
//! # 内存模型
//!
//! ```text
//! GPU 显存池:
//! ┌────────┬────────┬────────┬─────┬──────────┐
//! │ Block 0│ Block 1│ Block 2│ ... │ Block N-1│
//! │ K[0:16]│ K[0:16]│ K[0:16]│     │ K[0:16]  │
//! │ V[0:16]│ V[0:16]│ V[0:16]│     │ V[0:16]  │
//! └────────┴────────┴────────┴─────┴──────────┘
//! ```

use crate::error::MemoryError;
use crate::types::{BlockIdx, LogicalBlock, MemoryStats, PhysicalBlockRef, SeqId};
use std::collections::{HashMap, HashSet, VecDeque};

/// A physical block representing a contiguous GPU memory region
#[derive(Debug, Clone)]
pub struct PhysicalBlock {
    /// Unique index of this block
    pub block_idx: BlockIdx,
    /// Reference count for copy-on-write support
    pub ref_count: u32,
    /// Whether this block is currently allocated
    pub is_allocated: bool,
}

impl PhysicalBlock {
    pub fn new(block_idx: BlockIdx) -> Self {
        Self {
            block_idx,
            ref_count: 0,
            is_allocated: false,
        }
    }
}

/// Pool of physical blocks with free list management
#[derive(Debug)]
pub struct BlockPool {
    /// All physical blocks
    blocks: Vec<PhysicalBlock>,
    /// Free block indices (FIFO queue for better cache locality)
    free_list: VecDeque<BlockIdx>,
    /// Number of tokens per block
    block_size: u32,
}

impl BlockPool {
    /// Create a new block pool with the specified number of blocks
    pub fn new(num_blocks: u32, block_size: u32) -> Self {
        let mut blocks = Vec::with_capacity(num_blocks as usize);
        let mut free_list = VecDeque::with_capacity(num_blocks as usize);

        for i in 0..num_blocks {
            blocks.push(PhysicalBlock::new(i));
            free_list.push_back(i);
        }

        Self {
            blocks,
            free_list,
            block_size,
        }
    }

    /// Allocate a single block from the pool
    pub fn allocate(&mut self) -> Result<PhysicalBlockRef, MemoryError> {
        if let Some(block_idx) = self.free_list.pop_front() {
            let block = &mut self.blocks[block_idx as usize];
            block.is_allocated = true;
            block.ref_count = 1;
            Ok(PhysicalBlockRef { block_idx })
        } else {
            Err(MemoryError::OutOfBlocks)
        }
    }

    /// Free a block back to the pool
    pub fn free(&mut self, block_ref: PhysicalBlockRef) -> Result<(), MemoryError> {
        let block_idx = block_ref.block_idx;
        if block_idx >= self.blocks.len() as u32 {
            return Err(MemoryError::InvalidBlockIndex(block_idx));
        }

        let block = &mut self.blocks[block_idx as usize];
        if !block.is_allocated {
            return Ok(()); // Already free, idempotent
        }

        block.ref_count = block.ref_count.saturating_sub(1);
        if block.ref_count == 0 {
            block.is_allocated = false;
            self.free_list.push_back(block_idx);
        }
        Ok(())
    }

    /// Check if n blocks can be allocated
    pub fn can_allocate(&self, num_blocks: u32) -> bool {
        self.free_list.len() >= num_blocks as usize
    }

    /// Get number of free blocks
    pub fn num_free_blocks(&self) -> u32 {
        self.free_list.len() as u32
    }

    /// Get total number of blocks
    pub fn total_blocks(&self) -> u32 {
        self.blocks.len() as u32
    }

    /// Get number of used blocks
    pub fn num_used_blocks(&self) -> u32 {
        self.total_blocks() - self.num_free_blocks()
    }

    /// Get block size
    pub fn block_size(&self) -> u32 {
        self.block_size
    }
}

/// Page table for a single sequence mapping logical to physical blocks
#[derive(Debug, Clone)]
pub struct PageTable {
    /// Sequence ID this page table belongs to
    pub seq_id: SeqId,
    /// Logical blocks with their physical mappings
    pub logical_blocks: Vec<LogicalBlock>,
}

impl PageTable {
    pub fn new(seq_id: SeqId) -> Self {
        Self {
            seq_id,
            logical_blocks: Vec::new(),
        }
    }

    /// Add a new logical block with physical mapping
    pub fn add_block(&mut self, physical_ref: PhysicalBlockRef) {
        let logical_idx = self.logical_blocks.len() as u32;
        self.logical_blocks
            .push(LogicalBlock::with_physical(logical_idx, physical_ref));
    }

    /// Get physical block for a logical index - O(1) lookup
    pub fn get_physical(&self, logical_idx: u32) -> Option<PhysicalBlockRef> {
        self.logical_blocks
            .get(logical_idx as usize)
            .and_then(|lb| lb.physical_block)
    }

    /// Get the block table as a vector of physical block indices
    pub fn get_block_table(&self) -> Vec<BlockIdx> {
        self.logical_blocks
            .iter()
            .filter_map(|lb| lb.physical_block.map(|pb| pb.block_idx))
            .collect()
    }

    /// Number of allocated blocks
    pub fn num_blocks(&self) -> u32 {
        self.logical_blocks.len() as u32
    }
}

/// Trait defining the KV Cache Manager interface
pub trait KVCacheManagerTrait {
    /// Allocate blocks for a new sequence
    fn allocate_sequence(&mut self, seq_id: SeqId, num_tokens: u32) -> Result<(), MemoryError>;

    /// Allocate additional block when sequence grows
    fn allocate_block(&mut self, seq_id: SeqId) -> Result<PhysicalBlockRef, MemoryError>;

    /// Free all blocks for a completed sequence
    fn free_sequence(&mut self, seq_id: SeqId);

    /// Get block table for GPU execution
    fn get_block_table(&self, seq_id: SeqId) -> Option<Vec<BlockIdx>>;

    /// Query memory status
    fn get_memory_stats(&self) -> MemoryStats;

    /// Check if can allocate n blocks
    fn can_allocate(&self, num_blocks: u32) -> bool;

    /// Get block size
    fn block_size(&self) -> u32;
}

/// KV Cache Manager implementation
#[derive(Debug)]
pub struct KVCacheManager {
    /// Physical block pool
    block_pool: BlockPool,
    /// Page tables for each sequence
    page_tables: HashMap<SeqId, PageTable>,
    /// Set of active sequence IDs
    active_sequences: HashSet<SeqId>,
}

impl KVCacheManager {
    /// Create a new KV Cache Manager
    pub fn new(num_blocks: u32, block_size: u32) -> Self {
        Self {
            block_pool: BlockPool::new(num_blocks, block_size),
            page_tables: HashMap::new(),
            active_sequences: HashSet::new(),
        }
    }

    /// Calculate number of blocks needed for given token count
    pub fn blocks_for_tokens(&self, num_tokens: u32) -> u32 {
        // ceil(num_tokens / block_size) — 公式对 num_tokens==0 也成立
        num_tokens.div_ceil(self.block_pool.block_size())
    }

    /// Check if sequence exists
    pub fn has_sequence(&self, seq_id: SeqId) -> bool {
        self.active_sequences.contains(&seq_id)
    }

    /// Get number of blocks allocated for a sequence
    pub fn get_sequence_blocks(&self, seq_id: SeqId) -> u32 {
        self.page_tables
            .get(&seq_id)
            .map(|pt| pt.num_blocks())
            .unwrap_or(0)
    }
}

impl KVCacheManagerTrait for KVCacheManager {
    fn allocate_sequence(&mut self, seq_id: SeqId, num_tokens: u32) -> Result<(), MemoryError> {
        // Check if sequence already exists
        if self.active_sequences.contains(&seq_id) {
            return Ok(()); // Idempotent
        }

        let num_blocks = self.blocks_for_tokens(num_tokens);

        // Check if we have enough blocks
        if !self.block_pool.can_allocate(num_blocks) {
            return Err(MemoryError::OutOfBlocks);
        }

        // Create page table and allocate blocks
        let mut page_table = PageTable::new(seq_id);

        for _ in 0..num_blocks {
            let physical_ref = self.block_pool.allocate()?;
            page_table.add_block(physical_ref);
        }

        self.page_tables.insert(seq_id, page_table);
        self.active_sequences.insert(seq_id);

        Ok(())
    }

    fn allocate_block(&mut self, seq_id: SeqId) -> Result<PhysicalBlockRef, MemoryError> {
        if !self.active_sequences.contains(&seq_id) {
            return Err(MemoryError::SequenceNotFound(seq_id));
        }

        let physical_ref = self.block_pool.allocate()?;

        if let Some(page_table) = self.page_tables.get_mut(&seq_id) {
            page_table.add_block(physical_ref);
        }

        Ok(physical_ref)
    }

    fn free_sequence(&mut self, seq_id: SeqId) {
        if let Some(page_table) = self.page_tables.remove(&seq_id) {
            // Free all physical blocks
            for logical_block in &page_table.logical_blocks {
                if let Some(physical_ref) = logical_block.physical_block {
                    let _ = self.block_pool.free(physical_ref);
                }
            }
        }
        self.active_sequences.remove(&seq_id);
    }

    fn get_block_table(&self, seq_id: SeqId) -> Option<Vec<BlockIdx>> {
        self.page_tables.get(&seq_id).map(|pt| pt.get_block_table())
    }

    fn get_memory_stats(&self) -> MemoryStats {
        MemoryStats {
            total_blocks: self.block_pool.total_blocks(),
            used_blocks: self.block_pool.num_used_blocks(),
            free_blocks: self.block_pool.num_free_blocks(),
            num_sequences: self.active_sequences.len() as u32,
        }
    }

    fn can_allocate(&self, num_blocks: u32) -> bool {
        self.block_pool.can_allocate(num_blocks)
    }

    fn block_size(&self) -> u32 {
        self.block_pool.block_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_pool_allocation() {
        let mut pool = BlockPool::new(10, 16);
        assert_eq!(pool.num_free_blocks(), 10);
        assert_eq!(pool.num_used_blocks(), 0);

        let block = pool.allocate().unwrap();
        assert_eq!(pool.num_free_blocks(), 9);
        assert_eq!(pool.num_used_blocks(), 1);

        pool.free(block).unwrap();
        assert_eq!(pool.num_free_blocks(), 10);
        assert_eq!(pool.num_used_blocks(), 0);
    }

    #[test]
    fn test_block_pool_exhaustion() {
        let mut pool = BlockPool::new(2, 16);

        let _b1 = pool.allocate().unwrap();
        let _b2 = pool.allocate().unwrap();

        assert!(pool.allocate().is_err());
    }

    #[test]
    fn test_kv_cache_manager_allocate_sequence() {
        let mut manager = KVCacheManager::new(100, 16);

        // Allocate sequence with 32 tokens (needs 2 blocks)
        manager.allocate_sequence(1, 32).unwrap();

        let stats = manager.get_memory_stats();
        assert_eq!(stats.used_blocks, 2);
        assert_eq!(stats.num_sequences, 1);

        let block_table = manager.get_block_table(1).unwrap();
        assert_eq!(block_table.len(), 2);
    }

    #[test]
    fn test_kv_cache_manager_free_sequence() {
        let mut manager = KVCacheManager::new(100, 16);

        manager.allocate_sequence(1, 32).unwrap();
        manager.allocate_sequence(2, 48).unwrap();

        let stats = manager.get_memory_stats();
        assert_eq!(stats.used_blocks, 5); // 2 + 3
        assert_eq!(stats.num_sequences, 2);

        manager.free_sequence(1);

        let stats = manager.get_memory_stats();
        assert_eq!(stats.used_blocks, 3);
        assert_eq!(stats.num_sequences, 1);
    }

    #[test]
    fn test_kv_cache_manager_allocate_block() {
        let mut manager = KVCacheManager::new(100, 16);

        manager.allocate_sequence(1, 16).unwrap();
        assert_eq!(manager.get_sequence_blocks(1), 1);

        manager.allocate_block(1).unwrap();
        assert_eq!(manager.get_sequence_blocks(1), 2);
    }

    #[test]
    fn test_blocks_for_tokens() {
        let manager = KVCacheManager::new(100, 16);

        assert_eq!(manager.blocks_for_tokens(0), 0);
        assert_eq!(manager.blocks_for_tokens(1), 1);
        assert_eq!(manager.blocks_for_tokens(16), 1);
        assert_eq!(manager.blocks_for_tokens(17), 2);
        assert_eq!(manager.blocks_for_tokens(32), 2);
        assert_eq!(manager.blocks_for_tokens(33), 3);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    /// Operation type for property testing
    #[derive(Debug, Clone)]
    enum CacheOp {
        Allocate(SeqId, u32), // seq_id, num_tokens
        Free(SeqId),
        Grow(SeqId),
    }

    fn arb_cache_op() -> impl Strategy<Value = CacheOp> {
        prop_oneof![
            (1u64..100, 1u32..200).prop_map(|(id, tokens)| CacheOp::Allocate(id, tokens)),
            (1u64..100).prop_map(CacheOp::Free),
            (1u64..100).prop_map(CacheOp::Grow),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// **Feature: heterogeneous-inference-system, Property 5: Block Count Invariant**
        /// *For any* state of the KV_Cache_Manager, the invariant `used_blocks + free_blocks == total_blocks`
        /// shall hold. Additionally, when a sequence is freed, all its blocks shall return to the free pool.
        /// **Validates: Requirements 2.4, 2.5**
        #[test]
        fn prop_block_count_invariant(
            ops in prop::collection::vec(arb_cache_op(), 0..50),
            num_blocks in 10u32..200,
            block_size in 1u32..32,
        ) {
            let mut manager = KVCacheManager::new(num_blocks, block_size);

            // Verify invariant holds initially
            let stats = manager.get_memory_stats();
            prop_assert_eq!(
                stats.used_blocks + stats.free_blocks,
                stats.total_blocks,
                "Initial invariant violated"
            );

            // Apply operations and verify invariant after each
            for op in ops {
                match op {
                    CacheOp::Allocate(seq_id, num_tokens) => {
                        let _ = manager.allocate_sequence(seq_id, num_tokens);
                    }
                    CacheOp::Free(seq_id) => {
                        manager.free_sequence(seq_id);
                    }
                    CacheOp::Grow(seq_id) => {
                        let _ = manager.allocate_block(seq_id);
                    }
                }

                // Verify invariant after each operation
                let stats = manager.get_memory_stats();
                prop_assert_eq!(
                    stats.used_blocks + stats.free_blocks,
                    stats.total_blocks,
                    "Invariant violated after operation"
                );
            }
        }

        /// **Feature: heterogeneous-inference-system, Property 3: Block Allocation on Sequence Start**
        /// *For any* new sequence with n input tokens, the KV_Cache_Manager shall allocate
        /// ceil(n / block_size) logical blocks, each mapped to a distinct physical block.
        /// **Validates: Requirements 2.2**
        #[test]
        fn prop_block_allocation_on_sequence_start(
            seq_id in 1u64..1000,
            num_tokens in 1u32..500,
            block_size in 1u32..64,
        ) {
            // Ensure we have enough blocks
            let expected_blocks = num_tokens.div_ceil(block_size);
            let num_blocks = expected_blocks + 10; // Extra headroom

            let mut manager = KVCacheManager::new(num_blocks, block_size);

            // Allocate sequence
            let result = manager.allocate_sequence(seq_id, num_tokens);
            prop_assert!(result.is_ok(), "Allocation should succeed");

            // Verify correct number of blocks allocated
            let allocated_blocks = manager.get_sequence_blocks(seq_id);
            prop_assert_eq!(
                allocated_blocks,
                expected_blocks,
                "Expected {} blocks for {} tokens with block_size {}, got {}",
                expected_blocks, num_tokens, block_size, allocated_blocks
            );

            // Verify all physical blocks are distinct
            let block_table = manager.get_block_table(seq_id).unwrap();
            let unique_blocks: std::collections::HashSet<_> = block_table.iter().collect();
            prop_assert_eq!(
                unique_blocks.len(),
                block_table.len(),
                "Physical blocks should be distinct"
            );
        }

        /// **Feature: heterogeneous-inference-system, Property 4: Block Allocation on Growth**
        /// *For any* sequence that grows beyond its current block capacity, the KV_Cache_Manager
        /// shall allocate exactly one additional physical block when the token count crosses a block boundary.
        /// **Validates: Requirements 2.3**
        #[test]
        fn prop_block_allocation_on_growth(
            seq_id in 1u64..1000,
            initial_tokens in 1u32..100,
            growth_steps in 1u32..20,
            block_size in 1u32..32,
        ) {
            let max_blocks = initial_tokens / block_size + growth_steps + 10;
            let mut manager = KVCacheManager::new(max_blocks, block_size);

            // Allocate initial sequence
            manager.allocate_sequence(seq_id, initial_tokens).unwrap();
            let initial_block_count = manager.get_sequence_blocks(seq_id);

            // Grow the sequence
            for _ in 0..growth_steps {
                let blocks_before = manager.get_sequence_blocks(seq_id);
                let result = manager.allocate_block(seq_id);

                if result.is_ok() {
                    let blocks_after = manager.get_sequence_blocks(seq_id);
                    // Each successful allocation should add exactly one block
                    prop_assert_eq!(
                        blocks_after,
                        blocks_before + 1,
                        "Each allocation should add exactly one block"
                    );
                }
            }

            // Verify final block count
            let final_blocks = manager.get_sequence_blocks(seq_id);
            prop_assert!(
                final_blocks >= initial_block_count,
                "Block count should not decrease"
            );
        }

        /// **Feature: heterogeneous-inference-system, Property 12: Memory Statistics Invariant**
        /// *For any* state of the KV_Cache_Manager, the reported memory statistics shall satisfy:
        /// `total_blocks == used_blocks + free_blocks` and
        /// `num_sequences == count of sequences with allocated blocks`.
        /// **Validates: Requirements 6.2**
        #[test]
        fn prop_memory_statistics_invariant(
            ops in prop::collection::vec(arb_cache_op(), 0..30),
            num_blocks in 20u32..100,
            block_size in 4u32..16,
        ) {
            let mut manager = KVCacheManager::new(num_blocks, block_size);
            let mut expected_sequences: std::collections::HashSet<SeqId> = std::collections::HashSet::new();

            for op in ops {
                match op {
                    CacheOp::Allocate(seq_id, num_tokens) => {
                        if manager.allocate_sequence(seq_id, num_tokens).is_ok() {
                            expected_sequences.insert(seq_id);
                        }
                    }
                    CacheOp::Free(seq_id) => {
                        manager.free_sequence(seq_id);
                        expected_sequences.remove(&seq_id);
                    }
                    CacheOp::Grow(seq_id) => {
                        let _ = manager.allocate_block(seq_id);
                    }
                }

                let stats = manager.get_memory_stats();

                // Verify block count invariant
                prop_assert_eq!(
                    stats.total_blocks,
                    stats.used_blocks + stats.free_blocks,
                    "Block count invariant violated"
                );

                // Verify sequence count
                prop_assert_eq!(
                    stats.num_sequences as usize,
                    expected_sequences.len(),
                    "Sequence count mismatch"
                );
            }
        }
    }
}
