//! 内存类型

use super::BlockIdx;

/// 物理块引用
///
/// 表示对 GPU 显存中物理块的引用。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysicalBlockRef {
    /// 物理块索引
    pub block_idx: BlockIdx,
}

/// 逻辑块
///
/// 表示映射到物理块的逻辑块。
#[derive(Debug, Clone)]
pub struct LogicalBlock {
    /// 序列内的逻辑块索引
    pub block_idx: u32,

    /// 映射的物理块（未分配时为 None）
    pub physical_block: Option<PhysicalBlockRef>,
}

impl LogicalBlock {
    /// 创建未映射的逻辑块
    pub fn new(block_idx: u32) -> Self {
        Self {
            block_idx,
            physical_block: None,
        }
    }

    /// 创建已映射的逻辑块
    pub fn with_physical(block_idx: u32, physical: PhysicalBlockRef) -> Self {
        Self {
            block_idx,
            physical_block: Some(physical),
        }
    }
}

/// KV Cache 内存统计
///
/// 提供内存使用情况的快照。
#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryStats {
    /// 物理块总数
    pub total_blocks: u32,

    /// 已使用的物理块数
    pub used_blocks: u32,

    /// 空闲物理块数
    pub free_blocks: u32,

    /// 活跃序列数
    pub num_sequences: u32,
}

impl MemoryStats {
    /// 计算内存利用率
    pub fn utilization(&self) -> f32 {
        if self.total_blocks == 0 {
            0.0
        } else {
            self.used_blocks as f32 / self.total_blocks as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
