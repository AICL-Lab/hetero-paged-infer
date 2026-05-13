# PagedAttention

PagedAttention 是一种革命性的 KV Cache 内存管理技术，将内存浪费从传统的 40-60% 降低到 <5%。

## 核心思想

PagedAttention 借鉴了操作系统的虚拟内存概念：

1. **逻辑块 vs 物理块**：序列看到连续的逻辑块，实际映射到离散的物理块
2. **按需分配**：仅在需要时分配物理块，避免预分配浪费
3. **引用计数**：支持 Copy-on-Write，实现高效的序列共享

```mermaid
flowchart TB
    subgraph Sequence["序列视图 (逻辑)"]
        L0["逻辑块 0"]
        L1["逻辑块 1"]
        L2["逻辑块 2"]
    end
    
    subgraph PageTable["页表映射"]
        PT0["0 → 3"]
        PT1["1 → 7"]
        PT2["2 → 12"]
    end
    
    subgraph Physical["物理块池"]
        P3["块 3: ref=1"]
        P7["块 7: ref=1"]
        P12["块 12: ref=1"]
    end
    
    L0 --> PT0 --> P3
    L1 --> PT1 --> P7
    L2 --> PT2 --> P12
```

## 核心操作

### 块分配

```rust
impl KVCacheManager {
    /// 为序列分配新块
    pub fn allocate_block(&mut self, seq_id: SeqId) -> Result<BlockIdx, MemoryError> {
        // 1. 检查空闲列表
        if self.block_pool.free_list.is_empty() {
            return Err(MemoryError::OutOfMemory);
        }
        
        // 2. 从空闲列表弹出
        let block_idx = self.block_pool.free_list.pop_front().unwrap();
        
        // 3. 设置引用计数
        self.block_pool.blocks[block_idx].ref_count = 1;
        
        // 4. 添加到序列的页表
        self.page_tables.get_mut(&seq_id).unwrap().push(block_idx);
        
        Ok(block_idx)
    }
}
```

### Copy-on-Write (序列分支)

```rust
impl KVCacheManager {
    /// 分支序列 (用于 beam search 等)
    pub fn fork_sequence(&mut self, parent_id: SeqId) -> Result<SeqId, MemoryError> {
        let child_id = self.next_seq_id();
        
        // 复制页表 (增加引用计数)
        let parent_table = self.page_tables.get(&parent_id).unwrap();
        
        for block_idx in parent_table.logical_blocks.iter() {
            // 增加引用计数
            self.block_pool.blocks[*block_idx].ref_count += 1;
        }
        
        Ok(child_id)
    }
}
```

## 内存效率分析

### 传统静态分配 vs PagedAttention

| 方法 | 内存浪费 | 说明 |
|------|:--------:|------|
| 静态分配 | ~40-60% | 预分配最大长度，大量浪费 |
| **PagedAttention** | **<5%** | 按需分配，仅最后一个块有碎片 |

## 相关

- [内存管理](/zh/architecture/memory-management)
- [连续批处理](/zh/architecture/continuous-batching)
- [内存效率基准](/zh/benchmarks/memory-efficiency)