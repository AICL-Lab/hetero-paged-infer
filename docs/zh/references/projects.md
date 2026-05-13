# 项目引用

影响或与 Hetero-Paged-Infer 相关的开源项目。

## 核心参考项目

### vLLM

<div class="project-card">
<div class="project-header">
<span class="project-icon">🔥</span>
<span class="project-name">vLLM</span>
</div>
<div class="project-desc">高吞吐量 LLM 推理，实现 PagedAttention</div>

- **仓库**: [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
- **语言**: Python + CUDA C++
- **许可证**: Apache 2.0

**实现的关键特性**：
- PagedAttention 内存管理
- 连续批处理
- OpenAI 兼容 API 服务器
- 张量并行

**对比**：Hetero-Paged-Infer 用 Rust 实现类似技术，专注于类型安全和零成本抽象。

</div>

---

### llama.cpp

<div class="project-card">
<div class="project-header">
<span class="project-icon">🦙</span>
<span class="project-name">llama.cpp</span>
</div>
<div class="project-desc">纯 C/C++ 的 CPU 优化 LLM 推理</div>

- **仓库**: [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
- **语言**: C/C++
- **许可证**: MIT

**关键特性**：
- CPU-only 推理，SIMD 优化
- 量化支持（GGUF 格式）
- 跨平台支持
- 内存映射模型加载

**对比**：Hetero-Paged-Infer 专注于 GPU 推理和 PagedAttention，llama.cpp 擅长 CPU 部署。

</div>

---

## 特性对比

| 特性 | Hetero-Paged-Infer | vLLM | llama.cpp |
|------|:------------------:|:----:|:---------:|
| **语言** | Rust | Python | C++ |
| **PagedAttention** | ✅ | ✅ | ❌ |
| **连续批处理** | ✅ | ✅ | ❌ |
| **CPU 推理** | Mock | ❌ | ✅ |
| **GPU 推理** | 计划中 | ✅ | ✅ |
| **OpenAI API** | ✅ | ✅ | ✅ |
| **流式输出 (SSE)** | ✅ | ✅ | ✅ |
| **内存安全** | ✅ (Rust) | ⚠️ | ⚠️ |
| **属性测试** | ✅ | ❌ | ❌ |

## 为什么选择 Rust？

Hetero-Paged-Infer 选择 Rust 的原因：

1. **无 GC 的内存安全**：无 GC 停顿，对延迟敏感的推理至关重要
2. **零成本抽象**：基于 Trait 的设计无运行时开销
3. **无畏并发**：安全的 async/await 用于 HTTP 服务
4. **互操作性**：通过 `cudarc` 轻松与 CUDA 进行 FFI

<style>
.project-card {
  padding: 20px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-border);
  border-radius: 12px;
  margin: 16px 0;
}

.project-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 8px;
}

.project-icon {
  font-size: 1.5rem;
}

.project-name {
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--vp-c-text-1);
}

.project-desc {
  color: var(--vp-c-text-2);
  font-size: 0.95rem;
  margin-bottom: 12px;
}

.project-card ul {
  margin: 8px 0;
  padding-left: 20px;
}

.project-card li {
  margin: 4px 0;
  color: var(--vp-c-text-2);
}
</style>