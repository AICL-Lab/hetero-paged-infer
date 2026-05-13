# Projects

Open-source projects that influenced or are related to Hetero-Paged-Infer.

## Core Reference Projects

### vLLM

<div class="project-card">
<div class="project-header">
<span class="project-icon">🔥</span>
<span class="project-name">vLLM</span>
</div>
<div class="project-desc">High-throughput LLM serving with PagedAttention</div>

- **Repository**: [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
- **Language**: Python + CUDA C++
- **License**: Apache 2.0

**Key Features Implemented**:
- PagedAttention memory management
- Continuous batching
- OpenAI-compatible API server
- Tensor parallelism

**Comparison**: Hetero-Paged-Infer implements similar techniques in Rust with focus on type safety and zero-cost abstractions.

</div>

---

### llama.cpp

<div class="project-card">
<div class="project-header">
<span class="project-icon">🦙</span>
<span class="project-name">llama.cpp</span>
</div>
<div class="project-desc">CPU-optimized LLM inference in pure C/C++</div>

- **Repository**: [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
- **Language**: C/C++
- **License**: MIT

**Key Features**:
- CPU-only inference with SIMD optimization
- Quantization (GGUF format)
- Cross-platform support
- Memory-mapped model loading

**Comparison**: Hetero-Paged-Infer focuses on GPU inference with PagedAttention, while llama.cpp excels at CPU deployment.

</div>

---

### TensorRT-LLM

<div class="project-card">
<div class="project-header">
<span class="project-icon">⚡</span>
<span class="project-name">TensorRT-LLM</span>
</div>
<div class="project-desc">NVIDIA's optimized inference library for LLMs</div>

- **Repository**: [github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- **Language**: C++ / Python
- **License**: NVIDIA Software License

**Key Features**:
- Deep integration with NVIDIA hardware
- CUDA Graph support
- Advanced quantization (FP8, INT4)
- Multi-GPU support

**Comparison**: Hetero-Paged-Infer is hardware-agnostic (via trait abstraction), TensorRT-LLM is NVIDIA-specific.

</div>

---

## Feature Comparison

| Feature | Hetero-Paged-Infer | vLLM | llama.cpp | TensorRT-LLM |
|---------|:------------------:|:----:|:---------:|:------------:|
| **Language** | Rust | Python | C++ | C++ |
| **PagedAttention** | ✅ | ✅ | ❌ | ✅ |
| **Continuous Batching** | ✅ | ✅ | ❌ | ✅ |
| **CPU Inference** | Mock | ❌ | ✅ | ❌ |
| **GPU Inference** | Planned | ✅ | ✅ | ✅ |
| **OpenAI API** | ✅ | ✅ | ✅ | ✅ |
| **Streaming (SSE)** | ✅ | ✅ | ✅ | ✅ |
| **Memory Safety** | ✅ (Rust) | ⚠️ | ⚠️ | ⚠️ |
| **Property Testing** | ✅ | ❌ | ❌ | ❌ |

## Why Rust?

Hetero-Paged-Infer chose Rust for:

1. **Memory Safety Without GC**: No GC pauses, critical for latency-sensitive serving
2. **Zero-Cost Abstractions**: Trait-based design without runtime overhead
3. **Fearless Concurrency**: Safe async/await for HTTP serving
4. **Interoperability**: Easy FFI with CUDA via `cudarc`

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
