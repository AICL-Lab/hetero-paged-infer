---
layout: home
---

<div class="home-header">
  <div class="home-header-left">
    <div class="home-logo">HPI</div>
    <div>
      <span class="home-title">Hetero-Paged-Infer</span>
      <span class="home-subtitle">High-Performance LLM Inference</span>
    </div>
  </div>
  <div class="home-nav">
    <a href="./setup/quickstart">Quick Start</a>
    <a href="https://github.com/LessUp/hetero-paged-infer">GitHub</a>
    <a href="../zh/">中文</a>
  </div>
</div>

<div class="home-intro-row">
  <div class="home-intro">
    A high-performance LLM inference engine written in Rust, implementing PagedAttention and Continuous Batching. Achieves &lt;5% memory waste and 50%+ throughput improvement over static batching.
  </div>
  <div class="home-stats">
    <span><strong>Rust</strong> Native</span>
    <span><strong>PagedAttention</strong></span>
    <span><strong>OpenAI</strong> Compatible</span>
  </div>
</div>

## Key Metrics

<div class="stats-grid">
  <div class="stat-card teal">
    <div class="stat-value">&lt;5%</div>
    <div class="stat-label">Memory Waste</div>
  </div>
  <div class="stat-card amber">
    <div class="stat-value">+50%</div>
    <div class="stat-label">Throughput</div>
  </div>
  <div class="stat-card violet">
    <div class="stat-value">122+</div>
    <div class="stat-label">Tests</div>
  </div>
  <div class="stat-card green">
    <div class="stat-value">0</div>
    <div class="stat-label">Unsafe</div>
  </div>
</div>

## Core Features

<div class="feature-map">
  <div class="feature-card">
    <div class="feature-card-title">🧠 PagedAttention</div>
    <div class="feature-card-desc">
      Block-based KV Cache management with on-demand allocation. Memory waste reduced from 40-60% to &lt;5%.
    </div>
    <div class="feature-tags">
      <a href="./architecture/paged-attention" class="feature-tag">Deep Dive</a>
      <a href="./benchmarks/memory-efficiency" class="feature-tag">Benchmarks</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">⚡ Continuous Batching</div>
    <div class="feature-card-desc">
      Dynamic prefill/decode scheduling with decode priority. Maximizes GPU utilization while maintaining low latency.
    </div>
    <div class="feature-tags">
      <a href="./architecture/continuous-batching" class="feature-tag">How It Works</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">🛡️ Memory Pressure Awareness</div>
    <div class="feature-card-desc">
      Configurable OOM prevention with graceful degradation. Production-ready error handling and monitoring.
    </div>
    <div class="feature-tags">
      <a href="./architecture/memory-management" class="feature-tag">Details</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">🔧 Modular Architecture</div>
    <div class="feature-card-desc">
      Trait-based abstractions for easy customization. Clean separation between CPU scheduler and GPU executor.
    </div>
    <div class="feature-tags">
      <a href="./architecture/overview" class="feature-tag">Architecture</a>
      <a href="./api/core-types" class="feature-tag">API</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">🧪 Property Testing</div>
    <div class="feature-card-desc">
      122+ tests including unit, property-based, and integration tests. Property tests verify critical invariants.
    </div>
    <div class="feature-tags">
      <a href="https://github.com/LessUp/hetero-paged-infer/tree/master/tests" class="feature-tag">Test Suite</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">🚀 OpenAI Compatible</div>
    <div class="feature-card-desc">
      Built-in HTTP server with OpenAI-compatible API. Supports streaming via Server-Sent Events (SSE).
    </div>
    <div class="feature-tags">
      <a href="./api/core-types" class="feature-tag">API Reference</a>
    </div>
  </div>
</div>

## Quick Start

<div class="quick-start">
  <div class="quick-start-title">Get Started in Minutes</div>
  <div class="quick-start-content">
    <div class="command-block">
      <code>git clone https://github.com/LessUp/hetero-paged-infer.git</code>
    </div>
    <div class="command-block">
      <code>cd hetero-paged-infer && cargo build --release</code>
    </div>
    <p style="margin-top: 12px; color: var(--vp-c-text-2);">Run inference with the built binary or start the HTTP server.</p>
  </div>
</div>

## Documentation

<div class="docs-grid">
  <a href="./setup/quickstart" class="doc-card">
    <div class="doc-icon">🚀</div>
    <div class="doc-title">Quick Start</div>
    <div class="doc-desc">Install, configure, and run your first inference request</div>
  </a>
  <a href="./architecture/overview" class="doc-card">
    <div class="doc-icon">🏗️</div>
    <div class="doc-title">Architecture</div>
    <div class="doc-desc">Deep dive into system design and core components</div>
  </a>
  <a href="./api/core-types" class="doc-card">
    <div class="doc-icon">📚</div>
    <div class="doc-title">API Reference</div>
    <div class="doc-desc">Complete API documentation for all modules</div>
  </a>
  <a href="./comparison/" class="doc-card">
    <div class="doc-icon">⚖️</div>
    <div class="doc-title">Comparison</div>
    <div class="doc-desc">Compare with vLLM, TensorRT-LLM, and other engines</div>
  </a>
</div>
