---
layout: home
---

<div class="home-header">
  <div class="home-header-left">
    <div class="home-logo">HPI</div>
    <div>
      <span class="home-title">Hetero-Paged-Infer</span>
      <span class="home-subtitle">高性能 LLM 推理引擎</span>
    </div>
  </div>
  <div class="home-nav">
    <a href="./setup/quickstart">快速开始</a>
    <a href="https://github.com/LessUp/hetero-paged-infer">GitHub</a>
    <a href="../en/">English</a>
  </div>
</div>

<div class="home-intro-row">
  <div class="home-intro">
    一个用 Rust 编写的高性能 LLM 推理引擎，实现 PagedAttention 和 Continuous Batching 技术。内存浪费 &lt;5%，相比静态批处理吞吐提升 50%+。
  </div>
  <div class="home-stats">
    <span><strong>Rust</strong> 原生</span>
    <span><strong>PagedAttention</strong></span>
    <span><strong>OpenAI</strong> 兼容</span>
  </div>
</div>

## 核心指标

<div class="stats-grid">
  <div class="stat-card teal">
    <div class="stat-value">&lt;5%</div>
    <div class="stat-label">内存浪费</div>
  </div>
  <div class="stat-card amber">
    <div class="stat-value">+50%</div>
    <div class="stat-label">吞吐提升</div>
  </div>
  <div class="stat-card violet">
    <div class="stat-value">122+</div>
    <div class="stat-label">测试用例</div>
  </div>
  <div class="stat-card green">
    <div class="stat-value">0</div>
    <div class="stat-label">Unsafe</div>
  </div>
</div>

## 核心特性

<div class="feature-map">
  <div class="feature-card">
    <div class="feature-card-title">🧠 PagedAttention</div>
    <div class="feature-card-desc">
      基于块的 KV Cache 管理，按需分配。内存浪费从 40-60% 降至 &lt;5%。
    </div>
    <div class="feature-tags">
      <a href="./architecture/paged-attention" class="feature-tag">深入了解</a>
      <a href="./benchmarks/memory-efficiency" class="feature-tag">性能数据</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">⚡ 连续批处理</div>
    <div class="feature-card-desc">
      动态 prefill/decode 调度，decode 优先。最大化 GPU 利用率，同时保持低延迟。
    </div>
    <div class="feature-tags">
      <a href="./architecture/continuous-batching" class="feature-tag">工作原理</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">🛡️ 内存压力感知</div>
    <div class="feature-card-desc">
      可配置的 OOM 防护，优雅降级。生产级错误处理和监控。
    </div>
    <div class="feature-tags">
      <a href="./architecture/memory-management" class="feature-tag">详细说明</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">🔧 模块化架构</div>
    <div class="feature-card-desc">
      基于 Trait 的抽象设计，易于定制。CPU 调度器与 GPU 执行器清晰分离。
    </div>
    <div class="feature-tags">
      <a href="./architecture/overview" class="feature-tag">架构概览</a>
      <a href="./api/core-types" class="feature-tag">API</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">🧪 属性测试</div>
    <div class="feature-card-desc">
      122+ 测试，包括单元测试、属性测试和集成测试。属性测试验证关键不变量。
    </div>
    <div class="feature-tags">
      <a href="https://github.com/LessUp/hetero-paged-infer/tree/master/tests" class="feature-tag">测试套件</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">🚀 OpenAI 兼容</div>
    <div class="feature-card-desc">
      内置 HTTP 服务器，OpenAI API 兼容。支持 Server-Sent Events (SSE) 流式响应。
    </div>
    <div class="feature-tags">
      <a href="./api/core-types" class="feature-tag">API 参考</a>
    </div>
  </div>
</div>

## 快速开始

<div class="quick-start">
  <div class="quick-start-title">几分钟上手</div>
  <div class="quick-start-content">
    <div class="command-block">
      <code>git clone https://github.com/LessUp/hetero-paged-infer.git</code>
    </div>
    <div class="command-block">
      <code>cd hetero-paged-infer && cargo build --release</code>
    </div>
    <p style="margin-top: 12px; color: var(--vp-c-text-2);">使用编译好的二进制文件运行推理或启动 HTTP 服务器。</p>
  </div>
</div>

## 文档导航

<div class="docs-grid">
  <a href="./setup/quickstart" class="doc-card">
    <div class="doc-icon">🚀</div>
    <div class="doc-title">快速开始</div>
    <div class="doc-desc">安装、配置、运行第一个推理请求</div>
  </a>
  <a href="./architecture/overview" class="doc-card">
    <div class="doc-icon">🏗️</div>
    <div class="doc-title">架构设计</div>
    <div class="doc-desc">深入系统设计和核心组件</div>
  </a>
  <a href="./api/core-types" class="doc-card">
    <div class="doc-icon">📚</div>
    <div class="doc-title">API 参考</div>
    <div class="doc-desc">完整的模块 API 文档</div>
  </a>
  <a href="./comparison/" class="doc-card">
    <div class="doc-icon">⚖️</div>
    <div class="doc-title">项目对比</div>
    <div class="doc-desc">与 vLLM、TensorRT-LLM 等对比</div>
  </a>
</div>
