---
title: Hetero-Paged-Infer
hide:
  - navigation
  - toc
---

<div align="center" style="padding: 4rem 2rem;">

# Hetero-Paged-Infer

<p style="font-size: 1.25rem; color: var(--md-default-fg-color--light); max-width: 600px; margin: 1rem auto;">
High-Performance Heterogeneous Inference Engine for Large Language Models<br>
高性能异构推理引擎，支持大语言模型 CPU-GPU 协同执行
</p>

<div style="display: flex; gap: 2rem; justify-content: center; flex-wrap: wrap; margin-top: 3rem;">

<a href="en/" class="md-button" style="padding: 1.5rem 3rem; font-size: 1.1rem;">
  <strong>English</strong><br>
  <span style="font-size: 0.9rem; opacity: 0.8;">Documentation</span>
</a>

<a href="zh/" class="md-button" style="padding: 1.5rem 3rem; font-size: 1.1rem;">
  <strong>中文</strong><br>
  <span style="font-size: 0.9rem; opacity: 0.8;">文档</span>
</a>

</div>

<div style="margin-top: 3rem;">

[![CI](https://github.com/LessUp/hetero-paged-infer/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/hetero-paged-infer/actions/workflows/ci.yml)
&nbsp;
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
&nbsp;
[![Rust](https://img.shields.io/badge/Rust-2021-orange.svg)](https://www.rust-lang.org/)

</div>

</div>

---

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin: 2rem 0;">

<div style="padding: 1.5rem; background: var(--md-code-bg-color); border-radius: 8px;">
<h3>:material-memory: PagedAttention</h3>
<p>Block-based memory management with &lt;5% waste</p>
</div>

<div style="padding: 1.5rem; background: var(--md-code-bg-color); border-radius: 8px;">
<h3>:material-format-list-bulleted: Continuous Batching</h3>
<p>Dynamic scheduling for optimal GPU utilization</p>
</div>

<div style="padding: 1.5rem; background: var(--md-code-bg-color); border-radius: 8px;">
<h3>:material-cpu-64-bit: CPU-GPU Co-execution</h3>
<p>Heterogeneous computing architecture</p>
</div>

</div>

---

<div align="center" style="padding: 2rem;">

<p>
<a href="https://github.com/LessUp/hetero-paged-infer">GitHub Repository</a>
&bull;
<a href="https://github.com/LessUp/hetero-paged-infer/issues">Issues</a>
</p>

<p style="color: var(--md-default-fg-color--lighter); font-size: 0.9rem;">
Copyright &copy; 2026 LessUp<br>
Licensed under MIT License
</p>

</div>

<style>
.md-content__button {
  display: none;
}
.md-sidebar {
  display: none;
}
.md-main__inner {
  max-width: 900px;
  margin: 0 auto;
}
</style>
