# 论文引用

构成 Hetero-Paged-Infer 理论基础的学术论文。

## 核心论文

### Efficient Memory Management for Large Language Model Serving with PagedAttention

**Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hoffmann, Sean Yu, Joseph Zhang, Joseph E. Gonzalez, Ion Stoica**

*SOSP 2023*

<div class="cite-card">
<div class="cite-title">Efficient Memory Management for Large Language Model Serving with PagedAttention</div>
<div class="cite-authors">Woosuk Kwon, et al. (SOSP 2023)</div>
<div class="cite-links">
<a href="https://arxiv.org/abs/2309.06180">PDF (arXiv)</a>
<a href="https://github.com/vllm-project/vllm">代码 (vLLM)</a>
</div>

```bibtex
@inproceedings{kwon2023pagedattention,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Kwon, Woosuk and Li, Zhuohan and Zhuang, Siyuan and Sheng, Ying and Zheng, Lianmin and Hoffmann, Cody and Yu, Sean and Zhang, Joseph and Gonzalez, Joseph E and Stoica, Ion},
  booktitle={Proceedings of the 29th Symposium on Operating Systems Principles},
  pages={611--626},
  year={2023}
}
```
</div>

**核心贡献**：提出 PagedAttention，一种基于块的 KV Cache 内存管理技术，将内存浪费从 40-60% 降低到 <5%。

---

### Orca: A Distributed Serving System for Transformer-Based Generative Language Models

**Gyeongmin Yu, Gyeongmin Kim, Dongmin Kim, Soojeong Kim, Joo Seong Jeong, Joo Young Hwang, Junbeom Hur, Sungroh Yoon**

*OSDI 2022*

<div class="cite-card">
<div class="cite-title">Orca: A Distributed Serving System for Transformer-Based Generative Language Models</div>
<div class="cite-authors">Gyeongmin Yu, et al. (OSDI 2022)</div>
<div class="cite-links">
<a href="https://arxiv.org/abs/2205.11000">PDF (arXiv)</a>
</div>

```bibtex
@inproceedings{yu2022orca,
  title={Orca: A Distributed Serving System for Transformer-Based Generative Language Models},
  author={Yu, Gyeongmin and Kim, Gyeongmin and Kim, Dongmin and Kim, Soojeong and Jeong, Joo Seong and Hwang, Joo Young and Hur, Junbeom and Yoon, Sungroh},
  booktitle={16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22)},
  pages={845--861},
  year={2022}
}
```
</div>

**核心贡献**：提出连续批处理（迭代级调度），支持在推理迭代之间动态添加和移除序列。

---

### FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

**Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré**

*NeurIPS 2022*

<div class="cite-card">
<div class="cite-title">FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness</div>
<div class="cite-authors">Tri Dao, et al. (NeurIPS 2022)</div>
<div class="cite-links">
<a href="https://arxiv.org/abs/2205.14135">PDF (arXiv)</a>
<a href="https://github.com/Dao-AILab/flash-attention">代码</a>
</div>

```bibtex
@inproceedings{dao2022flashattention,
  title={FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness},
  author={Dao, Tri and Fu, Daniel Y and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle={Advances in Neural Information Processing Systems},
  volume={35},
  pages={16344--16359},
  year={2022}
}
```
</div>

**核心贡献**：提出 IO 感知的注意力计算，最小化 GPU HBM 读写次数，实现 2-4x 注意力加速。

---

## 引用本项目

如果您在研究中使用 Hetero-Paged-Infer，请引用：

```bibtex
@software{hetero-paged-infer2024,
  title={Hetero-Paged-Infer: High-Performance LLM Inference Engine},
  author={LessUp},
  year={2024},
  url={https://github.com/LessUp/hetero-paged-infer}
}
```

<style>
.cite-card {
  padding: 16px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-border);
  border-left: 4px solid #8b5cf6;
  border-radius: 8px;
  margin: 16px 0;
}

.cite-card .cite-title {
  font-weight: 700;
  font-size: 1rem;
  color: var(--vp-c-text-1);
  margin-bottom: 4px;
}

.cite-card .cite-authors {
  font-style: italic;
  color: var(--vp-c-text-2);
  font-size: 0.9rem;
  margin-bottom: 8px;
}

.cite-card .cite-links {
  display: flex;
  gap: 16px;
  margin-bottom: 12px;
}

.cite-card .cite-links a {
  color: #8b5cf6;
  font-size: 0.85rem;
  font-weight: 500;
  text-decoration: none;
}

.cite-card pre {
  margin: 0;
  padding: 8px;
  background: var(--vp-c-bg-alt);
  border-radius: 4px;
  font-size: 0.75rem;
  overflow-x: auto;
}
</style>