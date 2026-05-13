# Papers

Academic publications that form the theoretical foundation of Hetero-Paged-Infer.

## Core Papers

### Efficient Memory Management for Large Language Model Serving with PagedAttention

**Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hoffmann, Sean Yu, Joseph Zhang, Joseph E. Gonzalez, Ion Stoica**

*SOSP 2023*

<div class="cite-card">
<div class="cite-title">Efficient Memory Management for Large Language Model Serving with PagedAttention</div>
<div class="cite-authors">Woosuk Kwon, et al. (SOSP 2023)</div>
<div class="cite-links">
<a href="https://arxiv.org/abs/2309.06180">PDF (arXiv)</a>
<a href="https://github.com/vllm-project/vllm">Code (vLLM)</a>
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

**Key Contribution**: Introduced PagedAttention, a block-based memory management technique for KV cache that reduces memory waste from 40-60% to <5%.

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

**Key Contribution**: Introduced continuous batching (iteration-level scheduling), enabling dynamic addition and removal of sequences between inference iterations.

---

### FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

**Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré**

*NeurIPS 2022*

<div class="cite-card">
<div class="cite-title">FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness</div>
<div class="cite-authors">Tri Dao, et al. (NeurIPS 2022)</div>
<div class="cite-links">
<a href="https://arxiv.org/abs/2205.14135">PDF (arXiv)</a>
<a href="https://github.com/Dao-AILab/flash-attention">Code</a>
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

**Key Contribution**: Introduced IO-aware attention computation that minimizes GPU HBM reads/writes, enabling 2-4x speedup for attention operations.

---

### FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning

**Tri Dao**

*ICLR 2024*

<div class="cite-card">
<div class="cite-title">FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning</div>
<div class="cite-authors">Tri Dao (ICLR 2024)</div>
<div class="cite-links">
<a href="https://arxiv.org/abs/2307.08691">PDF (arXiv)</a>
<a href="https://github.com/Dao-AILab/flash-attention">Code</a>
</div>

```bibtex
@inproceedings{dao2024flashattention2,
  title={FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning},
  author={Dao, Tri},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```
</div>

**Key Contribution**: Improved parallelism and work partitioning, achieving 2x speedup over FlashAttention.

---

## Related Papers

### vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention

**Woosuk Kwon, et al.**

*Technical Report, 2023*

<div class="cite-card">
<div class="cite-authors">vLLM Project Paper</div>
<div class="cite-links">
<a href="https://arxiv.org/abs/2309.06180">PDF (arXiv)</a>
</div>

System implementation of PagedAttention with production features.
</div>

---

## Citation Summary

If you use Hetero-Paged-Infer in your research, please cite:

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
