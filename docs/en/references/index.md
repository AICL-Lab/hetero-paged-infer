# References

This section is a curated reading guide, not a local proof page. It points to the papers and open-source systems that shaped Hetero-Paged-Infer, and it labels borrowed headline numbers as external literature context rather than repository-specific proof.

## Reading map

- [Papers](/en/references/papers) — Academic publications behind the main design ideas
- [Projects](/en/references/projects) — Reference implementations and production systems worth comparing against

## How to read this section

- Use [Benchmark Methodology](/en/benchmarks/methodology) for repository evidence rules.
- Use the references pages for design rationale, prior art, and curated follow-up reading.
- Do **not** treat cited impact numbers here as reproduced by this repository unless a benchmark page explicitly proves them.

## Prior-art context, not local proof

| Technique | Source | Literature context | How to read it here |
|-----------|--------|--------------------|---------------------|
| PagedAttention | Kwon et al., SOSP 2023 | External prior art: the paper reports `<5%` memory waste | Explains why paged KV-cache allocation matters; not a local benchmark result |
| Continuous Batching | Yu et al., OSDI 2022 | External prior art: the paper reports roughly `+50%` throughput improvement | Motivates scheduler design; not a throughput claim proven in this repository |
| FlashAttention | Dao et al., NeurIPS 2022 | External prior art: the paper reports `2-4x` faster attention in its own setup | Shows why fused attention kernels matter; not a measured engine result here |
