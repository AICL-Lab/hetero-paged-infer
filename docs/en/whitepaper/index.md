# Whitepaper

This whitepaper is the entry point for understanding why Hetero-Paged-Infer exists, what it already proves, and where its production credibility still depends on future GPU work.

Hetero-Paged-Infer is a Rust-first exploration of modern LLM serving architecture: paged KV cache management, continuous batching, memory-pressure-aware scheduling, and a production-oriented API surface.

## Whitepaper map

This section is organized as a four-page reading path:

1. **Overview** *(this page)* — frames the problem, the thesis, and how to read the rest of the whitepaper.
2. [**Positioning**](./positioning) — explains what the project is, what it is not, and why that distinction matters.
3. [**Proof and Limits**](./proof) — separates implemented facts, tested behavior, inherited literature claims, and current boundaries.
4. [**Roadmap**](./roadmap) — outlines the next work required to turn architectural intent into stronger production credibility.

## Why this whitepaper exists

Modern LLM serving discussions often collapse architecture, implementation, and benchmark claims into the same story. This whitepaper keeps them separate. It describes the engine as it exists today, points to the evidence the repository can already support, and stays explicit about what still depends on a real CUDA backend.

## Recommended reading order

If you are new to the project, read the pages in sequence:

- Start here for the high-level framing.
- Continue to [Positioning](./positioning) for the project thesis.
- Then read [Proof and Limits](./proof) for the evidence boundary.
- Finish with [Roadmap](./roadmap) for the forward-looking work.
