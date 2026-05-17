# Positioning

## What Hetero-Paged-Infer is

Hetero-Paged-Infer is a study-grade, production-shaped inference engine. It is not yet a full CUDA production stack, but it already demonstrates the structural decisions that matter: memory management, scheduling, serving boundaries, testing, and explicit future extension points.

In practice, that means the project already has a real engine boundary, a scheduler with queue and memory-pressure rules, a paged KV cache manager, and a documentation/test surface that makes the design inspectable instead of aspirational. The value of the project today is not “we already beat mature inference stacks,” but “we can show, review, and evolve the architecture with precision.”

## What it is not

1. Not a benchmark winner in its current mock-executor form
2. Not a full replacement for vLLM or TensorRT-LLM
3. Not a toy CLI with no architecture story

Those negatives matter because they define how this whitepaper should be read. When this site discusses throughput, memory efficiency, or future GPU work, it separates three things: what the Rust implementation already proves, what is represented by interfaces and tests, and what still depends on real CUDA kernels before production claims become credible.

## Why this positioning matters

Mature inference systems are judged on kernels, scheduling, memory behavior, and operational boundaries together. Many educational projects explain the ideas but stop before turning them into a coherent engine. Many product pages do the opposite: they promise results without showing the internal seams.

Hetero-Paged-Infer aims for the middle ground. It is concrete enough to examine as a system, but honest enough to say where the last production mile has not been completed yet. That makes it useful for readers who want to understand the architecture decisions behind modern LLM serving without pretending the current codebase is already a drop-in production CUDA stack.

## Read this whitepaper as

- an architecture argument for why paged KV management and continuous batching belong together,
- a proof-oriented description of what the repository implements today,
- and a roadmap for what must change before production-grade GPU credibility is earned.

Continue with [Proof and Limits](./proof) or [Roadmap](./roadmap).
