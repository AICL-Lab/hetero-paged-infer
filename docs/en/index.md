---
title: Hetero-Paged-Infer
---

<FlagshipHero
  eyebrow="Rust LLM Serving Engine"
  title="A focused inference engine with clear subsystem boundaries"
  summary="Hetero-Paged-Infer concentrates on the core serving path: paged KV cache management, continuous batching, and OpenAI-compatible HTTP APIs."
  primary-text="Read Architecture"
  primary-href="/en/architecture/overview"
  secondary-text="Quick Start"
  secondary-href="/en/setup/quickstart"
/>

<ProofStrip :items="[
  { label: 'Language', value: 'Rust', detail: 'Memory-safe systems code with explicit interfaces.' },
  { label: 'Memory', value: 'Paged KV Cache', detail: 'Block-based allocation and accounting.' },
  { label: 'Scheduling', value: 'Continuous Batching', detail: 'Prefill/decode flow with decode-priority behavior.' },
  { label: 'Serving', value: 'OpenAI-Compatible', detail: 'Completions and chat endpoints with operational probes.' },
]" />

## Documentation map

<SectionGrid :cards="[
  { title: 'Architecture', summary: 'Engine structure, scheduling model, and memory management design.', href: '/en/architecture/overview' },
  { title: 'Setup', summary: 'Installation, configuration, and local usage.', href: '/en/setup/quickstart' },
  { title: 'API', summary: 'Core types and HTTP API references.', href: '/en/api/' },
  { title: 'Benchmarks', summary: 'Current measurements and methodology.', href: '/en/benchmarks/' },
  { title: 'Deployment', summary: 'Docker and production-oriented deployment notes.', href: '/en/deployment/' },
  { title: 'Development', summary: 'Contributing and validation workflow.', href: '/en/development/contributing' },
]" />

## Current scope

- The repository already provides a usable local engine, scheduler, KV cache manager, and HTTP serving layer.
- The docs site focuses on implementation and usage details, not changelog history.
- Real CUDA kernel execution remains planned work.
