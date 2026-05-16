---
title: Hetero-Paged-Infer
---

<FlagshipHero
  eyebrow="Rust LLM Serving Engine"
  title="A whitepaper-grade showcase of paged attention and continuous batching"
  summary="Hetero-Paged-Infer is a Rust-first inference engine that turns modern serving ideas into an inspectable, testable architecture: paged KV cache management, decode-prioritized scheduling, OpenAI-compatible serving, and proof-oriented documentation."
  primary-text="Read the Whitepaper"
  primary-href="/en/whitepaper/"
  secondary-text="Inspect the Architecture"
  secondary-href="/en/architecture/overview"
/>

<ProofStrip :items="[
  { label: 'Core language', value: 'Rust', detail: 'Trait-based architecture and memory-safe systems design.' },
  { label: 'Memory model', value: 'Paged', detail: 'Block-based KV cache with low waste and explicit accounting.' },
  { label: 'Serving model', value: 'Continuous', detail: 'Decode-priority scheduling with OpenAI-compatible endpoints.' },
  { label: 'Quality bar', value: 'CI + Tests', detail: 'Public claims must match code, docs, and benchmarks.' },
]" />

## Why this project matters

Most open-source inference engines are either highly optimized but difficult to study, or easy to read but weak in systems design. Hetero-Paged-Infer aims to sit in the middle: readable enough for learning, rigorous enough to earn technical trust.

## What makes it distinctive

1. Rust-first systems framing
2. Clear control-plane / compute-plane separation
3. Stronger testing and type-safety story than typical hobby inference repos
4. Whitepaper-style explanation instead of feature-list marketing

## Why now

Modern LLM serving is no longer judged only by raw throughput. Readers, reviewers, and future contributors also want to know how memory is managed, why decode work is prioritized, and whether the API surface reflects real deployment concerns.

Hetero-Paged-Infer matters now because it treats those questions as part of the product. The homepage is not just a launcher to internal pages; it is an argument that architecture, evidence, and documentation should reinforce each other.

- The [whitepaper](/en/whitepaper/) frames the problem and the technical position.
- The [architecture section](/en/architecture/overview) explains how the control plane and compute plane fit together.
- The [benchmarks section](/en/benchmarks/) separates current evidence from future performance claims.

## Architecture at a glance

The engine is presented as a set of legible subsystems instead of a monolith. You can trace how requests enter the orchestrator, how the scheduler builds mixed prefill and decode batches, and how the KV cache manager accounts for memory in fixed-size blocks.

<SectionGrid :cards="[
  { title: 'System overview', summary: 'Start with the control-plane / compute-plane split and the main execution loop.', href: '/en/architecture/overview' },
  { title: 'PagedAttention', summary: 'See how block-based KV allocation reduces waste and keeps memory accounting explicit.', href: '/en/architecture/paged-attention' },
  { title: 'Continuous batching', summary: 'Follow the decode-first scheduling model and how batch slots are filled over time.', href: '/en/architecture/continuous-batching' },
  { title: 'Design notes', summary: 'Read the broader architectural choices, trade-offs, and future-facing constraints.', href: '/en/architecture/design' },
]" />

## Proof, not slogans

This project is strongest when a claim can be checked from more than one direction: implementation structure, benchmark notes, API documentation, and references to prior art. That is why the site emphasizes traceability over promotional language.

<SectionGrid :cards="[
  { title: 'Benchmarks', summary: 'Review memory, throughput, and latency pages with explicit notes about current mock-GPU limits.', href: '/en/benchmarks/' },
  { title: 'API surface', summary: 'Inspect the OpenAI-compatible endpoints and the core request and response types.', href: '/en/api/' },
  { title: 'Comparison', summary: 'Place the project against other inference engines without pretending every capability is complete.', href: '/en/comparison/' },
  { title: 'References', summary: 'Trace the architecture back to papers and adjacent open-source systems.', href: '/en/references/' },
]" />

## Where this project is already strong

- **Readable systems decomposition** — the engine, scheduler, KV cache manager, and GPU executor are all named architectural units with clear responsibilities.
- **Evidence-oriented docs** — the whitepaper, architecture pages, and benchmark notes are organized to support technical review rather than feature browsing.
- **Rust-first design discipline** — traits and explicit interfaces make the control flow easier to test and reason about.
- **Serving realism** — the project discusses OpenAI-compatible serving, batching behavior, and memory pressure as operational concerns, not just algorithm names.

## Where it is still intentionally incomplete

This is not presented as a finished production serving stack. Some of its value comes from being honest about what is still under construction.

- Hardware-backed GPU performance work is still separate from the current mock-executor benchmark story.
- Deployment and production hardening are documented as directions, not as fully closed operational guarantees.
- The documentation platform is expanding toward a fuller whitepaper and reference experience, so some sections are stronger than others today.

That candor is part of the flagship narrative: the site should help readers distinguish what is already proven from what is still a deliberate next step.

## Suggested reading path

If you are new to the project, read it like a technical case study rather than a product brochure.

<SectionGrid :cards="[
  { title: '1. Whitepaper', summary: 'Start with the motivation, positioning, and limits of the engine.', href: '/en/whitepaper/' },
  { title: '2. Architecture', summary: 'Move into the subsystem layout and the control flow between scheduler, cache, and executor.', href: '/en/architecture/overview' },
  { title: '3. Benchmarks', summary: 'Check what evidence exists today and what is explicitly deferred.', href: '/en/benchmarks/' },
  { title: '4. Comparison', summary: 'See how the project situates itself against adjacent serving systems.', href: '/en/comparison/' },
  { title: '5. References', summary: 'End with the papers and projects that ground the design.', href: '/en/references/' },
]" />
