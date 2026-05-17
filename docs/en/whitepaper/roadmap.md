# Roadmap

## Near-term credibility work

The first priority is not adding more slogans; it is tightening the evidence chain. That means keeping architecture docs aligned with code, expanding correctness tests where scheduler and KV cache invariants matter most, and making benchmark methodology explicit once real GPU execution exists. The near-term goal is to reduce ambiguity between “designed,” “implemented,” and “measured.”

## Serving/runtime work

The engine boundary is already visible, but production serving needs more than a single-process core loop. Important next steps include request admission policy, cancellation and timeout semantics, richer metrics, clearer failure surfaces, and runtime ergonomics around configuration and deployment. This is the layer that turns a strong core into an operable service.

## GPU kernel work

Real production credibility depends on replacing the mock executor with an actual CUDA path. That work includes paged-attention kernels, decode-path optimization, CUDA graph capture where it materially reduces launch overhead, and disciplined benchmarking across realistic batch shapes. This is the most important “future” bucket because it converts architectural intent into hardware-backed proof.

## Research-grade extensions

Once the core path is credible, the project can explore more ambitious extensions: speculative decoding, sequence branching, richer memory-reuse policies, heterogeneous executor backends, or scheduling policies aimed at workload classes instead of a single default heuristic. These extensions are valuable because the current codebase already exposes the interfaces where such work would plug in.

## What will deliberately stay out of scope

Hetero-Paged-Infer should not try to become everything at once. Near-term work should stay out of large-scale distributed orchestration, every-model compatibility claims, or marketing-style benchmark wars against mature systems before the CUDA backend exists. The project becomes more credible by defending a focused thesis—clear architecture, explicit boundaries, honest proof—than by pretending to be a fully general replacement for the established serving stack today.
