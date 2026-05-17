# Continuous Batching

## Problem

Static batching wastes capacity whenever sequences finish at different times. In an autoregressive workload, that means the shortest request leaves holes behind while the longest request keeps the batch open, and new work waits outside the system even though some resources are already idle.

The harder problem is not just throughput; it is balancing throughput, latency, and memory pressure at once. Active decode requests want low step latency, while new prefill requests want admission fairness, and both compete for batch slots and KV cache space.

## Design choice

Hetero-Paged-Infer uses decode-priority continuous batching. The scheduler keeps separate pending, prefill, and decode states, then builds each execution step by taking decode work first, filling remaining capacity with prefill work, and only admitting brand-new prefills when memory pressure allows it.

This is an opinionated choice. The engine prefers keeping in-flight requests moving over maximizing immediate fairness for newly arrived prompts. It also couples admission policy to memory pressure instead of treating batching and KV allocation as unrelated subsystems.

## Trade-off

The advantage is that the engine behaves like a serving system instead of a batch demo. Decode-first scheduling reduces the risk that already-active requests stall behind large prefills, and memory-pressure gating makes overload behavior explicit rather than accidental.

The trade-off is that long or numerous prefills can wait longer, and the current policy is a single default heuristic rather than a full multi-tenant fairness framework. This design is optimized for architectural clarity and realistic serving behavior, not for proving that every workload gets the globally optimal schedule.

## Current implementation status

The scheduler implementation already enforces batch-size and token-count limits, tracks memory pressure, and exercises mixed prefill/decode behavior in tests. Integration coverage verifies that new work can enter while earlier requests are already running and that the engine continues operating under memory pressure.

What is not yet proven here is final GPU utilization. Because execution still goes through a mock backend, this page documents a real scheduling policy with correctness evidence, but not a finished hardware-backed throughput result.
