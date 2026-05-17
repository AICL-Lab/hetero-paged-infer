# PagedAttention

## Problem

KV cache memory is difficult because sequence lengths are unpredictable. If the system reserves large contiguous regions up front, short requests waste memory. If it relies on ad hoc growth without a clear mapping model, memory accounting becomes fragile and the serving layer loses control over admission decisions.

For an inference engine, the real problem is therefore twofold: reduce waste from variable-length requests while still presenting a stable block table to the execution backend. PagedAttention solves that by turning “sequence memory” into an explicit mapping problem.

## Design choice

Hetero-Paged-Infer uses a fixed-size physical block pool plus per-sequence page tables. Sequences see logical growth, while the engine allocates physical blocks on demand and records the mapping explicitly. Memory statistics, free-list state, and block ownership stay in Rust control-plane code instead of being hidden inside an opaque executor.

The block model is intentionally simple today. Reference counting is present so the design can grow toward copy-on-write style reuse later, but the current implementation focuses first on deterministic allocation, release, and accounting for the single-engine path that already exists.

## Trade-off

The main advantage is bounded waste and explicit memory policy. Fixed-size blocks make fragmentation easier to reason about, and page tables give the scheduler a concrete basis for admission and pressure decisions. This is a stronger systems story than “allocate whatever each request needs and hope the backend survives.”

The trade-off is indirection and tuning surface. Smaller blocks reduce internal waste but increase metadata and lookup overhead; larger blocks simplify metadata but waste more tail space. The current manager also favors a straightforward single-process design over concurrency-heavy optimization, which is a good match for the repository today but not the final word on production scaling.

## Current implementation status

The repository already includes a block pool, free-list allocation, per-sequence page tables, memory statistics, and cleanup on sequence completion. Those pieces are implemented, testable, and directly connected to scheduler behavior.

What remains unfinished is the last production mile: real GPU kernels consuming this structure at full speed, and more advanced reuse features beyond the current reference-count scaffolding. So the page-table design is real today, while the highest-performance backend that should eventually exploit it is still future work.
