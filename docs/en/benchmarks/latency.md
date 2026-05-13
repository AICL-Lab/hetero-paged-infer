# Latency

Response latency is critical for interactive applications.

## Latency Components

```mermaid
flowchart LR
    subgraph Latency["Total Latency"]
        TTFT["Time to First Token"]
        ITL["Inter-Token Latency"]
    end
    
    subgraph TTFT_Components["TTFT Components"]
        Queue["Queue Time"]
        Prefill["Prefill Time"]
    end
    
    subgraph ITL_Components["ITL Components"]
        Schedule["Schedule Time"]
        Decode["Decode Time"]
    end
    
    TTFT --> TTFT_Components
    ITL --> ITL_Components
```

## Time to First Token (TTFT)

TTFT is the time from request submission to first generated token.

| Prompt Length | TTFT (ms) | Dominant Factor |
|---------------|:---------:|:---------------:|
| 128 tokens | 50 | Prefill |
| 512 tokens | 150 | Prefill |
| 2048 tokens | 500 | Prefill |

**Optimization**: Chunked prefill prevents long prefill from blocking other requests.

## Inter-Token Latency (ITL)

ITL is the time between consecutive generated tokens.

| Batch Size | ITL (ms) | Notes |
|------------|:--------:|:------:|
| 1 | 15 | Single sequence |
| 8 | 18 | Small batch |
| 32 | 25 | Large batch |
| 64 | 35 | Max batch |

**Observation**: ITL increases with batch size due to memory bandwidth, but throughput increases.

## Scheduling Impact

### Decode-First Priority

```rust
// Prioritize decode to minimize ITL
fn schedule(&mut self) -> SchedulerOutput {
    // Decode first: low latency for existing sequences
    for seq in self.decode_queue.iter() {
        batch.add_decode(seq);
    }
    
    // Prefill second: new requests wait
    for seq in self.prefill_queue.iter() {
        if !batch.is_full() {
            batch.add_prefill(seq);
        }
    }
}
```

### Memory Pressure Handling

When memory is constrained, the scheduler:

1. **Pauses new prefills** — Prevents OOM
2. **Continues decodes** — Maintains low ITL for in-flight requests
3. **Preempts if needed** — Swaps out lowest-priority sequences

```mermaid
stateDiagram-v2
    [*] --> Normal
    Normal --> Pressure: Memory > 80%
    Pressure --> Critical: Memory > 95%
    Critical --> Pressure: Memory < 90%
    Pressure --> Normal: Memory < 70%
    
    state Normal {
        [*] --> AcceptAll
    }
    state Pressure {
        [*] --> PausePrefill
    }
    state Critical {
        [*] --> PreemptSequences
    }
```

## Latency vs Throughput Trade-off

```mermaid
xychart-beta
    title "Latency vs Throughput Trade-off"
    x-axis "Batch Size" [1, 8, 16, 32, 64]
    y-axis "Latency (ms)" 0 --> 50
    y-axis "Throughput (tokens/s)" 0 --> 2000
    line [15, 18, 20, 25, 35]
    bar [500, 1200, 1500, 1800, 2000]
```

**Guideline**: Choose batch size based on latency SLA:
- Interactive applications: batch ≤ 16 (ITL < 20ms)
- Batch processing: batch = 64 (max throughput)

## Related

- [Throughput Metrics](/en/benchmarks/throughput)
- [Memory Management](/en/architecture/memory-management)
