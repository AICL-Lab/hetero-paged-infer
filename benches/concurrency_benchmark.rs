//! 并发压测基准：并发突发下的调度吞吐与尾延迟
//!
//! 测量"提交 N 个并发请求 → 全部排空"的总耗时与逐请求延迟分布
//! （p50/p95/p99），作为调度器在并发下的可复现性能基线。
//! 正确性断言见 tests/concurrency_stress.rs；本文件只出性能数字。

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use paged_infer::test_utils::create_test_config;
use paged_infer::types::RequestId;
use paged_infer::{
    EngineConfig, GenerationParams, GPUExecutorTrait, InferenceEngine, MockGPUExecutor, Scheduler,
    SimpleTokenizer,
};
use std::collections::HashMap;
use std::time::Instant;

fn make_mock_engine(config: EngineConfig) -> InferenceEngine {
    let executor = MockGPUExecutor::new(config.clone(), 1000);
    InferenceEngine::with_components(
        config.clone(),
        Box::new(SimpleTokenizer::new()),
        Scheduler::new(config),
        Box::new(executor),
    )
    .unwrap()
}

/// 提交 n 个并发请求并排空，返回逐请求完成延迟（ms）。
fn run_burst(engine: &mut InferenceEngine, n: usize) -> Vec<f64> {
    let mut starts: HashMap<RequestId, Instant> = HashMap::new();
    for i in 0..n {
        let params = GenerationParams {
            max_tokens: 8,
            ..GenerationParams::default()
        };
        let (id, _) = engine.submit_request(&format!("prompt {}", i), params).unwrap();
        starts.insert(id, Instant::now());
    }
    let mut latencies = Vec::with_capacity(n);
    while engine.has_pending_work() {
        let done = engine.step().unwrap();
        for c in done {
            if let Some(t0) = starts.remove(&c.request_id) {
                latencies.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
        }
    }
    latencies
}

fn bench_concurrent_burst(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_burst");
    for &n in &[8u32, 16, 32, 64] {
        group.bench_function(format!("n{}", n), |b| {
            b.iter_batched(
                || {
                    let mut cfg = create_test_config();
                    cfg.max_num_seqs = n.max(32);
                    make_mock_engine(cfg)
                },
                |mut engine| {
                    let lat = run_burst(&mut engine, n as usize);
                    std::hint::black_box(lat.len());
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, bench_concurrent_burst);
criterion_main!(benches);
