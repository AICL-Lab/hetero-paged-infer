//! 并发压测基准：并发突发下的调度吞吐与尾延迟
//!
//! 测量"提交 N 个并发请求 → 全部排空"的总耗时与逐请求延迟分布
//! （p50/p95/p99），作为调度器在并发下的可复现性能基线。
//! 正确性断言见 tests/concurrency_stress.rs；本文件只出性能数字。

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use paged_infer::test_utils::create_test_config;
use paged_infer::types::RequestId;
use paged_infer::{
    EngineConfig, GenerationParams, InferenceEngine, MockGPUExecutor, Scheduler, SimpleTokenizer,
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
        let (id, _) = engine
            .submit_request(&format!("prompt {}", i), params)
            .unwrap();
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

/// 有序样本的百分位（p ∈ [0,1]，线性插值取最近秩）。
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
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

/// Continuous Batching on/off 对比：相同 N 并发请求，
/// - `cb_off`：`max_batch_size = 1`（一次只跑一个序列，串行）；
/// - `cb_on`：`max_batch_size = N`（全部并发组批）。
///
/// 输出总排空时间与 p50/p95/p99。使用 Mock 后端，只反映调度开销，
/// 不是 GPU 吞吐。
fn bench_cb_on_off(c: &mut Criterion) {
    let mut group = c.benchmark_group("cb_on_off");
    for &n in &[8u32, 16, 32] {
        for (label, max_batch_size) in [("cb_off", 1u32), ("cb_on", n)] {
            group.bench_function(format!("{label}_n{n}"), |b| {
                b.iter_batched(
                    || {
                        let mut cfg = create_test_config();
                        cfg.max_batch_size = max_batch_size;
                        cfg.max_num_seqs = n.max(32);
                        make_mock_engine(cfg)
                    },
                    |mut engine| {
                        let mut lat = run_burst(&mut engine, n as usize);
                        lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let total = lat.iter().sum::<f64>();
                        std::hint::black_box((
                            total,
                            percentile(&lat, 0.5),
                            percentile(&lat, 0.95),
                            percentile(&lat, 0.99),
                        ));
                    },
                    BatchSize::SmallInput,
                )
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_concurrent_burst, bench_cb_on_off);
criterion_main!(benches);
