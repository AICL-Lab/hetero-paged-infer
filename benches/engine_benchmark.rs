//! Performance benchmarks for Hetero-Paged-Infer
//!
//! Run with: cargo bench

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hetero_infer::{EngineConfig, GenerationParams, InferenceEngine, KVCacheManager, Scheduler};
use std::hint::black_box;

/// Benchmark engine creation
fn bench_engine_creation(c: &mut Criterion) {
    c.bench_function("engine_creation", |b| {
        b.iter(|| {
            let config = EngineConfig::default();
            InferenceEngine::new(config).unwrap()
        })
    });
}

/// Benchmark single request submission
fn bench_request_submission(c: &mut Criterion) {
    let config = EngineConfig::default();
    let params = GenerationParams {
        max_tokens: 10,
        temperature: 1.0,
        top_p: 0.9,
    };

    // 每批使用全新引擎：向同一引擎无限提交会超过 max_num_seqs 而 panic
    // （criterion 会执行上千次迭代）。引擎创建在 setup 闭包内，不计入测量。
    c.bench_function("request_submission", |b| {
        b.iter_batched(
            || InferenceEngine::new(config.clone()).unwrap(),
            |mut engine| {
                let id = engine.submit_request("Hello world", params).unwrap();
                black_box(id)
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

/// Benchmark engine step operation
fn bench_engine_step(c: &mut Criterion) {
    let config = EngineConfig::default();
    let params = GenerationParams {
        max_tokens: 10,
        temperature: 1.0,
        top_p: 0.9,
    };

    // 每批使用带待处理请求的全新引擎，保证每个被测 step 都在做真实的
    // 调度 + 执行，而不是在已清空的引擎上空转（空 step 耗时随时间漂移）。
    c.bench_function("engine_step", |b| {
        b.iter_batched(
            || {
                let mut engine = InferenceEngine::new(config.clone()).unwrap();
                for i in 0..10 {
                    engine
                        .submit_request(&format!("Request {}", i), params)
                        .unwrap();
                }
                engine
            },
            |mut engine| {
                let result = engine.step().unwrap();
                black_box(result)
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

/// Benchmark batch processing with the current mock executor pipeline.
/// These numbers are useful for relative engine behavior, not for claiming real CUDA throughput.
fn bench_batch_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing");

    for size in [1, 4, 8, 16, 32].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let config = EngineConfig {
                max_batch_size: 32,
                ..EngineConfig::default()
            };

            b.iter_batched(
                || {
                    let mut engine = InferenceEngine::new(config.clone()).unwrap();
                    let params = GenerationParams {
                        max_tokens: 5,
                        temperature: 1.0,
                        top_p: 0.9,
                    };
                    for i in 0..size {
                        engine
                            .submit_request(&format!("Request {}", i), params)
                            .unwrap();
                    }
                    engine
                },
                |mut engine| {
                    let completed = engine.run();
                    black_box(completed)
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

/// Benchmark KV Cache allocation
fn bench_kv_cache_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache");

    for num_tokens in [16, 32, 64, 128, 256].iter() {
        group.bench_with_input(
            BenchmarkId::new("allocate_sequence", num_tokens),
            num_tokens,
            |b, &num_tokens| {
                let mut manager = KVCacheManager::new(1000, 16);
                b.iter(|| {
                    let _ = manager.allocate_sequence(1, num_tokens);
                    manager.free_sequence(1);
                })
            },
        );
    }

    group.finish();
}

/// Benchmark KV Cache block growth
fn bench_kv_cache_growth(c: &mut Criterion) {
    let config = EngineConfig::default();

    // 每批使用全新管理器：对同一管理器无限增长会耗尽块池，
    // 之后测到的全是 OutOfBlocks 错误路径而非分配路径。
    c.bench_function("allocate_block", |b| {
        b.iter_batched(
            || {
                let mut manager = KVCacheManager::new(config.max_num_blocks, config.block_size);
                manager.allocate_sequence(1, 16).unwrap();
                manager
            },
            |mut manager| {
                let result = manager.allocate_block(1);
                black_box(result)
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

/// Benchmark scheduler operations
fn bench_scheduler(c: &mut Criterion) {
    let mut group = c.benchmark_group("scheduler");

    group.bench_function("schedule_empty", |b| {
        let config = EngineConfig::default();
        let mut scheduler = Scheduler::new(config);
        b.iter(|| {
            let output = scheduler.schedule();
            black_box(output);
        })
    });

    group.bench_function("schedule_with_requests", |b| {
        let config = EngineConfig::default();
        b.iter_batched(
            || {
                let mut scheduler = Scheduler::new(config.clone());
                use hetero_infer::{Request, RequestState};
                for i in 1..=5 {
                    let mut request =
                        Request::new(i, vec![1, 2, 3, 4, 5], GenerationParams::default());
                    request.state = RequestState::Pending;
                    let _ = scheduler.add_request(request);
                }
                scheduler
            },
            |mut scheduler| {
                let output = scheduler.schedule();
                black_box(output)
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

/// Benchmark configuration validation
fn bench_config(c: &mut Criterion) {
    let config = EngineConfig::default();
    c.bench_function("config_validation", |b| {
        b.iter(|| config.validate().unwrap())
    });
}

criterion_group!(
    benches,
    bench_engine_creation,
    bench_request_submission,
    bench_engine_step,
    bench_batch_sizes,
    bench_kv_cache_allocation,
    bench_kv_cache_growth,
    bench_scheduler,
    bench_config,
);

criterion_main!(benches);
