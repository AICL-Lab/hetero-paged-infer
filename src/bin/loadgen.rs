//! Serving 压测客户端（loadgen）：对 OpenAI 兼容 `/v1/completions`（SSE 流式）
//! 端点做可复现负载实验。
//!
//! 同一二进制零改动覆盖三个后端：paged-infer / llama-server / vLLM，
//! 保证横向可比（口径定义见 `benchmarks/serving/methodology.md`）。
//!
//! # 负载模型
//! - `--mode closed`：闭环饱和——固定 `--concurrency` 个并发槽，一个请求完成
//!   立即补发下一个。测最大吞吐与尾延迟。
//! - `--mode poisson`：开环泊松——按 `--rate` req/s 的指数间隔到达，请求独立
//!   并发。测 SLO 曲线（TTFT p95 vs 到达率）。
//!
//! # 指标口径（客户端侧）
//! - TTFT：响应头就绪（请求体发送完成）→ 首个**非空文本** chunk 的墙钟。
//!   空文本 chunk（如部分服务器的前导帧）不计，避免扭曲首 token 语义。
//! - ITL：相邻非空文本 chunk 的到达间隔（首 chunk 除外）。
//! - completion_tokens：优先取最终帧 `usage.completion_tokens`；缺失时回退
//!   非空 chunk 计数（记录中标记 `tokens_source="chunks"`）。
//! - 失败归类：timeout / http_429 / http_4xx / http_5xx / connection /
//!   stream_error（SSE 内 error 载荷）/ no_done（流提前结束）。
//!
//! # 输出
//! - `--out` 指定 per_request.jsonl（每行一条请求记录）；
//! - stdout 打印汇总（成功率、TTFT/ITL/TPOT 分位、吞吐）。
//!
//! # 示例
//! ```bash
//! cargo run --release --bin loadgen -- \
//!   --base-url http://127.0.0.1:3000 --mode closed --concurrency 4 \
//!   --dataset benchmarks/serving/datasets/synth/work.jsonl \
//!   --requests 64 --warmup-secs 30 --max-tokens 128 \
//!   --out benchmarks/serving/results/run1/per_request.jsonl
//! ```

use clap::Parser;
use futures_util::StreamExt;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(about = "OpenAI 兼容 serving 端点压测客户端（闭环/泊松双模式）")]
struct Args {
    /// 目标服务 base URL，如 http://127.0.0.1:3000
    #[arg(long)]
    base_url: String,

    /// 负载模式：closed（闭环饱和）| poisson（开环泊松到达）
    #[arg(long, default_value = "closed")]
    mode: String,

    /// 闭环并发槽数（closed 模式）
    #[arg(long, default_value_t = 1)]
    concurrency: usize,

    /// 平均到达率 req/s（poisson 模式）
    #[arg(long, default_value_t = 1.0)]
    rate: f64,

    /// 数据集文件（jsonl，每行 {"prompt": "...", "prompt_tokens": 可选}）
    #[arg(long)]
    dataset: String,

    /// 测量窗口内总请求数（warmup 不计入）
    #[arg(long, default_value_t = 64)]
    requests: usize,

    /// warmup 秒数：以相同负载形态运行并丢弃结果（0 = 跳过）
    #[arg(long, default_value_t = 30)]
    warmup_secs: u64,

    /// 每请求生成 token 上限
    #[arg(long, default_value_t = 128)]
    max_tokens: u32,

    /// 单请求超时（秒）
    #[arg(long, default_value_t = 120)]
    timeout_secs: u64,

    /// per_request.jsonl 输出路径
    #[arg(long, default_value = "per_request.jsonl")]
    out: String,

    /// model 字段（透传给 API，不影响路由）
    #[arg(long, default_value = "paged-infer")]
    model: String,
}

#[derive(Deserialize, Clone)]
struct DatasetEntry {
    prompt: String,
    /// 离线 tokenize 的 prompt token 数（元数据，仅用于报表核对）
    #[serde(default)]
    prompt_tokens: Option<u32>,
}

#[derive(Serialize, Clone)]
struct RequestRecord {
    request_id: usize,
    /// 测量窗口内序号（warmup 请求为 null）
    measured_index: Option<usize>,
    ok: bool,
    error_class: Option<String>,
    /// 错误详情（stream_error 的服务端消息等；聚合统计仍用 error_class）
    error_detail: Option<String>,
    ttft_ms: Option<f64>,
    itl_ms: Vec<f64>,
    duration_ms: f64,
    /// 非空文本 chunk 数
    chunks: u32,
    completion_tokens: Option<u32>,
    /// completion_tokens 来源："usage" | "chunks"（回退）
    tokens_source: Option<String>,
    finish_reason: Option<String>,
    prompt_tokens_meta: Option<u32>,
}

/// 单请求执行：发送流式 completions 请求，逐 chunk 打时间戳。
async fn run_request(
    client: &reqwest::Client,
    args: &Args,
    request_id: usize,
    measured_index: Option<usize>,
    entry: &DatasetEntry,
) -> RequestRecord {
    let body = serde_json::json!({
        "model": args.model,
        "prompt": entry.prompt,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "stream": true,
    });

    let t_start = Instant::now();
    let resp = client
        .post(format!("{}/v1/completions", args.base_url))
        .json(&body)
        .timeout(Duration::from_secs(args.timeout_secs))
        .send()
        .await;

    let resp = match resp {
        Ok(r) => r,
        Err(e) => {
            let class = if e.is_timeout() { "timeout" } else { "connection" };
            return RequestRecord {
                request_id,
                measured_index,
                ok: false,
                error_class: Some(class.to_string()),
                error_detail: Some(e.to_string()),
                ttft_ms: None,
                itl_ms: Vec::new(),
                duration_ms: t_start.elapsed().as_secs_f64() * 1000.0,
                chunks: 0,
                completion_tokens: None,
                tokens_source: None,
                finish_reason: None,
                prompt_tokens_meta: entry.prompt_tokens,
            };
        }
    };

    let status = resp.status();
    if !status.is_success() {
        let class = match status.as_u16() {
            429 => "http_429",
            400..=499 => "http_4xx",
            _ => "http_5xx",
        };
        return RequestRecord {
            request_id,
            measured_index,
            ok: false,
            error_class: Some(class.to_string()),
            error_detail: Some(format!("HTTP {status}")),
            ttft_ms: None,
            itl_ms: Vec::new(),
            duration_ms: t_start.elapsed().as_secs_f64() * 1000.0,
            chunks: 0,
            completion_tokens: None,
            tokens_source: None,
            finish_reason: None,
            prompt_tokens_meta: entry.prompt_tokens,
        };
    }

    // TTFT 起点：响应头就绪（请求体已发送完成）。
    let t_headers = Instant::now();
    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    let mut chunk_times: Vec<Instant> = Vec::new();
    let mut completion_tokens: Option<u32> = None;
    let mut finish_reason: Option<String> = None;
    let mut stream_error: Option<String> = None;
    let mut saw_done = false;

    'outer: while let Some(item) = stream.next().await {
        let bytes = match item {
            Ok(b) => b,
            Err(e) => {
                stream_error = Some(format!("stream read error: {e}"));
                break;
            }
        };
        buf.push_str(&String::from_utf8_lossy(&bytes));

        // SSE 事件以空行分隔；逐块解析已完整的事件。
        while let Some(pos) = buf.find("\n\n") {
            let event = buf[..pos].to_string();
            buf.drain(..pos + 2);

            for line in event.lines() {
                let Some(payload) = line.strip_prefix("data:") else {
                    continue; // 忽略 event:/id:/注释行
                };
                let payload = payload.trim();
                if payload == "[DONE]" {
                    saw_done = true;
                    break 'outer;
                }
                let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) else {
                    continue;
                };
                if let Some(err) = v.get("error") {
                    stream_error = Some(
                        err.get("message")
                            .and_then(|m| m.as_str())
                            .unwrap_or("unknown stream error")
                            .to_string(),
                    );
                    break 'outer;
                }
                // 最终帧：usage / finish_reason（text 通常为空）
                if let Some(usage) = v.get("usage") {
                    if let Some(ct) = usage.get("completion_tokens").and_then(|c| c.as_u64()) {
                        completion_tokens = Some(ct as u32);
                    }
                }
                if let Some(choice) = v.get("choices").and_then(|c| c.get(0)) {
                    if let Some(fr) = choice.get("finish_reason").and_then(|f| f.as_str()) {
                        if !fr.is_empty() {
                            finish_reason = Some(fr.to_string());
                        }
                    }
                    let text = choice
                        .get("text")
                        .and_then(|t| t.as_str())
                        .unwrap_or_default();
                    if !text.is_empty() {
                        chunk_times.push(Instant::now());
                    }
                }
            }
        }
    }

    let duration_ms = t_headers.elapsed().as_secs_f64() * 1000.0;

    // 失败归类：流内 error > 无 [DONE]（即使已收到部分 chunk 也判失败，
    // 成功率口径不掺水）> 正常
    let (ok, error_class, error_detail) = if let Some(msg) = &stream_error {
        (
            false,
            Some("stream_error".to_string()),
            Some(msg.clone()),
        )
    } else if !saw_done {
        (false, Some("no_done".to_string()), None)
    } else {
        (true, None, None)
    };

    let ttft_ms = chunk_times.first().map(|t| {
        (t.duration_since(t_headers)).as_secs_f64() * 1000.0
    });
    let itl_ms: Vec<f64> = chunk_times
        .windows(2)
        .map(|w| (w[1].duration_since(w[0])).as_secs_f64() * 1000.0)
        .collect();

    let (tokens, tokens_source) = match completion_tokens {
        Some(ct) => (Some(ct), Some("usage".to_string())),
        None if !chunk_times.is_empty() => {
            (Some(chunk_times.len() as u32), Some("chunks".to_string()))
        }
        _ => (None, None),
    };

    RequestRecord {
        request_id,
        measured_index,
        ok,
        error_class,
        error_detail,
        ttft_ms,
        itl_ms,
        duration_ms,
        chunks: chunk_times.len() as u32,
        completion_tokens: tokens,
        tokens_source,
        finish_reason,
        prompt_tokens_meta: entry.prompt_tokens,
    }
}

fn load_dataset(path: &str) -> Result<Vec<DatasetEntry>, String> {
    let content = std::fs::read_to_string(path).map_err(|e| format!("读取数据集失败: {e}"))?;
    let mut entries = Vec::new();
    for (i, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let entry: DatasetEntry = serde_json::from_str(line)
            .map_err(|e| format!("数据集第 {} 行解析失败: {e}", i + 1))?;
        entries.push(entry);
    }
    if entries.is_empty() {
        return Err("数据集为空".to_string());
    }
    Ok(entries)
}

fn percentile(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    // 最近秩法（与评测口径一致，避免插值引入的歧义）
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    Some(sorted[idx.min(sorted.len() - 1)])
}

fn print_summary(records: &[RequestRecord], wall_secs: f64, args: &Args) {
    let total = records.len();
    let ok_records: Vec<&RequestRecord> = records.iter().filter(|r| r.ok).collect();
    let ok = ok_records.len();

    let mut ttfts: Vec<f64> = ok_records.iter().filter_map(|r| r.ttft_ms).collect();
    ttfts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut itls: Vec<f64> = ok_records.iter().flat_map(|r| r.itl_ms.clone()).collect();
    itls.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // 逐请求 TPOT：(duration - ttft) / (tokens - 1)
    let mut tpots: Vec<f64> = ok_records
        .iter()
        .filter_map(|r| {
            let tokens = r.completion_tokens.unwrap_or(0);
            match (r.ttft_ms, tokens > 1) {
                (Some(ttft), true) => {
                    Some((r.duration_ms - ttft) / (tokens as f64 - 1.0))
                }
                _ => None,
            }
        })
        .collect();
    tpots.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let total_tokens: u64 = ok_records
        .iter()
        .map(|r| r.completion_tokens.unwrap_or(0) as u64)
        .sum();

    // 失败归类计数
    let mut error_counts: std::collections::BTreeMap<String, usize> = Default::default();
    for r in records.iter().filter(|r| !r.ok) {
        *error_counts
            .entry(r.error_class.clone().unwrap_or_else(|| "unknown".into()))
            .or_insert(0) += 1;
    }

    let fmt = |v: Option<f64>| v.map(|x| format!("{x:.2}")).unwrap_or_else(|| "-".into());
    println!("=== loadgen 汇总 ===");
    println!("模式: {} | 目标: {}", args.mode, args.base_url);
    println!(
        "请求: {total}（成功 {ok}，成功率 {:.2}%）| 测量窗口: {wall_secs:.1}s",
        if total > 0 {
            100.0 * ok as f64 / total as f64
        } else {
            0.0
        }
    );
    if !error_counts.is_empty() {
        let errs: Vec<String> = error_counts
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        println!("失败归类: {}", errs.join(", "));
    }
    println!(
        "TTFT ms  p50/p95/p99: {} / {} / {}",
        fmt(percentile(&ttfts, 50.0)),
        fmt(percentile(&ttfts, 95.0)),
        fmt(percentile(&ttfts, 99.0))
    );
    println!(
        "ITL  ms  p50/p95/p99: {} / {} / {}",
        fmt(percentile(&itls, 50.0)),
        fmt(percentile(&itls, 95.0)),
        fmt(percentile(&itls, 99.0))
    );
    println!(
        "TPOT ms  p50/p95/p99: {} / {} / {}",
        fmt(percentile(&tpots, 50.0)),
        fmt(percentile(&tpots, 95.0)),
        fmt(percentile(&tpots, 99.0))
    );
    if wall_secs > 0.0 {
        println!(
            "吞吐: {:.1} tok/s（客户端侧 Σ输出tokens/窗口时长）| {:.2} req/s",
            total_tokens as f64 / wall_secs,
            ok as f64 / wall_secs
        );
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    env_logger::init();

    if args.mode != "closed" && args.mode != "poisson" {
        eprintln!("--mode 必须是 closed 或 poisson");
        std::process::exit(2);
    }
    if args.mode == "closed" && args.concurrency == 0 {
        eprintln!("closed 模式 --concurrency 必须 >= 1");
        std::process::exit(2);
    }
    if args.mode == "poisson" && args.rate <= 0.0 {
        eprintln!("poisson 模式 --rate 必须 > 0");
        std::process::exit(2);
    }

    let dataset = load_dataset(&args.dataset)?;
    println!(
        "数据集: {} 条 | 模式: {} | 并发/速率: {} | 测量请求数: {} | warmup: {}s",
        dataset.len(),
        args.mode,
        if args.mode == "closed" {
            args.concurrency.to_string()
        } else {
            format!("{} req/s", args.rate)
        },
        args.requests,
        args.warmup_secs
    );

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .pool_max_idle_per_host(args.concurrency.max(16))
        .build()?;

    let args = Arc::new(args);
    let dataset = Arc::new(dataset);
    let records: Arc<std::sync::Mutex<Vec<RequestRecord>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let next_id = Arc::new(AtomicUsize::new(0));

    // 取下一个数据集条目（环绕）；返回 (全局请求 id, 条目)。
    let pick = |next_id: &Arc<AtomicUsize>, dataset: &Arc<Vec<DatasetEntry>>| {
        let id = next_id.fetch_add(1, Ordering::Relaxed);
        (id, dataset[id % dataset.len()].clone())
    };

    let measured_count = Arc::new(AtomicUsize::new(0));

    if args.mode == "closed" {
        // warmup 阶段：闭环跑 warmup_secs，结果丢弃。
        if args.warmup_secs > 0 {
            let stop = Arc::new(AtomicBool::new(false));
            let mut warmup_handles = Vec::new();
            for _ in 0..args.concurrency {
                let (client, args, dataset, next_id, stop) = (
                    client.clone(),
                    args.clone(),
                    dataset.clone(),
                    next_id.clone(),
                    stop.clone(),
                );
                warmup_handles.push(tokio::spawn(async move {
                    while !stop.load(Ordering::Relaxed) {
                        let (id, entry) = pick(&next_id, &dataset);
                        let rec = run_request(&client, &args, id, None, &entry).await;
                        if !rec.ok {
                            log::warn!("warmup 请求 {id} 失败: {:?}", rec.error_class);
                        }
                    }
                }));
            }
            tokio::time::sleep(Duration::from_secs(args.warmup_secs)).await;
            stop.store(true, Ordering::Relaxed);
            for h in warmup_handles {
                let _ = h.await;
            }
            println!("warmup 完成（{}s），开始测量窗口", args.warmup_secs);
        }
        let measure_start = Instant::now();

        // 测量阶段：闭环直到完成 args.requests 个请求。
        let mut handles = Vec::new();
        for _ in 0..args.concurrency {
            let (client, args, dataset, next_id, records, measured_count) = (
                client.clone(),
                args.clone(),
                dataset.clone(),
                next_id.clone(),
                records.clone(),
                measured_count.clone(),
            );
            handles.push(tokio::spawn(async move {
                loop {
                    let idx = measured_count.fetch_add(1, Ordering::Relaxed);
                    if idx >= args.requests {
                        break;
                    }
                    let (id, entry) = pick(&next_id, &dataset);
                    let rec = run_request(&client, &args, id, Some(idx), &entry).await;
                    records.lock().unwrap().push(rec);
                }
            }));
        }
        for h in handles {
            let _ = h.await;
        }
        let wall = measure_start.elapsed().as_secs_f64();
        let recs = finalize_records(&records);
        write_records(&args.out, &recs)?;
        print_summary(&recs, wall, &args);
    } else {
        // poisson 模式：指数间隔到达；warmup 期间同样到达但丢弃。
        let mut rng = rand::thread_rng();
        let mut warmup_handles: Vec<tokio::task::JoinHandle<RequestRecord>> = Vec::new();
        let mut handles: Vec<tokio::task::JoinHandle<()>> = Vec::new();
        let mut issued = 0usize;

        // 先发射 warmup 流量（不计入 issued 上限）。
        if args.warmup_secs > 0 {
            let warmup_until = Instant::now() + Duration::from_secs(args.warmup_secs);
            while Instant::now() < warmup_until {
                let dt = -(1.0 - rng.gen::<f64>()).ln() / args.rate;
                tokio::time::sleep(Duration::from_secs_f64(dt)).await;
                let (id, entry) = pick(&next_id, &dataset);
                let (client, args) = (client.clone(), args.clone());
                warmup_handles.push(tokio::spawn(async move {
                    run_request(&client, &args, id, None, &entry).await
                }));
            }
            for h in warmup_handles {
                let _ = h.await; // warmup 结果丢弃，仅起预热作用
            }
            println!("warmup 完成（{}s），开始测量窗口", args.warmup_secs);
        }
        let measure_start = Instant::now();

        while issued < args.requests {
            let dt = -(1.0 - rng.gen::<f64>()).ln() / args.rate;
            tokio::time::sleep(Duration::from_secs_f64(dt)).await;
            let (id, entry) = pick(&next_id, &dataset);
            let idx = issued;
            issued += 1;
            let (client, args, records) = (client.clone(), args.clone(), records.clone());
            handles.push(tokio::spawn(async move {
                let rec = run_request(&client, &args, id, Some(idx), &entry).await;
                records.lock().unwrap().push(rec);
            }));
        }
        for h in handles {
            let _ = h.await;
        }
        let wall = measure_start.elapsed().as_secs_f64();
        let recs = finalize_records(&records);
        write_records(&args.out, &recs)?;
        print_summary(&recs, wall, &args);
    }

    Ok(())
}

/// 按测量窗口序号排序后取出全部记录。
fn finalize_records(
    records: &Arc<std::sync::Mutex<Vec<RequestRecord>>>,
) -> Vec<RequestRecord> {
    let mut g = records.lock().unwrap();
    g.sort_by_key(|r| r.measured_index);
    (*g).clone()
}

fn write_records(path: &str, records: &[RequestRecord]) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut out = String::new();
    for r in records {
        out.push_str(&serde_json::to_string(r)?);
        out.push('\n');
    }
    std::fs::write(path, out)?;
    println!("逐请求记录已写入: {path}（{} 条）", records.len());
    Ok(())
}
