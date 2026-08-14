//! tiny-llm 真实后端接入的端到端测试（里程碑 4）。
//!
//! 门控：
//! - 编译期：`cargo test --features tiny-llm`（启用 C ABI 符号与适配器）
//! - 链接期：`TINY_LLM_DIR` 指向 tiny-llm 构建目录（build.rs）
//! - 运行期：`TINY_LLM_MODEL` 指向真实 GGUF 模型
//!
//! 使用 paged-infer 自带的 SimpleTokenizer（词表语义与 Qwen 不同），
//! 本测试验证的是**接入流程正确性**（引擎驱动、KV 生命周期、资源守恒、
//! 能力声明），而非文本质量；文本质量验证需接入与模型词表一致的 tokenizer。
//!
//! 注意：单个测试进程只 load 一次模型（0.5B 模型 + 多序列 KV 在 6GB 卡上
//! 无法容纳多次并发实例），因此能力检查与端到端流程合并在同一测试内。

#![cfg(feature = "tiny-llm")]

use paged_infer::test_utils::create_test_config;
use paged_infer::{
    EngineConfig, EngineError, GenerationParams, InferenceEngine, Scheduler, SimpleTokenizer,
    TinyLlmExecutor,
};

fn model_path() -> Option<String> {
    std::env::var("TINY_LLM_MODEL").ok()
}

#[test]
fn tiny_llm_backend_end_to_end() {
    let Some(path) = model_path() else {
        eprintln!("skip: set TINY_LLM_MODEL to a GGUF file to enable");
        return;
    };

    let mut config = create_test_config();
    config.max_num_blocks = 256;
    config.max_model_len = 256; // 与 tiny-llm KV 预留（context+512）匹配
    config.max_total_tokens = 1024;

    let executor = TinyLlmExecutor::new(&path, config.clone()).expect("tinyllm_load failed");
    let mut engine = InferenceEngine::with_components(
        config.clone(),
        Box::new(SimpleTokenizer::new()),
        Scheduler::new(config),
        Box::new(executor),
    )
    .unwrap();

    // 能力声明：非 greedy 参数应在 submit 阶段被诚实拒绝
    let params = GenerationParams {
        max_tokens: 4,
        temperature: 0.8,
        ..GenerationParams::default()
    };
    let err = engine.submit_request("sampling", params).unwrap_err();
    assert!(matches!(
        err,
        EngineError::UnsupportedGenerationMode(_)
    ));

    // 并发提交 3 个请求（greedy），驱动到全部完成
    for i in 0..3 {
        let params = GenerationParams {
            max_tokens: 4,
            ..GenerationParams::default()
        };
        engine
            .submit_request(&format!("prompt {}", i), params)
            .unwrap();
    }

    let completed = engine.run();
    assert_eq!(completed.len(), 3, "全部请求应完成");
    for c in &completed {
        assert!(c.success, "request {} failed: {:?}", c.request_id, c.error);
    }

    // 资源守恒：全部完成后 KV 归还
    let util = engine.memory_utilization();
    assert!(util < 0.05, "tiny-llm KV 未归还: {util}");

    let m = engine.get_metrics();
    assert_eq!(m.completed_requests, 3);
    assert_eq!(m.failed_requests, 0);
}
