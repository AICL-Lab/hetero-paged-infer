//! Qwen2 真实文本生成端到端验证（真实后端 + 与模型词表一致的 tokenizer）。
//!
//! 门控：
//! - 编译期：`cargo test --features tiny-llm`
//! - 运行期：`TINY_LLM_MODEL`（GGUF）、`PINF_TOKENIZER_JSON`（tokenizer.json）
//!
//! 与 `tiny_llm_backend.rs`（验证接入流程）不同，本测试验证**文本质量**：
//! 使用 HuggingFaceTokenizer（词表与模型一致），提交真实 prompt，断言输出
//! 可解码且非空、EOS（151645）能正确终止生成，并与 llama.cpp 生成对齐。

#![cfg(feature = "tiny-llm")]

use paged_infer::{
    EngineConfig, GenerationParams, HuggingFaceTokenizer, InferenceEngine, Scheduler,
    TinyLlmExecutor, TokenizerTrait,
};
use std::path::Path;

fn model_path() -> Option<String> {
    std::env::var("TINY_LLM_MODEL").ok()
}

fn tokenizer_path() -> Option<String> {
    std::env::var("PINF_TOKENIZER_JSON").ok()
}

fn build_engine(model: &str, tok_path: &str) -> InferenceEngine {
    let tokenizer =
        HuggingFaceTokenizer::from_file(Path::new(tok_path)).expect("load tokenizer.json");
    assert_eq!(tokenizer.eos_token_id(), 151645, "EOS 应为真实模型值");

    let mut config = EngineConfig::default();
    config.max_num_blocks = 256;
    config.max_model_len = 256;
    config.max_total_tokens = 1024;

    let executor = TinyLlmExecutor::new(model, config.clone()).expect("tinyllm_load failed");
    InferenceEngine::with_components(
        config.clone(),
        Box::new(tokenizer),
        Scheduler::new(config),
        Box::new(executor),
    )
    .unwrap()
}

#[test]
fn qwen2_text_generation_end_to_end() {
    let Some(model) = model_path() else {
        eprintln!("skip: set TINY_LLM_MODEL to a GGUF file to enable");
        return;
    };
    let Some(tok_path) = tokenizer_path() else {
        eprintln!("skip: set PINF_TOKENIZER_JSON to tokenizer.json to enable");
        return;
    };

    let mut engine = build_engine(&model, &tok_path);

    let prompt = "Hello, how are you?";
    let (request_id, prompt_tokens) = engine
        .submit_request(
            prompt,
            GenerationParams {
                max_tokens: 32,
                ..GenerationParams::default()
            },
        )
        .unwrap();

    let completed = engine.run();
    assert_eq!(completed.len(), 1);
    let c = &completed[0];
    assert!(c.success, "request {} failed: {:?}", request_id, c.error);
    assert!(!c.output_text.is_empty(), "输出文本不应为空");
    assert!(!c.output_tokens.is_empty(), "应生成至少 1 个 token");

    eprintln!("prompt_tokens={prompt_tokens} ({prompt})");
    eprintln!(
        "output_tokens({}) = {:?}",
        c.output_tokens.len(),
        c.output_tokens
    );
    eprintln!("output_text = {:#?}", c.output_text);
    eprintln!("finish_reason = {:?}", c.finish_reason);

    // EOS（151645）触发时停止，输出文本不应包含 <|im_end|> 原样串
    assert!(!c.output_text.contains("<|im_end|>"));
}

/// 与 llama.cpp（llama-cli `-st --no-jinja`，greedy）同 prompt 对比：
/// 使用相同的 chat 包装文本（`<|im_start|>user\n...<|im_start|>assistant\n`），
/// 验证 tiny-llm 后端生成的 token 序列与 llama.cpp 完全一致（词表 + 推理数值对齐）。
#[test]
fn qwen2_chat_prompt_matches_llama_cpp() {
    let Some(model) = model_path() else {
        eprintln!("skip: set TINY_LLM_MODEL to a GGUF file to enable");
        return;
    };
    let Some(tok_path) = tokenizer_path() else {
        eprintln!("skip: set PINF_TOKENIZER_JSON to tokenizer.json to enable");
        return;
    };

    let mut engine = build_engine(&model, &tok_path);

    let prompt = "<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\n";
    let (request_id, prompt_tokens) = engine
        .submit_request(
            prompt,
            GenerationParams {
                max_tokens: 64,
                ..GenerationParams::default()
            },
        )
        .unwrap();

    let completed = engine.run();
    assert_eq!(completed.len(), 1);
    let c = &completed[0];
    assert!(c.success, "request {} failed: {:?}", request_id, c.error);

    eprintln!("prompt_tokens={prompt_tokens}");
    eprintln!(
        "output_tokens({}) = {:?}",
        c.output_tokens.len(),
        c.output_tokens
    );
    eprintln!("output_text = {:#?}", c.output_text);
    eprintln!("finish_reason = {:?}", c.finish_reason);

    // llama-cli 参考（greedy，同一 chat prompt，`-n 32` 内 EOS 自然终止）：
    //   [9707 'Hello', 0 '!', 358 ' I', 2776 "'m", 1101 ' just', 264 ' a',
    //    6366 ' computer', 2025 ' program', 11 ',', 773 ' so', 358 ' I',
    //    1513 ' don', 944 "'t", 614 ' have', 15650 ' feelings', 13 '.',
    //    2585 ' How', 646 ' can', 358 ' I', 7789 ' assist', 498 ' you',
    //    3351 ' today', 30 '?', 151645 EOS]
    let reference: &[u32] = &[
        9707, 0, 358, 2776, 1101, 264, 6366, 2025, 11, 773, 358, 1513, 944, 614, 15650, 13, 2585,
        646, 358, 7789, 498, 3351, 30, 151645,
    ];
    // 引擎输出含 eos（finish_reason=Stop，eos 计入 output_tokens）
    eprintln!("对比: 期望 {} 个 token", reference.len());
    assert_eq!(
        c.output_tokens, reference,
        "tiny-llm 与 llama.cpp 生成序列不一致"
    );
    assert_eq!(
        c.output_text,
        "Hello! I'm just a computer program, so I don't have feelings. How can I assist you today?"
    );
}
