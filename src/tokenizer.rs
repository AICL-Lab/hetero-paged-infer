//! 分词器 - 文本与 Token 转换
//!
//! 提供简单的字符级分词器，用于测试目的。
//! 生产环境可替换为真实分词器（如 SentencePiece、tiktoken）。
//!
//! # 特殊 Token
//!
//! | Token ID | 名称 | 说明 |
//! |----------|------|------|
//! | 0 | PAD | 填充 token |
//! | 1 | BOS | 句首 token |
//! | 2 | EOS | 句尾 token |
//! | 3 | UNK | 未知 token |
//!
//! 注意：这些默认值可通过 [`SpecialTokenIds`] 配置。
//!
//! # 示例
//!
//! ```rust
//! use paged_infer::{SimpleTokenizer, TokenizerTrait};
//!
//! let tokenizer = SimpleTokenizer::new();
//!
//! let tokens = tokenizer.encode("Hello");
//! let text = tokenizer.decode(&tokens);
//! ```

use crate::config::{EngineConfig, SpecialTokenIds, TokenizerKind};
use crate::error::{ConfigError, EngineError};
use crate::types::TokenId;
use std::collections::{hash_map::Entry, HashMap};
use std::path::Path;
use tokenizers::Tokenizer;

/// 分词器 trait 接口
///
/// 定义分词器的标准接口。
pub trait TokenizerTrait: Send + Sync {
    /// 将文本编码为 token ID 序列（可失败）
    fn try_encode(&self, text: &str) -> Result<Vec<TokenId>, String>;

    /// 将 token ID 序列解码为文本（可失败）
    fn try_decode(&self, tokens: &[TokenId]) -> Result<String, String>;

    /// 将文本编码为 token ID 序列
    ///
    /// 这是 [`try_encode`](Self::try_encode) 的便捷封装。引擎内部使用
    /// `try_encode` 以优雅地处理错误；本方法面向不可能失败的调用场景。
    ///
    /// # Panics
    ///
    /// 底层 `try_encode` 失败时 panic。
    fn encode(&self, text: &str) -> Vec<TokenId> {
        self.try_encode(text)
            .unwrap_or_else(|err| panic!("tokenizer encode failed: {err}"))
    }

    /// 将 token ID 序列解码为文本
    ///
    /// 这是 [`try_decode`](Self::try_decode) 的便捷封装。引擎内部使用
    /// `try_decode` 以优雅地处理错误；本方法面向不可能失败的调用场景。
    ///
    /// # Panics
    ///
    /// 底层 `try_decode` 失败时 panic。
    fn decode(&self, tokens: &[TokenId]) -> String {
        self.try_decode(tokens)
            .unwrap_or_else(|err| panic!("tokenizer decode failed: {err}"))
    }

    /// 获取词表大小
    fn vocab_size(&self) -> u32;

    /// 获取 BOS token ID
    fn bos_token_id(&self) -> TokenId;

    /// 获取 EOS token ID
    fn eos_token_id(&self) -> TokenId;

    /// 获取 PAD token ID
    fn pad_token_id(&self) -> TokenId;

    /// 创建一个属于单个请求的增量解码器
    ///
    /// 流式响应必须保证：对任意成功生成的 token 序列，
    /// `push`/`finish` 输出的全部片段拼接 == 对同一序列一次性 `try_decode` 的文本。
    ///
    /// 解码器状态按请求持有，请求完成、取消或失败时随之销毁。
    fn create_decoder(&self) -> Box<dyn IncrementalDecoder>;
}

/// 每请求增量解码器
///
/// 禁止用“每次 decode 全前缀再取字符串后缀”的方式实现：某些 decoder 会
/// 修改尾部空格或 byte 序列，已发送的片段无法撤回。
pub trait IncrementalDecoder: Send {
    /// 喂入一个新生成的 token。
    ///
    /// 返回此刻可安全下发的文本片段；若解码器需要更多上下文
    /// （例如 byte/wordpiece 跨 token 边界），返回 `Ok(None)`。
    fn push(&mut self, token: TokenId) -> Result<Option<String>, String>;

    /// 冲刷末尾状态。请求到达终态时调用一次，返回值是最后一段文本。
    fn finish(&mut self) -> Result<Option<String>, String>;
}

/// 简单字符级分词器
///
/// 将每个 ASCII 字符映射为唯一的 token ID。
/// 特殊 token: PAD=0, BOS=1, EOS=2, UNK=3 (可通过配置修改)
/// 常规字符从 ID 4 开始。
#[derive(Debug, Clone)]
pub struct SimpleTokenizer {
    /// Character to token ID mapping
    char_to_id: HashMap<char, TokenId>,
    /// Token ID to character mapping
    id_to_char: HashMap<TokenId, char>,
    /// Vocabulary size
    vocab_size: u32,
    /// Special token IDs
    special_tokens: SpecialTokenIds,
    /// Whether encode adds BOS/EOS and decode skips them
    add_special_tokens: bool,
}

impl SimpleTokenizer {
    /// Create a new simple tokenizer with ASCII vocabulary and default special tokens
    pub fn new() -> Self {
        Self::with_special_tokens(SpecialTokenIds::default())
    }

    /// Create a new simple tokenizer with custom special token IDs
    pub fn with_special_tokens(special_tokens: SpecialTokenIds) -> Self {
        let mut char_to_id = HashMap::new();
        let mut id_to_char = HashMap::new();

        // Reserve special tokens
        let mut next_id: TokenId = 4;

        // Add printable ASCII characters (32-126)
        for c in (32u8..=126).map(|b| b as char) {
            char_to_id.insert(c, next_id);
            id_to_char.insert(next_id, c);
            next_id += 1;
        }

        // Add common whitespace
        for c in ['\n', '\r', '\t'] {
            if let Entry::Vacant(entry) = char_to_id.entry(c) {
                entry.insert(next_id);
                id_to_char.insert(next_id, c);
                next_id += 1;
            }
        }

        Self {
            char_to_id,
            id_to_char,
            vocab_size: next_id,
            special_tokens,
            add_special_tokens: true,
        }
    }

    /// Create a tokenizer that does not add/strip BOS/EOS (exact round-trip)
    pub fn without_special_tokens() -> Self {
        let mut t = Self::new();
        t.add_special_tokens = false;
        t
    }

    /// Encode a single character
    fn encode_char(&self, c: char) -> TokenId {
        *self.char_to_id.get(&c).unwrap_or(&self.special_tokens.unk)
    }

    /// Decode a single token
    fn decode_token(&self, token: TokenId) -> Option<char> {
        self.id_to_char.get(&token).copied()
    }

    /// Get the special token IDs configuration
    pub fn special_tokens(&self) -> &SpecialTokenIds {
        &self.special_tokens
    }
}

impl Default for SimpleTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenizerTrait for SimpleTokenizer {
    fn try_encode(&self, text: &str) -> Result<Vec<TokenId>, String> {
        let mut tokens = Vec::with_capacity(text.len() + 2);

        if self.add_special_tokens {
            tokens.push(self.special_tokens.bos);
        }

        for c in text.chars() {
            tokens.push(self.encode_char(c));
        }

        if self.add_special_tokens {
            tokens.push(self.special_tokens.eos);
        }

        Ok(tokens)
    }

    fn try_decode(&self, tokens: &[TokenId]) -> Result<String, String> {
        let mut result = String::with_capacity(tokens.len());

        for &token in tokens {
            if self.add_special_tokens
                && (token == self.special_tokens.bos
                    || token == self.special_tokens.eos
                    || token == self.special_tokens.pad)
            {
                continue;
            }

            if let Some(c) = self.decode_token(token) {
                result.push(c);
            }
        }

        Ok(result)
    }

    fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    fn bos_token_id(&self) -> TokenId {
        self.special_tokens.bos
    }

    fn eos_token_id(&self) -> TokenId {
        self.special_tokens.eos
    }

    fn pad_token_id(&self) -> TokenId {
        self.special_tokens.pad
    }

    fn create_decoder(&self) -> Box<dyn IncrementalDecoder> {
        Box::new(SimpleDecoder {
            id_to_char: self.id_to_char.clone(),
            special_tokens: self.special_tokens.clone(),
            add_special_tokens: self.add_special_tokens,
        })
    }
}

/// SimpleTokenizer 的增量解码器：token 与字符一一对应，可以逐 token 输出。
struct SimpleDecoder {
    id_to_char: HashMap<TokenId, char>,
    special_tokens: SpecialTokenIds,
    add_special_tokens: bool,
}

impl IncrementalDecoder for SimpleDecoder {
    fn push(&mut self, token: TokenId) -> Result<Option<String>, String> {
        // 与 SimpleTokenizer::try_decode 保持完全一致的跳过规则
        if self.add_special_tokens
            && (token == self.special_tokens.bos
                || token == self.special_tokens.eos
                || token == self.special_tokens.pad)
        {
            return Ok(None);
        }
        Ok(self.id_to_char.get(&token).map(|c| c.to_string()))
    }

    fn finish(&mut self) -> Result<Option<String>, String> {
        Ok(None)
    }
}
#[derive(Debug, Clone)]
pub struct HuggingFaceTokenizer {
    inner: Tokenizer,
    special_tokens: SpecialTokenIds,
}

/// 常见 BOS token 名（按优先级；Qwen2 系为 `<|endoftext|>`）
const BOS_TOKEN_NAMES: &[&str] = &["<|endoftext|>", "<s>", "<bos>", "<BOS>"];
/// 常见 EOS token 名（Qwen2 系为 `<|im_end|>`）
const EOS_TOKEN_NAMES: &[&str] = &["<|im_end|>", "</s>", "<eos>", "<EOS>", "<|endoftext|>"];
/// 常见 PAD token 名（Qwen2 系与 BOS 同为 `<|endoftext|>`）
const PAD_TOKEN_NAMES: &[&str] = &["<|endoftext|>", "<pad>", "<PAD>"];

/// 返回词表中第一个命中的 token id
fn first_token_id(inner: &Tokenizer, names: &[&str]) -> Option<u32> {
    names.iter().find_map(|name| inner.token_to_id(name))
}

impl HuggingFaceTokenizer {
    /// 从 tokenizer JSON 文件创建
    pub fn from_file(path: &Path) -> Result<Self, String> {
        Self::with_special_tokens_from_file(path, SpecialTokenIds::default())
    }

    /// 使用自定义 special token 配置从文件创建。
    ///
    /// 特殊 token ID 优先从词表探测真实值（Qwen2：BOS/PAD=151643、EOS=151645），
    /// 命中则覆盖传入配置；未命中才回退到配置值。
    pub fn with_special_tokens_from_file(
        path: &Path,
        mut special_tokens: SpecialTokenIds,
    ) -> Result<Self, String> {
        let inner = Tokenizer::from_file(path).map_err(|e| e.to_string())?;
        if let Some(id) = first_token_id(&inner, BOS_TOKEN_NAMES) {
            special_tokens.bos = id;
        }
        if let Some(id) = first_token_id(&inner, EOS_TOKEN_NAMES) {
            special_tokens.eos = id;
        }
        if let Some(id) = first_token_id(&inner, PAD_TOKEN_NAMES) {
            special_tokens.pad = id;
        }
        Ok(Self {
            inner,
            special_tokens,
        })
    }
}

impl TokenizerTrait for HuggingFaceTokenizer {
    fn try_encode(&self, text: &str) -> Result<Vec<TokenId>, String> {
        // 不自动添加特殊 token（Qwen2 等真实模型 add_bos=false，聊天模板由上层
        // 拼装），与 tiny-llm 的 tokenizer 语义保持一致，保证两侧词表对齐。
        self.inner
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|e| e.to_string())
    }

    fn try_decode(&self, tokens: &[TokenId]) -> Result<String, String> {
        self.inner.decode(tokens, true).map_err(|e| e.to_string())
    }

    fn vocab_size(&self) -> u32 {
        // tokenizer 的完整词表（含 added tokens 特殊 token）。Qwen2.5-0.5B 为
        // 151665；模型 embedding 虽是 151936，多出的 271 个是 llama.cpp 填充的
        // [PADnnnn] 占位符（未训练行），不影响编码一致性。
        self.inner.get_vocab_size(true) as u32
    }

    fn bos_token_id(&self) -> TokenId {
        self.special_tokens.bos
    }

    fn eos_token_id(&self) -> TokenId {
        self.special_tokens.eos
    }

    fn pad_token_id(&self) -> TokenId {
        self.special_tokens.pad
    }

    fn create_decoder(&self) -> Box<dyn IncrementalDecoder> {
        // 当前依赖的 tokenizers 版本不提供安全的逐 token streaming decode
        // （BPE/WordPiece 可能跨 token 修改尾部字节）。先缓冲 token、仅在
        // finish 时输出完整解码结果，不伪造增量语义。
        Box::new(BufferedDecoder {
            inner: self.inner.clone(),
            tokens: Vec::new(),
        })
    }
}

/// HuggingFace tokenizer 的增量解码器：缓冲全部 token，finish 时一次性解码。
struct BufferedDecoder {
    inner: Tokenizer,
    tokens: Vec<TokenId>,
}

impl IncrementalDecoder for BufferedDecoder {
    fn push(&mut self, token: TokenId) -> Result<Option<String>, String> {
        self.tokens.push(token);
        Ok(None)
    }

    fn finish(&mut self) -> Result<Option<String>, String> {
        if self.tokens.is_empty() {
            return Ok(None);
        }
        self.inner
            .decode(&self.tokens, true)
            .map(Some)
            .map_err(|e| e.to_string())
    }
}
pub fn build_tokenizer(config: &EngineConfig) -> Result<Box<dyn TokenizerTrait>, EngineError> {
    match config.tokenizer.kind {
        TokenizerKind::Simple => Ok(Box::new(SimpleTokenizer::with_special_tokens(
            config.special_tokens.clone(),
        ))),
        TokenizerKind::HuggingFace => {
            let path = config
                .tokenizer
                .path
                .as_deref()
                .ok_or(ConfigError::MissingTokenizerPath)?;
            let tokenizer = HuggingFaceTokenizer::with_special_tokens_from_file(
                path,
                config.special_tokens.clone(),
            )
            .map_err(EngineError::Tokenization)?;
            Ok(Box::new(tokenizer))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{EngineConfig, TokenizerConfig, TokenizerKind};
    use crate::test_utils::write_test_tokenizer_json;
    use std::fs;

    #[test]
    fn test_simple_tokenizer_encode() {
        let tokenizer = SimpleTokenizer::new();
        let special_tokens = tokenizer.special_tokens();

        let tokens = tokenizer.encode("Hi");

        // Should have BOS + 'H' + 'i' + EOS
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0], special_tokens.bos);
        assert_eq!(tokens[tokens.len() - 1], special_tokens.eos);
    }

    #[test]
    fn test_simple_tokenizer_decode() {
        let tokenizer = SimpleTokenizer::new();

        let tokens = tokenizer.encode("Hello");
        let decoded = tokenizer.decode(&tokens);

        assert_eq!(decoded, "Hello");
    }

    #[test]
    fn test_empty_string() {
        let tokenizer = SimpleTokenizer::new();

        let tokens = tokenizer.encode("");
        assert_eq!(tokens.len(), 2); // BOS + EOS

        let decoded = tokenizer.decode(&tokens);
        assert_eq!(decoded, "");
    }

    #[test]
    fn test_special_characters() {
        let tokenizer = SimpleTokenizer::new();

        let text = "Hello, World!\n";
        let tokens = tokenizer.encode(text);
        let decoded = tokenizer.decode(&tokens);

        assert_eq!(decoded, text);
    }

    #[test]
    fn test_round_trip_tokenizer() {
        let tokenizer = SimpleTokenizer::without_special_tokens();

        let text = "Hello World 123!";
        let tokens = tokenizer.encode(text);
        let decoded = tokenizer.decode(&tokens);

        assert_eq!(decoded, text);
    }

    #[test]
    fn test_vocab_size() {
        let tokenizer = SimpleTokenizer::new();

        // Should have special tokens + printable ASCII + whitespace
        assert!(tokenizer.vocab_size() > 100);
    }

    #[test]
    fn test_custom_special_tokens() {
        let custom_tokens = SpecialTokenIds {
            bos: 100,
            eos: 101,
            pad: 102,
            unk: 103,
        };
        let tokenizer = SimpleTokenizer::with_special_tokens(custom_tokens.clone());

        assert_eq!(tokenizer.bos_token_id(), 100);
        assert_eq!(tokenizer.eos_token_id(), 101);
        assert_eq!(tokenizer.pad_token_id(), 102);

        let tokens = tokenizer.encode("Hi");
        assert_eq!(tokens[0], 100); // BOS
        assert_eq!(tokens[tokens.len() - 1], 101); // EOS
    }

    #[test]
    fn test_huggingface_tokenizer_loads_and_round_trips() {
        let path = write_test_tokenizer_json();
        let tokenizer = HuggingFaceTokenizer::from_file(&path).unwrap();

        let tokens = tokenizer.encode("hello world");
        let decoded = tokenizer.decode(&tokens);

        assert!(!tokens.is_empty());
        assert_eq!(decoded, "hello world");

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_build_tokenizer_uses_huggingface_when_configured() {
        let path = write_test_tokenizer_json();
        let config = EngineConfig {
            tokenizer: TokenizerConfig {
                kind: TokenizerKind::HuggingFace,
                path: Some(path.clone()),
            },
            ..Default::default()
        };

        let tokenizer = build_tokenizer(&config).unwrap();
        let decoded = tokenizer.decode(&tokenizer.encode("hello world"));

        assert_eq!(decoded, "hello world");

        let _ = fs::remove_file(path);
    }

    /// PINF-102 流式等价性质：逐 token push 的片段拼接 + finish 输出
    /// 必须等于对同一 token 序列的一次性 decode。
    #[test]
    fn test_simple_decoder_streaming_equivalence() {
        let tokenizer = SimpleTokenizer::new();
        let tokens = tokenizer.encode("Hello, World!\n");

        let mut decoder = tokenizer.create_decoder();
        let mut streamed = String::new();
        for &token in &tokens {
            if let Some(chunk) = decoder.push(token).unwrap() {
                streamed.push_str(&chunk);
            }
        }
        if let Some(tail) = decoder.finish().unwrap() {
            streamed.push_str(&tail);
        }

        assert_eq!(streamed, tokenizer.decode(&tokens));
        assert_eq!(streamed, "Hello, World!\n");
    }

    #[test]
    fn test_simple_decoder_without_special_tokens_round_trip() {
        let tokenizer = SimpleTokenizer::without_special_tokens();
        let tokens = tokenizer.encode("abc 123");

        let mut decoder = tokenizer.create_decoder();
        let mut streamed = String::new();
        for &token in &tokens {
            if let Some(chunk) = decoder.push(token).unwrap() {
                streamed.push_str(&chunk);
            }
        }
        if let Some(tail) = decoder.finish().unwrap() {
            streamed.push_str(&tail);
        }

        assert_eq!(streamed, "abc 123");
    }

    /// 生成过程中混入特殊 token（如 EOS）时，增量输出与一次性 decode 一致跳过。
    #[test]
    fn test_simple_decoder_skips_special_tokens_like_one_shot() {
        let tokenizer = SimpleTokenizer::new();
        let specials = tokenizer.special_tokens().clone();
        let h = tokenizer.encode("H")[1]; // skip BOS
        let i = tokenizer.encode("i")[1];
        let tokens = vec![h, specials.eos, i, specials.pad];

        let mut decoder = tokenizer.create_decoder();
        let mut streamed = String::new();
        for &token in &tokens {
            if let Some(chunk) = decoder.push(token).unwrap() {
                streamed.push_str(&chunk);
            }
        }
        if let Some(tail) = decoder.finish().unwrap() {
            streamed.push_str(&tail);
        }

        assert_eq!(streamed, tokenizer.decode(&tokens));
        assert_eq!(streamed, "Hi");
    }

    /// 非 ASCII（中文/UTF-8）输入：SimpleTokenizer 把未知字符映射为 UNK，
    /// 增量与一次性 decode 一致地丢弃它——等价性质对损失路径同样成立。
    #[test]
    fn test_simple_decoder_utf8_equivalence() {
        let tokenizer = SimpleTokenizer::new();
        let tokens = tokenizer.encode("hi 你好");

        let mut decoder = tokenizer.create_decoder();
        let mut streamed = String::new();
        for &token in &tokens {
            if let Some(chunk) = decoder.push(token).unwrap() {
                streamed.push_str(&chunk);
            }
        }
        if let Some(tail) = decoder.finish().unwrap() {
            streamed.push_str(&tail);
        }

        assert_eq!(streamed, tokenizer.decode(&tokens));
        assert_eq!(streamed, "hi ");
    }

    /// HuggingFace 适配器缓冲 token、finish 时输出完整文本；
    /// 拼接性质仍然成立（中间片段为空）。
    #[test]
    fn test_huggingface_decoder_buffers_until_finish() {
        let path = write_test_tokenizer_json();
        let tokenizer = HuggingFaceTokenizer::from_file(&path).unwrap();
        let tokens = tokenizer.encode("hello world");

        let mut decoder = tokenizer.create_decoder();
        let mut streamed = String::new();
        for &token in &tokens {
            let chunk = decoder.push(token).unwrap();
            assert!(chunk.is_none(), "buffered decoder must not emit mid-stream");
        }
        if let Some(tail) = decoder.finish().unwrap() {
            streamed.push_str(&tail);
        }

        assert_eq!(streamed, tokenizer.decode(&tokens));
        assert_eq!(streamed, "hello world");

        let _ = fs::remove_file(path);
    }

    /// 真实 tokenizer（Qwen2 风格 added tokens）加载时，BOS/EOS/PAD 应从词表
    /// 探测真实值并覆盖配置默认值（1/2/0/3 与真实模型不符）。
    #[test]
    fn test_huggingface_tokenizer_detects_real_special_tokens() {
        let json = r###"{
          "version": "1.0",
          "truncation": null,
          "padding": null,
          "added_tokens": [
            {"id": 151643, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 151644, "content": "<|im_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 151645, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
          ],
          "normalizer": null,
          "pre_tokenizer": { "type": "Whitespace" },
          "post_processor": null,
          "decoder": { "type": "WordPiece", "prefix": "##", "cleanup": false },
          "model": {
            "type": "WordLevel",
            "vocab": {
              "[UNK]": 0,
              "hi": 1,
              "<|endoftext|>": 151643,
              "<|im_start|>": 151644,
              "<|im_end|>": 151645
            },
            "unk_token": "[UNK]"
          }
        }"###;
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("paged-test-special-{unique}.json"));
        fs::write(&path, json).unwrap();

        let tokenizer = HuggingFaceTokenizer::with_special_tokens_from_file(
            &path,
            SpecialTokenIds::default(), // 默认 1/2/0/3，应被探测覆盖
        )
        .unwrap();

        assert_eq!(tokenizer.bos_token_id(), 151643);
        assert_eq!(tokenizer.eos_token_id(), 151645);
        assert_eq!(tokenizer.pad_token_id(), 151643);
        // add_special_tokens=false：encode 不自动添加 BOS/EOS
        assert_eq!(tokenizer.encode("hi"), vec![1]);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_huggingface_decoder_finish_on_empty_is_none() {
        let path = write_test_tokenizer_json();
        let tokenizer = HuggingFaceTokenizer::from_file(&path).unwrap();

        let mut decoder = tokenizer.create_decoder();
        assert_eq!(decoder.finish().unwrap(), None);

        let _ = fs::remove_file(path);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// **Feature: paged-inference-system, Property 15: Tokenizer Round-Trip**
        /// *For any* valid text input, decoding the encoded tokens shall produce text
        /// equivalent to the original input (accounting for normalization).
        /// **Validates: Requirements 8.4**
        #[test]
        fn prop_tokenizer_round_trip(
            text in "[a-zA-Z0-9 .,!?\\-_:;'\"()\\[\\]{}@#$%^&*+=<>/\\\\|~`]{0,100}"
        ) {
            let tokenizer = SimpleTokenizer::without_special_tokens();

            let tokens = tokenizer.encode(&text);
            let decoded = tokenizer.decode(&tokens);

            prop_assert_eq!(
                decoded.clone(),
                text.clone(),
                "Round-trip failed: '{}' -> {:?} -> '{}'",
                text,
                tokens,
                decoded
            );
        }

        /// Property test for ASCII printable characters
        #[test]
        fn prop_ascii_round_trip(
            text in prop::collection::vec(32u8..=126, 0..100)
                .prop_map(|bytes| String::from_utf8(bytes).unwrap())
        ) {
            let tokenizer = SimpleTokenizer::without_special_tokens();

            let tokens = tokenizer.encode(&text);
            let decoded = tokenizer.decode(&tokens);

            prop_assert_eq!(
                decoded,
                text,
                "ASCII round-trip failed"
            );
        }

        /// Property test for encoding consistency
        #[test]
        fn prop_encoding_consistency(
            text in "[a-zA-Z0-9 ]{1,50}"
        ) {
            let tokenizer = SimpleTokenizer::new();

            // Encoding the same text twice should produce the same tokens
            let tokens1 = tokenizer.encode(&text);
            let tokens2 = tokenizer.encode(&text);

            prop_assert_eq!(tokens1, tokens2, "Encoding should be deterministic");
        }

        /// Property test for token count
        #[test]
        fn prop_token_count(
            text in "[a-zA-Z]{0,100}"
        ) {
            let tokenizer = SimpleTokenizer::new();

            let tokens = tokenizer.encode(&text);

            // Should have BOS + characters + EOS
            prop_assert_eq!(
                tokens.len(),
                text.len() + 2,
                "Token count should be text length + 2 (BOS + EOS)"
            );
        }
    }
}
