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
//! use hetero_infer::{SimpleTokenizer, TokenizerTrait};
//!
//! let tokenizer = SimpleTokenizer::new();
//!
//! let tokens = tokenizer.encode("Hello");
//! let text = tokenizer.decode(&tokens);
//! ```

use crate::config::SpecialTokenIds;
use crate::types::TokenId;
use std::collections::{hash_map::Entry, HashMap};

/// Special token IDs (deprecated constants, use SpecialTokenIds instead)
///
/// 这些常量保留用于向后兼容。新代码应使用 [`SpecialTokenIds`]。
#[deprecated(
    since = "0.2.0",
    note = "Use SpecialTokenIds from EngineConfig instead"
)]
pub const BOS_TOKEN_ID: TokenId = 1;
#[deprecated(
    since = "0.2.0",
    note = "Use SpecialTokenIds from EngineConfig instead"
)]
pub const EOS_TOKEN_ID: TokenId = 2;
#[deprecated(
    since = "0.2.0",
    note = "Use SpecialTokenIds from EngineConfig instead"
)]
pub const PAD_TOKEN_ID: TokenId = 0;
#[deprecated(
    since = "0.2.0",
    note = "Use SpecialTokenIds from EngineConfig instead"
)]
pub const UNK_TOKEN_ID: TokenId = 3;

/// 分词器 trait 接口
///
/// 定义分词器的标准接口。
pub trait TokenizerTrait: Send + Sync {
    /// 将文本编码为 token ID 序列
    fn encode(&self, text: &str) -> Vec<TokenId>;

    /// 将 token ID 序列解码为文本
    fn decode(&self, tokens: &[TokenId]) -> String;

    /// 获取词表大小
    fn vocab_size(&self) -> u32;

    /// 获取 BOS token ID
    fn bos_token_id(&self) -> TokenId;

    /// 获取 EOS token ID
    fn eos_token_id(&self) -> TokenId;

    /// 获取 PAD token ID
    fn pad_token_id(&self) -> TokenId;
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
        }
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
    fn encode(&self, text: &str) -> Vec<TokenId> {
        let mut tokens = Vec::with_capacity(text.len() + 2);

        // Add BOS token
        tokens.push(self.special_tokens.bos);

        // Encode each character
        for c in text.chars() {
            tokens.push(self.encode_char(c));
        }

        // Add EOS token
        tokens.push(self.special_tokens.eos);

        tokens
    }

    fn decode(&self, tokens: &[TokenId]) -> String {
        let mut result = String::with_capacity(tokens.len());

        for &token in tokens {
            // Skip special tokens
            if token == self.special_tokens.bos
                || token == self.special_tokens.eos
                || token == self.special_tokens.pad
            {
                continue;
            }

            if let Some(c) = self.decode_token(token) {
                result.push(c);
            }
            // UNK tokens are silently skipped
        }

        result
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
}

/// Tokenizer that preserves exact round-trip for ASCII text
#[derive(Debug, Clone)]
pub struct RoundTripTokenizer {
    inner: SimpleTokenizer,
}

impl RoundTripTokenizer {
    pub fn new() -> Self {
        Self {
            inner: SimpleTokenizer::new(),
        }
    }

    /// Create with custom special token IDs
    pub fn with_special_tokens(special_tokens: SpecialTokenIds) -> Self {
        Self {
            inner: SimpleTokenizer::with_special_tokens(special_tokens),
        }
    }
}

impl Default for RoundTripTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenizerTrait for RoundTripTokenizer {
    fn encode(&self, text: &str) -> Vec<TokenId> {
        // Don't add BOS/EOS for round-trip testing
        text.chars().map(|c| self.inner.encode_char(c)).collect()
    }

    fn decode(&self, tokens: &[TokenId]) -> String {
        let mut result = String::with_capacity(tokens.len());

        for &token in tokens {
            if let Some(c) = self.inner.decode_token(token) {
                result.push(c);
            }
        }

        result
    }

    fn vocab_size(&self) -> u32 {
        self.inner.vocab_size()
    }

    fn bos_token_id(&self) -> TokenId {
        self.inner.bos_token_id()
    }

    fn eos_token_id(&self) -> TokenId {
        self.inner.eos_token_id()
    }

    fn pad_token_id(&self) -> TokenId {
        self.inner.pad_token_id()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let tokenizer = RoundTripTokenizer::new();

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
        let custom_tokens = SpecialTokenIds::new(100, 101, 102, 103);
        let tokenizer = SimpleTokenizer::with_special_tokens(custom_tokens.clone());

        assert_eq!(tokenizer.bos_token_id(), 100);
        assert_eq!(tokenizer.eos_token_id(), 101);
        assert_eq!(tokenizer.pad_token_id(), 102);

        let tokens = tokenizer.encode("Hi");
        assert_eq!(tokens[0], 100); // BOS
        assert_eq!(tokens[tokens.len() - 1], 101); // EOS
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// **Feature: heterogeneous-inference-system, Property 15: Tokenizer Round-Trip**
        /// *For any* valid text input, decoding the encoded tokens shall produce text
        /// equivalent to the original input (accounting for normalization).
        /// **Validates: Requirements 8.4**
        #[test]
        fn prop_tokenizer_round_trip(
            text in "[a-zA-Z0-9 .,!?\\-_:;'\"()\\[\\]{}@#$%^&*+=<>/\\\\|~`]{0,100}"
        ) {
            let tokenizer = RoundTripTokenizer::new();

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
            let tokenizer = RoundTripTokenizer::new();

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
