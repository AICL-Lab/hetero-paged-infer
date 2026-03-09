//! Tokenizer for text-to-token and token-to-text conversion
//!
//! Provides a simple character-level tokenizer for testing purposes.
//! Can be replaced with a real tokenizer (e.g., SentencePiece, tiktoken) for production.

use std::collections::HashMap;
use crate::types::TokenId;

/// Special token IDs
pub const BOS_TOKEN_ID: TokenId = 1;
pub const EOS_TOKEN_ID: TokenId = 2;
pub const PAD_TOKEN_ID: TokenId = 0;
pub const UNK_TOKEN_ID: TokenId = 3;

/// Tokenizer trait defining the interface
pub trait TokenizerTrait: Send + Sync {
    /// Encode text to token IDs
    fn encode(&self, text: &str) -> Vec<TokenId>;
    
    /// Decode token IDs back to text
    fn decode(&self, tokens: &[TokenId]) -> String;
    
    /// Get vocabulary size
    fn vocab_size(&self) -> u32;
    
    /// Get BOS token ID
    fn bos_token_id(&self) -> TokenId;
    
    /// Get EOS token ID
    fn eos_token_id(&self) -> TokenId;
    
    /// Get PAD token ID
    fn pad_token_id(&self) -> TokenId;
}

/// Simple character-level tokenizer for testing
/// 
/// This tokenizer maps each ASCII character to a unique token ID.
/// Special tokens: PAD=0, BOS=1, EOS=2, UNK=3
/// Regular characters start at ID 4.
#[derive(Debug, Clone)]
pub struct SimpleTokenizer {
    /// Character to token ID mapping
    char_to_id: HashMap<char, TokenId>,
    /// Token ID to character mapping
    id_to_char: HashMap<TokenId, char>,
    /// Vocabulary size
    vocab_size: u32,
}

impl SimpleTokenizer {
    /// Create a new simple tokenizer with ASCII vocabulary
    pub fn new() -> Self {
        let mut char_to_id = HashMap::new();
        let mut id_to_char = HashMap::new();
        
        // Reserve special tokens 0-3
        let mut next_id: TokenId = 4;
        
        // Add printable ASCII characters (32-126)
        for c in (32u8..=126).map(|b| b as char) {
            char_to_id.insert(c, next_id);
            id_to_char.insert(next_id, c);
            next_id += 1;
        }
        
        // Add common whitespace
        for c in ['\n', '\r', '\t'] {
            if !char_to_id.contains_key(&c) {
                char_to_id.insert(c, next_id);
                id_to_char.insert(next_id, c);
                next_id += 1;
            }
        }
        
        Self {
            char_to_id,
            id_to_char,
            vocab_size: next_id,
        }
    }
    
    /// Encode a single character
    fn encode_char(&self, c: char) -> TokenId {
        *self.char_to_id.get(&c).unwrap_or(&UNK_TOKEN_ID)
    }
    
    /// Decode a single token
    fn decode_token(&self, token: TokenId) -> Option<char> {
        self.id_to_char.get(&token).copied()
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
        tokens.push(BOS_TOKEN_ID);
        
        // Encode each character
        for c in text.chars() {
            tokens.push(self.encode_char(c));
        }
        
        // Add EOS token
        tokens.push(EOS_TOKEN_ID);
        
        tokens
    }
    
    fn decode(&self, tokens: &[TokenId]) -> String {
        let mut result = String::with_capacity(tokens.len());
        
        for &token in tokens {
            // Skip special tokens
            if token == BOS_TOKEN_ID || token == EOS_TOKEN_ID || token == PAD_TOKEN_ID {
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
        BOS_TOKEN_ID
    }
    
    fn eos_token_id(&self) -> TokenId {
        EOS_TOKEN_ID
    }
    
    fn pad_token_id(&self) -> TokenId {
        PAD_TOKEN_ID
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
        BOS_TOKEN_ID
    }
    
    fn eos_token_id(&self) -> TokenId {
        EOS_TOKEN_ID
    }
    
    fn pad_token_id(&self) -> TokenId {
        PAD_TOKEN_ID
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tokenizer_encode() {
        let tokenizer = SimpleTokenizer::new();
        
        let tokens = tokenizer.encode("Hi");
        
        // Should have BOS + 'H' + 'i' + EOS
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0], BOS_TOKEN_ID);
        assert_eq!(tokens[tokens.len() - 1], EOS_TOKEN_ID);
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
