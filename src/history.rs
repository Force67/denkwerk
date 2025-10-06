use std::sync::Arc;

use crate::types::{ChatMessage, CompletionRequest, MessageRole};
use crate::{LLMError, LLMProvider};

#[derive(Debug, Clone, Default)]
pub struct ChatHistory {
    messages: Vec<ChatMessage>,
}

impl ChatHistory {
    pub fn new() -> Self {
        Self { messages: Vec::new() }
    }

    pub fn with_messages(messages: Vec<ChatMessage>) -> Self {
        Self { messages }
    }

    pub fn push(&mut self, message: ChatMessage) {
        self.messages.push(message);
    }

    pub fn push_user(&mut self, content: impl Into<String>) {
        self.push(ChatMessage::user(content));
    }

    pub fn push_assistant(&mut self, content: impl Into<String>) {
        self.push(ChatMessage::assistant(content));
    }

    pub fn push_system(&mut self, content: impl Into<String>) {
        self.push(ChatMessage::system(content));
    }

    pub fn push_tool(&mut self, id: impl Into<String>, content: impl Into<String>) {
        self.push(ChatMessage::tool(id, content));
    }

    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    pub fn into_messages(self) -> Vec<ChatMessage> {
        self.messages
    }

    pub fn iter(&self) -> std::slice::Iter<'_, ChatMessage> {
        self.messages.iter()
    }

    pub fn len(&self) -> usize {
        self.messages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    pub fn clear(&mut self) {
        self.messages.clear();
    }

    pub fn total_content_length(&self) -> usize {
        self.messages
            .iter()
            .filter_map(|message| message.text())
            .map(|value| value.len())
            .sum()
    }

    pub fn last(&self) -> Option<&ChatMessage> {
        self.messages.last()
    }

    pub fn compress<C: ChatHistoryCompressor>(&mut self, compressor: &mut C) -> bool {
        compressor.compress(self)
    }

    pub fn append(&mut self, other: &mut ChatHistory) {
        self.messages.append(&mut other.messages);
    }
}

pub trait ChatHistoryCompressor {
    fn compress(&mut self, history: &mut ChatHistory) -> bool;
}

pub struct NoopChatHistoryCompressor;

impl ChatHistoryCompressor for NoopChatHistoryCompressor {
    fn compress(&mut self, _history: &mut ChatHistory) -> bool {
        false
    }
}

pub trait ChatHistorySummarizer {
    fn summarize(&mut self, messages: &[ChatMessage]) -> Option<String>;
}

#[derive(Debug, Clone)]
pub struct ConciseSummarizer {
    max_chars: usize,
}

impl ConciseSummarizer {
    pub fn new(max_chars: usize) -> Self {
        Self { max_chars: max_chars.max(1) }
    }
}

impl Default for ConciseSummarizer {
    fn default() -> Self {
        Self::new(512)
    }
}

impl ChatHistorySummarizer for ConciseSummarizer {
    fn summarize(&mut self, messages: &[ChatMessage]) -> Option<String> {
        let mut combined = String::new();

        for message in messages {
            if let Some(text) = message.text() {
                if !combined.is_empty() {
                    combined.push(' ');
                }
                combined.push_str(text.trim());
            }
        }

        if combined.is_empty() {
            return None;
        }

        if combined.len() > self.max_chars {
            combined.truncate(self.max_chars);
            combined.push_str("...");
        }

        Some(combined)
    }
}

pub struct FixedWindowCompressor<S> {
    max_messages: usize,
    retain_messages: usize,
    summary_prefix: String,
    summarizer: S,
}

impl<S> FixedWindowCompressor<S> {
    pub fn new(max_messages: usize, summarizer: S) -> Self {
        Self {
            max_messages: max_messages.max(2),
            retain_messages: 6,
            summary_prefix: "Summary so far: ".to_string(),
            summarizer,
        }
    }

    pub fn with_retain_messages(mut self, retain_messages: usize) -> Self {
        self.retain_messages = retain_messages.max(1);
        self
    }

    pub fn with_summary_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.summary_prefix = prefix.into();
        self
    }
}

impl<S> ChatHistoryCompressor for FixedWindowCompressor<S>
where
    S: ChatHistorySummarizer,
{
    fn compress(&mut self, history: &mut ChatHistory) -> bool {
        if history.len() <= self.max_messages {
            return false;
        }

        let retain = self
            .retain_messages
            .min(self.max_messages.saturating_sub(1))
            .min(history.len());

        let boundary = history.len().saturating_sub(retain);
        if boundary == 0 {
            return false;
        }

        let summary_text = match self.summarizer.summarize(&history.messages[..boundary]) {
            Some(text) if !text.trim().is_empty() => text.trim().to_string(),
            _ => return false,
        };

        let mut summary = ChatMessage::system(format!("{}{}", self.summary_prefix, summary_text));
        summary.name = Some("history-summary".to_string());

        history.messages.drain(..boundary);
        history.messages.insert(0, summary);

        while history.len() > self.max_messages {
            history.messages.remove(1);
        }

        true
    }
}

pub struct LLMHistoryCompressor {
    provider: Arc<dyn LLMProvider>,
    model: String,
    summarizer_instructions: String,
    max_messages: usize,
    retain_messages: usize,
    summary_prefix: String,
}

impl LLMHistoryCompressor {
    pub fn new(provider: Arc<dyn LLMProvider>, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            summarizer_instructions: "Summarize the following conversation succinctly while preserving key facts.".into(),
            max_messages: 12,
            retain_messages: 6,
            summary_prefix: "Conversation summary: ".into(),
        }
    }

    pub fn with_max_messages(mut self, max_messages: usize) -> Self {
        self.max_messages = max_messages.max(2);
        self
    }

    pub fn with_retain_messages(mut self, retain_messages: usize) -> Self {
        self.retain_messages = retain_messages.max(1);
        self
    }

    pub fn with_summary_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.summary_prefix = prefix.into();
        self
    }

    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.summarizer_instructions = instructions.into();
        self
    }

    fn needs_compression(&self, history: &ChatHistory) -> bool {
        history.len() > self.max_messages
    }

    fn window_boundary(&self, history: &ChatHistory) -> usize {
        let retain = self
            .retain_messages
            .min(self.max_messages.saturating_sub(1))
            .min(history.len());
        history.len().saturating_sub(retain)
    }

    fn build_summary_prompt(&self, messages: &[ChatMessage]) -> Vec<ChatMessage> {
        let mut buffer = String::new();

        for message in messages {
            if let Some(text) = message.text() {
                let role = match message.role {
                    MessageRole::System => "System".to_string(),
                    MessageRole::User => "User".to_string(),
                    MessageRole::Assistant => message
                        .name
                        .as_deref()
                        .map(|name| format!("Assistant::{name}"))
                        .unwrap_or_else(|| "Assistant".to_string()),
                    MessageRole::Tool => "Tool".to_string(),
                };

                buffer.push_str(&format!("[{role}] {text}\n"));
            }
        }

        vec![
            ChatMessage::system(self.summarizer_instructions.clone()),
            ChatMessage::user(format!("Conversation so far:\n{}", buffer.trim())),
        ]
    }

    pub async fn compress(&self, history: &mut ChatHistory) -> Result<bool, LLMError> {
        if !self.needs_compression(history) {
            return Ok(false);
        }

        let boundary = self.window_boundary(history);
        if boundary == 0 {
            return Ok(false);
        }

        let prompt = self.build_summary_prompt(&history.messages[..boundary]);
        let request = CompletionRequest::new(self.model.clone(), prompt);
        let response = self.provider.complete(request).await?;
        let summary = response
            .message
            .text()
            .map(|text| text.trim().to_string())
            .filter(|text| !text.is_empty());

        let summary_text = match summary {
            Some(text) => text,
            None => return Ok(false),
        };

        let mut summary_message = ChatMessage::system(format!("{}{}", self.summary_prefix, summary_text));
        summary_message.name = Some("history-summary".to_string());

        history.messages.drain(..boundary);
        history.messages.insert(0, summary_message);

        while history.len() > self.max_messages {
            history.messages.remove(1);
        }

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::LLMProvider;
    use crate::LLMError;
    use async_trait::async_trait;
    use crate::types::MessageRole;
    use std::sync::Mutex;

    #[test]
    fn summarizer_truncates() {
        let mut summarizer = ConciseSummarizer::new(20);
        let history = vec![
            ChatMessage::user("This is a long message that should be truncated."),
            ChatMessage::assistant("Second part of the conversation."),
        ];

        let summary = summarizer.summarize(&history).expect("summary");
        assert!(summary.len() <= 23); // includes trailing ellipsis
    }

    #[test]
    fn compressor_creates_summary() {
        let mut history = ChatHistory::new();
        for index in 0..8 {
            history.push_user(format!("Message {index}"));
            history.push_assistant(format!("Reply {index}"));
        }

        let mut compressor = FixedWindowCompressor::new(6, ConciseSummarizer::new(80));
        let changed = history.compress(&mut compressor);
        assert!(changed);
        assert!(history.len() <= 6);
        assert_eq!(history.messages()[0].role, MessageRole::System);
        assert!(history.messages()[0].text().unwrap_or_default().starts_with("Summary"));
    }

    #[test]
    fn noop_preserves_history() {
        let mut history = ChatHistory::new();
        history.push_user("Hi");
        history.push_assistant("Hello");
        let mut compressor = NoopChatHistoryCompressor;
        assert!(!history.compress(&mut compressor));
        assert_eq!(history.len(), 2);
    }

    struct StubProvider {
        response: Mutex<String>,
    }

    impl StubProvider {
        fn new(summary: impl Into<String>) -> Self {
            Self {
                response: Mutex::new(summary.into()),
            }
        }
    }

    #[async_trait]
    impl LLMProvider for StubProvider {
        async fn complete(&self, _request: CompletionRequest) -> Result<crate::types::CompletionResponse, LLMError> {
            let content = self.response.lock().unwrap().clone();
            Ok(crate::types::CompletionResponse {
                message: ChatMessage::assistant(content),
                usage: None,
                reasoning: None,
            })
        }

        fn name(&self) -> &'static str {
            "stub"
        }
    }

    #[tokio::test]
    async fn llm_compressor_uses_provider() {
        let provider = Arc::new(StubProvider::new("A concise summary."));
        let provider_trait = provider.clone() as Arc<dyn LLMProvider>;
        let compressor = LLMHistoryCompressor::new(provider_trait, "test-model")
            .with_max_messages(4)
            .with_retain_messages(2);

        let mut history = ChatHistory::new();
        for index in 0..5 {
            history.push_user(format!("Message {index}"));
            history.push_assistant(format!("Reply {index}"));
        }

        let changed = compressor.compress(&mut history).await.expect("compress");
        assert!(changed);
        assert!(history.len() <= 4);
        assert_eq!(history.messages()[0].role, MessageRole::System);
        assert!(history.messages()[0]
            .text()
            .unwrap_or_default()
            .contains("A concise summary"));
    }
}
