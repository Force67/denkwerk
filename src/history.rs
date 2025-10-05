use crate::types::ChatMessage;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MessageRole;

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
}
