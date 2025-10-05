use futures_core::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::pin::Pin;

use crate::functions::{FunctionRegistry, Tool, ToolCall, ToolChoice};

pub type CompletionStream = Pin<Box<dyn Stream<Item = Result<StreamEvent, crate::LLMError>> + Send>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: MessageRole,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
}

impl ChatMessage {
    pub fn new(role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: Some(content.into()),
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new(MessageRole::System, content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new(MessageRole::User, content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(MessageRole::Assistant, content)
    }

    pub fn tool(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Tool,
            content: Some(content.into()),
            name: None,
            tool_call_id: Some(id.into()),
            tool_calls: Vec::new(),
        }
    }

    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        self.tool_calls = tool_calls;
        self
    }

    pub fn text(&self) -> Option<&str> {
        self.content.as_deref()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
}

impl CompletionRequest {
    pub fn new(model: impl Into<String>, messages: Vec<ChatMessage>) -> Self {
        Self {
            model: model.into(),
            messages,
            max_tokens: None,
            temperature: None,
            top_p: None,
            response_format: None,
            tools: Vec::new(),
            tool_choice: None,
        }
    }

    pub fn with_max_tokens(mut self, value: u32) -> Self {
        self.max_tokens = Some(value);
        self
    }

    pub fn with_temperature(mut self, value: f32) -> Self {
        self.temperature = Some(value);
        self
    }

    pub fn with_top_p(mut self, value: f32) -> Self {
        self.top_p = Some(value);
        self
    }

    pub fn with_response_format(mut self, value: Value) -> Self {
        self.response_format = Some(value);
        self
    }

    pub fn with_tool(mut self, tool: Tool) -> Self {
        self.tools.push(tool);
        self
    }

    pub fn with_tools<I>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = Tool>,
    {
        self.tools.extend(tools);
        self
    }

    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    pub fn with_function_registry(mut self, registry: &FunctionRegistry) -> Self {
        self.tools.extend(registry.tools());
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub message: ChatMessage,
    pub usage: Option<TokenUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Vec<ReasoningTrace>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningTrace {
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    MessageDelta(String),
    ReasoningDelta(String),
    ToolCallDelta { index: usize, arguments: String },
    Completed(CompletionResponse),
}

#[derive(Debug, Clone)]
pub struct ImageUploadRequest {
    pub purpose: String,
    pub filename: String,
    pub bytes: Vec<u8>,
    pub mime_type: String,
}

impl ImageUploadRequest {
    pub fn new(
        purpose: impl Into<String>,
        filename: impl Into<String>,
        mime_type: impl Into<String>,
        bytes: Vec<u8>,
    ) -> Self {
        Self {
            purpose: purpose.into(),
            filename: filename.into(),
            bytes,
            mime_type: mime_type.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUploadResponse {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<u64>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ProviderCapabilities {
    pub supports_streaming: bool,
    pub supports_reasoning_stream: bool,
    pub supports_image_uploads: bool,
}

impl ProviderCapabilities {
    pub const fn new(
        supports_streaming: bool,
        supports_reasoning_stream: bool,
        supports_image_uploads: bool,
    ) -> Self {
        Self {
            supports_streaming,
            supports_reasoning_stream,
            supports_image_uploads,
        }
    }
}
