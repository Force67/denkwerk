use futures_core::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::pin::Pin;

use crate::functions::{FunctionRegistry, Tool, ToolCall, ToolChoice};

pub type CompletionStream = Pin<Box<dyn Stream<Item = Result<StreamEvent, crate::LLMError>> + Send>>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

impl EmbeddingRequest {
    pub fn new(model: impl Into<String>, input: Vec<String>) -> Self {
        Self {
            model: model.into(),
            input,
            user: None,
        }
    }

    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub data: Vec<Embedding>,
    pub model: String,
    pub usage: Option<EmbeddingUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ProviderCapabilities {
    pub supports_streaming: bool,
    pub supports_reasoning_stream: bool,
    pub supports_image_uploads: bool,
    pub supports_embeddings: bool,
}

impl ProviderCapabilities {
    pub const fn new(
        supports_streaming: bool,
        supports_reasoning_stream: bool,
        supports_image_uploads: bool,
        supports_embeddings: bool,
    ) -> Self {
        Self {
            supports_streaming,
            supports_reasoning_stream,
            supports_image_uploads,
            supports_embeddings,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelPricing {
    pub prompt_per_token: Option<f64>,
    pub completion_per_token: Option<f64>,
    pub image_per_token: Option<f64>,
    pub request_per_call: Option<f64>,
    pub web_search_per_call: Option<f64>,
    pub internal_reasoning_per_token: Option<f64>,
    pub image_output_per_token: Option<f64>,
    pub discount: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelCapabilities {
    pub supports_reasoning: bool,
    pub supports_function_calling: bool,
    pub supports_tools: bool,
    pub supports_tool_choice: bool,
    pub supports_vision: bool,
    pub supports_streaming: bool,
    pub supports_json_schema: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReasoningConfig {
    pub start_token: Option<String>,
    pub end_token: Option<String>,
    pub system_prompt: Option<String>,
    pub return_mechanism: Option<String>,
    pub is_mandatory_reasoning: Option<bool>,
    pub should_send_reasoning_text_in_text_content: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub provider: String,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
    pub context_length: Option<u32>,
    pub max_completion_tokens: Option<u32>,
    pub input_modalities: Vec<String>,
    pub output_modalities: Vec<String>,
    pub pricing: ModelPricing,
    pub capabilities: ModelCapabilities,
    pub reasoning_config: Option<ReasoningConfig>,
}
