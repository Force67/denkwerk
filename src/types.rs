use futures_core::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::pin::Pin;

use crate::functions::{FunctionRegistry, Tool, ToolCall, ToolChoice};

/// Controls the reasoning effort for models that support extended thinking.
///
/// Maps to provider-specific parameters:
/// - **OpenAI / Azure**: `reasoning_effort` field (`"low"`, `"medium"`, `"high"`)
/// - **OpenRouter**: `reasoning` object with `effort` field
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

pub type CompletionStream =
    Pin<Box<dyn Stream<Item = Result<StreamEvent, crate::LLMError>> + Send>>;

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
    #[serde(default, skip_serializing_if = "Vec::is_empty", deserialize_with = "crate::providers::deserialize_null_as_empty_vec")]
    pub tool_calls: Vec<ToolCall>,
    /// Optional image data URLs (e.g. `data:image/jpeg;base64,...`) for multimodal messages.
    /// Skipped during normal serde; the provider serializer handles these specially.
    #[serde(skip)]
    pub images: Vec<String>,
    /// Provider-separated reasoning/thinking trace tied to this message. Populated by
    /// providers that expose thinking as a distinct field (e.g. Ollama native API) and
    /// echoed back on subsequent turns when the provider preserves thinking.
    #[serde(skip)]
    pub thinking: Option<String>,
}

impl ChatMessage {
    pub fn new(role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: Some(content.into()),
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
            images: Vec::new(),
            thinking: None,
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
            images: Vec::new(),
            thinking: None,
        }
    }

    /// Create a user message with attached images (data URLs).
    pub fn user_with_images(content: impl Into<String>, images: Vec<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: Some(content.into()),
            name: None,
            tool_call_id: None,
            tool_calls: Vec::new(),
            images,
            thinking: None,
        }
    }

    /// Attach a reasoning/thinking trace to this message.
    pub fn with_thinking(mut self, thinking: impl Into<String>) -> Self {
        self.thinking = Some(thinking.into());
        self
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
    /// Optional reasoning effort level for models that support extended thinking.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
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
            reasoning_effort: None,
        }
    }

    pub fn with_max_tokens(mut self, value: u32) -> Self {
        self.max_tokens = Some(value);
        self
    }

    /// Clear any previously-set output cap. `None` means "no client-imposed
    /// limit" — useful for reasoning models that need room to think. Each
    /// provider's default then applies: Ollama generates up to context
    /// exhaustion or natural stop; OpenAI picks a sensible bound from the
    /// remaining context window.
    pub fn without_max_tokens(mut self) -> Self {
        self.max_tokens = None;
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

    pub fn with_reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
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
    pub dimensions: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

impl EmbeddingRequest {
    pub fn new(model: impl Into<String>, input: Vec<String>) -> Self {
        Self {
            model: model.into(),
            input,
            dimensions: None,
            user: None,
        }
    }

    pub fn with_dimensions(mut self, dimensions: u32) -> Self {
        self.dimensions = Some(dimensions);
        self
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

#[cfg(test)]
mod tests {
    use super::{CompletionRequest, EmbeddingRequest};

    #[test]
    fn embedding_request_defaults_dimensions_to_none() {
        let request = EmbeddingRequest::new("model", vec!["input".to_string()]);

        assert!(request.dimensions.is_none());
    }

    #[test]
    fn embedding_request_sets_dimensions_via_builder() {
        let request =
            EmbeddingRequest::new("model", vec!["input".to_string()]).with_dimensions(1536);

        assert_eq!(request.dimensions, Some(1536));
    }

    #[test]
    fn completion_request_without_max_tokens_clears_cap() {
        let request = CompletionRequest::new("m", vec![])
            .with_max_tokens(128)
            .without_max_tokens();
        assert!(request.max_tokens.is_none());
    }
}
