use std::time::Duration;

use async_stream::try_stream;
use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::{Client, RequestBuilder};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    error::LLMError,
    functions::{Tool, ToolChoice},
    providers::LLMProvider,
    types::{
        ChatMessage, CompletionRequest, CompletionResponse, CompletionStream, EmbeddingRequest,
        EmbeddingResponse, ModelCapabilities, ModelInfo, ProviderCapabilities, ReasoningTrace,
        StreamEvent, TokenUsage,
    },
};

const DEFAULT_BASE_URL: &str = "http://localhost:11434";

#[derive(Debug, Clone)]
pub struct OllamaConfig {
    pub base_url: String,
    pub request_timeout: Duration,
    pub keep_alive: String,
    pub num_ctx: Option<u32>,
}

impl OllamaConfig {
    pub fn new() -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            request_timeout: Duration::from_secs(120),
            keep_alive: "30m".to_string(),
            num_ctx: None,
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_keep_alive(mut self, duration: impl Into<String>) -> Self {
        self.keep_alive = duration.into();
        self
    }

    pub fn with_num_ctx(mut self, ctx: u32) -> Self {
        self.num_ctx = Some(ctx);
        self
    }
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct Ollama {
    client: Client,
    config: OllamaConfig,
}

impl Ollama {
    pub fn new() -> Result<Self, LLMError> {
        Self::from_config(OllamaConfig::new())
    }

    pub fn from_config(config: OllamaConfig) -> Result<Self, LLMError> {
        let client = Client::builder()
            .timeout(config.request_timeout)
            .build()?;
        Ok(Self { client, config })
    }

    pub fn from_env() -> Result<Self, LLMError> {
        let base_url = std::env::var("OLLAMA_BASE_URL")
            .unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
        let keep_alive =
            std::env::var("OLLAMA_KEEP_ALIVE").unwrap_or_else(|_| "30m".to_string());
        let config = OllamaConfig {
            base_url,
            request_timeout: Duration::from_secs(120),
            keep_alive,
            num_ctx: None,
        };
        Self::from_config(config)
    }

    fn endpoint(&self, path: &str) -> String {
        format!(
            "{}/{}",
            self.config.base_url.trim_end_matches('/'),
            path.trim_start_matches('/')
        )
    }

    fn prepare_request(&self, builder: RequestBuilder) -> RequestBuilder {
        builder.header("Content-Type", "application/json")
    }
}

#[derive(Debug, Serialize)]
struct OllamaRequestBody {
    model: String,
    messages: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<String>,
}

fn chat_message_to_json(msg: &ChatMessage) -> Value {
    if msg.images.is_empty() {
        return serde_json::to_value(msg).unwrap_or_default();
    }

    let mut content_parts: Vec<Value> = Vec::with_capacity(1 + msg.images.len());
    if let Some(text) = &msg.content {
        content_parts.push(serde_json::json!({
            "type": "text",
            "text": text,
        }));
    }
    for image_url in &msg.images {
        content_parts.push(serde_json::json!({
            "type": "image_url",
            "image_url": { "url": image_url },
        }));
    }

    let mut obj = serde_json::json!({
        "role": msg.role,
        "content": content_parts,
    });

    if let Some(name) = &msg.name {
        obj["name"] = serde_json::json!(name);
    }
    if let Some(tool_call_id) = &msg.tool_call_id {
        obj["tool_call_id"] = serde_json::json!(tool_call_id);
    }
    if !msg.tool_calls.is_empty() {
        obj["tool_calls"] = serde_json::to_value(&msg.tool_calls).unwrap_or_default();
    }

    obj
}

#[derive(Debug, Deserialize)]
struct OllamaChoice {
    message: OllamaMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OllamaMessage {
    role: String,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    tool_calls: Vec<crate::functions::ToolCall>,
}

#[derive(Debug, Deserialize)]
struct OllamaResponse {
    choices: Vec<OllamaChoice>,
    #[serde(default)]
    usage: Option<OllamaUsage>,
}

#[derive(Debug, Deserialize)]
struct OllamaUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
    #[serde(default)]
    prompt_tokens_details: Option<OllamaPromptTokensDetails>,
}

#[derive(Debug, Deserialize, Default)]
struct OllamaPromptTokensDetails {
    #[serde(default)]
    cached_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct OllamaErrorBody {
    error: Option<OllamaErrorDetail>,
}

#[derive(Debug, Deserialize)]
struct OllamaErrorDetail {
    message: String,
}

#[derive(Debug, Serialize)]
struct OllamaEmbeddingRequestBody {
    model: String,
    input: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct OllamaEmbeddingResponse {
    data: Vec<OllamaEmbeddingData>,
    model: String,
    #[serde(default)]
    usage: Option<crate::types::EmbeddingUsage>,
}

#[derive(Debug, Deserialize)]
struct OllamaEmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Debug, Deserialize)]
struct OllamaModelsResponse {
    data: Vec<OllamaModel>,
}

#[derive(Debug, Deserialize)]
struct OllamaModel {
    id: String,
    #[serde(default)]
    owned_by: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OllamaStreamChunk {
    #[serde(default)]
    choices: Vec<OllamaStreamChoice>,
    #[serde(default)]
    usage: Option<OllamaUsage>,
}

#[derive(Debug, Deserialize)]
struct OllamaStreamChoice {
    #[serde(default)]
    delta: Option<OllamaStreamDelta>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct OllamaStreamDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<crate::functions::ToolCall>>,
}

fn parse_error_body(text: &str) -> LLMError {
    if let Ok(error_body) = serde_json::from_str::<OllamaErrorBody>(text) {
        if let Some(error) = error_body.error {
            return LLMError::Provider(error.message);
        }
    }
    LLMError::Provider(text.to_string())
}

fn ollama_usage_to_token_usage(usage: Option<OllamaUsage>) -> Option<TokenUsage> {
    usage.map(|u| TokenUsage {
        prompt_tokens: u.prompt_tokens,
        completion_tokens: u.completion_tokens,
        total_tokens: u.total_tokens,
        cached_tokens: u
            .prompt_tokens_details
            .and_then(|d| d.cached_tokens),
    })
}

fn ollama_message_to_chat(msg: OllamaMessage) -> ChatMessage {
    // Keep assistant-visible text clean: reasoning is returned separately via
    // `CompletionResponse.reasoning` and should not be injected into message content.
    let content = msg.content.unwrap_or_default();

    let mut chat_msg = ChatMessage::assistant(content);
    if !msg.tool_calls.is_empty() {
        chat_msg = chat_msg.with_tool_calls(msg.tool_calls);
    }
    chat_msg
}

#[async_trait]
impl LLMProvider for Ollama {
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, LLMError> {
        let CompletionRequest {
            model,
            messages,
            max_tokens,
            temperature,
            top_p,
            response_format,
            tools,
            tool_choice,
        } = request;

        let body = OllamaRequestBody {
            model,
            messages: messages.iter().map(chat_message_to_json).collect(),
            max_tokens,
            temperature,
            top_p,
            response_format,
            tools: if tools.is_empty() {
                None
            } else {
                Some(tools)
            },
            tool_choice,
            stream: None,
            keep_alive: Some(self.config.keep_alive.clone()),
        };

        let builder = self
            .prepare_request(self.client.post(self.endpoint("v1/chat/completions")))
            .json(&body);

        let response = builder.send().await?;
        let status = response.status();

        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(parse_error_body(&text));
        }

        let parsed: OllamaResponse = response.json().await?;
        let choice = parsed
            .choices
            .into_iter()
            .next()
            .ok_or(LLMError::InvalidResponse("response did not contain any choices"))?;

        let reasoning = choice
            .message
            .reasoning
            .as_ref()
            .filter(|r| !r.is_empty())
            .cloned()
            .map(|content| {
                vec![ReasoningTrace {
                    content,
                    finish_reason: choice.finish_reason.clone(),
                }]
            });

        let message = ollama_message_to_chat(choice.message);
        let usage = ollama_usage_to_token_usage(parsed.usage);

        Ok(CompletionResponse {
            message,
            usage,
            reasoning,
        })
    }

    async fn stream_completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionStream, LLMError> {
        let CompletionRequest {
            model,
            messages,
            max_tokens,
            temperature,
            top_p,
            response_format,
            tools,
            tool_choice,
        } = request;

        let body = OllamaRequestBody {
            model,
            messages: messages.iter().map(chat_message_to_json).collect(),
            max_tokens,
            temperature,
            top_p,
            response_format,
            tools: if tools.is_empty() {
                None
            } else {
                Some(tools)
            },
            tool_choice,
            stream: Some(true),
            keep_alive: Some(self.config.keep_alive.clone()),
        };

        let builder = self
            .prepare_request(self.client.post(self.endpoint("v1/chat/completions")))
            .header("Accept", "text/event-stream")
            .header("Cache-Control", "no-cache")
            .json(&body);

        let response = builder.send().await?;
        let status = response.status();

        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(parse_error_body(&text));
        }

        let stream = try_stream! {
            let mut buffer = Vec::new();
            let mut message = String::new();
            let mut reasoning_buffer = String::new();
            let mut usage: Option<TokenUsage> = None;
            let mut finish_reason: Option<String> = None;
            let mut finished = false;
            let mut body_stream = response.bytes_stream();

            while let Some(chunk) = body_stream.next().await {
                let chunk = chunk?;
                buffer.extend_from_slice(&chunk);

                while let Some(event) = extract_sse_event(&mut buffer) {
                    if event.is_empty() {
                        continue;
                    }

                    let payload = extract_data_payload(&event)?;
                    let payload = payload.trim();

                    if payload.is_empty() {
                        continue;
                    }

                    if payload == "[DONE]" {
                        let reasoning = if reasoning_buffer.is_empty() {
                            None
                        } else {
                            Some(vec![ReasoningTrace {
                                content: reasoning_buffer.clone(),
                                finish_reason: finish_reason.clone(),
                            }])
                        };

                        let completion = CompletionResponse {
                            message: ChatMessage::assistant(message.clone()),
                            usage: usage.clone(),
                            reasoning,
                        };

                        yield StreamEvent::Completed(completion);
                        finished = true;
                        break;
                    }

                    let chunk: OllamaStreamChunk = match serde_json::from_str(payload) {
                        Ok(c) => c,
                        Err(_) => continue,
                    };

                    if let Some(chunk_usage) = chunk.usage {
                        usage = ollama_usage_to_token_usage(Some(chunk_usage));
                    }

                    for choice in chunk.choices {
                        if let Some(delta) = choice.delta {
                            if let Some(text) = delta.content {
                                if !text.is_empty() {
                                    message.push_str(&text);
                                    yield StreamEvent::MessageDelta(text);
                                }
                            }

                            if let Some(reasoning) = delta.reasoning {
                                if !reasoning.is_empty() {
                                    reasoning_buffer.push_str(&reasoning);
                                    yield StreamEvent::ReasoningDelta(reasoning);
                                }
                            }

                            if let Some(tool_calls) = delta.tool_calls {
                                for (idx, tc) in tool_calls.into_iter().enumerate() {
                                    let args = tc
                                        .function
                                        .raw_arguments
                                        .clone()
                                        .unwrap_or_else(|| tc.function.arguments.to_string());
                                    yield StreamEvent::ToolCallDelta {
                                        index: idx,
                                        arguments: args,
                                    };
                                }
                            }
                        }

                        if let Some(reason) = choice.finish_reason {
                            finish_reason = Some(reason);
                        }
                    }
                }

                if finished {
                    break;
                }
            }
        };

        Ok(Box::pin(stream))
    }

    async fn create_embeddings(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LLMError> {
        let body = OllamaEmbeddingRequestBody {
            model: request.model,
            input: request.input,
        };

        let builder = self
            .prepare_request(self.client.post(self.endpoint("v1/embeddings")))
            .json(&body);

        let response = builder.send().await?;
        let status = response.status();

        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(parse_error_body(&text));
        }

        let parsed: OllamaEmbeddingResponse = response.json().await?;

        Ok(EmbeddingResponse {
            data: parsed
                .data
                .into_iter()
                .map(|e| crate::types::Embedding {
                    object: "embedding".to_string(),
                    embedding: e.embedding,
                    index: e.index,
                })
                .collect(),
            model: parsed.model,
            usage: parsed.usage,
        })
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new(true, true, false, true)
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, LLMError> {
        let response = self
            .prepare_request(self.client.get(self.endpoint("v1/models")))
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(parse_error_body(&text));
        }

        let parsed: OllamaModelsResponse = response.json().await?;
        let models = parsed
            .data
            .into_iter()
            .map(|m| ModelInfo {
                id: m.id.clone(),
                name: m.id,
                provider: "ollama".to_string(),
                created_at: None,
                updated_at: None,
                context_length: self.config.num_ctx,
                max_completion_tokens: None,
                input_modalities: vec!["text".to_string()],
                output_modalities: vec!["text".to_string()],
                pricing: crate::types::ModelPricing::default(),
                capabilities: ModelCapabilities {
                    supports_reasoning: false,
                    supports_function_calling: true,
                    supports_tools: true,
                    supports_tool_choice: true,
                    supports_vision: false,
                    supports_streaming: true,
                    supports_json_schema: false,
                },
                reasoning_config: None,
            })
            .collect();

        Ok(models)
    }

    fn name(&self) -> &'static str {
        "ollama"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_builder_sets_fields() {
        let config = OllamaConfig::new()
            .with_base_url("http://192.168.1.1:11434")
            .with_keep_alive("1h")
            .with_num_ctx(32768);

        assert_eq!(config.base_url, "http://192.168.1.1:11434");
        assert_eq!(config.keep_alive, "1h");
        assert_eq!(config.num_ctx, Some(32768));
    }

    #[test]
    fn config_default_has_sane_values() {
        let config = OllamaConfig::default();
        assert_eq!(config.base_url, "http://localhost:11434");
        assert_eq!(config.keep_alive, "30m");
        assert!(config.num_ctx.is_none());
    }

    #[test]
    fn chat_message_to_json_text_only() {
        let msg = ChatMessage::user("Hello");
        let json = chat_message_to_json(&msg);
        assert_eq!(json["content"].as_str(), Some("Hello"));
        assert_eq!(json["role"].as_str(), Some("user"));
    }

    #[test]
    fn chat_message_to_json_multimodal() {
        let msg = ChatMessage::user_with_images(
            "What is this?",
            vec!["data:image/png;base64,AAAA".to_string()],
        );
        let json = chat_message_to_json(&msg);
        let content = json["content"].as_array().expect("content should be array");
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"].as_str(), Some("text"));
        assert_eq!(content[1]["type"].as_str(), Some("image_url"));
    }

    #[test]
    fn ollama_usage_maps_cached_tokens() {
        let usage = Some(OllamaUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            prompt_tokens_details: Some(OllamaPromptTokensDetails {
                cached_tokens: Some(80),
            }),
        });
        let token_usage = ollama_usage_to_token_usage(usage).unwrap();
        assert_eq!(token_usage.prompt_tokens, 100);
        assert_eq!(token_usage.completion_tokens, 50);
        assert_eq!(token_usage.cached_tokens, Some(80));
    }

    #[test]
    fn ollama_usage_handles_missing_details() {
        let usage = Some(OllamaUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            prompt_tokens_details: None,
        });
        let token_usage = ollama_usage_to_token_usage(usage).unwrap();
        assert_eq!(token_usage.cached_tokens, None);
    }

    #[test]
    fn parse_error_body_structured() {
        let err = parse_error_body(r#"{"error":{"message":"model not found"}}"#);
        assert!(matches!(err, LLMError::Provider(msg) if msg == "model not found"));
    }

    #[test]
    fn parse_error_body_fallback() {
        let err = parse_error_body("something went wrong");
        assert!(matches!(err, LLMError::Provider(msg) if msg == "something went wrong"));
    }

    #[test]
    fn ollama_message_ignores_reasoning_in_content() {
        let msg = OllamaMessage {
            role: "assistant".to_string(),
            content: Some("The answer is 42".to_string()),
            reasoning: Some("Let me think...".to_string()),
            tool_calls: vec![],
        };
        let chat = ollama_message_to_chat(msg);
        let content = chat.content.unwrap();
        assert_eq!(content, "The answer is 42");
    }

    #[test]
    fn ollama_message_content_only() {
        let msg = OllamaMessage {
            role: "assistant".to_string(),
            content: Some("Just text".to_string()),
            reasoning: None,
            tool_calls: vec![],
        };
        let chat = ollama_message_to_chat(msg);
        assert_eq!(chat.content.as_deref(), Some("Just text"));
    }

    #[test]
    fn endpoint_formatting() {
        let ollama = Ollama::from_config(OllamaConfig::new()).unwrap();
        assert_eq!(ollama.endpoint("v1/chat/completions"), "http://localhost:11434/v1/chat/completions");
        assert_eq!(ollama.endpoint("/v1/models"), "http://localhost:11434/v1/models");
    }
}

fn extract_sse_event(buffer: &mut Vec<u8>) -> Option<Vec<u8>> {
    if let Some(pos) = buffer.windows(2).position(|w| w == b"\n\n") {
        let event = buffer[..pos].to_vec();
        buffer.drain(..pos + 2);
        return Some(event);
    }

    if let Some(pos) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
        let event = buffer[..pos].to_vec();
        buffer.drain(..pos + 4);
        return Some(event);
    }

    None
}

fn extract_data_payload(event: &[u8]) -> Result<String, LLMError> {
    let text = String::from_utf8(event.to_vec())
        .map_err(|_| LLMError::InvalidResponse("stream event contained invalid utf-8"))?;

    let mut payload = String::new();
    for line in text.lines() {
        if let Some(value) = line.strip_prefix("data:") {
            if !payload.is_empty() {
                payload.push('\n');
            }
            payload.push_str(value.trim_start());
        }
    }

    Ok(payload)
}
