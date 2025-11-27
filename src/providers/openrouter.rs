use std::time::Duration;

use async_stream::try_stream;
use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::{multipart::Form, Client, RequestBuilder};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    error::LLMError,
    providers::LLMProvider,
    functions::{Tool, ToolChoice},
    types::{
        ChatMessage, CompletionRequest, CompletionResponse, CompletionStream, ImageUploadRequest,
        ImageUploadResponse, ProviderCapabilities, ReasoningTrace, StreamEvent, TokenUsage,
        EmbeddingRequest, EmbeddingResponse,
    },
};

const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";

#[derive(Debug, Clone)]
pub struct OpenRouterConfig {
    pub api_key: String,
    pub base_url: String,
    pub request_timeout: Duration,
    pub referer: Option<String>,
    pub title: Option<String>,
}

impl OpenRouterConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            request_timeout: Duration::from_secs(30),
            referer: None,
            title: Some("denkwerk".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OpenRouter {
    client: Client,
    config: OpenRouterConfig,
}

impl OpenRouter {
    pub fn new(api_key: impl Into<String>) -> Result<Self, LLMError> {
        Self::from_config(OpenRouterConfig::new(api_key))
    }

    pub fn from_env() -> Result<Self, LLMError> {
        let api_key = std::env::var("OPENROUTER_API_KEY")
            .map_err(|_| LLMError::MissingApiKey("OPENROUTER_API_KEY"))?;
        Self::new(api_key)
    }

    pub fn from_config(config: OpenRouterConfig) -> Result<Self, LLMError> {
        let client = Client::builder()
            .timeout(config.request_timeout)
            .build()?;

        Ok(Self { client, config })
    }

    fn endpoint(&self, path: &str) -> String {
        format!(
            "{}/{}",
            self.config.base_url.trim_end_matches('/'),
            path.trim_start_matches('/')
        )
    }

    fn with_default_headers(&self, builder: RequestBuilder) -> RequestBuilder {
        let mut builder = builder.bearer_auth(&self.config.api_key);

        if let Some(ref referer) = self.config.referer {
            builder = builder.header("HTTP-Referer", referer);
        }

        if let Some(ref title) = self.config.title {
            builder = builder.header("X-Title", title);
        }

        builder
    }
}

#[derive(Debug, Serialize)]
struct OpenRouterRequestBody {
    model: String,
    messages: Vec<ChatMessage>,
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
}

#[derive(Debug, Deserialize)]
struct OpenRouterChoice {
    message: ChatMessage,
}

#[derive(Debug, Deserialize)]
struct OpenRouterResponseBody {
    choices: Vec<OpenRouterChoice>,
    usage: Option<TokenUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterErrorBody {
    error: Option<ProviderError>,
}

#[derive(Debug, Deserialize)]
struct ProviderError {
    message: String,
}

#[derive(Debug, Serialize)]
struct OpenRouterEmbeddingRequestBody {
    model: String,
    input: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterEmbeddingResponse {
    data: Vec<OpenRouterEmbedding>,
    model: String,
    usage: Option<crate::types::EmbeddingUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterEmbedding {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Debug, Deserialize)]
struct OpenRouterStreamBody {
    #[serde(default)]
    choices: Vec<OpenRouterStreamChoice>,
    #[serde(default)]
    usage: Option<TokenUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterStreamChoice {
    #[serde(default)]
    delta: Option<OpenRouterStreamDelta>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct OpenRouterStreamDelta {
    #[serde(default)]
    content: Vec<OpenRouterStreamContent>,
    #[serde(default)]
    reasoning: Vec<OpenRouterStreamContent>,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenRouterStreamContent {
    #[serde(rename = "type")]
    #[serde(default)]
    _kind: Option<String>,
    #[serde(default)]
    text: Option<String>,
}

#[async_trait]
impl LLMProvider for OpenRouter {
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

        let body = OpenRouterRequestBody {
            model,
            messages,
            max_tokens,
            temperature,
            top_p,
            response_format,
            tools: if tools.is_empty() { None } else { Some(tools) },
            tool_choice,
            stream: None,
        };

        let builder = self
            .with_default_headers(self.client.post(self.endpoint("chat/completions")))
            .json(&body);

        let response = builder.send().await?;
        let status = response.status();

        if !status.is_success() {
            let text = response.text().await?;
            if let Ok(error_body) = serde_json::from_str::<OpenRouterErrorBody>(&text) {
                if let Some(error) = error_body.error {
                    return Err(LLMError::Provider(error.message));
                }
            }

            return Err(LLMError::Provider(format!(
                "unexpected status {status}: {text}"
            )));
        }

        let parsed: OpenRouterResponseBody = response.json().await?;
        let choice = parsed
            .choices
            .into_iter()
            .next()
            .ok_or(LLMError::InvalidResponse("response did not contain any choices"))?;

        Ok(CompletionResponse {
            message: choice.message,
            usage: parsed.usage,
            reasoning: None,
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

        let body = OpenRouterRequestBody {
            model,
            messages,
            max_tokens,
            temperature,
            top_p,
            response_format,
            tools: if tools.is_empty() { None } else { Some(tools) },
            tool_choice,
            stream: Some(true),
        };

        let builder = self
            .with_default_headers(self.client.post(self.endpoint("chat/completions")))
            .header("Accept", "text/event-stream")
            .header("Cache-Control", "no-cache")
            .json(&body);

        let response = builder.send().await?;
        let status = response.status();

        if !status.is_success() {
            let text = response.text().await?;
            if let Ok(error_body) = serde_json::from_str::<OpenRouterErrorBody>(&text) {
                if let Some(error) = error_body.error {
                    return Err(LLMError::Provider(error.message));
                }
            }

            return Err(LLMError::Provider(format!(
                "unexpected status {status}: {text}"
            )));
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

                    let chunk: OpenRouterStreamBody = serde_json::from_str(payload)?;

                    if let Some(chunk_usage) = chunk.usage {
                        usage = Some(chunk_usage);
                    }

                    for choice in chunk.choices {
                        if let Some(delta) = choice.delta {
                            for block in delta.content {
                                if let Some(text) = block.text {
                                    if text.is_empty() {
                                        continue;
                                    }
                                    message.push_str(&text);
                                    yield StreamEvent::MessageDelta(text);
                                }
                            }

                            for block in delta.reasoning {
                                if let Some(text) = block.text {
                                    if text.is_empty() {
                                        continue;
                                    }
                                    reasoning_buffer.push_str(&text);
                                    yield StreamEvent::ReasoningDelta(text);
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

    async fn upload_image(
        &self,
        request: ImageUploadRequest,
    ) -> Result<ImageUploadResponse, LLMError> {
        let ImageUploadRequest {
            purpose,
            filename,
            bytes,
            mime_type,
        } = request;

        let file_part = reqwest::multipart::Part::bytes(bytes)
            .file_name(filename.clone())
            .mime_str(&mime_type)?;

        let form = Form::new()
            .text("purpose", purpose)
            .part("file", file_part);

        let builder = self
            .with_default_headers(self.client.post(self.endpoint("files")))
            .multipart(form);

        let response = builder.send().await?;
        let status = response.status();

        if !status.is_success() {
            let text = response.text().await?;
            if let Ok(error_body) = serde_json::from_str::<OpenRouterErrorBody>(&text) {
                if let Some(error) = error_body.error {
                    return Err(LLMError::Provider(error.message));
                }
            }

            return Err(LLMError::Provider(format!(
                "unexpected status {status}: {text}"
            )));
        }

        let parsed: serde_json::Value = response.json().await?;
        let id = parsed
            .get("id")
            .and_then(|value| value.as_str())
            .ok_or(LLMError::InvalidResponse("file upload missing id"))?
            .to_string();

        let bytes = parsed
            .get("bytes")
            .and_then(|value| value.as_u64())
            .map(|value| value as usize);
        let created_at = parsed.get("created_at").and_then(|value| value.as_u64());

        Ok(ImageUploadResponse {
            id,
            bytes,
            created_at,
        })
    }

    async fn create_embeddings(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LLMError> {
        let body = OpenRouterEmbeddingRequestBody {
            model: request.model,
            input: request.input,
        };

        let builder = self
            .with_default_headers(self.client.post(self.endpoint("embeddings")))
            .json(&body);

        let response = builder.send().await?;
        let status = response.status();

        if !status.is_success() {
            let text = response.text().await?;
            if let Ok(error_body) = serde_json::from_str::<OpenRouterErrorBody>(&text) {
                if let Some(error) = error_body.error {
                    return Err(LLMError::Provider(error.message));
                }
            }

            return Err(LLMError::Provider(format!(
                "unexpected status {status}: {text}"
            )));
        }

        let parsed: OpenRouterEmbeddingResponse = response.json().await?;

        Ok(EmbeddingResponse {
            data: parsed.data.into_iter().map(|embedding| crate::types::Embedding {
                object: "embedding".to_string(),
                embedding: embedding.embedding,
                index: embedding.index,
            }).collect(),
            model: parsed.model,
            usage: parsed.usage,
        })
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new(true, true, true, true)
    }

    fn name(&self) -> &'static str {
        "openrouter"
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
