use std::sync::Arc;
use std::time::{Duration, Instant};

use async_stream::try_stream;
use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::{multipart::Form, Client, RequestBuilder};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::RwLock;

use crate::{
    error::LLMError,
    providers::LLMProvider,
    functions::{Tool, ToolChoice},
    types::{
        ChatMessage, CompletionRequest, CompletionResponse, CompletionStream, ImageUploadRequest,
        ImageUploadResponse, ProviderCapabilities, ReasoningTrace, StreamEvent, TokenUsage,
        EmbeddingRequest, EmbeddingResponse, ModelInfo, ModelPricing, ModelCapabilities,
        ReasoningConfig,
    },
};

const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";
const OPENROUTER_CATALOG_URL: &str = "https://openrouter.ai/api/frontend/models";

#[derive(Debug, Clone)]
pub struct OpenRouterConfig {
    pub api_key: String,
    pub base_url: String,
    pub request_timeout: Duration,
    pub referer: Option<String>,
    pub title: Option<String>,
    pub model_catalog_ttl: Duration,
}

impl OpenRouterConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            request_timeout: Duration::from_secs(30),
            referer: None,
            title: Some("denkwerk".to_string()),
            model_catalog_ttl: Duration::from_secs(600),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OpenRouter {
    client: Client,
    config: OpenRouterConfig,
    model_catalog_cache: Arc<RwLock<Option<ModelCatalogCache>>>,
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

        Ok(Self {
            client,
            config,
            model_catalog_cache: Arc::new(RwLock::new(None)),
        })
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

    async fn fetch_model_catalog(&self) -> Result<Vec<ModelInfo>, LLMError> {
        if let Some(cached) = self.read_cached_models().await {
            return Ok(cached);
        }

        let response = self
            .with_default_headers(self.client.get(OPENROUTER_CATALOG_URL))
            .send()
            .await?;
        let status = response.status();

        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(LLMError::Provider(format!(
                "unexpected status {status}: {text}"
            )));
        }

        let parsed: OpenRouterCatalogResponse = response.json().await?;
        let models = parsed
            .data
            .into_iter()
            .map(|model| model.into_model_info())
            .collect::<Vec<_>>();

        self.write_cached_models(models.clone()).await;
        Ok(models)
    }

    async fn read_cached_models(&self) -> Option<Vec<ModelInfo>> {
        let cache = self.model_catalog_cache.read().await;
        let cached = cache.as_ref()?;
        if cached.fetched_at.elapsed() > self.config.model_catalog_ttl {
            return None;
        }
        Some(cached.models.clone())
    }

    async fn write_cached_models(&self, models: Vec<ModelInfo>) {
        let mut cache = self.model_catalog_cache.write().await;
        *cache = Some(ModelCatalogCache {
            fetched_at: Instant::now(),
            models,
        });
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

#[derive(Debug, Deserialize)]
struct OpenRouterCatalogResponse {
    data: Vec<OpenRouterCatalogModel>,
}

#[derive(Debug, Clone)]
struct ModelCatalogCache {
    fetched_at: Instant,
    models: Vec<ModelInfo>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterCatalogModel {
    slug: String,
    name: String,
    #[serde(default)]
    created_at: Option<String>,
    #[serde(default)]
    updated_at: Option<String>,
    #[serde(default)]
    context_length: Option<u32>,
    #[serde(default)]
    max_completion_tokens: Option<u32>,
    #[serde(default)]
    input_modalities: Vec<String>,
    #[serde(default)]
    output_modalities: Vec<String>,
    #[serde(default)]
    supported_parameters: Vec<String>,
    #[serde(default)]
    supports_tool_parameters: Option<bool>,
    #[serde(default)]
    supports_reasoning: Option<bool>,
    #[serde(default)]
    pricing: Option<OpenRouterPricing>,
    #[serde(default)]
    reasoning_config: Option<OpenRouterReasoningConfig>,
    #[serde(default)]
    features: Option<OpenRouterModelFeatures>,
}

#[derive(Debug, Deserialize, Default)]
struct OpenRouterPricing {
    #[serde(default)]
    prompt: Option<PriceValue>,
    #[serde(default)]
    completion: Option<PriceValue>,
    #[serde(default)]
    image: Option<PriceValue>,
    #[serde(default)]
    request: Option<PriceValue>,
    #[serde(default)]
    web_search: Option<PriceValue>,
    #[serde(default)]
    internal_reasoning: Option<PriceValue>,
    #[serde(default)]
    image_output: Option<PriceValue>,
    #[serde(default)]
    discount: Option<PriceValue>,
}

#[derive(Debug, Deserialize, Default, Clone)]
struct OpenRouterReasoningConfig {
    #[serde(default)]
    start_token: Option<String>,
    #[serde(default)]
    end_token: Option<String>,
    #[serde(default)]
    system_prompt: Option<String>,
}

#[derive(Debug, Deserialize, Default, Clone)]
struct OpenRouterModelFeatures {
    #[serde(default)]
    reasoning_return_mechanism: Option<String>,
    #[serde(default)]
    is_mandatory_reasoning: Option<bool>,
    #[serde(default)]
    should_send_reasoning_text_in_text_content: Option<bool>,
    #[serde(default)]
    supports_tool_choice: Option<OpenRouterSupportsToolChoice>,
    #[serde(default)]
    reasoning_config: Option<OpenRouterReasoningConfig>,
}

#[derive(Debug, Deserialize, Default, Clone)]
struct OpenRouterSupportsToolChoice {
    #[serde(default)]
    #[allow(dead_code)]
    type_function: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum PriceValue {
    Number(f64),
    String(String),
}

fn parse_price(value: Option<PriceValue>) -> Option<f64> {
    match value {
        Some(PriceValue::Number(value)) => Some(value),
        Some(PriceValue::String(value)) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                None
            } else {
                trimmed.parse::<f64>().ok()
            }
        }
        None => None,
    }
}

fn string_or_none(value: Option<String>) -> Option<String> {
    value.and_then(|v| {
        let trimmed = v.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

impl OpenRouterCatalogModel {
    fn into_model_info(self) -> ModelInfo {
        let supported_parameters = self.supported_parameters;
        let supports_tools = self.supports_tool_parameters.unwrap_or(false)
            || supported_parameters.iter().any(|item| item == "tools");
        let supports_tool_choice = supported_parameters.iter().any(|item| item == "tool_choice")
            || self
                .features
                .as_ref()
                .and_then(|features| features.supports_tool_choice.as_ref())
                .and_then(|choice| choice.type_function)
                .unwrap_or(false);
        let supports_json_schema = supported_parameters.iter().any(|item| item == "response_format");
        let supports_streaming = supported_parameters.iter().any(|item| item == "stream");
        let supports_reasoning = self.supports_reasoning.unwrap_or(false);
        let supports_vision = self.input_modalities.iter().any(|m| m != "text")
            || self.output_modalities.iter().any(|m| m != "text");
        let base_reasoning_config = self
            .reasoning_config
            .or_else(|| self.features.as_ref().and_then(|f| f.reasoning_config.clone()));
        let has_feature_reasoning = self.features.as_ref().map_or(false, |features| {
            features.reasoning_return_mechanism.is_some()
                || features.is_mandatory_reasoning.is_some()
                || features.should_send_reasoning_text_in_text_content.is_some()
        });
        let reasoning_config = if base_reasoning_config.is_some() || has_feature_reasoning {
            let config = base_reasoning_config.unwrap_or_default();
            Some(ReasoningConfig {
                start_token: string_or_none(config.start_token),
                end_token: string_or_none(config.end_token),
                system_prompt: string_or_none(config.system_prompt),
                return_mechanism: string_or_none(
                    self.features
                        .as_ref()
                        .and_then(|f| f.reasoning_return_mechanism.clone()),
                ),
                is_mandatory_reasoning: self
                    .features
                    .as_ref()
                    .and_then(|f| f.is_mandatory_reasoning),
                should_send_reasoning_text_in_text_content: self
                    .features
                    .as_ref()
                    .and_then(|f| f.should_send_reasoning_text_in_text_content),
            })
        } else {
            None
        };

        let pricing = self.pricing.unwrap_or_default();

        ModelInfo {
            id: self.slug,
            name: self.name,
            provider: "openrouter".to_string(),
            created_at: self.created_at,
            updated_at: self.updated_at,
            context_length: self.context_length,
            max_completion_tokens: self.max_completion_tokens,
            input_modalities: self.input_modalities,
            output_modalities: self.output_modalities,
            pricing: ModelPricing {
                prompt_per_token: parse_price(pricing.prompt),
                completion_per_token: parse_price(pricing.completion),
                image_per_token: parse_price(pricing.image),
                request_per_call: parse_price(pricing.request),
                web_search_per_call: parse_price(pricing.web_search),
                internal_reasoning_per_token: parse_price(pricing.internal_reasoning),
                image_output_per_token: parse_price(pricing.image_output),
                discount: parse_price(pricing.discount),
            },
            capabilities: ModelCapabilities {
                supports_reasoning,
                supports_function_calling: supports_tools,
                supports_tools,
                supports_tool_choice,
                supports_vision,
                supports_streaming,
                supports_json_schema,
            },
            reasoning_config,
        }
    }
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

    async fn model_info(&self, id: &str) -> Result<ModelInfo, LLMError> {
        let models = self.list_models().await?;
        models
            .into_iter()
            .find(|model| model.id == id)
            .ok_or_else(|| LLMError::Provider(format!("model not found: {id}")))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, LLMError> {
        self.fetch_model_catalog().await
    }

    fn name(&self) -> &'static str {
        "openrouter"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_openrouter_catalog_into_model_info() {
        let payload = include_str!("../../tests/fixtures/openrouter_models_sample.json");
        let response: OpenRouterCatalogResponse = serde_json::from_str(payload).unwrap();
        let mut models = response.data.into_iter();

        let first = models.next().unwrap().into_model_info();
        assert_eq!(first.id, "openai/gpt-4o-mini");
        assert_eq!(first.pricing.prompt_per_token, Some(0.00000015));
        assert!(first.capabilities.supports_function_calling);
        assert!(first.capabilities.supports_tool_choice);
        assert!(first.capabilities.supports_reasoning);
        assert!(first.capabilities.supports_vision);
        assert!(first.capabilities.supports_json_schema);
        assert_eq!(
            first.reasoning_config.as_ref().and_then(|r| r.start_token.as_deref()),
            Some("<reasoning>")
        );

        let second = models.next().unwrap().into_model_info();
        assert_eq!(second.id, "anthropic/claude-3-haiku");
        assert_eq!(second.pricing.completion_per_token, Some(0.00000125));
        assert!(!second.capabilities.supports_reasoning);
        assert!(!second.capabilities.supports_vision);
        assert!(second.reasoning_config.is_none());
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
