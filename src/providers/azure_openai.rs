use std::{env, time::Duration};

use async_stream::try_stream;
use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::{Client, RequestBuilder};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    error::LLMError,
    providers::LLMProvider,
    functions::{FunctionCall, Tool, ToolCall, ToolChoice},
    types::{
        ChatMessage, CompletionRequest, CompletionResponse, CompletionStream, MessageRole,
        ProviderCapabilities, ReasoningTrace, StreamEvent, TokenUsage,
    },
};

const DEFAULT_API_VERSION: &str = "2024-08-01-preview";

#[derive(Debug, Clone)]
pub struct AzureOpenAIConfig {
    pub api_key: String,
    pub endpoint: String,
    pub api_version: String,
    pub request_timeout: Duration,
}

impl AzureOpenAIConfig {
    pub fn new(api_key: impl Into<String>, endpoint: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            endpoint: endpoint.into(),
            api_version: DEFAULT_API_VERSION.to_string(),
            request_timeout: Duration::from_secs(30),
        }
    }

    pub fn with_api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = version.into();
        self
    }

    pub fn with_timeout(mut self, request_timeout: Duration) -> Self {
        self.request_timeout = request_timeout;
        self
    }
}

#[derive(Debug, Clone)]
pub struct AzureOpenAI {
    client: Client,
    config: AzureOpenAIConfig,
}

impl AzureOpenAI {
    pub fn new(api_key: impl Into<String>, endpoint: impl Into<String>) -> Result<Self, LLMError> {
        Self::from_config(AzureOpenAIConfig::new(api_key, endpoint))
    }

    pub fn from_env() -> Result<Self, LLMError> {
        let api_key = env::var("AZURE_OPENAI_KEY")
            .map_err(|_| LLMError::MissingApiKey("AZURE_OPENAI_KEY"))?;
        let endpoint = env::var("AZURE_OPENAI_ENDPOINT")
            .map_err(|_| LLMError::MissingApiKey("AZURE_OPENAI_ENDPOINT"))?;

        let mut config = AzureOpenAIConfig::new(api_key, endpoint);
        if let Ok(version) = env::var("AZURE_OPENAI_API_VERSION") {
            config.api_version = version;
        }
        if let Ok(timeout_ms) = env::var("AZURE_OPENAI_REQUEST_TIMEOUT_MS") {
            if let Ok(ms) = timeout_ms.parse::<u64>() {
                config.request_timeout = Duration::from_millis(ms);
            }
        }

        Self::from_config(config)
    }

    pub fn from_config(config: AzureOpenAIConfig) -> Result<Self, LLMError> {
        let client = Client::builder()
            .timeout(config.request_timeout)
            .build()?;

        Ok(Self { client, config })
    }

    fn endpoint(&self, deployment: &str) -> String {
        format!(
            "{}/openai/deployments/{}/chat/completions?api-version={}",
            self.config.endpoint.trim_end_matches('/'),
            deployment,
            self.config.api_version
        )
    }

    fn with_default_headers(&self, builder: RequestBuilder) -> RequestBuilder {
        builder.header("api-key", &self.config.api_key)
    }
}

#[derive(Debug, Serialize)]
struct AzureChatRequestBody {
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
struct AzureChatResponse {
    choices: Vec<AzureResponseChoice>,
    usage: Option<TokenUsage>,
}

#[derive(Debug, Deserialize)]
struct AzureResponseChoice {
    message: ChatMessage,
}

#[derive(Debug, Deserialize)]
struct AzureChatChunk {
    #[serde(default)]
    choices: Vec<AzureChatChunkChoice>,
    #[serde(default)]
    usage: Option<TokenUsage>,
}

#[derive(Debug, Deserialize)]
struct AzureChatChunkChoice {
    #[serde(default)]
    delta: Option<AzureChunkDelta>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct AzureChunkDelta {
    #[serde(default)]
    content: Vec<AzureChunkContent>,
    #[serde(default)]
    reasoning: Vec<AzureChunkContent>,
    #[serde(default)]
    tool_calls: Vec<AzureChunkToolCall>,
}

#[derive(Debug, Clone, Deserialize)]
struct AzureChunkContent {
    #[serde(rename = "type")]
    #[serde(default)]
    _kind: Option<String>,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct AzureChunkToolCall {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<AzureChunkToolCallFunction>,
}

#[derive(Debug, Clone, Deserialize)]
struct AzureChunkToolCallFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Default, Clone)]
struct ToolCallAccumulator {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

impl ToolCallAccumulator {
    fn update(&mut self, delta: &AzureChunkToolCall) {
        if let Some(id) = &delta.id {
            if !id.is_empty() {
                self.id = Some(id.clone());
            }
        }

        if let Some(function) = &delta.function {
            if let Some(name) = &function.name {
                if !name.is_empty() {
                    self.name = Some(name.clone());
                }
            }

            if let Some(arguments) = &function.arguments {
                self.arguments.push_str(arguments);
            }
        }
    }

    fn build(self) -> Result<ToolCall, LLMError> {
        let name = self
            .name
            .ok_or_else(|| LLMError::InvalidResponse("tool call missing function name"))?;
        let arguments = if self.arguments.trim().is_empty() {
            Value::Object(serde_json::Map::new())
        } else {
            serde_json::from_str(&self.arguments)
                .map_err(|_| LLMError::InvalidResponse("tool call arguments contained invalid json"))?
        };

        Ok(ToolCall {
            id: self.id,
            kind: crate::functions::ToolCallType::Function,
            function: FunctionCall {
                name,
                arguments,
                raw_arguments: Some(self.arguments),
            },
        })
    }
}

#[derive(Debug, Deserialize)]
struct AzureErrorEnvelope {
    error: AzureError,
}

#[derive(Debug, Deserialize)]
struct AzureError {
    message: String,
}

#[async_trait]
impl LLMProvider for AzureOpenAI {
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

        let body = AzureChatRequestBody {
            model: model.clone(),
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
            .with_default_headers(self.client.post(self.endpoint(&model)))
            .json(&body);

        let response = builder.send().await?;
        let status = response.status();

        if !status.is_success() {
            let text = response.text().await?;
            if let Ok(error) = serde_json::from_str::<AzureErrorEnvelope>(&text) {
                return Err(LLMError::Provider(error.error.message));
            }

            return Err(LLMError::Provider(format!("unexpected status {status}: {text}")));
        }

        let parsed: AzureChatResponse = response.json().await?;
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

        let body = AzureChatRequestBody {
            model: model.clone(),
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
            .with_default_headers(self.client.post(self.endpoint(&model)))
            .header("Accept", "text/event-stream")
            .header("Cache-Control", "no-cache")
            .json(&body);

        let response = builder.send().await?;
        let status = response.status();

        if !status.is_success() {
            let text = response.text().await?;
            if let Ok(error) = serde_json::from_str::<AzureErrorEnvelope>(&text) {
                return Err(LLMError::Provider(error.error.message));
            }

            return Err(LLMError::Provider(format!("unexpected status {status}: {text}")));
        }

        let stream = try_stream! {
            let mut buffer = Vec::new();
            let mut message = String::new();
            let mut reasoning_buffer = String::new();
            let mut usage: Option<TokenUsage> = None;
            let mut finish_reason: Option<String> = None;
            let mut tool_call_accumulators: Vec<ToolCallAccumulator> = Vec::new();
            let mut body_stream = response.bytes_stream();
            let mut finished = false;

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

                        let resolved_tool_calls: Vec<ToolCall> = tool_call_accumulators
                            .clone()
                            .into_iter()
                            .map(|builder| builder.build())
                            .collect::<Result<_, _>>()?;

                        let content = if message.is_empty() {
                            None
                        } else {
                            Some(message.clone())
                        };

                        let completion_message = ChatMessage {
                            role: MessageRole::Assistant,
                            content,
                            name: None,
                            tool_call_id: None,
                            tool_calls: resolved_tool_calls.clone(),
                        };

                        let completion = CompletionResponse {
                            message: completion_message,
                            usage: usage.clone(),
                            reasoning,
                        };

                        yield StreamEvent::Completed(completion);
                        finished = true;
                        break;
                    }

                    let chunk: AzureChatChunk = serde_json::from_str(payload)?;

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

                            for tool_delta in delta.tool_calls {
                                let index = tool_delta.index;
                                if tool_call_accumulators.len() <= index {
                                    tool_call_accumulators
                                        .resize_with(index + 1, ToolCallAccumulator::default);
                                }

                                if let Some(function) = &tool_delta.function {
                                    if let Some(arguments) = &function.arguments {
                                        if !arguments.is_empty() {
                                            yield StreamEvent::ToolCallDelta {
                                                index,
                                                arguments: arguments.clone(),
                                            };
                                        }
                                    }
                                }

                                if let Some(accumulator) = tool_call_accumulators.get_mut(index) {
                                    accumulator.update(&tool_delta);
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

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new(true, true, false)
    }

    fn name(&self) -> &'static str {
        "azure-openai"
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
