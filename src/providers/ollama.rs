//! Ollama provider — native `/api/chat` and `/api/embed` endpoints.
//!
//! Uses Ollama's native API (not the OpenAI-compat shim) so we can:
//! * Control `think` (on/off per request, or mapped from `reasoning_effort`).
//! * Preserve assistant `thinking` across turns for thinking models like
//!   Qwen3.6 — see [`OllamaConfig::with_preserve_thinking`].
//! * Forward native `options` (num_ctx, num_predict, top_k, …) and `keep_alive`.
//!
//! The native streaming protocol is NDJSON (one JSON object per line), not SSE.

use std::time::Duration;

use async_stream::try_stream;
use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::{Client, RequestBuilder};
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::{
    error::LLMError,
    functions::{FunctionCall, Tool, ToolCall, ToolCallType},
    providers::LLMProvider,
    types::{
        ChatMessage, CompletionRequest, CompletionResponse, CompletionStream, EmbeddingRequest,
        EmbeddingResponse, MessageRole, ModelCapabilities, ModelInfo, ProviderCapabilities,
        ReasoningEffort, ReasoningTrace, StreamEvent, TokenUsage,
    },
};

const DEFAULT_BASE_URL: &str = "http://localhost:11434";

/// How the provider decides the value of the native `think` parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkMode {
    /// Enable thinking iff the request carries a `reasoning_effort`.
    Auto,
    /// Always disable thinking.
    Off,
    /// Always enable thinking.
    On,
}

impl Default for ThinkMode {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Debug, Clone)]
pub struct OllamaConfig {
    pub base_url: String,
    pub request_timeout: Duration,
    pub keep_alive: String,
    pub num_ctx: Option<u32>,
    pub think_mode: ThinkMode,
    /// Echo assistant `thinking` back on subsequent turns — the Ollama
    /// equivalent of Qwen3.6's `preserve_thinking=true` chat-template kwarg.
    pub preserve_thinking: bool,
}

impl OllamaConfig {
    pub fn new() -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            request_timeout: Duration::from_secs(120),
            keep_alive: "30m".to_string(),
            num_ctx: None,
            think_mode: ThinkMode::Auto,
            preserve_thinking: false,
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

    pub fn with_think_mode(mut self, mode: ThinkMode) -> Self {
        self.think_mode = mode;
        self
    }

    pub fn with_preserve_thinking(mut self, preserve: bool) -> Self {
        self.preserve_thinking = preserve;
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
        let client = Client::builder().timeout(config.request_timeout).build()?;
        Ok(Self { client, config })
    }

    pub fn from_env() -> Result<Self, LLMError> {
        let base_url =
            std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
        let keep_alive =
            std::env::var("OLLAMA_KEEP_ALIVE").unwrap_or_else(|_| "30m".to_string());
        let config = OllamaConfig {
            base_url,
            keep_alive,
            ..OllamaConfig::new()
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

    fn prepare(&self, builder: RequestBuilder) -> RequestBuilder {
        builder.header("Content-Type", "application/json")
    }

    /// Fetch the model's max context length (tokens) from `/api/show`.
    /// Returns `None` if the server doesn't advertise one for this model.
    pub async fn max_context_length(&self, model: &str) -> Result<Option<u32>, LLMError> {
        let response = self
            .prepare(self.client.post(self.endpoint("api/show")))
            .json(&json!({ "model": model }))
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(parse_error_text(&text));
        }

        let parsed: OllamaShowResponse = response.json().await?;
        if let Some(err) = parsed.error {
            return Err(LLMError::Provider(err));
        }

        Ok(max_context_from_model_info(&parsed.model_info))
    }

    fn resolve_think(&self, effort: Option<ReasoningEffort>) -> Option<bool> {
        // `Auto` is caller-driven: thinking is on iff the request carried a
        // `reasoning_effort`. We explicitly emit `think=false` when effort
        // is absent rather than letting the server default apply, so the
        // behavior is deterministic across models (qwen3.6 defaults to
        // thinking ON when `think` is omitted).
        match self.config.think_mode {
            ThinkMode::Off => Some(false),
            ThinkMode::On => Some(true),
            ThinkMode::Auto => Some(effort.is_some()),
        }
    }

    fn build_chat_body(&self, request: CompletionRequest, stream: bool) -> Result<Value, LLMError> {
        let CompletionRequest {
            model,
            messages,
            max_tokens,
            temperature,
            top_p,
            response_format,
            tools,
            tool_choice: _,
            reasoning_effort,
        } = request;

        let mut body = Map::new();
        body.insert("model".into(), Value::String(model));
        body.insert(
            "messages".into(),
            Value::Array(
                messages
                    .iter()
                    .map(|m| message_to_ollama(m, self.config.preserve_thinking))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
        );
        body.insert("stream".into(), Value::Bool(stream));
        body.insert(
            "keep_alive".into(),
            Value::String(self.config.keep_alive.clone()),
        );

        if let Some(think) = self.resolve_think(reasoning_effort) {
            body.insert("think".into(), Value::Bool(think));
        }

        if !tools.is_empty() {
            body.insert(
                "tools".into(),
                Value::Array(tools.iter().map(tool_to_ollama).collect()),
            );
        }

        // Ollama does not honor OpenAI's tool_choice. Ignore silently.

        if let Some(fmt) = response_format {
            if let Some(converted) = response_format_to_native(&fmt) {
                body.insert("format".into(), converted);
            }
        }

        let mut options = Map::new();
        if let Some(ctx) = self.config.num_ctx {
            options.insert("num_ctx".into(), json!(ctx));
        }
        if let Some(t) = max_tokens {
            options.insert("num_predict".into(), json!(t));
        }
        if let Some(t) = temperature {
            options.insert("temperature".into(), json!(t));
        }
        if let Some(t) = top_p {
            options.insert("top_p".into(), json!(t));
        }
        if !options.is_empty() {
            body.insert("options".into(), Value::Object(options));
        }

        Ok(Value::Object(body))
    }
}

// ---------------------------------------------------------------------------
// Message / tool serialization — native Ollama shape
// ---------------------------------------------------------------------------

fn message_to_ollama(msg: &ChatMessage, preserve_thinking: bool) -> Result<Value, LLMError> {
    let role = match msg.role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    };

    let mut obj = Map::new();
    obj.insert("role".into(), Value::String(role.into()));

    if let Some(content) = &msg.content {
        obj.insert("content".into(), Value::String(content.clone()));
    } else {
        obj.insert("content".into(), Value::String(String::new()));
    }

    if !msg.images.is_empty() {
        let images: Result<Vec<Value>, LLMError> = msg
            .images
            .iter()
            .map(|img| image_to_ollama_base64(img).map(Value::String))
            .collect();
        obj.insert("images".into(), Value::Array(images?));
    }

    if matches!(msg.role, MessageRole::Assistant) {
        if preserve_thinking {
            if let Some(thinking) = &msg.thinking {
                if !thinking.is_empty() {
                    obj.insert("thinking".into(), Value::String(thinking.clone()));
                }
            }
        }

        if !msg.tool_calls.is_empty() {
            obj.insert(
                "tool_calls".into(),
                Value::Array(msg.tool_calls.iter().map(tool_call_to_ollama).collect()),
            );
        }
    }

    if matches!(msg.role, MessageRole::Tool) {
        if let Some(name) = &msg.name {
            obj.insert("tool_name".into(), Value::String(name.clone()));
        }
    }

    Ok(Value::Object(obj))
}

/// Ollama wants raw base64 in `images`. Accept either a data URL or a bare
/// base64 string.
fn image_to_ollama_base64(image: &str) -> Result<String, LLMError> {
    if let Some(idx) = image.find("base64,") {
        return Ok(image[idx + "base64,".len()..].to_string());
    }
    if image.starts_with("http://") || image.starts_with("https://") {
        // Ollama native API does not fetch remote URLs — require the caller to
        // have downloaded the image and attached it as a data URL.
        return Err(LLMError::InvalidResponse(
            "ollama requires base64 image data; remote URLs are not supported",
        ));
    }
    // Assume the caller already passed raw base64; the server will reject
    // anything malformed with a clear error.
    Ok(image.to_string())
}

fn tool_call_to_ollama(tc: &ToolCall) -> Value {
    // Native Ollama expects `arguments` as a JSON object (not a string).
    let mut function = Map::new();
    function.insert("name".into(), Value::String(tc.function.name.clone()));
    function.insert("arguments".into(), tc.function.arguments.clone());

    let mut outer = Map::new();
    if let Some(id) = &tc.id {
        outer.insert("id".into(), Value::String(id.clone()));
    }
    outer.insert("function".into(), Value::Object(function));
    Value::Object(outer)
}

fn tool_to_ollama(tool: &Tool) -> Value {
    // The Tool definition shape is already OpenAI-compatible and Ollama
    // accepts it verbatim.
    serde_json::to_value(tool).unwrap_or_else(|_| Value::Null)
}

/// Convert the OpenAI-style `response_format` wrapper to Ollama's `format`.
/// * `{"type": "json_object"}` → `"json"`
/// * `{"type": "json_schema", "json_schema": {"schema": <S>, ...}}` → `<S>`
/// * otherwise → pass through as-is (assume caller already used native shape).
fn response_format_to_native(fmt: &Value) -> Option<Value> {
    let kind = fmt.get("type").and_then(|v| v.as_str());
    match kind {
        Some("json_object") => Some(Value::String("json".into())),
        Some("json_schema") => fmt
            .get("json_schema")
            .and_then(|js| js.get("schema"))
            .cloned(),
        _ => Some(fmt.clone()),
    }
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Default)]
struct OllamaChatResponse {
    #[serde(default)]
    message: OllamaMessage,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    done_reason: Option<String>,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
    #[serde(default)]
    eval_count: Option<u32>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct OllamaMessage {
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    thinking: Option<String>,
    #[serde(default)]
    tool_calls: Vec<OllamaToolCall>,
}

#[derive(Debug, Deserialize, Default)]
struct OllamaToolCall {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: OllamaToolCallFunction,
}

#[derive(Debug, Deserialize, Default)]
struct OllamaToolCallFunction {
    #[serde(default)]
    name: String,
    #[serde(default)]
    arguments: Value,
}

impl OllamaToolCall {
    fn into_tool_call(self, index: usize) -> ToolCall {
        let raw = if self.function.arguments.is_null() {
            None
        } else {
            serde_json::to_string(&self.function.arguments).ok()
        };
        let args = if self.function.arguments.is_null() {
            Value::Object(Map::new())
        } else {
            self.function.arguments
        };

        let id = self.id.unwrap_or_else(|| format!("call_ollama_{index}"));
        ToolCall {
            id: Some(id),
            kind: ToolCallType::Function,
            function: FunctionCall {
                name: self.function.name,
                arguments: args,
                raw_arguments: raw,
            },
        }
    }
}

#[derive(Debug, Deserialize)]
struct OllamaEmbedResponse {
    #[serde(default)]
    embeddings: Vec<Vec<f32>>,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaTag>,
}

#[derive(Debug, Deserialize)]
struct OllamaTag {
    name: String,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    modified_at: Option<String>,
    #[serde(default)]
    details: Option<OllamaTagDetails>,
}

#[derive(Debug, Deserialize, Default)]
struct OllamaTagDetails {
    #[serde(default)]
    family: Option<String>,
    #[serde(default)]
    families: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Default)]
struct OllamaShowResponse {
    #[serde(default)]
    capabilities: Vec<String>,
    #[serde(default)]
    details: Option<OllamaTagDetails>,
    #[serde(default)]
    model_info: Map<String, Value>,
    #[serde(default)]
    error: Option<String>,
}

/// Extract the model's maximum context length from an `/api/show` `model_info`
/// map. Ollama exposes it as `<architecture>.context_length` where
/// `<architecture>` comes from `general.architecture`. We fall back to any
/// other `*.context_length` key (excluding RoPE scaling sentinels like
/// `*.rope.scaling.original_context_length`, which reports the pre-scaling
/// value).
fn max_context_from_model_info(model_info: &Map<String, Value>) -> Option<u32> {
    fn as_u32(v: &Value) -> Option<u32> {
        v.as_u64().and_then(|n| u32::try_from(n).ok())
    }

    if let Some(arch) = model_info
        .get("general.architecture")
        .and_then(|v| v.as_str())
    {
        let key = format!("{arch}.context_length");
        if let Some(v) = model_info.get(&key).and_then(as_u32) {
            return Some(v);
        }
    }

    model_info
        .iter()
        .filter(|(k, _)| k.ends_with(".context_length") && !k.contains(".rope.scaling."))
        .filter_map(|(_, v)| as_u32(v))
        .max()
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

fn parse_error_text(text: &str) -> LLMError {
    if let Ok(value) = serde_json::from_str::<Value>(text) {
        if let Some(msg) = value.get("error").and_then(|v| v.as_str()) {
            return LLMError::Provider(msg.to_string());
        }
    }
    LLMError::Provider(text.to_string())
}

fn usage_from_counts(prompt: Option<u32>, eval: Option<u32>) -> Option<TokenUsage> {
    match (prompt, eval) {
        (None, None) => None,
        _ => {
            let p = prompt.unwrap_or(0);
            let c = eval.unwrap_or(0);
            Some(TokenUsage {
                prompt_tokens: p,
                completion_tokens: c,
                total_tokens: p + c,
                cached_tokens: None,
            })
        }
    }
}

fn assistant_message_from(
    msg: OllamaMessage,
    tool_calls: Vec<ToolCall>,
    thinking: Option<String>,
) -> ChatMessage {
    ChatMessage {
        role: MessageRole::Assistant,
        content: msg.content.filter(|s| !s.is_empty()),
        name: None,
        tool_call_id: None,
        tool_calls,
        images: Vec::new(),
        thinking: thinking.filter(|s| !s.is_empty()),
    }
}

// ---------------------------------------------------------------------------
// LLMProvider impl
// ---------------------------------------------------------------------------

#[async_trait]
impl LLMProvider for Ollama {
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, LLMError> {
        let body = self.build_chat_body(request, false)?;

        let response = self
            .prepare(self.client.post(self.endpoint("api/chat")))
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(parse_error_text(&text));
        }

        let parsed: OllamaChatResponse = response.json().await?;
        if let Some(err) = parsed.error {
            return Err(LLMError::Provider(err));
        }

        let tool_calls: Vec<ToolCall> = parsed
            .message
            .tool_calls
            .into_iter()
            .enumerate()
            .map(|(i, tc)| tc.into_tool_call(i))
            .collect();

        let thinking_for_reasoning = parsed.message.thinking.clone();
        let message = assistant_message_from(
            OllamaMessage {
                role: parsed.message.role.clone(),
                content: parsed.message.content.clone(),
                thinking: parsed.message.thinking.clone(),
                tool_calls: Vec::new(),
            },
            tool_calls,
            parsed.message.thinking,
        );

        let reasoning = thinking_for_reasoning
            .filter(|s| !s.is_empty())
            .map(|content| {
                vec![ReasoningTrace {
                    content,
                    finish_reason: parsed.done_reason.clone(),
                }]
            });

        Ok(CompletionResponse {
            message,
            usage: usage_from_counts(parsed.prompt_eval_count, parsed.eval_count),
            reasoning,
        })
    }

    async fn stream_completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionStream, LLMError> {
        let body = self.build_chat_body(request, true)?;

        let response = self
            .prepare(self.client.post(self.endpoint("api/chat")))
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(parse_error_text(&text));
        }

        let stream = try_stream! {
            let mut body_stream = response.bytes_stream();
            let mut buffer = Vec::<u8>::new();
            let mut message = String::new();
            let mut thinking_buf = String::new();
            let mut tool_calls: Vec<ToolCall> = Vec::new();
            let mut prompt_eval: Option<u32> = None;
            let mut eval_count: Option<u32> = None;
            let mut done_reason: Option<String> = None;
            let mut done = false;

            while let Some(chunk) = body_stream.next().await {
                let chunk = chunk?;
                buffer.extend_from_slice(&chunk);

                while let Some(line) = pop_ndjson_line(&mut buffer) {
                    if line.is_empty() {
                        continue;
                    }

                    let parsed: OllamaChatResponse = match serde_json::from_slice(&line) {
                        Ok(p) => p,
                        Err(_) => continue,
                    };

                    if let Some(err) = parsed.error {
                        Err(LLMError::Provider(err))?;
                    }

                    if let Some(text) = parsed.message.content {
                        if !text.is_empty() {
                            message.push_str(&text);
                            yield StreamEvent::MessageDelta(text);
                        }
                    }

                    if let Some(tr) = parsed.message.thinking {
                        if !tr.is_empty() {
                            thinking_buf.push_str(&tr);
                            yield StreamEvent::ReasoningDelta(tr);
                        }
                    }

                    for (i, tc) in parsed.message.tool_calls.into_iter().enumerate() {
                        let idx = tool_calls.len();
                        let call = tc.into_tool_call(idx);
                        let args = call
                            .function
                            .raw_arguments
                            .clone()
                            .unwrap_or_else(|| call.function.arguments.to_string());
                        tool_calls.push(call);
                        yield StreamEvent::ToolCallDelta {
                            index: idx,
                            arguments: args,
                        };
                        let _ = i;
                    }

                    if parsed.prompt_eval_count.is_some() {
                        prompt_eval = parsed.prompt_eval_count;
                    }
                    if parsed.eval_count.is_some() {
                        eval_count = parsed.eval_count;
                    }
                    if let Some(reason) = parsed.done_reason {
                        done_reason = Some(reason);
                    }

                    if parsed.done {
                        done = true;
                    }
                }

                if done {
                    break;
                }
            }

            let reasoning = if thinking_buf.is_empty() {
                None
            } else {
                Some(vec![ReasoningTrace {
                    content: thinking_buf.clone(),
                    finish_reason: done_reason.clone(),
                }])
            };

            let completion_message = ChatMessage {
                role: MessageRole::Assistant,
                content: if message.is_empty() { None } else { Some(message) },
                name: None,
                tool_call_id: None,
                tool_calls,
                images: Vec::new(),
                thinking: if thinking_buf.is_empty() { None } else { Some(thinking_buf) },
            };

            yield StreamEvent::Completed(CompletionResponse {
                message: completion_message,
                usage: usage_from_counts(prompt_eval, eval_count),
                reasoning,
            });
        };

        Ok(Box::pin(stream))
    }

    async fn create_embeddings(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LLMError> {
        let body = json!({
            "model": request.model,
            "input": request.input,
        });

        let response = self
            .prepare(self.client.post(self.endpoint("api/embed")))
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(parse_error_text(&text));
        }

        let parsed: OllamaEmbedResponse = response.json().await?;
        if let Some(err) = parsed.error {
            return Err(LLMError::Provider(err));
        }

        let data = parsed
            .embeddings
            .into_iter()
            .enumerate()
            .map(|(index, embedding)| crate::types::Embedding {
                object: "embedding".to_string(),
                embedding,
                index,
            })
            .collect();

        let usage = parsed.prompt_eval_count.map(|prompt_tokens| {
            crate::types::EmbeddingUsage {
                prompt_tokens,
                total_tokens: prompt_tokens,
            }
        });

        Ok(EmbeddingResponse {
            data,
            model: request.model,
            usage,
        })
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new(true, true, false, true)
    }

    async fn model_info(&self, id: &str) -> Result<ModelInfo, LLMError> {
        let response = self
            .prepare(self.client.post(self.endpoint("api/show")))
            .json(&json!({ "model": id }))
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(parse_error_text(&text));
        }

        let parsed: OllamaShowResponse = response.json().await?;
        if let Some(err) = parsed.error {
            return Err(LLMError::Provider(err));
        }

        let max_context = max_context_from_model_info(&parsed.model_info);

        Ok(model_info_from_show(
            id,
            parsed.capabilities,
            parsed.details.unwrap_or_default(),
            max_context,
        ))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, LLMError> {
        let response = self
            .prepare(self.client.get(self.endpoint("api/tags")))
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(parse_error_text(&text));
        }

        let parsed: OllamaTagsResponse = response.json().await?;

        // /api/tags doesn't include capabilities — leave them unknown.
        // Callers who need accurate caps should use model_info(id).
        let models = parsed
            .models
            .into_iter()
            .map(|m| {
                let details = m.details.unwrap_or_default();
                ModelInfo {
                    id: m.name.clone(),
                    name: m.model.unwrap_or(m.name),
                    provider: "ollama".to_string(),
                    created_at: None,
                    updated_at: m.modified_at,
                    // Lazy: /api/tags doesn't carry the max context; callers
                    // who need it should use `model_info(id)` or
                    // `max_context_length(id)`.
                    context_length: None,
                    max_completion_tokens: None,
                    input_modalities: vec!["text".to_string()],
                    output_modalities: vec!["text".to_string()],
                    pricing: crate::types::ModelPricing::default(),
                    capabilities: caps_from_family(details.family.as_deref()),
                    reasoning_config: None,
                }
            })
            .collect();

        Ok(models)
    }

    fn name(&self) -> &'static str {
        "ollama"
    }
}

fn model_info_from_show(
    id: &str,
    capabilities: Vec<String>,
    details: OllamaTagDetails,
    context_length: Option<u32>,
) -> ModelInfo {
    let has = |c: &str| capabilities.iter().any(|s| s == c);
    let supports_vision = has("vision")
        || details
            .families
            .as_deref()
            .map(|fs| fs.iter().any(|f| f == "clip"))
            .unwrap_or(false);

    let mut input_modalities = vec!["text".to_string()];
    if supports_vision {
        input_modalities.push("image".to_string());
    }

    ModelInfo {
        id: id.to_string(),
        name: id.to_string(),
        provider: "ollama".to_string(),
        created_at: None,
        updated_at: None,
        context_length,
        max_completion_tokens: None,
        input_modalities,
        output_modalities: vec!["text".to_string()],
        pricing: crate::types::ModelPricing::default(),
        capabilities: ModelCapabilities {
            supports_reasoning: has("thinking"),
            supports_function_calling: has("tools"),
            supports_tools: has("tools"),
            supports_tool_choice: false,
            supports_vision,
            supports_streaming: has("completion") || has("chat") || true,
            supports_json_schema: true,
        },
        reasoning_config: None,
    }
}

/// Fallback capability guess when /api/tags doesn't return a capabilities array.
fn caps_from_family(family: Option<&str>) -> ModelCapabilities {
    let lower = family.map(|s| s.to_ascii_lowercase()).unwrap_or_default();
    ModelCapabilities {
        supports_reasoning: lower.contains("qwen3") || lower.contains("gpt-oss"),
        supports_function_calling: true,
        supports_tools: true,
        supports_tool_choice: false,
        supports_vision: false,
        supports_streaming: true,
        supports_json_schema: true,
    }
}

/// Pop one `\n`-delimited line from the buffer. Returns `None` when there's no
/// complete line yet.
fn pop_ndjson_line(buffer: &mut Vec<u8>) -> Option<Vec<u8>> {
    let pos = buffer.iter().position(|&b| b == b'\n')?;
    let mut line = buffer[..pos].to_vec();
    buffer.drain(..=pos);
    // Tolerate CRLF.
    if line.last() == Some(&b'\r') {
        line.pop();
    }
    Some(line)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functions::FunctionCall;
    use serde_json::json;

    #[test]
    fn config_builder_sets_fields() {
        let config = OllamaConfig::new()
            .with_base_url("http://192.168.1.1:11434")
            .with_keep_alive("1h")
            .with_num_ctx(32768)
            .with_think_mode(ThinkMode::On)
            .with_preserve_thinking(true);

        assert_eq!(config.base_url, "http://192.168.1.1:11434");
        assert_eq!(config.keep_alive, "1h");
        assert_eq!(config.num_ctx, Some(32768));
        assert_eq!(config.think_mode, ThinkMode::On);
        assert!(config.preserve_thinking);
    }

    #[test]
    fn endpoint_formatting() {
        let ollama = Ollama::from_config(OllamaConfig::new()).unwrap();
        assert_eq!(ollama.endpoint("api/chat"), "http://localhost:11434/api/chat");
        assert_eq!(ollama.endpoint("/api/tags"), "http://localhost:11434/api/tags");
    }

    #[test]
    fn message_text_only() {
        let msg = ChatMessage::user("Hello");
        let v = message_to_ollama(&msg, false).unwrap();
        assert_eq!(v["role"], "user");
        assert_eq!(v["content"], "Hello");
        assert!(v.get("images").is_none());
    }

    #[test]
    fn message_with_images_strips_data_url_prefix() {
        let msg = ChatMessage::user_with_images(
            "describe",
            vec!["data:image/png;base64,AAAA".into()],
        );
        let v = message_to_ollama(&msg, false).unwrap();
        let imgs = v["images"].as_array().unwrap();
        assert_eq!(imgs.len(), 1);
        assert_eq!(imgs[0], "AAAA");
    }

    #[test]
    fn assistant_preserves_thinking_when_enabled() {
        let mut msg = ChatMessage::assistant("result");
        msg.thinking = Some("step 1…".into());

        let off = message_to_ollama(&msg, false).unwrap();
        assert!(off.get("thinking").is_none());

        let on = message_to_ollama(&msg, true).unwrap();
        assert_eq!(on["thinking"], "step 1…");
    }

    #[test]
    fn tool_call_arguments_are_object_not_string() {
        let mut msg = ChatMessage::assistant("");
        msg.tool_calls = vec![ToolCall {
            id: Some("call_1".into()),
            kind: ToolCallType::Function,
            function: FunctionCall::new("get_weather", json!({"city":"sf"})),
        }];
        let v = message_to_ollama(&msg, false).unwrap();
        let tc = &v["tool_calls"][0];
        assert_eq!(tc["id"], "call_1");
        assert_eq!(tc["function"]["name"], "get_weather");
        // Object, not string:
        assert!(tc["function"]["arguments"].is_object());
        assert_eq!(tc["function"]["arguments"]["city"], "sf");
    }

    #[test]
    fn think_mode_resolution() {
        let auto = Ollama::from_config(OllamaConfig::new()).unwrap();
        // Auto is caller-driven: explicit false when no effort, explicit
        // true when effort is set. Never passthrough (would let the server
        // default surface non-deterministically).
        assert_eq!(auto.resolve_think(None), Some(false));
        assert_eq!(auto.resolve_think(Some(ReasoningEffort::Low)), Some(true));

        let off = Ollama::from_config(OllamaConfig::new().with_think_mode(ThinkMode::Off)).unwrap();
        assert_eq!(off.resolve_think(Some(ReasoningEffort::High)), Some(false));

        let on = Ollama::from_config(OllamaConfig::new().with_think_mode(ThinkMode::On)).unwrap();
        assert_eq!(on.resolve_think(None), Some(true));
    }

    #[test]
    fn response_format_mapping() {
        assert_eq!(
            response_format_to_native(&json!({"type": "json_object"})),
            Some(Value::String("json".into()))
        );
        let schema = json!({"type":"json_schema","json_schema":{"schema":{"type":"object"}}});
        assert_eq!(
            response_format_to_native(&schema),
            Some(json!({"type":"object"}))
        );
    }

    #[test]
    fn parse_error_text_structured_and_plain() {
        let e = parse_error_text(r#"{"error":"model not found"}"#);
        assert!(matches!(e, LLMError::Provider(ref m) if m == "model not found"));
        let e = parse_error_text("boom");
        assert!(matches!(e, LLMError::Provider(ref m) if m == "boom"));
    }

    #[test]
    fn pop_ndjson_handles_crlf() {
        let mut buf = b"{\"a\":1}\r\n{\"b\":2}\n".to_vec();
        let l1 = pop_ndjson_line(&mut buf).unwrap();
        assert_eq!(l1, b"{\"a\":1}");
        let l2 = pop_ndjson_line(&mut buf).unwrap();
        assert_eq!(l2, b"{\"b\":2}");
        assert!(pop_ndjson_line(&mut buf).is_none());
    }

    #[test]
    fn max_context_uses_architecture_prefix() {
        let mut mi = Map::new();
        mi.insert("general.architecture".into(), json!("qwen35moe"));
        mi.insert("qwen35moe.context_length".into(), json!(262144));
        mi.insert("qwen35moe.rope.scaling.original_context_length".into(), json!(4096));
        assert_eq!(max_context_from_model_info(&mi), Some(262144));
    }

    #[test]
    fn max_context_falls_back_to_any_matching_key() {
        let mut mi = Map::new();
        // No general.architecture field.
        mi.insert("foo.context_length".into(), json!(8192));
        mi.insert("foo.rope.scaling.original_context_length".into(), json!(4096));
        assert_eq!(max_context_from_model_info(&mi), Some(8192));
    }

    #[test]
    fn max_context_returns_none_when_absent() {
        let mut mi = Map::new();
        mi.insert("general.architecture".into(), json!("bar"));
        assert_eq!(max_context_from_model_info(&mi), None);
    }

    #[test]
    fn usage_counts_sum() {
        let u = usage_from_counts(Some(10), Some(5)).unwrap();
        assert_eq!(u.prompt_tokens, 10);
        assert_eq!(u.completion_tokens, 5);
        assert_eq!(u.total_tokens, 15);
        assert!(usage_from_counts(None, None).is_none());
    }
}
