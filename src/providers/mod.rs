use async_trait::async_trait;
use serde::Deserialize;

use crate::types::{
    CompletionRequest, CompletionResponse, CompletionStream, ImageUploadRequest,
    ImageUploadResponse, ProviderCapabilities, EmbeddingRequest, EmbeddingResponse, ModelInfo,
};
use crate::LLMError;

pub mod openai;
pub mod openrouter;
pub mod ollama;
pub mod scripted;
pub mod azure_openai;

/// A single content block in a streaming delta. All OpenAI-compatible APIs use this shape
/// for structured content, but the standard chat completions API sends `delta.content` as
/// a plain string. The [`deserialize_content_blocks`] function handles both representations.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct StreamContentBlock {
    #[serde(rename = "type")]
    #[serde(default)]
    pub _kind: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
}

/// Deserializes a `Vec<T>` that tolerates `null` (→ empty vec) in addition to a proper array.
/// Many non-OpenAI providers (e.g. Kimi K2) return `"tool_calls": null` instead of omitting the
/// field entirely, which trips up serde's default `Vec` deserialization.
pub(crate) fn deserialize_null_as_empty_vec<'de, D, T>(
    deserializer: D,
) -> Result<Vec<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de>,
{
    Option::<Vec<T>>::deserialize(deserializer).map(|opt| opt.unwrap_or_default())
}

/// Deserializes a streaming delta content field that may arrive as:
/// - a plain string  (standard chat completions: `"content": "Hello"`)
/// - an array of content blocks (structured format: `"content": [{"type":"text","text":"Hello"}]`)
/// - null / absent
pub(crate) fn deserialize_content_blocks<'de, D>(
    deserializer: D,
) -> Result<Vec<StreamContentBlock>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;

    struct ContentVisitor;

    impl<'de> de::Visitor<'de> for ContentVisitor {
        type Value = Vec<StreamContentBlock>;

        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a string, null, or array of content blocks")
        }

        fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
            Ok(vec![StreamContentBlock { _kind: None, text: Some(v.to_owned()) }])
        }

        fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(Vec::new())
        }

        fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(Vec::new())
        }

        fn visit_seq<A: de::SeqAccess<'de>>(self, seq: A) -> Result<Self::Value, A::Error> {
            Vec::<StreamContentBlock>::deserialize(de::value::SeqAccessDeserializer::new(seq))
        }
    }

    deserializer.deserialize_any(ContentVisitor)
}

/// Marker tokens used by some providers (e.g. Kimi K2) to embed tool calls as text in the
/// assistant message content instead of using the structured `tool_calls` JSON field.
///
/// Format:
/// ```text
/// <|tool_calls_section_begin|>
/// <|tool_call_begin|>functions.name:call_id<|tool_call_argument_begin|>{"key":"value"}<|tool_call_end|>
/// <|tool_calls_section_end|>
/// ```
const TOOL_SECTION_BEGIN: &str = "<|tool_calls_section_begin|>";
const TOOL_SECTION_END: &str = "<|tool_calls_section_end|>";
const TOOL_CALL_BEGIN: &str = "<|tool_call_begin|>";
const TOOL_CALL_END: &str = "<|tool_call_end|>";
const TOOL_CALL_ARG_BEGIN: &str = "<|tool_call_argument_begin|>";

/// Parse text-embedded tool calls (Kimi K2 format) from message content.
/// Returns the extracted tool calls and the content with the tool-call section stripped.
pub(crate) fn parse_text_tool_calls(content: &str) -> (Vec<crate::functions::ToolCall>, String) {
    let Some(section_start) = content.find(TOOL_SECTION_BEGIN) else {
        return (Vec::new(), content.to_string());
    };
    let section_end_pos = content
        .find(TOOL_SECTION_END)
        .map(|p| p + TOOL_SECTION_END.len())
        .unwrap_or(content.len());

    let section = &content[section_start..section_end_pos];
    let cleaned = format!(
        "{}{}",
        content[..section_start].trim_end(),
        content[section_end_pos..].trim_start(),
    );

    let mut calls = Vec::new();
    let mut remaining = section;

    while let Some(call_start) = remaining.find(TOOL_CALL_BEGIN) {
        remaining = &remaining[call_start + TOOL_CALL_BEGIN.len()..];

        let call_end = remaining.find(TOOL_CALL_END).unwrap_or(remaining.len());
        let call_body = &remaining[..call_end];
        remaining = &remaining[call_end..];

        // Split header from arguments. The argument separator is either the explicit
        // <|tool_call_argument_begin|> token, or (when absent) the first '{' character.
        let (header, args_raw) = if let Some(sep) = call_body.find(TOOL_CALL_ARG_BEGIN) {
            (
                call_body[..sep].trim(),
                call_body[sep + TOOL_CALL_ARG_BEGIN.len()..].trim(),
            )
        } else if let Some(brace) = call_body.find('{') {
            (call_body[..brace].trim(), call_body[brace..].trim())
        } else {
            // No arguments found at all — skip this call
            continue;
        };

        // Parse header: "functions.name:id" or just "name:id" or "name"
        let func_name;
        let call_id;
        let header_clean = header.strip_prefix("functions.").unwrap_or(header);
        if let Some((name, id)) = header_clean.rsplit_once(':') {
            func_name = name.trim();
            call_id = Some(id.trim().to_string());
        } else {
            func_name = header_clean;
            call_id = None;
        }

        let arguments: serde_json::Value = serde_json::from_str(args_raw)
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

        calls.push(
            crate::functions::ToolCall::new(
                crate::functions::FunctionCall::new(func_name, arguments)
                    .with_raw_arguments(args_raw),
            )
            .with_id(call_id.unwrap_or_else(|| format!("call_text_{}", calls.len()))),
        );
    }

    (calls, cleaned)
}

/// Pop one SSE event (terminated by `\n\n` or `\r\n\r\n`) from a byte buffer.
/// Returns `None` if no complete event is buffered yet.
pub(crate) fn extract_sse_event(buffer: &mut Vec<u8>) -> Option<Vec<u8>> {
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

/// Join all `data:` lines from an SSE event into a single payload string.
pub(crate) fn extract_data_payload(event: &[u8]) -> Result<String, LLMError> {
    let text = std::str::from_utf8(event)
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

#[async_trait]
pub trait LLMProvider: Send + Sync {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LLMError>;

    async fn stream_completion(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionStream, LLMError> {
        Err(LLMError::Unsupported("streaming completions"))
    }

    async fn upload_image(
        &self,
        _request: ImageUploadRequest,
    ) -> Result<ImageUploadResponse, LLMError> {
        Err(LLMError::Unsupported("image uploads"))
    }

    async fn create_embeddings(
        &self,
        _request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LLMError> {
        Err(LLMError::Unsupported("embeddings"))
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::default()
    }

    async fn model_info(&self, _id: &str) -> Result<ModelInfo, LLMError> {
        Err(LLMError::Unsupported("model info"))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, LLMError> {
        Err(LLMError::Unsupported("model list"))
    }

    fn name(&self) -> &'static str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_kimi_k2_tool_calls() {
        let content = "Some preamble text\n<|tool_calls_section_begin|>\n<|tool_call_begin|>functions.code_execution:13<|tool_call_argument_begin|>{\"language\": \"bash\", \"code\": \"echo hello\"}<|tool_call_end|>\n<|tool_calls_section_end|>";
        let (calls, cleaned) = parse_text_tool_calls(content);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "code_execution");
        assert_eq!(calls[0].id.as_deref(), Some("13"));
        assert_eq!(calls[0].function.arguments["language"], "bash");
        assert_eq!(calls[0].function.arguments["code"], "echo hello");
        assert_eq!(cleaned, "Some preamble text");
    }

    #[test]
    fn parse_multiple_text_tool_calls() {
        let content = "<|tool_calls_section_begin|><|tool_call_begin|>functions.search:1<|tool_call_argument_begin|>{\"q\":\"rust\"}<|tool_call_end|><|tool_call_begin|>functions.read_file:2<|tool_call_argument_begin|>{\"path\":\"/tmp/a\"}<|tool_call_end|><|tool_calls_section_end|>";
        let (calls, cleaned) = parse_text_tool_calls(content);

        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "search");
        assert_eq!(calls[0].id.as_deref(), Some("1"));
        assert_eq!(calls[1].function.name, "read_file");
        assert_eq!(calls[1].id.as_deref(), Some("2"));
        assert!(cleaned.is_empty());
    }

    #[test]
    fn no_text_tool_calls_returns_original() {
        let content = "Just a normal message with no tool calls";
        let (calls, cleaned) = parse_text_tool_calls(content);

        assert!(calls.is_empty());
        assert_eq!(cleaned, content);
    }

    #[test]
    fn parse_kimi_k2_without_argument_begin_token() {
        let content = "<|tool_calls_section_begin|>   <|tool_call_begin|>  functions.code_execution:14 {\"language\": \"python\", \"code\": \"print('hello')\"} <|tool_call_end|> <|tool_calls_section_end|>";
        let (calls, cleaned) = parse_text_tool_calls(content);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "code_execution");
        assert_eq!(calls[0].id.as_deref(), Some("14"));
        assert_eq!(calls[0].function.arguments["language"], "python");
        assert!(cleaned.is_empty());
    }

    #[test]
    fn parse_tool_call_without_functions_prefix() {
        let content = "<|tool_calls_section_begin|><|tool_call_begin|>my_tool:42<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>";
        let (calls, cleaned) = parse_text_tool_calls(content);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "my_tool");
        assert_eq!(calls[0].id.as_deref(), Some("42"));
        assert!(cleaned.is_empty());
    }
}
