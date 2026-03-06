use async_trait::async_trait;
use serde::Deserialize;

use crate::types::{
    CompletionRequest, CompletionResponse, CompletionStream, ImageUploadRequest,
    ImageUploadResponse, ProviderCapabilities, EmbeddingRequest, EmbeddingResponse, ModelInfo,
};
use crate::LLMError;

pub mod openai;
pub mod openrouter;
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
