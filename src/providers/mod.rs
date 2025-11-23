use async_trait::async_trait;

use crate::types::{
    CompletionRequest, CompletionResponse, CompletionStream, ImageUploadRequest,
    ImageUploadResponse, ProviderCapabilities,
};
use crate::LLMError;

pub mod openai;
pub mod openrouter;
pub mod scripted;
pub mod azure_openai;

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

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::default()
    }

    fn name(&self) -> &'static str;
}
