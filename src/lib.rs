pub mod error;
pub mod providers;
pub mod types;
pub mod functions;

pub use error::LLMError;
pub use providers::LLMProvider;
pub use types::{
    ChatMessage, CompletionRequest, CompletionResponse, CompletionStream, ImageUploadRequest,
    ImageUploadResponse, MessageRole, ProviderCapabilities, ReasoningTrace, StreamEvent, TokenUsage,
};
pub use functions::{
    DynKernelFunction, FunctionCall, FunctionDefinition, FunctionRegistry, Tool, ToolCall,
    ToolCallType, ToolChoice, ToolChoiceFunction, ToolChoiceKind, ToolChoiceSimple,
};
pub use schemars::JsonSchema;
pub use denkwerk_macros::{kernel_function, kernel_module};
