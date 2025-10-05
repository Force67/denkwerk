pub mod error;
pub mod providers;
pub mod types;
pub mod functions;
pub mod agents;
pub mod magentic;
pub mod sequential;
pub mod concurrent;
pub mod group_chat;
pub mod plugins;
pub mod history;

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
pub use agents::{
    Agent,
    AgentAction,
    AgentError,
    HandoffEvent,
    HandoffOrchestrator,
    HandoffSession,
    HandoffTurn,
};
pub use magentic::{
    MagenticDecision,
    MagenticEvent,
    MagenticManager,
    MagenticOrchestrator,
    MagenticRun,
};
pub use sequential::{
    SequentialEvent,
    SequentialOrchestrator,
    SequentialRun,
};
pub use concurrent::{
    ConcurrentEvent,
    ConcurrentOrchestrator,
    ConcurrentResult,
    ConcurrentRun,
};
 pub use group_chat::{
     GroupChatEvent,
     GroupChatManager,
     GroupChatOrchestrator,
     GroupChatRun,
     RoundRobinGroupChatManager,
 };
pub use plugins::math;
pub use schemars::JsonSchema;
pub use denkwerk_macros::{kernel_function, kernel_module};
pub use history::{
    ChatHistory,
    ChatHistoryCompressor,
    ChatHistorySummarizer,
    ConciseSummarizer,
    FixedWindowCompressor,
    NoopChatHistoryCompressor,
};
extern crate self as denkwerk;
