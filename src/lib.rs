 pub mod error;
 pub mod providers;
 pub mod types;
 pub mod functions;
 pub mod agents;
 pub mod flows;
 pub mod plugins;
 pub mod history;
 pub mod eval;

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
pub use agents::{Agent, AgentError};
pub use flows::handoffflow::{
    AgentAction,
    HandoffEvent,
    HandoffOrchestrator,
    HandoffSession,
    HandoffTurn,
};
pub use flows::magentic::{
    MagenticDecision,
    MagenticEvent,
    MagenticManager,
    MagenticOrchestrator,
    MagenticRun,
};
pub use flows::sequential::{
    SequentialEvent,
    SequentialOrchestrator,
    SequentialRun,
};
pub use flows::concurrent::{
    ConcurrentEvent,
    ConcurrentOrchestrator,
    ConcurrentResult,
    ConcurrentRun,
};
pub use flows::group_chat::{
    GroupChatEvent,
    GroupChatManager,
    GroupChatOrchestrator,
    GroupChatRun,
    RoundRobinGroupChatManager,
};
 pub use plugins::math;
 pub use schemars::JsonSchema;
 pub use denkwerk_macros::{kernel_function, kernel_module};
 pub use eval::{
     scenario::{DecisionSource, EvalScenario, ExpectStep, ExpectedTrace, ScriptedTurn},
     report::{CaseReport, EvalReport},
     runner::EvalRunner,
 };
 pub use history::{
    ChatHistory,
    ChatHistoryCompressor,
    ChatHistorySummarizer,
    ConciseSummarizer,
    FixedWindowCompressor,
    NoopChatHistoryCompressor,
};
extern crate self as denkwerk;
