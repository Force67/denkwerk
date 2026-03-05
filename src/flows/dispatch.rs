//! # Dispatch Orchestrator (Hub-and-Spoke Pattern)
//!
//! A hub agent routes work to specialised spoke agents via a `dispatch` tool.
//! Spokes execute with their own focused tool registries, return results, and
//! the hub synthesises a final response.
//!
//! ## Key Features
//!
//! * **Parallel dispatch** – the hub can call `dispatch` multiple times in a
//!   single turn; all spoke invocations run concurrently.
//! * **Input pre-routing** – deterministic keyword / regex / predicate matching
//!   on the user's message *before* any LLM call, routing directly to a spoke
//!   for zero-overhead classification of obvious intents.
//! * **Configurable spokes** – each spoke has its own max tool-calling rounds,
//!   context window depth, and optional model / provider overrides.
//!
//! ## Typical flow
//!
//! ```text
//! User message
//!     │
//!     ├─ input route match? ──▶ Spoke handles directly (1 LLM call)
//!     │
//!     └─ no match ──▶ Hub processes
//!                        ├─ responds directly (1 LLM call)
//!                        └─ calls dispatch(spoke, task)
//!                              └─ spoke executes (parallel if multiple)
//!                                    └─ results return to hub
//!                                          └─ hub synthesises (1 extra LLM round)
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;

use crate::{
    functions::{
        json_schema_for, FunctionDefinition, FunctionParameter, FunctionRegistry, KernelFunction,
        ToolCall,
    },
    metrics::{AgentMetrics, MetricsCollector},
    types::{ChatMessage, CompletionRequest, TokenUsage},
    Agent, AgentError, LLMError, LLMProvider,
};

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// How to match a user's input for pre-routing.
#[derive(Clone)]
pub enum InputMatcher {
    /// Match if **any** keyword appears (case-insensitive, substring).
    KeywordsAny(Vec<String>),
    /// Match if **all** keywords appear (case-insensitive, substring).
    KeywordsAll(Vec<String>),
    /// Full regex match on the raw input.
    Regex(Regex),
    /// Arbitrary predicate: `(transcript, user_input) → bool`.
    Predicate(Arc<dyn Fn(&[ChatMessage], &str) -> bool + Send + Sync>),
}

/// A pre-routing rule checked against the user's message *before* any agent
/// runs.  When matched the message is forwarded directly to the target spoke.
pub struct InputRoute {
    pub target: String,
    pub matcher: InputMatcher,
}

impl InputRoute {
    /// Create a route that fires when **any** keyword is present.
    pub fn keywords_any(target: impl Into<String>, keywords: &[&str]) -> Self {
        Self {
            target: target.into(),
            matcher: InputMatcher::KeywordsAny(
                keywords.iter().map(|s| s.to_lowercase()).collect(),
            ),
        }
    }

    /// Create a route that fires when **all** keywords are present.
    pub fn keywords_all(target: impl Into<String>, keywords: &[&str]) -> Self {
        Self {
            target: target.into(),
            matcher: InputMatcher::KeywordsAll(
                keywords.iter().map(|s| s.to_lowercase()).collect(),
            ),
        }
    }

    /// Create a route that fires on a regex match.
    pub fn regex(target: impl Into<String>, regex: Regex) -> Self {
        Self {
            target: target.into(),
            matcher: InputMatcher::Regex(regex),
        }
    }

    /// Create a route with a custom predicate.
    pub fn predicate(
        target: impl Into<String>,
        pred: impl Fn(&[ChatMessage], &str) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            target: target.into(),
            matcher: InputMatcher::Predicate(Arc::new(pred)),
        }
    }
}

/// Configuration for a spoke agent.
pub struct SpokeConfig {
    pub agent: Agent,
    /// Maximum tool-calling rounds the spoke may use (default: 4).
    pub max_rounds: usize,
    /// How many recent transcript messages to include as context (default: 10).
    pub context_window: usize,
}

impl SpokeConfig {
    pub fn new(agent: Agent) -> Self {
        Self {
            agent,
            max_rounds: 4,
            context_window: 10,
        }
    }

    pub fn with_max_rounds(mut self, rounds: usize) -> Self {
        self.max_rounds = rounds;
        self
    }

    pub fn with_context_window(mut self, window: usize) -> Self {
        self.context_window = window;
        self
    }
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// Observable events emitted during a dispatch turn.
#[derive(Debug, Clone)]
pub enum DispatchEvent {
    /// Hub produced a final text message.
    HubMessage { message: String },
    /// A spoke was dispatched with a task.
    SpokeDispatched { spoke: String, task: String },
    /// A spoke finished its task.
    SpokeCompleted { spoke: String, result: String },
    /// Multiple spokes were dispatched in the same hub turn (parallel).
    ParallelDispatch { spokes: Vec<String> },
    /// User input was pre-routed to a spoke (no hub call).
    InputRouted { target: String },
    /// A regular (non-dispatch) tool was called by the hub.
    HubToolCalled { name: String },
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// The result of a single spoke execution.
#[derive(Debug, Clone)]
pub struct SpokeResult {
    /// Spoke name.
    pub name: String,
    /// The spoke's final text response.
    pub response: String,
    /// All tool calls the spoke made during execution.
    pub tool_calls: Vec<ToolCall>,
    /// Cumulative token usage across all rounds.
    pub usage: Option<TokenUsage>,
}

/// The output of a single `send()` call.
#[derive(Debug)]
pub struct DispatchTurn {
    /// Final reply text (may be `None` if the hub produced no text).
    pub reply: Option<String>,
    /// Chronological events from this turn.
    pub events: Vec<DispatchEvent>,
    /// Spoke invocations that occurred (empty for pure-conversation turns).
    pub spoke_results: Vec<SpokeResult>,
    /// Aggregated metrics (if a collector is configured).
    pub metrics: Option<AgentMetrics>,
    /// Which agent produced the final reply.
    pub responding_agent: String,
}

// ---------------------------------------------------------------------------
// Internal: dispatch tool
// ---------------------------------------------------------------------------

const DISPATCH_TOOL_NAME: &str = "dispatch";

/// A no-op kernel function whose invocation is intercepted by the orchestrator.
/// The tool definition tells the hub which spokes are available.
struct DispatchToolStub {
    spoke_names: Vec<String>,
}

#[async_trait]
impl KernelFunction for DispatchToolStub {
    fn definition(&self) -> FunctionDefinition {
        let roster = self.spoke_names.join(", ");

        let mut def = FunctionDefinition::new(DISPATCH_TOOL_NAME).with_description(format!(
            "Dispatch a task to a specialist agent who will execute it using their own tools and \
             return the result. Available specialists: [{roster}]. You may call this tool multiple \
             times in the same turn to run several specialists in parallel."
        ));

        let agent_schema = serde_json::json!({
            "type": "string",
            "enum": self.spoke_names,
        });
        def.add_parameter(
            FunctionParameter {
                name: "agent".to_string(),
                schema: agent_schema,
                description: Some("The specialist agent to dispatch to.".to_string()),
                required: true,
                default: None,
            },
        );
        def.add_parameter(
            FunctionParameter::new("task", json_schema_for::<String>()).with_description(
                "Detailed task description with all relevant context from the conversation. \
                 The specialist cannot see the chat history, so include everything they need.",
            ),
        );

        def
    }

    async fn invoke(&self, _arguments: &Value) -> Result<Value, LLMError> {
        // Never actually called — the orchestrator intercepts dispatch tool
        // calls before they reach the registry.
        Ok(serde_json::json!({"error": "dispatch stub: should not be invoked directly"}))
    }
}

// ---------------------------------------------------------------------------
// Spoke execution (free function to avoid borrow issues during parallel runs)
// ---------------------------------------------------------------------------

async fn execute_spoke(
    spoke_name: &str,
    config: &SpokeConfig,
    transcript_tail: &[ChatMessage],
    task: &str,
    default_provider: &Arc<dyn LLMProvider>,
    default_model: &str,
    timeout_ms: u64,
) -> Result<SpokeResult, AgentError> {
    let provider = config
        .agent
        .provider_override()
        .unwrap_or_else(|| default_provider.clone());
    let model = config.agent.model_override().unwrap_or(default_model);

    // Build messages: [System] + [context tail] + [User: task]
    let mut messages = Vec::with_capacity(transcript_tail.len() + 2);
    messages.push(ChatMessage::system(config.agent.instructions().to_string()));
    messages.extend_from_slice(transcript_tail);
    messages.push(ChatMessage::user(task.to_string()));

    let registry = config.agent.function_registry();
    let mut all_tool_calls: Vec<ToolCall> = Vec::new();
    let mut cumulative_usage: Option<TokenUsage> = None;
    let mut last_content = String::new();

    for round in 0..config.max_rounds {
        let mut request = CompletionRequest::new(model.to_string(), messages.clone());
        if let Some(t) = config.agent.max_tokens() {
            request = request.with_max_tokens(t);
        }
        if let Some(t) = config.agent.temperature() {
            request = request.with_temperature(t);
        }
        if let Some(t) = config.agent.top_p() {
            request = request.with_top_p(t);
        }
        if let Some(ref reg) = registry {
            request = request.with_function_registry(reg);
        }

        let response = tokio::time::timeout(
            std::time::Duration::from_millis(timeout_ms),
            provider.complete(request),
        )
        .await
        .map_err(|_| AgentError::ProviderTimeout)?
        .map_err(AgentError::Provider)?;

        // Accumulate token usage.
        if let Some(usage) = &response.usage {
            cumulative_usage = Some(match cumulative_usage {
                Some(mut prev) => {
                    prev.prompt_tokens += usage.prompt_tokens;
                    prev.completion_tokens += usage.completion_tokens;
                    prev.total_tokens += usage.total_tokens;
                    prev
                }
                None => usage.clone(),
            });
        }

        let mut assistant_msg = response.message;

        // Ensure every tool call has an ID.
        for (i, call) in assistant_msg.tool_calls.iter_mut().enumerate() {
            if call.id.is_none() {
                call.id = Some(format!("spoke_{spoke_name}_{round}_{i}"));
            }
        }

        last_content = assistant_msg.text().unwrap_or_default().to_string();
        all_tool_calls.extend(assistant_msg.tool_calls.clone());
        messages.push(assistant_msg.clone());

        if assistant_msg.tool_calls.is_empty() {
            break;
        }

        let Some(ref reg) = registry else { break };

        for call in &assistant_msg.tool_calls {
            let id = call
                .id
                .clone()
                .unwrap_or_else(|| format!("spoke_{spoke_name}_{round}_x"));
            let result = match reg.invoke(&call.function).await {
                Ok(value) => serde_json::to_string(&value).unwrap_or_default(),
                Err(err) => serde_json::json!({"error": err.to_string()}).to_string(),
            };
            messages.push(ChatMessage::tool(id, result));
        }
    }

    // If the spoke used tools but produced no text (e.g. the model returned an
    // empty final response after tool use), do one more LLM call WITHOUT tools
    // to force a text summary of what happened.
    if last_content.trim().is_empty() && !all_tool_calls.is_empty() {
        // Inject a nudge so the model knows it must produce a text summary.
        messages.push(ChatMessage::system(
            "You have completed your tool calls. Now write a brief, natural response \
             to the user summarising what you did. Do NOT call any tools."
                .to_string(),
        ));

        let mut request = CompletionRequest::new(model.to_string(), messages);
        if let Some(t) = config.agent.max_tokens() {
            request = request.with_max_tokens(t);
        }
        if let Some(t) = config.agent.temperature() {
            request = request.with_temperature(t);
        }
        // Deliberately omit function_registry so the model MUST reply with text.
        match tokio::time::timeout(
            std::time::Duration::from_millis(timeout_ms),
            provider.complete(request),
        )
        .await
        {
            Ok(Ok(response)) => {
                let forced = response.message.text().unwrap_or_default().to_string();
                if !forced.trim().is_empty() {
                    last_content = forced;
                }
                if let Some(usage) = &response.usage {
                    cumulative_usage = Some(match cumulative_usage {
                        Some(mut prev) => {
                            prev.prompt_tokens += usage.prompt_tokens;
                            prev.completion_tokens += usage.completion_tokens;
                            prev.total_tokens += usage.total_tokens;
                            prev
                        }
                        None => usage.clone(),
                    });
                }
            }
            Ok(Err(_)) | Err(_) => {
                // Best-effort — if the forced call fails, we proceed with empty
                // text and let the caller handle it (e.g. hub synthesis fallback).
            }
        }
    }

    Ok(SpokeResult {
        name: spoke_name.to_string(),
        response: last_content,
        tool_calls: all_tool_calls,
        usage: cumulative_usage,
    })
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

pub struct DispatchOrchestrator {
    provider: Arc<dyn LLMProvider>,
    model: String,
    hub: Agent,
    spokes: HashMap<String, SpokeConfig>,
    input_routes: Vec<InputRoute>,
    /// Maximum tool-calling rounds for the hub agent (default: 5).
    max_hub_rounds: usize,
    /// Timeout per LLM call in milliseconds (default: 60 000).
    llm_timeout_ms: u64,
    event_callback: Option<Arc<dyn Fn(&DispatchEvent) + Send + Sync>>,
    metrics_collector: Option<Arc<dyn MetricsCollector>>,
}

impl DispatchOrchestrator {
    pub fn new(provider: Arc<dyn LLMProvider>, model: impl Into<String>, hub: Agent) -> Self {
        Self {
            provider,
            model: model.into(),
            hub,
            spokes: HashMap::new(),
            input_routes: Vec::new(),
            max_hub_rounds: 5,
            llm_timeout_ms: 60_000,
            event_callback: None,
            metrics_collector: None,
        }
    }

    /// Register a spoke agent under the given name.
    pub fn register_spoke(mut self, name: impl Into<String>, config: SpokeConfig) -> Self {
        self.spokes.insert(name.into(), config);
        self
    }

    /// Add a pre-routing rule.  Rules are evaluated in order; first match wins.
    pub fn define_input_route(mut self, route: InputRoute) -> Self {
        self.input_routes.push(route);
        self
    }

    pub fn with_max_hub_rounds(mut self, rounds: usize) -> Self {
        self.max_hub_rounds = rounds;
        self
    }

    pub fn with_llm_timeout_ms(mut self, ms: u64) -> Self {
        self.llm_timeout_ms = ms;
        self
    }

    pub fn with_event_callback(
        mut self,
        cb: impl Fn(&DispatchEvent) + Send + Sync + 'static,
    ) -> Self {
        self.event_callback = Some(Arc::new(cb));
        self
    }

    pub fn with_metrics_collector(mut self, collector: Arc<dyn MetricsCollector>) -> Self {
        self.metrics_collector = Some(collector);
        self
    }

    pub fn hub(&self) -> &Agent {
        &self.hub
    }

    pub fn spoke(&self, name: &str) -> Option<&SpokeConfig> {
        self.spokes.get(name)
    }

    pub fn spoke_names(&self) -> Vec<&str> {
        self.spokes.keys().map(|s| s.as_str()).collect()
    }

    /// Create a new conversation session.
    pub fn session(&self) -> DispatchSession<'_> {
        DispatchSession {
            orchestrator: self,
            transcript: Vec::new(),
        }
    }

    // -- private helpers --

    fn emit(&self, event: &DispatchEvent) {
        if let Some(cb) = &self.event_callback {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (cb)(event)));
        }
    }

    fn match_input_routes(&self, transcript: &[ChatMessage], user_input: &str) -> Option<&str> {
        let lower = user_input.to_lowercase();
        for route in &self.input_routes {
            let matched = match &route.matcher {
                InputMatcher::KeywordsAny(kws) => kws.iter().any(|kw| lower.contains(kw)),
                InputMatcher::KeywordsAll(kws) => kws.iter().all(|kw| lower.contains(kw)),
                InputMatcher::Regex(re) => re.is_match(user_input),
                InputMatcher::Predicate(pred) => pred(transcript, user_input),
            };
            if matched && self.spokes.contains_key(&route.target) {
                return Some(&route.target);
            }
        }
        None
    }

    /// Build a `FunctionRegistry` for the hub that includes its own tools plus
    /// the `dispatch` stub.
    fn build_hub_registry(&self) -> FunctionRegistry {
        let mut reg = FunctionRegistry::new();
        if let Some(agent_reg) = self.hub.function_registry() {
            reg.extend_from(&agent_reg);
        }
        let spoke_names: Vec<String> = self.spokes.keys().cloned().collect();
        reg.register(Arc::new(DispatchToolStub { spoke_names }) as Arc<dyn KernelFunction>);
        reg
    }
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

pub struct DispatchSession<'a> {
    orchestrator: &'a DispatchOrchestrator,
    transcript: Vec<ChatMessage>,
}

impl<'a> DispatchSession<'a> {
    pub fn transcript(&self) -> &[ChatMessage] {
        &self.transcript
    }

    pub fn set_history(&mut self, history: Vec<ChatMessage>) {
        self.transcript = history;
    }

    pub fn push_message(&mut self, msg: ChatMessage) {
        self.transcript.push(msg);
    }

    // -- public entry point --

    pub async fn send(&mut self, user_input: impl Into<String>) -> Result<DispatchTurn, AgentError> {
        let user_input = user_input.into();
        self.transcript.push(ChatMessage::user(user_input.clone()));

        // 1. Try deterministic pre-routing.
        if let Some(target) = self.orchestrator.match_input_routes(&self.transcript, &user_input) {
            let target = target.to_string(); // release borrow on orchestrator
            return self.handle_pre_routed(&target, &user_input).await;
        }

        // 2. Hub-mediated turn.
        self.handle_hub_turn().await
    }

    // -- pre-routed path --

    async fn handle_pre_routed(
        &mut self,
        target: &str,
        user_input: &str,
    ) -> Result<DispatchTurn, AgentError> {
        let config = self
            .orchestrator
            .spokes
            .get(target)
            .ok_or_else(|| AgentError::UnknownAgent(target.to_string()))?;

        self.orchestrator
            .emit(&DispatchEvent::InputRouted { target: target.to_string() });

        let tail = self.transcript_tail(config.context_window);
        let result = execute_spoke(
            target,
            config,
            tail,
            user_input,
            &self.orchestrator.provider,
            &self.orchestrator.model,
            self.orchestrator.llm_timeout_ms,
        )
        .await?;

        let mut reply = result.response.clone();

        // If the spoke used tools but produced no text, ask the hub to
        // synthesise a response based on the spoke's work.
        if reply.trim().is_empty() && !result.tool_calls.is_empty() {
            let tools_summary: Vec<&str> = result
                .tool_calls
                .iter()
                .map(|tc| tc.function.name.as_str())
                .collect();
            let spoke_summary = serde_json::json!({
                "agent": target,
                "result": "(spoke produced no text)",
                "tools_used": tools_summary,
            });

            let orch = self.orchestrator;
            let hub_provider =
                orch.hub.provider_override().unwrap_or_else(|| orch.provider.clone());
            let hub_model = orch.hub.model_override().unwrap_or(&orch.model);

            let mut synth_msgs = Vec::with_capacity(self.transcript.len() + 3);
            synth_msgs.push(ChatMessage::system(orch.hub.instructions().to_string()));
            synth_msgs.extend(self.transcript.iter().cloned());
            synth_msgs.push(ChatMessage::system(format!(
                "A specialist agent ({target}) handled the user's request and used \
                 these tools: {spoke_summary}. Write a brief, natural response to \
                 the user summarising what happened. Do NOT call any tools."
            )));

            let request = CompletionRequest::new(hub_model.to_string(), synth_msgs);
            if let Ok(Ok(response)) = tokio::time::timeout(
                std::time::Duration::from_millis(orch.llm_timeout_ms),
                hub_provider.complete(request),
            )
            .await
            {
                let hub_reply = response.message.text().unwrap_or_default().to_string();
                if !hub_reply.trim().is_empty() {
                    reply = hub_reply;
                }
            }
        }

        self.transcript.push(ChatMessage::assistant(reply.clone()));

        self.orchestrator
            .emit(&DispatchEvent::SpokeCompleted {
                spoke: target.to_string(),
                result: reply.clone(),
            });

        let responding = if reply == result.response {
            target.to_string()
        } else {
            self.orchestrator.hub.name().to_string()
        };

        Ok(DispatchTurn {
            reply: Some(reply),
            events: vec![
                DispatchEvent::InputRouted { target: target.to_string() },
                DispatchEvent::SpokeCompleted {
                    spoke: target.to_string(),
                    result: result.response.clone(),
                },
            ],
            spoke_results: vec![result],
            metrics: None,
            responding_agent: responding,
        })
    }

    // -- hub-mediated path --

    async fn handle_hub_turn(&mut self) -> Result<DispatchTurn, AgentError> {
        let orch = self.orchestrator;
        let hub_registry = orch.build_hub_registry();

        let hub_provider = orch.hub.provider_override().unwrap_or_else(|| orch.provider.clone());
        let hub_model = orch.hub.model_override().unwrap_or(&orch.model);

        // Assemble messages: [System: hub instructions] + transcript
        let mut messages = Vec::with_capacity(self.transcript.len() + 1);
        messages.push(ChatMessage::system(orch.hub.instructions().to_string()));
        messages.extend(self.transcript.iter().cloned());

        let mut events: Vec<DispatchEvent> = Vec::new();
        let mut spoke_results: Vec<SpokeResult> = Vec::new();
        let mut last_content = String::new();

        for round in 0..orch.max_hub_rounds {
            let mut request =
                CompletionRequest::new(hub_model.to_string(), messages.clone())
                    .with_function_registry(&hub_registry);

            if let Some(t) = orch.hub.max_tokens() {
                request = request.with_max_tokens(t);
            }
            if let Some(t) = orch.hub.temperature() {
                request = request.with_temperature(t);
            }
            if let Some(t) = orch.hub.top_p() {
                request = request.with_top_p(t);
            }

            let response = tokio::time::timeout(
                std::time::Duration::from_millis(orch.llm_timeout_ms),
                hub_provider.complete(request),
            )
            .await
            .map_err(|_| AgentError::ProviderTimeout)?
            .map_err(AgentError::Provider)?;

            let mut assistant_msg = response.message;

            // Ensure every tool call has an ID.
            for (i, call) in assistant_msg.tool_calls.iter_mut().enumerate() {
                if call.id.is_none() {
                    call.id = Some(format!("hub_{round}_{i}"));
                }
            }

            last_content = assistant_msg.text().unwrap_or_default().to_string();
            messages.push(assistant_msg.clone());

            // No tool calls → final text response.
            if assistant_msg.tool_calls.is_empty() {
                break;
            }

            // Partition tool calls into dispatch vs regular.
            let mut dispatch_calls: Vec<&ToolCall> = Vec::new();
            let mut regular_calls: Vec<&ToolCall> = Vec::new();

            for call in &assistant_msg.tool_calls {
                if call.function.name == DISPATCH_TOOL_NAME {
                    dispatch_calls.push(call);
                } else {
                    regular_calls.push(call);
                }
            }

            // Execute regular tool calls sequentially.
            for call in &regular_calls {
                let id = call.id.clone().unwrap_or_default();
                let result = match hub_registry.invoke(&call.function).await {
                    Ok(value) => serde_json::to_string(&value).unwrap_or_default(),
                    Err(err) => serde_json::json!({"error": err.to_string()}).to_string(),
                };
                events.push(DispatchEvent::HubToolCalled {
                    name: call.function.name.clone(),
                });
                messages.push(ChatMessage::tool(id, result));
            }

            // Execute dispatch calls in parallel.
            if !dispatch_calls.is_empty() {
                let dispatch_results =
                    self.run_dispatches(&dispatch_calls, &mut events).await;

                for (call_id, result) in dispatch_results {
                    match result {
                        Ok(sr) => {
                            let tool_result = serde_json::json!({
                                "agent": sr.name,
                                "result": sr.response,
                                "tools_used": sr.tool_calls.iter()
                                    .map(|tc| tc.function.name.as_str())
                                    .collect::<Vec<_>>(),
                            });
                            events.push(DispatchEvent::SpokeCompleted {
                                spoke: sr.name.clone(),
                                result: sr.response.clone(),
                            });
                            spoke_results.push(sr);
                            messages.push(ChatMessage::tool(call_id, tool_result.to_string()));
                        }
                        Err(err) => {
                            let tool_result = serde_json::json!({
                                "error": err.to_string(),
                            });
                            messages.push(ChatMessage::tool(call_id, tool_result.to_string()));
                        }
                    }
                }
            }
        }

        // If the hub used tools but produced no final text, force a text-only
        // LLM call so the user gets a response.
        let hub_used_tools = !spoke_results.is_empty()
            || events.iter().any(|e| matches!(e, DispatchEvent::HubToolCalled { .. }));
        if last_content.trim().is_empty() && hub_used_tools {
            let request =
                CompletionRequest::new(hub_model.to_string(), messages.clone());
            if let Ok(Ok(response)) = tokio::time::timeout(
                std::time::Duration::from_millis(orch.llm_timeout_ms),
                hub_provider.complete(request),
            )
            .await
            {
                last_content = response.message.text().unwrap_or_default().to_string();
            }
        }

        // Persist hub's final reply in transcript.
        if !last_content.trim().is_empty() {
            self.transcript
                .push(ChatMessage::assistant(last_content.clone()));
            orch.emit(&DispatchEvent::HubMessage {
                message: last_content.clone(),
            });
            events.push(DispatchEvent::HubMessage {
                message: last_content.clone(),
            });
        }

        Ok(DispatchTurn {
            reply: if last_content.trim().is_empty() {
                None
            } else {
                Some(last_content)
            },
            events,
            spoke_results,
            metrics: None,
            responding_agent: orch.hub.name().to_string(),
        })
    }

    // -- helpers --

    /// Run all dispatch tool calls concurrently and return `(call_id, result)`
    /// pairs in the same order as the input.
    async fn run_dispatches(
        &self,
        calls: &[&ToolCall],
        events: &mut Vec<DispatchEvent>,
    ) -> Vec<(String, Result<SpokeResult, AgentError>)> {
        // Collect the parsed dispatch parameters.
        struct Parsed {
            call_id: String,
            agent: String,
            task: String,
        }

        let mut parsed: Vec<Parsed> = Vec::with_capacity(calls.len());
        for call in calls {
            let call_id = call.id.clone().unwrap_or_default();
            let agent = call
                .function
                .arguments
                .get("agent")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            let task = call
                .function
                .arguments
                .get("task")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            parsed.push(Parsed { call_id, agent, task });
        }

        // Emit parallel event if > 1 dispatch.
        if parsed.len() > 1 {
            let names: Vec<String> = parsed.iter().map(|p| p.agent.clone()).collect();
            let evt = DispatchEvent::ParallelDispatch { spokes: names };
            self.orchestrator.emit(&evt);
            events.push(evt);
        }

        for p in &parsed {
            let evt = DispatchEvent::SpokeDispatched {
                spoke: p.agent.clone(),
                task: p.task.clone(),
            };
            self.orchestrator.emit(&evt);
            events.push(evt);
        }

        // Build futures.  Each future captures only shared references so they
        // can all run concurrently.
        let orch = self.orchestrator;
        let transcript = &self.transcript;

        let futures = parsed.iter().map(|p| {
            let provider = &orch.provider;
            let model = &orch.model;
            let timeout = orch.llm_timeout_ms;

            async move {
                let config = match orch.spokes.get(&p.agent) {
                    Some(c) => c,
                    None => {
                        return (
                            p.call_id.clone(),
                            Err(AgentError::UnknownAgent(p.agent.clone())),
                        );
                    }
                };

                let start = transcript.len().saturating_sub(config.context_window);
                let tail = &transcript[start..];

                let result =
                    execute_spoke(&p.agent, config, tail, &p.task, provider, model, timeout).await;

                (p.call_id.clone(), result)
            }
        });

        futures_util::future::join_all(futures).await
    }

    /// Return the last `n` messages from the transcript (excluding the current
    /// user message which was just appended).
    fn transcript_tail(&self, n: usize) -> &[ChatMessage] {
        let len = self.transcript.len();
        let start = len.saturating_sub(n);
        &self.transcript[start..]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChatMessage;

    // -- InputRoute matching tests --

    #[test]
    fn keywords_any_matches() {
        let route = InputRoute::keywords_any("enforcer", &["rule", "punishment"]);
        let orch = DispatchOrchestrator::new(
            Arc::new(MockProvider),
            "test",
            Agent::from_string("hub", "you are a hub"),
        )
        .register_spoke(
            "enforcer",
            SpokeConfig::new(Agent::from_string("enforcer", "you enforce")),
        )
        .define_input_route(route);

        assert_eq!(
            orch.match_input_routes(&[], "check my rules"),
            Some("enforcer")
        );
        assert_eq!(
            orch.match_input_routes(&[], "how are you?"),
            None
        );
    }

    #[test]
    fn keywords_all_requires_all() {
        let route = InputRoute::keywords_all("special", &["rule", "punishment"]);
        let orch = DispatchOrchestrator::new(
            Arc::new(MockProvider),
            "test",
            Agent::from_string("hub", "you are a hub"),
        )
        .register_spoke(
            "special",
            SpokeConfig::new(Agent::from_string("special", "you are special")),
        )
        .define_input_route(route);

        assert_eq!(
            orch.match_input_routes(&[], "rule and punishment"),
            Some("special")
        );
        assert_eq!(
            orch.match_input_routes(&[], "just a rule"),
            None
        );
    }

    #[test]
    fn first_route_wins() {
        let orch = DispatchOrchestrator::new(
            Arc::new(MockProvider),
            "test",
            Agent::from_string("hub", "hub"),
        )
        .register_spoke("a", SpokeConfig::new(Agent::from_string("a", "a")))
        .register_spoke("b", SpokeConfig::new(Agent::from_string("b", "b")))
        .define_input_route(InputRoute::keywords_any("a", &["rule"]))
        .define_input_route(InputRoute::keywords_any("b", &["rule"]));

        // First match wins.
        assert_eq!(orch.match_input_routes(&[], "check my rule"), Some("a"));
    }

    #[test]
    fn route_skips_unregistered_spoke() {
        let orch = DispatchOrchestrator::new(
            Arc::new(MockProvider),
            "test",
            Agent::from_string("hub", "hub"),
        )
        .define_input_route(InputRoute::keywords_any("ghost", &["rule"]));

        // Target "ghost" isn't registered, so the route is skipped.
        assert_eq!(orch.match_input_routes(&[], "check my rule"), None);
    }

    // Minimal mock provider for compile-only / routing tests.
    struct MockProvider;

    #[async_trait]
    impl LLMProvider for MockProvider {
        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> Result<crate::CompletionResponse, LLMError> {
            Ok(crate::CompletionResponse {
                message: ChatMessage::assistant("mock".to_string()),
                usage: None,
                reasoning: None,
            })
        }

        fn capabilities(&self) -> crate::ProviderCapabilities {
            crate::ProviderCapabilities::default()
        }

        fn name(&self) -> &'static str {
            "mock"
        }
    }
}
