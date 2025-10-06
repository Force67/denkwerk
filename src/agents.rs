use std::{
    collections::HashMap,
    fmt,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use handlebars::Handlebars;
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use tokio::time;

use crate::{
    functions::{FunctionRegistry, ToolChoice},
    types::{ChatMessage, CompletionRequest},
    LLMError, LLMProvider,
};

#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("unknown agent: {0}")]
    UnknownAgent(String),
    #[error("template file not found: {0}")]
    TemplateNotFound(PathBuf),
    #[error("template error: {0}")]
    Template(#[from] handlebars::TemplateError),
    #[error("template render error: {0}")]
    TemplateRender(#[from] handlebars::RenderError),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("maximum handoffs reached")]
    MaxHandoffsReached,
    #[error("maximum rounds reached")]
    MaxRoundsReached,
    #[error("no agents registered")]
    NoAgentsRegistered,
    #[error("manager produced invalid decision: {0}")]
    InvalidManagerDecision(String),
    #[error("provider call timed out")]
    ProviderTimeout,
    #[error(transparent)]
    Provider(#[from] LLMError),
}

#[derive(Clone)]
pub struct Agent {
    name: String,
    description: Option<String>,
    instructions: String,
    functions: Option<Arc<FunctionRegistry>>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<u32>,
    tool_choice: Option<ToolChoice>,
    provider_override: Option<Arc<dyn LLMProvider>>,
    model_override: Option<String>,
}

impl fmt::Debug for Agent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Agent")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("has_functions", &self.functions.is_some())
            .field("temperature", &self.temperature)
            .field("top_p", &self.top_p)
            .field("max_tokens", &self.max_tokens)
            .finish()
    }
}

impl Agent {
    pub fn from_string(name: impl Into<String>, instructions: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            instructions: instructions.into(),
            functions: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            tool_choice: None,
            provider_override: None,
            model_override: None,
        }
    }

    pub fn from_handlebars_file<T: Serialize>(
        name: impl Into<String>,
        template_path: impl AsRef<Path>,
        data: &T,
    ) -> Result<Self, AgentError> {
        let template_path = template_path.as_ref();
        let template = fs::read_to_string(template_path)
            .map_err(|_| AgentError::TemplateNotFound(template_path.to_path_buf()))?;
        let hb = Handlebars::new();
        let rendered = hb.render_template(&template, data)?;

        Ok(Self::from_string(name, rendered))
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn instructions(&self) -> &str {
        &self.instructions
    }

    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn with_function_registry(mut self, registry: Arc<FunctionRegistry>) -> Self {
        self.functions = Some(registry);
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    pub fn with_provider(mut self, provider: Arc<dyn LLMProvider>) -> Self {
        self.provider_override = Some(provider);
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model_override = Some(model.into());
        self
    }

    pub(crate) async fn execute(
        &self,
        provider: &(dyn LLMProvider + Send + Sync),
        model: &str,
        history: &[ChatMessage],
    ) -> Result<AgentTurn, LLMError> {
        let mut messages = Vec::with_capacity(history.len() + 1);
        messages.push(ChatMessage::system(self.instructions.clone()));
        messages.extend(history.iter().cloned());

        let active_provider: &(dyn LLMProvider + Send + Sync) = match &self.provider_override {
            Some(custom) => custom.as_ref(),
            None => provider,
        };

        let target_model = self.model_override.as_deref().unwrap_or(model);

        let mut request = CompletionRequest::new(target_model.to_string(), messages);

        if let Some(max_tokens) = self.max_tokens {
            request = request.with_max_tokens(max_tokens);
        }

        if let Some(temperature) = self.temperature {
            request = request.with_temperature(temperature);
        }

        if let Some(top_p) = self.top_p {
            request = request.with_top_p(top_p);
        }

        if let Some(functions) = &self.functions {
            request = request.with_function_registry(functions.as_ref());
        }

        if let Some(tool_choice) = &self.tool_choice {
            request = request.with_tool_choice(tool_choice.clone());
        }

        let response = active_provider.complete(request).await?;
        let content = response.message.text().unwrap_or_default();
        let action = AgentAction::from_response(content);

        Ok(AgentTurn { action })
    }
}

#[derive(Debug, Clone)]
pub enum AgentAction {
    Respond { message: String },
    HandOff { target: String, message: Option<String> },
    Complete { message: Option<String> },
}

// Natural-language cues for handoff/complete
static RE_HANDOFF: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?ix)
        \b
        (?:hand[\s-]*off|handoff|transfer|delegate|connect|route)\b
        (?:[^A-Za-z0-9@]+(?:to|with)\b)?
        [^A-Za-z0-9@]*    # optional punctuation/space
        (?:agent|assistant|team|specialist|@)?\s*
        (?P<target>[A-Za-z0-9_.\- ]{1,64})
        "
    )
    .unwrap()
});

static RE_COMPLETE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)\b(done|complete|completed|finish(?:ed)?|that'?s all|all set|nothing further)\b").unwrap());

fn normalize_agent_key(s: &str) -> String {
    s.trim().to_lowercase()
}

impl AgentAction {
    fn from_response(content: &str) -> Self {
        // Try JSON parsing first (handles both pure JSON and mixed content)
        if let Some(action) = Self::parse_json_block(content) {
            return action;
        }

        // NL handoff (e.g., "handoff to Travel", "transfer to @billing")
        if let Some(caps) = RE_HANDOFF.captures(content) {
            if let Some(m) = caps.name("target") {
                let mut target = m.as_str().trim().to_string();
                target = target
                    .trim_matches(|c: char| c == '@' || c.is_ascii_punctuation())
                    .to_string();
                if !target.is_empty() {
                    return AgentAction::HandOff {
                        target,
                        message: None,
                    };
                }
            }
        }

        // NL completion (e.g., "done", "that's all")
        if RE_COMPLETE.is_match(content) {
            let msg = content.trim();
            let message = if msg.is_empty() { None } else { Some(msg.to_string()) };
            return AgentAction::Complete { message };
        }

        // Default to responding with the entire content
        let trimmed = content.trim();
        if trimmed.is_empty() {
            return AgentAction::Respond {
                message: String::new(),
            };
        }

        AgentAction::Respond {
            message: trimmed.to_string(),
        }
    }

    fn parse_json_block(content: &str) -> Option<Self> {
        // Try parsing the entire content as JSON first
        if let Ok(action) = serde_json::from_str::<ActionEnvelope>(content.trim()) {
            return Some(action.into());
        }

        // Try extracting JSON from fenced code blocks
        if let Some(fenced) = extract_json_from_fenced_block(content) {
            if let Ok(action) = serde_json::from_str::<ActionEnvelope>(&fenced) {
                return Some(action.into());
            }
        }

        // Try extracting JSON objects from mixed text content
        if let Some(json_str) = extract_json_from_mixed_content(content) {
            if let Ok(action) = serde_json::from_str::<ActionEnvelope>(&json_str) {
                return Some(action.into());
            }
        }

        None
    }

    pub fn message(&self) -> Option<&str> {
        match self {
            AgentAction::Respond { message } => Some(message.as_str()),
            AgentAction::HandOff { message, .. } => message.as_deref(),
            AgentAction::Complete { message } => message.as_deref(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
enum ActionEnvelope {
    #[serde(alias = "respond", alias = "reply")]
    Respond {
        #[serde(alias = "response", alias = "text")]
        message: String,
    },
    #[serde(alias = "hand_off", alias = "handoff")]
    HandOff {
        #[serde(alias = "to", alias = "target_agent")]
        target: String,
        #[serde(default)]
        #[serde(alias = "message", alias = "note", alias = "reason")]
        message: Option<String>,
    },
    #[serde(alias = "complete", alias = "done")]
    Complete {
        #[serde(default)]
        #[serde(alias = "message", alias = "response", alias = "text")]
        message: Option<String>,
    },
}

impl From<ActionEnvelope> for AgentAction {
    fn from(value: ActionEnvelope) -> Self {
        match value {
            ActionEnvelope::Respond { message } => AgentAction::Respond { message },
            ActionEnvelope::HandOff { target, message } => AgentAction::HandOff { target, message },
            ActionEnvelope::Complete { message } => AgentAction::Complete { message },
        }
    }
}

fn extract_json_from_fenced_block(content: &str) -> Option<String> {
    let start = content.find("```json").or_else(|| content.find("```"))?;
    let remainder = &content[start..];
    let after_language = remainder.find('\n')?;
    let body = &remainder[after_language + 1..];
    let end = body.find("```")?;
    Some(body[..end].trim().to_string())
}

// Quote/escape-aware extractor: last complete top-level JSON object
fn extract_json_from_mixed_content(content: &str) -> Option<String> {
    let bytes = content.as_bytes();
    let mut start_pos = None;
    let mut end_pos = None;
    let mut depth: i32 = 0;

    let mut in_str = false;
    let mut escaped = false;

    for (i, &b) in bytes.iter().enumerate() {
        if in_str {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        } else if b == b'"' {
            in_str = true;
            continue;
        }

        match b {
            b'{' => {
                if depth == 0 {
                    start_pos = Some(i);
                }
                depth += 1;
            }
            b'}' => {
                if depth > 0 {
                    depth -= 1;
                    if depth == 0 && start_pos.is_some() {
                        end_pos = Some(i + 1);
                    }
                }
            }
            _ => {}
        }
    }

    if let (Some(s), Some(e)) = (start_pos, end_pos) {
        let candidate = &content[s..e];
        if candidate.contains("\"action\"") {
            return Some(candidate.to_string());
        }
    }
    None
}

#[derive(Debug)]
pub(crate) struct AgentTurn {
    pub(crate) action: AgentAction,
}

#[derive(Debug, Clone)]
pub enum HandoffEvent {
    Message { agent: String, message: String },
    HandOff { from: String, to: String },
    Completed { agent: String },
}

#[derive(Debug, Clone)]
pub struct HandoffTurn {
    pub reply: Option<String>,
    pub events: Vec<HandoffEvent>,
}

pub struct HandoffOrchestrator {
    provider: Arc<dyn LLMProvider>,
    model: String,
    agents: HashMap<String, Agent>,
    max_handoffs: Option<usize>,
    max_rounds: usize,
    llm_timeout_ms: u64,
    event_callback: Option<Arc<dyn Fn(&HandoffEvent) + Send + Sync>>,
}

impl HandoffOrchestrator {
    pub fn new(provider: Arc<dyn LLMProvider>, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            agents: HashMap::new(),
            max_handoffs: Some(4),
            max_rounds: 32,
            llm_timeout_ms: 60_000,
            event_callback: None,
        }
    }

    pub fn register_agent(&mut self, agent: Agent) -> Option<Agent> {
        let previous = self.agents.insert(agent.name.clone(), agent);
        self.refresh_handoff_instructions();
        previous
    }

    fn create_handoff_agent(&self, mut agent: Agent) -> Agent {
        // Kept for API compatibility, but actual roster injection happens in refresh_handoff_instructions()
        let available_agents: Vec<&str> = self.agents.keys().map(|s| s.as_str()).collect();

        let has_handoff_instructions = agent.instructions.to_lowercase().contains("delegate")
            || agent.instructions.to_lowercase().contains("hand off")
            || agent.instructions.to_lowercase().contains("coordinator");

        let handoff_instructions = if has_handoff_instructions {
            format!(
                r#"
{}

Available agents: {}

Just respond naturally - the system handles the handoffs automatically based on what you say."#,
                agent.instructions,
                available_agents.join(", ")
            )
        } else if available_agents.is_empty() {
            format!(
                r#"
{}

You can respond naturally to the user. When you want to complete the conversation, use: {{"action": "complete", "message": "final response"}}"#,
                agent.instructions
            )
        } else {
            format!(
                r#"
{}

You can respond naturally to the user. The system will automatically detect when you want to:
- Hand off to another agent (mention "hand off", "transfer", "connect", etc.)
- Complete the task (mention "complete", "done", "finished", etc.)

Available agents: {}

Just respond naturally - the system handles the handoffs automatically based on what you say."#,
                agent.instructions,
                available_agents.join(", ")
            )
        };

        agent.instructions = handoff_instructions;
        agent
    }

    fn refresh_handoff_instructions(&mut self) {
        let roster: Vec<String> = self.agents.keys().cloned().collect();
        for agent in self.agents.values_mut() {
            let _has_handoff_instructions = agent.instructions.to_lowercase().contains("delegate")
                || agent.instructions.to_lowercase().contains("hand off")
                || agent.instructions.to_lowercase().contains("coordinator");

            let appendix = if roster.is_empty() {
                r#"
You can respond naturally to the user. When you want to complete the conversation, use: {"action": "complete", "message": "..."}"#
                    .to_string()
            } else {
                format!(
                    r#"

You can respond naturally to the user. The system will automatically detect when you want to:
- Hand off to another agent (mention "hand off", "transfer", "connect", etc.)
- Complete the task (mention "complete", "done", "finished", etc.)

Available agents: {}

Just respond naturally - the system handles the handoffs automatically based on what you say."#,
                    roster.join(", ")
                )
            };

            if !agent.instructions.contains("Available agents:") {
                agent.instructions = format!("{}\n{}", agent.instructions, appendix);
            }
        }
    }

    pub fn with_max_handoffs(mut self, max_handoffs: Option<usize>) -> Self {
        self.max_handoffs = max_handoffs;
        self
    }

    pub fn with_max_rounds(mut self, rounds: usize) -> Self {
        self.max_rounds = rounds;
        self
    }

    pub fn with_llm_timeout_ms(mut self, ms: u64) -> Self {
        self.llm_timeout_ms = ms;
        self
    }

    pub fn agent(&self, name: &str) -> Option<&Agent> {
        self.agents.get(name)
    }

    pub fn with_event_callback(
        mut self,
        callback: impl Fn(&HandoffEvent) + Send + Sync + 'static,
    ) -> Self {
        self.event_callback = Some(Arc::new(callback));
        self
    }

    fn emit_event(&self, event: &HandoffEvent) {
        if let Some(callback) = &self.event_callback {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (callback)(event)));
        }
    }

    fn resolve_target(&self, current: &str, raw_target: &str) -> Result<String, AgentError> {
        let want = normalize_agent_key(raw_target.trim().trim_start_matches('@'));
        if want.is_empty() {
            return Err(AgentError::UnknownAgent(raw_target.to_string()));
        }

        // 1) exact case-insensitive
        if let Some((k, _)) = self
            .agents
            .iter()
            .find(|(k, _)| normalize_agent_key(k) == want)
        {
            if normalize_agent_key(k) == normalize_agent_key(current) {
                return Err(AgentError::InvalidManagerDecision(
                    "self handoff not allowed".into(),
                ));
            }
            return Ok(k.to_string());
        }

        // 2) prefix
        if let Some((k, _)) = self
            .agents
            .iter()
            .find(|(k, _)| normalize_agent_key(k).starts_with(&want))
        {
            if normalize_agent_key(k) == normalize_agent_key(current) {
                return Err(AgentError::InvalidManagerDecision(
                    "self handoff not allowed".into(),
                ));
            }
            return Ok(k.to_string());
        }

        // 3) fuzzy (accept close matches)
        let mut best: Option<(&String, usize)> = None;
        for (k, _) in &self.agents {
            let d = strsim::levenshtein(&want, &normalize_agent_key(k));
            if best.map_or(true, |(_, bd)| d < bd) {
                best = Some((k, d));
            }
        }
        if let Some((k, dist)) = best {
            if dist <= 3 {
                if normalize_agent_key(k) == normalize_agent_key(current) {
                    return Err(AgentError::InvalidManagerDecision(
                        "self handoff not allowed".into(),
                    ));
                }
                return Ok(k.clone());
            }
        }

        Err(AgentError::UnknownAgent(raw_target.to_string()))
    }

    pub fn session<'a>(
        &'a self,
        initial_agent: impl Into<String>,
    ) -> Result<HandoffSession<'a>, AgentError> {
        let agent_name = initial_agent.into();
        if !self.agents.contains_key(&agent_name) {
            return Err(AgentError::UnknownAgent(agent_name));
        }

        Ok(HandoffSession {
            orchestrator: self,
            transcript: Vec::new(),
            active_agent: agent_name,
            remaining_handoffs: self.max_handoffs,
        })
    }
}

pub struct HandoffSession<'a> {
    orchestrator: &'a HandoffOrchestrator,
    transcript: Vec<ChatMessage>,
    active_agent: String,
    remaining_handoffs: Option<usize>,
}

impl<'a> HandoffSession<'a> {
    pub fn active_agent(&self) -> &str {
        &self.active_agent
    }

    pub fn transcript(&self) -> &[ChatMessage] {
        &self.transcript
    }

    pub fn set_history(&mut self, history: Vec<ChatMessage>) {
        self.transcript = history;
    }

    pub fn set_max_handoffs(&mut self, max: Option<usize>) {
        self.remaining_handoffs = max;
    }

    pub fn max_handoffs(&self) -> Option<usize> {
        self.remaining_handoffs
    }

    pub async fn send(&mut self, user_input: impl Into<String>) -> Result<HandoffTurn, AgentError> {
        self.transcript.push(ChatMessage::user(user_input.into()));
        let mut events = Vec::new();
        let mut rounds = 0usize;

        loop {
            rounds += 1;
            if rounds > self.orchestrator.max_rounds {
                return Err(AgentError::MaxRoundsReached);
            }

            let agent = self
                .orchestrator
                .agents
                .get(&self.active_agent)
                .ok_or_else(|| AgentError::UnknownAgent(self.active_agent.clone()))?;

            let fut = agent.execute(
                self.orchestrator.provider.as_ref(),
                &self.orchestrator.model,
                &self.transcript,
            );

            let turn = match time::timeout(
                std::time::Duration::from_millis(self.orchestrator.llm_timeout_ms),
                fut,
            )
            .await
            {
                Ok(res) => res?,
                Err(_) => return Err(AgentError::ProviderTimeout),
            };

            match turn.action {
                AgentAction::Respond { message } => {
                    if !message.trim().is_empty() {
                        let mut assistant = ChatMessage::assistant(message.clone());
                        assistant.name = Some(agent.name.clone());
                        self.transcript.push(assistant);
                        let event = HandoffEvent::Message {
                            agent: agent.name.clone(),
                            message: message.clone(),
                        };
                        self.orchestrator.emit_event(&event);
                        events.push(event);
                    }

                    return Ok(HandoffTurn {
                        reply: Some(message),
                        events,
                    });
                }
                AgentAction::HandOff { target, message } => {
                    if let Some(limit) = self.remaining_handoffs.as_mut() {
                        if *limit == 0 {
                            return Err(AgentError::MaxHandoffsReached);
                        }
                        *limit -= 1;
                    }

                    if let Some(msg) = message.filter(|m| !m.trim().is_empty()) {
                        let mut assistant = ChatMessage::assistant(msg.clone());
                        assistant.name = Some(agent.name.clone());
                        self.transcript.push(assistant);
                        let event = HandoffEvent::Message {
                            agent: agent.name.clone(),
                            message: msg,
                        };
                        self.orchestrator.emit_event(&event);
                        events.push(event);
                    }

                    let resolved = self
                        .orchestrator
                        .resolve_target(&self.active_agent, &target)?;

                    let event = HandoffEvent::HandOff {
                        from: agent.name.clone(),
                        to: resolved.clone(),
                    };
                    self.orchestrator.emit_event(&event);
                    events.push(event);

                    self.active_agent = resolved;
                    continue;
                }
                AgentAction::Complete { message } => {
                    if let Some(msg) = message.clone().filter(|m| !m.trim().is_empty()) {
                        let mut assistant = ChatMessage::assistant(msg.clone());
                        assistant.name = Some(agent.name.clone());
                        self.transcript.push(assistant);
                        let event = HandoffEvent::Message {
                            agent: agent.name.clone(),
                            message: msg,
                        };
                        self.orchestrator.emit_event(&event);
                        events.push(event);
                    }

                    let event = HandoffEvent::Completed {
                        agent: agent.name.clone(),
                    };
                    self.orchestrator.emit_event(&event);
                    events.push(event);

                    return Ok(HandoffTurn {
                        reply: message,
                        events,
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AgentAction;

    #[test]
    fn parses_inline_json() {
        let json = r#"{"action":"hand_off","target":"travel","message":"Need itinerary."}"#;
        if let AgentAction::HandOff { target, message } = AgentAction::from_response(json) {
            assert_eq!(target, "travel");
            assert_eq!(message.as_deref(), Some("Need itinerary."));
        } else {
            panic!("expected handoff action");
        }
    }

    #[test]
    fn parses_fenced_json() {
        let json = r#"```json
{"action":"respond","message":"All set."}
```"#;
        if let AgentAction::Respond { message } = AgentAction::from_response(json) {
            assert_eq!(message, "All set.");
        } else {
            panic!("expected respond action");
        }
    }

    #[test]
    fn falls_back_to_plain_text() {
        let content = "Hello there";
        if let AgentAction::Respond { message } = AgentAction::from_response(content) {
            assert_eq!(message, "Hello there");
        } else {
            panic!("expected respond action");
        }
    }

    #[test]
    fn parses_json_handoff() {
        let content =
            r#"{"action": "handoff", "target": "travel", "message": "Need itinerary help"}"#;
        if let AgentAction::HandOff { target, message } = AgentAction::from_response(content) {
            assert_eq!(target, "travel");
            assert_eq!(message.unwrap(), "Need itinerary help");
        } else {
            panic!("expected handoff action");
        }
    }

    #[test]
    fn parses_json_complete() {
        let content = r#"{"action": "complete", "message": "All done"}"#;
        if let AgentAction::Complete { message } = AgentAction::from_response(content) {
            assert_eq!(message.unwrap(), "All done");
        } else {
            panic!("expected complete action");
        }
    }

    #[test]
    fn falls_back_to_response_for_plain_text() {
        let content = "This is just a regular response from the agent.";
        if let AgentAction::Respond { message } = AgentAction::from_response(content) {
            assert_eq!(message, content);
        } else {
            panic!("expected respond action");
        }
    }

    #[test]
    fn parses_json_from_mixed_content() {
        let content = r#"Sure, I'll help with that!

{"action": "handoff", "target": "travel", "message": "Please help with travel planning"}"#;
        if let AgentAction::HandOff { target, message } = AgentAction::from_response(content) {
            assert_eq!(target, "travel");
            assert_eq!(message.unwrap(), "Please help with travel planning");
        } else {
            panic!("expected handoff action");
        }
    }

    #[test]
    fn nl_handoff_variants() {
        let samples = [
            "Can you hand off to Travel?",
            "please transfer to @billing",
            "delegate with data scientist",
            "connect me to Research team",
        ];
        for s in samples {
            if let AgentAction::HandOff { target, .. } = AgentAction::from_response(s) {
                assert!(
                    !target.trim().is_empty(),
                    "target should be present for '{s}'"
                );
            } else {
                panic!("expected NL handoff for '{s}'");
            }
        }
    }

    #[test]
    fn nl_complete_variants() {
        for s in ["done", "completed", "that's all", "all set", "we are finished here"] {
            match AgentAction::from_response(s) {
                AgentAction::Complete { .. } => {}
                _ => panic!("expected completion for '{s}'"),
            }
        }
    }

    #[test]
    fn json_extractor_handles_braces_in_strings() {
        let content =
            r#"noise {"action":"respond","message":"brace in value: {ok}"} trailing"#;
        if let AgentAction::Respond { message } = AgentAction::from_response(content) {
            assert!(message.contains("brace in value"));
        } else {
            panic!("expected respond action");
        }
    }
}
