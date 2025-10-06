use std::{
    collections::HashMap,
    sync::Arc,
};

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::time;

use crate::{
    functions::{FunctionRegistry, ToolChoice, json_schema_for, to_value},
    types::ChatMessage,
    Agent, AgentError, LLMError, LLMProvider,
};

/// What to do if a rule matches
#[derive(Debug, Clone)]
pub struct HandoffDirective {
    pub target: String,
    pub message: Option<String>,
}

/// How to match; you can add more variants over time
#[derive(Clone)]
pub enum HandoffMatcher {
    /// Simple keyword presence (any/ all)
    KeywordsAny(Vec<String>),
    KeywordsAll(Vec<String>),

    /// Regex over the assistant text
    Regex(Regex),

    /// Full-power check: (transcript, last_assistant_text) -> Option<Directive>
    Predicate(Arc<dyn Fn(&[ChatMessage], &str) -> Option<HandoffDirective> + Send + Sync>),
}

/// One rule = matcher + target resolver (static or dynamic)
pub struct HandoffRule {
    pub matcher: HandoffMatcher,
    pub resolve: Arc<dyn Fn(&[ChatMessage], &str) -> Option<HandoffDirective> + Send + Sync>,
}

impl HandoffRule {
    pub fn to(target: impl Into<String>, matcher: HandoffMatcher) -> Self {
        let target = target.into();
        Self {
            matcher,
            resolve: Arc::new(move |_t, _txt| {
                Some(HandoffDirective { target: target.clone(), message: None })
            }),
        }
    }
}

#[derive(Debug)]
pub struct HandoffTurn {
    pub reply: Option<String>,
    pub events: Vec<HandoffEvent>,
}

struct HandoffFunction;

#[async_trait::async_trait]
impl crate::functions::KernelFunction for HandoffFunction {
    fn definition(&self) -> crate::functions::FunctionDefinition {
        let mut def = crate::functions::FunctionDefinition::new("handoff")
            .with_description("Route the conversation to another agent. Use this whenever another specialist should take over.");
        def.add_parameter(crate::functions::FunctionParameter::new("to", json_schema_for::<String>()).with_description("Target agent name (e.g., travel, weather)"));
        def.add_parameter(crate::functions::FunctionParameter::new("message", json_schema_for::<Option<String>>()).optional().with_description("Optional handoff note"));
        def
    }

    async fn invoke(&self, arguments: &Value) -> Result<Value, LLMError> {
        let to = arguments.get("to").and_then(|v| v.as_str()).unwrap_or("");
        let message = arguments.get("message").and_then(|v| v.as_str()).map(|s| s.to_string());
        Ok(to_value(ActionEnvelope::HandOff {
            target: to.to_string(),
            message
        }))
    }
}

struct CompleteFunction;

#[async_trait::async_trait]
impl crate::functions::KernelFunction for CompleteFunction {
    fn definition(&self) -> crate::functions::FunctionDefinition {
        let mut def = crate::functions::FunctionDefinition::new("complete")
            .with_description("Mark the task as complete and return the final answer.");
        def.add_parameter(crate::functions::FunctionParameter::new("message", json_schema_for::<Option<String>>()).optional().with_description("Optional final response"));
        def
    }

    async fn invoke(&self, arguments: &Value) -> Result<Value, LLMError> {
        let message = arguments.get("message").and_then(|v| v.as_str()).map(|s| s.to_string());
        Ok(to_value(ActionEnvelope::Complete { message }))
    }
}

#[derive(Debug)]
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
    pub fn from_response(content: &str) -> Self {
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

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ActionEnvelope {
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
    pub(crate) tool_calls: Vec<crate::functions::ToolCall>,
}

#[derive(Debug, Clone)]
pub enum HandoffEvent {
    Message { agent: String, message: String },
    HandOff { from: String, to: String },
    Completed { agent: String },
}

pub struct HandoffOrchestrator {
    provider: Arc<dyn LLMProvider>,
    model: String,
    agents: HashMap<String, Agent>,
    rules: Vec<HandoffRule>,
    aliases: HashMap<String, String>,
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
            rules: Vec::new(),
            aliases: HashMap::new(),
            max_handoffs: Some(4),
            max_rounds: 32,
            llm_timeout_ms: 60_000,
            event_callback: None,
        }
    }

    pub fn register_agent(&mut self, agent: Agent) -> Option<Agent> {
        let name = agent.name().to_string();
        let previous = self.agents.insert(name, agent);
        self.refresh_handoff_instructions();
        previous
    }

    pub fn define_handoff(&mut self, rule: HandoffRule) -> &mut Self {
        self.rules.push(rule);
        self
    }

    pub fn add_alias(&mut self, alias: impl Into<String>, target: impl Into<String>) -> &mut Self {
        self.aliases.insert(alias.into(), target.into());
        self
    }

    fn internal_tools(&self) -> FunctionRegistry {
        let mut reg = FunctionRegistry::new();

        reg.register(Arc::new(HandoffFunction) as Arc<dyn crate::functions::KernelFunction>);
        reg.register(Arc::new(CompleteFunction) as Arc<dyn crate::functions::KernelFunction>);

        reg
    }

    fn match_rules(&self, transcript: &[ChatMessage], last_message: &str) -> Option<HandoffDirective> {
        for rule in &self.rules {
            let matches = match &rule.matcher {
                HandoffMatcher::KeywordsAny(keywords) => {
                    keywords.iter().any(|kw| last_message.to_lowercase().contains(&kw.to_lowercase()))
                }
                HandoffMatcher::KeywordsAll(keywords) => {
                    keywords.iter().all(|kw| last_message.to_lowercase().contains(&kw.to_lowercase()))
                }
                HandoffMatcher::Regex(regex) => {
                    regex.is_match(last_message)
                }
                HandoffMatcher::Predicate(pred) => {
                    pred(transcript, last_message).is_some()
                }
            };

            if matches {
                return (rule.resolve)(transcript, last_message);
            }
        }
        None
    }

    fn create_handoff_agent(&self, agent: Agent) -> Agent {
        // Kept for API compatibility, but actual roster injection happens in refresh_handoff_instructions()
        let available_agents: Vec<&str> = self.agents.keys().map(|s| s.as_str()).collect();

        let has_handoff_instructions = agent.instructions().to_lowercase().contains("delegate")
            || agent.instructions().to_lowercase().contains("hand off")
            || agent.instructions().to_lowercase().contains("coordinator");

        let handoff_instructions = if has_handoff_instructions {
            format!(
                r#"
{}
 
Available agents: {}
 
Just respond naturally - the system handles the handoffs automatically based on what you say."#,
                agent.instructions(),
                available_agents.join(", ")
            )
        } else if available_agents.is_empty() {
            format!(
                r#"
{}
 
You can respond naturally to the user. When you want to complete the conversation, use: {{"action": "complete", "message": "final response"}}"#,
                agent.instructions()
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
                agent.instructions(),
                available_agents.join(", ")
            )
        };

        // Note: We can't modify agent.instructions directly since it's private
        // This method is kept for API compatibility but the actual modification
        // happens in refresh_handoff_instructions()
        let _ = handoff_instructions;
        agent
    }

    fn refresh_handoff_instructions(&mut self) {
        // Note: Since Agent.instructions is private, we can't modify it here.
        // The original logic would append agent roster information to instructions,
        // but this needs to be redesigned since instructions are immutable after Agent creation.
        // For now, this method is kept for API compatibility but doesn't modify anything.
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

        // Check aliases first
        let resolved_want = self.aliases.get(&want).unwrap_or(&want).clone();

        // 1) exact case-insensitive
        if let Some((k, _)) = self
            .agents
            .iter()
            .find(|(k, _)| normalize_agent_key(k) == resolved_want)
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
            .find(|(k, _)| normalize_agent_key(k).starts_with(&resolved_want))
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
            let d = strsim::levenshtein(&resolved_want, &normalize_agent_key(k));
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

            let internal_tools = self.orchestrator.internal_tools();
            let fut = agent.execute_with_tools(
                self.orchestrator.provider.as_ref(),
                &self.orchestrator.model,
                &self.transcript,
                Some(&internal_tools),
                Some(ToolChoice::auto()),
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

            let mut action = turn.action;

            // If action is Respond, run deterministic rules.
            if let AgentAction::Respond { ref message } = action {
                if let Some(dir) = self.orchestrator.match_rules(&self.transcript, message) {
                    action = AgentAction::HandOff { target: dir.target, message: dir.message };
                }
            }

            match action {
                AgentAction::Respond { message } => {
                    if !message.trim().is_empty() {
                        let mut assistant = ChatMessage::assistant(message.clone());
                        assistant.name = Some(agent.name().to_string());
                        self.transcript.push(assistant);
                        let event = HandoffEvent::Message {
                            agent: agent.name().to_string(),
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
                        assistant.name = Some(agent.name().to_string());
                        self.transcript.push(assistant);
                        let event = HandoffEvent::Message {
                            agent: agent.name().to_string(),
                            message: msg,
                        };
                        self.orchestrator.emit_event(&event);
                        events.push(event);
                    }

                    let resolved = self
                        .orchestrator
                        .resolve_target(&self.active_agent, &target)?;

                    let event = HandoffEvent::HandOff {
                        from: agent.name().to_string(),
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
                        assistant.name = Some(agent.name().to_string());
                        self.transcript.push(assistant);
                        let event = HandoffEvent::Message {
                            agent: agent.name().to_string(),
                            message: msg,
                        };
                        self.orchestrator.emit_event(&event);
                        events.push(event);
                    }

                    let event = HandoffEvent::Completed {
                        agent: agent.name().to_string(),
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
    use super::{AgentAction, HandoffMatcher, HandoffRule};
    use regex::Regex;

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

    #[test]
    fn handoff_rule_keywords_any() {
        let matcher = HandoffMatcher::KeywordsAny(vec!["flight".to_string(), "hotel".to_string()]);
        let rule = HandoffRule::to("travel", matcher);
        let transcript = vec![];
        let message = "I need to book a flight and hotel for my trip.";

        let directive = (rule.resolve)(&transcript, message);
        assert_eq!(directive.unwrap().target, "travel");
    }

    #[test]
    fn handoff_rule_regex() {
        let matcher = HandoffMatcher::Regex(Regex::new(r"weather").unwrap());
        let rule = HandoffRule::to("weather", matcher);
        let transcript = vec![];
        let message = "What's the weather like today?";

        let directive = (rule.resolve)(&transcript, message);
        assert_eq!(directive.unwrap().target, "weather");
    }
}