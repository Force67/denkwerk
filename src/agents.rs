use std::{
    collections::HashMap,
    fmt,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use handlebars::Handlebars;
use serde::{Deserialize, Serialize};

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

impl AgentAction {
    fn from_response(content: &str) -> Self {
        if let Some(action) = Self::parse_json_block(content) {
            return action;
        }

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
        if let Ok(action) = serde_json::from_str::<ActionEnvelope>(content) {
            return Some(action.into());
        }

        let fenced = extract_json_from_fenced_block(content)?;
        serde_json::from_str::<ActionEnvelope>(&fenced)
            .ok()
            .map(Into::into)
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
    event_callback: Option<Arc<dyn Fn(&HandoffEvent) + Send + Sync>>,
}

impl HandoffOrchestrator {
    pub fn new(provider: Arc<dyn LLMProvider>, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            agents: HashMap::new(),
            max_handoffs: Some(4),
            event_callback: None,
        }
    }

    pub fn register_agent(&mut self, agent: Agent) -> Option<Agent> {
        self.agents.insert(agent.name.clone(), agent)
    }

    pub fn with_max_handoffs(mut self, max_handoffs: Option<usize>) -> Self {
        self.max_handoffs = max_handoffs;
        self
    }

    pub fn agent(&self, name: &str) -> Option<&Agent> {
        self.agents.get(name)
    }

    pub fn with_event_callback(mut self, callback: impl Fn(&HandoffEvent) + Send + Sync + 'static) -> Self {
        self.event_callback = Some(Arc::new(callback));
        self
    }

    fn emit_event(&self, event: &HandoffEvent) {
        if let Some(callback) = &self.event_callback {
            callback(event);
        }
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

        loop {
            let agent = self
                .orchestrator
                .agents
                .get(&self.active_agent)
                .ok_or_else(|| AgentError::UnknownAgent(self.active_agent.clone()))?;

            let turn = agent
                .execute(self.orchestrator.provider.as_ref(), &self.orchestrator.model, &self.transcript)
                .await?;

            match turn.action {
                AgentAction::Respond { message } => {
                    let mut assistant = ChatMessage::assistant(message.clone());
                    assistant.name = Some(agent.name.clone());
                    self.transcript.push(assistant);
                    let event = HandoffEvent::Message {
                        agent: agent.name.clone(),
                        message: message.clone(),
                    };
                    self.orchestrator.emit_event(&event);
                    events.push(event);

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

                    if let Some(message) = message {
                        let mut assistant = ChatMessage::assistant(message.clone());
                        assistant.name = Some(agent.name.clone());
                        self.transcript.push(assistant);
                        let event = HandoffEvent::Message {
                            agent: agent.name.clone(),
                            message,
                        };
                        self.orchestrator.emit_event(&event);
                        events.push(event);
                    }

                    if !self.orchestrator.agents.contains_key(&target) {
                        return Err(AgentError::UnknownAgent(target));
                    }

                    let event = HandoffEvent::HandOff {
                        from: agent.name.clone(),
                        to: target.clone(),
                    };
                    self.orchestrator.emit_event(&event);
                    events.push(event);

                    self.active_agent = target;
                    continue;
                }
                AgentAction::Complete { message } => {
                    if let Some(message) = message.clone() {
                        let mut assistant = ChatMessage::assistant(message.clone());
                        assistant.name = Some(agent.name.clone());
                        self.transcript.push(assistant);
                        let event = HandoffEvent::Message {
                            agent: agent.name.clone(),
                            message,
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
}
