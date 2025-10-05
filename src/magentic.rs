use std::{
    collections::HashMap,
    fmt::Write,
    sync::Arc,
};

use serde::Deserialize;

use crate::{
    agents::{Agent, AgentAction, AgentError},
    types::{ChatMessage, MessageRole},
    LLMProvider,
};

/// Guides the multi-agent collaboration by emitting structured delegation commands.
#[derive(Clone)]
pub struct MagenticManager {
    agent: Agent,
}

impl MagenticManager {
    pub fn new(agent: Agent) -> Self {
        Self { agent }
    }

    /// Returns a manager that is configured with sensible defaults for JSON based delegation.
    pub fn standard() -> Self {
        let instructions = r#"
You coordinate a team of domain experts to complete the user's task.
Carefully review the task, the progress so far, and each agent's description before you answer.

Always respond with a single JSON object using one of these shapes:
- {"action":"delegate","target":"<agent name>","instructions":"<what the agent should do next>","progress_note":"<optional summary to share>"}
- {"action":"message","message":"<status update or clarifying question>"}
- {"action":"complete","result":"<final answer for the user>"}

Rules:
- Only delegate to agents listed in the roster.
- Make incremental progress. Break large tasks into focused instructions.
- Use the message action when you must ask the user for more information.
- Use the complete action only when you are confident the overall task is finished.
- Never include additional text outside the JSON object.
"#;

        Self::new(Agent::from_string("manager", instructions))
    }

    pub fn name(&self) -> &str {
        self.agent.name()
    }

    pub(crate) fn agent(&self) -> &Agent {
        &self.agent
    }
}

#[derive(Debug, Clone)]
pub enum MagenticDecision {
    Delegate {
        target: String,
        instructions: String,
        progress_note: Option<String>,
    },
    Message { content: String },
    Complete { result: String },
}

impl MagenticDecision {
    fn parse(content: &str) -> Result<Self, AgentError> {
        if let Some(envelope) = parse_json_envelope(content) {
            return Ok(match envelope {
                ManagerEnvelope::Delegate {
                    target,
                    instructions,
                    progress_note,
                } => MagenticDecision::Delegate {
                    target,
                    instructions,
                    progress_note,
                },
                ManagerEnvelope::Message { message } => MagenticDecision::Message { content: message },
                ManagerEnvelope::Complete { result } => MagenticDecision::Complete { result },
            });
        }

        let trimmed = content.trim();
        if trimmed.is_empty() {
            return Err(AgentError::InvalidManagerDecision("empty manager response".into()));
        }

        Ok(MagenticDecision::Message {
            content: trimmed.to_string(),
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
enum ManagerEnvelope {
    #[serde(alias = "delegate_agent", alias = "call_agent")]
    Delegate {
        #[serde(alias = "agent", alias = "target_agent")]
        target: String,
        #[serde(alias = "message", alias = "task", alias = "instruction")]
        instructions: String,
        #[serde(default)]
        #[serde(alias = "progress", alias = "note", alias = "summary")]
        progress_note: Option<String>,
    },
    #[serde(alias = "respond", alias = "status", alias = "say")]
    Message {
        #[serde(alias = "content", alias = "text")]
        message: String,
    },
    #[serde(alias = "final", alias = "finalize")]
    Complete {
        #[serde(alias = "message", alias = "response")]
        result: String,
    },
}

fn parse_json_envelope(content: &str) -> Option<ManagerEnvelope> {
    if let Ok(envelope) = serde_json::from_str::<ManagerEnvelope>(content) {
        return Some(envelope);
    }

    let fenced = extract_json_from_fenced_block(content)?;
    serde_json::from_str::<ManagerEnvelope>(&fenced).ok()
}

fn extract_json_from_fenced_block(content: &str) -> Option<String> {
    let start = content.find("```json").or_else(|| content.find("```"))?;
    let remainder = &content[start..];
    let line_break = remainder.find('\n')?;
    let after_language = &remainder[line_break + 1..];
    let end = after_language.find("```")?;
    Some(after_language[..end].trim().to_string())
}

#[derive(Debug, Clone)]
pub enum MagenticEvent {
    ManagerMessage { message: String },
    ManagerDelegation {
        target: String,
        instructions: String,
        progress_note: Option<String>,
    },
    AgentMessage { agent: String, message: String },
    AgentCompletion { agent: String, message: Option<String> },
    Completed { message: String },
}

#[derive(Debug, Clone)]
pub struct MagenticRun {
    pub final_result: Option<String>,
    pub events: Vec<MagenticEvent>,
    pub rounds: usize,
    pub transcript: Vec<ChatMessage>,
}

pub struct MagenticOrchestrator {
    provider: Arc<dyn LLMProvider>,
    model: String,
    manager: MagenticManager,
    roster: Vec<Agent>,
    agents: HashMap<String, Agent>,
    max_rounds: usize,
    event_callback: Option<Arc<dyn Fn(&MagenticEvent) + Send + Sync>>,
}

impl MagenticOrchestrator {
    pub fn new(
        provider: Arc<dyn LLMProvider>,
        model: impl Into<String>,
        manager: MagenticManager,
    ) -> Self {
        Self {
            provider,
            model: model.into(),
            manager,
            roster: Vec::new(),
            agents: HashMap::new(),
            max_rounds: 12,
            event_callback: None,
        }
    }

    pub fn register_agent(&mut self, agent: Agent) -> Result<(), AgentError> {
        let name = agent.name().to_string();
        if self.agents.contains_key(&name) {
            return Err(AgentError::InvalidManagerDecision(format!(
                "duplicate agent name '{name}'"
            )));
        }
        self.agents.insert(name.clone(), agent.clone());
        self.roster.push(agent);
        Ok(())
    }

    pub fn with_max_rounds(mut self, rounds: usize) -> Self {
        self.max_rounds = rounds.max(1);
        self
    }

    pub fn with_event_callback(mut self, callback: impl Fn(&MagenticEvent) + Send + Sync + 'static) -> Self {
        self.event_callback = Some(Arc::new(callback));
        self
    }

    fn emit_event(&self, event: &MagenticEvent) {
        if let Some(callback) = &self.event_callback {
            callback(event);
        }
    }

    pub async fn run(&self, task: impl Into<String>) -> Result<MagenticRun, AgentError> {
        let task = task.into();
        let mut transcript = vec![ChatMessage::user(task.clone())];
        let mut events = Vec::new();

        for round in 0..self.max_rounds {
            let manager_prompt = build_manager_prompt(
                &task,
                round + 1,
                &self.manager,
                &self.roster,
                &transcript,
            );

            let manager_messages = vec![ChatMessage::user(manager_prompt)];
            let manager_turn = self
                .manager
                .agent()
                .execute(self.provider.as_ref(), &self.model, &manager_messages)
                .await?;

            let manager_text = manager_turn
                .action
                .message()
                .map(|m| m.to_string())
                .ok_or_else(|| AgentError::InvalidManagerDecision("manager response missing content".into()))?;

            let decision = MagenticDecision::parse(&manager_text)?;

            match decision {
                MagenticDecision::Delegate {
                    target,
                    instructions,
                    progress_note,
                } => {
                    if let Some(note) = progress_note.clone() {
                        push_manager_message(&mut transcript, &self.manager, note.clone());
                        let event = MagenticEvent::ManagerMessage { message: note };
                        self.emit_event(&event);
                        events.push(event);
                    }

                    let agent = self
                        .agents
                        .get(&target)
                        .ok_or_else(|| AgentError::UnknownAgent(target.clone()))?
                        .clone();

                    push_manager_message(&mut transcript, &self.manager, instructions.clone());
                    let event = MagenticEvent::ManagerDelegation {
                        target: target.clone(),
                        instructions: instructions.clone(),
                        progress_note: progress_note.clone(),
                    };
                    self.emit_event(&event);
                    events.push(event);

                    let turn = agent
                        .execute(self.provider.as_ref(), &self.model, &transcript)
                        .await?;

                    match turn.action {
                        AgentAction::Respond { message } => {
                            push_agent_message(&mut transcript, &agent, &message);
                            let event = MagenticEvent::AgentMessage {
                                agent: agent.name().to_string(),
                                message,
                            };
                            self.emit_event(&event);
                            events.push(event);
                        }
                        AgentAction::HandOff { target: _, message } => {
                            let text = message.unwrap_or_default();
                            push_agent_message(&mut transcript, &agent, &text);
                            let event = MagenticEvent::AgentMessage {
                                agent: agent.name().to_string(),
                                message: text,
                            };
                            self.emit_event(&event);
                            events.push(event);
                        }
                        AgentAction::Complete { message } => {
                            if let Some(text) = message.clone() {
                                push_agent_message(&mut transcript, &agent, &text);
                            }
                            let event = MagenticEvent::AgentCompletion {
                                agent: agent.name().to_string(),
                                message,
                            };
                            self.emit_event(&event);
                            events.push(event);
                        }
                    }
                }
                MagenticDecision::Message { content } => {
                    push_manager_message(&mut transcript, &self.manager, content.clone());
                    let event = MagenticEvent::ManagerMessage { message: content };
                    self.emit_event(&event);
                    events.push(event);
                }
                MagenticDecision::Complete { result } => {
                    push_manager_message(&mut transcript, &self.manager, result.clone());
                    let event = MagenticEvent::Completed {
                        message: result.clone(),
                    };
                    self.emit_event(&event);
                    events.push(event);
                    return Ok(MagenticRun {
                        final_result: Some(result),
                        events,
                        rounds: round + 1,
                        transcript,
                    });
                }
            }
        }

        Err(AgentError::MaxRoundsReached)
    }
}

fn push_manager_message(transcript: &mut Vec<ChatMessage>, manager: &MagenticManager, content: String) {
    let mut message = ChatMessage::assistant(content);
    message.name = Some(manager.name().to_string());
    transcript.push(message);
}

fn push_agent_message(transcript: &mut Vec<ChatMessage>, agent: &Agent, content: &str) {
    let mut message = ChatMessage::assistant(content.to_string());
    message.name = Some(agent.name().to_string());
    transcript.push(message);
}

fn build_manager_prompt(
    task: &str,
    round: usize,
    manager: &MagenticManager,
    roster: &[Agent],
    transcript: &[ChatMessage],
) -> String {
    let mut prompt = String::new();
    let _ = writeln!(prompt, "You are {} coordinating a collaboration.", manager.name());
    let _ = writeln!(prompt, "Task: {task}");
    let _ = writeln!(prompt, "Round: {round}");
    let _ = writeln!(prompt, "Agent roster:");
    for agent in roster {
        let description = agent
            .description()
            .map(|d| d.to_string())
            .unwrap_or_else(|| "No description provided.".to_string());
        let _ = writeln!(prompt, "- {}: {}", agent.name(), description);
    }

    let _ = writeln!(prompt, "\nConversation so far:");
    if transcript.is_empty() {
        let _ = writeln!(prompt, "(no messages yet)");
    } else {
        for message in transcript {
            let speaker = match (&message.role, &message.name) {
                (MessageRole::User, _) => "User".to_string(),
                (MessageRole::System, _) => "System".to_string(),
                (MessageRole::Assistant, Some(name)) => format!("Assistant::{name}"),
                (MessageRole::Assistant, None) => "Assistant".to_string(),
                (MessageRole::Tool, Some(name)) => format!("Tool::{name}"),
                (MessageRole::Tool, None) => "Tool".to_string(),
            };
            let text = message.text().unwrap_or_default();
            let _ = writeln!(prompt, "- {speaker}: {text}");
        }
    }

    let _ = writeln!(prompt, "\nProduce your JSON decision now.");
    prompt
}

#[cfg(test)]
mod tests {
    use super::{extract_json_from_fenced_block, MagenticDecision};

    #[test]
    fn parses_delegation() {
        let json = r#"{"action":"delegate","target":"Research","instructions":"Find usage stats."}"#;
        match MagenticDecision::parse(json).expect("decision") {
            MagenticDecision::Delegate { target, instructions, .. } => {
                assert_eq!(target, "Research");
                assert_eq!(instructions, "Find usage stats.");
            }
            _ => panic!("expected delegation"),
        }
    }

    #[test]
    fn parses_fenced_complete() {
        let response = r#"```json
{"action":"complete","result":"All done."}
```"#;
        match MagenticDecision::parse(response).expect("decision") {
            MagenticDecision::Complete { result } => assert_eq!(result, "All done."),
            _ => panic!("expected completion"),
        }
    }

    #[test]
    fn falls_back_to_message() {
        let text = "Standing by for more details.";
        match MagenticDecision::parse(text).expect("decision") {
            MagenticDecision::Message { content } => assert_eq!(content, text),
            _ => panic!("expected message"),
        }
    }

    #[test]
    fn extracts_json_block() {
        let content = r#"random text
```json
{"action":"message","message":"ok"}
```"#;
        let extracted = extract_json_from_fenced_block(content).expect("json");
        assert!(extracted.contains("\"action\""));
    }
}
