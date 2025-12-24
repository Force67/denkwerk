use std::{
    collections::HashMap,
    fmt::Write,
    sync::Arc,
};

use serde::Deserialize;

use crate::{
    agents::{Agent, AgentError},
    metrics::{AgentMetrics, ExecutionTimer, MetricsCollector, WithMetrics},
    skills::SkillRuntime,
    types::{ChatMessage, CompletionRequest, MessageRole},
    LLMProvider,
};

use super::handoffflow::AgentAction;
use crate::shared_state::SharedStateContext;

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
    pub metrics: Option<AgentMetrics>,
}

pub struct MagenticOrchestrator {
    provider: Arc<dyn LLMProvider>,
    model: String,
    manager: MagenticManager,
    roster: Vec<Agent>,
    agents: HashMap<String, Agent>,
    max_rounds: usize,
    event_callback: Option<Arc<dyn Fn(&MagenticEvent) + Send + Sync>>,
    shared_state: Option<Arc<dyn SharedStateContext>>,
    skill_runtime: Option<Arc<SkillRuntime>>,
    metrics_collector: Option<Arc<dyn MetricsCollector>>,
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
            shared_state: None,
            skill_runtime: None,
            metrics_collector: None,
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

    pub fn with_shared_state(mut self, shared_state: Arc<dyn SharedStateContext>) -> Self {
        self.shared_state = Some(shared_state);
        self
    }

    pub fn with_skill_runtime(mut self, runtime: Arc<SkillRuntime>) -> Self {
        self.skill_runtime = Some(runtime);
        self
    }

    pub fn shared_state(&self) -> Option<&Arc<dyn SharedStateContext>> {
        self.shared_state.as_ref()
    }

    pub fn with_metrics_collector(mut self, collector: Arc<dyn MetricsCollector>) -> Self {
        self.metrics_collector = Some(collector);
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
        let mut metrics = self
            .metrics_collector
            .as_ref()
            .map(|_| AgentMetrics::new("magentic_workflow".to_string()));
        let execution_timer = ExecutionTimer::new();

        for round in 0..self.max_rounds {
            let manager_prompt = build_manager_prompt(
                &task,
                round + 1,
                &self.manager,
                &self.roster,
                &transcript,
            );

            let manager_messages = vec![ChatMessage::user(manager_prompt)];
            // Execute the manager directly without Agent action parsing
            // to avoid issues with tool calls or malformed responses
            let request = CompletionRequest::new(self.model.clone(), manager_messages);
            let response = match self.provider.complete(request).await {
                Ok(response) => response,
                Err(err) => {
                    if let (Some(ref mut m), Some(collector)) = (&mut metrics, &self.metrics_collector) {
                        m.record_error(&err);
                        m.execution.total_duration = execution_timer.elapsed();
                        m.finalize(false, 0, round);
                        collector.record_metrics(m.clone());
                    }
                    return Err(err.into());
                }
            };
            if let (Some(ref mut m), Some(usage)) = (&mut metrics, response.usage.as_ref()) {
                let input_cost = m.token_usage.cost_per_input_token;
                let output_cost = m.token_usage.cost_per_output_token;
                m.record_token_usage(usage, input_cost, output_cost);
            }
            let manager_text = response.message.text().unwrap_or_default();

            if manager_text.trim().is_empty() {
                return Err(AgentError::InvalidManagerDecision(
                    "manager response is empty or contains no text".into()
                ));
            }

            let decision = MagenticDecision::parse(manager_text)?;

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

                    let skill_tools = self
                        .skill_runtime
                        .as_ref()
                        .and_then(|runtime| runtime.registry_for_agent(&agent, &transcript));
                    let turn = agent
                        .execute_with_tools(
                            self.provider.as_ref(),
                            &self.model,
                            &transcript,
                            skill_tools.as_ref(),
                            None,
                        )
                        .await;

                    let turn = match turn {
                        Ok(turn) => turn,
                        Err(err) => {
                            if let (Some(ref mut m), Some(collector)) = (&mut metrics, &self.metrics_collector) {
                                m.record_error(&err);
                                m.execution.total_duration = execution_timer.elapsed();
                                m.finalize(false, 0, round + 1);
                                collector.record_metrics(m.clone());
                            }
                            return Err(AgentError::Provider(err));
                        }
                    };

                    if let (Some(ref mut m), Some(usage)) = (&mut metrics, turn.usage.as_ref()) {
                        let input_cost = m.token_usage.cost_per_input_token;
                        let output_cost = m.token_usage.cost_per_output_token;
                        m.record_token_usage(usage, input_cost, output_cost);
                    }

                    if let Some(ref mut m) = metrics {
                        for tool_call in &turn.tool_calls {
                            m.record_function_call(
                                &tool_call.function.name,
                                execution_timer.elapsed(),
                                true,
                            );
                        }
                    }

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
                    let metrics = if let (Some(mut metrics), Some(collector)) = (metrics, &self.metrics_collector) {
                        metrics.execution.total_duration = execution_timer.elapsed();
                        metrics.finalize(true, result.len(), round + 1);
                        collector.record_metrics(metrics.clone());
                        Some(metrics)
                    } else {
                        None
                    };
                    return Ok(MagenticRun {
                        final_result: Some(result),
                        events,
                        rounds: round + 1,
                        transcript,
                        metrics,
                    });
                }
            }
        }

        if let (Some(mut metrics), Some(collector)) = (metrics, &self.metrics_collector) {
            metrics.execution.total_duration = execution_timer.elapsed();
            metrics.finalize(false, 0, self.max_rounds);
            collector.record_metrics(metrics.clone());
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

impl WithMetrics for MagenticOrchestrator {
    fn with_metrics_collector(mut self, collector: Arc<dyn MetricsCollector>) -> Self {
        self.metrics_collector = Some(collector);
        self
    }
}
