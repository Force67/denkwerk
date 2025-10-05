use std::sync::Arc;

use crate::{
    agents::{Agent, AgentAction, AgentError},
    types::ChatMessage,
    LLMProvider,
};

pub trait GroupChatManager: Send + Sync {
    /// Called before the orchestration starts so the manager can reset its state.
    fn on_start(&mut self, roster: &[Agent]);

    /// Returns the name of the agent that should speak next.
    fn select_next_agent(
        &mut self,
        roster: &[Agent],
        transcript: &[ChatMessage],
        round: usize,
    ) -> Option<String>;

    /// Determines whether the chat should terminate before the given round.
    fn should_terminate(&self, round: usize, transcript: &[ChatMessage]) -> bool;

    /// Optional hard limit on conversation rounds.
    fn max_rounds(&self) -> Option<usize> {
        None
    }
}

#[derive(Debug, Clone)]
pub struct RoundRobinGroupChatManager {
    pub maximum_rounds: Option<usize>,
    index: usize,
}

impl RoundRobinGroupChatManager {
    pub fn new() -> Self {
        Self {
            maximum_rounds: Some(6),
            index: 0,
        }
    }

    pub fn with_maximum_rounds(mut self, rounds: Option<usize>) -> Self {
        self.maximum_rounds = rounds;
        self
    }
}

impl Default for RoundRobinGroupChatManager {
    fn default() -> Self {
        Self::new()
    }
}

impl GroupChatManager for RoundRobinGroupChatManager {
    fn on_start(&mut self, roster: &[Agent]) {
        let _ = roster;
        self.index = 0;
    }

    fn select_next_agent(
        &mut self,
        roster: &[Agent],
        _transcript: &[ChatMessage],
        _round: usize,
    ) -> Option<String> {
        if roster.is_empty() {
            return None;
        }

        let agent = roster.get(self.index % roster.len())?;
        self.index = (self.index + 1) % roster.len();
        Some(agent.name().to_string())
    }

    fn should_terminate(&self, round: usize, _transcript: &[ChatMessage]) -> bool {
        if let Some(limit) = self.maximum_rounds {
            return round >= limit;
        }
        false
    }

    fn max_rounds(&self) -> Option<usize> {
        self.maximum_rounds
    }
}

#[derive(Debug, Clone)]
pub enum GroupChatEvent {
    AgentMessage { agent: String, message: String },
    AgentCompletion { agent: String, message: Option<String> },
    Terminated { reason: String },
}

#[derive(Debug, Clone)]
pub struct GroupChatRun {
    pub final_output: Option<String>,
    pub events: Vec<GroupChatEvent>,
    pub transcript: Vec<ChatMessage>,
    pub rounds: usize,
}

pub struct GroupChatOrchestrator<M: GroupChatManager + 'static> {
    provider: Arc<dyn LLMProvider>,
    model: String,
    agents: Vec<Agent>,
    manager: M,
}

impl<M: GroupChatManager + 'static> GroupChatOrchestrator<M> {
    pub fn new(provider: Arc<dyn LLMProvider>, model: impl Into<String>, manager: M) -> Self {
        Self {
            provider,
            model: model.into(),
            agents: Vec::new(),
            manager,
        }
    }

    pub fn add_agent(&mut self, agent: Agent) {
        self.agents.push(agent);
    }

    pub fn with_agents<I>(mut self, agents: I) -> Self
    where
        I: IntoIterator<Item = Agent>,
    {
        self.agents.extend(agents);
        self
    }

    pub async fn run(&mut self, task: impl Into<String>) -> Result<GroupChatRun, AgentError> {
        if self.agents.is_empty() {
            return Err(AgentError::NoAgentsRegistered);
        }

        let task = task.into();
        let mut transcript = vec![ChatMessage::user(task.clone())];
        let mut events = Vec::new();
        let mut final_output = None;
        let mut rounds = 0usize;

        self.manager.on_start(&self.agents);

        loop {
            if self.manager.should_terminate(rounds, &transcript) {
                events.push(GroupChatEvent::Terminated {
                    reason: "manager requested termination".to_string(),
                });
                break;
            }

            if let Some(limit) = self.manager.max_rounds() {
                if rounds >= limit {
                    events.push(GroupChatEvent::Terminated {
                        reason: format!("maximum rounds {limit} reached"),
                    });
                    break;
                }
            }

            let next = self
                .manager
                .select_next_agent(&self.agents, &transcript, rounds)
                .ok_or_else(|| AgentError::InvalidManagerDecision("manager returned no agent".into()))?;

            let agent = self
                .agents
                .iter()
                .find(|candidate| candidate.name() == next)
                .cloned()
                .ok_or_else(|| AgentError::UnknownAgent(next.clone()))?;

            let turn = agent
                .execute(self.provider.as_ref(), &self.model, &transcript)
                .await?;

            rounds += 1;

            match turn.action {
                AgentAction::Respond { message } => {
                    push_agent_message(&mut transcript, &agent, &message);
                    final_output = Some(message.clone());
                    events.push(GroupChatEvent::AgentMessage {
                        agent: agent.name().to_string(),
                        message,
                    });
                }
                AgentAction::HandOff { target: _, message } => {
                    let text = message.unwrap_or_default();
                    push_agent_message(&mut transcript, &agent, &text);
                    final_output = Some(text.clone());
                    events.push(GroupChatEvent::AgentMessage {
                        agent: agent.name().to_string(),
                        message: text,
                    });
                }
                AgentAction::Complete { message } => {
                    if let Some(ref content) = message {
                        push_agent_message(&mut transcript, &agent, content);
                        final_output = Some(content.clone());
                    }
                    events.push(GroupChatEvent::AgentCompletion {
                        agent: agent.name().to_string(),
                        message: message.clone(),
                    });
                    break;
                }
            }
        }

        Ok(GroupChatRun {
            final_output,
            events,
            transcript,
            rounds,
        })
    }
}

fn push_agent_message(transcript: &mut Vec<ChatMessage>, agent: &Agent, content: &str) {
    let mut message = ChatMessage::assistant(content.to_string());
    message.name = Some(agent.name().to_string());
    transcript.push(message);
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;

    use crate::{
        agents::{Agent, AgentError},
        providers::LLMProvider,
        types::{ChatMessage, CompletionRequest, CompletionResponse},
        LLMError,
    };

    use super::{GroupChatEvent, GroupChatManager, GroupChatOrchestrator, RoundRobinGroupChatManager};

    struct TestProvider {
        responses: Mutex<Vec<String>>,
    }

    impl TestProvider {
        fn new(responses: Vec<String>) -> Self {
            Self {
                responses: Mutex::new(responses),
            }
        }
    }

    #[async_trait]
    impl LLMProvider for TestProvider {
        async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, LLMError> {
            let content = {
                let mut guard = self.responses.lock().unwrap();
                guard.remove(0)
            };

            Ok(CompletionResponse {
                message: ChatMessage::assistant(content),
                usage: None,
                reasoning: None,
            })
        }

        fn name(&self) -> &'static str {
            "test"
        }
    }

    #[tokio::test]
    async fn rotates_between_agents() {
        let provider: Arc<dyn LLMProvider> = Arc::new(TestProvider::new(vec![
            "writer response".to_string(),
            "editor response".to_string(),
        ]));

        let manager = RoundRobinGroupChatManager::new().with_maximum_rounds(Some(2));
        let mut orchestrator = GroupChatOrchestrator::new(provider, "model", manager)
            .with_agents(vec![
                Agent::from_string("Writer", "Draft copy."),
                Agent::from_string("Editor", "Review copy."),
            ]);

        let run = orchestrator
            .run("Create a slogan")
            .await
            .expect("run should succeed");

        assert_eq!(run.rounds, 2);
        assert_eq!(run.final_output.as_deref(), Some("editor response"));
        assert!(matches!(run.events.first(), Some(GroupChatEvent::AgentMessage { agent, .. }) if agent == "Writer"));
    }

    #[tokio::test]
    async fn errors_when_no_agents() {
        let provider: Arc<dyn LLMProvider> = Arc::new(TestProvider::new(vec![]));
        let manager = RoundRobinGroupChatManager::default();
        let mut orchestrator = GroupChatOrchestrator::new(provider, "model", manager);
        let error = orchestrator.run("task").await.unwrap_err();
        assert!(matches!(error, AgentError::NoAgentsRegistered));
    }
}
