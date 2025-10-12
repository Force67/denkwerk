use std::{
    sync::Arc,
};

use crate::{
    agents::{Agent, AgentError},
    types::ChatMessage,
    LLMProvider,
};

use super::handoffflow::AgentAction;
use crate::shared_state::SharedStateContext;

#[derive(Debug, Clone)]
pub enum SequentialEvent {
    Step {
        agent: String,
        output: String,
    },
    Completed {
        agent: String,
        output: Option<String>,
    },
}

#[derive(Debug, Clone)]
pub struct SequentialRun {
    pub final_output: Option<String>,
    pub events: Vec<SequentialEvent>,
    pub transcript: Vec<ChatMessage>,
}

pub struct SequentialOrchestrator {
    provider: Arc<dyn LLMProvider>,
    model: String,
    pipeline: Vec<Agent>,
    event_callback: Option<Arc<dyn Fn(&SequentialEvent) + Send + Sync>>,
    shared_state: Option<Arc<dyn SharedStateContext>>,
}

impl SequentialOrchestrator {
    pub fn new(provider: Arc<dyn LLMProvider>, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            pipeline: Vec::new(),
            event_callback: None,
            shared_state: None,
        }
    }

    pub fn add_agent(&mut self, agent: Agent) {
        self.pipeline.push(agent);
    }

    pub fn with_agents<I>(mut self, agents: I) -> Self
    where
        I: IntoIterator<Item = Agent>,
    {
        self.pipeline.extend(agents);
        self
    }

    pub fn with_event_callback(mut self, callback: impl Fn(&SequentialEvent) + Send + Sync + 'static) -> Self {
        self.event_callback = Some(Arc::new(callback));
        self
    }

    pub fn with_shared_state(mut self, shared_state: Arc<dyn SharedStateContext>) -> Self {
        self.shared_state = Some(shared_state);
        self
    }

    pub fn shared_state(&self) -> Option<&Arc<dyn SharedStateContext>> {
        self.shared_state.as_ref()
    }

    fn emit_event(&self, event: &SequentialEvent) {
        if let Some(callback) = &self.event_callback {
            callback(event);
        }
    }

    pub async fn run(&self, task: impl Into<String>) -> Result<SequentialRun, AgentError> {
        if self.pipeline.is_empty() {
            return Err(AgentError::NoAgentsRegistered);
        }

        let task = task.into();
        let mut transcript = vec![ChatMessage::user(task.clone())];
        let mut events = Vec::new();
        let mut payload = task;

        for (index, agent) in self.pipeline.iter().enumerate() {
            let turn = agent
                .execute(self.provider.as_ref(), &self.model, &transcript)
                .await?;

            match turn.action {
                AgentAction::Respond { message } => {
                    push_agent_message(&mut transcript, agent, &message);
                    payload = message.clone();
                    let event = SequentialEvent::Step {
                        agent: agent.name().to_string(),
                        output: message,
                    };
                    self.emit_event(&event);
                    events.push(event);
                }
                AgentAction::HandOff { target: _, message } => {
                    let text = message.unwrap_or_default();
                    push_agent_message(&mut transcript, agent, &text);
                    if !text.is_empty() {
                        payload = text.clone();
                    }
                    let event = SequentialEvent::Step {
                        agent: agent.name().to_string(),
                        output: text,
                    };
                    self.emit_event(&event);
                    events.push(event);
                }
                AgentAction::Complete { message } => {
                    let text = message.clone();
                    if let Some(ref content) = text {
                        push_agent_message(&mut transcript, agent, content);
                        payload = content.clone();
                    }
                    let event = SequentialEvent::Completed {
                        agent: agent.name().to_string(),
                        output: text.clone(),
                    };
                    self.emit_event(&event);
                    events.push(event);

                    return Ok(SequentialRun {
                        final_output: text.or_else(|| Some(payload.clone())),
                        events,
                        transcript,
                    });
                }
            }

            // If this was the last agent, mark completion with its output.
            if index == self.pipeline.len() - 1 {
                let event = SequentialEvent::Completed {
                    agent: agent.name().to_string(),
                    output: Some(payload.clone()),
                };
                self.emit_event(&event);
                events.push(event);

                return Ok(SequentialRun {
                    final_output: Some(payload),
                    events,
                    transcript,
                });
            }
        }

        // Should not reach here because loop returns on last agent.
        unreachable!("sequential orchestrator exited without completion");
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

    use super::{SequentialEvent, SequentialOrchestrator};

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
            let mut guard = self.responses.lock().unwrap();
            let content = guard.remove(0);
            drop(guard);

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
    async fn runs_agents_in_sequence() {
        let provider: Arc<dyn LLMProvider> = Arc::new(TestProvider::new(vec![
            "features: speed".to_string(),
            "copy: fast product".to_string(),
            "final: polished".to_string(),
        ]));

        let agent_a = Agent::from_string("Analyst", "Identify features.");
        let agent_b = Agent::from_string("Writer", "Write marketing copy.");
        let agent_c = Agent::from_string("Editor", "Polish the draft.");

        let orchestrator = SequentialOrchestrator::new(provider, "model").with_agents(vec![
            agent_a,
            agent_b,
            agent_c,
        ]);

        let run = orchestrator
            .run("Describe the product")
            .await
            .expect("run should succeed");

        assert_eq!(run.final_output.as_deref(), Some("final: polished"));
        assert_eq!(run.events.len(), 4);
        match run.events.last() {
            Some(SequentialEvent::Completed { agent, output }) => {
                assert_eq!(agent, "Editor");
                assert_eq!(output.as_deref(), Some("final: polished"));
            }
            other => panic!("unexpected event: {other:?}"),
        }
        assert_eq!(run.transcript.len(), 4); // initial user + three agent replies
    }

    #[tokio::test]
    async fn errors_when_no_agents() {
        let provider: Arc<dyn LLMProvider> = Arc::new(TestProvider::new(vec![]));
        let orchestrator = SequentialOrchestrator::new(provider, "model");
        let error = orchestrator.run("task").await.unwrap_err();
        assert!(matches!(error, AgentError::NoAgentsRegistered));
    }

    #[tokio::test]
    async fn fires_event_callback() {
        let provider: Arc<dyn LLMProvider> = Arc::new(TestProvider::new(vec![
            "step one".to_string(),
            "step two".to_string(),
        ]));

        let events: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let events_clone = Arc::clone(&events);

        let orchestrator = SequentialOrchestrator::new(provider, "model")
            .with_agents(vec![
                Agent::from_string("A", "first"),
                Agent::from_string("B", "second"),
            ])
            .with_event_callback(move |event| {
                if let SequentialEvent::Step { agent, .. } = event {
                    events_clone.lock().unwrap().push(agent.clone());
                }
            });

        let _ = orchestrator.run("task").await.unwrap();
        let recorded = events.lock().unwrap().clone();
        assert_eq!(recorded, vec!["A".to_string(), "B".to_string()]);
    }
}
