use std::sync::Arc;

use futures_util::{stream::FuturesUnordered, StreamExt};

use crate::{
    agents::{Agent, AgentAction, AgentError},
    types::ChatMessage,
    LLMProvider,
};

#[derive(Debug, Clone)]
pub enum ConcurrentEvent {
    Message { agent: String, output: String },
    Completed { agent: String, output: Option<String> },
}

#[derive(Debug, Clone)]
pub struct ConcurrentResult {
    pub agent: String,
    pub output: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ConcurrentRun {
    pub results: Vec<ConcurrentResult>,
    pub events: Vec<ConcurrentEvent>,
    pub transcript: Vec<ChatMessage>,
}

pub struct ConcurrentOrchestrator {
    provider: Arc<dyn LLMProvider>,
    model: String,
    agents: Vec<Agent>,
}

impl ConcurrentOrchestrator {
    pub fn new(provider: Arc<dyn LLMProvider>, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            agents: Vec::new(),
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

    pub async fn run(&self, task: impl Into<String>) -> Result<ConcurrentRun, AgentError> {
        if self.agents.is_empty() {
            return Err(AgentError::NoAgentsRegistered);
        }

        let task = task.into();
        let mut transcript = vec![ChatMessage::user(task.clone())];
        let mut events = Vec::new();
        let mut results = Vec::new();

        let mut futures = FuturesUnordered::new();
        for agent in &self.agents {
            let agent = agent.clone();
            let provider = Arc::clone(&self.provider);
            let model = self.model.clone();
            let task_clone = task.clone();

            futures.push(async move {
                let history = vec![ChatMessage::user(task_clone)];
                let turn = agent.execute(provider.as_ref(), &model, &history).await;
                match turn {
                    Ok(turn) => Ok((agent, turn.action)),
                    Err(err) => Err(AgentError::from(err)),
                }
            });
        }

        while let Some(result) = futures.next().await {
            let (agent, action) = result?;
            let name = agent.name().to_string();

            match action {
                AgentAction::Respond { message } => {
                    push_agent_message(&mut transcript, &agent, &message);
                    events.push(ConcurrentEvent::Message {
                        agent: name.clone(),
                        output: message.clone(),
                    });
                    results.push(ConcurrentResult {
                        agent: name,
                        output: Some(message),
                    });
                }
                AgentAction::HandOff { target: _, message } => {
                    let text = message.unwrap_or_default();
                    push_agent_message(&mut transcript, &agent, &text);
                    events.push(ConcurrentEvent::Message {
                        agent: name.clone(),
                        output: text.clone(),
                    });
                    results.push(ConcurrentResult {
                        agent: name,
                        output: Some(text),
                    });
                }
                AgentAction::Complete { message } => {
                    if let Some(ref content) = message {
                        push_agent_message(&mut transcript, &agent, content);
                    }
                    events.push(ConcurrentEvent::Completed {
                        agent: name.clone(),
                        output: message.clone(),
                    });
                    results.push(ConcurrentResult { agent: name, output: message });
                }
            }
        }

        Ok(ConcurrentRun {
            results,
            events,
            transcript,
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
    use std::time::Duration;

    use async_trait::async_trait;
    use tokio::time::sleep;

    use crate::{
        agents::{Agent, AgentError},
        providers::LLMProvider,
        types::{ChatMessage, CompletionRequest, CompletionResponse},
        LLMError,
    };

    use super::{ConcurrentEvent, ConcurrentOrchestrator};

    struct TestProvider {
        responses: Mutex<Vec<(String, Option<Duration>)>>,
    }

    impl TestProvider {
        fn new(responses: Vec<(String, Option<Duration>)>) -> Self {
            Self {
                responses: Mutex::new(responses),
            }
        }
    }

    #[async_trait]
    impl LLMProvider for TestProvider {
        async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, LLMError> {
            let entry = {
                let mut guard = self.responses.lock().unwrap();
                guard.remove(0)
            };

            if let Some(delay) = entry.1 {
                sleep(delay).await;
            }

            Ok(CompletionResponse {
                message: ChatMessage::assistant(entry.0),
                usage: None,
                reasoning: None,
            })
        }

        fn name(&self) -> &'static str {
            "test"
        }
    }

    #[tokio::test]
    async fn collects_concurrent_results() {
        let provider: Arc<dyn LLMProvider> = Arc::new(TestProvider::new(vec![
            ("physics".to_string(), Some(Duration::from_millis(50))),
            ("chemistry".to_string(), Some(Duration::from_millis(10))),
        ]));

        let orchestrator = ConcurrentOrchestrator::new(provider, "model").with_agents(vec![
            Agent::from_string("Physics", "Explain physics."),
            Agent::from_string("Chemistry", "Explain chemistry."),
        ]);

        let run = orchestrator
            .run("What is temperature?")
            .await
            .expect("run should succeed");

        assert_eq!(run.results.len(), 2);
        assert!(run
            .events
            .iter()
            .any(|event| matches!(event, ConcurrentEvent::Message { agent, .. } if agent == "Chemistry")));
        assert_eq!(run.transcript.len(), 3); // user + two replies
    }

    #[tokio::test]
    async fn errors_when_no_agents() {
        let provider: Arc<dyn LLMProvider> = Arc::new(TestProvider::new(vec![]));
        let orchestrator = ConcurrentOrchestrator::new(provider, "model");
        let error = orchestrator.run("task").await.unwrap_err();
        assert!(matches!(error, AgentError::NoAgentsRegistered));
    }
}
