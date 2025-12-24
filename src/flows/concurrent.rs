use std::sync::Arc;

use futures_util::{stream::FuturesUnordered, StreamExt};

use crate::{
    agents::{Agent, AgentError},
    metrics::{AgentMetrics, ExecutionTimer, MetricsCollector, WithMetrics},
    skills::SkillRuntime,
    types::ChatMessage,
    LLMProvider,
};

use super::handoffflow::AgentAction;
use crate::shared_state::SharedStateContext;

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
    pub metrics: Option<Vec<AgentMetrics>>,
}

pub struct ConcurrentOrchestrator {
    provider: Arc<dyn LLMProvider>,
    model: String,
    agents: Vec<Agent>,
    event_callback: Option<Arc<dyn Fn(&ConcurrentEvent) + Send + Sync>>,
    shared_state: Option<Arc<dyn SharedStateContext>>,
    skill_runtime: Option<Arc<SkillRuntime>>,
    metrics_collector: Option<Arc<dyn MetricsCollector>>,
}

impl ConcurrentOrchestrator {
    pub fn new(provider: Arc<dyn LLMProvider>, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            agents: Vec::new(),
            event_callback: None,
            shared_state: None,
            skill_runtime: None,
            metrics_collector: None,
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

    pub fn with_event_callback(mut self, callback: impl Fn(&ConcurrentEvent) + Send + Sync + 'static) -> Self {
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

    fn emit_event(&self, event: &ConcurrentEvent) {
        if let Some(callback) = &self.event_callback {
            callback(event);
        }
    }

    pub async fn run(&self, task: impl Into<String>) -> Result<ConcurrentRun, AgentError> {
        if self.agents.is_empty() {
            return Err(AgentError::NoAgentsRegistered);
        }

        let task = task.into();
        let mut transcript = vec![ChatMessage::user(task.clone())];
        let mut events = Vec::new();
        let mut results = Vec::new();
        let mut collected_metrics = self.metrics_collector.as_ref().map(|_| Vec::new());

        let mut futures = FuturesUnordered::new();
        let metrics_collector = self.metrics_collector.clone();
        let skill_runtime = self.skill_runtime.clone();
        for agent in &self.agents {
            let agent = agent.clone();
            let provider = Arc::clone(&self.provider);
            let model = self.model.clone();
            let task_clone = task.clone();
            let metrics_collector = metrics_collector.clone();
            let skill_runtime = skill_runtime.clone();

            futures.push(async move {
                let mut metrics = metrics_collector
                    .as_ref()
                    .map(|_| AgentMetrics::new(agent.name().to_string()));
                let timer = ExecutionTimer::new();
                let history = vec![ChatMessage::user(task_clone)];
                let skill_tools = skill_runtime
                    .as_ref()
                    .and_then(|runtime| runtime.registry_for_agent(&agent, &history));
                let turn = agent
                    .execute_with_tools(
                        provider.as_ref(),
                        &model,
                        &history,
                        skill_tools.as_ref(),
                        None,
                    )
                    .await;
                match turn {
                    Ok(turn) => {
                        if let (Some(ref mut m), Some(usage)) = (&mut metrics, turn.usage.as_ref()) {
                            let input_cost = m.token_usage.cost_per_input_token;
                            let output_cost = m.token_usage.cost_per_output_token;
                            m.record_token_usage(usage, input_cost, output_cost);
                        }

                        if let Some(ref mut m) = metrics {
                            for tool_call in &turn.tool_calls {
                                m.record_function_call(
                                    &tool_call.function.name,
                                    timer.elapsed(),
                                    true,
                                );
                            }
                        }

                        let action = turn.action;
                        if let Some(ref mut m) = metrics {
                            let output_length = match &action {
                                AgentAction::Respond { message } => message.len(),
                                AgentAction::HandOff { message, .. } => message.as_ref().map(|m| m.len()).unwrap_or(0),
                                AgentAction::Complete { message } => message.as_ref().map(|m| m.len()).unwrap_or(0),
                            };
                            m.execution.total_duration = timer.elapsed();
                            m.finalize(true, output_length, 1);
                            if let Some(ref collector) = metrics_collector {
                                collector.record_metrics(m.clone());
                            }
                        }

                        Ok((agent, action, metrics))
                    }
                    Err(err) => {
                        if let Some(ref mut m) = metrics {
                            m.record_error(&err);
                            m.execution.total_duration = timer.elapsed();
                            m.finalize(false, 0, 1);
                            if let Some(ref collector) = metrics_collector {
                                collector.record_metrics(m.clone());
                            }
                        }
                        Err(AgentError::from(err))
                    }
                }
            });
        }

        while let Some(result) = futures.next().await {
            let (agent, action, metrics) = result?;
            if let (Some(ref mut bucket), Some(metric)) = (&mut collected_metrics, metrics) {
                bucket.push(metric);
            }
            let name = agent.name().to_string();

            match action {
                AgentAction::Respond { message } => {
                    push_agent_message(&mut transcript, &agent, &message);
                    let event = ConcurrentEvent::Message {
                        agent: name.clone(),
                        output: message.clone(),
                    };
                    self.emit_event(&event);
                    events.push(event);
                    results.push(ConcurrentResult {
                        agent: name,
                        output: Some(message),
                    });
                }
                AgentAction::HandOff { target: _, message } => {
                    let text = message.unwrap_or_default();
                    push_agent_message(&mut transcript, &agent, &text);
                    let event = ConcurrentEvent::Message {
                        agent: name.clone(),
                        output: text.clone(),
                    };
                    self.emit_event(&event);
                    events.push(event);
                    results.push(ConcurrentResult {
                        agent: name,
                        output: Some(text),
                    });
                }
                AgentAction::Complete { message } => {
                    if let Some(ref content) = message {
                        push_agent_message(&mut transcript, &agent, content);
                    }
                    let event = ConcurrentEvent::Completed {
                        agent: name.clone(),
                        output: message.clone(),
                    };
                    self.emit_event(&event);
                    events.push(event);
                    results.push(ConcurrentResult { agent: name, output: message });
                }
            }
        }

        Ok(ConcurrentRun {
            results,
            events,
            transcript,
            metrics: collected_metrics,
        })
    }
}

fn push_agent_message(transcript: &mut Vec<ChatMessage>, agent: &Agent, content: &str) {
    let mut message = ChatMessage::assistant(content.to_string());
    message.name = Some(agent.name().to_string());
    transcript.push(message);
}

impl WithMetrics for ConcurrentOrchestrator {
    fn with_metrics_collector(mut self, collector: Arc<dyn MetricsCollector>) -> Self {
        self.metrics_collector = Some(collector);
        self
    }
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
