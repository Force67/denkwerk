use std::sync::Arc;

use denkwerk::providers::openrouter::OpenRouter;
use denkwerk::{
    Agent, GroupChatEvent, GroupChatManager, GroupChatOrchestrator, GroupChatRun, LLMProvider,
};
use serde_json::json;

struct AlternatingManager {
    max_rounds: usize,
    next: usize,
}

impl AlternatingManager {
    fn new(max_rounds: usize) -> Self {
        Self { max_rounds, next: 0 }
    }
}

impl GroupChatManager for AlternatingManager {
    fn on_start(&mut self, _roster: &[Agent]) {
        self.next = 0;
    }

    fn select_next_agent(
        &mut self,
        roster: &[Agent],
        _transcript: &[denkwerk::ChatMessage],
        _round: usize,
    ) -> Option<String> {
        if roster.is_empty() {
            return None;
        }
        let agent = &roster[self.next % roster.len()];
        self.next = (self.next + 1) % roster.len();
        Some(agent.name().to_string())
    }

    fn should_terminate(&self, round: usize, _transcript: &[denkwerk::ChatMessage]) -> bool {
        round >= self.max_rounds
    }

    fn should_request_user_input(&self, round: usize, _transcript: &[denkwerk::ChatMessage]) -> bool {
        round == 1
    }
}

fn print_run(run: &GroupChatRun) {
    for event in &run.events {
        match event {
            GroupChatEvent::AgentMessage { agent, message } => println!("{agent}: {message}"),
            GroupChatEvent::AgentCompletion { agent, message } => {
                println!("{agent} complete: {}", message.clone().unwrap_or_default());
            }
            GroupChatEvent::UserMessage { message } => println!("[User]: {message}"),
            GroupChatEvent::Terminated { reason } => println!("[Manager terminated] {reason}"),
        }
    }

    if let Some(final_output) = &run.final_output {
        println!("\nConsensus: {final_output}");
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider: Arc<dyn LLMProvider> = Arc::new(OpenRouter::from_env()?);

    let concept = Agent::from_handlebars_file(
        "Strategist",
        "examples/prompts/group_writer.hbs",
        &json!({ "name": "Strategist" }),
    )?
    .with_description("Proposes punchy slogans.");

    let critic = Agent::from_handlebars_file(
        "Critic",
        "examples/prompts/group_editor.hbs",
        &json!({ "name": "Critic" }),
    )?
    .with_description("Evaluates slogans and requests revisions.");

    let mut orchestrator = GroupChatOrchestrator::new(provider, "openai/gpt-4o-mini", AlternatingManager::new(6))
        .with_agents(vec![concept, critic])
        .with_user_input_callback(|transcript| {
            if let Some(last) = transcript.last() {
                println!("\nManager asked for feedback after: {}", last.text().unwrap_or(""));
            }
            Some("Let's keep it playful but highlight affordability.".to_string())
        })
        .with_event_callback(|event| match event {
            GroupChatEvent::AgentMessage { agent, .. } => println!("(callback) {agent} contributed."),
            _ => {}
        });

    let run = orchestrator
        .run("We need a slogan for an affordable, fun electric SUV.")
        .await?;

    print_run(&run);

    Ok(())
}
