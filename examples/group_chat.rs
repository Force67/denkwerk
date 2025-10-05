use std::sync::Arc;

use denkwerk::providers::openrouter::OpenRouter;
use denkwerk::{
    Agent, GroupChatEvent, GroupChatOrchestrator, LLMProvider, RoundRobinGroupChatManager,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider: Arc<dyn LLMProvider> = Arc::new(OpenRouter::from_env()?);

    let writer = Agent::from_handlebars_file(
        "CopyWriter",
        "examples/prompts/group_writer.hbs",
        &json!({ "name": "CopyWriter" }),
    )?
    .with_description("Drafts concise marketing slogans with dry humor.");

    let reviewer = Agent::from_handlebars_file(
        "Reviewer",
        "examples/prompts/group_editor.hbs",
        &json!({ "name": "Reviewer" }),
    )?
    .with_description("Evaluates slogans through an art director lens.");

    let manager = RoundRobinGroupChatManager::new().with_maximum_rounds(Some(6));
    let mut orchestrator = GroupChatOrchestrator::new(provider, "openai/gpt-4o-mini", manager)
        .with_agents(vec![writer, reviewer]);

    let run = orchestrator
        .run("Create a slogan for a new electric SUV that is affordable and fun to drive.")
        .await?;

    for event in &run.events {
        match event {
            GroupChatEvent::AgentMessage { agent, message } => {
                println!("{agent}: {message}\n");
            }
            GroupChatEvent::AgentCompletion { agent, message } => {
                if let Some(text) = message {
                    println!("{agent} signaled completion with: {text}\n");
                } else {
                    println!("{agent} signaled completion.\n");
                }
            }
            GroupChatEvent::UserMessage { message } => {
                println!("[User]: {message}\n");
            }
            GroupChatEvent::Terminated { reason } => {
                println!("[Manager terminated] {reason}\n");
            }
        }
    }

    if let Some(final_output) = run.final_output {
        println!("Final outcome:\n{final_output}");
    }

    Ok(())
}
