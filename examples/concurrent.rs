use std::sync::Arc;

use denkwerk::providers::openrouter::OpenRouter;
use denkwerk::{
    Agent, ConcurrentEvent, ConcurrentOrchestrator, ConcurrentResult, LLMProvider,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider: Arc<dyn LLMProvider> = Arc::new(OpenRouter::from_env()?);

    let physics = Agent::from_handlebars_file(
        "PhysicsExpert",
        "examples/prompts/concurrent_physics.hbs",
        &json!({ "name": "PhysicsExpert" }),
    )?
    .with_description("Explains phenomena from a physics standpoint.");

    let chemistry = Agent::from_handlebars_file(
        "ChemistryExpert",
        "examples/prompts/concurrent_chemistry.hbs",
        &json!({ "name": "ChemistryExpert" }),
    )?
    .with_description("Explains phenomena using chemistry concepts.");

    let orchestrator = ConcurrentOrchestrator::new(provider, "openai/gpt-4o-mini")
        .with_agents(vec![physics, chemistry]);

    let run = orchestrator.run("What is temperature and why does it matter?").await?;

    println!("Collected responses (order reflects completion):\n");
    for ConcurrentResult { agent, output } in &run.results {
        match output {
            Some(text) => println!("[{agent}] {text}\n"),
            None => println!("[{agent}] (no textual output)\n"),
        }
    }

    println!("Events:\n");
    for event in &run.events {
        match event {
            ConcurrentEvent::Message { agent, output } => {
                println!("Message from {agent}: {output}");
            }
            ConcurrentEvent::Completed { agent, output } => {
                if let Some(text) = output {
                    println!("{agent} completed with: {text}");
                } else {
                    println!("{agent} completed without a message");
                }
            }
        }
    }

    Ok(())
}
