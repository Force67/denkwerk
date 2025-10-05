use std::sync::Arc;

use denkwerk::providers::openrouter::OpenRouter;
use denkwerk::{
    Agent, LLMProvider, SequentialEvent, SequentialOrchestrator,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider: Arc<dyn LLMProvider> = Arc::new(OpenRouter::from_env()?);

    let analyst = Agent::from_handlebars_file(
        "Analyst",
        "examples/prompts/sequential_analyst.hbs",
        &json!({ "name": "Analyst" }),
    )?
    .with_description("Extracts features, audience, and differentiators.");

    let writer = Agent::from_handlebars_file(
        "Writer",
        "examples/prompts/sequential_writer.hbs",
        &json!({ "name": "Writer" }),
    )?
    .with_description("Creates persuasive marketing copy.");

    let editor = Agent::from_handlebars_file(
        "Editor",
        "examples/prompts/sequential_editor.hbs",
        &json!({ "name": "Editor" }),
    )?
    .with_description("Polishes tone, grammar, and flow.");

    let orchestrator = SequentialOrchestrator::new(provider, "openai/gpt-4o-mini")
        .with_agents(vec![analyst, writer, editor]);

    let product_description = "An eco-friendly stainless steel water bottle that keeps drinks cold for 24 hours.";

    let run = orchestrator.run(product_description).await?;

    for event in &run.events {
        match event {
            SequentialEvent::Step { agent, output } => {
                println!("{agent}:\n{output}\n");
            }
            SequentialEvent::Completed { agent, output } => {
                if let Some(result) = output {
                    println!("-- {agent} finalized the copy --\n{result}\n");
                } else {
                    println!("-- {agent} signaled completion --");
                }
            }
        }
    }

    if let Some(final_output) = run.final_output {
        println!("Final copy:\n{final_output}");
    }

    Ok(())
}
