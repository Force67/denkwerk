use std::sync::Arc;

use denkwerk::providers::openrouter::OpenRouter;
use denkwerk::{
    Agent, MagenticEvent, MagenticManager, MagenticOrchestrator, LLMProvider,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider: Arc<dyn LLMProvider> = Arc::new(OpenRouter::from_env()?);

    let manager = MagenticManager::standard();

    let research = Agent::from_handlebars_file(
        "ResearchAgent",
        "examples/prompts/magentic_researcher.hbs",
        &json!({ "name": "ResearchAgent" }),
    )?
    .with_description("Surfaces background information and credible references.");

    let coder = Agent::from_handlebars_file(
        "CoderAgent",
        "examples/prompts/magentic_coder.hbs",
        &json!({ "name": "CoderAgent" }),
    )?
    .with_description("Runs quantitative reasoning and structures results with tables.");

    let mut orchestrator = MagenticOrchestrator::new(provider, "openai/gpt-4o-mini", manager)
        .with_max_rounds(8);

    orchestrator.register_agent(research)?;
    orchestrator.register_agent(coder)?;

    let brief = concat!(
        "Create an energy efficiency brief comparing ResNet-50, BERT-base, and GPT-2. ",
        "Estimate both training and inference energy usage plus the associated CO2 emissions ",
        "for a 24-hour window on Azure Standard_NC6s_v3 infrastructure. ",
        "Conclude with recommendations for the most efficient model per workload."
    );

    let run = orchestrator.run(brief).await?;

    for event in run.events {
        match event {
            MagenticEvent::ManagerMessage { message } => {
                println!("Manager: {message}");
            }
            MagenticEvent::ManagerDelegation {
                target,
                instructions,
                progress_note,
            } => {
                if let Some(note) = progress_note {
                    println!("Manager note: {note}");
                }
                println!("[delegate] -> {target}: {instructions}");
            }
            MagenticEvent::AgentMessage { agent, message } => {
                println!("{agent}: {message}");
            }
            MagenticEvent::AgentCompletion { agent, message } => {
                if let Some(msg) = message {
                    println!("{agent} (complete): {msg}");
                } else {
                    println!("{agent} signaled completion.");
                }
            }
            MagenticEvent::Completed { message } => {
                println!("\nManager final answer:\n{message}");
            }
        }
    }

    if let Some(final_result) = run.final_result {
        println!("\nFinal result:\n{final_result}");
    } else {
        println!("\nNo final result was produced within the configured round limit.");
    }

    Ok(())
}
