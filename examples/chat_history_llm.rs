use std::sync::Arc;

use denkwerk::history::{ChatHistory, LLMHistoryCompressor};
use denkwerk::providers::openrouter::OpenRouter;
use denkwerk::LLMProvider;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider: Arc<dyn LLMProvider> = Arc::new(OpenRouter::from_env()?);
    let compressor = LLMHistoryCompressor::new(Arc::clone(&provider), "openai/gpt-4o-mini")
        .with_max_messages(8)
        .with_retain_messages(4)
        .with_summary_prefix("Earlier context: ");

    let mut history = ChatHistory::new();
    history.push_system("You are capturing user preferences.");
    history.push_user("I'm looking for weekend getaway ideas.");
    history.push_assistant("Sure! Do you prefer mountains or beaches?");
    history.push_user("Definitely beaches, and somewhere budget friendly.");
    history.push_assistant("Consider Santa Cruz or San Diego—they're fun and relatively affordable.");
    history.push_user("Can you remind me of the travel time from San Francisco?");
    history.push_assistant("Roughly 75 minutes to Santa Cruz by car, closer to 8 hours to San Diego.");

    println!("Messages before compression: {}", history.len());

    let changed = compressor.compress(&mut history).await?;
    if changed {
        println!("History compressed to {} messages", history.len());
    } else {
        println!("History not compressed; below threshold.");
    }

    for (index, message) in history.iter().enumerate() {
        println!("{}: {:?} -> {}", index, message.role, message.text().unwrap_or(""));
    }

    Ok(())
}
