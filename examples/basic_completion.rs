use denkwerk::providers::openrouter::OpenRouter;
use denkwerk::{types::{ChatMessage, CompletionRequest}, LLMProvider};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = OpenRouter::from_env()?;

    let request = CompletionRequest::new(
        "openai/gpt-4o",
        vec![
            ChatMessage::system("You are a concise assistant."),
            ChatMessage::user("Give me a short interesting fact about space."),
        ],
    )
    .with_max_tokens(100)
    .with_temperature(0.7);

    let response = provider.complete(request).await?;

    println!("{}", response.message.text().unwrap_or_default());

    Ok(())
}
