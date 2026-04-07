use denkwerk::providers::ollama::{Ollama, OllamaConfig};
use denkwerk::{ChatMessage, CompletionRequest, LLMProvider};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base_url = std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
    let model = std::env::args().nth(1).unwrap_or_else(|| "gemma4:31b".to_string());

    let config = OllamaConfig::new()
        .with_base_url(&base_url)
        .with_keep_alive("30m");
    let provider = Ollama::from_config(config)?;

    println!("Connecting to Ollama at {base_url} with model {model}");

    let request = CompletionRequest::new(
        model,
        vec![ChatMessage::user("Say hello in exactly 3 words.")],
    )
    .with_max_tokens(30)
    .with_temperature(0.1);

    let response = provider.complete(request).await?;
    println!("Response: {:?}", response.message.content);

    if let Some(usage) = response.usage {
        println!(
            "Tokens: prompt={}, completion={}, total={}, cached={:?}",
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
            usage.cached_tokens
        );
    }

    let models = provider.list_models().await?;
    println!("Available models: {}", models.len());
    for m in models.iter().take(5) {
        println!("  - {}", m.id);
    }

    Ok(())
}
