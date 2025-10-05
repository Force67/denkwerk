use denkwerk::providers::openai::OpenAI;
use denkwerk::{ChatMessage, CompletionRequest, LLMProvider, StreamEvent};
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = OpenAI::from_env()?;

    let request = CompletionRequest::new(
        "gpt-4o-mini",
        vec![
            ChatMessage::system("You are an assistant who thinks aloud before responding."),
            ChatMessage::user("Explain why the sky appears blue in simple terms."),
        ],
    )
    .with_temperature(0.7);

    let mut stream = provider.stream_completion(request).await?;

    println!("=== Streaming response ===");

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::MessageDelta(delta) => {
                print!("{delta}");
            }
            StreamEvent::ReasoningDelta(delta) => {
                eprintln!("[reasoning] {delta}");
            }
            StreamEvent::ToolCallDelta { index, arguments } => {
                eprintln!("[tool #{index}] {arguments}");
            }
            StreamEvent::Completed(response) => {
                println!("\n---\nFull response: {}", response.message.text().unwrap_or_default());
                if let Some(reasoning) = response.reasoning {
                    for trace in reasoning {
                        eprintln!("Reasoning trace: {}", trace.content);
                    }
                }
                if let Some(usage) = response.usage {
                    println!(
                        "Usage — prompt: {}, completion: {}, total: {}",
                        usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                    );
                }
            }
        }
    }

    Ok(())
}
