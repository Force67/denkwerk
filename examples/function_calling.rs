use std::error::Error;
use std::sync::Arc;

use denkwerk::providers::openai::OpenAI;
use denkwerk::{
    kernel_function, kernel_module, ChatMessage, CompletionRequest, FunctionRegistry, LLMProvider,
};
use serde_json::to_string;

#[kernel_function(name = "greet_contact", description = "Greet a person by name.")]
fn greet(
    #[param(description = "Name of the person to greet")] name: String,
) -> Result<String, std::convert::Infallible> {
    Ok(format!("Hello, {name}!"))
}

#[derive(Clone)]
struct Calculator;

#[kernel_module]
impl Calculator {
    #[kernel_function(name = "add_numbers", description = "Add two floating point values.")]
    fn add(
        &self,
        #[param(description = "First operand")] a: f64,
        #[param(description = "Second operand")] b: f64,
    ) -> Result<f64, std::convert::Infallible> {
        Ok(a + b)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let provider = OpenAI::from_env()?;

    let mut registry = FunctionRegistry::new();
    registry.register(greet_kernel());
    Arc::new(Calculator).register_kernel_functions(&mut registry);

    let mut messages = vec![
        ChatMessage::system("You are an assistant that must use tools whenever possible."),
        ChatMessage::user("Say hello to Sam and compute 4.5 + 5.25 using the tools."),
    ];

    let mut request = CompletionRequest::new("gpt-4o-mini", messages.clone())
        .with_function_registry(&registry);

    let mut response = provider.complete(request).await?;
    println!("assistant: {}", response.message.text().unwrap_or("<tool call>"));
    messages.push(response.message.clone());

    if !response.message.tool_calls.is_empty() {
        // Execute tool calls in parallel but preserve order for LLM expectations
        use futures_util::future::join_all;

        let tool_futures: Vec<_> = response.message.tool_calls.iter()
            .map(|call| registry.invoke(&call.function))
            .collect();

        let results = join_all(tool_futures).await;

        // Process results in original call order
        for (call, result) in response.message.tool_calls.iter().zip(results) {
            let result = result?;
            let payload = to_string(&result)?;
            let tool_id = call
                .id
                .clone()
                .unwrap_or_else(|| call.function.name.clone());
            println!("tool {} -> {}", tool_id, payload);
            messages.push(ChatMessage::tool(tool_id, payload));
        }

        request = CompletionRequest::new("gpt-4o-mini", messages.clone())
            .with_function_registry(&registry);
        response = provider.complete(request).await?;
        println!(
            "assistant (final): {}",
            response.message.text().unwrap_or("<no response>")
        );
    }

    Ok(())
}
