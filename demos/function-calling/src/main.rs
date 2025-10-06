use clap::Parser;
use denkwerk::providers::openrouter::OpenRouter;
use denkwerk::{
    ChatMessage, CompletionRequest, FunctionRegistry, LLMProvider, kernel_function, kernel_module,
};
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "function-calling-demo")]
#[command(about = "Comprehensive demo of function calling capabilities")]
struct Args {
    /// OpenRouter API token
    #[arg(long)]
    token: String,
}

#[kernel_function(name = "greet_person", description = "Greet a person by name with optional title.")]
fn greet(
    #[param(description = "Name of the person to greet")] name: String,
    #[param(description = "Optional title (Mr, Ms, Dr, etc.)")] title: Option<String>,
) -> Result<String, std::convert::Infallible> {
    let greeting = if let Some(t) = title {
        format!("Hello, {} {}!", t, name)
    } else {
        format!("Hello, {}!", name)
    };
    Ok(greeting)
}

#[kernel_function(name = "get_weather", description = "Get current weather for a city.")]
fn get_weather(
    #[param(description = "City name")] city: String,
    #[param(description = "Country code (optional)")] country: Option<String>,
) -> Result<String, std::convert::Infallible> {
    // Mock weather data
    let temp = match city.to_lowercase().as_str() {
        "tokyo" => 22,
        "london" => 15,
        "new york" => 25,
        "paris" => 18,
        _ => 20,
    };
    let location = if let Some(c) = country {
        format!("{}, {}", city, c)
    } else {
        city
    };
    Ok(format!("Current weather in {}: {}°C, partly cloudy", location, temp))
}

#[derive(Clone)]
struct MathOperations;

#[kernel_module]
impl MathOperations {
    #[kernel_function(name = "add_numbers", description = "Add two numbers.")]
    fn add(&self, #[param(description = "First number")] a: f64, #[param(description = "Second number")] b: f64) -> Result<f64, std::convert::Infallible> {
        Ok(a + b)
    }

    #[kernel_function(name = "multiply_numbers", description = "Multiply two numbers.")]
    fn multiply(&self, #[param(description = "First number")] a: f64, #[param(description = "Second number")] b: f64) -> Result<f64, std::convert::Infallible> {
        Ok(a * b)
    }

    #[kernel_function(name = "calculate_average", description = "Calculate average of a list of numbers.")]
    fn average(&self, #[param(description = "List of numbers")] numbers: Vec<f64>) -> Result<f64, std::convert::Infallible> {
        if numbers.is_empty() {
            Ok(0.0)
        } else {
            Ok(numbers.iter().sum::<f64>() / numbers.len() as f64)
        }
    }
}

#[kernel_function(name = "search_web", description = "Search the web for information.")]
fn search_web(
    #[param(description = "Search query")] query: String,
    #[param(description = "Maximum number of results")] limit: Option<i32>,
) -> Result<String, std::convert::Infallible> {
    let max_results = limit.unwrap_or(5);
    // Mock search results
    let results = format!("Search results for '{}':\n1. Result 1\n2. Result 2\n3. Result 3\n... (showing {} results)", query, max_results);
    Ok(results)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Set the OPENROUTER_API_KEY environment variable
    std::env::set_var("OPENROUTER_API_KEY", args.token);

    let provider: Arc<dyn LLMProvider> = Arc::new(OpenRouter::from_env()?);

    println!("=== Function Calling Demo ===\n");

    let mut registry = FunctionRegistry::new();
    registry.register(greet_kernel());
    registry.register(get_weather_kernel());
    Arc::new(MathOperations).register_kernel_functions(&mut registry);
    registry.register(search_web_kernel());

    // Demo 1: Simple greeting
    println!("Demo 1: Simple greeting");
    run_demo(
        &provider,
        &registry,
        "Say hello to Dr. Smith using the greeting function.",
    ).await?;
    println!();

    // Demo 2: Math operations
    println!("Demo 2: Math operations");
    run_demo(
        &provider,
        &registry,
        "Calculate 15.5 + 24.3 and then multiply the result by 2.",
    ).await?;
    println!();

    // Demo 3: Weather and search
    println!("Demo 3: Weather and search");
    run_demo(
        &provider,
        &registry,
        "What's the weather like in Tokyo? Also, search for 'best sushi restaurants in Tokyo'.",
    ).await?;
    println!();

    // Demo 4: Complex multi-step
    println!("Demo 4: Complex multi-step operations");
    run_demo(
        &provider,
        &registry,
        "Greet Alice Johnson as Ms. Johnson, get the weather for London, calculate the average of 10, 20, 30, 40, and search for 'London attractions'.",
    ).await?;
    println!();

    // Demo 5: Error handling scenario
    println!("Demo 5: Handling multiple requests");
    run_demo(
        &provider,
        &registry,
        "First greet Bob, then calculate 100 * 5, get weather for Paris, and finally search for 'Paris museums'.",
    ).await?;
    println!();

    println!("All function calling demos completed!");

    Ok(())
}

async fn run_demo(
    provider: &Arc<dyn LLMProvider>,
    registry: &FunctionRegistry,
    user_message: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("User: {}", user_message);

    let mut messages = vec![
        ChatMessage::system("You are a helpful assistant with access to various tools. Use them whenever appropriate to provide accurate information."),
        ChatMessage::user(user_message.to_string()),
    ];

    let mut request = CompletionRequest::new("openai/gpt-4o-mini", messages.clone())
        .with_function_registry(registry);

    let mut response = provider.complete(request).await?;
    println!("Assistant: {}", response.message.text().unwrap_or("<tool call>"));
    messages.push(response.message.clone());

    // Handle tool calls
    let mut tool_call_count = 0;
    while !response.message.tool_calls.is_empty() && tool_call_count < 5 {
        for call in &response.message.tool_calls {
            let result = registry.invoke(&call.function).await?;
            let payload = serde_json::to_string(&result)?;
            let tool_id = call
                .id
                .clone()
                .unwrap_or_else(|| call.function.name.clone());
            println!("Tool {} -> {}", tool_id, payload);
            messages.push(ChatMessage::tool(tool_id, payload));
        }

        request = CompletionRequest::new("openai/gpt-4o-mini", messages.clone())
            .with_function_registry(registry);
        response = provider.complete(request).await?;
        println!("Assistant: {}", response.message.text().unwrap_or("<response>"));
        messages.push(response.message.clone());

        tool_call_count += 1;
    }

    Ok(())
}