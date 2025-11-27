//! Simplified example using the new Flow API
//!
//! This demonstrates how much easier it is to work with flows using the new simplified interface.

use denkwerk::{Flow, kernel_function};
use std::env;
use std::sync::Arc;
use dotenvy::from_path;

// Define a custom function - no boilerplate needed!
#[kernel_function(name = "math_add", description = "Add two numbers")]
fn math_add(a: f64, b: f64) -> Result<f64, String> {
    Ok(a + b)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables from .env file
    let example_dir = env::current_dir()?
        .join("examples")
        .join("prompt_from_yaml_demo");

    if let Err(e) = from_path(example_dir.join(".env")) {
        eprintln!("Warning: Failed to load .env file: {}", e);
        eprintln!("Make sure to copy .env.example to .env and add your API keys");
    }

    // Get the task from command line or use default
    let task = env::args()
        .nth(1)
        .unwrap_or_else(|| "Summarize why Rust is great for backend services.".to_string());

    // Get optional mode parameter
    let mode = env::args().nth(2).or_else(|| env::var("FLOW_MODE").ok());

    println!("Running simplified flow example...");
    println!("Task: {}", task);
    if let Some(mode) = &mode {
        println!("Mode: {}", mode);
    }
    println!();

    // SIMPLIFIED API - Clean and intuitive interface
    let mut flow_builder = Flow::from_directory("examples/prompt_from_yaml_demo")?;

    // Add optional context variable
    if let Some(mode) = mode {
        flow_builder = flow_builder.with_context_var("mode", mode);
    }

    // Add our custom function (auto-discovery would find this too)
    flow_builder = flow_builder.with_function(math_add_kernel());

    // Create provider (auto-detect from environment variables)
    let provider = if let Ok(_) = env::var("AZURE_OPENAI_KEY") {
        println!("Using Azure OpenAI provider");
        Arc::new(denkwerk::providers::azure_openai::AzureOpenAI::from_env()?) as Arc<dyn denkwerk::LLMProvider>
    } else if let Ok(_) = env::var("OPENAI_API_KEY") {
        println!("Using OpenAI provider");
        Arc::new(denkwerk::providers::openai::OpenAI::from_env()?) as Arc<dyn denkwerk::LLMProvider>
    } else {
        return Err("No LLM provider configured. Set AZURE_OPENAI_KEY or OPENAI_API_KEY environment variable in .env file".into());
    };

    // Run the flow!
    let result = flow_builder
        .with_provider(provider)
        .run(task).await?;

    // Display results
    println!("Flow completed successfully!");
    println!();
    println!("Execution Summary:");
    println!("  - Agent steps: {}", result.agent_steps());
    println!("  - Tool executions: {}", result.tool_executions());
    println!("  - Success: {}", result.is_success());
    println!();

    println!("Final Output:");
    println!("{}", result.output());

    Ok(())
}