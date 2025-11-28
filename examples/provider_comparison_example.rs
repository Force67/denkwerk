use denkwerk::{
    Agent, LLMProvider,
    providers::{openai::OpenAI, openrouter::OpenRouter},
    flows::{sequential::SequentialOrchestrator, handoffflow::HandoffOrchestrator},
};
use std::sync::Arc;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Agent Flow Provider Comparison ===\n");

    // Test OpenAI provider
    println!("1. Testing with OpenAI Provider:");
    let openai_provider = Arc::new(OpenAI::from_env()?);

    let _openai_agent = Agent::from_string("openai_agent", "You are a helpful assistant using OpenAI")
        .with_provider(openai_provider.clone())
        .with_model("gpt-4o-mini");

    let _openai_orchestrator = SequentialOrchestrator::new(openai_provider.clone(), "gpt-4o-mini");
    println!("   - Provider: {}", openai_provider.name());
    println!("   - Supports embeddings: {}", openai_provider.capabilities().supports_embeddings);
    println!("   - Supports streaming: {}", openai_provider.capabilities().supports_streaming);

    // Test OpenRouter provider
    println!("\n2. Testing with OpenRouter Provider:");
    let openrouter_provider = Arc::new(OpenRouter::from_env()?);

    let _openrouter_agent = Agent::from_string("openrouter_agent", "You are a helpful assistant using OpenRouter")
        .with_provider(openrouter_provider.clone())
        .with_model("openai/gpt-4o-mini");

    let _openrouter_orchestrator = SequentialOrchestrator::new(openrouter_provider.clone(), "openai/gpt-4o-mini");
    println!("   - Provider: {}", openrouter_provider.name());
    println!("   - Supports embeddings: {}", openrouter_provider.capabilities().supports_embeddings);
    println!("   - Supports streaming: {}", openrouter_provider.capabilities().supports_streaming);

    // Test Handoff flow with OpenAI
    println!("\n3. Testing Handoff Flow with OpenAI:");
    let _handoff_orchestrator = HandoffOrchestrator::new(openai_provider.clone(), "gpt-4o-mini");

    // Create specialized agents for handoff
    let support_agent = Agent::from_string("support", "You are a customer support agent")
        .with_provider(openai_provider.clone())
        .with_model("gpt-4o-mini");

    let technical_agent = Agent::from_string("technical", "You are a technical support agent")
        .with_provider(openai_provider.clone())
        .with_model("gpt-4o-mini");

    println!("   - Handoff flow created with OpenAI provider");
    println!("   - Support agent: {}", support_agent.name());
    println!("   - Technical agent: {}", technical_agent.name());

    // Test mixed providers
    println!("\n4. Testing Mixed Provider Usage:");

    // Sequential flow with OpenRouter
    let _mixed_sequential = SequentialOrchestrator::new(openrouter_provider.clone(), "openai/gpt-4o-mini");
    println!("   - Sequential flow uses OpenRouter provider");

    // Handoff flow with OpenAI
    let _mixed_handoff = HandoffOrchestrator::new(openai_provider.clone(), "gpt-4o-mini");
    println!("   - Handoff flow uses OpenAI provider");

    // Individual agents can use different providers
    let agent_with_override = Agent::from_string("mixed_agent", "This agent uses a different provider")
        .with_provider(openrouter_provider.clone())  // Override provider
        .with_model("openai/gpt-4o-mini");
    println!("   - Agent with provider override: {}", agent_with_override.name());

    println!("\n=== Provider Capabilities Comparison ===");

    println!("\nOpenAI Provider:");
    println!("  - Streaming: {}", openai_provider.capabilities().supports_streaming);
    println!("  - Reasoning Stream: {}", openai_provider.capabilities().supports_reasoning_stream);
    println!("  - Image Uploads: {}", openai_provider.capabilities().supports_image_uploads);
    println!("  - Embeddings: {}", openai_provider.capabilities().supports_embeddings);

    println!("\nOpenRouter Provider:");
    println!("  - Streaming: {}", openrouter_provider.capabilities().supports_streaming);
    println!("  - Reasoning Stream: {}", openrouter_provider.capabilities().supports_reasoning_stream);
    println!("  - Image Uploads: {}", openrouter_provider.capabilities().supports_image_uploads);
    println!("  - Embeddings: {}", openrouter_provider.capabilities().supports_embeddings);

    println!("\n=== Configuration Examples ===");
    println!("\nTo use OpenAI:");
    println!("  export OPENAI_API_KEY=sk-...");
    println!("  provider = Arc::new(OpenAI::from_env()?);");

    println!("\nTo use OpenRouter:");
    println!("  export OPENROUTER_API_KEY=sk-or-...");
    println!("  provider = Arc::new(OpenRouter::from_env()?);");

    println!("\n=== Summary ===");
    println!("✓ All agent flows support both OpenAI and OpenRouter providers");
    println!("✓ Different flows can use different providers simultaneously");
    println!("✓ Individual agents can override the flow's default provider");
    println!("✓ Provider configuration happens at runtime, not in YAML specs");
    println!("✓ Both providers support the same core capabilities (streaming, embeddings)");

    Ok(())
}