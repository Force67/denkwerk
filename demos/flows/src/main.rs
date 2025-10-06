use clap::Parser;
use denkwerk::providers::openrouter::OpenRouter;
use denkwerk::{
    Agent, ChatMessage, CompletionRequest, ConcurrentOrchestrator, FunctionRegistry,
    GroupChatOrchestrator, HandoffOrchestrator, LLMProvider, MagenticManager, MagenticOrchestrator,
    RoundRobinGroupChatManager, SequentialOrchestrator, kernel_function, kernel_module,
};
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "flows-demo")]
#[command(about = "Demo app showcasing all denkwerk agent flows")]
struct Args {
    /// OpenRouter API token
    #[arg(long)]
    token: String,
}

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
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Set the OPENROUTER_API_KEY environment variable
    std::env::set_var("OPENROUTER_API_KEY", args.token);

    let provider: Arc<dyn LLMProvider> = Arc::new(OpenRouter::from_env()?);

    println!("=== Denkwerk Flows Demo ===\n");

    // 1. Function Calling Demo
    println!("1. Function Calling Demo");
    demo_function_calling(&provider).await?;
    println!();

    // 2. Sequential Orchestrator Demo
    println!("2. Sequential Orchestrator Demo");
    demo_sequential(&provider).await?;
    println!();

    // 3. Concurrent Orchestrator Demo
    println!("3. Concurrent Orchestrator Demo");
    demo_concurrent(&provider).await?;
    println!();

    // 4. Group Chat Demo
    println!("4. Group Chat Demo");
    demo_group_chat(&provider).await?;
    println!();

    // 5. Magentic Demo
    println!("5. Magentic Demo");
    demo_magentic(&provider).await?;
    println!();

    // 6. Handoff Demo
    println!("6. Handoff Demo");
    demo_handoff(&provider).await?;
    println!();

    println!("All demos completed!");

    Ok(())
}

async fn demo_function_calling(provider: &Arc<dyn LLMProvider>) -> Result<(), Box<dyn std::error::Error>> {
    let mut registry = FunctionRegistry::new();
    registry.register(greet_kernel());
    Arc::new(Calculator).register_kernel_functions(&mut registry);

    let messages = vec![
        ChatMessage::system("You are an assistant that must use tools whenever possible."),
        ChatMessage::user("Say hello to Alice and compute 10.5 + 7.25 using the tools."),
    ];

    let request = CompletionRequest::new("openai/gpt-4o-mini", messages)
        .with_function_registry(&registry);

    let response = provider.complete(request).await?;
    println!("Assistant: {}", response.message.text().unwrap_or("<tool call>"));

    Ok(())
}

async fn demo_sequential(provider: &Arc<dyn LLMProvider>) -> Result<(), Box<dyn std::error::Error>> {
    let analyst = Agent::from_string(
        "Analyst",
        "You are an analyst. Extract key features from product descriptions.",
    )
    .with_description("Extracts features from product descriptions.");

    let writer = Agent::from_string(
        "Writer",
        "You are a writer. Create marketing copy based on analysis.",
    )
    .with_description("Creates marketing copy.");

    let orchestrator = SequentialOrchestrator::new(provider.clone(), "openai/gpt-4o-mini")
        .with_agents(vec![analyst, writer]);

    let product_description = "A compact wireless charger that works with all Qi-enabled devices.";

    let run = orchestrator.run(product_description).await?;

    for event in &run.events {
        match event {
            denkwerk::SequentialEvent::Step { agent, output } => {
                println!("{agent}: {output}");
            }
            denkwerk::SequentialEvent::Completed { agent, output } => {
                if let Some(result) = output {
                    println!("{agent} finalized: {result}");
                }
            }
        }
    }

    Ok(())
}

async fn demo_concurrent(provider: &Arc<dyn LLMProvider>) -> Result<(), Box<dyn std::error::Error>> {
    let physics = Agent::from_string(
        "PhysicsExpert",
        "Explain phenomena from physics perspective.",
    )
    .with_description("Explains from physics standpoint.");

    let chemistry = Agent::from_string(
        "ChemistryExpert",
        "Explain phenomena using chemistry concepts.",
    )
    .with_description("Explains using chemistry.");

    let orchestrator = ConcurrentOrchestrator::new(provider.clone(), "openai/gpt-4o-mini")
        .with_agents(vec![physics, chemistry]);

    let run = orchestrator.run("What is temperature?").await?;

    for result in &run.results {
        println!("[{}] {}", result.agent, result.output.as_deref().unwrap_or("(no output)"));
    }

    Ok(())
}

async fn demo_group_chat(provider: &Arc<dyn LLMProvider>) -> Result<(), Box<dyn std::error::Error>> {
    let writer = Agent::from_string(
        "CopyWriter",
        "Create marketing slogans.",
    )
    .with_description("Drafts slogans.");

    let reviewer = Agent::from_string(
        "Reviewer",
        "Evaluate slogans.",
    )
    .with_description("Evaluates slogans.");

    let manager = RoundRobinGroupChatManager::new().with_maximum_rounds(Some(3));
    let mut orchestrator = GroupChatOrchestrator::new(provider.clone(), "openai/gpt-4o-mini", manager)
        .with_agents(vec![writer, reviewer]);

    let run = orchestrator.run("Create a slogan for an eco-friendly water bottle.").await?;

    for event in &run.events {
        match event {
            denkwerk::GroupChatEvent::AgentMessage { agent, message } => {
                println!("{agent}: {message}");
            }
            denkwerk::GroupChatEvent::AgentCompletion { agent, message } => {
                if let Some(text) = message {
                    println!("{agent} completed: {text}");
                }
            }
            _ => {}
        }
    }

    Ok(())
}

async fn demo_magentic(provider: &Arc<dyn LLMProvider>) -> Result<(), Box<dyn std::error::Error>> {
    let manager = MagenticManager::standard();

    let research = Agent::from_string(
        "ResearchAgent",
        "Research and provide information.",
    )
    .with_description("Surfaces information.");

    let coder = Agent::from_string(
        "CoderAgent",
        "Analyze data and structure results.",
    )
    .with_description("Structures results.");

    let mut orchestrator = MagenticOrchestrator::new(provider.clone(), "openai/gpt-4o-mini", manager)
        .with_max_rounds(5);

    orchestrator.register_agent(research)?;
    orchestrator.register_agent(coder)?;

    let brief = "Compare energy efficiency of ResNet-50 and BERT-base models.";

    let run = orchestrator.run(brief).await?;

    for event in run.events {
        match event {
            denkwerk::MagenticEvent::AgentMessage { agent, message } => {
                println!("{agent}: {message}");
            }
            denkwerk::MagenticEvent::Completed { message } => {
                println!("Final: {message}");
            }
            _ => {}
        }
    }

    Ok(())
}

async fn demo_handoff(provider: &Arc<dyn LLMProvider>) -> Result<(), Box<dyn std::error::Error>> {
    let weather_agent = Agent::from_string(
        "weather",
        "Provide weather briefings. Reply with JSON: {\"action\":\"respond\",\"message\":\"<guidance>\"} or {\"action\":\"complete\",\"message\":\"<remark>\"}",
    );

    let travel_agent = Agent::from_string(
        "travel",
        "Plan travel itineraries. Can handoff to weather agent.",
    );

    let concierge = Agent::from_string(
        "concierge",
        "Coordinate travel planning. Can handoff to travel or weather agents.",
    );

    let mut orchestrator = HandoffOrchestrator::new(provider.clone(), "openai/gpt-4o-mini")
        .with_max_handoffs(Some(3));

    orchestrator.register_agent(concierge);
    orchestrator.register_agent(travel_agent);
    orchestrator.register_agent(weather_agent);

    let mut session = orchestrator.session("concierge")?;

    let user_message = "Help me plan a trip to Paris next week.";
    println!("User: {user_message}");

    let turn = session.send(user_message).await?;

    for event in &turn.events {
        match event {
            denkwerk::HandoffEvent::Message { agent, message } => {
                println!("{agent}: {message}");
            }
            denkwerk::HandoffEvent::HandOff { from, to } => {
                println!("[handoff] {from} -> {to}");
            }
            _ => {}
        }
    }

    Ok(())
}