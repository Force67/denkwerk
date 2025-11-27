use clap::Parser;
use colored::Colorize;
use denkwerk::providers::openrouter::OpenRouter;
use denkwerk::{
    Agent, ChatMessage, CompletionRequest, ConcurrentOrchestrator, FunctionRegistry,
    GroupChatOrchestrator, HandoffOrchestrator, LLMProvider, MagenticManager, MagenticOrchestrator,
    RoundRobinGroupChatManager, SequentialOrchestrator, kernel_function, kernel_module,
};
use std::sync::Arc;

fn colorize_agent(agent_name: &str) -> colored::ColoredString {
    match agent_name.to_lowercase().as_str() {
        // Sequential demo agents
        "analyst" => agent_name.bright_blue().bold(),
        "writer" => agent_name.bright_green().bold(),

        // Concurrent demo agents
        "physicsexpert" => agent_name.bright_magenta().bold(),
        "chemistryexpert" => agent_name.bright_cyan().bold(),

        // Group chat demo agents
        "copywriter" => agent_name.bright_yellow().bold(),
        "reviewer" => agent_name.bright_red().bold(),

        // Magentic demo agents
        "researchagent" => agent_name.bright_blue().bold(),
        "coderagent" => agent_name.bright_green().bold(),

        // Handoff demo agents
        "concierge" => agent_name.bright_cyan().bold(),
        "travel" => agent_name.bright_yellow().bold(),
        "weather" => agent_name.bright_magenta().bold(),

        // Default fallback
        _ => agent_name.white().bold(),
    }
}

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

    /*
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
    println!();*/

    // 5. Magentic Demo
    //println!("5. Magentic Demo");
    //demo_magentic(&provider).await?;
    //println!();

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

    let mut messages = vec![
        ChatMessage::system("You are an assistant that must use tools whenever possible."),
        ChatMessage::user("Say hello to Alice and compute 10.5 + 7.25 using the tools."),
    ];

    let mut request = CompletionRequest::new("openai/gpt-4o-mini", messages.clone())
        .with_function_registry(&registry);

    let mut response = provider.complete(request).await?;
    println!("Assistant: {}", response.message.text().unwrap_or("<tool call>"));
    messages.push(response.message.clone());

    // Handle tool calls
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
            let payload = serde_json::to_string(&result)?;
            let tool_id = call
                .id
                .clone()
                .unwrap_or_else(|| call.function.name.clone());
            println!("tool {} -> {}", tool_id, payload);
            messages.push(ChatMessage::tool(tool_id, payload));
        }

        request = CompletionRequest::new("openai/gpt-4o-mini", messages.clone())
            .with_function_registry(&registry);
        response = provider.complete(request).await?;
        println!(
            "Assistant (final): {}",
            response.message.text().unwrap_or("<no response>")
        );
    }

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
                println!("{}: {}", colorize_agent(agent), output);
            }
            denkwerk::SequentialEvent::Completed { agent, output } => {
                if let Some(result) = output {
                    println!("{} finalized: {}", colorize_agent(agent), result);
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
        println!("[{}] {}", colorize_agent(&result.agent), result.output.as_deref().unwrap_or("(no output)"));
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
                println!("{}: {}", colorize_agent(agent), message);
            }
            denkwerk::GroupChatEvent::AgentCompletion { agent, message } => {
                if let Some(text) = message {
                    println!("{} completed: {}", colorize_agent(agent), text);
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
        "Research and provide information. When you have gathered relevant data, respond with your findings and suggest next steps.",
    )
    .with_description("Surfaces information.");

    let coder = Agent::from_string(
        "CoderAgent",
        "Analyze data and structure results. When you have completed your analysis, provide a clear summary and conclusion.",
    )
    .with_description("Structures results.");

    let mut orchestrator = MagenticOrchestrator::new(provider.clone(), "openai/gpt-4o-mini", manager)
        .with_max_rounds(10);

    orchestrator.register_agent(research)?;
    orchestrator.register_agent(coder)?;

    let brief = "Compare the training energy efficiency of ResNet-50 and BERT-base models. Focus on key differences and provide a brief analysis.";

    let run = orchestrator.run(brief).await?;

    for event in run.events {
        match event {
            denkwerk::MagenticEvent::ManagerMessage { message } => {
                println!("{}: {}", colorize_agent("manager"), message);
            }
            denkwerk::MagenticEvent::ManagerDelegation { target, instructions, progress_note } => {
                println!("{}", format!("{} delegates to {}: {}", colorize_agent("manager"), colorize_agent(&target), instructions).cyan().bold());
                if let Some(note) = progress_note {
                    println!("{}: {}", colorize_agent("manager"), note);
                }
            }
            denkwerk::MagenticEvent::AgentMessage { agent, message } => {
                println!("{}: {}", colorize_agent(&agent), message);
            }
            denkwerk::MagenticEvent::AgentCompletion { agent, message } => {
                if let Some(msg) = message {
                    println!("{} completes: {}", colorize_agent(&agent), msg);
                } else {
                    println!("{} completes task", colorize_agent(&agent));
                }
            }
            denkwerk::MagenticEvent::Completed { message } => {
                println!("{}: {}", "Final Result".bright_green().bold(), message);
            }
        }
    }

    Ok(())
}

async fn demo_handoff(provider: &Arc<dyn LLMProvider>) -> Result<(), Box<dyn std::error::Error>> {
    let weather_agent = Agent::from_string(
        "weather",
        "You are the Weather Specialist. Provide detailed weather forecasts, temperature ranges, and packing recommendations for specific destinations and dates.
        Focus only on weather information.",
    );

    let travel_agent = Agent::from_string(
        "travel",
        r#"You are a Travel Planner. Focus on flights, hotels, and transportation. Provide specific recommendations and booking advice.
        When you receive a travel planning request that mentions weather information, first provide your travel recommendations, then hand off to the weather specialist.
        If weather is not mentioned, complete the travel planning with {"action": "complete", "message": "Travel planning complete"}."#,
    );

    let concierge = Agent::from_string(
        "concierge",
        r#"You are a Concierge Coordinator. Your role is to coordinate travel planning - DO NOT provide specific flight, hotel, or weather details yourself.
        Instead, delegate to the appropriate specialists for detailed information.
        When you receive a travel planning request, start by handing off to the travel agent for the main planning.
        The travel agent will handle weather information by coordinating with the weather specialist if needed. Just start with the travel agent."#,
    );

    let mut orchestrator = HandoffOrchestrator::new(provider.clone(), "openai/gpt-4o-mini")
        .with_max_handoffs(Some(5));

    orchestrator.register_agent(concierge);
    orchestrator.register_agent(travel_agent);
    orchestrator.register_agent(weather_agent);

    let mut session = orchestrator.session("concierge")?;

    let user_message = "I need to plan a business trip to Paris from Denver. I'll be there for 3 days next week. Can you help me with flights, hotels, and weather information?";
    println!("User: {user_message}");

    let turn = session.send(user_message).await?;

    let mut last_agent_message: Option<(String, String)> = None;

    for event in &turn.events {
        match event {
            denkwerk::HandoffEvent::Message { agent, message } => {
                println!("{}: {}", colorize_agent(agent), message);
                last_agent_message = Some((agent.clone(), message.clone()));
            }
            denkwerk::HandoffEvent::HandOff { from, to } => {
                // Show the agent's reasoning for the handoff in color
                if let Some((agent, reasoning)) = &last_agent_message {
                    if agent == from {
                        println!("{}", format!("🤔 {} thinks: \"{reasoning}\"", colorize_agent(agent)).cyan().bold());
                    }
                }
                println!("{}", format!("🔄 [handoff] {} -> {}", colorize_agent(from), colorize_agent(to)).yellow().bold());
            }
            denkwerk::HandoffEvent::Completed { agent } => {
                println!("{} completed the task", colorize_agent(agent));
            }
        }
    }

    Ok(())
}
