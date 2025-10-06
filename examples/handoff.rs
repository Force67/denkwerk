use std::sync::Arc;

use colored::Colorize;
use denkwerk::{
    Agent, AgentError, HandoffEvent, HandoffOrchestrator, HandoffSession, HandoffTurn,
    LLMProvider,
};
use denkwerk::providers::openrouter::OpenRouter;

fn colorize_agent(agent_name: &str) -> colored::ColoredString {
    match agent_name.to_lowercase().as_str() {
        "concierge" => agent_name.bright_cyan().bold(),
        "travel" => agent_name.bright_yellow().bold(),
        "weather" => agent_name.bright_magenta().bold(),
        _ => agent_name.white().bold(),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider: Arc<dyn LLMProvider> = Arc::new(OpenRouter::from_env()?);

    let weather_agent = Agent::from_string(
        "weather",
        "You are the Weather Advisor. Provide weather briefings and packing suggestions for destinations. Share typical seasonal expectations and remind travelers to verify current conditions.",
    );

    let travel_agent = Agent::from_string(
        "travel",
        "You are a Travel Planner. Research flights, suggest hotels, and provide transportation advice. Create concise travel itineraries with key details and reminders.",
    );

    let concierge = Agent::from_string(
        "concierge",
        "You are a Concierge Coordinator. Help users with travel planning requests. Coordinate between travel and weather specialists as needed.",
    );

    let mut orchestrator = HandoffOrchestrator::new(provider, "openai/gpt-4o-mini")
        .with_max_handoffs(Some(4));

    orchestrator.register_agent(concierge);
    orchestrator.register_agent(travel_agent);
    orchestrator.register_agent(weather_agent);

    let mut session = orchestrator.session("concierge")?;

    run_demo(&mut session).await?;

    Ok(())
}

async fn run_demo(session: &mut HandoffSession<'_>) -> Result<(), AgentError> {
    let script = [
        "Hi there! I need help planning a trip to Seattle the week of October 7th for work.",
        "I'll be leaving from Denver and morning flights would be best.",
        "Could you also tell me what kind of weather to expect that week?",
        "Great, thanks for the help!",
    ];

    for user in script {
        println!("\nUser: {user}");
        let turn = session.send(user).await?;
        render_turn(&turn);

        if turn
            .events
            .iter()
            .any(|event| matches!(event, HandoffEvent::Completed { .. }))
        {
            break;
        }
    }

    Ok(())
}

fn render_turn(turn: &HandoffTurn) {
    let mut last_agent_message: Option<(String, String)> = None;

    for event in &turn.events {
        match event {
            HandoffEvent::Message { agent, message } => {
                println!("{}: {}", colorize_agent(agent), message);
                last_agent_message = Some((agent.clone(), message.clone()));
            }
            HandoffEvent::HandOff { from, to } => {
                // Show the agent's reasoning for the handoff in color
                if let Some((agent, reasoning)) = &last_agent_message {
                    if agent == from {
                        println!("{}", format!("🤔 {} thinks: \"{reasoning}\"", colorize_agent(agent)).cyan().bold());
                    }
                }
                println!("{}", format!("🔄 [handoff] {} -> {}", colorize_agent(from), colorize_agent(to)).yellow().bold());
            }
            HandoffEvent::Completed { agent } => {
                println!("{}", format!("[completed by {}]", colorize_agent(agent)).green().bold());
            }
        }
    }
}
