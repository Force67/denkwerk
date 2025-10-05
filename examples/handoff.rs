use std::sync::Arc;

use denkwerk::{
    Agent, AgentError, HandoffEvent, HandoffOrchestrator, HandoffSession, HandoffTurn,
    LLMProvider,
};
use denkwerk::providers::openrouter::OpenRouter;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider: Arc<dyn LLMProvider> = Arc::new(OpenRouter::from_env()?);

    let weather_agent = Agent::from_string(
        "weather",
        r#"
You are the Weather Advisor. Offer short weather briefings and packing suggestions based on the destination and travel dates.

Reply with JSON only:
- {"action":"respond","message":"<weather guidance>"}
- {"action":"complete","message":"<closing remark>"}

If you do not have live data, share typical seasonal expectations and remind the traveler to verify conditions closer to departure.
"#,
    );

    let travel_agent = Agent::from_handlebars_file(
        "travel",
        "examples/prompts/travel_agent.hbs",
        &json!({
            "name": "Travel Planner",
            "services": [
                "Flight research",
                "Hotel suggestions",
                "Ground transportation advice",
            ],
            "weather_agent": "weather",
        }),
    )?;

    let concierge = Agent::from_handlebars_file(
        "concierge",
        "examples/prompts/front_desk_agent.hbs",
        &json!({
            "name": "Concierge Coordinator",
            "agents": [
                {"id": "travel", "description": "Designs full travel itineraries"},
                {"id": "weather", "description": "Summarizes expected weather"},
            ],
        }),
    )?;

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
    for event in &turn.events {
        match event {
            HandoffEvent::Message { agent, message } => {
                println!("{agent}: {message}");
            }
            HandoffEvent::HandOff { from, to } => {
                println!("[handoff] {from} -> {to}");
            }
            HandoffEvent::Completed { agent } => {
                println!("[completed by {agent}]");
            }
        }
    }
}
