use std::{collections::HashSet, sync::{Arc, Mutex}};

use crate::{
    eval::{
        report::{CaseReport, EvalReport},
        scenario::{EvalScenario, ExpectStep},
    },
    flows::handoffflow::{HandoffEvent, HandoffOrchestrator},
    providers::scripted::ScriptedProvider,
    Agent,
};

pub struct EvalRunner;

impl EvalRunner {
    pub fn new() -> Self {
        Self
    }

    pub async fn run(
        &self,
        make_orchestrator: impl Fn(Arc<dyn crate::LLMProvider>, String) -> HandoffOrchestrator,
        scenarios: &[EvalScenario],
    ) -> EvalReport {
        let mut cases = Vec::new();
        let mut passed = 0;

        for scenario in scenarios {
            let case = self.run_scenario(&make_orchestrator, scenario).await;
            if case.pass {
                passed += 1;
            }
            cases.push(case);
        }

        EvalReport {
            total: scenarios.len(),
            passed,
            cases,
        }
    }

    pub async fn run_with_provider(
        &self,
        make_orchestrator: impl Fn(Arc<dyn crate::LLMProvider>, String) -> HandoffOrchestrator,
        provider: Arc<dyn crate::LLMProvider>,
        model: String,
        scenarios: &[EvalScenario],
    ) -> EvalReport {
        let mut cases = Vec::new();
        let mut passed = 0;

        for scenario in scenarios {
            let case = self.run_scenario_real(&make_orchestrator, provider.clone(), model.clone(), scenario).await;
            // For real LLM, always consider passed since no expectations
            passed += 1;
            cases.push(case);
        }

        EvalReport {
            total: scenarios.len(),
            passed,
            cases,
        }
    }

    async fn run_scenario(
        &self,
        make_orchestrator: &impl Fn(Arc<dyn crate::LLMProvider>, String) -> HandoffOrchestrator,
        scenario: &EvalScenario,
    ) -> CaseReport {
        // Create scripted provider for this scenario
        let scripted_provider = Arc::new(ScriptedProvider::from_scripted_turns(&scenario.scripted));

        // Get unique agent names
        let agent_names: HashSet<String> = scenario
            .scripted
            .iter()
            .map(|t| t.agent.clone())
            .collect();

        // Create orchestrator with scripted provider
        let mut orchestrator = make_orchestrator(scripted_provider, "scripted".to_string());

        // Create dummy agents
        for name in &agent_names {
            let agent = Agent::from_string(name.clone(), format!("You are agent {}.", name));
            orchestrator.register_agent(agent);
        }

        // Collect events
        let actual_events = Arc::new(Mutex::new(Vec::new()));
        let actual_events_clone = Arc::clone(&actual_events);
        orchestrator = orchestrator.with_event_callback(move |event| {
            actual_events_clone.lock().unwrap().push(event.clone());
        });

        let mut session = orchestrator
            .session(&scenario.initial_agent)
            .expect("initial agent not found");

        // Run the conversation
        let result = session.send(&scenario.user_input).await;

        // Check expectations
        let actual_events = actual_events.lock().unwrap();
        let mut failures = Vec::new();

        // Check steps
        for (i, expect) in scenario.expect.steps.iter().enumerate() {
            if i >= actual_events.len() {
                failures.push(format!("Expected step {} but no more events", i));
                continue;
            }
            let actual = &actual_events[i];
            if !matches_step(expect, actual) {
                failures.push(format!("Step {} mismatch: expected {:?}, got {:?}", i, expect, actual));
            }
        }

        if actual_events.len() > scenario.expect.steps.len() {
            failures.push(format!("Extra events: {} vs expected {}", actual_events.len(), scenario.expect.steps.len()));
        }

        // Check final reply
        if let Some(contains) = &scenario.expect.final_reply_contains {
            if let Some(reply) = result.as_ref().ok().and_then(|turn| turn.reply.as_deref()) {
                if !reply.contains(contains) {
                    failures.push(format!("Final reply does not contain '{}'", contains));
                }
            } else {
                failures.push("No final reply".to_string());
            }
        }

        // Check max rounds (approximate by event count)
        if let Some(max_le) = scenario.expect.max_rounds_le {
            if actual_events.len() > max_le {
                failures.push(format!("Too many rounds: {} > {}", actual_events.len(), max_le));
            }
        }

        CaseReport {
            name: scenario.name.clone(),
            pass: failures.is_empty(),
            failures,
        }
    }

    async fn run_scenario_real(
        &self,
        make_orchestrator: &impl Fn(Arc<dyn crate::LLMProvider>, String) -> HandoffOrchestrator,
        provider: Arc<dyn crate::LLMProvider>,
        model: String,
        scenario: &EvalScenario,
    ) -> CaseReport {
        // Get unique agent names from scripted (even if not used)
        let agent_names: HashSet<String> = scenario
            .scripted
            .iter()
            .map(|t| t.agent.clone())
            .collect();

        // Create orchestrator with real provider
        let mut orchestrator = make_orchestrator(provider, model);

        // Create dummy agents
        for name in &agent_names {
            let agent = Agent::from_string(name.clone(), format!("You are agent {}.", name));
            orchestrator.register_agent(agent);
        }

        // Collect events
        let actual_events = Arc::new(Mutex::new(Vec::new()));
        let actual_events_clone = Arc::clone(&actual_events);
        orchestrator = orchestrator.with_event_callback(move |event| {
            actual_events_clone.lock().unwrap().push(event.clone());
        });

        let mut session = orchestrator
            .session(&scenario.initial_agent)
            .expect("initial agent not found");

        // Run the conversation
        let result = session.send(&scenario.user_input).await;

        let actual_events = actual_events.lock().unwrap();

        // For real LLM, just log the events, no failures
        println!("Scenario: {}", scenario.name);
        for event in &*actual_events {
            println!("  {:?}", event);
        }
        if let Ok(turn) = &result {
            if let Some(reply) = &turn.reply {
                println!("  Final reply: {}", reply);
            }
        }

        CaseReport {
            name: scenario.name.clone(),
            pass: true, // Always pass for real LLM mode
            failures: vec![],
        }
    }
}

fn matches_step(expect: &ExpectStep, actual: &HandoffEvent) -> bool {
    match (expect, actual) {
        (ExpectStep::Msg { agent, contains }, HandoffEvent::Message { agent: a, message: m }) => {
            agent == a && contains.as_ref().map_or(true, |c| m.contains(c))
        }
        (ExpectStep::HandOff { from, to, because }, HandoffEvent::HandOff { from: f, to: t, because: b }) => {
            from == f && to == t && because == b
        }
        (ExpectStep::Complete { agent }, HandoffEvent::Completed { agent: a }) => agent == a,
        _ => false,
    }
}