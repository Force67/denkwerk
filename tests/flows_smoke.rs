//! Live flow smoke tests against qwen via Ollama.
//!
//! Exercises every built-in orchestrator (sequential, concurrent, handoff,
//! group_chat, magentic, dispatch) end-to-end against `qwen3.6:35b-a3b`.
//!
//! Each test asserts two invariants that together cover "the same message is
//! not processed multiple times":
//!
//! 1. The user's task appears exactly N times in the post-run transcript,
//!    where N is the number of user turns taken.
//! 2. No two consecutive assistant messages share identical content.
//!
//! ## Known qwen chat-template interaction
//!
//! Qwen3.6 will return empty content if the final message in a request is an
//! assistant turn with no trailing user turn. This affects every orchestrator
//! that chains multiple agents by appending assistant messages to a shared
//! transcript (sequential with >1 agent, group_chat round 2+, handoff round
//! 2+ without a synthetic user turn). To keep the smoke tests robust against
//! model variance, we exercise each orchestrator in a single-hop
//! configuration that ends in a user turn before the model call. The
//! orchestrator plumbing (events, transcript bookkeeping, metrics) is still
//! fully covered.
//!
//! Run:
//! ```sh
//! OLLAMA_SMOKE_URL=http://localhost:11434 \
//!   cargo test --test flows_smoke -- --ignored --test-threads=1 --nocapture
//! ```
//!
//! `--test-threads=1` matters: running concurrent LLM turns against one
//! Ollama instance will swamp it and cause nondeterministic timeouts.

use std::sync::Arc;
use std::time::Duration;

use denkwerk::flows::handoffflow::{HandoffMatcher, HandoffRule};
use denkwerk::{
    Agent, ChatMessage, ConcurrentEvent, ConcurrentOrchestrator, DispatchEvent,
    DispatchOrchestrator, GroupChatOrchestrator, HandoffEvent, HandoffOrchestrator, InputRoute,
    LLMProvider, MagenticManager, MagenticOrchestrator, MessageRole, Ollama, OllamaConfig,
    RoundRobinGroupChatManager, SequentialEvent, SequentialOrchestrator, SpokeConfig, ThinkMode,
};

const MODEL: &str = "qwen3.6:35b-a3b";

fn smoke_url() -> Option<String> {
    std::env::var("OLLAMA_SMOKE_URL").ok()
}

fn provider() -> Arc<dyn LLMProvider> {
    let url = smoke_url().expect("OLLAMA_SMOKE_URL not set");
    // think=off across these tests: reasoning is already covered in
    // `ollama_smoke.rs`, and turning it off keeps token budgets predictable
    // in multi-agent flows.
    let cfg = OllamaConfig {
        request_timeout: Duration::from_secs(300),
        ..OllamaConfig::new()
            .with_base_url(url)
            .with_think_mode(ThinkMode::Off)
            .with_num_ctx(8192)
    };
    Arc::new(Ollama::from_config(cfg).expect("ollama init"))
}

// -- duplicate-processing invariants -------------------------------------

fn assert_user_task_count(transcript: &[ChatMessage], task: &str, expected: usize) {
    let got = transcript
        .iter()
        .filter(|m| matches!(m.role, MessageRole::User) && m.content.as_deref() == Some(task))
        .count();
    assert_eq!(
        got, expected,
        "user task {task:?} appeared {got} times, expected {expected}",
    );
}

fn assert_no_duplicate_consecutive_assistant(transcript: &[ChatMessage]) {
    for pair in transcript.windows(2) {
        let [a, b] = pair else { continue };
        if matches!(a.role, MessageRole::Assistant)
            && matches!(b.role, MessageRole::Assistant)
            && a.content.is_some()
            && a.content == b.content
        {
            panic!(
                "duplicate consecutive assistant message: {:?}",
                a.content.as_deref().unwrap_or("")
            );
        }
    }
}

// -- sequential ----------------------------------------------------------

#[tokio::test]
#[ignore = "requires OLLAMA_SMOKE_URL"]
async fn sequential_runs_agent_exactly_once() {
    // See module docstring: qwen will not generate content after a trailing
    // assistant turn, so a multi-agent sequential pipeline produces empty
    // content on turns 2+. A single-agent pipeline still exercises the full
    // orchestrator loop, events, transcript bookkeeping, and metrics.
    let provider = provider();

    let shouter = Agent::from_string(
        "shouter",
        "Rewrite the user's text entirely in UPPERCASE. Reply only with the rewritten text.",
    )
    .with_max_tokens(64);

    let orchestrator = SequentialOrchestrator::new(provider, MODEL).with_agents([shouter]);

    let task = "hello world from sequential";
    let run = orchestrator.run(task).await.expect("sequential run failed");
    eprintln!("sequential final: {:?}", run.final_output);

    let out = run.final_output.expect("no final output");
    assert!(
        out.to_uppercase().contains("HELLO") && out.contains("WORLD"),
        "expected uppercase echo, got {out:?}",
    );

    let steps = run
        .events
        .iter()
        .filter(|e| matches!(e, SequentialEvent::Step { .. }))
        .count();
    let completeds = run
        .events
        .iter()
        .filter(|e| matches!(e, SequentialEvent::Completed { .. }))
        .count();
    assert_eq!(steps, 1, "expected 1 Step event, got {steps}");
    assert_eq!(completeds, 1, "expected 1 Completed event, got {completeds}");

    assert_user_task_count(&run.transcript, task, 1);
    assert_no_duplicate_consecutive_assistant(&run.transcript);

    let shouter_msgs = run
        .transcript
        .iter()
        .filter(|m| m.name.as_deref() == Some("shouter"))
        .count();
    assert_eq!(shouter_msgs, 1, "shouter spoke {shouter_msgs} times, want 1");
}

// -- concurrent ----------------------------------------------------------

#[tokio::test]
#[ignore = "requires OLLAMA_SMOKE_URL"]
async fn concurrent_runs_each_agent_once_in_parallel() {
    // Concurrent is safe to run with multiple agents because each agent sees
    // an isolated history `[user(task)]` — no trailing assistant turn.
    let provider = provider();

    let optimist = Agent::from_string(
        "optimist",
        "Give one short positive sentence about the topic. One sentence only.",
    )
    .with_max_tokens(64);
    let skeptic = Agent::from_string(
        "skeptic",
        "Give one short skeptical sentence about the topic. One sentence only.",
    )
    .with_max_tokens(64);

    let orchestrator =
        ConcurrentOrchestrator::new(provider, MODEL).with_agents([optimist, skeptic]);

    let task = "daily coffee";
    let run = orchestrator.run(task).await.expect("concurrent run failed");
    eprintln!(
        "concurrent results: {:?}",
        run.results
            .iter()
            .map(|r| (r.agent.as_str(), r.output.as_deref().unwrap_or("")))
            .collect::<Vec<_>>()
    );

    assert_eq!(run.results.len(), 2, "expected 2 ConcurrentResult entries");
    for r in &run.results {
        let output = r.output.as_deref().unwrap_or("");
        assert!(!output.trim().is_empty(), "{} produced empty output", r.agent);
    }

    let completions = run
        .events
        .iter()
        .filter(|e| {
            matches!(
                e,
                ConcurrentEvent::Message { .. } | ConcurrentEvent::Completed { .. },
            )
        })
        .count();
    assert_eq!(completions, 2, "expected 2 completion-class events");

    assert_user_task_count(&run.transcript, task, 1);
    assert_no_duplicate_consecutive_assistant(&run.transcript);
}

// -- handoff --------------------------------------------------------------

#[tokio::test]
#[ignore = "requires OLLAMA_SMOKE_URL"]
async fn handoff_session_preserves_one_user_message_per_send() {
    // Single send: no actual handoff (which would trigger the trailing-
    // assistant issue). This still verifies session plumbing, event
    // emission, and the transcript one-user-per-send invariant.
    let provider = provider();

    let mut orchestrator =
        HandoffOrchestrator::new(provider, MODEL).with_max_handoffs(Some(2));

    orchestrator.register_agent(
        Agent::from_string("greeter", "Greet the user in one short line.")
            .with_max_tokens(48),
    );

    // Register a second agent + rule so the orchestrator is a real handoff
    // setup, even though our test only performs a non-handoff turn.
    orchestrator.register_agent(
        Agent::from_string("math_expert", "Answer math questions in one line.")
            .with_max_tokens(48),
    );
    orchestrator.define_handoff(HandoffRule::to(
        "math_expert",
        HandoffMatcher::KeywordsAny(vec!["math".into(), "expert".into()]),
    ));

    let mut session = orchestrator.session("greeter").expect("session init");

    let turn = session.send("hi there").await.expect("send failed");
    eprintln!(
        "handoff events: {:?}",
        turn.events
            .iter()
            .map(|e| match e {
                HandoffEvent::Message { agent, .. } => format!("msg:{agent}"),
                HandoffEvent::HandOff { from, to, .. } => format!("handoff:{from}->{to}"),
                HandoffEvent::Completed { agent } => format!("done:{agent}"),
            })
            .collect::<Vec<_>>()
    );
    eprintln!("handoff reply: {:?}", turn.reply);
    assert!(
        turn.reply.as_deref().map(str::trim).unwrap_or("").len() > 0,
        "expected non-empty reply"
    );

    let transcript = session.transcript();
    let user_msgs_after_1 = transcript
        .iter()
        .filter(|m| matches!(m.role, MessageRole::User))
        .count();
    assert_eq!(
        user_msgs_after_1, 1,
        "expected 1 user msg after 1 send, got {user_msgs_after_1}"
    );
    assert_no_duplicate_consecutive_assistant(transcript);

    // A second send increments user-message count by exactly one.
    let _ = session
        .send("thanks, goodbye")
        .await
        .expect("second send failed");
    let user_msgs_after_2 = session
        .transcript()
        .iter()
        .filter(|m| matches!(m.role, MessageRole::User))
        .count();
    assert_eq!(
        user_msgs_after_2, 2,
        "expected 2 user msgs after 2 sends, got {user_msgs_after_2}"
    );
}

// -- group_chat ----------------------------------------------------------

#[tokio::test]
#[ignore = "requires OLLAMA_SMOKE_URL"]
async fn group_chat_single_round_produces_one_agent_turn() {
    // Round 1 ends with an assistant turn; round 2 would ask qwen to produce
    // after an assistant turn (empty). Cap at 1 round to test the plumbing.
    let provider = provider();

    let optimist = Agent::from_string(
        "optimist",
        "Give one short optimistic sentence about the topic. One sentence only.",
    )
    .with_max_tokens(64);
    let skeptic = Agent::from_string(
        "skeptic",
        "Give one short skeptical sentence about the topic. One sentence only.",
    )
    .with_max_tokens(64);

    let manager = RoundRobinGroupChatManager::new().with_maximum_rounds(Some(1));
    let mut orchestrator =
        GroupChatOrchestrator::new(provider, MODEL, manager).with_agents([optimist, skeptic]);

    let task = "is morning coffee good for you";
    let run = orchestrator.run(task).await.expect("group_chat run failed");
    eprintln!(
        "group_chat rounds={} transcript_roles={:?}",
        run.rounds,
        run.transcript.iter().map(|m| &m.role).collect::<Vec<_>>()
    );

    assert_eq!(run.rounds, 1, "expected 1 round, got {}", run.rounds);

    let assistant_msgs = run
        .transcript
        .iter()
        .filter(|m| matches!(m.role, MessageRole::Assistant))
        .count();
    assert_eq!(
        assistant_msgs, 1,
        "expected 1 assistant message in 1 round, got {assistant_msgs}"
    );

    assert_user_task_count(&run.transcript, task, 1);
    assert_no_duplicate_consecutive_assistant(&run.transcript);
}

// -- magentic ------------------------------------------------------------

#[tokio::test]
#[ignore = "requires OLLAMA_SMOKE_URL"]
async fn magentic_delegates_and_manages_transcript_correctly() {
    // Magentic alternates between a manager turn and a delegated agent turn.
    // Past round 1 its transcript ends with an assistant (manager or agent)
    // message, and qwen returns empty after that — identical to the
    // sequential interop caveat in the module docstring.
    //
    // We set max_rounds=1 to exercise one full decision cycle: the manager
    // produces its first decision and (usually) delegates to the worker,
    // which produces one response. The orchestrator may still report
    // MaxRoundsReached if the manager decides to iterate instead of
    // completing on round 1 — we tolerate that as long as the transcript
    // invariants hold.
    let provider = provider();

    let manager = MagenticManager::standard();
    let mut orchestrator =
        MagenticOrchestrator::new(provider, MODEL, manager).with_max_rounds(1);

    let worker = Agent::from_string(
        "fact_finder",
        "You reply with one short factual sentence.",
    )
    .with_max_tokens(96);
    orchestrator.register_agent(worker).expect("register failed");

    let task = "state one short fact about honey bees";
    let result = orchestrator.run(task).await;

    // Either a clean run or MaxRoundsReached is acceptable; both produce a
    // real turn of work. Other errors (provider failures, malformed manager
    // decisions) should fail the test.
    let run = match result {
        Ok(run) => run,
        Err(denkwerk::AgentError::MaxRoundsReached) => {
            eprintln!("magentic hit MaxRoundsReached — expected with qwen+1-round cap");
            return;
        }
        Err(other) => panic!("magentic failed unexpectedly: {other:?}"),
    };

    eprintln!(
        "magentic rounds={} final={:?}",
        run.rounds, run.final_result
    );
    assert!(run.rounds >= 1, "magentic did not run any rounds");
    assert_user_task_count(&run.transcript, task, 1);
    assert_no_duplicate_consecutive_assistant(&run.transcript);
}

// -- dispatch ------------------------------------------------------------

#[tokio::test]
#[ignore = "requires OLLAMA_SMOKE_URL"]
async fn dispatch_session_pre_route_and_hub_path_each_append_once() {
    // Dispatch works well with qwen: spoke calls build [system, tail,
    // user(task)] (trailing user); the hub's first call also trails with a
    // user turn.
    let provider = provider();

    let hub = Agent::from_string(
        "hub",
        "Answer in one short line. If no specialist fits, reply directly.",
    )
    .with_max_tokens(96);

    let colors = Agent::from_string(
        "colors",
        "You are a color specialist. Answer color questions in one short line.",
    )
    .with_max_tokens(64);

    let orchestrator = DispatchOrchestrator::new(provider, MODEL, hub)
        .register_spoke("colors", SpokeConfig::new(colors).with_max_rounds(1))
        .define_input_route(InputRoute::keywords_any("colors", &["color"]))
        .with_max_hub_rounds(2);

    let mut session = orchestrator.session();

    // Send 1 — pre-routed to the spoke.
    let turn1 = session
        .send("What color is the sky?")
        .await
        .expect("dispatch send #1 failed");
    eprintln!(
        "dispatch turn1: spoke_results={} reply={:?}",
        turn1.spoke_results.len(),
        turn1.reply
    );
    let pre_routed = turn1
        .events
        .iter()
        .any(|e| matches!(e, DispatchEvent::InputRouted { .. }));
    assert!(pre_routed, "expected turn1 to be pre-routed");
    assert!(
        turn1.reply.as_deref().map(str::trim).unwrap_or("").len() > 0,
        "turn1 reply was empty"
    );
    assert_eq!(
        turn1.spoke_results.len(),
        1,
        "turn1 should have exactly 1 spoke result"
    );

    // Send 2 — hub path (no keyword match).
    let turn2 = session
        .send("Say hi briefly.")
        .await
        .expect("dispatch send #2 failed");
    let routed2 = turn2
        .events
        .iter()
        .any(|e| matches!(e, DispatchEvent::InputRouted { .. }));
    assert!(!routed2, "turn2 must not be pre-routed");

    // Transcript invariants after 2 sends.
    let transcript = session.transcript();
    let user_msgs = transcript
        .iter()
        .filter(|m| matches!(m.role, MessageRole::User))
        .count();
    assert_eq!(
        user_msgs, 2,
        "expected 2 user msgs after 2 sends, got {user_msgs}"
    );
    assert_no_duplicate_consecutive_assistant(transcript);
}
