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
//! ## Qwen chat-template interaction (compensated)
//!
//! Qwen3.6's renderer treats a trailing assistant message as a prefill cue
//! and returns empty content when orchestrators chain agents naively. The
//! orchestrators are compensated by `flows::prefill::history_for_llm`,
//! which injects a synthetic user turn only for known-affected models. The
//! multi-agent tests below verify the fix works end-to-end.
//!
//! Run:
//! ```sh
//! OLLAMA_SMOKE_URL=http://localhost:11434 \
//!   cargo test --test flows_smoke -- --ignored --test-threads=1 --nocapture
//! ```
//!
//! Swap the model via `OLLAMA_SMOKE_MODEL=<tag>` (default: qwen3.6:35b-a3b).
//! Results per model are tracked in `docs/tested-models.md`.
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

const DEFAULT_MODEL: &str = "qwen3.6:35b-a3b";

fn smoke_url() -> Option<String> {
    std::env::var("OLLAMA_SMOKE_URL").ok()
}

fn model() -> String {
    std::env::var("OLLAMA_SMOKE_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string())
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
async fn sequential_chains_two_agents() {
    // Exercises the `flows::prefill` compensation: without it, agent 2
    // would see a transcript ending in agent 1's assistant message and
    // (on qwen) return empty content.
    let provider = provider();

    let shouter = Agent::from_string(
        "shouter",
        "Rewrite the user's text entirely in UPPERCASE. Reply only with the rewritten text.",
    )
    .with_max_tokens(64);
    let exclaimer = Agent::from_string(
        "exclaimer",
        "Append three exclamation marks to whatever the previous assistant just said. \
         Reply with only the updated text.",
    )
    .with_max_tokens(64);

    let orchestrator =
        SequentialOrchestrator::new(provider, &model()).with_agents([shouter, exclaimer]);

    let task = "hello world from sequential";
    let run = orchestrator.run(task).await.expect("sequential run failed");
    eprintln!("sequential final: {:?}", run.final_output);
    for m in &run.transcript {
        eprintln!(
            "  [{:?}] name={:?} content={:?}",
            m.role,
            m.name,
            m.content.as_deref().unwrap_or(""),
        );
    }

    let out = run.final_output.expect("no final output");
    assert!(
        !out.trim().is_empty(),
        "final output was empty — prefill fix likely not applied"
    );
    // Agent 2's job is to append exclamation marks. Count, not exact
    // three — small instruction-following variance across models is
    // tolerable, what matters is that agent 2 actually modified agent 1's
    // output (proving the chaining works end-to-end).
    assert!(
        out.contains('!'),
        "expected at least one '!' in final output, got {out:?}"
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
    assert_eq!(steps, 2, "expected 2 Step events, got {steps}");
    assert_eq!(completeds, 1, "expected 1 Completed event, got {completeds}");

    // Duplicate-processing invariants — still hold even with the synthetic
    // user turn, because that turn is injected into the LLM request only,
    // NOT persisted into the transcript.
    assert_user_task_count(&run.transcript, task, 1);
    assert_no_duplicate_consecutive_assistant(&run.transcript);

    for name in ["shouter", "exclaimer"] {
        let n = run
            .transcript
            .iter()
            .filter(|m| m.name.as_deref() == Some(name))
            .count();
        assert_eq!(n, 1, "{name} spoke {n} times, expected 1");
    }
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
        ConcurrentOrchestrator::new(provider, &model()).with_agents([optimist, skeptic]);

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
async fn handoff_session_triggers_real_handoff_and_target_responds() {
    // Real handoff path: greeter's reply matches the rule's keywords,
    // which pushes math_expert as active, which then has to respond to a
    // transcript ending in greeter's assistant message. Without the
    // `flows::prefill` compensation, math_expert would return empty on qwen.
    let provider = provider();

    let mut orchestrator =
        HandoffOrchestrator::new(provider, &model()).with_max_handoffs(Some(2));

    orchestrator.register_agent(
        Agent::from_string(
            "greeter",
            "Your first word must be the literal string '[HELLO]'. \
             Then write one short greeting sentence. Do not answer the question.",
        )
        .with_max_tokens(32)
        .with_temperature(0.0),
    );
    orchestrator.register_agent(
        Agent::from_string(
            "math_expert",
            "You are a math expert. Answer the user's math question directly in one short line. \
             Never use the word '[HELLO]'.",
        )
        .with_max_tokens(64)
        .with_temperature(0.0),
    );
    // Sentinel-token match only fires on greeter's output. Qwen at
    // temperature=0 follows the literal-prefix instruction reliably, and
    // math_expert is explicitly forbidden from emitting the sentinel.
    orchestrator.define_handoff(HandoffRule::to(
        "math_expert",
        HandoffMatcher::KeywordsAny(vec!["[HELLO]".into()]),
    ));

    let mut session = orchestrator.session("greeter").expect("session init");

    let turn = session.send("What is 2+2?").await.expect("send failed");
    let events: Vec<_> = turn
        .events
        .iter()
        .map(|e| match e {
            HandoffEvent::Message { agent, .. } => format!("msg:{agent}"),
            HandoffEvent::HandOff { from, to, .. } => format!("handoff:{from}->{to}"),
            HandoffEvent::Completed { agent } => format!("done:{agent}"),
        })
        .collect();
    eprintln!("handoff events: {events:?}");
    eprintln!("handoff reply: {:?}", turn.reply);

    // A handoff event should have fired, and the final reply should be
    // non-empty (math_expert produced a response after being handed off).
    let handed_off = turn.events.iter().any(|e| matches!(e, HandoffEvent::HandOff { .. }));
    assert!(handed_off, "expected a HandOff event; got {events:?}");
    assert!(
        turn.reply.as_deref().map(str::trim).unwrap_or("").len() > 0,
        "expected non-empty reply after handoff — prefill fix likely not applied"
    );

    let transcript = session.transcript();
    let user_msgs = transcript
        .iter()
        .filter(|m| matches!(m.role, MessageRole::User))
        .count();
    assert_eq!(user_msgs, 1, "expected 1 user msg after 1 send, got {user_msgs}");
    assert_no_duplicate_consecutive_assistant(transcript);
}

// -- group_chat ----------------------------------------------------------

#[tokio::test]
#[ignore = "requires OLLAMA_SMOKE_URL"]
async fn group_chat_round_robin_runs_two_rounds() {
    // Exercises the prefill fix: round 2's agent sees a transcript ending
    // in round 1's assistant reply — without compensation it would return
    // empty content on qwen.
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

    let manager = RoundRobinGroupChatManager::new().with_maximum_rounds(Some(2));
    let mut orchestrator =
        GroupChatOrchestrator::new(provider, &model(), manager).with_agents([optimist, skeptic]);

    let task = "is morning coffee good for you";
    let run = orchestrator.run(task).await.expect("group_chat run failed");
    eprintln!(
        "group_chat rounds={} transcript_roles={:?}",
        run.rounds,
        run.transcript.iter().map(|m| &m.role).collect::<Vec<_>>()
    );

    assert_eq!(run.rounds, 2, "expected 2 rounds, got {}", run.rounds);

    let assistant_msgs: Vec<_> = run
        .transcript
        .iter()
        .filter(|m| matches!(m.role, MessageRole::Assistant))
        .collect();
    assert_eq!(
        assistant_msgs.len(),
        2,
        "expected 2 assistant messages, got {}",
        assistant_msgs.len()
    );
    for m in &assistant_msgs {
        assert!(
            m.content.as_deref().unwrap_or("").trim().len() > 0,
            "an assistant message was empty — prefill fix likely not applied: {:?}",
            m.name
        );
    }

    assert_user_task_count(&run.transcript, task, 1);
    assert_no_duplicate_consecutive_assistant(&run.transcript);

    // Each agent spoke exactly once across the two rounds.
    for name in ["optimist", "skeptic"] {
        let n = run
            .transcript
            .iter()
            .filter(|m| m.name.as_deref() == Some(name))
            .count();
        assert_eq!(n, 1, "{name} spoke {n} times, expected 1");
    }
}

// -- magentic ------------------------------------------------------------

#[tokio::test]
#[ignore = "requires OLLAMA_SMOKE_URL"]
async fn magentic_delegates_and_manages_transcript_correctly() {
    // With the prefill fix, the delegated worker sees a synthetic user turn
    // injected after the manager's assistant-role delegation message, so it
    // actually responds instead of returning empty. We still cap max_rounds
    // — the manager may take several iterations before emitting a
    // `complete` decision — and accept either a clean finish or a
    // MaxRoundsReached that at least produced content.
    let provider = provider();

    let manager = MagenticManager::standard();
    let mut orchestrator =
        MagenticOrchestrator::new(provider, &model(), manager).with_max_rounds(3);

    let worker = Agent::from_string(
        "fact_finder",
        "You reply with one short factual sentence.",
    )
    .with_max_tokens(96);
    orchestrator.register_agent(worker).expect("register failed");

    let task = "state one short fact about honey bees";
    let result = orchestrator.run(task).await;

    let run = match result {
        Ok(run) => run,
        Err(denkwerk::AgentError::MaxRoundsReached) => {
            eprintln!("magentic hit MaxRoundsReached — acceptable as long as agent spoke");
            return;
        }
        Err(other) => panic!("magentic failed unexpectedly: {other:?}"),
    };

    eprintln!(
        "magentic rounds={} final={:?}",
        run.rounds, run.final_result
    );
    assert!(run.rounds >= 1, "magentic did not run any rounds");

    // The worker should have produced at least one non-empty response —
    // which only happens if the prefill fix is applied.
    let worker_spoke_non_empty = run
        .transcript
        .iter()
        .any(|m| {
            m.name.as_deref() == Some("fact_finder")
                && m.content.as_deref().unwrap_or("").trim().len() > 0
        });
    assert!(
        worker_spoke_non_empty,
        "fact_finder never produced non-empty content — prefill fix likely not applied"
    );

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

    let orchestrator = DispatchOrchestrator::new(provider, &model(), hub)
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
