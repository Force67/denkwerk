//! Live smoke tests against a running Ollama instance.
//!
//! All tests are `#[ignore]` by default. Run them with:
//!
//! ```sh
//! OLLAMA_SMOKE_URL=http://localhost:11434 \
//!   cargo test --test ollama_smoke -- --ignored --nocapture
//! ```
//!
//! They verify behaviors that unit tests can't: that `think` actually toggles,
//! that `preserve_thinking` is accepted, that streaming reaches Completed, and
//! that vision models produce text from base64 images.

use std::time::Duration;

use denkwerk::{
    types::StreamEvent, ChatMessage, CompletionRequest, LLMProvider, Ollama, OllamaConfig,
    ThinkMode,
};
use futures_util::StreamExt;

const TEXT_MODEL: &str = "qwen3.6:35b-a3b";
const VISION_MODEL: &str = "qwen3-vl:8b-instruct";

fn smoke_url() -> Option<String> {
    std::env::var("OLLAMA_SMOKE_URL").ok()
}

fn provider(think: ThinkMode, preserve_thinking: bool) -> Ollama {
    let url = smoke_url().expect("OLLAMA_SMOKE_URL not set");
    let cfg = OllamaConfig::new()
        .with_base_url(url)
        .with_think_mode(think)
        .with_preserve_thinking(preserve_thinking);
    // Give the model long enough to load/warm up on a cold cache.
    let cfg = OllamaConfig {
        request_timeout: Duration::from_secs(300),
        ..cfg
    };
    Ollama::from_config(cfg).expect("failed to construct Ollama")
}

#[tokio::test]
#[ignore = "requires OLLAMA_SMOKE_URL"]
async fn text_and_reasoning_via_complete() {
    let ollama = provider(ThinkMode::On, false);
    let req = CompletionRequest::new(
        TEXT_MODEL,
        vec![ChatMessage::user("What is 2+2? Reply with just the number.")],
    )
    .with_max_tokens(256);

    let res = ollama.complete(req).await.expect("complete failed");
    eprintln!("content: {:?}", res.message.content);
    eprintln!(
        "reasoning present: {}",
        res.reasoning.as_ref().map(|r| r.len()).unwrap_or(0)
    );

    let content = res.message.content.expect("content missing");
    assert!(content.contains('4'), "expected '4' in reply, got {content:?}");
    let reasoning = res.reasoning.expect("reasoning trace missing");
    assert!(!reasoning.is_empty());
    assert!(!reasoning[0].content.is_empty(), "reasoning[0] was empty");

    let usage = res.usage.expect("usage missing");
    assert!(usage.prompt_tokens > 0 && usage.completion_tokens > 0);
}

#[tokio::test]
#[ignore = "requires OLLAMA_SMOKE_URL"]
async fn think_off_suppresses_reasoning() {
    let ollama = provider(ThinkMode::Off, false);
    let req = CompletionRequest::new(
        TEXT_MODEL,
        vec![ChatMessage::user("Say the word 'hi' exactly once.")],
    )
    .with_max_tokens(64);

    let res = ollama.complete(req).await.expect("complete failed");
    let content = res.message.content.unwrap_or_default();
    eprintln!("content with think=false: {content:?}");
    assert!(!content.is_empty(), "expected non-empty content");
    assert!(
        res.reasoning.is_none(),
        "expected no reasoning when think=false, got {:?}",
        res.reasoning
    );
    // Content must not contain the thinking markers the OpenAI-compat shim
    // would leak.
    assert!(!content.contains("<think>"), "raw <think> tags leaked into content");
}

#[tokio::test]
#[ignore = "requires OLLAMA_SMOKE_URL"]
async fn preserve_thinking_roundtrips() {
    let ollama = provider(ThinkMode::On, true);

    // Turn 1 — give plenty of headroom so thinking doesn't starve the
    // content budget on this 35B model.
    let req1 = CompletionRequest::new(
        TEXT_MODEL,
        vec![ChatMessage::user(
            "Pick a random integer between 10 and 20 and remember it. \
             Reply with only that number.",
        )],
    )
    .with_max_tokens(4096);

    let res1 = ollama.complete(req1).await.expect("turn1 failed");
    let first_reply = res1.message.content.clone().expect("turn1 content missing");
    let first_thinking = res1.message.thinking.clone();
    assert!(first_thinking.is_some(), "turn1 should carry thinking");
    eprintln!("turn1 content: {first_reply}");
    eprintln!(
        "turn1 thinking: {} chars",
        first_thinking.as_deref().unwrap_or("").len()
    );

    // Turn 2: feed the assistant message back (carrying its thinking) and ask
    // a follow-up. With preserve_thinking=true the serializer should send the
    // prior `thinking` field. Any server-side rejection would surface here.
    let mut assistant_msg = ChatMessage::assistant(first_reply.clone());
    assistant_msg.thinking = first_thinking.clone();

    let req2 = CompletionRequest::new(
        TEXT_MODEL,
        vec![
            ChatMessage::user(
                "Pick a random integer between 10 and 20 and remember it. \
                 Reply with only that number.",
            ),
            assistant_msg,
            ChatMessage::user("What was that number? Reply with only the number."),
        ],
    )
    .with_max_tokens(4096);

    let res2 = ollama.complete(req2).await.expect("turn2 failed");
    let reply2 = res2.message.content.expect("turn2 content missing");
    eprintln!("turn2 content: {reply2}");
    // We don't assert the exact number (models are noisy), only that the
    // server accepted the request and produced output.
    assert!(!reply2.trim().is_empty(), "turn2 reply was empty");
}

#[tokio::test]
#[ignore = "requires OLLAMA_SMOKE_URL"]
async fn streaming_reaches_completed() {
    // Streaming + think=off is the deterministic path: content must arrive via
    // MessageDelta. Reasoning streaming is covered by the other tests.
    let ollama = provider(ThinkMode::Off, false);
    let req = CompletionRequest::new(
        TEXT_MODEL,
        vec![ChatMessage::user("Count from 1 to 3, comma-separated.")],
    )
    .with_max_tokens(128);

    let mut stream = ollama
        .stream_completion(req)
        .await
        .expect("stream_completion failed");

    let mut message_deltas = 0usize;
    let mut got_completed = false;
    let mut got_reasoning = false;
    let mut final_content = String::new();

    while let Some(event) = stream.next().await {
        let event = event.expect("stream error");
        match event {
            StreamEvent::MessageDelta(_) => message_deltas += 1,
            StreamEvent::ReasoningDelta(_) => got_reasoning = true,
            StreamEvent::Completed(resp) => {
                got_completed = true;
                final_content = resp.message.content.unwrap_or_default();
            }
            StreamEvent::ToolCallDelta { .. } => {}
        }
    }

    eprintln!(
        "deltas={message_deltas} reasoning={got_reasoning} completed={got_completed} final={final_content:?}"
    );
    assert!(got_completed, "stream never reached Completed");
    assert!(!got_reasoning, "think=off must not emit reasoning deltas");
    assert!(message_deltas > 0, "stream produced no MessageDelta events");
    assert!(
        final_content.contains('1') && final_content.contains('3'),
        "final text missing expected content: {final_content:?}"
    );
}

#[tokio::test]
#[ignore = "requires OLLAMA_SMOKE_URL"]
async fn vision_qwen3_vl_describes_image() {
    let ollama = provider(ThinkMode::Off, false);

    // A valid 32×32 pure-red PNG, base64-encoded. Larger than the single-pixel
    // minimum so the model runner has enough patches to work with.
    const TINY_RED_PNG_B64: &str =
        "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAKElEQVR4nO3NsQ0AAAzCMP5/un0CNkuZ41wybXsHAAAAAAAAAAAAxR4yw/wuPL6QkAAAAABJRU5ErkJggg==";

    let req = CompletionRequest::new(
        VISION_MODEL,
        vec![ChatMessage::user_with_images(
            "What color is this tiny image? Reply in one short sentence.",
            vec![format!("data:image/png;base64,{TINY_RED_PNG_B64}")],
        )],
    )
    .with_max_tokens(128);

    let res = ollama.complete(req).await.expect("vision complete failed");
    let content = res.message.content.expect("vision content missing");
    eprintln!("vision content: {content}");
    assert!(!content.trim().is_empty(), "vision reply was empty");
    // The image is pure red — the model should mention it.
    assert!(
        content.to_lowercase().contains("red"),
        "expected 'red' in vision reply, got: {content}"
    );
    let usage = res.usage.expect("usage missing");
    // Vision inputs add tokens on top of text.
    assert!(usage.prompt_tokens > 0);
}
