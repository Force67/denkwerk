//! Chat-template prefill compensation for multi-agent orchestrators.
//!
//! # Why this exists
//!
//! Some model families — notably the Qwen3 family served by Ollama's
//! `qwen3.5` renderer — treat a trailing `assistant` message in the messages
//! array as a **prefill cue**: a partial assistant turn the model should
//! *continue* rather than a completed turn it should *respond after*.
//! Concretely, when the request ends with an assistant turn, the renderer
//! does not append `<|im_end|>\n<|im_start|>assistant\n` (the generation
//! prompt). The prompt is left mid-turn, and the model's first generated
//! token is typically `<|im_end|>` (close the turn), producing empty content
//! and `eval_count: 1`.
//!
//! This is a legitimate feature for prefill workflows (e.g. "the answer is "
//! → model fills in), but it breaks multi-agent orchestrators that chain
//! agents by appending each agent's reply as an assistant message to a
//! shared transcript. Agent B, called with `[user, assistant_A(...)]`, gets
//! interpreted as "continue A's completed response" and returns empty.
//!
//! Not every model does this. In live tests:
//! - `qwen3.6:35b-a3b` → prefills (empty response).
//! - `gemma4:e4b` → generates normally.
//! - `gpt-oss:120b` → generates normally.
//!
//! The gating here is conservative: we only inject a synthetic user turn
//! for model names we've confirmed prefill. Other models are untouched.
//!
//! # Fix
//!
//! Before calling the next agent, if the transcript currently ends with an
//! assistant message and the model is known to prefill, append a minimal
//! user-role turn (`"Please continue."`) to the history we send to the LLM
//! — **without** persisting it to the orchestrator's shared transcript.
//! The model then sees a proper alternation and opens a fresh assistant
//! turn as expected.
//!
//! # Non-goals
//!
//! - We do not try to detect prefill behavior dynamically (e.g. by looking
//!   at `eval_count == 1` on a prior call) — a one-shot retry would double
//!   latency on legitimate short answers.
//! - We do not strip the nudge from the response — it only exists in the
//!   per-call `Vec<ChatMessage>` passed to the provider, not the transcript.

use std::borrow::Cow;

use crate::types::{ChatMessage, MessageRole};

/// Default synthetic user turn text. Kept deliberately bland — anything
/// more directive risks biasing the downstream agent's response.
const DEFAULT_NUDGE: &str = "Please continue.";

/// Returns `true` when the given model is known to use a chat template /
/// renderer that treats a trailing `assistant` turn as a prefill cue (and
/// will therefore return empty content unless the caller inserts a user
/// turn between consecutive assistant messages).
///
/// Match is case-insensitive and scans for substrings so it works across
/// Ollama tags (`qwen3.6:35b-a3b`), OpenRouter slugs (`qwen/qwen3-...`),
/// and vendor-prefixed names.
///
/// Currently matches the **Qwen2 / Qwen2.5 / Qwen3 / Qwen3.6** families —
/// all of which use Ollama's `qwen35` renderer or an equivalent upstream
/// chat template with the same prefill behavior. Extend this list as new
/// families are confirmed (via a live probe) to prefill on trailing
/// assistant turns.
pub fn needs_user_alternation(model: &str) -> bool {
    let m = model.to_ascii_lowercase();
    // Ordered by specificity just to aid readability; any substring match
    // is sufficient.
    m.contains("qwen3.6")
        || m.contains("qwen3.5")
        || m.contains("qwen3")
        || m.contains("qwen2.5")
        || m.contains("qwen2")
}

/// Produce the message history to send to the LLM for the next agent turn.
///
/// If the `transcript`'s last message is an assistant turn *and* `model`
/// needs alternation, returns an owned copy with a minimal user turn
/// appended. Otherwise borrows the transcript as-is (zero-cost).
///
/// The synthetic turn is **not** persisted to the caller's transcript; it
/// only appears in the history passed to the provider. That keeps the
/// orchestrator's durable conversation state clean for users who inspect
/// it (telemetry, UI replay, history export).
pub fn history_for_llm<'a>(transcript: &'a [ChatMessage], model: &str) -> Cow<'a, [ChatMessage]> {
    let last_is_assistant = matches!(
        transcript.last().map(|m| &m.role),
        Some(MessageRole::Assistant)
    );
    if !last_is_assistant || !needs_user_alternation(model) {
        return Cow::Borrowed(transcript);
    }

    let mut patched = transcript.to_vec();
    patched.push(ChatMessage::user(DEFAULT_NUDGE.to_string()));
    Cow::Owned(patched)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gates_by_model_name_case_insensitive() {
        assert!(needs_user_alternation("qwen3.6:35b-a3b"));
        assert!(needs_user_alternation("Qwen3-235B-A22B-Instruct"));
        assert!(needs_user_alternation("qwen/qwen2.5-coder-32b"));
        assert!(!needs_user_alternation("gemma4:e4b"));
        assert!(!needs_user_alternation("gpt-oss:120b"));
        assert!(!needs_user_alternation("gpt-4o"));
    }

    #[test]
    fn no_alternation_needed_when_last_is_user() {
        let transcript = vec![ChatMessage::user("hi")];
        let out = history_for_llm(&transcript, "qwen3.6:35b-a3b");
        assert!(matches!(out, Cow::Borrowed(_)));
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn no_alternation_needed_for_non_qwen() {
        let transcript = vec![
            ChatMessage::user("hi"),
            ChatMessage::assistant("hello"),
        ];
        let out = history_for_llm(&transcript, "gemma4:e4b");
        assert!(matches!(out, Cow::Borrowed(_)));
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn injects_nudge_for_qwen_with_trailing_assistant() {
        let transcript = vec![
            ChatMessage::user("hi"),
            ChatMessage::assistant("hello"),
        ];
        let out = history_for_llm(&transcript, "qwen3.6:35b-a3b");
        assert!(matches!(out, Cow::Owned(_)));
        assert_eq!(out.len(), 3);
        assert!(matches!(out[2].role, MessageRole::User));
        assert_eq!(out[2].content.as_deref(), Some(DEFAULT_NUDGE));
    }

    #[test]
    fn empty_transcript_does_not_panic() {
        let transcript: Vec<ChatMessage> = Vec::new();
        let out = history_for_llm(&transcript, "qwen3.6:35b-a3b");
        assert_eq!(out.len(), 0);
    }
}
