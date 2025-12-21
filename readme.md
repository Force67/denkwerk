# denkwerk

`denkwerk` is a LLM orchestration library for rust. It adapts some ideas from microsofts semantic kernel and autogen.

## Why?
I made this, because i wanted to have a rust native library that is simple and easy to use. Therefore, here we are.

## Features at a glance

- **Function-calling** – register async Rust functions that models can call and let the crate handle tool schemas, argument parsing, and invocation.
- **Agent orchestrations** – drop-in managers for sequential pipelines, concurrent fan-out, handoffs, magentic-style planners, and moderated group chats.
- **Chat history utilities** – append, summarize, and compress transcripts with either quick heuristics or an optional LLM-powered condenser.

Most modules work on their own. You only opt into the layers you need.

## Getting started

Add the dependency:

```toml
[dependencies]
denkwerk = { git = "https://github.com/Force67/denkwerk" }
```

Pick a provider (OpenRouter is bundled) and issue a completion:

```rust
use denkwerk::providers::openrouter::OpenRouter;
use denkwerk::{ChatMessage, CompletionRequest, LLMProvider};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let provider = OpenRouter::from_env()?;
    let request = CompletionRequest::new(
        "openai/gpt-4o-mini",
        vec![ChatMessage::user("Give me a weekend project idea.")],
    );

    let response = provider.complete(request).await?;
    println!("{}", response.message.text().unwrap_or_default());
    Ok(())
}
```

## Orchestrations in practice

Each orchestration is just a thin layer on the provider trait. They add routing logic and event hooks but stay out of the model’s way.

### How flows work (technical)

- **Common shape**: An orchestrator holds an `Arc<dyn LLMProvider>`, a model name, and one or more `Agent`s. Runs build a transcript (`Vec<ChatMessage>`) and repeatedly call `provider.complete(...)` through the active agent.
- **Agents + tools**: `Agent::execute_with_tools` prepends the agent’s system instructions, attaches tool definitions (if any), and returns both assistant text and tool calls.
- **Structured outcomes**: Some flows interpret the assistant output as an action (respond / route / finish) and emit events as the transcript evolves.

Flow types (current implementations):

- **Sequential** (`SequentialOrchestrator`): runs agents in order; each agent sees the growing transcript and produces the next message.
- **Concurrent** (`ConcurrentOrchestrator`): fans out to multiple agents in parallel and aggregates the results.
- **Handoff** (`HandoffOrchestrator`): keeps a single active agent and switches to another agent when it detects a handoff action (via tool call, JSON-in-text, natural-language cue, or deterministic rule match).
- **Group chat** (`GroupChatOrchestrator`): multi-agent conversation with a moderator/selector deciding who speaks next.
- **Magentic** (`MagenticOrchestrator`): planner/executor style orchestration (plan, then run steps) using agents and tools.

### Sequential pipeline

```rust
use std::sync::Arc;
use denkwerk::{Agent, SequentialOrchestrator, LLMProvider};
use denkwerk::providers::openrouter::OpenRouter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let provider: Arc<dyn LLMProvider> = Arc::new(OpenRouter::from_env()?);

    let analyst = Agent::from_string("Analyst", "Extract key facts.");
    let writer = Agent::from_string("Writer", "Turn facts into a newsletter blurb.");
    let editor = Agent::from_string("Editor", "Polish the draft.");

    let orchestrator = SequentialOrchestrator::new(provider, "openai/gpt-4o-mini")
        .with_agents(vec![analyst, writer, editor]);

    let run = orchestrator.run("Summarize trends in home coffee gear.").await?;
    println!("{}", run.final_output.unwrap_or_default());
    Ok(())
}
```

### Handoff: forcing tool-based handoffs

By default, the handoff flow accepts handoff directives from either tool calling or parsed assistant text. If you want to *only* accept model-initiated handoffs via the internal `handoff` function call (deterministic rules can still route):

```rust
let orchestrator = HandoffOrchestrator::new(provider, "openai/gpt-4o-mini")
    .with_force_handoff_tool(true);
```

## License

MIT or Apache 2.0, whichever suits your project.

---

If you reach the limits of the provided orchestrations, keep extending—everything here is meant to stay modular so your own patterns can live alongside the defaults.
