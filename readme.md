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


## License

MIT or Apache 2.0, whichever suits your project.

---

If you reach the limits of the provided orchestrations, keep extending—everything here is meant to stay modular so your own patterns can live alongside the defaults.
