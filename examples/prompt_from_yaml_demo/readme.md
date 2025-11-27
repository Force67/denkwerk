# Prompt-from-YAML (Azure) Demo

This example loads a flow defined in `flow.yaml`, pulls prompts from the `prompts/` folder, and runs it using the Azure OpenAI connector. The flow exercises most node types: input, decision, parallel fan-out + merge, subflow, loop, a tool-enabled agent, and output. Parallel branches are executed sequentially in this demo for simplicity.

## Setup
1. Copy `.env.example` to `.env` in this folder and set your Azure resource values:
   - `AZURE_OPENAI_KEY`
   - `AZURE_OPENAI_ENDPOINT` (e.g., `https://your-resource.openai.azure.com`)
   - `AZURE_OPENAI_DEPLOYMENT` (deployment name to use as the model)
   - `AZURE_OPENAI_API_VERSION` (optional, defaults to 2024-08-01-preview)

## Run
From the repo root:

```bash
cargo run --example prompt_from_yaml_demo -- "Summarize why Rust excels at backend services."

# take the short decision branch (skips the facts subflow)
cargo run --example prompt_from_yaml_demo -- "Summarize why Rust excels at backend services." short
# or
FLOW_MODE=short cargo run --example prompt_from_yaml_demo -- "Summarize why Rust excels at backend services."
```

The example automatically loads `.env` from this folder before creating the provider and orchestration pipeline.
