use std::{collections::HashMap, path::PathBuf, sync::Arc};

use denkwerk::{
    ExecutionStep,
    FlowBuilder,
    FlowContext,
    FlowNodeKind,
    LLMProvider,
    SequentialEvent,
};
use denkwerk::flows::spec::FlowRunError;
use denkwerk::providers::azure_openai::AzureOpenAI;
use denkwerk::kernel_function;
use denkwerk::functions::KernelFunction;
use dotenvy::from_path;

#[kernel_function(name = "math_add", description = "Add two numbers")]
fn math_add(a: f64, b: f64) -> Result<f64, String> {
    Ok(a + b)
}

fn example_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("prompt_from_yaml_demo")
}

fn load_env() {
    let path = example_dir().join(".env");
    if let Err(err) = from_path(&path) {
        eprintln!(
            "No .env loaded from {:?}: {} (set AZURE_OPENAI_* in your environment)",
            path,
            err
        );
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    load_env();

    let deployment =
        std::env::var("AZURE_OPENAI_DEPLOYMENT").unwrap_or_else(|_| "gpt-4o-mini".to_string());

    let yaml_path = example_dir().join("flow.yaml");
    let yaml = std::fs::read_to_string(&yaml_path)?;
    let yaml = yaml.replace("${AZURE_OPENAI_DEPLOYMENT}", &deployment);

    let task = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Summarize why Rust is great for backend services.".to_string());
    let mode = std::env::args().nth(2).or_else(|| std::env::var("FLOW_MODE").ok());

    let ctx = if let Some(mode) = mode.clone() {
        FlowContext::default().with_var("mode", mode)
    } else {
        FlowContext::default()
    };

    let builder = FlowBuilder::from_yaml_str(example_dir(), &yaml)?;

    let mut functions: HashMap<String, Arc<dyn KernelFunction>> = HashMap::new();
    functions.insert("math_add".to_string(), math_add_kernel());
    let tool_registries = builder.build_tool_registries(&functions)?;

    let plan = builder.build_execution_plan("main", &ctx, &tool_registries)?;

    println!("Flow file: {}", yaml_path.display());
    println!("Deployment: {deployment}");
    if let Some(flow) = builder.document().flows.iter().find(|f| f.id == "main") {
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for node in &flow.nodes {
            let label = match node.kind {
                FlowNodeKind::Input {} => "input",
                FlowNodeKind::Output {} => "output",
                FlowNodeKind::Agent { .. } => "agent",
                FlowNodeKind::Decision { .. } => "decision",
                FlowNodeKind::Tool { .. } => "tool",
                FlowNodeKind::Merge {} => "merge",
                FlowNodeKind::Parallel { .. } => "parallel",
                FlowNodeKind::Loop { .. } => "loop",
                FlowNodeKind::Subflow { .. } => "subflow",
            };
            *counts.entry(label).or_default() += 1;
        }
        println!("Node summary:");
        for (label, count) in counts {
            println!("  {label}: {count}");
        }
    }
    if let Some(mode) = &mode {
        println!("Decision context: mode={mode}");
    }
    println!("Planned steps:");
    let task_with_tools = task.clone();
    for (idx, step) in plan.iter().enumerate() {
        match step {
            ExecutionStep::Agent(agent) => println!("  {idx}: agent -> {}", agent.name()),
            ExecutionStep::Tool { tool, arguments } => {
                if let Some(args) = arguments {
                    println!("  {idx}: tool -> {tool} (args from YAML)");
                    println!("        args: {args}");
                } else {
                    println!("  {idx}: tool -> {tool}");
                }
            }
            ExecutionStep::Parallel { branches, .. } => {
                println!("  {idx}: parallel -> {} branches", branches.len());
            }
        }
    }

    let provider: Arc<dyn LLMProvider> = Arc::new(AzureOpenAI::from_env()?);
    let event_logger = |event: &SequentialEvent| match event {
        SequentialEvent::Step { agent, output } => {
            println!("\n[{agent}] produced:\n{output}\n");
        }
        SequentialEvent::Completed { agent, output } => {
            if let Some(text) = output {
                println!("[{agent}] completed with:\n{text}\n");
            }
        }
    };

    let (mut run, tool_runs) = match builder
        .run_sequential_flow(
            "main",
            &ctx,
            &tool_registries,
            provider,
            task_with_tools,
            Some(event_logger),
        )
        .await
    {
        Ok(result) => result,
        Err(FlowRunError::Tool(err)) => {
            eprintln!("Tool execution failed: {err}");
            std::process::exit(1);
        }
        Err(FlowRunError::NoAgents(flow)) => {
            eprintln!("Flow '{flow}' contains no agent steps.");
            std::process::exit(1);
        }
        Err(other) => return Err(Box::new(other) as Box<dyn std::error::Error>),
    };

    if let Some(uuid) = tool_runs.iter().find_map(|run| {
        run.value
            .get("body")
            .and_then(|b| b.get("uuid"))
            .and_then(|u| u.as_str())
    }) {
        if let Some(text) = run.final_output.as_mut() {
            text.push_str(&format!("\n\nUUID (from http_uuid): {uuid}"));
        }
    }

    println!("--- Final output ---\n{}", run.final_output.clone().unwrap_or_default());

    Ok(())
}
