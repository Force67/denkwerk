use std::{collections::HashMap, path::PathBuf, sync::Arc};

use denkwerk::{
    ExecutionStep,
    FlowBuilder,
    FlowContext,
    FlowNodeKind,
    LLMProvider,
    SequentialOrchestrator,
    SequentialEvent,
};
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
    for (idx, step) in plan.iter().enumerate() {
        match step {
            ExecutionStep::Agent(agent) => println!("  {idx}: agent -> {}", agent.name()),
            ExecutionStep::Parallel { branches, .. } => {
                println!("  {idx}: parallel -> {} branches", branches.len());
            }
        }
    }

    let mut pipeline = Vec::new();
    for step in &plan {
        match step {
            ExecutionStep::Agent(agent) => pipeline.push(agent.clone()),
            ExecutionStep::Parallel { branches, .. } => {
                println!("Note: parallel branches will run sequentially in this demo.");
                for branch in branches {
                    for agent in branch {
                        pipeline.push(agent.clone());
                    }
                }
            }
        }
    }

    let model = builder
        .document()
        .agents
        .iter()
        .find(|a| pipeline.first().map(|p| p.name()) == Some(a.id.as_str()))
        .map(|a| a.model.clone())
        .unwrap_or_else(|| deployment.clone());

    let provider: Arc<dyn LLMProvider> = Arc::new(AzureOpenAI::from_env()?);
    let orchestrator = SequentialOrchestrator::new(provider, model)
        .with_agents(pipeline)
        .with_event_callback(|event| match event {
            SequentialEvent::Step { agent, output } => {
                println!("\n[{agent}] produced:\n{output}\n");
            }
            SequentialEvent::Completed { agent, output } => {
                if let Some(text) = output {
                    println!("[{agent}] completed with:\n{text}\n");
                }
            }
        });

    let run = orchestrator.run(task).await?;
    println!("--- Final output ---\n{}", run.final_output.unwrap_or_default());

    Ok(())
}
