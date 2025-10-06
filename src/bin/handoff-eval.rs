use std::{fs, path::PathBuf};

use clap::Parser;
use denkwerk::{
    eval::{scenario::EvalScenario, runner::EvalRunner},
    flows::handoffflow::HandoffOrchestrator,
    providers::openrouter::OpenRouter,
    LLMProvider,
};

#[derive(Parser)]
#[command(name = "handoff-eval")]
#[command(about = "Run handoff evaluation scenarios")]
struct Args {
    /// Path to scenarios directory or file
    #[arg(short, long)]
    scenarios: PathBuf,

    /// Use OpenRouter provider instead of scripted
    #[arg(long)]
    use_openrouter: bool,

    /// OpenRouter API token (can also set OPENROUTER_API_KEY env var)
    #[arg(long)]
    token: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Load scenarios
    let scenarios: Vec<EvalScenario> = if args.scenarios.is_dir() {
        let mut scenarios = Vec::new();
        for entry in fs::read_dir(&args.scenarios)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                let content = fs::read_to_string(&path)?;
                let scenario: EvalScenario = serde_json::from_str(&content)?;
                scenarios.push(scenario);
            }
        }
        scenarios
    } else {
        let content = fs::read_to_string(&args.scenarios)?;
        vec![serde_json::from_str(&content)?]
    };

    let runner = EvalRunner::new();

    let report = if args.use_openrouter {
        let api_key = args.token
            .or_else(|| std::env::var("OPENROUTER_API_KEY").ok())
            .expect("OpenRouter API key not provided. Use --token or set OPENROUTER_API_KEY env var");
        let provider = std::sync::Arc::new(OpenRouter::new(api_key).unwrap());
        let make_orchestrator = |p: std::sync::Arc<dyn LLMProvider>, m: String| {
            HandoffOrchestrator::new(p, m)
        };
        runner.run_with_provider(make_orchestrator, provider, "anthropic/claude-3-haiku".to_string(), &scenarios).await
    } else {
        // For evaluation, we create orchestrator with the scripted provider
        let make_orchestrator = |provider: std::sync::Arc<dyn crate::LLMProvider>, model: String| {
            HandoffOrchestrator::new(provider, model)
        };
        runner.run(make_orchestrator, &scenarios).await
    };

    println!("Total: {}, Passed: {}", report.total, report.passed);

    for case in &report.cases {
        if !case.pass {
            println!("Failed: {}", case.name);
            for failure in &case.failures {
                println!("  - {}", failure);
            }
        }
    }

    if report.passed == report.total {
        std::process::exit(0);
    } else {
        std::process::exit(1);
    }
}