use std::{
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

use clap::{Parser, ValueEnum};
use denkwerk::{
    bench::{load_cases, run_case, BenchCase},
    providers::{azure_openai::AzureOpenAI, openai::OpenAI, openrouter::OpenRouter},
    LLMProvider,
};

#[derive(Debug, Clone, ValueEnum)]
enum ProviderKind {
    Openai,
    Openrouter,
    AzureOpenai,
}

#[derive(Parser)]
#[command(name = "bench-tool-adherence")]
#[command(about = "Run tool-calling adherence benchmark cases")]
struct Args {
    /// Path to cases directory or case file (YAML/JSON)
    #[arg(long, default_value = "bench/cases")]
    cases: PathBuf,

    /// Provider to use
    #[arg(long, value_enum, default_value = "openrouter")]
    provider: ProviderKind,

    /// Model identifier (provider-specific)
    #[arg(long)]
    model: String,

    /// Output path for JSONL results
    #[arg(long)]
    out: Option<PathBuf>,

    /// Maximum completion->tool loop rounds per case
    #[arg(long, default_value_t = 8)]
    max_rounds: usize,

    /// Run only cases whose id contains this substring (repeatable)
    #[arg(long)]
    filter: Vec<String>,

    /// Stop at first failure/provider error
    #[arg(long)]
    fail_fast: bool,
}

fn default_system_prompt() -> &'static str {
    "You are a careful tool-calling agent.\n\
Follow tool JSON schemas exactly.\n\
Call tools only when required by the user instructions.\n\
After receiving tool results, produce the final answer."
}

fn ensure_parent_dir(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

fn default_out_path() -> PathBuf {
    let ts = chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
    PathBuf::from(format!("bench/runs/{ts}.jsonl"))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let cases = load_cases(&args.cases)?;
    let cases = filter_cases(cases, &args.filter);
    if cases.is_empty() {
        eprintln!("No cases matched.");
        std::process::exit(2);
    }

    let provider: Arc<dyn LLMProvider> = match args.provider {
        ProviderKind::Openai => Arc::new(OpenAI::from_env()?),
        ProviderKind::Openrouter => Arc::new(OpenRouter::from_env()?),
        ProviderKind::AzureOpenai => Arc::new(AzureOpenAI::from_env()?),
    };

    let out_path = args.out.unwrap_or_else(default_out_path);
    ensure_parent_dir(&out_path)?;
    let file = fs::File::create(&out_path)?;
    let mut writer = BufWriter::new(file);

    let mut total = 0usize;
    let mut passed = 0usize;
    let mut sum_total = 0.0f64;
    let mut sum_validity = 0.0f64;
    let mut sum_selection = 0.0f64;
    let mut sum_sequence = 0.0f64;
    let mut sum_efficiency = 0.0f64;
    let mut sum_final = 0.0f64;

    for case in cases {
        total += 1;

        let result = run_case(
            provider.as_ref(),
            &args.model,
            default_system_prompt(),
            &case,
            args.max_rounds,
        )
        .await;

        match result {
            Ok(case_result) => {
                if case_result.pass {
                    passed += 1;
                } else {
                    eprintln!("FAIL {} (score {:.3})", case_result.id, case_result.scores.total);
                    for f in &case_result.failures {
                        eprintln!("  - {f}");
                    }
                    if args.fail_fast {
                        serde_json::to_writer(&mut writer, &case_result)?;
                        writer.write_all(b"\n")?;
                        writer.flush()?;
                        break;
                    }
                }

                sum_total += case_result.scores.total;
                sum_validity += case_result.scores.validity;
                sum_selection += case_result.scores.selection;
                sum_sequence += case_result.scores.sequence;
                sum_efficiency += case_result.scores.efficiency;
                sum_final += case_result.scores.final_answer;

                serde_json::to_writer(&mut writer, &case_result)?;
                writer.write_all(b"\n")?;
            }
            Err(err) => {
                eprintln!("ERROR {}: {err}", case.id);
                if args.fail_fast {
                    break;
                }
            }
        }
    }

    writer.flush()?;

    let denom = (total.max(1)) as f64;
    println!(
        "Provider: {}, Model: {}, Results: {passed}/{total} passed, AvgTotal: {:.3}, Output: {}",
        provider.name(),
        args.model,
        sum_total / denom,
        out_path.display()
    );
    println!(
        "Avg: validity {:.3}, selection {:.3}, sequence {:.3}, efficiency {:.3}, final {:.3}",
        sum_validity / denom,
        sum_selection / denom,
        sum_sequence / denom,
        sum_efficiency / denom,
        sum_final / denom
    );

    if passed == total {
        Ok(())
    } else {
        std::process::exit(1);
    }
}

fn filter_cases(mut cases: Vec<BenchCase>, filters: &[String]) -> Vec<BenchCase> {
    if filters.is_empty() {
        return cases;
    }
    cases.retain(|c| filters.iter().any(|f| c.id.contains(f)));
    cases
}
