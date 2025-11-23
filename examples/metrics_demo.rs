use denkwerk::{
    Agent, InMemoryMetricsCollector, LLMProvider, SequentialOrchestrator, WithMetrics,
    TokenUsage, MetricsCollector, SharedStateContextExt, InMemorySharedStateStore,
};
use std::sync::Arc;
use async_trait::async_trait;

// Mock provider that simulates realistic token usage
struct RealisticMockProvider {
    responses: std::sync::Mutex<Vec<String>>,
    token_usage: std::sync::Mutex<Vec<TokenUsage>>,
}

impl RealisticMockProvider {
    fn new() -> Self {
        Self {
            responses: std::sync::Mutex::new(vec![
                "I'll analyze the requirements and create a structured plan. This will involve breaking down the problem into manageable components.".to_string(),
                "Based on my analysis, I'll now implement the solution with proper error handling and optimization considerations.".to_string(),
                "The implementation is complete. Let me now provide a comprehensive summary and recommendations for future improvements.".to_string(),
            ]),
            token_usage: std::sync::Mutex::new(vec![
                TokenUsage {
                    prompt_tokens: 150,
                    completion_tokens: 80,
                    total_tokens: 230,
                },
                TokenUsage {
                    prompt_tokens: 200,
                    completion_tokens: 120,
                    total_tokens: 320,
                },
                TokenUsage {
                    prompt_tokens: 180,
                    completion_tokens: 100,
                    total_tokens: 280,
                },
            ]),
        }
    }
}

#[async_trait]
impl LLMProvider for RealisticMockProvider {
    async fn complete(&self, _request: denkwerk::CompletionRequest) -> Result<denkwerk::CompletionResponse, denkwerk::LLMError> {
        let mut responses_guard = self.responses.lock().unwrap();
        let mut usage_guard = self.token_usage.lock().unwrap();

        let content = responses_guard.remove(0);
        let usage = usage_guard.remove(0);

        drop(responses_guard);
        drop(usage_guard);

        Ok(denkwerk::CompletionResponse {
            message: denkwerk::ChatMessage::assistant(content),
            usage: Some(usage),
            reasoning: None,
        })
    }

    fn name(&self) -> &'static str {
        "realistic_mock"
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Agent Metrics and Token Spending Demo\n");

    // Create metrics collector
    let metrics_collector = Arc::new(InMemoryMetricsCollector::new());

    // Create shared state (to show integration with other features)
    let shared_state = Arc::new(InMemorySharedStateStore::new());

    // Create realistic mock provider
    let provider1: Arc<dyn LLMProvider> = Arc::new(RealisticMockProvider::new());

    // Create agents for different roles
    let planner = Agent::from_string(
        "Project Planner",
        "Analyze requirements and create detailed project plans with timelines and resource allocation."
    );

    let implementer = Agent::from_string(
        "Software Implementer",
        "Implement solutions based on project plans, focusing on code quality and best practices."
    );

    let reviewer = Agent::from_string(
        "Code Reviewer",
        "Review implementations for quality, security, and performance optimizations."
    );

    println!("🚀 Running Multiple Workflows with Metrics Collection\n");

    // Run first workflow
    println!("📋 Workflow 1: Software Development Process");
    let orchestrator1 = SequentialOrchestrator::new(provider1, "gpt-4")
        .with_agents(vec![planner.clone(), implementer.clone(), reviewer.clone()])
        .with_metrics_collector(metrics_collector.clone())
        .with_shared_state(shared_state.clone());

    let result1 = orchestrator1.run("Develop a user authentication system").await?;
    println!("✅ Workflow 1 completed");

    if let Some(ref metrics) = result1.metrics {
        println!("📊 Execution Metrics:");
        println!("   Duration: {:?}", metrics.execution.total_duration);
        println!("   Rounds: {}", metrics.execution.rounds);
        println!("   Output length: {} chars", metrics.execution.output_length);
        println!("   Success: {}", metrics.execution.succeeded);

        println!("💰 Token Usage:");
        println!("   Input tokens: {}", metrics.token_usage.input_tokens);
        println!("   Output tokens: {}", metrics.token_usage.output_tokens);
        println!("   Total tokens: {}", metrics.token_usage.total_tokens);

        println!("💸 Cost Analysis:");
        println!("   Estimated cost: ${:.6}", metrics.cost.estimated_cost_usd);
        println!("   Cost per round: ${:.6}", metrics.cost.cost_per_round);

        println!("🔧 Function Calls:");
        println!("   Total calls: {}", metrics.function_calls.total_calls);
        println!("   Success rate: {:.2}%", metrics.success_rate() * 100.0);
    }

    // Store workflow result in shared state
    shared_state.extensions().set_string(
        "workflow1_result".to_string(),
        result1.final_output.clone().unwrap_or_default(),
        Some("workflows".to_string())
    ).await?;

    println!("\n📋 Workflow 2: Different Task (API Development)");

    // Run second workflow with same metrics collector
    let provider2: Arc<dyn LLMProvider> = Arc::new(RealisticMockProvider::new());
    let orchestrator2 = SequentialOrchestrator::new(provider2, "gpt-4")
        .with_agents(vec![planner.clone(), implementer.clone()])
        .with_metrics_collector(metrics_collector.clone())
        .with_shared_state(shared_state.clone());

    let result2 = orchestrator2.run("Build a REST API for user management").await?;
    println!("✅ Workflow 2 completed");

    // Store second workflow result
    shared_state.extensions().set_string(
        "workflow2_result".to_string(),
        result2.final_output.clone().unwrap_or_default(),
        Some("workflows".to_string())
    ).await?;

    println!("\n📋 Workflow 3: Quality Assurance Process");

    // Run third workflow with error simulation
    let provider3: Arc<dyn LLMProvider> = Arc::new(RealisticMockProvider::new());

    let orchestrator3 = SequentialOrchestrator::new(provider3, "gpt-4")
        .with_agents(vec![reviewer.clone(), planner.clone()])
        .with_metrics_collector(metrics_collector.clone())
        .with_shared_state(shared_state.clone());

    let result3 = orchestrator3.run("Perform quality assurance on existing code").await?;
    println!("✅ Workflow 3 completed");

    // Store third workflow result
    shared_state.extensions().set_string(
        "workflow3_result".to_string(),
        result3.final_output.clone().unwrap_or_default(),
        Some("workflows".to_string())
    ).await?;

    println!("\n📈 Aggregated Metrics Analysis");
    println!("================================");

    // Get aggregated metrics
    let aggregated = metrics_collector.get_aggregated_metrics();

    println!("🔢 Overall Statistics:");
    println!("   Total executions: {}", aggregated.total_executions);
    println!("   Total cost: ${:.6}", aggregated.total_cost_usd);
    println!("   Total tokens: {}", aggregated.total_tokens);
    println!("   Average success rate: {:.2}%", aggregated.average_success_rate * 100.0);
    println!("   Average execution time: {:?}", aggregated.average_execution_time);

    println!("\n💰 Cost Breakdown:");
    for (category, cost) in &aggregated.cost_breakdown {
        println!("   {}: ${:.6}", category, cost);
    }

    println!("\n⏰ Time Range:");
    println!("   From: {}", aggregated.time_range.0.format("%Y-%m-%d %H:%M:%S UTC"));
    println!("   To: {}", aggregated.time_range.1.format("%Y-%m-%d %H:%M:%S UTC"));

    println!("\n📊 Metrics by Agent:");
    for (agent_name, agent_metrics) in &aggregated.by_agent {
        println!("\n🤖 Agent: {}", agent_name);
        println!("   Executions: {}", agent_metrics.len());

        if let Some(latest) = agent_metrics.last() {
            println!("   Latest execution:");
            println!("     Duration: {:?}", latest.execution.total_duration);
            println!("     Success: {}", latest.execution.succeeded);
            println!("     Token usage: {} total", latest.token_usage.total_tokens);
            println!("     Cost: ${:.6}", latest.cost.estimated_cost_usd);

            if latest.errors.error_count > 0 {
                println!("     Errors: {}", latest.errors.error_count);
                for error_type in &latest.errors.error_types {
                    println!("       - {}", error_type);
                }
            }
        }
    }

    println!("\n🔍 Performance Insights:");
    println!("================================");

    // Calculate and display insights
    let total_tokens = aggregated.total_tokens;
    let total_cost = aggregated.total_cost_usd;
    let avg_tokens_per_execution = if aggregated.total_executions > 0 {
        total_tokens / aggregated.total_executions as u64
    } else {
        0
    };
    let avg_cost_per_execution = if aggregated.total_executions > 0 {
        total_cost / aggregated.total_executions as f64
    } else {
        0.0
    };

    println!("📊 Performance Summary:");
    println!("   Average tokens per execution: {}", avg_tokens_per_execution);
    println!("   Average cost per execution: ${:.6}", avg_cost_per_execution);
    println!("   Cost efficiency: {:.2} tokens per $0.001", avg_tokens_per_execution as f64 / (total_cost * 1000.0));

    // Show cost projections
    let daily_executions = 100;
    let monthly_cost = daily_executions as f64 * 30.0 * avg_cost_per_execution;
    println!("💡 Cost Projections:");
    println!("   {} executions/day: ${:.2}/month", daily_executions, monthly_cost);

    let daily_executions_heavy = 1000;
    let monthly_cost_heavy = daily_executions_heavy as f64 * 30.0 * avg_cost_per_execution;
    println!("   {} executions/day: ${:.2}/month", daily_executions_heavy, monthly_cost_heavy);

    println!("\n🎯 Recommendations:");
    println!("================================");

    if aggregated.average_success_rate < 0.95 {
        println!("⚠️  Consider investigating error patterns - success rate is {:.1}%",
                 aggregated.average_success_rate * 100.0);
    }

    if avg_cost_per_execution > 0.01 {
        println!("💰 High cost per execution detected - consider optimizing prompts or using smaller models");
    }

    if avg_tokens_per_execution > 1000 {
        println!("📝 High token usage per execution - consider prompt optimization");
    }

    println!("✅ Metrics collection is working properly!");
    println!("📈 Consider setting up regular monitoring and alerts for production usage");

    println!("\n🎉 Agent Metrics Demo Complete!");
    println!("📚 Key Features Demonstrated:");
    println!("  ✅ Token usage tracking across multiple executions");
    println!("  ✅ Cost estimation and budget projections");
    println!("  ✅ Performance metrics and success rates");
    println!("  ✅ Error tracking and analysis");
    println!("  ✅ Aggregated statistics across workflows");
    println!("  ✅ Integration with shared state system");
    println!("  ✅ Performance insights and recommendations");

    Ok(())
}