use denkwerk::{
    Agent, InMemorySharedStateStore, LLMProvider, SequentialOrchestrator, ConcurrentOrchestrator,
    GroupChatOrchestrator, RoundRobinGroupChatManager, SharedStateContextExt, SharedStateContext,
};
use std::sync::Arc;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

// Mock provider for demonstration
struct MockProvider {
    responses: std::sync::Mutex<Vec<String>>,
}

impl MockProvider {
    fn new(responses: Vec<String>) -> Self {
        Self {
            responses: std::sync::Mutex::new(responses),
        }
    }
}

#[async_trait]
impl LLMProvider for MockProvider {
    async fn complete(&self, _request: denkwerk::CompletionRequest) -> Result<denkwerk::CompletionResponse, denkwerk::LLMError> {
        let mut guard = self.responses.lock().unwrap();
        let content = guard.remove(0);
        drop(guard);

        Ok(denkwerk::CompletionResponse {
            message: denkwerk::ChatMessage::assistant(content),
            usage: None,
            reasoning: None,
        })
    }

    fn name(&self) -> &'static str {
        "mock"
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct TaskResult {
    agent: String,
    result: String,
    timestamp: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Shared States with Multiple Orchestrators Demo\n");

    // Create a shared state store
    let shared_state = Arc::new(InMemorySharedStateStore::new());

    // Create mock provider with responses for different scenarios
    let provider: Arc<dyn LLMProvider> = Arc::new(MockProvider::new(vec![
        "Data collected: User preferences, system metrics, error logs".to_string(),
        "Analysis complete: Patterns identified, recommendations generated".to_string(),
        "Report drafted: Executive summary, detailed findings, action items".to_string(),
    ]));

    // Create agents for different stages
    let collector = Agent::from_string(
        "Data Collector",
        "Collect relevant data and information for analysis."
    );

    let analyzer = Agent::from_string(
        "Data Analyzer",
        "Analyze collected data and identify patterns and insights."
    );

    let reporter = Agent::from_string(
        "Report Writer",
        "Create comprehensive reports based on analysis results."
    );

    println!("📋 Scenario 1: Sequential Workflow with Shared State\n");

    // Run sequential workflow
    let sequential_orchestrator = SequentialOrchestrator::new(provider.clone(), "mock-model")
        .with_agents(vec![collector.clone(), analyzer.clone(), reporter.clone()])
        .with_shared_state(shared_state.clone());

    // Store initial context
    shared_state.extensions().set_string(
        "workflow_type".to_string(),
        "sequential_analysis".to_string(),
        Some("scenario1".to_string())
    ).await?;

    let sequential_result = sequential_orchestrator.run("Analyze system performance").await?;
    println!("Sequential workflow completed");
    println!("Final output: {:?}", sequential_result.final_output);

    // Store results from sequential workflow
    let task_result = TaskResult {
        agent: "sequential_workflow".to_string(),
        result: sequential_result.final_output.unwrap_or_default(),
        timestamp: chrono::Utc::now().to_rfc3339(),
    };

    shared_state.extensions().set_object(
        "sequential_result".to_string(),
        &task_result,
        Some("scenario1".to_string())
    ).await?;

    println!("\n📋 Scenario 2: Concurrent Workflow with Shared State\n");

    // Create another mock provider for concurrent execution
    let concurrent_provider: Arc<dyn LLMProvider> = Arc::new(MockProvider::new(vec![
        "Network analysis: Latency measured, bandwidth calculated".to_string(),
        "Security audit: Vulnerabilities scanned, risks assessed".to_string(),
        "Performance review: CPU usage monitored, memory analyzed".to_string(),
    ]));

    // Create concurrent agents
    let network_analyst = Agent::from_string("Network Analyst", "Analyze network performance and connectivity.");
    let security_auditor = Agent::from_string("Security Auditor", "Audit system security and identify vulnerabilities.");
    let performance_reviewer = Agent::from_string("Performance Reviewer", "Review system performance metrics.");

    // Run concurrent workflow
    let concurrent_orchestrator = ConcurrentOrchestrator::new(concurrent_provider, "mock-model")
        .with_agents(vec![network_analyst, security_auditor, performance_reviewer])
        .with_shared_state(shared_state.clone());

    // Update workflow context
    shared_state.extensions().set_string(
        "workflow_type".to_string(),
        "concurrent_analysis".to_string(),
        Some("scenario2".to_string())
    ).await?;

    let concurrent_result = concurrent_orchestrator.run("Perform comprehensive system analysis").await?;
    println!("Concurrent workflow completed");
    println!("Concurrent results count: {}", concurrent_result.results.len());

    // Store concurrent results
    let concurrent_summary = format!("Concurrent analysis completed with {} parallel tasks", concurrent_result.results.len());
    shared_state.extensions().set_string(
        "concurrent_summary".to_string(),
        concurrent_summary,
        Some("scenario2".to_string())
    ).await?;

    println!("\n📋 Scenario 3: Group Chat with Shared State\n");

    // Create another mock provider for group chat
    let chat_provider: Arc<dyn LLMProvider> = Arc::new(MockProvider::new(vec![
        "Initial assessment: System shows moderate performance".to_string(),
        "Network perspective: Latency is within acceptable range".to_string(),
        "Security viewpoint: No critical vulnerabilities detected".to_string(),
        "Final consensus: System is healthy with minor optimizations needed".to_string(),
    ]));

    // Create group chat manager
    let manager = RoundRobinGroupChatManager::new()
        .with_maximum_rounds(Some(4));

    // Create group chat participants
    let coordinator = Agent::from_string("Coordinator", "Coordinate the discussion and ensure all perspectives are considered.");
    let network_specialist = Agent::from_string("Network Specialist", "Provide expertise on network-related issues.");
    let security_specialist = Agent::from_string("Security Specialist", "Provide expertise on security and compliance.");

    // Run group chat workflow
    let mut group_chat_orchestrator = GroupChatOrchestrator::new(chat_provider, "mock-model", manager)
        .with_agents(vec![coordinator, network_specialist, security_specialist])
        .with_shared_state(shared_state.clone());

    // Update workflow context
    shared_state.extensions().set_string(
        "workflow_type".to_string(),
        "group_discussion".to_string(),
        Some("scenario3".to_string())
    ).await?;

    let group_chat_result = group_chat_orchestrator.run("Discuss overall system health").await?;
    println!("Group chat completed");
    println!("💬 Discussion rounds: {}", group_chat_result.rounds);

    // Store group chat results
    shared_state.extensions().set_string(
        "group_chat_consensus".to_string(),
        group_chat_result.final_output.unwrap_or_default(),
        Some("scenario3".to_string())
    ).await?;

    println!("\nAnalyzing Cross-Orchestrator Shared State\n");

    // Show that all orchestrators have been sharing the same state store
    println!("Shared State Analysis:");

    for scope in ["scenario1", "scenario2", "scenario3", "global"] {
        let states = shared_state.list_state_ids(Some(scope)).await?;
        if !states.is_empty() {
            println!("Scope '{}': {} entries", scope, states.len());

            for state_id in states {
                if let Ok(Some(value)) = shared_state.read_state(&state_id, Some(scope)).await {
                    match scope {
                        "scenario1" | "scenario2" | "scenario3" => {
                            println!("  {}: {:?}", state_id, value);
                        }
                        "global" => {
                            println!("  🌐 {}: {:?}", state_id, value);
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // Demonstrate cross-workflow data access
    println!("\n🔗 Cross-Workflow Data Access:");

    if let Some(sequential_result) = shared_state.extensions().get_object::<TaskResult>(
        "sequential_result",
        Some("scenario1")
    ).await? {
        println!("Retrieved sequential workflow result from group chat context:");
        println!("   Agent: {}", sequential_result.agent);
        println!("   Result: {}", sequential_result.result);
        println!("   Timestamp: {}", sequential_result.timestamp);
    }

    // Store a global insight that was discovered through the multiple workflows
    let global_insight = "System analysis reveals healthy performance across all dimensions with opportunities for minor optimizations";
    shared_state.extensions().set_string(
        "global_insight".to_string(),
        global_insight.to_string(),
        Some("global".to_string())
    ).await?;

    println!("\nMulti-Orchestrator Shared States Demo Complete!");
    println!("📈 Key Features Demonstrated:");
    println!("  Same shared state store used across different orchestrator types");
    println!("  Sequential workflow: Data persisted for subsequent access");
    println!("  Concurrent workflow: Parallel access to shared state");
    println!("  Group chat: Collaborative state management");
    println!("  Cross-workflow data sharing and access patterns");
    println!("  Scoped organization for different workflow scenarios");
    println!("  Persistent state across multiple workflow executions");

    Ok(())
}