use denkwerk::{
    kernel_function, Agent, FunctionRegistry, InMemorySharedStateStore,
    LLMProvider, SequentialOrchestrator, SharedStateContextExt, SharedStateContext,
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

// Global shared state reference for functions
static mut SHARED_STATE: Option<Arc<InMemorySharedStateStore>> = None;

fn set_shared_state_reference(state: Arc<InMemorySharedStateStore>) {
    unsafe {
        SHARED_STATE = Some(state);
    }
}

fn get_shared_state() -> Option<Arc<InMemorySharedStateStore>> {
    unsafe {
        SHARED_STATE.clone()
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct ResearchData {
    topic: String,
    findings: Vec<String>,
    confidence: f64,
}

#[derive(Serialize, Deserialize, Debug)]
struct CompiledReport {
    summary: String,
    recommendations: Vec<String>,
    data_sources: Vec<String>,
}

// Functions that can access shared state
#[kernel_function]
async fn store_research_finding(topic: String, finding: String, confidence: f64) -> Result<String, String> {
    let shared_state = get_shared_state().ok_or("Shared state not initialized")?;

    let mut findings = shared_state.extensions().get_object::<Vec<String>>(&format!("findings_{}", topic), Some("research")).await
        .map_err(|e| format!("Failed to read findings: {}", e))?
        .unwrap_or_default();

    findings.push(finding.clone());

    shared_state.extensions().set_object(
        format!("findings_{}", topic),
        &findings,
        Some("research".to_string())
    ).await.map_err(|e| format!("Failed to store findings: {}", e))?;

    // Update confidence score
    shared_state.extensions().set_object(
        format!("confidence_{}", topic),
        &confidence,
        Some("research".to_string())
    ).await.map_err(|e| format!("Failed to store confidence: {}", e))?;

    Ok(format!("Stored finding: {} with confidence {:.2}", finding, confidence))
}

#[kernel_function]
async fn compile_research_report(topic: String) -> Result<String, String> {
    let shared_state = get_shared_state().ok_or("Shared state not initialized")?;

    let findings = shared_state.extensions().get_object::<Vec<String>>(&format!("findings_{}", topic), Some("research")).await
        .map_err(|e| format!("Failed to read findings: {}", e))?
        .unwrap_or_default();

    let confidence = shared_state.extensions().get_object::<f64>(&format!("confidence_{}", topic), Some("research")).await
        .map_err(|e| format!("Failed to read confidence: {}", e))?
        .unwrap_or(0.0);

    let report = CompiledReport {
        summary: format!("Research summary for topic: {}", topic),
        recommendations: vec![
            "Analyze findings further".to_string(),
            "Consider additional research".to_string(),
        ],
        data_sources: vec![format!("{} findings ({})", topic, findings.len())],
    };

    shared_state.extensions().set_object(
        format!("report_{}", topic),
        &report,
        Some("reports".to_string())
    ).await.map_err(|e| format!("Failed to store report: {}", e))?;

    Ok(format!("Compiled report with {} findings, confidence: {:.2}", findings.len(), confidence))
}

#[kernel_function]
async fn get_shared_state_count(scope: String) -> Result<String, String> {
    let shared_state = get_shared_state().ok_or("Shared state not initialized")?;

    let count = shared_state.list_state_ids(Some(&scope)).await
        .map_err(|e| format!("Failed to list states: {}", e))?
        .len();

    Ok(format!("Shared state has {} entries in scope '{}'", count, scope))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔗 Shared States with Agent Functions Demo\n");

    // Create shared state store
    let shared_state = Arc::new(InMemorySharedStateStore::new());
    set_shared_state_reference(shared_state.clone());

    // Create function registry
    let mut function_registry = FunctionRegistry::new();

    // Register our shared state functions
    function_registry.register(store_research_finding_kernel());
    function_registry.register(compile_research_report_kernel());
    function_registry.register(get_shared_state_count_kernel());

    // Create mock provider that will trigger function calls
    let provider: Arc<dyn LLMProvider> = Arc::new(MockProvider::new(vec![
        "I'll research this topic and store my findings. [store_research_finding(topic='climate_change', finding='Global temperatures rising', confidence=0.95)] [store_research_finding(topic='climate_change', finding='Ice caps melting rapidly', confidence=0.92)] [get_shared_state_count(scope='research')]".to_string(),
        "Now I'll compile a comprehensive report based on the research findings. [compile_research_report(topic='climate_change')] [get_shared_state_count(scope='reports')]".to_string(),
        "Final review completed. The report is ready for publication. [get_shared_state_count(scope='research')] [get_shared_state_count(scope='reports')]".to_string(),
    ]));

    // Create agents with function access
    let mut researcher_registry = FunctionRegistry::new();
    researcher_registry.register(store_research_finding_kernel());
    researcher_registry.register(get_shared_state_count_kernel());

    let mut analyst_registry = FunctionRegistry::new();
    analyst_registry.register(compile_research_report_kernel());
    analyst_registry.register(get_shared_state_count_kernel());

    let mut reviewer_registry = FunctionRegistry::new();
    reviewer_registry.register(get_shared_state_count_kernel());

    let researcher = Agent::from_string(
        "Research Specialist",
        "Conduct thorough research on the given topic and store findings using the available functions."
    ).with_function_registry(Arc::new(researcher_registry));

    let analyst = Agent::from_string(
        "Report Analyst",
        "Analyze research findings and compile comprehensive reports using the available functions."
    ).with_function_registry(Arc::new(analyst_registry));

    let reviewer = Agent::from_string(
        "Final Reviewer",
        "Review the compiled report and ensure it's ready for publication. Check the shared state status."
    ).with_function_registry(Arc::new(reviewer_registry));

    // Create orchestrator with shared state
    let orchestrator = SequentialOrchestrator::new(provider, "mock-model")
        .with_agents(vec![researcher, analyst, reviewer])
        .with_shared_state(shared_state.clone());

    println!("🔬 Running research workflow with shared state functions...\n");

    // Store initial context
    shared_state.extensions().set_string(
        "workflow_type".to_string(),
        "research_analysis".to_string(),
        Some("metadata".to_string())
    ).await?;

    // Run the workflow
    let result = orchestrator.run("Research and analyze climate change impacts").await?;

    println!("✅ Research workflow completed!");
    println!("📋 Final output: {:?}", result.final_output);

    // Demonstrate the shared state contents
    println!("\n🗂️  Shared State Analysis:");

    // Check different scopes
    for scope in ["research", "reports", "metadata"] {
        let states = shared_state.list_state_ids(Some(scope)).await?;
        println!("Scope '{}': {} entries", scope, states.len());

        for state_id in states {
            let value = shared_state.read_state(&state_id, Some(scope)).await?;
            println!("  - {}: {:?}", state_id, value);
        }
    }

    // Show structured data retrieval
    if let Some(report) = shared_state.extensions().get_object::<CompiledReport>("report_climate_change", Some("reports")).await? {
        println!("\n📊 Compiled Report:");
        println!("  Summary: {}", report.summary);
        println!("  Recommendations: {:?}", report.recommendations);
        println!("  Data Sources: {:?}", report.data_sources);
    }

    println!("\n🎉 Shared States Agent Functions Demo Complete!");
    println!("📈 Key Features Demonstrated:");
    println!("  ✅ Agent functions can access and modify shared state");
    println!("  ✅ Persistent state across multiple workflow steps");
    println!("  ✅ Scoped storage for organized data management");
    println!("  ✅ Both structured and unstructured data support");
    println!("  ✅ Real-time state inspection during workflow execution");

    Ok(())
}