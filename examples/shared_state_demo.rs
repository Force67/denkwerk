use denkwerk::{
    Agent, InMemorySharedStateStore, LLMProvider, SequentialOrchestrator, SharedStateContextExt,
    SharedStateContext,
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
struct DocumentAnalysis {
    title: String,
    key_points: Vec<String>,
    sentiment: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct DraftContent {
    introduction: String,
    main_body: String,
    conclusion: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Shared States Demo\n");

    // Create a shared state store
    let shared_state = Arc::new(InMemorySharedStateStore::new());

    // Create mock provider with sequential responses
    let provider: Arc<dyn LLMProvider> = Arc::new(MockProvider::new(vec![
        "Document analyzed. Title: 'Market Trends Report', Key points: ['Growth in AI', 'Consumer behavior shift'], Sentiment: positive".to_string(),
        "Draft created based on analysis. Introduction: 'The market is evolving rapidly...', Main body: 'Key trends include AI adoption...', Conclusion: 'Businesses should adapt...'".to_string(),
        "Final polish completed. Document is now ready for publication.".to_string(),
    ]));

    // Create agents for different stages
    let analyzer = Agent::from_string(
        "Document Analyzer",
        "Analyze the document and extract key information including title, key points, and sentiment."
    );

    let drafter = Agent::from_string(
        "Content Drafter",
        "Create a draft document based on the analysis. Structure it with introduction, main body, and conclusion."
    );

    let editor = Agent::from_string(
        "Final Editor",
        "Review and polish the draft for final publication."
    );

    // Create orchestrator with shared state
    let orchestrator = SequentialOrchestrator::new(provider, "mock-model")
        .with_agents(vec![analyzer, drafter, editor])
        .with_shared_state(shared_state.clone());

    // Store initial context in shared state
    shared_state.extensions().set_string(
        "workflow_context".to_string(),
        "Analyzing market trends report".to_string(),
        Some("demo".to_string())
    ).await?;

    println!("Running document processing workflow with shared states...\n");

    // Run the workflow
    let result = orchestrator.run("Process this market trends document").await?;

    println!("Workflow completed successfully!");
    println!("Final output: {:?}", result.final_output);

    // Demonstrate shared state usage
    println!("\nChecking shared state contents:");

    // List all states in the "demo" scope
    let demo_states = shared_state.list_state_ids(Some("demo")).await?;
    println!("States in 'demo' scope: {:?}", demo_states);

    // Show that we can store and retrieve structured data
    let analysis = DocumentAnalysis {
        title: "Market Trends Report".to_string(),
        key_points: vec![
            "Growth in AI".to_string(),
            "Consumer behavior shift".to_string(),
        ],
        sentiment: "positive".to_string(),
    };

    shared_state.extensions().set_object(
        "document_analysis".to_string(),
        &analysis,
        Some("processing".to_string())
    ).await?;

    let retrieved_analysis: Option<DocumentAnalysis> = shared_state.extensions().get_object(
        "document_analysis",
        Some("processing")
    ).await?;

    println!("Retrieved analysis: {:?}", retrieved_analysis);

    // Store draft content
    let draft = DraftContent {
        introduction: "The market is evolving rapidly...".to_string(),
        main_body: "Key trends include AI adoption...".to_string(),
        conclusion: "Businesses should adapt...".to_string(),
    };

    shared_state.extensions().set_object(
        "draft_content".to_string(),
        &draft,
        Some("processing".to_string())
    ).await?;

    // Show all processing states
    let processing_states = shared_state.list_state_ids(Some("processing")).await?;
    println!("States in 'processing' scope: {:?}", processing_states);

    println!("\nShared States Demo Complete!");
    println!("Summary:");
    println!("  - Created shared state store with scoped storage");
    println!("  - Ran workflow with shared state context");
    println!("  - Stored and retrieved both string and structured data");
    println!("  - Demonstrated namespacing with different scopes");

    Ok(())
}