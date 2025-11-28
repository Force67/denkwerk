use denkwerk::{providers::azure_openai::AzureOpenAI, types::EmbeddingRequest, LLMProvider};
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Note: This example requires the AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT
    // environment variables to be set
    // You can also set AZURE_OPENAI_API_VERSION

    // Create an Azure OpenAI provider
    let provider = AzureOpenAI::from_env()?;

    println!("Testing Azure OpenAI embeddings functionality...");

    // Create an embedding request using your deployment name
    let request = EmbeddingRequest::new(
        "text-embedding-ada-002", // Your Azure deployment name
        vec![
            "Hello, Azure OpenAI!".to_string(),
            "The quick brown fox jumps over the lazy dog.".to_string(),
            "Azure AI services are powerful.".to_string(),
        ],
    );

    println!("Request created with {} texts to embed", request.input.len());

    // Test if the provider supports embeddings
    let capabilities = provider.capabilities();
    println!("Provider supports embeddings: {}", capabilities.supports_embeddings);

    if capabilities.supports_embeddings {
        // Create embeddings
        println!("Creating embeddings...");
        match provider.create_embeddings(request).await {
            Ok(response) => {
                println!("Embeddings created successfully!");
                println!("Model used: {}", response.model);
                println!("Number of embeddings: {}", response.data.len());

                if let Some(usage) = response.usage {
                    println!("Prompt tokens: {}", usage.prompt_tokens);
                    println!("Total tokens: {}", usage.total_tokens);
                }

                // Print details of each embedding
                for (i, embedding) in response.data.iter().enumerate() {
                    println!("Embedding {}: {} (dimension: {})",
                            i, embedding.object, embedding.embedding.len());
                }

                // Test similarity between first two embeddings
                if response.data.len() >= 2 {
                    let emb1 = &response.data[0].embedding;
                    let emb2 = &response.data[1].embedding;
                    let similarity = cosine_similarity(emb1, emb2);
                    println!("Cosine similarity between first two texts: {:.4}", similarity);
                }
            }
            Err(e) => {
                eprintln!("Error creating embeddings: {}", e);
            }
        }
    } else {
        println!("This provider does not support embeddings");
    }

    Ok(())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}