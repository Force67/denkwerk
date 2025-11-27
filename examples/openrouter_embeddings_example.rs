use denkwerk::{providers::openrouter::OpenRouter, types::EmbeddingRequest, LLMProvider};
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Note: This example requires the OPENROUTER_API_KEY environment variable to be set

    // Create an OpenRouter provider
    let provider = OpenRouter::from_env()?;

    println!("Testing OpenRouter embeddings functionality...");

    // Create an embedding request using OpenRouter's model format
    let request = EmbeddingRequest::new(
        "openai/text-embedding-3-small", // OpenRouter uses provider/model format
        vec![
            "Hello, OpenRouter!".to_string(),
            "The quick brown fox jumps over the lazy dog.".to_string(),
            "OpenRouter provides access to multiple AI models.".to_string(),
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

                // Example of using different OpenRouter embedding models
                println!("\nTesting with a different model...");
                let large_model_request = EmbeddingRequest::new(
                    "openai/text-embedding-3-large",
                    vec!["This is a test for the larger embedding model.".to_string()],
                );

                match provider.create_embeddings(large_model_request).await {
                    Ok(large_response) => {
                        if let Some(embedding) = large_response.data.first() {
                            println!("Large model embedding dimension: {} (expected: 3072)",
                                    embedding.embedding.len());
                        }
                    }
                    Err(e) => {
                        println!("Error with large model: {}", e);
                    }
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