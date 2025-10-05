use thiserror::Error;

#[derive(Debug, Error)]
pub enum LLMError {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("provider error: {0}")]
    Provider(String),

    #[error("missing API key: set the {0} environment variable")]
    MissingApiKey(&'static str),

    #[error("invalid response from provider: {0}")]
    InvalidResponse(&'static str),

    #[error("operation not supported: {0}")]
    Unsupported(&'static str),

    #[error("unknown function: {0}")]
    UnknownFunction(String),

    #[error("invalid function arguments: {0}")]
    InvalidFunctionArguments(String),

    #[error("kernel function execution failed ({function}): {message}")]
    FunctionExecution { function: String, message: String },
}
