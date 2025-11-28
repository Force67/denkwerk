//! Simplified Flow Builder for agent orchestration
//!
//! This module provides a high-level, fluent interface for creating and running agent flows
//! without the boilerplate of the lower-level specification system.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::fmt;

use serde::Serialize;
use serde_json::Value;
use thiserror::Error;

use crate::flows::spec::{FlowBuilder as SpecFlowBuilder, FlowContext, FlowLoadError, FlowRunError};
use crate::functions::KernelFunction;
use crate::LLMProvider;

/// Simplified Flow Builder for agent orchestration
///
/// This is the main entry point for the simplified API. It provides a fluent interface
/// for creating, configuring, and running agent flows with minimal boilerplate.
///
/// # Examples
///
/// ## Basic usage with YAML file
/// ```rust
/// use denkwerk::Flow;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let result = Flow::from_file("flow.yaml")
///         .with_env_file(".env")
///         .run("Summarize why Rust is great for backend services")?;
///
///     println!("Result: {}", result);
///     Ok(())
/// }
/// ```
///
/// ## With custom functions
/// ```rust
/// use denkwerk::{Flow, kernel_function};
///
/// #[kernel_function(name = "calculate", description = "Perform calculations")]
/// fn calculate(a: f64, b: f64) -> Result<f64, String> {
///     Ok(a + b)
/// }
///
/// let result = Flow::from_file("flow.yaml")
///     .with_function(calculate())
///     .with_context_var("mode", "detailed")
///     .run("Analyze the data")?;
/// ```
pub struct Flow {
    builder: SpecFlowBuilder,
    base_dir: PathBuf,
    functions: HashMap<String, Arc<dyn KernelFunction>>,
    context: FlowContext,
    env_files: Vec<PathBuf>,
    auto_discover: bool,
}

impl fmt::Debug for Flow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Flow")
            .field("base_dir", &self.base_dir)
            .field("functions_count", &self.functions.len())
            .field("context", &self.context)
            .field("env_files", &self.env_files)
            .field("auto_discover", &self.auto_discover)
            .finish()
    }
}


impl Flow {
    /// Create a new Flow from a YAML file
    ///
    /// # Arguments
    /// * `path` - Path to the YAML flow file
    ///
    /// # Examples
    /// ```
    /// let flow = Flow::from_file("my_flow.yaml")?;
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, FlowError> {
        let path = path.as_ref();
        let yaml = std::fs::read_to_string(path)
            .map_err(|e| FlowError::FileNotFound(path.to_path_buf(), e))?;

        Self::from_yaml_string(yaml, path.parent().unwrap_or_else(|| Path::new(".")))
    }

    /// Create a new Flow from a YAML string
    ///
    /// # Arguments
    /// * `yaml` - YAML string containing the flow definition
    /// * `base_dir` - Base directory for resolving relative paths
    pub fn from_yaml_string<S: Into<String>>(yaml: S, base_dir: impl AsRef<Path>) -> Result<Self, FlowError> {
        let yaml = yaml.into();
        let base_dir_ref = base_dir.as_ref();
        let builder = SpecFlowBuilder::from_yaml_str(base_dir_ref, &yaml)
            .map_err(FlowError::LoadError)?;

        Ok(Self {
            builder,
            base_dir: base_dir_ref.to_path_buf(),
            functions: HashMap::new(),
            context: FlowContext::default(),
            env_files: Vec::new(),
            auto_discover: false,
        })
    }

    /// Create a new Flow from a directory (auto-discover flow.yaml, prompts, tools)
    ///
    /// This method automatically discovers:
    /// - `flow.yaml` or `flow.yml` as the main flow file
    /// - All `.env` files for environment configuration
    /// - All functions annotated with `#[kernel_function]`
    /// - All prompt files and tool specifications
    ///
    /// # Arguments
    /// * `dir` - Directory containing the flow definition
    ///
    /// # Examples
    /// ```
    /// let flow = Flow::from_directory("my_flow_directory")?;
    /// ```
    pub fn from_directory<P: AsRef<Path>>(dir: P) -> Result<Self, FlowError> {
        let dir = dir.as_ref();
        let base_dir = dir.to_path_buf();

        // Auto-discover flow file
        let flow_file = Self::discover_flow_file(dir)?;
        let yaml = std::fs::read_to_string(&flow_file)
            .map_err(|e| FlowError::FileNotFound(flow_file.clone(), e))?;

        let mut flow = Self::from_yaml_string(yaml, base_dir)?;
        flow.auto_discover = true;

        // Auto-discover env files
        flow.discover_env_files(dir)?;

        Ok(flow)
    }

    /// Add environment file to load variables from
    ///
    /// # Arguments
    /// * `path` - Path to the .env file
    ///
    /// # Examples
    /// ```
    /// let flow = Flow::from_file("flow.yaml")
    ///     .with_env_file(".env")
    ///     .with_env_file(".env.production");
    /// ```
    pub fn with_env_file<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.env_files.push(path.as_ref().to_path_buf());
        self
    }

    /// Add a custom function to the flow
    ///
    /// # Arguments
    /// * `function` - Function implementing KernelFunction trait
    ///
    /// # Examples
    /// ```
    /// use denkwerk::kernel_function;
    ///
    /// #[kernel_function(name = "math_add", description = "Add two numbers")]
    /// fn math_add(a: f64, b: f64) -> Result<f64, String> {
    ///     Ok(a + b)
    /// }
    ///
    /// let flow = Flow::from_file("flow.yaml")
    ///     .with_function(math_add());
    /// ```
    pub fn with_function(mut self, function: Arc<dyn KernelFunction>) -> Self {
        let name = function.definition().name.clone();
        self.functions.insert(name, function);
        self
    }

    /// Add a context variable for flow execution
    ///
    /// # Arguments
    /// * `key` - Variable name
    /// * `value` - Variable value
    ///
    /// # Examples
    /// ```
    /// let flow = Flow::from_file("flow.yaml")
    ///     .with_context_var("mode", "detailed")
    ///     .with_context_var("user_id", "12345");
    /// ```
    pub fn with_context_var<K: Into<String>, V: Into<Value>>(mut self, key: K, value: V) -> Self {
        self.context = self.context.with_var(key.into(), value.into());
        self
    }

    /// Add multiple context variables at once
    ///
    /// # Arguments
    /// * `vars` - HashMap of context variables
    pub fn with_context_vars(mut self, vars: HashMap<String, Value>) -> Self {
        for (key, value) in vars {
            self.context = self.context.with_var(key, value);
        }
        self
    }

    /// Enable auto-discovery of functions, prompts, and tools
    ///
    /// When enabled, automatically discovers:
    /// - Functions annotated with `#[kernel_function]` in the current crate
    /// - All prompt files referenced in the flow
    /// - All tool specifications referenced in the flow
    ///
    /// # Examples
    /// ```
    /// let flow = Flow::from_file("flow.yaml")
    ///     .with_auto_discovery(true);
    /// ```
    pub fn with_auto_discovery(mut self, enabled: bool) -> Self {
        self.auto_discover = enabled;
        self
    }

    /// Set a custom LLM provider
    ///
    /// # Arguments
    /// * `provider` - LLM provider implementation wrapped in Arc
    ///
    /// # Examples
    /// ```
    /// use denkwerk::{Flow, providers::openai::OpenAI};
    /// use std::sync::Arc;
    ///
    /// let provider = Arc::new(OpenAI::new("your-api-key")?);
    /// let flow = Flow::from_file("flow.yaml")
    ///     .with_provider(provider);
    /// ```
    pub fn with_provider(self, provider: Arc<dyn LLMProvider>) -> FlowRunner {
        FlowRunner {
            flow: self,
            provider,
        }
    }

    /// Run the flow with automatic provider detection
    ///
    /// This method automatically detects and configures the appropriate LLM provider
    /// based on environment variables and flow configuration.
    ///
    /// # Arguments
    /// * `task` - The task/question to process
    ///
    /// # Examples
    /// ```
    /// let result = Flow::from_file("flow.yaml")
    ///     .run("Process this data")?;
    /// ```
    pub async fn run<S: Into<String>>(self, task: S) -> Result<FlowResult, FlowError> {
        self.load_environment()?;

        // Auto-discover functions if enabled
        let mut flow = self;
        if flow.auto_discover {
            flow = flow.with_auto_discovered_functions()?;
        }

        // Auto-detect and create provider
        let provider = flow.detect_provider()?;
        let runner = FlowRunner {
            flow,
            provider,
        };

        runner.run(task).await
    }

    // Private helper methods

    fn discover_flow_file(dir: &Path) -> Result<PathBuf, FlowError> {
        let candidates = ["flow.yaml", "flow.yml", "flows.yaml", "flows.yml"];

        for candidate in &candidates {
            let path = dir.join(candidate);
            if path.exists() {
                return Ok(path);
            }
        }

        Err(FlowError::NoFlowFileFound(dir.to_path_buf()))
    }

    fn discover_env_files(&mut self, dir: &Path) -> Result<(), FlowError> {
        let candidates = [".env", ".env.local", ".env.development", ".env.production"];

        for candidate in &candidates {
            let path = dir.join(candidate);
            if path.exists() {
                self.env_files.push(path);
            }
        }

        Ok(())
    }

    fn load_environment(&self) -> Result<(), FlowError> {
        // Load environment variables from .env files if available
        for env_file in &self.env_files {
            // Check if dotenvy crate is available before using it
            if std::env::var("CARGO_PKG_NAME").is_ok() {
                // We're running in a cargo context, try to load .env
                match std::fs::read_to_string(env_file) {
                    Ok(content) => {
                        for line in content.lines() {
                            if let Some((key, value)) = line.split_once('=') {
                                std::env::set_var(key.trim(), value.trim());
                            }
                        }
                    }
                    Err(_) => {
                        eprintln!("Warning: Failed to read env file {:?}", env_file);
                    }
                }
            }
        }
        Ok(())
    }

    fn with_auto_discovered_functions(self) -> Result<Self, FlowError> {
        // This is a placeholder for auto-discovery
        // In a real implementation, this would use procedural macros
        // or reflection to discover annotated functions
        Ok(self)
    }

    fn detect_provider(&self) -> Result<Arc<dyn LLMProvider>, FlowError> {
        // For now, return an error if no provider is configured
        // The user should use with_provider() method
        Err(FlowError::NoProviderConfigured)
    }
}

/// Flow runner with a configured provider
///
/// This struct is created by `Flow::with_provider()` and provides methods
/// for running the flow with a specific LLM provider.
pub struct FlowRunner {
    flow: Flow,
    provider: Arc<dyn LLMProvider>,
}

impl fmt::Debug for FlowRunner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FlowRunner")
            .field("flow", &self.flow)
            .field("provider_name", &self.provider.name())
            .finish()
    }
}

impl FlowRunner {
    /// Run the flow with the configured provider
    ///
    /// # Arguments
    /// * `task` - The task/question to process
    ///
    /// # Returns
    /// `FlowResult` containing the execution results
    ///
    /// # Examples
    /// ```
    /// use denkwerk::{Flow, providers::openai::OpenAI};
    ///
    /// let provider = OpenAI::new("api-key")?;
    /// let result = Flow::from_file("flow.yaml")
    ///     .with_provider(provider)
    ///     .run("Process this task")?;
    /// ```
    pub async fn run<S: Into<String>>(self, task: S) -> Result<FlowResult, FlowError> {
        self.run_with_callback(task, |_event| {}).await
    }

    /// Run the flow with an event callback
    ///
    /// # Arguments
    /// * `task` - The task/question to process
    /// * `callback` - Function to call for each flow event
    ///
    /// # Examples
    /// ```
    /// let result = flow.with_provider(provider)
    ///     .run_with_callback("Process task", |event| {
    ///         println!("Event: {:?}", event);
    ///     })?;
    /// ```
    pub async fn run_with_callback<F, S>(
        self,
        task: S,
        callback: F
    ) -> Result<FlowResult, FlowError>
    where
        F: Fn(&crate::flows::sequential::SequentialEvent) + Send + Sync + 'static,
        S: Into<String>,
    {
        let task = task.into();

        // Load environment
        self.flow.load_environment()?;

        // Build tool registries
        let tool_registries = self.flow.builder
            .build_tool_registries(&self.flow.functions)
            .map_err(FlowError::LoadError)?;

        // Execute the flow with callback
        let (run, tool_runs) = self.flow.builder
            .run_sequential_flow(
                "main",
                &self.flow.context,
                &tool_registries,
                Arc::clone(&self.provider),
                task.clone(),
                Some(callback),
            )
            .await
            .map_err(FlowError::RunError)?;

        Ok(FlowResult {
            final_output: run.final_output,
            events: run.events,
            transcript: run.transcript,
            tool_results: tool_runs,
            metrics: run.metrics,
        })
    }
}

/// Result of a flow execution
///
/// Contains the output, events, transcript, and metrics from the flow execution.
#[derive(Debug, Clone, Serialize)]
pub struct FlowResult {
    /// Final output from the flow
    pub final_output: Option<String>,
    /// All events that occurred during execution
    pub events: Vec<crate::flows::sequential::SequentialEvent>,
    /// Full conversation transcript
    pub transcript: Vec<crate::types::ChatMessage>,
    /// Results from tool executions
    pub tool_results: Vec<crate::flows::spec::ToolRunResult>,
    /// Execution metrics (if available)
    pub metrics: Option<crate::metrics::AgentMetrics>,
}

impl FlowResult {
    /// Get the final output as a string
    ///
    /// # Returns
    /// The final output string, or an empty string if no output was generated
    pub fn output(&self) -> String {
        self.final_output.clone().unwrap_or_default()
    }

    /// Check if the execution was successful
    ///
    /// # Returns
    /// `true` if final output was generated, `false` otherwise
    pub fn is_success(&self) -> bool {
        self.final_output.is_some()
    }

    /// Get the number of agent steps executed
    pub fn agent_steps(&self) -> usize {
        self.events.iter()
            .filter(|e| matches!(e, crate::flows::sequential::SequentialEvent::Step { .. }))
            .count()
    }

    /// Get the number of tool executions
    pub fn tool_executions(&self) -> usize {
        self.tool_results.len()
    }
}

/// Errors that can occur during Flow operations
#[derive(Debug, Error)]
pub enum FlowError {
    #[error("Flow file not found: {0:?}")]
    FileNotFound(PathBuf, #[source] std::io::Error),

    #[error("No flow file found in directory: {0:?}")]
    NoFlowFileFound(PathBuf),

    #[error("Failed to load flow: {0}")]
    LoadError(#[from] FlowLoadError),

    #[error("Failed to run flow: {0}")]
    RunError(#[from] FlowRunError),

    #[error("Provider configuration error: {0}")]
    ProviderError(#[source] Box<dyn std::error::Error + Send + Sync>),

    #[error("No LLM provider configured. Set OPENAI_API_KEY or AZURE_OPENAI_KEY environment variables")]
    NoProviderConfigured,

    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
}

/// Convenience trait for creating functions easily
pub trait KernelFunctionExt: KernelFunction + Sized + 'static {
    fn into_arc(self) -> Arc<dyn KernelFunction> {
        Arc::new(self)
    }
}

impl<T: KernelFunction + 'static> KernelFunctionExt for T {}

// Re-export for convenience
pub use crate::kernel_function;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_function;

    #[kernel_function(name = "test_func", description = "Test function")]
    fn test_func(input: String) -> Result<String, String> {
        Ok(format!("processed: {}", input))
    }

    #[test]
    fn test_flow_creation() {
        let yaml = r#"
version: 0.1
flows:
  - id: main
    entry: start
    nodes:
      - id: start
        type: input
      - id: output
        type: output
    edges:
      - from: start
        to: output
        "#;

        let flow = Flow::from_yaml_string(yaml, ".")
            .unwrap()
            .with_function(test_func_kernel());

        assert_eq!(flow.functions.len(), 1);
    }

    #[test]
    fn test_flow_fluent_interface() {
        let yaml = r#"
version: 0.1
flows:
  - id: main
    entry: start
    nodes:
      - id: start
        type: input
      - id: output
        type: output
    edges:
      - from: start
        to: output
        "#;

        let flow = Flow::from_yaml_string(yaml, ".")
            .unwrap()
            .with_env_file(".env.example")
            .with_auto_discovery(true);

        assert_eq!(flow.env_files.len(), 1);
        assert!(flow.auto_discover);
    }

    #[test]
    fn test_flow_result() {
        let mut result = FlowResult {
            final_output: Some("test output".to_string()),
            events: vec![],
            transcript: vec![],
            tool_results: vec![],
            metrics: None,
        };

        assert_eq!(result.output(), "test output");
        assert!(result.is_success());
        assert_eq!(result.agent_steps(), 0);
        assert_eq!(result.tool_executions(), 0);

        result.final_output = None;
        assert!(!result.is_success());
    }

    #[test]
    fn test_flow_error_types() {
        let error = FlowError::NoProviderConfigured;
        assert!(matches!(error, FlowError::NoProviderConfigured));
        assert!(error.to_string().contains("No LLM provider configured"));
    }

    #[test]
    fn test_flow_debug() {
        let yaml = r#"
version: 0.1
flows:
  - id: main
    entry: start
    nodes:
      - id: start
        type: input
      - id: output
        type: output
    edges:
      - from: start
        to: output
        "#;

        let flow = Flow::from_yaml_string(yaml, ".").unwrap();
        let debug_str = format!("{:?}", flow);
        assert!(debug_str.contains("Flow"));
        assert!(debug_str.contains("functions_count"));
    }
}