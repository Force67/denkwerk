use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

use crate::TokenUsage;

/// Comprehensive metrics for agent execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetrics {
    /// Agent name
    pub agent_name: String,

    /// Execution metrics
    pub execution: ExecutionMetrics,

    /// Token usage metrics
    pub token_usage: TokenUsageMetrics,

    /// Function call metrics
    pub function_calls: FunctionCallMetrics,

    /// Error metrics
    pub errors: ErrorMetrics,

    /// Cost metrics (estimated)
    pub cost: CostMetrics,

    /// Timing information
    pub timestamp: DateTime<Utc>,
}

/// Execution-related metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    /// Total execution time
    pub total_duration: Duration,
    /// Number of rounds/turns
    pub rounds: usize,
    /// Success status
    pub succeeded: bool,
    /// Final output length
    pub output_length: usize,
}

/// Token usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsageMetrics {
    /// Input tokens used
    pub input_tokens: u32,
    /// Output tokens generated
    pub output_tokens: u32,
    /// Total tokens
    pub total_tokens: u32,
    /// Cost per token (estimated)
    pub cost_per_input_token: f64,
    pub cost_per_output_token: f64,
}

/// Function call metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallMetrics {
    /// Total function calls attempted
    pub total_calls: u32,
    /// Successful function calls
    pub successful_calls: u32,
    /// Failed function calls
    pub failed_calls: u32,
    /// List of called functions
    pub called_functions: Vec<String>,
    /// Average execution time per function call
    pub avg_function_duration: Option<Duration>,
}

/// Error tracking metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMetrics {
    /// Number of errors encountered
    pub error_count: u32,
    /// Types of errors encountered
    pub error_types: Vec<String>,
    /// Error messages (sanitized)
    pub error_messages: Vec<String>,
}

/// Cost estimation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostMetrics {
    /// Estimated total cost in USD
    pub estimated_cost_usd: f64,
    /// Cost breakdown by category
    pub cost_breakdown: HashMap<String, f64>,
    /// Cost per round
    pub cost_per_round: f64,
}

impl AgentMetrics {
    /// Create new metrics for an agent
    pub fn new(agent_name: String) -> Self {
        Self {
            agent_name,
            execution: ExecutionMetrics::default(),
            token_usage: TokenUsageMetrics::default(),
            function_calls: FunctionCallMetrics::default(),
            errors: ErrorMetrics::default(),
            cost: CostMetrics::default(),
            timestamp: Utc::now(),
        }
    }

    /// Record token usage
    pub fn record_token_usage(&mut self, usage: &TokenUsage, input_cost: f64, output_cost: f64) {
        self.token_usage.input_tokens += usage.prompt_tokens;
        self.token_usage.output_tokens += usage.completion_tokens;
        self.token_usage.total_tokens += usage.total_tokens;

        self.token_usage.cost_per_input_token = input_cost;
        self.token_usage.cost_per_output_token = output_cost;

        // Update cost metrics
        let token_cost = (usage.prompt_tokens as f64 * input_cost) +
                        (usage.completion_tokens as f64 * output_cost);
        self.cost.estimated_cost_usd += token_cost;
        self.cost.cost_breakdown.insert("tokens".to_string(), token_cost);
    }

    /// Record a function call
    pub fn record_function_call(&mut self, function_name: &str, duration: Duration, success: bool) {
        self.function_calls.total_calls += 1;

        if success {
            self.function_calls.successful_calls += 1;
        } else {
            self.function_calls.failed_calls += 1;
        }

        if !self.function_calls.called_functions.contains(&function_name.to_string()) {
            self.function_calls.called_functions.push(function_name.to_string());
        }

        // Update average duration
        let total_duration = if let Some(avg) = self.function_calls.avg_function_duration {
            avg * (self.function_calls.total_calls - 1) as u32 + duration
        } else {
            duration
        };
        self.function_calls.avg_function_duration = Some(
            total_duration / self.function_calls.total_calls as u32
        );

        // Update cost for function call (estimated)
        let function_cost = self.estimate_function_call_cost(function_name, duration);
        self.cost.estimated_cost_usd += function_cost;
        self.cost.cost_breakdown.insert(
            format!("functions:{}", function_name),
            function_cost
        );
    }

    /// Record an error
    pub fn record_error<E: std::error::Error + ?Sized>(&mut self, error: &E) {
        self.errors.error_count += 1;
        self.errors.error_types.push(std::any::type_name_of_val(error).to_string());

        // Sanitize error message for logging
        let error_msg = format!("{}", error);
        let sanitized_msg = if error_msg.len() > 200 {
            format!("{}...", &error_msg[..200])
        } else {
            error_msg
        };
        self.errors.error_messages.push(sanitized_msg);
    }

    /// Finalize metrics after execution
    pub fn finalize(&mut self, succeeded: bool, output_length: usize, rounds: usize) {
        self.execution.succeeded = succeeded;
        self.execution.output_length = output_length;
        self.execution.rounds = rounds;

        if rounds > 0 {
            self.cost.cost_per_round = self.cost.estimated_cost_usd / rounds as f64;
        }
    }

    /// Estimate function call cost (very rough approximation)
    fn estimate_function_call_cost(&self, function_name: &str, duration: Duration) -> f64 {
        // Base cost per function call (in USD)
        let base_cost = 0.0001;

        // Additional cost based on duration (per second)
        let duration_cost = duration.as_secs_f64() * 0.00001;

        // Additional cost based on function complexity (heuristic)
        let complexity_cost = match function_name {
            name if name.contains("search") || name.contains("query") => 0.0005,
            name if name.contains("generate") || name.contains("create") => 0.001,
            name if name.contains("analyze") || name.contains("process") => 0.002,
            _ => 0.0002,
        };

        base_cost + duration_cost + complexity_cost
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.function_calls.total_calls == 0 {
            1.0
        } else {
            self.function_calls.successful_calls as f64 / self.function_calls.total_calls as f64
        }
    }

    /// Get error rate
    pub fn error_rate(&self) -> f64 {
        if self.errors.error_count == 0 {
            0.0
        } else {
            self.errors.error_count as f64 / (self.errors.error_count + self.execution.rounds as u32) as f64
        }
    }
}

impl Default for ExecutionMetrics {
    fn default() -> Self {
        Self {
            total_duration: Duration::default(),
            rounds: 0,
            succeeded: false,
            output_length: 0,
        }
    }
}

impl Default for TokenUsageMetrics {
    fn default() -> Self {
        Self {
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            cost_per_input_token: 0.000001,  // $0.001 per 1K tokens (example rate)
            cost_per_output_token: 0.000002, // $0.002 per 1K tokens (example rate)
        }
    }
}

impl Default for FunctionCallMetrics {
    fn default() -> Self {
        Self {
            total_calls: 0,
            successful_calls: 0,
            failed_calls: 0,
            called_functions: Vec::new(),
            avg_function_duration: None,
        }
    }
}

impl Default for ErrorMetrics {
    fn default() -> Self {
        Self {
            error_count: 0,
            error_types: Vec::new(),
            error_messages: Vec::new(),
        }
    }
}

impl Default for CostMetrics {
    fn default() -> Self {
        Self {
            estimated_cost_usd: 0.0,
            cost_breakdown: HashMap::new(),
            cost_per_round: 0.0,
        }
    }
}

/// Trait for collecting and aggregating metrics
pub trait MetricsCollector: Send + Sync {
    /// Record metrics for an agent execution
    fn record_metrics(&self, metrics: AgentMetrics);

    /// Get aggregated metrics for all agents
    fn get_aggregated_metrics(&self) -> AggregatedMetrics;

    /// Get metrics for a specific agent
    fn get_agent_metrics(&self, agent_name: &str) -> Option<Vec<AgentMetrics>>;

    /// Clear all metrics
    fn clear_metrics(&self);
}

/// Aggregated metrics across multiple agent executions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedMetrics {
    /// Total number of executions
    pub total_executions: usize,

    /// Total cost across all executions
    pub total_cost_usd: f64,

    /// Total token usage
    pub total_tokens: u64,

    /// Average success rate
    pub average_success_rate: f64,

    /// Average execution time
    pub average_execution_time: Duration,

    /// Metrics by agent
    pub by_agent: HashMap<String, Vec<AgentMetrics>>,

    /// Cost breakdown
    pub cost_breakdown: HashMap<String, f64>,

    /// Time range of metrics
    pub time_range: (DateTime<Utc>, DateTime<Utc>),
}

/// In-memory metrics collector implementation
#[derive(Debug, Default)]
pub struct InMemoryMetricsCollector {
    metrics: Arc<std::sync::RwLock<Vec<AgentMetrics>>>,
}

impl InMemoryMetricsCollector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            metrics: Arc::new(std::sync::RwLock::new(Vec::with_capacity(capacity))),
        }
    }
}

impl MetricsCollector for InMemoryMetricsCollector {
    fn record_metrics(&self, metrics: AgentMetrics) {
        let mut metrics_lock = self.metrics.write().unwrap();
        metrics_lock.push(metrics);
    }

    fn get_aggregated_metrics(&self) -> AggregatedMetrics {
        let metrics_lock = self.metrics.read().unwrap();
        let metrics = &*metrics_lock;

        if metrics.is_empty() {
            return AggregatedMetrics {
                total_executions: 0,
                total_cost_usd: 0.0,
                total_tokens: 0,
                average_success_rate: 0.0,
                average_execution_time: Duration::default(),
                by_agent: HashMap::new(),
                cost_breakdown: HashMap::new(),
                time_range: (Utc::now(), Utc::now()),
            };
        }

        let mut by_agent: HashMap<String, Vec<AgentMetrics>> = HashMap::new();
        let mut total_cost = 0.0;
        let mut total_tokens = 0u64;
        let mut total_success_rate = 0.0;
        let mut total_execution_time = Duration::default();

        let mut min_time = metrics[0].timestamp;
        let mut max_time = metrics[0].timestamp;

        for metric in metrics {
            by_agent
                .entry(metric.agent_name.clone())
                .or_default()
                .push(metric.clone());

            total_cost += metric.cost.estimated_cost_usd;
            total_tokens += metric.token_usage.total_tokens as u64;
            total_success_rate += metric.success_rate();
            total_execution_time += metric.execution.total_duration;

            if metric.timestamp < min_time {
                min_time = metric.timestamp;
            }
            if metric.timestamp > max_time {
                max_time = metric.timestamp;
            }
        }

        let execution_count = metrics.len();
        let average_success_rate = total_success_rate / execution_count as f64;
        let average_execution_time = total_execution_time / execution_count as u32;

        // Aggregate cost breakdown
        let mut cost_breakdown = HashMap::new();
        for metric in metrics {
            for (category, cost) in &metric.cost.cost_breakdown {
                *cost_breakdown.entry(category.clone()).or_insert(0.0) += cost;
            }
        }

        AggregatedMetrics {
            total_executions: execution_count,
            total_cost_usd: total_cost,
            total_tokens,
            average_success_rate,
            average_execution_time,
            by_agent,
            cost_breakdown,
            time_range: (min_time, max_time),
        }
    }

    fn get_agent_metrics(&self, agent_name: &str) -> Option<Vec<AgentMetrics>> {
        let metrics_lock = self.metrics.read().unwrap();
        Some(
            metrics_lock
                .iter()
                .filter(|m| m.agent_name == agent_name)
                .cloned()
                .collect()
        )
    }

    fn clear_metrics(&self) {
        let mut metrics_lock = self.metrics.write().unwrap();
        metrics_lock.clear();
    }
}

/// Helper struct for timing agent execution
pub struct ExecutionTimer {
    start_time: Instant,
}

impl ExecutionTimer {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}

impl Default for ExecutionTimer {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility trait for adding metrics to orchestrators
pub trait WithMetrics {
    fn with_metrics_collector(self, collector: Arc<dyn MetricsCollector>) -> Self;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TokenUsage;

    #[test]
    fn test_agent_metrics_creation() {
        let metrics = AgentMetrics::new("test_agent".to_string());
        assert_eq!(metrics.agent_name, "test_agent");
        assert_eq!(metrics.token_usage.total_tokens, 0);
        assert_eq!(metrics.function_calls.total_calls, 0);
        assert_eq!(metrics.errors.error_count, 0);
    }

    #[test]
    fn test_token_usage_recording() {
        let mut metrics = AgentMetrics::new("test_agent".to_string());
        let usage = TokenUsage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
        };

        metrics.record_token_usage(&usage, 0.000001, 0.000002);

        assert_eq!(metrics.token_usage.input_tokens, 100);
        assert_eq!(metrics.token_usage.output_tokens, 50);
        assert_eq!(metrics.token_usage.total_tokens, 150);
        assert!(metrics.cost.estimated_cost_usd > 0.0);
    }

    #[test]
    fn test_function_call_recording() {
        let mut metrics = AgentMetrics::new("test_agent".to_string());
        let duration = Duration::from_millis(100);

        metrics.record_function_call("test_function", duration, true);

        assert_eq!(metrics.function_calls.total_calls, 1);
        assert_eq!(metrics.function_calls.successful_calls, 1);
        assert_eq!(metrics.function_calls.failed_calls, 0);
        assert!(metrics.function_calls.called_functions.contains(&"test_function".to_string()));
    }

    #[test]
    fn test_success_rate_calculation() {
        let mut metrics = AgentMetrics::new("test_agent".to_string());

        assert_eq!(metrics.success_rate(), 1.0); // No calls yet

        metrics.record_function_call("test1", Duration::default(), true);
        metrics.record_function_call("test2", Duration::default(), false);

        assert_eq!(metrics.success_rate(), 0.5);
    }

    #[test]
    fn test_in_memory_metrics_collector() {
        let collector = InMemoryMetricsCollector::new();
        let mut metrics = AgentMetrics::new("test_agent".to_string());
        metrics.finalize(true, 100, 3);

        collector.record_metrics(metrics);

        let aggregated = collector.get_aggregated_metrics();
        assert_eq!(aggregated.total_executions, 1);
        assert_eq!(aggregated.by_agent.get("test_agent").unwrap().len(), 1);
    }
}
