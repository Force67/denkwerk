use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::fs;
#[cfg(test)]
use serde_json::Value;
use meval;

use super::sequential::SequentialOrchestrator;
use crate::{agents::Agent, functions::FunctionRegistry, LLMProvider};

fn default_version() -> String {
    "0.1".to_string()
}

#[derive(Debug, Error)]
pub enum FlowSchemaError {
    #[error("failed to parse flow YAML: {0}")]
    Parse(#[from] serde_yaml::Error),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FlowDocument {
    #[serde(default = "default_version")]
    pub version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<FlowMetadata>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub agents: Vec<AgentDefinition>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prompts: Vec<PromptDefinition>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub flows: Vec<FlowDefinition>,
}

impl FlowDocument {
    pub fn from_yaml_str(input: &str) -> Result<Self, FlowSchemaError> {
        Ok(serde_yaml::from_str(input)?)
    }

    pub fn to_yaml_string(&self) -> Result<String, FlowSchemaError> {
        Ok(serde_yaml::to_string(self)?)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FlowMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AgentDefinition {
    pub id: String,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub defaults: Option<CallSettings>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ToolDefinition {
    pub id: String,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spec: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PromptDefinition {
    pub id: String,
    pub file: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FlowDefinition {
    pub id: String,
    pub entry: String,
    pub nodes: Vec<FlowNode>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub edges: Vec<FlowEdge>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FlowEdge {
    pub from: String,
    pub to: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FlowNode {
    #[serde(flatten)]
    pub base: NodeBase,
    #[serde(flatten)]
    pub kind: FlowNodeKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct NodeBase {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<NodeInput>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<NodeOutput>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout: Option<NodeLayout>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FlowNodeKind {
    Input {},
    Output {},
    Agent {
        agent: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prompt: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tools: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parameters: Option<CallSettings>,
    },
    Decision {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prompt: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        strategy: Option<DecisionStrategy>,
    },
    Tool {
        tool: String,
    },
    Merge {},
    Parallel {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        converge: Option<bool>,
    },
    Loop {
        #[serde(default)]
        max_iterations: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        condition: Option<String>,
    },
    Subflow {
        flow: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DecisionStrategy {
    Llm,
    Rule,
}

impl std::fmt::Display for DecisionStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecisionStrategy::Llm => write!(f, "llm"),
            DecisionStrategy::Rule => write!(f, "rule"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct NodeInput {
    pub from: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct NodeOutput {
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct NodeLayout {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CallSettings {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry: Option<RetryPolicy>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RetryPolicy {
    pub max: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backoff_ms: Option<u64>,
}

#[derive(Debug, Error)]
pub enum FlowLoadError {
    #[error(transparent)]
    Schema(#[from] FlowSchemaError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("flow not found: {0}")]
    FlowNotFound(String),
    #[error("agent not found: {0}")]
    AgentNotFound(String),
    #[error("tool not found: {0}")]
    ToolNotFound(String),
    #[error("function not found for tool {0}: {1}")]
    FunctionNotFound(String, String),
    #[error("node not found: {0}")]
    NodeNotFound(String),
    #[error("flow is not sequential near node: {0}")]
    NonSequential(String),
    #[error("flow {0} does not terminate in an output node")]
    MissingOutput(String),
    #[error("unsupported node type {0} in sequential planner")]
    UnsupportedNode(String),
    #[error("no matching edge from node {0}")]
    NoMatchingEdge(String),
    #[error("subflow recursion detected at {0}")]
    SubflowCycle(String),
}

pub struct FlowBuilder {
    document: FlowDocument,
    base_dir: PathBuf,
}

impl FlowBuilder {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, FlowLoadError> {
        let path = path.as_ref();
        let content = fs::read_to_string(path)?;
        Self::from_yaml_str(path.parent().unwrap_or_else(|| Path::new(".")), &content)
    }

    pub fn from_yaml_str(base_dir: impl AsRef<Path>, yaml: &str) -> Result<Self, FlowLoadError> {
        Ok(Self {
            document: FlowDocument::from_yaml_str(yaml)?,
            base_dir: base_dir.as_ref().to_path_buf(),
        })
    }

    pub fn document(&self) -> &FlowDocument {
        &self.document
    }

    pub fn build_agents(
        &self,
        tool_registries: &HashMap<String, Arc<FunctionRegistry>>,
    ) -> Result<HashMap<String, Agent>, FlowLoadError> {
        let mut agents = HashMap::new();

        for def in &self.document.agents {
            let mut agent = Agent::from_string(def.id.clone(), load_instructions(&self.base_dir, def.system_prompt.as_deref())?);

            if let Some(desc) = &def.description {
                agent = agent.with_description(desc.clone());
            }

            // Global defaults applied first
            agent = apply_call_settings(agent, Some(&CallSettings {
                model: Some(def.model.clone()),
                temperature: None,
                top_p: None,
                max_tokens: None,
                timeout_ms: None,
                retry: None,
            }));

            if let Some(defaults) = &def.defaults {
                agent = apply_call_settings(agent, Some(defaults));
            }

            if !def.tools.is_empty() {
                let mut combined = FunctionRegistry::new();
                let mut any = false;
                for tool_id in &def.tools {
                    if let Some(registry) = tool_registries.get(tool_id) {
                        combined.extend_from(registry);
                        any = true;
                    }
                }
                if any {
                    agent = agent.with_function_registry(Arc::new(combined));
                }
            }

            agents.insert(def.id.clone(), agent);
        }

        Ok(agents)
    }

    pub fn build_sequential_orchestrator(
        &self,
        provider: Arc<dyn LLMProvider>,
        flow_id: &str,
        tool_registries: &HashMap<String, Arc<FunctionRegistry>>,
    ) -> Result<SequentialOrchestrator, FlowLoadError> {
        let agents = self.build_agents(tool_registries)?;
        let mut visited = Vec::new();
        let planned = self.plan_nodes(flow_id, &FlowContext::default(), &mut visited)?;

        if planned.is_empty() {
            return Err(FlowLoadError::MissingOutput(flow_id.to_string()));
        }

        let pipeline = planned
            .iter()
            .map(|step| {
                agents
                    .get(&step.id)
                    .cloned()
                    .map(|agent| apply_call_settings(agent, step.params.as_ref()))
                    .ok_or_else(|| FlowLoadError::AgentNotFound(step.id.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let model_lookup: HashMap<String, String> = self
            .document
            .agents
            .iter()
            .map(|a| (a.id.clone(), a.model.clone()))
            .collect();

        let model = model_lookup
            .get(&planned[0].id)
            .cloned()
            .unwrap_or_else(|| "gpt-4o".to_string());

        Ok(SequentialOrchestrator::new(provider, model).with_agents(pipeline))
    }

    pub fn build_concurrent_orchestrator(
        &self,
        provider: Arc<dyn LLMProvider>,
        flow_id: &str,
        tool_registries: &HashMap<String, Arc<FunctionRegistry>>,
    ) -> Result<crate::flows::concurrent::ConcurrentOrchestrator, FlowLoadError> {
        let agents = self.build_agents(tool_registries)?;
        let roster = self.flow_agents(flow_id)?;
        let model = self
            .document
            .agents
            .iter()
            .find(|a| roster.contains(&a.id))
            .map(|a| a.model.clone())
            .unwrap_or_else(|| "gpt-4o".to_string());

        let pipeline = roster
            .into_iter()
            .filter_map(|id| agents.get(&id).cloned())
            .collect::<Vec<_>>();

        Ok(crate::flows::concurrent::ConcurrentOrchestrator::new(provider, model).with_agents(pipeline))
    }

    pub fn build_group_chat_orchestrator(
        &self,
        provider: Arc<dyn LLMProvider>,
        flow_id: &str,
        tool_registries: &HashMap<String, Arc<FunctionRegistry>>,
    ) -> Result<crate::flows::group_chat::GroupChatOrchestrator<crate::flows::group_chat::RoundRobinGroupChatManager>, FlowLoadError> {
        let agents = self.build_agents(tool_registries)?;
        let roster = self.flow_agents(flow_id)?;
        let model = self
            .document
            .agents
            .iter()
            .find(|a| roster.contains(&a.id))
            .map(|a| a.model.clone())
            .unwrap_or_else(|| "gpt-4o".to_string());

        let pipeline = roster
            .into_iter()
            .filter_map(|id| agents.get(&id).cloned())
            .collect::<Vec<_>>();

        Ok(crate::flows::group_chat::GroupChatOrchestrator::new(
            provider,
            model,
            crate::flows::group_chat::RoundRobinGroupChatManager::new(),
        )
        .with_agents(pipeline))
    }

    pub fn build_handoff_orchestrator(
        &self,
        provider: Arc<dyn LLMProvider>,
        flow_id: &str,
        tool_registries: &HashMap<String, Arc<FunctionRegistry>>,
    ) -> Result<crate::flows::handoffflow::HandoffOrchestrator, FlowLoadError> {
        let agents = self.build_agents(tool_registries)?;
        let roster = self.flow_agents(flow_id)?;
        let model = self
            .document
            .agents
            .iter()
            .find(|a| roster.contains(&a.id))
            .map(|a| a.model.clone())
            .unwrap_or_else(|| "gpt-4o".to_string());

        let mut orchestrator = crate::flows::handoffflow::HandoffOrchestrator::new(provider, model);
        for id in roster {
            if let Some(agent) = agents.get(&id) {
                orchestrator.register_agent(agent.clone());
            }
        }
        Ok(orchestrator)
    }

    pub fn build_tool_registries(
        &self,
        functions: &HashMap<String, Arc<dyn crate::functions::KernelFunction>>,
    ) -> Result<HashMap<String, Arc<FunctionRegistry>>, FlowLoadError> {
        let mut registries = HashMap::new();

        for tool in &self.document.tools {
            let mut registry = FunctionRegistry::new();
            if let Some(function_name) = &tool.function {
                let func = functions
                    .get(function_name)
                    .ok_or_else(|| FlowLoadError::FunctionNotFound(tool.id.clone(), function_name.clone()))?;
                registry.register(func.clone());
            } else {
                return Err(FlowLoadError::ToolNotFound(tool.id.clone()));
            }
            registries.insert(tool.id.clone(), Arc::new(registry));
        }

        Ok(registries)
    }

    pub fn plan_sequential_path(
        &self,
        flow_id: &str,
        ctx: &FlowContext,
        tool_registries: &HashMap<String, Arc<FunctionRegistry>>,
    ) -> Result<Vec<Agent>, FlowLoadError> {
        let agents = self.build_agents(tool_registries)?;
        let mut visited_flows = Vec::new();
        let path = self.plan_nodes(flow_id, ctx, &mut visited_flows)?;
        let mut result = Vec::new();
        for step in path {
            let agent = agents
                .get(&step.id)
                .cloned()
                .map(|a| apply_call_settings(a, step.params.as_ref()))
                .ok_or_else(|| FlowLoadError::AgentNotFound(step.id.clone()))?;
            result.push(agent);
        }
        Ok(result)
    }

    fn plan_nodes(
        &self,
        flow_id: &str,
        ctx: &FlowContext,
        visited_flows: &mut Vec<String>,
    ) -> Result<Vec<PlannedAgent>, FlowLoadError> {
        if visited_flows.contains(&flow_id.to_string()) {
            return Err(FlowLoadError::SubflowCycle(flow_id.to_string()));
        }
        visited_flows.push(flow_id.to_string());

        let flow = self
            .document
            .flows
            .iter()
            .find(|f| f.id == flow_id)
            .ok_or_else(|| FlowLoadError::FlowNotFound(flow_id.to_string()))?;

        let mut current = flow.entry.clone();
        let mut path = Vec::new();
        let mut loop_counters: HashMap<String, u32> = HashMap::new();

        loop {
            let node = flow
                .nodes
                .iter()
                .find(|n| n.base.id == current)
                .ok_or_else(|| FlowLoadError::NodeNotFound(current.clone()))?;

            match &node.kind {
                FlowNodeKind::Input {} => {}
                FlowNodeKind::Agent { agent, parameters, .. } => {
                    let id = agent.clone();
                    path.push(PlannedAgent {
                        id,
                        params: parameters.clone(),
                    });
                }
                FlowNodeKind::Decision { .. } => {}
                FlowNodeKind::Tool { .. } => {}
                FlowNodeKind::Merge {} => {}
                FlowNodeKind::Parallel { .. } => {
                    return Err(FlowLoadError::UnsupportedNode(node.base.id.clone()));
                }
                FlowNodeKind::Loop { .. } => {}
                FlowNodeKind::Subflow { flow } => {
                    let mut nested = self.plan_nodes(flow, ctx, visited_flows)?;
                    path.append(&mut nested);
                }
                FlowNodeKind::Output {} => break,
            }

            let outgoing: Vec<&FlowEdge> = flow
                .edges
                .iter()
                .filter(|e| edge_base(&e.from) == node.base.id)
                .collect();

            if outgoing.is_empty() {
                break;
            }

            let mut selected: Option<String> = None;
            let iteration_value = *loop_counters.get(&node.base.id).unwrap_or(&0);
            for edge in &outgoing {
                if condition_matches(edge.condition.as_deref(), ctx, Some(iteration_value)) {
                    selected = Some(edge.to.clone());
                    break;
                }
            }

            if selected.is_none() {
                // Try a default if conditions are on outputs
                for edge in &outgoing {
                    if edge.condition.is_none() {
                        selected = Some(edge.to.clone());
                        break;
                    }
                }
            }

            let next = selected.ok_or_else(|| FlowLoadError::NoMatchingEdge(node.base.id.clone()))?;

            if let FlowNodeKind::Loop { .. } = &node.kind {
                let counter = loop_counters.entry(node.base.id.clone()).or_insert(0);
                *counter += 1;
            }

            current = next;
        }

        visited_flows.retain(|f| f != flow_id);
        Ok(path)
    }

    fn flow_agents(&self, flow_id: &str) -> Result<Vec<String>, FlowLoadError> {
        let flow = self
            .document
            .flows
            .iter()
            .find(|f| f.id == flow_id)
            .ok_or_else(|| FlowLoadError::FlowNotFound(flow_id.to_string()))?;

        let mut ids = Vec::new();
        for node in &flow.nodes {
            if let FlowNodeKind::Agent { agent, .. } = &node.kind {
                ids.push(agent.clone());
            }
        }
        Ok(ids)
    }
}

#[derive(Debug, Default, Clone)]
pub struct FlowContext {
    pub vars: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone)]
struct PlannedAgent {
    id: String,
    params: Option<CallSettings>,
}

impl FlowContext {
    pub fn with_var(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.vars.insert(key.into(), value.into());
        self
    }

}

fn condition_matches(condition: Option<&str>, ctx: &FlowContext, iteration: Option<u32>) -> bool {
    match condition {
        None => true,
        Some(text) if text.trim().eq_ignore_ascii_case("else") => true,
        Some(text) => {
            if let Some(parsed) = parse_binary_condition(text, ctx, iteration) {
                return parsed;
            }

            let mut vars = meval::Context::new();
            if let Some(iter) = iteration {
                vars.var("iteration", iter as f64);
            }
            for (k, v) in &ctx.vars {
                if let Some(num) = v.as_f64() {
                    vars.var(k, num);
                }
            }
            meval::eval_str_with_context(text, &vars).map(|v| v != 0.0).unwrap_or(false)
        }
    }
}

fn parse_binary_condition(text: &str, ctx: &FlowContext, iteration: Option<u32>) -> Option<bool> {
    for op in ["<=", ">=", "==", "!=", "<", ">"] {
        if let Some((left, right)) = text.split_once(op) {
            let l = left.trim();
            let r = right.trim();
            match op {
                "==" | "!=" => {
                    let lval = extract_string_value(l, ctx, iteration)?;
                    let rval = extract_string_value(r, ctx, iteration)?;
                    return Some(match op {
                        "==" => lval == rval,
                        "!=" => lval != rval,
                        _ => false,
                    });
                }
                "<" | "<=" | ">" | ">=" => {
                    let lnum = extract_number_value(l, ctx, iteration)?;
                    let rnum = extract_number_value(r, ctx, iteration)?;
                    return Some(match op {
                        "<" => lnum < rnum,
                        "<=" => lnum <= rnum,
                        ">" => lnum > rnum,
                        ">=" => lnum >= rnum,
                        _ => false,
                    });
                }
                _ => {}
            }
        }
    }
    None
}

fn extract_number_value(token: &str, ctx: &FlowContext, iteration: Option<u32>) -> Option<f64> {
    if let Ok(n) = token.parse::<f64>() {
        return Some(n);
    }
    if let Some(iter) = iteration {
        if token == "iteration" {
            return Some(iter as f64);
        }
    }
    if let Some(value) = ctx.vars.get(token) {
        if let Some(num) = value.as_f64() {
            return Some(num);
        }
        if let Some(s) = value.as_str() {
            if let Ok(n) = s.parse::<f64>() {
                return Some(n);
            }
        }
    }
    None
}

fn extract_string_value(token: &str, ctx: &FlowContext, iteration: Option<u32>) -> Option<String> {
    let stripped = token.trim_matches('\'').trim_matches('"');
    if stripped != token {
        return Some(stripped.to_string());
    }
    if let Some(iter) = iteration {
        if token == "iteration" {
            return Some(iter.to_string());
        }
    }
    if let Some(value) = ctx.vars.get(token) {
        if let Some(s) = value.as_str() {
            return Some(s.to_string());
        }
        if let Some(n) = value.as_f64() {
            return Some(n.to_string());
        }
    }
    Some(token.to_string())
}

fn edge_base(edge: &str) -> String {
    edge.split(':').next().unwrap_or(edge).to_string()
}

fn apply_call_settings(agent: Agent, settings: Option<&CallSettings>) -> Agent {
    let mut agent = agent;
    if let Some(settings) = settings {
        if let Some(model) = &settings.model {
            agent = agent.with_model(model.clone());
        }
        if let Some(temp) = settings.temperature {
            agent = agent.with_temperature(temp);
        }
        if let Some(top_p) = settings.top_p {
            agent = agent.with_top_p(top_p);
        }
        if let Some(max_tokens) = settings.max_tokens {
            agent = agent.with_max_tokens(max_tokens);
        }
    }
    agent
}

fn load_instructions(base_dir: &Path, prompt: Option<&str>) -> Result<String, FlowLoadError> {
    match prompt {
        Some(content_or_path) => {
            let candidate = base_dir.join(content_or_path);
            if candidate.exists() {
                Ok(fs::read_to_string(candidate)?)
            } else {
                Ok(content_or_path.to_string())
            }
        }
        None => Ok(String::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::scripted::ScriptedProvider;
    use crate::eval::scenario::ScriptedTurn;
    use std::time::{SystemTime, UNIX_EPOCH};
    use crate::functions::{KernelFunction, FunctionDefinition, FunctionParameter, json_schema_for};

    fn temp_dir() -> PathBuf {
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("flow_builder_{ts}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn parses_minimal_flow_and_applies_defaults() {
        let yaml = r#"
flows:
  - id: main
    entry: start
    nodes:
      - id: start
        type: input
"#;

        let doc = FlowDocument::from_yaml_str(yaml).expect("minimal flow should parse");

        assert_eq!(doc.version, "0.1");
        assert!(doc.metadata.is_none());
        assert!(doc.agents.is_empty());
        assert!(doc.tools.is_empty());
        assert!(doc.prompts.is_empty());
        assert_eq!(doc.flows.len(), 1);

        let flow = &doc.flows[0];
        assert_eq!(flow.id, "main");
        assert_eq!(flow.entry, "start");
        assert!(flow.edges.is_empty());
        assert_eq!(flow.nodes.len(), 1);
        let node = &flow.nodes[0];
        assert_eq!(node.base.id, "start");
        assert!(matches!(node.kind, FlowNodeKind::Input {}));
    }

    #[test]
    fn parses_complete_document_sections() {
        let yaml = r#"
version: "1.2"
metadata:
  name: Demo Flow
  description: End-to-end flow
  tags: [demo, test]
agents:
  - id: analyst
    model: gpt-4o
    name: Analyst
    description: Extract insights
    system_prompt: Analyze carefully
    tools: [search, calculator]
    defaults:
      model: gpt-4o-mini
      temperature: 0.7
      top_p: 0.9
      max_tokens: 128
      timeout_ms: 5000
      retry:
        max: 3
        backoff_ms: 250
tools:
  - id: search
    kind: http
    description: Search tool
    spec: specs/search.yaml
    function: search
prompts:
  - id: analysis_prompt
    file: prompts/analysis.txt
    description: Main analysis prompt
flows:
  - id: parent
    entry: decide
    nodes:
      - id: decide
        type: decision
        prompt: analysis_prompt
        strategy: rule
"#;

        let doc = FlowDocument::from_yaml_str(yaml).expect("complete flow should parse");

        assert_eq!(doc.version, "1.2");
        assert_eq!(
            doc.metadata,
            Some(FlowMetadata {
                name: Some("Demo Flow".to_string()),
                description: Some("End-to-end flow".to_string()),
                tags: vec!["demo".to_string(), "test".to_string()],
            })
        );
        assert_eq!(
            doc.agents,
            vec![AgentDefinition {
                id: "analyst".to_string(),
                model: "gpt-4o".to_string(),
                name: Some("Analyst".to_string()),
                description: Some("Extract insights".to_string()),
                system_prompt: Some("Analyze carefully".to_string()),
                tools: vec!["search".to_string(), "calculator".to_string()],
                defaults: Some(CallSettings {
                    model: Some("gpt-4o-mini".to_string()),
                    temperature: Some(0.7),
                    top_p: Some(0.9),
                    max_tokens: Some(128),
                    timeout_ms: Some(5000),
                    retry: Some(RetryPolicy {
                        max: 3,
                        backoff_ms: Some(250),
                    }),
                }),
            }]
        );
        assert_eq!(
            doc.tools,
            vec![ToolDefinition {
                id: "search".to_string(),
                kind: "http".to_string(),
                description: Some("Search tool".to_string()),
                spec: Some("specs/search.yaml".to_string()),
                function: Some("search".to_string()),
            }]
        );
        assert_eq!(
            doc.prompts,
            vec![PromptDefinition {
                id: "analysis_prompt".to_string(),
                file: "prompts/analysis.txt".to_string(),
                description: Some("Main analysis prompt".to_string()),
            }]
        );
        assert_eq!(doc.flows.len(), 1);
        assert!(matches!(
            doc.flows[0].nodes[0].kind,
            FlowNodeKind::Decision { strategy: Some(DecisionStrategy::Rule), .. }
        ));
    }

    #[test]
    fn parses_all_node_kinds_and_edges() {
        let yaml = r#"
agents:
  - id: analyst
    model: gpt-4o
prompts:
  - id: agent_prompt
    file: prompts/agent.txt
  - id: route_prompt
    file: prompts/route.txt
flows:
  - id: main
    entry: input
    nodes:
      - id: input
        name: Start
        type: input
        outputs:
          - label: to_decision
      - id: agent
        type: agent
        agent: analyst
        prompt: agent_prompt
        tools: [search, math]
        parameters:
          model: custom-model
          temperature: 0.6
          top_p: 0.8
          max_tokens: 32
          retry:
            max: 2
      - id: decision
        type: decision
        prompt: route_prompt
        strategy: llm
      - id: tool
        type: tool
        tool: search
      - id: merge
        type: merge
      - id: loop
        type: loop
        max_iterations: 3
        condition: "x < 10"
      - id: parallel
        type: parallel
        converge: false
      - id: subflow
        type: subflow
        flow: child_flow
      - id: output
        type: output
    edges:
      - from: input:to_decision
        to: decision
      - from: decision
        to: tool
        label: use_tool
      - from: decision
        to: agent
        condition: "route == 'agent'"
      - from: tool
        to: merge
      - from: agent
        to: merge
      - from: merge
        to: loop
      - from: loop
        to: parallel
      - from: parallel
        to: subflow
      - from: subflow
        to: output
"#;

        let doc = FlowDocument::from_yaml_str(yaml).expect("all node kinds should parse");
        let flow = &doc.flows[0];

        assert_eq!(flow.nodes.len(), 9);
        assert!(matches!(flow.nodes[0].kind, FlowNodeKind::Input {}));
        assert!(matches!(flow.nodes[8].kind, FlowNodeKind::Output {}));

        if let FlowNodeKind::Agent {
            agent,
            prompt,
            tools,
            parameters,
        } = &flow.nodes[1].kind
        {
            assert_eq!(agent, "analyst");
            assert_eq!(prompt.as_deref(), Some("agent_prompt"));
            assert_eq!(tools, &["search".to_string(), "math".to_string()]);
            let params = parameters.as_ref().expect("agent parameters");
            assert_eq!(params.model.as_deref(), Some("custom-model"));
            assert_eq!(params.temperature, Some(0.6));
            assert_eq!(params.top_p, Some(0.8));
            assert_eq!(params.max_tokens, Some(32));
            assert_eq!(params.retry.as_ref().map(|r| r.max), Some(2));
        } else {
            panic!("expected agent node");
        }

        if let FlowNodeKind::Decision { strategy, prompt } = &flow.nodes[2].kind {
            assert_eq!(prompt.as_deref(), Some("route_prompt"));
            assert!(matches!(strategy, Some(DecisionStrategy::Llm)));
        } else {
            panic!("expected decision node");
        }

        assert!(matches!(&flow.nodes[3].kind, FlowNodeKind::Tool { tool } if tool == "search"));
        assert!(matches!(flow.nodes[4].kind, FlowNodeKind::Merge {}));
        assert!(matches!(flow.nodes[5].kind, FlowNodeKind::Loop { max_iterations: 3, condition: Some(_)}));
        assert!(matches!(flow.nodes[6].kind, FlowNodeKind::Parallel { converge: Some(false) }));
        assert!(matches!(&flow.nodes[7].kind, FlowNodeKind::Subflow { flow } if flow == "child_flow"));

        assert_eq!(flow.edges.len(), 9);
        assert_eq!(flow.edges[1].label.as_deref(), Some("use_tool"));
        assert_eq!(flow.edges[2].condition.as_deref(), Some("route == 'agent'"));
    }

    #[test]
    fn roundtrips_through_yaml_serialization() {
        let document = FlowDocument {
            version: "0.2".to_string(),
            metadata: Some(FlowMetadata {
                name: Some("Roundtrip Flow".to_string()),
                description: Some("Ensures serialization survives roundtrip".to_string()),
                tags: vec!["roundtrip".to_string()],
            }),
            agents: vec![AgentDefinition {
                id: "researcher".to_string(),
                model: "gpt-4o".to_string(),
                name: Some("Researcher".to_string()),
                description: None,
                system_prompt: Some("Be concise".to_string()),
                tools: vec!["browser".to_string()],
                defaults: Some(CallSettings {
                    model: Some("gpt-4o-mini".to_string()),
                    temperature: Some(0.2),
                    top_p: None,
                    max_tokens: Some(64),
                    timeout_ms: Some(2000),
                    retry: Some(RetryPolicy {
                        max: 2,
                        backoff_ms: Some(100),
                    }),
                }),
            }],
            tools: vec![ToolDefinition {
                id: "browser".to_string(),
                kind: "http".to_string(),
                description: Some("Browse the web".to_string()),
                spec: Some("specs/browser.yaml".to_string()),
                function: Some("browse".to_string()),
            }],
            prompts: vec![PromptDefinition {
                id: "research_prompt".to_string(),
                file: "prompts/research.txt".to_string(),
                description: None,
            }],
            flows: vec![FlowDefinition {
                id: "parent".to_string(),
                entry: "start".to_string(),
                nodes: vec![
                    FlowNode {
                        base: NodeBase {
                            id: "start".to_string(),
                            name: Some("Start".to_string()),
                            description: Some("Entry point".to_string()),
                            inputs: vec![],
                            outputs: vec![NodeOutput {
                                label: "next".to_string(),
                                condition: None,
                            }],
                            layout: Some(NodeLayout { x: 0.0, y: 0.0 }),
                        },
                        kind: FlowNodeKind::Input {},
                    },
                    FlowNode {
                        base: NodeBase {
                            id: "agent".to_string(),
                            name: None,
                            description: None,
                            inputs: vec![NodeInput {
                                from: "start:next".to_string(),
                            }],
                            outputs: vec![],
                            layout: None,
                        },
                        kind: FlowNodeKind::Agent {
                            agent: "researcher".to_string(),
                            prompt: Some("research_prompt".to_string()),
                            tools: vec!["browser".to_string()],
                            parameters: Some(CallSettings {
                                model: Some("gpt-4o-mini".to_string()),
                                temperature: Some(0.1),
                                top_p: Some(0.95),
                                max_tokens: Some(32),
                                timeout_ms: None,
                                retry: Some(RetryPolicy {
                                    max: 1,
                                    backoff_ms: None,
                                }),
                            }),
                        },
                    },
                    FlowNode {
                        base: NodeBase {
                            id: "end".to_string(),
                            name: None,
                            description: None,
                            inputs: vec![NodeInput {
                                from: "agent".to_string(),
                            }],
                            outputs: vec![],
                            layout: None,
                        },
                        kind: FlowNodeKind::Output {},
                    },
                ],
                edges: vec![
                    FlowEdge {
                        from: "start:next".to_string(),
                        to: "agent".to_string(),
                        condition: None,
                        label: Some("handover".to_string()),
                    },
                    FlowEdge {
                        from: "agent".to_string(),
                        to: "end".to_string(),
                        condition: Some("done".to_string()),
                        label: None,
                    },
                ],
            }],
        };

        let yaml = document
            .to_yaml_string()
            .expect("serializing to yaml should work");
        let parsed = FlowDocument::from_yaml_str(&yaml).expect("roundtrip parse should work");

        assert_eq!(parsed, document);
    }

    #[test]
    fn returns_error_on_invalid_yaml() {
        let yaml = "flows:\n  - id: bad\n    entry: [unbalanced";
        let err = FlowDocument::from_yaml_str(yaml).expect_err("invalid yaml should fail");
        match err {
            FlowSchemaError::Parse(_) => {}
        }
    }

    #[test]
    fn builds_agents_and_loads_prompts_from_files() {
        let dir = temp_dir();
        let prompt_path = dir.join("prompt.txt");
        fs::write(&prompt_path, "hello from file").unwrap();

        let yaml = format!(
            r#"
agents:
  - id: a1
    model: m1
    system_prompt: {}
  - id: a2
    model: m2
    system_prompt: inline prompt
flows:
  - id: main
    entry: n1
    nodes:
      - id: n1
        type: input
      - id: n2
        type: output
"#,
            prompt_path.file_name().unwrap().to_string_lossy()
        );

        let builder = FlowBuilder::from_yaml_str(&dir, &yaml).expect("builder");
        let agents = builder.build_agents(&HashMap::new()).expect("agents");

        assert_eq!(agents.len(), 2);
        assert_eq!(agents.get("a1").unwrap().instructions(), "hello from file");
        assert_eq!(agents.get("a2").unwrap().instructions(), "inline prompt");
    }

    #[tokio::test]
    async fn builds_sequential_orchestrator_for_linear_flow() {
        let yaml = r#"
agents:
  - id: first
    model: scripted
    system_prompt: first agent
  - id: second
    model: scripted
    system_prompt: second agent
flows:
  - id: main
    entry: start
    nodes:
      - id: start
        type: input
      - id: a1
        type: agent
        agent: first
      - id: a2
        type: agent
        agent: second
      - id: end
        type: output
    edges:
      - from: start
        to: a1
      - from: a1
        to: a2
      - from: a2
        to: end
"#;

        let builder = FlowBuilder::from_yaml_str(".", yaml).expect("builder");

        let provider = Arc::new(ScriptedProvider::from_scripted_turns(&[
            ScriptedTurn { agent: "first".to_string(), response: "hello".to_string(), latency_ms: None },
            ScriptedTurn { agent: "second".to_string(), response: "done".to_string(), latency_ms: None },
        ]));

        let orchestrator = builder
            .build_sequential_orchestrator(provider, "main", &HashMap::new())
            .expect("orchestrator");

        let run = orchestrator.run("task").await.expect("run");
        assert_eq!(run.events.len(), 2);
        assert_eq!(run.final_output.as_deref(), Some("done"));
    }

    #[test]
    fn rejects_non_sequential_flow() {
        let yaml = r#"
agents:
  - id: router
    model: m
    system_prompt: decide
flows:
  - id: branching
    entry: input
    nodes:
      - id: input
        type: input
      - id: decide
        type: decision
      - id: out
        type: output
    edges:
      - from: input
        to: decide
      - from: decide
        to: out
      - from: decide
        to: out
"#;

        let builder = FlowBuilder::from_yaml_str(".", yaml).expect("builder");
        let result = builder.build_sequential_orchestrator(
            Arc::new(ScriptedProvider::new()),
            "branching",
            &HashMap::new(),
        );
        assert!(matches!(result, Err(FlowLoadError::MissingOutput(_))));
    }

    #[test]
    fn builds_tool_registries_from_definitions() {
        struct EchoFn;
        #[async_trait::async_trait]
        impl KernelFunction for EchoFn {
            fn definition(&self) -> FunctionDefinition {
                let mut def = FunctionDefinition::new("echo");
                def.add_parameter(FunctionParameter::new("text", json_schema_for::<String>()));
                def
            }

            async fn invoke(&self, arguments: &Value) -> Result<Value, crate::LLMError> {
                Ok(arguments.get("text").cloned().unwrap_or(Value::Null))
            }
        }

        let yaml = r#"
tools:
  - id: echo_tool
    kind: function
    function: echo
flows:
  - id: main
    entry: n1
    nodes:
      - id: n1
        type: input
      - id: n2
        type: output
"#;

        let mut functions = HashMap::new();
        functions.insert("echo".to_string(), Arc::new(EchoFn) as Arc<dyn KernelFunction>);

        let builder = FlowBuilder::from_yaml_str(".", yaml).expect("builder");
        let registries = builder.build_tool_registries(&functions).expect("registries");

        let reg = registries.get("echo_tool").expect("echo registry");
        assert_eq!(reg.definitions().len(), 1);
        assert_eq!(reg.definitions()[0].name, "echo");
    }

    #[test]
    fn plans_decision_branch_with_context() {
        let yaml = r#"
agents:
  - id: a1
    model: m
    system_prompt: p1
  - id: a2
    model: m
    system_prompt: p2
flows:
  - id: main
    entry: start
    nodes:
      - id: start
        type: input
      - id: decide
        type: decision
      - id: agent1
        type: agent
        agent: a1
      - id: agent2
        type: agent
        agent: a2
      - id: end
        type: output
    edges:
      - from: start
        to: decide
      - from: decide
        to: agent1
        condition: "route == 'a1'"
      - from: decide
        to: agent2
        condition: "else"
      - from: agent1
        to: end
      - from: agent2
        to: end
"#;

        let builder = FlowBuilder::from_yaml_str(".", yaml).expect("builder");
        let ctx = FlowContext::default().with_var("route", "a2");
        let plan = builder.plan_sequential_path("main", &ctx, &HashMap::new()).expect("plan");
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].name(), "a2");
    }

    #[test]
    fn plans_loop_with_iteration_guard() {
        let yaml = r#"
agents:
  - id: worker
    model: m
    system_prompt: work
flows:
  - id: loop_flow
    entry: start
    nodes:
      - id: start
        type: input
      - id: loop
        type: loop
        max_iterations: 2
        condition: "iteration < 2"
      - id: worker
        type: agent
        agent: worker
      - id: end
        type: output
    edges:
      - from: start
        to: loop
      - from: loop
        to: worker
        condition: "iteration < 2"
      - from: loop
        to: end
        condition: "else"
      - from: worker
        to: loop
"#;

        let builder = FlowBuilder::from_yaml_str(".", yaml).expect("builder");
        let ctx = FlowContext::default();
        let plan = builder.plan_sequential_path("loop_flow", &ctx, &HashMap::new()).expect("plan");
        assert_eq!(plan.len(), 2, "worker should run twice due to max_iterations=2");
    }

    #[test]
    fn plans_subflow_inline() {
        let yaml = r#"
agents:
  - id: main_agent
    model: m
    system_prompt: main
  - id: child_agent
    model: m
    system_prompt: child
flows:
  - id: child
    entry: c_start
    nodes:
      - id: c_start
        type: input
      - id: c_agent
        type: agent
        agent: child_agent
      - id: c_end
        type: output
    edges:
      - from: c_start
        to: c_agent
      - from: c_agent
        to: c_end
  - id: main
    entry: m_start
    nodes:
      - id: m_start
        type: input
      - id: sub
        type: subflow
        flow: child
      - id: m_agent
        type: agent
        agent: main_agent
      - id: m_end
        type: output
    edges:
      - from: m_start
        to: sub
      - from: sub
        to: m_agent
      - from: m_agent
        to: m_end
"#;

        let builder = FlowBuilder::from_yaml_str(".", yaml).expect("builder");
        let ctx = FlowContext::default();
        let plan = builder.plan_sequential_path("main", &ctx, &HashMap::new()).expect("plan");
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].name(), "child_agent");
        assert_eq!(plan[1].name(), "main_agent");
    }

    #[tokio::test]
    async fn builds_and_runs_concurrent_orchestrator() {
        let yaml = r#"
agents:
  - id: a1
    model: scripted
    system_prompt: one
  - id: a2
    model: scripted
    system_prompt: two
flows:
  - id: main
    entry: start
    nodes:
      - id: start
        type: input
      - id: agent1
        type: agent
        agent: a1
      - id: agent2
        type: agent
        agent: a2
      - id: end
        type: output
"#;

        let builder = FlowBuilder::from_yaml_str(".", yaml).expect("builder");
        let provider = Arc::new(ScriptedProvider::from_scripted_turns(&[
            ScriptedTurn { agent: "a1".to_string(), response: "r1".to_string(), latency_ms: None },
            ScriptedTurn { agent: "a2".to_string(), response: "r2".to_string(), latency_ms: None },
        ]));
        let orch = builder.build_concurrent_orchestrator(provider, "main", &HashMap::new()).expect("concurrent");
        let run = orch.run("task").await.expect("run");
        assert_eq!(run.results.len(), 2);
    }

    #[tokio::test]
    async fn builds_and_runs_group_chat_orchestrator() {
        let yaml = r#"
agents:
  - id: speaker
    model: scripted
    system_prompt: chat
flows:
  - id: main
    entry: start
    nodes:
      - id: start
        type: input
      - id: talker
        type: agent
        agent: speaker
      - id: end
        type: output
"#;

        let builder = FlowBuilder::from_yaml_str(".", yaml).expect("builder");
        let provider = Arc::new(ScriptedProvider::from_scripted_turns(&[
            ScriptedTurn { agent: "speaker".to_string(), response: "hello".to_string(), latency_ms: None },
            ScriptedTurn { agent: "speaker".to_string(), response: "hello".to_string(), latency_ms: None },
            ScriptedTurn { agent: "speaker".to_string(), response: "hello".to_string(), latency_ms: None },
            ScriptedTurn { agent: "speaker".to_string(), response: "hello".to_string(), latency_ms: None },
            ScriptedTurn { agent: "speaker".to_string(), response: "hello".to_string(), latency_ms: None },
            ScriptedTurn { agent: "speaker".to_string(), response: "hello".to_string(), latency_ms: None },
        ]));

        let mut orch = builder.build_group_chat_orchestrator(provider, "main", &HashMap::new()).expect("group chat");
        let run = orch.run("hi").await.expect("run");
        assert_eq!(run.rounds, 6);
    }

    #[test]
    fn builds_handoff_orchestrator() {
        let yaml = r#"
agents:
  - id: concierge
    model: m
    system_prompt: frontdesk
flows:
  - id: main
    entry: start
    nodes:
      - id: start
        type: input
      - id: concierge_node
        type: agent
        agent: concierge
      - id: end
        type: output
"#;

        let builder = FlowBuilder::from_yaml_str(".", yaml).expect("builder");
        let provider = Arc::new(ScriptedProvider::new());
        let orch = builder.build_handoff_orchestrator(provider, "main", &HashMap::new()).expect("handoff");
        assert!(orch.agent("concierge").is_some());
    }
}
