use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use meval;

use super::sequential::{SequentialEvent, SequentialOrchestrator, SequentialRun};
use crate::flows::handoffflow::{HandoffDirective, HandoffMatcher, HandoffRule};
use crate::functions::http::load_http_function;
use crate::{
    agents::{Agent, AgentError},
    functions::{FunctionCall, FunctionRegistry},
    LLMProvider,
};

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

// ...

    pub fn build_agents(
        &self,
        tool_registries: &HashMap<String, Arc<FunctionRegistry>>,
    ) -> Result<HashMap<String, Agent>, FlowLoadError> {
        let mut agents = HashMap::new();
        
        let prompt_map: HashMap<String, &PromptDefinition> = self
            .document
            .prompts
            .iter()
            .map(|p| (p.id.clone(), p))
            .collect();

        for def in &self.document.agents {
            let instructions = if let Some(ref sys_prompt) = def.system_prompt {
                if let Some(prompt_def) = prompt_map.get(sys_prompt) {
                    // Found a matching prompt definition
                    if let Some(ref text) = prompt_def.text {
                        text.clone()
                    } else if let Some(ref file) = prompt_def.file {
                        load_instructions(&self.base_dir, Some(file))?
                    } else {
                        // Fallback or empty if definition exists but no content
                        String::new()
                    }
                } else {
                    // No matching definition, treat as file path or inline text
                    load_instructions(&self.base_dir, Some(sys_prompt))?
                }
            } else {
                String::new()
            };

            let mut agent = Agent::from_string(def.id.clone(), instructions);

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
        let flow = self.flow(flow_id)?;
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

        let manager = if let Some(opts) = &flow.group_chat {
            crate::flows::group_chat::RoundRobinGroupChatManager::new()
                .with_maximum_rounds(opts.maximum_rounds)
                .with_user_prompt_frequency(opts.user_prompt_frequency)
        } else {
            crate::flows::group_chat::RoundRobinGroupChatManager::new()
        };

        Ok(crate::flows::group_chat::GroupChatOrchestrator::new(
            provider,
            model,
            manager,
        )
        .with_agents(pipeline))
    }

    pub fn build_handoff_orchestrator(
        &self,
        provider: Arc<dyn LLMProvider>,
        flow_id: &str,
        tool_registries: &HashMap<String, Arc<FunctionRegistry>>,
    ) -> Result<crate::flows::handoffflow::HandoffOrchestrator, FlowLoadError> {
        let flow = self.flow(flow_id)?;
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

        if let Some(opts) = &flow.handoff {
            if let Some(max_handoffs) = opts.max_handoffs {
                orchestrator = orchestrator.with_max_handoffs(Some(max_handoffs));
            }
            if let Some(rounds) = opts.max_rounds {
                orchestrator = orchestrator.with_max_rounds(rounds);
            }
            if let Some(timeout) = opts.llm_timeout_ms {
                orchestrator = orchestrator.with_llm_timeout_ms(timeout);
            }

            for alias in &opts.aliases {
                orchestrator.add_alias(alias.alias.clone(), alias.target.clone());
            }

            for rule in &opts.rules {
                orchestrator.define_handoff(handoff_rule_from_definition(rule)?);
            }
        }

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
        let mut resolved_functions: HashMap<String, Arc<dyn crate::functions::KernelFunction>> = functions.clone();

        // Auto-load HTTP specs into the local function map when no function is supplied.
        for tool in &self.document.tools {
            if tool.function.is_none() && tool.kind == "http" {
                if let Some(spec_path) = &tool.spec {
                    // Only load if not already provided
                    if !resolved_functions.contains_key(&tool.id) {
                        match load_http_function(&self.base_dir, spec_path, &tool.id) {
                            Ok(func) => {
                                resolved_functions.insert(tool.id.clone(), func.clone());
                                resolved_functions.insert(spec_path.clone(), func.clone());
                                if let Some(abs) = self.base_dir.join(spec_path).to_str() {
                                    resolved_functions.insert(format!("http:{}", abs), func.clone());
                                }
                            }
                            Err(err) => {
                                return Err(FlowLoadError::ToolResolution(tool.id.clone(), err.to_string()));
                            }
                        }
                    }
                }
            }
        }

        for tool in &self.document.tools {
            let mut registry = FunctionRegistry::new();
            let func = self.resolve_tool_function(tool, &resolved_functions)?;
            registry.register(func);
            registries.insert(tool.id.clone(), Arc::new(registry));
        }

        Ok(registries)
    }

    fn resolve_tool_function(
        &self,
        tool: &ToolDefinition,
        functions: &HashMap<String, Arc<dyn crate::functions::KernelFunction>>,
    ) -> Result<Arc<dyn crate::functions::KernelFunction>, FlowLoadError> {
        if let Some(function_name) = &tool.function {
            let func = functions
                .get(function_name)
                .ok_or_else(|| FlowLoadError::FunctionNotFound(tool.id.clone(), function_name.clone()))?;
            return Ok(func.clone());
        }

        let mut candidates = Vec::new();
        candidates.push(tool.id.clone());
        if let Some(spec) = &tool.spec {
            candidates.push(spec.clone());
            candidates.push(format!("{}:{}", tool.kind, spec));
            let joined = self.base_dir.join(spec);
            if let Some(path) = joined.to_str() {
                candidates.push(path.to_string());
                candidates.push(format!("{}:{}", tool.kind, path));
            }
        }

        for key in candidates {
            if let Some(func) = functions.get(&key) {
                return Ok(func.clone());
            }
        }

        let detail = tool
            .spec
            .clone()
            .or_else(|| tool.function.clone())
            .unwrap_or_else(|| "no function or spec provided".to_string());
        Err(FlowLoadError::ToolResolution(tool.id.clone(), detail))
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

    pub fn plan_execution_steps(
        &self,
        flow_id: &str,
        ctx: &FlowContext,
    ) -> Result<Vec<PlannedStep>, FlowLoadError> {
        let mut visited_flows = Vec::new();
        self.plan_steps(flow_id, ctx, &mut visited_flows)
    }

    pub fn build_execution_plan(
        &self,
        flow_id: &str,
        ctx: &FlowContext,
        tool_registries: &HashMap<String, Arc<FunctionRegistry>>,
    ) -> Result<Vec<ExecutionStep>, FlowLoadError> {
        let planned = self.plan_execution_steps(flow_id, ctx)?;
        let agents = self.build_agents(tool_registries)?;

        planned
            .into_iter()
            .map(|step| match step {
                PlannedStep::Agent(plan) => {
                    let agent = agents
                        .get(&plan.id)
                        .cloned()
                        .ok_or_else(|| FlowLoadError::AgentNotFound(plan.id.clone()))?;
                    Ok(ExecutionStep::Agent(apply_call_settings(agent, plan.params.as_ref())))
                }
                PlannedStep::Parallel { branches, converge } => {
                    let mapped = branches
                        .into_iter()
                        .map(|branch| {
                            branch
                                .into_iter()
                                .map(|plan| {
                                    let agent = agents
                                        .get(&plan.id)
                                        .cloned()
                                        .ok_or_else(|| FlowLoadError::AgentNotFound(plan.id.clone()))?;
                                    Ok(apply_call_settings(agent, plan.params.as_ref()))
                                })
                                .collect::<Result<Vec<_>, FlowLoadError>>()
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(ExecutionStep::Parallel {
                        branches: mapped,
                        converge,
                    })
                }
                PlannedStep::Tool { tool, arguments } => Ok(ExecutionStep::Tool { tool, arguments }),
            })
            .collect()
    }

    /// Convenience: execute tool nodes in the plan, flatten the remaining agent
    /// pipeline, and optionally emit step events through the provided callback.
    pub async fn run_sequential_flow<F>(
        &self,
        flow_id: &str,
        ctx: &FlowContext,
        tool_registries: &HashMap<String, Arc<FunctionRegistry>>,
        provider: Arc<dyn LLMProvider>,
        task: String,
        event_callback: Option<F>,
    ) -> Result<(SequentialRun, Vec<ToolRunResult>), FlowRunError>
    where
        F: Fn(&SequentialEvent) + Send + Sync + 'static,
    {
        let plan = self.build_execution_plan(flow_id, ctx, tool_registries)?;

        let tool_runs = execute_tool_steps(&plan, tool_registries).await?;
        let mut task_with_tools = task.clone();
        for run in &tool_runs {
            task_with_tools.push_str(&format!("\n[tool:{}] {}\n", run.tool, run.value));
        }

        let pipeline = flatten_agent_pipeline(&plan);
        if pipeline.is_empty() {
            return Err(FlowRunError::NoAgents(flow_id.to_string()));
        }

        let model_lookup: HashMap<String, String> = self
            .document
            .agents
            .iter()
            .map(|a| (a.id.clone(), a.model.clone()))
            .collect();

        let model = model_lookup
            .get(pipeline[0].name())
            .cloned()
            .unwrap_or_else(|| "gpt-4o".to_string());

        let orchestrator = {
            let base = SequentialOrchestrator::new(provider, model).with_agents(pipeline);
            if let Some(cb) = event_callback {
                base.with_event_callback(cb)
            } else {
                base
            }
        };

        let run = orchestrator.run(task_with_tools).await?;

        Ok((run, tool_runs))
    }

    fn plan_steps(
        &self,
        flow_id: &str,
        ctx: &FlowContext,
        visited_flows: &mut Vec<String>,
    ) -> Result<Vec<PlannedStep>, FlowLoadError> {
        if visited_flows.contains(&flow_id.to_string()) {
            return Err(FlowLoadError::SubflowCycle(flow_id.to_string()));
        }
        visited_flows.push(flow_id.to_string());

        let flow = self.flow(flow_id)?;
        let mut current = flow.entry.clone();
        let mut steps = Vec::new();
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
                    steps.push(PlannedStep::Agent(PlannedAgent {
                        id: agent.clone(),
                        params: parameters.clone(),
                    }));
                }
                FlowNodeKind::Tool { tool, arguments } => {
                    steps.push(PlannedStep::Tool { tool: tool.clone(), arguments: arguments.clone() });
                }
                FlowNodeKind::Parallel { converge } => {
                    let converge = converge.unwrap_or(true);
                    let outgoing: Vec<&FlowEdge> = flow
                        .edges
                        .iter()
                        .filter(|e| edge_base(&e.from) == node.base.id)
                        .collect();

                    if outgoing.is_empty() {
                        return Err(FlowLoadError::MissingParallelBranches(node.base.id.clone()));
                    }

                    let mut branches = Vec::new();
                    let mut join_target: Option<Option<String>> = None;
                    for edge in outgoing {
                        let (branch, join) = self.collect_parallel_branch(
                            flow,
                            &edge.to,
                            ctx,
                            visited_flows,
                            HashMap::new(),
                        )?;
                        branches.push(branch);

                        if converge {
                            if let Some(existing) = &join_target {
                                if existing != &join {
                                    return Err(FlowLoadError::ParallelConvergence(node.base.id.clone()));
                                }
                            }
                            if join_target.is_none() {
                                join_target = Some(join.clone());
                            }
                        } else if join_target.is_none() {
                            join_target = Some(join.clone());
                        }
                    }

                    steps.push(PlannedStep::Parallel { branches, converge });

                    if let Some(next) = join_target.flatten() {
                        current = next;
                        continue;
                    } else {
                        break;
                    }
                }
                FlowNodeKind::Decision { .. } => {}
                FlowNodeKind::Merge {} => {}
                FlowNodeKind::Loop { .. } => {}
                FlowNodeKind::Subflow { flow } => {
                    let mut nested = self.plan_steps(flow, ctx, visited_flows)?;
                    steps.append(&mut nested);
                }
                FlowNodeKind::Output {} => break,
            }

            let next = self.next_node(flow, node, ctx, &mut loop_counters)?;
            match next {
                Some(next_id) => {
                    current = next_id;
                }
                None => break,
            }
        }

        visited_flows.retain(|f| f != flow_id);
        Ok(steps)
    }

    fn collect_parallel_branch(
        &self,
        flow: &FlowDefinition,
        start: &str,
        ctx: &FlowContext,
        visited_flows: &mut Vec<String>,
        mut loop_counters: HashMap<String, u32>,
    ) -> Result<(Vec<PlannedAgent>, Option<String>), FlowLoadError> {
        let mut current = start.to_string();
        let mut branch = Vec::new();

        loop {
            let node = flow
                .nodes
                .iter()
                .find(|n| n.base.id == current)
                .ok_or_else(|| FlowLoadError::NodeNotFound(current.clone()))?;

            match &node.kind {
                FlowNodeKind::Input {} => {}
                FlowNodeKind::Agent { agent, parameters, .. } => {
                    branch.push(PlannedAgent {
                        id: agent.clone(),
                        params: parameters.clone(),
                    });
                }
                FlowNodeKind::Decision { .. } => {}
                FlowNodeKind::Tool { .. } => {}
                FlowNodeKind::Merge {} => return Ok((branch, Some(node.base.id.clone()))),
                FlowNodeKind::Parallel { .. } => return Err(FlowLoadError::UnsupportedNode(node.base.id.clone())),
                FlowNodeKind::Loop { .. } => {}
                FlowNodeKind::Subflow { flow } => {
                    let mut nested = self.plan_nodes(flow, ctx, visited_flows)?;
                    branch.append(&mut nested);
                }
                FlowNodeKind::Output {} => return Ok((branch, None)),
            }

            let next = self.next_node(flow, node, ctx, &mut loop_counters)?;
            match next {
                Some(next_id) => current = next_id,
                None => return Ok((branch, None)),
            }
        }
    }

    fn next_node(
        &self,
        flow: &FlowDefinition,
        node: &FlowNode,
        ctx: &FlowContext,
        loop_counters: &mut HashMap<String, u32>,
    ) -> Result<Option<String>, FlowLoadError> {
        let outgoing: Vec<&FlowEdge> = flow
            .edges
            .iter()
            .filter(|e| edge_base(&e.from) == node.base.id)
            .collect();

        if outgoing.is_empty() {
            return Ok(None);
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
            for edge in &outgoing {
                if edge.condition.is_none() {
                    selected = Some(edge.to.clone());
                    break;
                }
            }
        }

        if let FlowNodeKind::Loop { .. } = &node.kind {
            let counter = loop_counters.entry(node.base.id.clone()).or_insert(0);
            *counter += 1;
        }

        selected
            .ok_or_else(|| FlowLoadError::NoMatchingEdge(node.base.id.clone()))
            .map(Some)
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

#[derive(Debug, Default, Clone)]
pub struct FlowContext {
    pub vars: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlannedAgent {
    id: String,
    params: Option<CallSettings>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PlannedStep {
    Agent(PlannedAgent),
    Tool {
        tool: String,
        arguments: Option<serde_json::Value>,
    },
    Parallel {
        branches: Vec<Vec<PlannedAgent>>,
        converge: bool,
    },
}

#[derive(Debug, Clone)]
pub enum ExecutionStep {
    Agent(Agent),
    Tool {
        tool: String,
        arguments: Option<serde_json::Value>,
    },
    Parallel {
        branches: Vec<Vec<Agent>>,
        converge: bool,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolRunResult {
    pub tool: String,
    pub value: serde_json::Value,
}

/// Flatten a mixed execution plan (agents, tools, parallel branches) into a simple
/// sequential agent pipeline. Tool steps are skipped and parallel branches are
/// concatenated in order, which mirrors the simple demo semantics.
pub fn flatten_agent_pipeline(steps: &[ExecutionStep]) -> Vec<Agent> {
    let mut pipeline = Vec::new();
    for step in steps {
        match step {
            ExecutionStep::Agent(agent) => pipeline.push(agent.clone()),
            ExecutionStep::Parallel { branches, .. } => {
                for branch in branches {
                    for agent in branch {
                        pipeline.push(agent.clone());
                    }
                }
            }
            ExecutionStep::Tool { .. } => {}
        }
    }
    pipeline
}

/// Execute all tool steps in a plan, returning their outputs in order.
/// Tool arguments must be JSON objects; a missing or invalid registry results in an error.
pub async fn execute_tool_steps(
    steps: &[ExecutionStep],
    tool_registries: &HashMap<String, Arc<FunctionRegistry>>,
) -> Result<Vec<ToolRunResult>, ToolExecutionError> {
    let mut results = Vec::new();

    for step in steps {
        if let ExecutionStep::Tool { tool, arguments } = step {
            let registry = tool_registries
                .get(tool)
                .ok_or_else(|| ToolExecutionError::RegistryMissing(tool.clone()))?;

            let args = arguments.clone().unwrap_or_else(|| Value::Object(Default::default()));
            let arguments_obj = match args {
                Value::Object(map) => Value::Object(map),
                _ => return Err(ToolExecutionError::InvalidArguments(tool.clone())),
            };

            let call = FunctionCall {
                name: registry
                    .definitions()
                    .first()
                    .map(|d| d.name.clone())
                    .unwrap_or_else(|| tool.clone()),
                arguments: arguments_obj,
                raw_arguments: None,
            };

            let value = registry
                .invoke(&call)
                .await
                .map_err(|err| ToolExecutionError::InvocationFailed(tool.clone(), err.to_string()))?;

            results.push(ToolRunResult {
                tool: tool.clone(),
                value,
            });
        }
    }

    Ok(results)
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

fn handoff_rule_from_definition(def: &HandoffRuleDefinition) -> Result<HandoffRule, FlowLoadError> {
    let matcher = handoff_matcher_from_definition(&def.matcher)?;
    let directive = HandoffDirective {
        target: def.target.clone(),
        message: def.message.clone(),
    };

    Ok(HandoffRule {
        id: def.id.clone().unwrap_or_default(),
        matcher,
        resolve: Arc::new(move |_t, _txt| Some(directive.clone())),
    })
}

fn handoff_matcher_from_definition(def: &HandoffMatcherDefinition) -> Result<HandoffMatcher, FlowLoadError> {
    match def {
        HandoffMatcherDefinition::KeywordsAny { keywords } => Ok(HandoffMatcher::KeywordsAny(keywords.clone())),
        HandoffMatcherDefinition::KeywordsAll { keywords } => Ok(HandoffMatcher::KeywordsAll(keywords.clone())),
        HandoffMatcherDefinition::Regex { pattern } => {
            let regex = regex::Regex::new(pattern)
                .map_err(|err| FlowLoadError::InvalidRegex(pattern.clone(), err.to_string()))?;
            Ok(HandoffMatcher::Regex(regex))
        }
    }
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
    use std::sync::Mutex;
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

        assert!(matches!(&flow.nodes[3].kind, FlowNodeKind::Tool { tool, arguments: _ } if tool == "search"));
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
                file: Some("prompts/research.txt".to_string()),
                text: None,
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
                group_chat: None,
                handoff: None,
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

    #[test]
    fn plans_parallel_execution_steps() {
        let yaml = r#"
agents:
  - id: left
    model: m
    system_prompt: left
  - id: right
    model: m
    system_prompt: right
  - id: closer
    model: m
    system_prompt: close
flows:
  - id: main
    entry: start
    nodes:
      - id: start
        type: input
      - id: fork
        type: parallel
        converge: true
      - id: left_node
        type: agent
        agent: left
      - id: right_node
        type: agent
        agent: right
      - id: merge
        type: merge
      - id: final
        type: agent
        agent: closer
      - id: end
        type: output
    edges:
      - from: start
        to: fork
      - from: fork
        to: left_node
      - from: fork
        to: right_node
      - from: left_node
        to: merge
      - from: right_node
        to: merge
      - from: merge
        to: final
      - from: final
        to: end
"#;

        let builder = FlowBuilder::from_yaml_str(".", yaml).expect("builder");
        let steps = builder.plan_execution_steps("main", &FlowContext::default()).expect("plan");

        assert_eq!(steps.len(), 2, "parallel block + final agent");
        match &steps[0] {
            PlannedStep::Parallel { branches, converge } => {
                assert_eq!(branches.len(), 2);
                assert!(converge, "parallel should converge by default");
                let branch_sizes: Vec<_> = branches.iter().map(|b| b.len()).collect();
                assert_eq!(branch_sizes, vec![1, 1]);
            }
            other => panic!("expected parallel step, got {other:?}"),
        }

        match &steps[1] {
            PlannedStep::Agent(agent) => assert_eq!(agent.id, "closer"),
            other => panic!("expected final agent, got {other:?}"),
        }
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

    #[tokio::test]
    async fn applies_group_chat_manager_from_yaml() {
        let yaml = r#"
agents:
  - id: speaker
    model: scripted
    system_prompt: chat
flows:
  - id: main
    entry: start
    group_chat:
      maximum_rounds: 2
      user_prompt_frequency: 1
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
            ScriptedTurn { agent: "speaker".to_string(), response: "first".to_string(), latency_ms: None },
            ScriptedTurn { agent: "speaker".to_string(), response: "second".to_string(), latency_ms: None },
        ]));

        let mut orch = builder.build_group_chat_orchestrator(provider, "main", &HashMap::new()).expect("group chat");
        let prompts: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
        let tracker = prompts.clone();
        orch = orch.with_user_input_callback(move |_transcript| {
            let mut guard = tracker.lock().unwrap();
            *guard += 1;
            Some("user input".to_string())
        });

        let run = orch.run("hi").await.expect("run");
        assert_eq!(run.rounds, 2, "maximum rounds from YAML should be respected");
        assert!(run
            .events
            .iter()
            .any(|e| matches!(e, crate::flows::group_chat::GroupChatEvent::UserMessage { .. })));
        assert_eq!(*prompts.lock().unwrap(), 1);
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

    #[test]
    fn resolves_tool_by_spec_identifier() {
        use crate::functions::{KernelFunction, FunctionDefinition};

        struct Dummy;
        #[async_trait::async_trait]
        impl KernelFunction for Dummy {
            fn definition(&self) -> FunctionDefinition {
                FunctionDefinition::new("dummy")
            }

            async fn invoke(&self, _arguments: &Value) -> Result<Value, crate::LLMError> {
                Ok(Value::Null)
            }
        }

        let dir = temp_dir();
        let spec_path = dir.join("tools").join("search.json");
        std::fs::create_dir_all(spec_path.parent().unwrap()).unwrap();
        std::fs::write(&spec_path, "{}").unwrap();

        let yaml = format!(
            r#"
tools:
  - id: search_tool
    kind: http
    spec: {}
flows:
  - id: main
    entry: n1
    nodes:
      - id: n1
        type: input
      - id: n2
        type: output
"#,
            spec_path
                .file_name()
                .unwrap()
                .to_string_lossy()
        );

        let mut functions = HashMap::new();
        functions.insert(
            format!("http:{}", spec_path.display()),
            Arc::new(Dummy) as Arc<dyn KernelFunction>,
        );

        let builder = FlowBuilder::from_yaml_str(spec_path.parent().unwrap(), &yaml).expect("builder");
        let registries = builder.build_tool_registries(&functions).expect("registries");
        assert!(registries.contains_key("search_tool"));
        let defs = registries.get("search_tool").unwrap().definitions();
        assert_eq!(defs[0].name, "dummy");
    }

    #[test]
    fn plans_tool_nodes_into_execution_steps() {
        let yaml = r#"
agents:
  - id: a1
    model: m
tools:
  - id: t1
    kind: internal
    function: f1
flows:
  - id: main
    entry: n1
    nodes:
      - id: n1
        type: input
      - id: tool
        type: tool
        tool: t1
      - id: n2
        type: output
    edges:
      - from: n1
        to: tool
      - from: tool
        to: n2
"#;

        let builder = FlowBuilder::from_yaml_str(".", yaml).expect("builder");
        let steps = builder
            .plan_execution_steps("main", &FlowContext::default())
            .expect("steps");
        assert!(matches!(&steps[0], PlannedStep::Tool { tool, arguments: None } if tool == "t1"));
    }

    #[test]
    fn autoloads_http_tool_from_spec_file() {
        let temp_dir = temp_dir().join("http_tool_autoload");
        std::fs::create_dir_all(&temp_dir).unwrap();
        let spec_path = temp_dir.join("search.yaml");
        std::fs::write(
            &spec_path,
            r#"
name: search_tool
description: Simple search proxy
method: GET
url: https://example.com/search
query:
  q:
    type: string
    description: Query
"#,
        )
        .unwrap();

        let yaml = r#"
tools:
  - id: search_tool
    kind: http
    spec: search.yaml
flows:
  - id: main
    entry: n1
    nodes:
      - id: n1
        type: input
      - id: n2
        type: output
"#;

        let functions = HashMap::new();

        let builder = FlowBuilder::from_yaml_str(&temp_dir, yaml).expect("builder");
        let registries = builder.build_tool_registries(&functions).expect("registries");
        let defs = registries.get("search_tool").unwrap().definitions();
        assert_eq!(defs[0].name, "search_tool");
        assert_eq!(defs[0].description.as_deref(), Some("Simple search proxy"));
    }

    #[tokio::test]
    async fn builds_handoff_rules_from_yaml() {
        let yaml = r#"
agents:
  - id: concierge
    model: scripted
    system_prompt: frontdesk
  - id: weather
    model: scripted
    system_prompt: forecast
flows:
  - id: main
    entry: start
    handoff:
      aliases:
        - alias: wx
          target: weather
      rules:
        - id: weather_rule
          target: wx
          matcher: keywords_any
          keywords: ["weather"]
      max_handoffs: 1
      max_rounds: 5
      llm_timeout_ms: 5000
    nodes:
      - id: start
        type: input
      - id: concierge_node
        type: agent
        agent: concierge
      - id: weather_node
        type: agent
        agent: weather
      - id: end
        type: output
"#;

        let builder = FlowBuilder::from_yaml_str(".", yaml).expect("builder");
        let provider = Arc::new(ScriptedProvider::from_scripted_turns(&[
            ScriptedTurn { agent: "concierge".to_string(), response: "the weather is needed".to_string(), latency_ms: None },
            ScriptedTurn { agent: "weather".to_string(), response: "clear skies".to_string(), latency_ms: None },
        ]));

        let orch = builder.build_handoff_orchestrator(provider, "main", &HashMap::new()).expect("handoff");
        let mut session = orch.session("concierge").expect("session");
        let turn = session.send("hi").await.expect("send");

        assert!(matches!(
            turn.events.iter().find(|e| matches!(e, crate::flows::handoffflow::HandoffEvent::HandOff { .. })),
            Some(crate::flows::handoffflow::HandoffEvent::HandOff { to, .. }) if to == "weather"
        ));
        assert_eq!(turn.reply.as_deref(), Some("clear skies"));
    }
}
