use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::agents::Agent;
use crate::functions::{FunctionDefinition, FunctionParameter, FunctionRegistry, KernelFunction, ToolChoice, json_schema_for, to_value};
use crate::types::ChatMessage;
use crate::{LLMError, LLMProvider};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SkillDefinition {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_agent_tools: Option<bool>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_tools: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub disallowed_tools: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SkillStub {
    pub id: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LoadedSkill {
    pub id: String,
    pub description: Option<String>,
    pub prompt: String,
    pub tools: Vec<String>,
    pub include_agent_tools: bool,
    pub allowed_tools: Vec<String>,
    pub disallowed_tools: Vec<String>,
}

#[derive(Debug, Error)]
pub enum SkillLoadError {
    #[error("unknown skill: {0}")]
    UnknownSkill(String),
    #[error("skill has no text or file: {0}")]
    MissingContent(String),
    #[error("skill file not found: {0}")]
    FileNotFound(PathBuf),
    #[error("skill file read error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Clone)]
pub struct SkillCatalog {
    base_dir: PathBuf,
    skills: HashMap<String, SkillDefinition>,
}

impl SkillCatalog {
    pub fn new(base_dir: PathBuf, skills: Vec<SkillDefinition>) -> Self {
        let mut map = HashMap::new();
        for skill in skills {
            map.insert(skill.id.clone(), skill);
        }
        Self { base_dir, skills: map }
    }

    pub fn stub(&self, id: &str) -> Option<SkillStub> {
        self.skills.get(id).map(|skill| SkillStub {
            id: skill.id.clone(),
            description: skill.description.clone(),
        })
    }

    pub fn load(&self, id: &str) -> Result<LoadedSkill, SkillLoadError> {
        let skill = self
            .skills
            .get(id)
            .ok_or_else(|| SkillLoadError::UnknownSkill(id.to_string()))?;

        let prompt = match (&skill.text, &skill.file) {
            (Some(text), _) => text.clone(),
            (None, Some(file)) => {
                let candidate = self.base_dir.join(file);
                if !candidate.exists() {
                    return Err(SkillLoadError::FileNotFound(candidate));
                }
                fs::read_to_string(candidate)?
            }
            (None, None) => return Err(SkillLoadError::MissingContent(id.to_string())),
        };

        Ok(LoadedSkill {
            id: skill.id.clone(),
            description: skill.description.clone(),
            prompt,
            tools: skill.tools.clone(),
            include_agent_tools: skill.include_agent_tools.unwrap_or(true),
            allowed_tools: skill.allowed_tools.clone(),
            disallowed_tools: skill.disallowed_tools.clone(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SpawnSkillRequest {
    pub skill_id: String,
    #[serde(default)]
    pub input: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillResult {
    pub skill_id: String,
    pub summary: String,
    #[serde(default)]
    pub payload: Value,
}

#[derive(Clone)]
pub struct SkillRuntime {
    inner: Arc<SkillRuntimeInner>,
}

#[derive(Clone)]
struct SkillRuntimeInner {
    catalog: Arc<SkillCatalog>,
    tool_registries: Arc<HashMap<String, Arc<FunctionRegistry>>>,
    provider: Arc<dyn LLMProvider>,
    model: String,
    max_depth: usize,
}

impl SkillRuntime {
    pub fn new(
        catalog: Arc<SkillCatalog>,
        tool_registries: Arc<HashMap<String, Arc<FunctionRegistry>>>,
        provider: Arc<dyn LLMProvider>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            inner: Arc::new(SkillRuntimeInner {
                catalog,
                tool_registries,
                provider,
                model: model.into(),
                max_depth: 4,
            }),
        }
    }

    pub fn registry_for_agent(
        &self,
        agent: &Agent,
        context: &[ChatMessage],
    ) -> Option<FunctionRegistry> {
        if agent.skill_ids().is_empty() {
            return None;
        }

        let function = SpawnSkillFunction::new(
            Arc::clone(&self.inner),
            agent.skill_ids(),
            agent.tool_ids(),
            context.to_vec(),
            0,
        );

        let mut registry = FunctionRegistry::new();
        registry.register(Arc::new(function));
        Some(registry)
    }
}

#[derive(Debug, Error)]
enum SkillRuntimeError {
    #[error("skill not allowed: {0}")]
    SkillNotAllowed(String),
    #[error("skill tool not found: {0}")]
    ToolNotFound(String),
    #[error("skill depth exceeded")]
    DepthExceeded,
    #[error("skill load error: {0}")]
    Load(#[from] SkillLoadError),
    #[error("invalid arguments: {0}")]
    InvalidArguments(String),
}

struct SpawnSkillFunction {
    runtime: Arc<SkillRuntimeInner>,
    allowed_skills: Vec<String>,
    agent_tool_ids: Vec<String>,
    context: Vec<ChatMessage>,
    depth: usize,
}

impl SpawnSkillFunction {
    fn new(
        runtime: Arc<SkillRuntimeInner>,
        allowed_skills: Vec<String>,
        agent_tool_ids: Vec<String>,
        context: Vec<ChatMessage>,
        depth: usize,
    ) -> Self {
        Self {
            runtime,
            allowed_skills,
            agent_tool_ids,
            context,
            depth,
        }
    }

    fn build_definition() -> FunctionDefinition {
        let mut def = FunctionDefinition::new("spawn_skill")
            .with_description("Load and run a skill as a separate subtask, returning a concise summary and payload.");
        def.add_parameter(
            FunctionParameter::new("skill_id", json_schema_for::<String>())
                .with_description("ID of the skill to load."),
        );
        def.add_parameter(
            FunctionParameter::new("input", json_schema_for::<Option<String>>())
                .optional()
                .with_description("Optional input for the skill."),
        );
        def
    }

    fn skill_directory(&self) -> Option<String> {
        if self.allowed_skills.is_empty() {
            return None;
        }

        let mut out = String::from("Available skills (load with spawn_skill):\n");
        for id in &self.allowed_skills {
            let desc = self
                .runtime
                .catalog
                .stub(id)
                .and_then(|stub| stub.description)
                .unwrap_or_else(|| "No description provided.".to_string());
            out.push_str(&format!("- {}: {}\n", id, desc));
        }
        Some(out.trim_end().to_string())
    }

    fn build_registry_for_skill(
        &self,
        skill: &LoadedSkill,
    ) -> Result<FunctionRegistry, SkillRuntimeError> {
        let mut ids = Vec::new();
        ids.extend(skill.tools.iter().cloned());
        if skill.include_agent_tools {
            ids.extend(self.agent_tool_ids.iter().cloned());
        }

        if !skill.allowed_tools.is_empty() {
            ids.retain(|id| skill.allowed_tools.iter().any(|allowed| allowed == id));
        }

        if !skill.disallowed_tools.is_empty() {
            ids.retain(|id| !skill.disallowed_tools.iter().any(|blocked| blocked == id));
        }

        let mut registry = FunctionRegistry::new();
        for tool_id in ids {
            let reg = self
                .runtime
                .tool_registries
                .get(&tool_id)
                .ok_or_else(|| SkillRuntimeError::ToolNotFound(tool_id.clone()))?;
            registry.extend_from(reg);
        }

        if self.depth + 1 <= self.runtime.max_depth {
            let nested = SpawnSkillFunction::new(
                Arc::clone(&self.runtime),
                self.allowed_skills.clone(),
                self.agent_tool_ids.clone(),
                self.context.clone(),
                self.depth + 1,
            );
            registry.register(Arc::new(nested));
        }

        Ok(registry)
    }

    async fn run_skill(
        &self,
        skill: &LoadedSkill,
        input: Option<String>,
        registry: &FunctionRegistry,
    ) -> Result<SkillResult, LLMError> {
        let mut history = Vec::new();
        if !self.context.is_empty() {
            history.extend(self.context.iter().cloned());
        }
        if let Some(input) = input {
            history.push(ChatMessage::user(input));
        }

        let mut system = String::new();
        system.push_str(&skill.prompt);
        if let Some(directory) = self.skill_directory() {
            system.push_str("\n\n");
            system.push_str(&directory);
        }
        system.push_str("\n\nReturn a JSON object with fields \"summary\" and \"payload\".\n");
        system.push_str("summary: 1-3 concise sentences for the parent context.\n");
        system.push_str("payload: concise structured details from the skill.\n");

        let skill_agent = Agent::from_string(format!("skill:{}", skill.id), system);
        let turn = skill_agent
            .execute_with_tools(
                self.runtime.provider.as_ref(),
                &self.runtime.model,
                &history,
                Some(registry),
                Some(ToolChoice::auto()),
            )
            .await?;

        let summary_payload = parse_skill_output(&turn.raw_content);
        Ok(SkillResult {
            skill_id: skill.id.clone(),
            summary: summary_payload.summary,
            payload: summary_payload.payload,
        })
    }
}

#[async_trait]
impl KernelFunction for SpawnSkillFunction {
    fn definition(&self) -> FunctionDefinition {
        Self::build_definition()
    }

    async fn invoke(&self, arguments: &Value) -> Result<Value, LLMError> {
        if self.depth >= self.runtime.max_depth {
            return Err(LLMError::FunctionExecution {
                function: "spawn_skill".to_string(),
                message: SkillRuntimeError::DepthExceeded.to_string(),
            });
        }

        let request: SpawnSkillRequest = serde_json::from_value(arguments.clone()).map_err(|err| {
            LLMError::InvalidFunctionArguments(SkillRuntimeError::InvalidArguments(err.to_string()).to_string())
        })?;

        if !self.allowed_skills.iter().any(|id| id == &request.skill_id) {
            return Err(LLMError::FunctionExecution {
                function: "spawn_skill".to_string(),
                message: SkillRuntimeError::SkillNotAllowed(request.skill_id).to_string(),
            });
        }

        let loaded = self.runtime.catalog.load(&request.skill_id).map_err(|err| {
            LLMError::FunctionExecution {
                function: "spawn_skill".to_string(),
                message: SkillRuntimeError::Load(err).to_string(),
            }
        })?;

        let registry = self.build_registry_for_skill(&loaded).map_err(|err| {
            LLMError::FunctionExecution {
                function: "spawn_skill".to_string(),
                message: err.to_string(),
            }
        })?;

        let result = self
            .run_skill(&loaded, request.input, &registry)
            .await?;

        Ok(to_value(result))
    }
}

#[derive(Debug)]
struct ParsedSkillOutput {
    summary: String,
    payload: Value,
}

fn parse_skill_output(content: &str) -> ParsedSkillOutput {
    if let Some(json) = extract_json_from_mixed_content(content) {
        if let Ok(value) = serde_json::from_str::<Value>(&json) {
            let summary = value
                .get("summary")
                .and_then(|v| v.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| fallback_summary(content));
            let payload = value.get("payload").cloned().unwrap_or(Value::Null);
            return ParsedSkillOutput { summary, payload };
        }
    }

    ParsedSkillOutput {
        summary: fallback_summary(content),
        payload: Value::String(truncate_payload(content)),
    }
}

fn fallback_summary(content: &str) -> String {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return "Skill completed with no output.".to_string();
    }

    let mut summary = String::new();
    let mut sentences = 0;
    for part in trimmed.split_terminator(|c| c == '.' || c == '!' || c == '?') {
        let piece = part.trim();
        if piece.is_empty() {
            continue;
        }
        if !summary.is_empty() {
            summary.push_str(". ");
        }
        summary.push_str(piece);
        sentences += 1;
        if sentences >= 2 {
            break;
        }
    }

    if summary.is_empty() {
        truncate_payload(trimmed)
    } else {
        summary
    }
}

fn truncate_payload(content: &str) -> String {
    let trimmed = content.trim();
    let limit = 2000;
    if trimmed.len() > limit {
        let mut out = trimmed[..limit].to_string();
        out.push_str("...");
        out
    } else {
        trimmed.to_string()
    }
}

fn extract_json_from_mixed_content(content: &str) -> Option<String> {
    let bytes = content.as_bytes();
    let mut start_pos = None;
    let mut end_pos = None;
    let mut depth: i32 = 0;
    let mut in_str = false;
    let mut escaped = false;

    for (i, &b) in bytes.iter().enumerate() {
        let c = b as char;
        if in_str {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }

        if c == '"' {
            in_str = true;
            continue;
        }

        if c == '{' {
            if depth == 0 {
                start_pos = Some(i);
            }
            depth += 1;
        } else if c == '}' {
            depth -= 1;
            if depth == 0 {
                end_pos = Some(i);
            }
        }
    }

    match (start_pos, end_pos) {
        (Some(s), Some(e)) if e > s => Some(content[s..=e].to_string()),
        _ => None,
    }
}
