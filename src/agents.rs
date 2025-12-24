use std::{
    fmt,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use handlebars::Handlebars;
use serde::Serialize;

use crate::{
    functions::{FunctionRegistry, ToolChoice},
    skills::SkillStub,
    types::{ChatMessage, CompletionRequest},
    flows::handoffflow::{AgentAction, AgentTurn, ActionEnvelope},
    LLMError, LLMProvider,
};

#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("unknown agent: {0}")]
    UnknownAgent(String),
    #[error("template file not found: {0}")]
    TemplateNotFound(PathBuf),
    #[error("template error: {0}")]
    Template(#[from] handlebars::TemplateError),
    #[error("template render error: {0}")]
    TemplateRender(#[from] handlebars::RenderError),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("maximum handoffs reached")]
    MaxHandoffsReached,
    #[error("maximum rounds reached")]
    MaxRoundsReached,
    #[error("no agents registered")]
    NoAgentsRegistered,
    #[error("manager produced invalid decision: {0}")]
    InvalidManagerDecision(String),
    #[error("provider call timed out")]
    ProviderTimeout,
    #[error(transparent)]
    Provider(#[from] LLMError),
}

#[derive(Clone)]
pub struct Agent {
    name: String,
    description: Option<String>,
    instructions: String,
    functions: Option<Arc<FunctionRegistry>>,
    tool_ids: Vec<String>,
    skills: Vec<SkillStub>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<u32>,
    tool_choice: Option<ToolChoice>,
    provider_override: Option<Arc<dyn LLMProvider>>,
    model_override: Option<String>,
}

impl fmt::Debug for Agent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Agent")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("has_functions", &self.functions.is_some())
            .field("temperature", &self.temperature)
            .field("top_p", &self.top_p)
            .field("max_tokens", &self.max_tokens)
            .finish()
    }
}

impl Agent {
    pub fn from_string(name: impl Into<String>, instructions: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            instructions: instructions.into(),
            functions: None,
            tool_ids: Vec::new(),
            skills: Vec::new(),
            temperature: None,
            top_p: None,
            max_tokens: None,
            tool_choice: None,
            provider_override: None,
            model_override: None,
        }
    }

    pub fn from_handlebars_file<T: Serialize>(
        name: impl Into<String>,
        template_path: impl AsRef<Path>,
        data: &T,
    ) -> Result<Self, AgentError> {
        let template_path = template_path.as_ref();
        let template = fs::read_to_string(template_path)
            .map_err(|_| AgentError::TemplateNotFound(template_path.to_path_buf()))?;
        let hb = Handlebars::new();
        let rendered = hb.render_template(&template, data)?;

        Ok(Self::from_string(name, rendered))
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn instructions(&self) -> &str {
        &self.instructions
    }

    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn with_tool_ids(mut self, tool_ids: Vec<String>) -> Self {
        self.tool_ids = tool_ids;
        self
    }

    pub fn tool_ids(&self) -> Vec<String> {
        self.tool_ids.clone()
    }

    pub fn with_skills(mut self, skills: Vec<SkillStub>) -> Self {
        self.skills = skills;
        self
    }

    pub fn skills(&self) -> &[SkillStub] {
        &self.skills
    }

    pub fn skill_ids(&self) -> Vec<String> {
        self.skills.iter().map(|skill| skill.id.clone()).collect()
    }

    pub fn with_function_registry(mut self, registry: Arc<FunctionRegistry>) -> Self {
        self.functions = Some(registry);
        self
    }

    pub fn function_registry(&self) -> Option<Arc<FunctionRegistry>> {
        self.functions.clone()
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    pub fn with_provider(mut self, provider: Arc<dyn LLMProvider>) -> Self {
        self.provider_override = Some(provider);
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model_override = Some(model.into());
        self
    }



    pub(crate) async fn execute(
        &self,
        provider: &(dyn LLMProvider + Send + Sync),
        model: &str,
        history: &[ChatMessage],
    ) -> Result<AgentTurn, LLMError> {
        self.execute_with_tools(provider, model, history, None, None).await
    }

    pub(crate) async fn execute_with_tools(
        &self,
        provider: &(dyn LLMProvider + Send + Sync),
        model: &str,
        history: &[ChatMessage],
        additional_functions: Option<&FunctionRegistry>,
        tool_choice: Option<ToolChoice>,
    ) -> Result<AgentTurn, LLMError> {
        let mut messages = Vec::with_capacity(history.len() + 1);
        messages.push(ChatMessage::system(self.instructions.clone()));
        messages.extend(history.iter().cloned());

        let active_provider: &(dyn LLMProvider + Send + Sync) = match &self.provider_override {
            Some(custom) => custom.as_ref(),
            None => provider,
        };

        let target_model = self.model_override.as_deref().unwrap_or(model);

        let mut request = CompletionRequest::new(target_model.to_string(), messages.clone());

        if let Some(max_tokens) = self.max_tokens {
            request = request.with_max_tokens(max_tokens);
        }

        if let Some(temperature) = self.temperature {
            request = request.with_temperature(temperature);
        }

        if let Some(top_p) = self.top_p {
            request = request.with_top_p(top_p);
        }

        // Merge internal/extra functions with agent functions when both exist.
        // The merged registry must live long enough, so we store it in an Option outside the match.
        let agent_functions = self.functions.as_ref().map(|arc| arc.as_ref());
        let merged_storage = match (additional_functions, agent_functions) {
            (Some(additional), Some(agent_funcs)) => {
                let mut registry = FunctionRegistry::new();
                registry.extend_from(agent_funcs);
                registry.extend_from(additional);
                Some(registry)
            }
            _ => None,
        };
        let functions_to_use: Option<&FunctionRegistry> = merged_storage
            .as_ref()
            .map(|r| r as &FunctionRegistry)
            .or_else(|| match (additional_functions, agent_functions) {
                (Some(additional), None) => Some(additional),
                (None, Some(agent_funcs)) => Some(agent_funcs),
                _ => None,
            });

        if let Some(functions) = functions_to_use {
            request = request.with_function_registry(functions);
        }

        let effective_tool_choice = tool_choice.or_else(|| self.tool_choice.clone());
        if let Some(tool_choice) = &effective_tool_choice {
            request = request.with_tool_choice(tool_choice.clone());
        }

        let max_tool_rounds = 4;
        let mut all_tool_calls = Vec::new();
        let mut last_usage = None;
        let mut last_content = String::new();
        let mut action_override: Option<AgentAction> = None;

        for round in 0..max_tool_rounds {
            let response = active_provider.complete(request).await?;
            let mut assistant_msg = response.message.clone();
            last_usage = response.usage;

            for (i, call) in assistant_msg.tool_calls.iter_mut().enumerate() {
                if call.id.is_none() {
                    call.id = Some(format!("tool_call_{round}_{i}"));
                }
            }

            last_content = assistant_msg.text().unwrap_or_default().to_string();
            all_tool_calls.extend(assistant_msg.tool_calls.clone());
            messages.push(assistant_msg.clone());

            if assistant_msg.tool_calls.is_empty() {
                break;
            }

            let Some(functions) = functions_to_use else {
                break;
            };

            for call in assistant_msg.tool_calls {
                let id = call.id.clone().unwrap_or_else(|| format!("tool_call_{round}_x"));
                let tool_result = functions.invoke(&call.function).await;
                let tool_value = match tool_result {
                    Ok(value) => value,
                    Err(err) => serde_json::json!({ "error": err.to_string() }),
                };

                if action_override.is_none() {
                    if let Ok(action_envelope) = serde_json::from_value::<ActionEnvelope>(tool_value.clone()) {
                        action_override = Some(action_envelope.into());
                    }
                }

                let tool_content = serde_json::to_string(&tool_value)
                    .unwrap_or_else(|_| "{\"error\":\"failed to serialize tool result\"}".to_string());
                messages.push(ChatMessage::tool(id, tool_content));
            }

            if action_override.is_some() {
                break;
            }

            let mut next_request = CompletionRequest::new(target_model.to_string(), messages.clone());
            if let Some(max_tokens) = self.max_tokens {
                next_request = next_request.with_max_tokens(max_tokens);
            }
            if let Some(temperature) = self.temperature {
                next_request = next_request.with_temperature(temperature);
            }
            if let Some(top_p) = self.top_p {
                next_request = next_request.with_top_p(top_p);
            }
            if let Some(functions) = functions_to_use {
                next_request = next_request.with_function_registry(functions);
            }
            if let Some(tool_choice) = &effective_tool_choice {
                next_request = next_request.with_tool_choice(tool_choice.clone());
            }

            request = next_request;
        }

        let action = action_override.unwrap_or_else(|| AgentAction::from_response(&last_content));

        Ok(AgentTurn {
            action,
            tool_calls: all_tool_calls,
            usage: last_usage,
            raw_content: last_content,
        })
    }
}
