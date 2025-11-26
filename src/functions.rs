use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize, Serializer};
use serde::ser::SerializeStruct;
use serde_json::Value;

use crate::LLMError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: FunctionParameters,
}

impl FunctionDefinition {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            parameters: FunctionParameters::new(),
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn add_parameter(&mut self, parameter: FunctionParameter) {
        let FunctionParameter {
            name,
            mut schema,
            description,
            required,
            default,
        } = parameter;

        if let Some(description) = description {
            schema
                .as_object_mut()
                .map(|object| object.insert("description".to_string(), Value::String(description)));
        }

        if let Some(default) = default {
            schema
                .as_object_mut()
                .map(|object| object.insert("default".to_string(), default));
        }

        if required {
            self.parameters.required.push(name.clone());
        }

        self.parameters.properties.insert(name, schema);
    }

    pub fn to_tool(&self) -> Tool {
        Tool::from(self.clone())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionParameters {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub properties: BTreeMap<String, Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub additional_properties: Option<bool>,
}

impl FunctionParameters {
    pub fn new() -> Self {
        Self {
            kind: "object".to_string(),
            properties: BTreeMap::new(),
            required: Vec::new(),
            additional_properties: Some(false),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FunctionParameter {
    pub name: String,
    pub schema: Value,
    pub description: Option<String>,
    pub required: bool,
    pub default: Option<Value>,
}

impl FunctionParameter {
    pub fn new(name: impl Into<String>, schema: Value) -> Self {
        Self {
            name: name.into(),
            schema,
            description: None,
            required: true,
            default: None,
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }

    pub fn with_default(mut self, default: Value) -> Self {
        self.default = Some(default);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub kind: ToolType,
    pub function: FunctionDefinition,
}

impl From<FunctionDefinition> for Tool {
    fn from(function: FunctionDefinition) -> Self {
        Self {
            kind: ToolType::Function,
            function,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    Function,
}

#[derive(Debug, Clone)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: Value,
    pub raw_arguments: Option<String>,
}

impl FunctionCall {
    pub fn new(name: impl Into<String>, arguments: Value) -> Self {
        Self {
            name: name.into(),
            arguments,
            raw_arguments: None,
        }
    }

    pub fn with_raw_arguments(mut self, raw: impl Into<String>) -> Self {
        self.raw_arguments = Some(raw.into());
        self
    }
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: Option<String>,
    pub kind: ToolCallType,
    pub function: FunctionCall,
}

impl ToolCall {
    pub fn new(function: FunctionCall) -> Self {
        Self {
            id: None,
            kind: ToolCallType::Function,
            function,
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ToolCallType {
    Function,
}

impl Serialize for ToolCall {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("ToolCall", 3)?;
        if let Some(id) = &self.id {
            state.serialize_field("id", id)?;
        }
        state.serialize_field("type", &self.kind)?;
        state.serialize_field("function", &SerializableFunctionCall(&self.function))?;
        state.end()
    }
}

impl Serialize for ToolCallType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            ToolCallType::Function => serializer.serialize_str("function"),
        }
    }
}

impl<'de> Deserialize<'de> for ToolCall {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawFunctionCall {
            name: String,
            arguments: String,
        }

        #[derive(Deserialize)]
        struct RawToolCall {
            id: Option<String>,
            #[serde(rename = "type")]
            kind: String,
            function: RawFunctionCall,
        }

        let raw = RawToolCall::deserialize(deserializer)?;
        let kind = match raw.kind.as_str() {
            "function" => ToolCallType::Function,
            other => {
                return Err(serde::de::Error::custom(format!(
                    "unsupported tool call type '{other}'"
                )))
            }
        };

        let arguments: Value = serde_json::from_str(&raw.function.arguments)
            .map_err(|error| serde::de::Error::custom(format!(
                "failed to parse function arguments: {error}"
            )))?;

        Ok(Self {
            id: raw.id,
            kind,
            function: FunctionCall {
                name: raw.function.name,
                arguments,
                raw_arguments: Some(raw.function.arguments),
            },
        })
    }
}

impl<'de> Deserialize<'de> for ToolCallType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        match value.as_str() {
            "function" => Ok(ToolCallType::Function),
            other => Err(serde::de::Error::custom(format!(
                "unsupported tool call type '{other}'"
            ))),
        }
    }
}

struct SerializableFunctionCall<'a>(&'a FunctionCall);

impl<'a> Serialize for SerializableFunctionCall<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("function", 2)?;
        state.serialize_field("name", &self.0.name)?;
        let raw = if let Some(raw) = &self.0.raw_arguments {
            raw.clone()
        } else {
            serde_json::to_string(&self.0.arguments)
                .map_err(|error| serde::ser::Error::custom(error.to_string()))?
        };
        state.serialize_field("arguments", &raw)?;
        state.end()
    }
}

#[async_trait]
pub trait KernelFunction: Send + Sync {
    fn definition(&self) -> FunctionDefinition;

    async fn invoke(&self, arguments: &Value) -> Result<Value, LLMError>;
}

pub type DynKernelFunction = Arc<dyn KernelFunction>;

#[derive(Default)]
pub struct FunctionRegistry {
    functions: BTreeMap<String, DynKernelFunction>,
    cached_definitions: std::sync::Mutex<Option<Vec<FunctionDefinition>>>,
    cached_tools: std::sync::Mutex<Option<Vec<Tool>>>,
}

impl FunctionRegistry {
    pub fn new() -> Self {
        Self {
            functions: BTreeMap::new(),
            cached_definitions: std::sync::Mutex::new(None),
            cached_tools: std::sync::Mutex::new(None),
        }
    }

    pub fn register(&mut self, function: DynKernelFunction) {
        let name = function.definition().name;
        self.functions.insert(name, function);
        self.invalidate_cache();
    }

    pub fn register_all<I>(&mut self, functions: I)
    where
        I: IntoIterator<Item = DynKernelFunction>,
    {
        for function in functions {
            self.register(function);
        }
    }

    pub fn extend_from(&mut self, other: &FunctionRegistry) {
        for (name, func) in &other.functions {
            self.functions.insert(name.clone(), func.clone());
        }
        self.invalidate_cache();
    }

    fn invalidate_cache(&mut self) {
        *self.cached_definitions.lock().unwrap() = None;
        *self.cached_tools.lock().unwrap() = None;
    }

    pub fn get(&self, name: &str) -> Option<&DynKernelFunction> {
        self.functions.get(name)
    }

    pub fn definitions(&self) -> Vec<FunctionDefinition> {
        let mut cache = self.cached_definitions.lock().unwrap();
        if let Some(ref defs) = *cache {
            return defs.clone();
        }
        let defs: Vec<FunctionDefinition> = self.functions
            .values()
            .map(|function| function.definition())
            .collect();
        *cache = Some(defs.clone());
        defs
    }

    pub fn tools(&self) -> Vec<Tool> {
        let mut cache = self.cached_tools.lock().unwrap();
        if let Some(ref tools) = *cache {
            return tools.clone();
        }
        let tools: Vec<Tool> = self.definitions()
            .into_iter()
            .map(|definition| definition.into())
            .collect();
        *cache = Some(tools.clone());
        tools
    }

    pub async fn invoke(&self, call: &FunctionCall) -> Result<Value, LLMError> {
        let function = self
            .get(&call.name)
            .ok_or_else(|| LLMError::UnknownFunction(call.name.clone()))?;
        function.invoke(&call.arguments).await
    }
}

pub fn json_schema_for<T: JsonSchema>() -> Value {
    let schema = schemars::schema_for!(T);
    serde_json::to_value(schema.schema).expect("schema serialization should not fail")
}

pub fn to_value<T: serde::Serialize>(value: T) -> Value {
    serde_json::to_value(value).expect("value serialization should not fail")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Simple(ToolChoiceSimple),
    Function {
        #[serde(rename = "type")]
        kind: ToolChoiceKind,
        function: ToolChoiceFunction,
    },
}

impl ToolChoice {
    pub fn auto() -> Self {
        Self::Simple(ToolChoiceSimple::Auto)
    }

    pub fn none() -> Self {
        Self::Simple(ToolChoiceSimple::None)
    }

    pub fn required() -> Self {
        Self::Simple(ToolChoiceSimple::Required)
    }

    pub fn function(name: impl Into<String>) -> Self {
        Self::Function {
            kind: ToolChoiceKind::Function,
            function: ToolChoiceFunction { name: name.into() },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceSimple {
    None,
    Auto,
    Required,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceKind {
    Function,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceFunction {
    pub name: String,
}
