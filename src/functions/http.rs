use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use reqwest::Method;
use serde::Deserialize;
use serde_json::{json, Value};
use thiserror::Error;

use crate::functions::{FunctionDefinition, FunctionParameter, KernelFunction};
use crate::error::LLMError;

#[derive(Debug, Error)]
pub enum HttpToolError {
    #[error("failed to read http tool spec: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse http tool spec: {0}")]
    Parse(#[from] serde_yaml::Error),
    #[error("invalid http tool spec: {0}")]
    Invalid(String),
    #[error("missing env var {0} for bearer auth")]
    MissingEnv(String),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AuthKind {
    Bearer,
    Header,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AuthSpec {
    #[serde(rename = "type")]
    kind: AuthKind,
    #[serde(default)]
    env: Option<String>,
    #[serde(default)]
    header: Option<String>,
    #[serde(default)]
    value: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ParamSpec {
    #[serde(rename = "type")]
    ty: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    required: Option<bool>,
    #[serde(default)]
    default: Option<Value>,
    #[serde(default)]
    #[serde(rename = "enum")]
    enum_values: Option<Vec<Value>>,
}

impl ParamSpec {
    fn to_schema(&self) -> Value {
        let mut schema = json!({ "type": self.ty });
        if let Some(values) = &self.enum_values {
            if !values.is_empty() {
                schema
                    .as_object_mut()
                    .map(|obj| obj.insert("enum".to_string(), Value::Array(values.clone())));
            }
        }
        schema
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct HttpToolSpec {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    pub method: String,
    pub url: String,
    #[serde(default)]
    pub headers: HashMap<String, String>,
    #[serde(default)]
    pub auth: Option<AuthSpec>,
    #[serde(default)]
    pub query: HashMap<String, ParamSpec>,
    #[serde(default)]
    pub body: HashMap<String, ParamSpec>,
}

pub fn load_http_function(
    base_dir: &Path,
    spec_path: &str,
    fallback_name: &str,
) -> Result<Arc<dyn KernelFunction>, HttpToolError> {
    let mut path = PathBuf::from(spec_path);
    if path.is_relative() {
        path = base_dir.join(path);
    }
    let content = std::fs::read_to_string(&path)?;
    let spec: HttpToolSpec = serde_yaml::from_str(&content)?;
    let name = spec.name.clone().unwrap_or_else(|| fallback_name.to_string());
    Ok(Arc::new(HttpFunction::new(name, spec)))
}

#[derive(Clone)]
pub struct HttpFunction {
    definition: FunctionDefinition,
    spec: HttpToolSpec,
    client: reqwest::Client,
}

impl HttpFunction {
    pub fn new(name: impl Into<String>, spec: HttpToolSpec) -> Self {
        let definition = build_definition(name.into(), &spec);
        Self {
            definition,
            spec,
            client: reqwest::Client::new(),
        }
    }

    fn apply_auth(&self, req: reqwest::RequestBuilder) -> Result<reqwest::RequestBuilder, HttpToolError> {
        if let Some(auth) = &self.spec.auth {
            match auth.kind {
                AuthKind::Bearer => {
                    let env_key = auth
                        .env
                        .clone()
                        .ok_or_else(|| HttpToolError::Invalid("bearer auth requires env key".into()))?;
                    let token = env::var(&env_key).map_err(|_| HttpToolError::MissingEnv(env_key.clone()))?;
                    Ok(req.bearer_auth(token))
                }
                AuthKind::Header => {
                    let header = auth
                        .header
                        .clone()
                        .ok_or_else(|| HttpToolError::Invalid("header auth requires header name".into()))?;
                    let value = auth
                        .value
                        .clone()
                        .ok_or_else(|| HttpToolError::Invalid("header auth requires value".into()))?;
                    Ok(req.header(header, value))
                }
            }
        } else {
            Ok(req)
        }
    }
}

fn build_definition(name: String, spec: &HttpToolSpec) -> FunctionDefinition {
    let mut def = FunctionDefinition::new(name.clone());
    if let Some(desc) = &spec.description {
        def = def.with_description(desc.clone());
    }

    for (param, meta) in spec.query.iter().chain(spec.body.iter()) {
        let mut fp = FunctionParameter::new(param, meta.to_schema());
        if let Some(desc) = &meta.description {
            fp = fp.with_description(desc.clone());
        }
        if !meta.required.unwrap_or(true) {
            fp = fp.optional();
        }
        if let Some(default) = &meta.default {
            fp = fp.with_default(default.clone());
        }
        def.add_parameter(fp);
    }

    def
}

#[async_trait]
impl KernelFunction for HttpFunction {
    fn definition(&self) -> FunctionDefinition {
        self.definition.clone()
    }

    async fn invoke(&self, arguments: &Value) -> Result<Value, LLMError> {
        let mut request = self
            .client
            .request(
                Method::from_bytes(self.spec.method.as_bytes())
                    .map_err(|e| LLMError::InvalidFunctionArguments(e.to_string()))?,
                &self.spec.url,
            );

        // Headers from spec
        for (k, v) in &self.spec.headers {
            request = request.header(k, v);
        }
        request = self
            .apply_auth(request)
            .map_err(|e| LLMError::FunctionExecution { function: self.definition.name.clone(), message: e.to_string() })?;

        let args = arguments
            .as_object()
            .cloned()
            .ok_or_else(|| LLMError::InvalidFunctionArguments("arguments must be an object".into()))?;

        // Query params
        if !self.spec.query.is_empty() {
            let mut pairs = Vec::new();
            for key in self.spec.query.keys() {
                if let Some(value) = args.get(key) {
                    if let Some(s) = value.as_str() {
                        pairs.push((key.as_str(), s.to_string()));
                    } else {
                        pairs.push((key.as_str(), value.to_string()));
                    }
                }
            }
            if !pairs.is_empty() {
                request = request.query(&pairs);
            }
        }

        // Body params
        if !self.spec.body.is_empty() {
            let mut body = serde_json::Map::new();
            for key in self.spec.body.keys() {
                if let Some(value) = args.get(key) {
                    body.insert(key.clone(), value.clone());
                }
            }
            request = request.json(&Value::Object(body));
        }

        let response = request.send().await?;
        let status = response.status();
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|h| h.to_str().ok())
            .unwrap_or_default()
            .to_string();

        if content_type.contains("application/json") {
            let json: Value = response.json().await?;
            Ok(json!({ "status": status.as_u16(), "body": json }))
        } else {
            let text = response.text().await?;
            Ok(json!({ "status": status.as_u16(), "body": text }))
        }
    }
}
