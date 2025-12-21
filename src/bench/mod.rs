use std::{
    collections::HashMap,
    fs,
    path::Path,
    sync::Arc,
};

use jsonschema::{Draft, JSONSchema};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    functions::KernelFunction,
    types::ChatMessage,
    CompletionRequest, FunctionDefinition, FunctionRegistry, LLMError, LLMProvider, ToolChoice,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchCase {
    pub id: String,
    #[serde(default)]
    pub description: Option<String>,
    pub prompt: String,
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub max_rounds: Option<usize>,
    #[serde(default)]
    pub tools: Vec<ToolSpec>,
    pub oracle: OracleSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<Value>,
    #[serde(default)]
    pub fixtures: Vec<ToolFixture>,
    #[serde(default)]
    pub default: Option<ToolResultSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFixture {
    pub when: Value,
    pub then: ToolResultSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolResultSpec {
    Ok { value: Value },
    Err { message: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleSpec {
    #[serde(default)]
    pub allowed_tools: Option<Vec<String>>,
    #[serde(default)]
    pub forbidden_tools: Vec<String>,
    #[serde(default)]
    pub required_calls: Vec<ExpectedCall>,
    #[serde(default)]
    pub sequence: SequenceMode,
    #[serde(default)]
    pub max_calls: Option<usize>,
    #[serde(default)]
    pub final_contains: Vec<String>,
    #[serde(default)]
    pub final_not_contains: Vec<String>,
    #[serde(default)]
    pub final_json_schema: Option<Value>,
    #[serde(default)]
    pub weights: Option<ScoreWeights>,
    #[serde(default)]
    pub pass_threshold: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedCall {
    pub name: String,
    #[serde(default)]
    pub arguments: Option<Value>,
    #[serde(default)]
    pub arguments_subset: Option<Value>,
    #[serde(default)]
    pub arguments_schema: Option<Value>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SequenceMode {
    Strict,
    InOrder,
}

impl Default for SequenceMode {
    fn default() -> Self {
        Self::Strict
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolCallRecord {
    pub id: String,
    pub name: String,
    pub arguments: Value,
    pub raw_arguments: Option<String>,
    pub schema_valid: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub schema_errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CaseRunResult {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub pass: bool,
    pub scores: ScoreBreakdown,
    #[serde(default)]
    pub failures: Vec<String>,
    pub final_answer: String,
    pub tool_calls: Vec<ToolCallRecord>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ScoreBreakdown {
    pub validity: f64,
    pub selection: f64,
    pub sequence: f64,
    pub efficiency: f64,
    pub final_answer: f64,
    pub total: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ScoreWeights {
    pub validity: f64,
    pub selection: f64,
    pub sequence: f64,
    pub efficiency: f64,
    pub final_answer: f64,
}

impl Default for ScoreWeights {
    fn default() -> Self {
        Self {
            validity: 0.25,
            selection: 0.25,
            sequence: 0.20,
            efficiency: 0.10,
            final_answer: 0.20,
        }
    }
}

pub fn load_cases(path: impl AsRef<Path>) -> Result<Vec<BenchCase>, std::io::Error> {
    let path = path.as_ref();
    if path.is_dir() {
        let mut cases = Vec::new();
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let p = entry.path();
            let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("");
            let is_case = matches!(ext, "yaml" | "yml" | "json");
            if !is_case {
                continue;
            }
            let bytes = fs::read(&p)?;
            let case: BenchCase = if ext == "json" {
                serde_json::from_slice(&bytes)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
            } else {
                serde_yaml::from_slice(&bytes)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
            };
            cases.push(case);
        }
        cases.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(cases)
    } else {
        let bytes = fs::read(path)?;
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        let case: BenchCase = if ext == "json" {
            serde_json::from_slice(&bytes)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
        } else {
            serde_yaml::from_slice(&bytes)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
        };
        Ok(vec![case])
    }
}

pub fn build_stub_registry(case: &BenchCase) -> FunctionRegistry {
    let mut registry = FunctionRegistry::new();
    for tool in &case.tools {
        registry.register(Arc::new(StubKernelFunction::new(tool.clone())));
    }
    registry
}

pub async fn run_case(
    provider: &(dyn LLMProvider + Send + Sync),
    model: &str,
    global_system_prompt: &str,
    case: &BenchCase,
    default_max_rounds: usize,
) -> Result<CaseRunResult, LLMError> {
    let registry = build_stub_registry(case);
    let validators = build_schema_validators(&registry)?;

    let mut messages = Vec::new();
    let mut system = global_system_prompt.to_string();
    if let Some(extra) = case.system_prompt.as_deref() {
        if !extra.trim().is_empty() {
            system.push_str("\n\n");
            system.push_str(extra);
        }
    }
    messages.push(ChatMessage::system(system));
    messages.push(ChatMessage::user(case.prompt.clone()));

    let max_rounds = case.max_rounds.unwrap_or(default_max_rounds);
    let max_calls = case.oracle.max_calls.unwrap_or(usize::MAX);

    let mut tool_calls: Vec<ToolCallRecord> = Vec::new();
    let mut final_answer = String::new();
    let mut hit_max_calls = false;

    for round in 0..max_rounds {
        let request = CompletionRequest::new(model.to_string(), messages.clone())
            .with_function_registry(&registry)
            .with_tool_choice(ToolChoice::auto());

        let response = provider.complete(request).await?;
        let mut assistant_msg = response.message.clone();

        for (i, call) in assistant_msg.tool_calls.iter_mut().enumerate() {
            if call.id.is_none() {
                call.id = Some(format!("bench_call_{round}_{i}"));
            }
        }

        final_answer = assistant_msg.text().unwrap_or_default().to_string();
        messages.push(assistant_msg.clone());

        if assistant_msg.tool_calls.is_empty() {
            break;
        }

        for call in assistant_msg.tool_calls {
            let id = call.id.clone().unwrap_or_else(|| format!("bench_call_{round}_x"));
            let name = call.function.name.clone();
            let arguments = call.function.arguments.clone();
            let raw_arguments = call.function.raw_arguments.clone();

            let (schema_valid, schema_errors) =
                validate_call(&validators, &name, &arguments).unwrap_or_else(|e| (false, vec![e]));

            tool_calls.push(ToolCallRecord {
                id: id.clone(),
                name: name.clone(),
                arguments: arguments.clone(),
                raw_arguments,
                schema_valid,
                schema_errors,
            });

            if tool_calls.len() > max_calls {
                hit_max_calls = true;
                continue;
            }

            let tool_result = registry.invoke(&call.function).await;
            let tool_content = match tool_result {
                Ok(value) => serde_json::to_string(&value)?,
                Err(err) => serde_json::to_string(&serde_json::json!({
                    "error": err.to_string()
                }))?,
            };
            messages.push(ChatMessage::tool(id, tool_content));
        }

        if hit_max_calls {
            break;
        }
    }

    Ok(score_case(case, final_answer, tool_calls))
}

fn build_schema_validators(
    registry: &FunctionRegistry,
) -> Result<HashMap<String, JSONSchema>, LLMError> {
    let mut map = HashMap::new();
    for def in registry.definitions() {
        let schema = serde_json::to_value(&def.parameters)?;
        let compiled = JSONSchema::options()
            .with_draft(Draft::Draft7)
            .compile(&schema)
            .map_err(|e| LLMError::InvalidFunctionArguments(e.to_string()))?;
        map.insert(def.name, compiled);
    }
    Ok(map)
}

fn validate_call(
    validators: &HashMap<String, JSONSchema>,
    tool_name: &str,
    args: &Value,
) -> Result<(bool, Vec<String>), String> {
    let Some(schema) = validators.get(tool_name) else {
        return Ok((false, vec![format!("unknown tool: {tool_name}")]));
    };
    match schema.validate(args) {
        Ok(()) => Ok((true, Vec::new())),
        Err(errors) => Ok((
            false,
            errors
                .map(|e| e.to_string())
                .collect::<Vec<_>>(),
        )),
    }
}

fn score_case(case: &BenchCase, final_answer: String, tool_calls: Vec<ToolCallRecord>) -> CaseRunResult {
    let mut failures: Vec<String> = Vec::new();

    let max_calls_ok = match case.oracle.max_calls {
        Some(max_calls) if tool_calls.len() > max_calls => {
            failures.push(format!(
                "tool call limit exceeded: max {}, got {}",
                max_calls,
                tool_calls.len()
            ));
            false
        }
        _ => true,
    };

    let validity = if tool_calls.iter().all(|c| c.schema_valid) {
        1.0
    } else {
        failures.push("one or more tool calls failed schema validation".to_string());
        0.0
    };

    let mut allowed_ok = true;
    if let Some(allowed) = &case.oracle.allowed_tools {
        for c in &tool_calls {
            if !allowed.iter().any(|t| t == &c.name) {
                allowed_ok = false;
                failures.push(format!("tool not in allowed set: {}", c.name));
            }
        }
    }

    let mut forbidden_hit = false;
    for c in &tool_calls {
        if case.oracle.forbidden_tools.iter().any(|t| t == &c.name) {
            forbidden_hit = true;
            failures.push(format!("forbidden tool called: {}", c.name));
        }
    }

    let required_ok = required_calls_satisfied(case, &tool_calls, &mut failures);
    let selection = if max_calls_ok && !forbidden_hit && required_ok {
        1.0
    } else {
        0.0
    };
    let selection = if selection == 1.0 && allowed_ok { 1.0 } else { 0.0 };

    let sequence_ok = match case.oracle.sequence {
        SequenceMode::Strict => required_ok,
        SequenceMode::InOrder => sequence_satisfied(case, &tool_calls, &mut failures),
    };
    let sequence = if sequence_ok { 1.0 } else { 0.0 };

    let required_len = case.oracle.required_calls.len().max(1);
    let extra = tool_calls.len().saturating_sub(case.oracle.required_calls.len());
    let efficiency = if extra == 0 {
        1.0
    } else {
        (1.0 - (extra as f64 / required_len as f64)).max(0.0)
    };

    let final_contains_ok = case
        .oracle
        .final_contains
        .iter()
        .all(|needle| final_answer.contains(needle));
    if !final_contains_ok {
        for needle in &case.oracle.final_contains {
            if !final_answer.contains(needle) {
                failures.push(format!("final answer missing required substring: {needle}"));
            }
        }
    }

    let final_not_contains_ok = case
        .oracle
        .final_not_contains
        .iter()
        .all(|needle| !final_answer.contains(needle));
    if !final_not_contains_ok {
        for needle in &case.oracle.final_not_contains {
            if final_answer.contains(needle) {
                failures.push(format!("final answer contains forbidden substring: {needle}"));
            }
        }
    }

    let final_schema_ok = match &case.oracle.final_json_schema {
        None => true,
        Some(schema) => match serde_json::from_str::<Value>(&final_answer) {
            Ok(json) => match JSONSchema::options().with_draft(Draft::Draft7).compile(schema) {
                Ok(compiled) => match compiled.validate(&json) {
                    Ok(()) => true,
                    Err(errors) => {
                        for e in errors.take(5) {
                            failures.push(format!("final json schema violation: {e}"));
                        }
                        false
                    }
                },
                Err(e) => {
                    failures.push(format!("failed to compile final_json_schema: {e}"));
                    false
                }
            },
            Err(e) => {
                failures.push(format!("final answer is not valid JSON: {e}"));
                false
            }
        },
    };

    let final_answer_score = if final_contains_ok && final_not_contains_ok && final_schema_ok {
        1.0
    } else {
        0.0
    };

    let weights = normalize_weights(case.oracle.weights.unwrap_or_default());
    let total = weights.validity * validity
        + weights.selection * selection
        + weights.sequence * sequence
        + weights.efficiency * efficiency
        + weights.final_answer * final_answer_score;

    let pass_threshold = case.oracle.pass_threshold.unwrap_or(0.99);
    let pass = total >= pass_threshold;

    CaseRunResult {
        id: case.id.clone(),
        description: case.description.clone(),
        pass,
        scores: ScoreBreakdown {
            validity,
            selection,
            sequence,
            efficiency,
            final_answer: final_answer_score,
            total,
        },
        failures,
        final_answer,
        tool_calls,
    }
}

fn normalize_weights(mut weights: ScoreWeights) -> ScoreWeights {
    let sum =
        weights.validity + weights.selection + weights.sequence + weights.efficiency + weights.final_answer;
    if sum <= 0.0 {
        return ScoreWeights::default();
    }
    weights.validity /= sum;
    weights.selection /= sum;
    weights.sequence /= sum;
    weights.efficiency /= sum;
    weights.final_answer /= sum;
    weights
}

fn required_calls_satisfied(
    case: &BenchCase,
    tool_calls: &[ToolCallRecord],
    failures: &mut Vec<String>,
) -> bool {
    if case.oracle.required_calls.is_empty() {
        return true;
    }

    match case.oracle.sequence {
        SequenceMode::Strict => {
            if tool_calls.len() != case.oracle.required_calls.len() {
                failures.push(format!(
                    "expected exactly {} tool call(s), got {}",
                    case.oracle.required_calls.len(),
                    tool_calls.len()
                ));
                return false;
            }

            for (i, expected) in case.oracle.required_calls.iter().enumerate() {
                let actual = &tool_calls[i];
                if !call_matches_expected(expected, actual, failures, true) {
                    return false;
                }
            }
            true
        }
        SequenceMode::InOrder => {
            let mut cursor = 0usize;
            for expected in &case.oracle.required_calls {
                let mut found = false;
                while cursor < tool_calls.len() {
                    let actual = &tool_calls[cursor];
                    cursor += 1;
                    if call_matches_expected(expected, actual, failures, false) {
                        found = true;
                        break;
                    }
                }
                if !found {
                    failures.push(format!("missing required call: {}", expected.name));
                    return false;
                }
            }
            true
        }
    }
}

fn sequence_satisfied(case: &BenchCase, tool_calls: &[ToolCallRecord], failures: &mut Vec<String>) -> bool {
    if case.oracle.required_calls.is_empty() {
        return true;
    }
    match case.oracle.sequence {
        SequenceMode::Strict => true,
        SequenceMode::InOrder => {
            let mut last_index: isize = -1;
            for expected in &case.oracle.required_calls {
                let mut found_index: Option<usize> = None;
                for (i, actual) in tool_calls.iter().enumerate() {
                    if call_matches_expected(expected, actual, failures, false) {
                        found_index = Some(i);
                        break;
                    }
                }
                let Some(i) = found_index else {
                    failures.push(format!("missing required call for ordering: {}", expected.name));
                    return false;
                };
                if i as isize <= last_index {
                    failures.push("required calls not in order".to_string());
                    return false;
                }
                last_index = i as isize;
            }
            true
        }
    }
}

fn call_matches_expected(
    expected: &ExpectedCall,
    actual: &ToolCallRecord,
    failures: &mut Vec<String>,
    record_failures: bool,
) -> bool {
    if actual.name != expected.name {
        if record_failures {
            failures.push(format!(
                "tool name mismatch: expected {}, got {}",
                expected.name, actual.name
            ));
        }
        return false;
    }

    let matcher_count = expected.arguments.is_some() as u8
        + expected.arguments_subset.is_some() as u8
        + expected.arguments_schema.is_some() as u8;
    if matcher_count > 1 {
        if record_failures {
            failures.push(format!(
                "invalid expected call matcher for {}: set only one of arguments/arguments_subset/arguments_schema",
                expected.name
            ));
        }
        return false;
    }

    if let Some(exact) = &expected.arguments {
        if &actual.arguments != exact {
            if record_failures {
                failures.push(format!(
                    "arguments mismatch for {}: expected {:?}, got {:?}",
                    expected.name, exact, actual.arguments
                ));
            }
            return false;
        }
    }

    if let Some(subset) = &expected.arguments_subset {
        if !value_is_subset(subset, &actual.arguments) {
            if record_failures {
                failures.push(format!(
                    "arguments subset mismatch for {}: expected subset {:?}, got {:?}",
                    expected.name, subset, actual.arguments
                ));
            }
            return false;
        }
    }

    if let Some(schema) = &expected.arguments_schema {
        match JSONSchema::options().with_draft(Draft::Draft7).compile(schema) {
            Ok(compiled) => {
                if let Err(errors) = compiled.validate(&actual.arguments) {
                    if record_failures {
                        for e in errors.take(5) {
                            failures.push(format!(
                                "expected args schema violation for {}: {e}",
                                expected.name
                            ));
                        }
                    }
                    return false;
                }
            }
            Err(e) => {
                if record_failures {
                    failures.push(format!(
                        "failed to compile expected arguments_schema for {}: {e}",
                        expected.name
                    ));
                }
                return false;
            }
        }
    }

    true
}

#[derive(Clone)]
struct StubKernelFunction {
    definition: FunctionDefinition,
    fixtures: Vec<ToolFixture>,
    default: Option<ToolResultSpec>,
}

impl StubKernelFunction {
    fn new(tool: ToolSpec) -> Self {
        let mut def = FunctionDefinition::new(tool.name.clone());
        if let Some(desc) = tool.description.clone() {
            def = def.with_description(desc);
        }
        if let Some(params) = tool.parameters.clone() {
            def.parameters = serde_json::from_value(normalize_parameters(params))
                .unwrap_or_else(|_| def.parameters);
        }

        Self {
            definition: def,
            fixtures: tool.fixtures.clone(),
            default: tool.default.clone(),
        }
    }

    fn match_fixture<'a>(&'a self, args: &Value) -> Option<&'a ToolResultSpec> {
        for fixture in &self.fixtures {
            if value_is_subset(&fixture.when, args) {
                return Some(&fixture.then);
            }
        }
        self.default.as_ref()
    }
}

#[async_trait::async_trait]
impl KernelFunction for StubKernelFunction {
    fn definition(&self) -> FunctionDefinition {
        self.definition.clone()
    }

    async fn invoke(&self, arguments: &Value) -> Result<Value, LLMError> {
        let Some(result) = self.match_fixture(arguments) else {
            return Err(LLMError::InvalidFunctionArguments(
                "no matching fixture and no default specified".to_string(),
            ));
        };

        match result {
            ToolResultSpec::Ok { value } => Ok(value.clone()),
            ToolResultSpec::Err { message } => Err(LLMError::FunctionExecution {
                function: self.definition.name.clone(),
                message: message.clone(),
            }),
        }
    }
}

fn value_is_subset(expected: &Value, actual: &Value) -> bool {
    let (Value::Object(expected), Value::Object(actual)) = (expected, actual) else {
        return expected == actual;
    };
    expected.iter().all(|(k, v)| actual.get(k).is_some_and(|av| value_is_subset(v, av)))
}

fn normalize_parameters(mut value: Value) -> Value {
    let Value::Object(map) = &mut value else {
        return value;
    };
    if map.contains_key("additionalProperties") && !map.contains_key("additional_properties") {
        if let Some(v) = map.remove("additionalProperties") {
            map.insert("additional_properties".to_string(), v);
        }
    }
    value
}
