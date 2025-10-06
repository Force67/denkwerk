use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalScenario {
    pub name: String,
    pub seed: u64,
    pub initial_agent: String,
    pub user_input: String,
    pub scripted: Vec<ScriptedTurn>,
    pub expect: ExpectedTrace,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScriptedTurn {
    pub agent: String,
    pub response: String,
    pub latency_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedTrace {
    pub steps: Vec<ExpectStep>,
    pub final_reply_contains: Option<String>,
    pub max_rounds_le: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExpectStep {
    Msg { agent: String, contains: Option<String> },
    HandOff { from: String, to: String, because: DecisionSource },
    Complete { agent: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DecisionSource {
    Rule,
    Tool,
    Parser,
}