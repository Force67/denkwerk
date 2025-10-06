use async_trait::async_trait;

use crate::{
    eval::scenario::ScriptedTurn,
    providers::LLMProvider,
    types::{ChatMessage, CompletionRequest, CompletionResponse},
    LLMError,
};

pub struct ScriptedProvider {
    responses: Vec<String>,
    current: usize,
}

impl ScriptedProvider {
    pub fn new() -> Self {
        Self {
            responses: Vec::new(),
            current: 0,
        }
    }

    pub fn from_scripted_turns(turns: &[ScriptedTurn]) -> Self {
        let responses = turns.iter().map(|t| t.response.clone()).collect();
        Self {
            responses,
            current: 0,
        }
    }

    fn next_response(&mut self) -> Option<String> {
        if self.current < self.responses.len() {
            let response = self.responses[self.current].clone();
            self.current += 1;
            Some(response)
        } else {
            None
        }
    }
}

#[async_trait]
impl LLMProvider for ScriptedProvider {
    async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, LLMError> {
        let provider = self as *const Self as *mut Self;
        unsafe {
            if let Some(response) = (*provider).next_response() {
                Ok(CompletionResponse {
                    message: ChatMessage::assistant(response),
                    usage: None,
                    reasoning: None,
                })
            } else {
                Err(LLMError::Provider("no more scripted responses".to_string()))
            }
        }
    }

    fn name(&self) -> &'static str {
        "scripted"
    }
}