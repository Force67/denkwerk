use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::RwLock;

use crate::LLMError;

/// Represents a shared state entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedStateEntry {
    /// The stored value
    pub value: Value,
    /// Optional scope name for namespacing
    pub scope: Option<String>,
    /// Timestamp when the state was created
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Optional description of the state
    pub description: Option<String>,
}

impl SharedStateEntry {
    pub fn new(value: Value) -> Self {
        Self {
            value,
            scope: None,
            created_at: chrono::Utc::now(),
            description: None,
        }
    }

    pub fn with_scope(mut self, scope: impl Into<String>) -> Self {
        self.scope = Some(scope.into());
        self
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// Trait for shared state operations within workflows
#[async_trait]
pub trait SharedStateContext: Send + Sync {
    /// Store a value with the given ID and optional scope
    async fn queue_state_update(
        &self,
        id: String,
        value: Value,
        scope: Option<String>,
    ) -> Result<(), LLMError>;

    /// Store a value with the given ID and scope name
    async fn queue_state_scoped(
        &self,
        id: String,
        value: Value,
        scope_name: String,
    ) -> Result<(), LLMError> {
        self.queue_state_update(id, value, Some(scope_name)).await
    }

    /// Read a value by ID, optionally filtered by scope
    async fn read_state(
        &self,
        id: &str,
        scope: Option<&str>,
    ) -> Result<Option<Value>, LLMError>;

    /// Read a value by ID and scope name
    async fn read_state_scoped(
        &self,
        id: &str,
        scope_name: &str,
    ) -> Result<Option<Value>, LLMError> {
        self.read_state(id, Some(scope_name)).await
    }

    /// List all state IDs, optionally filtered by scope
    async fn list_state_ids(&self, scope: Option<&str>) -> Result<Vec<String>, LLMError>;

    /// Remove a state entry by ID, optionally filtered by scope
    async fn remove_state(
        &self,
        id: &str,
        scope: Option<&str>,
    ) -> Result<bool, LLMError>;

    /// Clear all states, optionally filtered by scope
    async fn clear_states(&self, scope: Option<&str>) -> Result<usize, LLMError>;
}

/// In-memory shared state store implementation
#[derive(Debug, Default)]
pub struct InMemorySharedStateStore {
    states: Arc<RwLock<HashMap<String, SharedStateEntry>>>,
}

impl InMemorySharedStateStore {
    pub fn new() -> Self {
        Self::default()
    }

    fn generate_key(&self, id: &str, scope: Option<&str>) -> String {
        match scope {
            Some(scope) => format!("{}:{}", scope, id),
            None => id.to_string(),
        }
    }
}

#[async_trait]
impl SharedStateContext for InMemorySharedStateStore {
    async fn queue_state_update(
        &self,
        id: String,
        value: Value,
        scope: Option<String>,
    ) -> Result<(), LLMError> {
        let key = self.generate_key(&id, scope.as_deref());
        let entry = SharedStateEntry::new(value).with_scope(scope.unwrap_or_default());

        let mut states = self.states.write().await;
        states.insert(key, entry);
        Ok(())
    }

    async fn read_state(
        &self,
        id: &str,
        scope: Option<&str>,
    ) -> Result<Option<Value>, LLMError> {
        let key = self.generate_key(id, scope);
        let states = self.states.read().await;
        Ok(states.get(&key).map(|entry| entry.value.clone()))
    }

    async fn list_state_ids(&self, scope: Option<&str>) -> Result<Vec<String>, LLMError> {
        let states = self.states.read().await;
        let ids: Vec<String> = states
            .keys()
            .filter_map(|key| {
                match scope {
                    Some(scope_filter) => {
                        if key.starts_with(&format!("{}:", scope_filter)) {
                            key.strip_prefix(&format!("{}:", scope_filter))
                        } else if *key == scope_filter {
                            Some(key.as_str())
                        } else {
                            None
                        }
                    }
                    None => Some(key.as_str()),
                }
                .map(|s| s.to_string())
            })
            .collect();
        Ok(ids)
    }

    async fn remove_state(
        &self,
        id: &str,
        scope: Option<&str>,
    ) -> Result<bool, LLMError> {
        let key = self.generate_key(id, scope);
        let mut states = self.states.write().await;
        Ok(states.remove(&key).is_some())
    }

    async fn clear_states(&self, scope: Option<&str>) -> Result<usize, LLMError> {
        let mut states = self.states.write().await;
        match scope {
            Some(scope_filter) => {
                let prefix = format!("{}:", scope_filter);
                let keys_to_remove: Vec<String> = states
                    .keys()
                    .filter(|key| key.starts_with(&prefix) || *key == scope_filter)
                    .cloned()
                    .collect();
                let count = keys_to_remove.len();
                for key in keys_to_remove {
                    states.remove(&key);
                }
                Ok(count)
            }
            None => {
                let count = states.len();
                states.clear();
                Ok(count)
            }
        }
    }
}

/// Convenience extension methods for common shared state operations
pub struct SharedStateExtensions<'a, C: SharedStateContext + ?Sized> {
    context: &'a C,
}

impl<'a, C: SharedStateContext + ?Sized> SharedStateExtensions<'a, C> {
    pub fn new(context: &'a C) -> Self {
        Self { context }
    }

    /// Store a string value
    pub async fn set_string(
        &self,
        id: String,
        value: String,
        scope: Option<String>,
    ) -> Result<(), LLMError> {
        self.context
            .queue_state_update(id, Value::String(value), scope)
            .await
    }

    /// Store a serializable object
    pub async fn set_object<T: Serialize + Send>(
        &self,
        id: String,
        value: &T,
        scope: Option<String>,
    ) -> Result<(), LLMError> {
        let json_value = serde_json::to_value(value)
            .map_err(|e| LLMError::Serialization(e))?;
        self.context.queue_state_update(id, json_value, scope).await
    }

    /// Read a string value
    pub async fn get_string(
        &self,
        id: &str,
        scope: Option<&str>,
    ) -> Result<Option<String>, LLMError> {
        let value = self.context.read_state(id, scope).await?;
        Ok(value.and_then(|v| v.as_str().map(|s| s.to_string())))
    }

    /// Read and deserialize an object
    pub async fn get_object<T: for<'de> Deserialize<'de>>(
        &self,
        id: &str,
        scope: Option<&str>,
    ) -> Result<Option<T>, LLMError> {
        let value = self.context.read_state(id, scope).await?;
        match value {
            Some(v) => {
                let obj: T = serde_json::from_value(v)
                    .map_err(|e| LLMError::Serialization(e))?;
                Ok(Some(obj))
            }
            None => Ok(None),
        }
    }
}

/// Extension trait to add convenience methods to any SharedStateContext
pub trait SharedStateContextExt: SharedStateContext {
    fn extensions(&self) -> SharedStateExtensions<Self>;
}

impl<C: SharedStateContext> SharedStateContextExt for C {
    fn extensions(&self) -> SharedStateExtensions<Self> {
        SharedStateExtensions::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_basic_state_operations() {
        let store = InMemorySharedStateStore::new();

        // Store a value
        store
            .queue_state_update("test_key".to_string(), json!("test_value"), None)
            .await
            .unwrap();

        // Read the value
        let value = store.read_state("test_key", None).await.unwrap();
        assert_eq!(value, Some(json!("test_value")));
    }

    #[tokio::test]
    async fn test_scoped_state_operations() {
        let store = InMemorySharedStateStore::new();

        // Store values in different scopes
        store
            .queue_state_scoped("key1".to_string(), json!("value1"), "scope1".to_string())
            .await
            .unwrap();
        store
            .queue_state_scoped("key1".to_string(), json!("value2"), "scope2".to_string())
            .await
            .unwrap();

        // Read values from different scopes
        let value1 = store.read_state_scoped("key1", "scope1").await.unwrap();
        let value2 = store.read_state_scoped("key1", "scope2").await.unwrap();

        assert_eq!(value1, Some(json!("value1")));
        assert_eq!(value2, Some(json!("value2")));
    }

    #[tokio::test]
    async fn test_convenience_methods() {
        let store = InMemorySharedStateStore::new();
        let ext = store.extensions();

        // Store a string
        ext.set_string("greeting".to_string(), "Hello, World!".to_string(), None).await.unwrap();

        // Read the string
        let greeting = ext.get_string("greeting", None).await.unwrap();
        assert_eq!(greeting, Some("Hello, World!".to_string()));
    }

    #[tokio::test]
    async fn test_list_and_remove() {
        let store = InMemorySharedStateStore::new();

        // Store multiple values
        store
            .queue_state_update("key1".to_string(), json!("value1"), None)
            .await
            .unwrap();
        store
            .queue_state_scoped("key2".to_string(), json!("value2"), "test_scope".to_string())
            .await
            .unwrap();

        // List all IDs
        let all_ids = store.list_state_ids(None).await.unwrap();
        assert_eq!(all_ids.len(), 2);

        // List IDs in specific scope
        let scoped_ids = store.list_state_ids(Some("test_scope")).await.unwrap();
        assert_eq!(scoped_ids, vec!["key2"]);

        // Remove a state
        let removed = store.remove_state("key1", None).await.unwrap();
        assert!(removed);

        let remaining_ids = store.list_state_ids(None).await.unwrap();
        assert_eq!(remaining_ids.len(), 1);
    }
}