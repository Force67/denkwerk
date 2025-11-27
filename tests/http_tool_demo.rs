use std::{collections::HashMap, sync::Arc};

use denkwerk::FlowBuilder;
use denkwerk::functions::KernelFunction;
use denkwerk::kernel_function;

// Integration-style check: loads the http tool spec and executes it against httpbin.org/uuid.
#[tokio::test]
async fn http_tool_executes_and_returns_body() {
    // Load the flow fixture and replace the model placeholder to keep parsing simple.
    let flow_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("prompt_from_yaml_demo")
        .join("flow.yaml");
    let raw = std::fs::read_to_string(&flow_path).expect("flow file");
    let yaml = raw.replace("${AZURE_OPENAI_DEPLOYMENT}", "stub-model");

    let builder = FlowBuilder::from_yaml_str(flow_path.parent().unwrap(), &yaml)
        .expect("builder");

    #[kernel_function(name = "math_add")]
    fn math_add(a: f64, b: f64) -> Result<f64, String> {
        Ok(a + b)
    }

    let mut functions: HashMap<String, Arc<dyn KernelFunction>> = HashMap::new();
    functions.insert("math_add".to_string(), math_add_kernel());
    let registries = builder.build_tool_registries(&functions).expect("registries");

    // Get the http tool registry and invoke the tool function (no params required).
    let reg = registries.get("http_uuid").expect("http_uuid registry");
    let defs = reg.definitions();
    let call = denkwerk::FunctionCall {
        name: defs[0].name.clone(),
        arguments: serde_json::json!({}),
        raw_arguments: None,
    };

    let result = reg.invoke(&call).await.expect("tool invoke");
    let status = result.get("status").and_then(|v| v.as_u64()).unwrap_or(0);
    assert_eq!(status, 200, "expected HTTP 200 from httpbin");
    let uuid = result
        .get("body")
        .and_then(|b| b.get("uuid"))
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    assert!(
        !uuid.is_empty(),
        "body.uuid should be present; got result: {result}"
    );
}
