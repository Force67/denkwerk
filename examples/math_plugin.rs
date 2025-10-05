use denkwerk::functions::{FunctionCall, FunctionRegistry};
use denkwerk::plugins::math;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut registry = FunctionRegistry::new();
    math::register_math_functions(&mut registry);

    let expressions = [
        "2 * (3 + 4)",
        "sin(pi / 2)",
        "sqrt(16) + 5",
    ];

    for expression in expressions {
        let call = FunctionCall::new("math_evaluate", json!({ "expression": expression }));
        let value = registry.invoke(&call).await?;
        let result = value
            .as_f64()
            .ok_or_else(|| "math plugin returned a non-numeric value".to_string())?;
        println!("{expression} = {result}");
    }

    let agent = math::agent_with_math_tools(
        "MathAssistant",
        "Use the math_evaluate tool whenever a calculation is needed before replying.",
    );
    println!("Created agent '{}' with math tools.", agent.name());

    Ok(())
}
