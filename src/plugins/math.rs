use std::sync::Arc;

use meval::Expr;

use crate::functions::{
    json_schema_for,
    FunctionDefinition,
    FunctionParameter,
    FunctionRegistry,
};
use crate::{kernel_function, Agent};

#[kernel_function]
async fn evaluate_expression(expression: String) -> Result<f64, String> {
    let expr = expression.parse::<Expr>().map_err(|err| err.to_string())?;
    expr.eval().map_err(|err| err.to_string())
}

pub fn register_math_functions(registry: &mut FunctionRegistry) {
    let mut function = FunctionDefinition::new("math_evaluate")
        .with_description("Evaluate a mathematical expression using standard operators and parentheses.");

    function.add_parameter(
        FunctionParameter::new("expression", json_schema_for::<String>())
            .with_description("The mathematical expression to evaluate."),
    );

    registry.register(evaluate_expression_kernel());
}

pub fn agent_with_math_tools(name: &str, instructions: &str) -> Agent {
    let mut registry = FunctionRegistry::new();
    register_math_functions(&mut registry);

    Agent::from_string(name, instructions.to_string()).with_function_registry(Arc::new(registry))
}
