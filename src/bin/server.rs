use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use denkwerk::{
    Flow, FlowDocument, FlowResult, LLMProvider, ModelInfo,
};
use denkwerk::providers::{openai::OpenAI, openrouter::OpenRouter};
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, path::PathBuf, sync::Arc};
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info,server=debug".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let flows_dir = PathBuf::from("examples/flows");
    if !flows_dir.exists() {
        std::fs::create_dir_all(&flows_dir).unwrap();
    }

    let app_state = Arc::new(AppState { flows_dir });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/api/flows", get(list_flows).post(create_flow))
        .route("/api/flows/{id}", get(get_flow).put(update_flow))
        .route("/api/flows/execute", post(execute_flow))
        .route("/api/models", get(list_models))
        .route("/api/models/{id}", get(get_model_info))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(app_state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 3002));
    tracing::info!("listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    tracing::info!("bound to {}", addr);
    axum::serve(listener, app).await.unwrap();
}

struct AppState {
    flows_dir: PathBuf,
}

#[derive(Deserialize)]
struct ExecuteFlowRequest {
    flow: FlowDocument,
    input: String,
    #[serde(default)]
    context: serde_json::Map<String, serde_json::Value>,
}

async fn list_flows(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut flows = Vec::new();
    
    // Simple directory walk
    if let Ok(entries) = std::fs::read_dir(&state.flows_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "yaml" || ext == "yml") {
                if let Ok(content) = std::fs::read_to_string(&path) {
                     if let Ok(doc) = serde_yaml::from_str::<FlowDocument>(&content) {
                         flows.push(doc);
                     }
                }
            }
        }
    }
    
    Json(ApiResponse {
        data: flows,
        message: None,
        success: true,
    })
}

async fn get_flow(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let path_yaml = state.flows_dir.join(format!("{}.yaml", id));
    let path_yml = state.flows_dir.join(format!("{}.yml", id));
    
    let path = if path_yaml.exists() {
        path_yaml
    } else if path_yml.exists() {
        path_yml
    } else {
        return (StatusCode::NOT_FOUND, Json(ApiResponse {
            data: (),
            message: Some("Flow not found".into()),
            success: false,
        })).into_response();
    };

    match std::fs::read_to_string(path) {
        Ok(content) => {
            match serde_yaml::from_str::<FlowDocument>(&content) {
                Ok(doc) => Json(ApiResponse {
                    data: doc,
                    message: None,
                    success: true,
                }).into_response(),
                Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse {
                    data: (),
                    message: Some(format!("Failed to parse flow: {}", e)),
                    success: false,
                })).into_response()
            }
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse {
            data: (),
            message: Some(format!("Failed to read flow: {}", e)),
            success: false,
        })).into_response()
    }
}

async fn create_flow(
    State(state): State<Arc<AppState>>,
    Json(flow): Json<FlowDocument>,
) -> impl IntoResponse {
    let id = flow.flows.first().map(|f| f.id.clone()).unwrap_or_else(|| "unknown".to_string());
    let filename = format!("{}.yaml", id);
    let path = state.flows_dir.join(filename);

    match serde_yaml::to_string(&flow) {
        Ok(yaml) => {
            if let Err(e) = std::fs::write(path, yaml) {
                return (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse {
                    data: (),
                    message: Some(format!("Failed to write flow: {}", e)),
                    success: false,
                })).into_response();
            }
            
            Json(ApiResponse {
                data: serde_json::json!({ "id": id }),
                message: Some("Flow created".into()),
                success: true,
            }).into_response()
        },
        Err(e) => (StatusCode::BAD_REQUEST, Json(ApiResponse {
            data: (),
            message: Some(format!("Failed to serialize flow: {}", e)),
            success: false,
        })).into_response()
    }
}

async fn update_flow(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(flow): Json<FlowDocument>,
) -> impl IntoResponse {
    let filename = format!("{}.yaml", id);
    let path = state.flows_dir.join(filename);

    match serde_yaml::to_string(&flow) {
        Ok(yaml) => {
            if let Err(e) = std::fs::write(path, yaml) {
                 return (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse {
                    data: (),
                    message: Some(format!("Failed to write flow: {}", e)),
                    success: false,
                })).into_response();
            }
            Json(ApiResponse {
                data: flow,
                message: Some("Flow updated".into()),
                success: true,
            }).into_response()
        },
        Err(e) => (StatusCode::BAD_REQUEST, Json(ApiResponse {
            data: (),
            message: Some(format!("Failed to serialize flow: {}", e)),
            success: false,
        })).into_response()
    }
}


async fn execute_flow(
    headers: HeaderMap,
    Json(req): Json<ExecuteFlowRequest>,
) -> impl IntoResponse {
    let yaml_str = match serde_yaml::to_string(&req.flow) {
        Ok(s) => s,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(ApiResponse {
            data: Option::<FlowResult>::None,
            message: Some(format!("Failed to serialize flow to YAML: {}", e)),
            success: false,
        })).into_response(),
    };

    let mut flow = match Flow::from_yaml_string(yaml_str, ".") {
        Ok(f) => f,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(ApiResponse {
            data: Option::<FlowResult>::None,
            message: Some(format!("Failed to build flow: {}", e)),
            success: false,
        })).into_response(),
    };

    if let Some(auth_header) = headers.get("Authorization") {
        if let Ok(auth_str) = auth_header.to_str() {
            if auth_str.starts_with("Bearer ") {
                let token = &auth_str[7..];
                
                let provider: Result<Arc<dyn LLMProvider>, _> = if token.starts_with("sk-or-") {
                    OpenRouter::new(token).map(|p| Arc::new(p) as Arc<dyn LLMProvider>)
                } else {
                    OpenAI::new(token).map(|p| Arc::new(p) as Arc<dyn LLMProvider>)
                };

                if let Ok(provider) = provider {
                    for (k, v) in req.context {
                        flow = flow.with_context_var(k, v);
                    }
                     
                     let runner = flow.with_provider(provider);
                     
                     match runner.run(req.input).await {
                        Ok(result) => return Json(ApiResponse {
                            data: Some(result),
                            message: None,
                            success: true,
                        }).into_response(),
                        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse {
                            data: Option::<FlowResult>::None,
                            message: Some(format!("Execution error: {}", e)),
                            success: false,
                        })).into_response(),
                     }
                }
            }
        }
    }

    for (k, v) in req.context {
        flow = flow.with_context_var(k, v);
    }
    
    match flow.run(req.input).await {
        Ok(result) => Json(ApiResponse {
            data: Some(result),
            message: None,
            success: true,
        }).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse {
            data: Option::<FlowResult>::None,
            message: Some(format!("Execution error: {}", e)),
            success: false,
        })).into_response(),
    }
}

async fn list_models(headers: HeaderMap) -> impl IntoResponse {
    let provider = match provider_from_headers(&headers) {
        Ok(provider) => provider,
        Err(response) => return response,
    };

    match provider.list_models().await {
        Ok(models) => Json(ApiResponse {
            data: models,
            message: None,
            success: true,
        })
        .into_response(),
        Err(e) => (StatusCode::BAD_REQUEST, Json(ApiResponse {
            data: Vec::<ModelInfo>::new(),
            message: Some(format!("Failed to list models: {}", e)),
            success: false,
        }))
        .into_response(),
    }
}

async fn get_model_info(
    headers: HeaderMap,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let provider = match provider_from_headers(&headers) {
        Ok(provider) => provider,
        Err(response) => return response,
    };

    match provider.model_info(&id).await {
        Ok(info) => Json(ApiResponse {
            data: info,
            message: None,
            success: true,
        })
        .into_response(),
        Err(e) => (StatusCode::BAD_REQUEST, Json(ApiResponse {
            data: (),
            message: Some(format!("Failed to fetch model info: {}", e)),
            success: false,
        }))
        .into_response(),
    }
}

fn provider_from_headers(
    headers: &HeaderMap,
) -> Result<Arc<dyn LLMProvider>, Response> {
    let Some(auth_header) = headers.get("Authorization") else {
        return Err((StatusCode::UNAUTHORIZED, Json(ApiResponse {
            data: (),
            message: Some("Missing Authorization header".into()),
            success: false,
        })).into_response());
    };

    let Ok(auth_str) = auth_header.to_str() else {
        return Err((StatusCode::BAD_REQUEST, Json(ApiResponse {
            data: (),
            message: Some("Invalid Authorization header".into()),
            success: false,
        })).into_response());
    };

    if !auth_str.starts_with("Bearer ") {
        return Err((StatusCode::BAD_REQUEST, Json(ApiResponse {
            data: (),
            message: Some("Authorization header must be Bearer token".into()),
            success: false,
        })).into_response());
    }

    let token = &auth_str[7..];

    let provider: Result<Arc<dyn LLMProvider>, _> = if token.starts_with("sk-or-") {
        OpenRouter::new(token).map(|p| Arc::new(p) as Arc<dyn LLMProvider>)
    } else {
        OpenAI::new(token).map(|p| Arc::new(p) as Arc<dyn LLMProvider>)
    };

    provider.map_err(|e| {
        (StatusCode::BAD_REQUEST, Json(ApiResponse {
            data: (),
            message: Some(format!("Failed to create provider: {}", e)),
            success: false,
        }))
        .into_response()
    })
}

#[derive(Serialize)]
struct ApiResponse<T> {
    data: T,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
    success: bool,
}
