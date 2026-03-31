//! MCP (Model Context Protocol) server for AI-powered level generation.
//!
//! Exposes the Euca Engine's [`GenerationService`] and level pipeline as MCP
//! tools that Claude Code can invoke during development conversations.
//!
//! Communicates via JSON-RPC 2.0 over stdio (stdin/stdout).
//!
//! # Tools exposed
//!
//! - `generate_heightmap` — Generate a terrain heightmap PNG from a text prompt.
//! - `generate_skybox` — Generate a 360° skybox from a text prompt.
//! - `generate_prop` — Generate a 3D model GLB from a text prompt.
//! - `generate_scene` — Generate a full 3D scene GLB via World Labs Marble.
//! - `list_providers` — List available AI generation providers.
//! - `import_tiled` — Import a Tiled JSON map into LevelData format.
//! - `import_ldtk` — Import an LDtk JSON file into LevelData format.

use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use euca_asset::ai_gen::service::GenerationService;
use euca_asset::ai_gen::{GenerationKind, GenerationRequest, GenerationStatus, Quality};
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------------------
// JSON-RPC types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Serialize)]
struct JsonRpcResponse {
    jsonrpc: &'static str,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

// ---------------------------------------------------------------------------
// MCP protocol types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct McpToolInfo {
    name: String,
    description: String,
    #[serde(rename = "inputSchema")]
    input_schema: Value,
}

// ---------------------------------------------------------------------------
// Tool handlers
// ---------------------------------------------------------------------------

struct McpServer {
    service: GenerationService,
}

impl McpServer {
    fn new(output_dir: PathBuf) -> Self {
        Self {
            service: GenerationService::new(output_dir),
        }
    }

    fn handle_request(&mut self, req: &JsonRpcRequest) -> Value {
        match req.method.as_str() {
            "initialize" => self.handle_initialize(),
            "tools/list" => self.handle_tools_list(),
            "tools/call" => self.handle_tools_call(&req.params),
            "notifications/initialized" | "notifications/cancelled" => Value::Null,
            _ => serde_json::json!({
                "error": format!("unknown method: {}", req.method)
            }),
        }
    }

    fn handle_initialize(&self) -> Value {
        serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "euca-level-mcp",
                "version": env!("CARGO_PKG_VERSION")
            }
        })
    }

    fn handle_tools_list(&self) -> Value {
        let tools = vec![
            McpToolInfo {
                name: "generate_heightmap".into(),
                description: "Generate a terrain heightmap PNG from a text prompt via Stability AI".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "prompt": { "type": "string", "description": "Text description of the terrain" },
                        "output_path": { "type": "string", "description": "File path to save the heightmap PNG" }
                    },
                    "required": ["prompt"]
                }),
            },
            McpToolInfo {
                name: "generate_skybox".into(),
                description: "Generate a 360° panoramic skybox from a text prompt via Blockade Labs".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "prompt": { "type": "string", "description": "Text description of the sky/environment" },
                        "output_path": { "type": "string", "description": "File path to save the skybox" }
                    },
                    "required": ["prompt"]
                }),
            },
            McpToolInfo {
                name: "generate_prop".into(),
                description: "Generate a 3D model GLB from a text prompt via Meshy/Tripo".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "prompt": { "type": "string", "description": "Text description of the 3D model" },
                        "provider": { "type": "string", "description": "Provider name (tripo, meshy, rodin, hunyuan)", "default": "tripo" },
                        "quality": { "type": "string", "enum": ["low", "medium", "high"], "default": "medium" },
                        "output_path": { "type": "string", "description": "File path to save the GLB" }
                    },
                    "required": ["prompt"]
                }),
            },
            McpToolInfo {
                name: "generate_scene".into(),
                description: "Generate a full 3D scene GLB via World Labs Marble (room-scale)".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "prompt": { "type": "string", "description": "Text description of the scene" },
                        "output_path": { "type": "string", "description": "File path to save the scene GLB" }
                    },
                    "required": ["prompt"]
                }),
            },
            McpToolInfo {
                name: "list_providers".into(),
                description: "List all registered AI generation providers and their availability".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            },
            McpToolInfo {
                name: "import_tiled".into(),
                description: "Import a Tiled JSON map file and convert to LevelData JSON".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Path to the Tiled JSON file" },
                        "cell_size": { "type": "number", "description": "World-space cell size", "default": 1.0 },
                        "output_path": { "type": "string", "description": "Path to save the LevelData JSON" }
                    },
                    "required": ["path"]
                }),
            },
            McpToolInfo {
                name: "import_ldtk".into(),
                description: "Import an LDtk JSON file and convert to LevelData JSON".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Path to the LDtk JSON file" },
                        "cell_size": { "type": "number", "description": "World-space cell size", "default": 1.0 },
                        "output_path": { "type": "string", "description": "Path to save the LevelData JSON" }
                    },
                    "required": ["path"]
                }),
            },
        ];

        serde_json::json!({ "tools": tools })
    }

    fn handle_tools_call(&mut self, params: &Value) -> Value {
        let tool_name = params["name"].as_str().unwrap_or("");
        let args = &params["arguments"];

        let result = match tool_name {
            "generate_heightmap" => self.tool_generate(args, "stability", GenerationKind::Heightmap),
            "generate_skybox" => self.tool_generate(args, "blockade_labs", GenerationKind::Skybox),
            "generate_prop" => {
                let provider = args["provider"].as_str().unwrap_or("tripo");
                self.tool_generate(args, provider, GenerationKind::Model3D)
            }
            "generate_scene" => self.tool_generate(args, "world_labs", GenerationKind::Scene),
            "list_providers" => self.tool_list_providers(),
            "import_tiled" => self.tool_import_tiled(args),
            "import_ldtk" => self.tool_import_ldtk(args),
            _ => Err(format!("unknown tool: {tool_name}")),
        };

        match result {
            Ok(content) => serde_json::json!({
                "content": [{ "type": "text", "text": content }]
            }),
            Err(e) => serde_json::json!({
                "content": [{ "type": "text", "text": format!("Error: {e}") }],
                "isError": true
            }),
        }
    }

    fn tool_generate(
        &mut self,
        args: &Value,
        default_provider: &str,
        kind: GenerationKind,
    ) -> Result<String, String> {
        let prompt = args["prompt"]
            .as_str()
            .ok_or("missing 'prompt' argument")?;

        let quality = match args["quality"].as_str().unwrap_or("medium") {
            "low" => Quality::Low,
            "high" => Quality::High,
            _ => Quality::Medium,
        };

        let req = GenerationRequest {
            prompt: Some(prompt.to_owned()),
            quality,
            kind,
            ..Default::default()
        };

        let task_id = self
            .service
            .start(default_provider, &req)
            .map_err(|e| e.to_string())?;

        // Poll until complete (blocking — MCP tools are expected to return
        // results, not start async work).
        loop {
            let status = self
                .service
                .update(&task_id)
                .map_err(|e| e.to_string())?;

            match status {
                GenerationStatus::Complete { .. } => {
                    let path = self
                        .service
                        .file_path(&task_id)
                        .ok_or("file path not available after completion")?;

                    // If output_path is specified, copy the file there.
                    if let Some(out) = args["output_path"].as_str() {
                        std::fs::copy(path, out).map_err(|e| e.to_string())?;
                        return Ok(format!("Generated and saved to: {out}"));
                    }

                    return Ok(format!("Generated: {}", path.display()));
                }
                GenerationStatus::Failed { error } => {
                    return Err(format!("Generation failed: {error}"));
                }
                GenerationStatus::Pending { progress } => {
                    log::debug!("task {task_id}: {:.0}% complete", progress * 100.0);
                    std::thread::sleep(std::time::Duration::from_secs(2));
                }
            }
        }
    }

    fn tool_list_providers(&self) -> Result<String, String> {
        let registered = self.service.registered_providers();
        let available = self.service.available_providers();

        let mut lines = Vec::new();
        for name in &registered {
            let status = if available.contains(name) {
                "available (API key configured)"
            } else {
                "unavailable (missing API key)"
            };
            lines.push(format!("- {name}: {status}"));
        }

        Ok(lines.join("\n"))
    }

    fn tool_import_tiled(&self, args: &Value) -> Result<String, String> {
        let path = args["path"]
            .as_str()
            .ok_or("missing 'path' argument")?;
        let cell_size = args["cell_size"].as_f64().unwrap_or(1.0) as f32;

        let level = euca_terrain::tiled_import::load_tiled_json(
            &PathBuf::from(path),
            cell_size,
        )
        .map_err(|e| e.to_string())?;

        let json = level.to_json().map_err(|e| e.to_string())?;

        if let Some(out) = args["output_path"].as_str() {
            std::fs::write(out, &json).map_err(|e| e.to_string())?;
            return Ok(format!(
                "Imported Tiled map ({}x{}, {} entities) → {out}",
                level.width,
                level.height,
                level.entities.len()
            ));
        }

        Ok(json)
    }

    fn tool_import_ldtk(&self, args: &Value) -> Result<String, String> {
        let path = args["path"]
            .as_str()
            .ok_or("missing 'path' argument")?;
        let cell_size = args["cell_size"].as_f64().unwrap_or(1.0) as f32;

        let level = euca_terrain::ldtk_import::load_ldtk_json(
            &PathBuf::from(path),
            cell_size,
        )
        .map_err(|e| e.to_string())?;

        let json = level.to_json().map_err(|e| e.to_string())?;

        if let Some(out) = args["output_path"].as_str() {
            std::fs::write(out, &json).map_err(|e| e.to_string())?;
            return Ok(format!(
                "Imported LDtk level ({}x{}, {} entities) → {out}",
                level.width,
                level.height,
                level.entities.len()
            ));
        }

        Ok(json)
    }
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .target(env_logger::Target::Stderr) // MCP uses stdout for protocol
        .init();

    let output_dir = std::env::var("EUCA_GEN_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let dir = std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join("generated_assets");
            dir
        });

    log::info!("euca-level-mcp starting (output_dir: {})", output_dir.display());

    let mut server = McpServer::new(output_dir);
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                log::error!("stdin read error: {e}");
                break;
            }
        };

        if line.trim().is_empty() {
            continue;
        }

        let req: JsonRpcRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                log::error!("invalid JSON-RPC: {e}");
                let err_resp = JsonRpcResponse {
                    jsonrpc: "2.0",
                    id: Value::Null,
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32700,
                        message: format!("parse error: {e}"),
                    }),
                };
                let _ = writeln!(stdout, "{}", serde_json::to_string(&err_resp).unwrap());
                let _ = stdout.flush();
                continue;
            }
        };

        // Notifications (no id) don't get responses.
        if req.id.is_none() {
            server.handle_request(&req);
            continue;
        }

        let result = server.handle_request(&req);
        let response = if result.is_null() {
            // Method returned null — likely a notification handler.
            JsonRpcResponse {
                jsonrpc: "2.0",
                id: req.id.unwrap_or(Value::Null),
                result: Some(Value::Object(serde_json::Map::new())),
                error: None,
            }
        } else {
            JsonRpcResponse {
                jsonrpc: "2.0",
                id: req.id.unwrap_or(Value::Null),
                result: Some(result),
                error: None,
            }
        };

        let json = serde_json::to_string(&response).unwrap();
        if let Err(e) = writeln!(stdout, "{json}") {
            log::error!("stdout write error: {e}");
            break;
        }
        let _ = stdout.flush();
    }
}
