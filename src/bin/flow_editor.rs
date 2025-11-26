use std::fs;
use denkwerk::{
    DecisionStrategy, FlowDocument, FlowEdge, FlowNode, FlowNodeBase, FlowNodeKind, NodeLayout,
    NodeOutput,
};
use iced::widget::{
    button, canvas, checkbox, column, container, pick_list, row, scrollable, text, text_input,
    Canvas, Column, Row,
};
use iced::{
    alignment, executor, mouse, Application, Color, Command, Element, Length, Point, Rectangle,
    Renderer, Settings, Subscription, Theme, Vector,
};

const NODE_WIDTH: f32 = 160.0;
const NODE_HEIGHT: f32 = 90.0;
const GRID: f32 = 20.0;

fn main() -> iced::Result {
    if !prepare_display_env() {
        return Ok(());
    }

    FlowEditor::run(Settings {
        antialiasing: true,
        ..Settings::default()
    })
}

fn prepare_display_env() -> bool {
    #[derive(Copy, Clone, Debug)]
    enum Backend {
        Wayland,
        X11,
    }

    let wayland_display = std::env::var("WAYLAND_DISPLAY").ok();
    let x11_display = std::env::var("DISPLAY").ok();
    let backend_hint = std::env::var("DENKWERK_BACKEND")
        .ok()
        .or_else(|| std::env::var("WINIT_UNIX_BACKEND").ok());

    let mut order = Vec::new();
    match backend_hint.as_deref() {
        Some("wayland") => order.push(Backend::Wayland),
        Some("x11") => order.push(Backend::X11),
        _ => {
            if wayland_display.is_some() && x11_display.is_some() {
                order.push(Backend::X11);
                order.push(Backend::Wayland);
                eprintln!(
                    "Both WAYLAND_DISPLAY and DISPLAY are set; defaulting to X11. \
                     Set DENKWERK_BACKEND=wayland to force the Wayland backend."
                );
            } else if wayland_display.is_some() {
                order.push(Backend::Wayland);
            } else if x11_display.is_some() {
                order.push(Backend::X11);
            }
        }
    }

    if order.is_empty() {
        eprintln!(
            "No display found (neither DISPLAY nor WAYLAND_DISPLAY set). \
             Start an X11/Wayland session or run under Xvfb, e.g. \
             `Xvfb :99 -screen 0 1280x720x24 & DISPLAY=:99 cargo run --bin flow_editor`."
        );
        return false;
    }

    let mut errors: Vec<String> = Vec::new();

    for backend in order {
        match backend {
            Backend::Wayland => {
                let display = match wayland_display.as_deref() {
                    Some(value) => value,
                    None => {
                        errors.push("Wayland backend requested but WAYLAND_DISPLAY is not set.".to_string());
                        continue;
                    }
                };

                match check_wayland(display) {
                    Ok(()) => return true,
                    Err(err) => errors.push(err),
                }
            }
            Backend::X11 => {
                let display = match x11_display.as_deref() {
                    Some(value) => value,
                    None => {
                        errors.push("X11 backend selected but DISPLAY is not set.".to_string());
                        continue;
                    }
                };

                match check_x11(display) {
                    Ok(()) => {
                        std::env::remove_var("WAYLAND_DISPLAY");
                        return true;
                    }
                    Err(err) => errors.push(err),
                }
            }
        }
    }

    if errors.is_empty() {
        eprintln!(
            "No usable display backend detected. \
             Start a Wayland compositor or X server, or run under Xvfb, e.g. \
             `Xvfb :99 -screen 0 1280x720x24 & DISPLAY=:99 cargo run --bin flow_editor`."
        );
    } else {
        for err in errors {
            eprintln!("{err}");
        }
    }
    false
}

fn check_wayland(display: &str) -> Result<(), String> {
    let socket = wayland_socket_path(display);
    if !socket.exists() {
        return Err(format!(
            "WAYLAND_DISPLAY is set ({display:?}) but no compositor socket found at {:?}. \
             Start a Wayland compositor or run under Xvfb/X11 (set DENKWERK_BACKEND=x11).",
            socket
        ));
    }

    if !has_wayland_client_lib() {
        return Err(
            "Wayland libraries are missing (libwayland-client). Install them or set DENKWERK_BACKEND=x11."
                .to_string(),
        );
    }

    Ok(())
}

fn check_x11(display: &str) -> Result<(), String> {
    if let Some(socket) = x11_socket_path(display) {
        if !socket.exists() {
            return Err(format!(
                "DISPLAY is set ({display}) but no X11 socket found at {:?}. \
                 Start an X server or run under Xvfb, e.g. \
                 `Xvfb :99 -screen 0 1280x720x24 & DISPLAY=:99 cargo run --bin flow_editor`.",
                socket
            ));
        }
    }

    if !has_x11_client_lib() {
        return Err(
            "X11 libraries are missing (libX11). Install them or set DENKWERK_BACKEND=wayland if you prefer Wayland."
                .to_string(),
        );
    }

    Ok(())
}

fn wayland_socket_path(display: &str) -> std::path::PathBuf {
    let base = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/run/user/1000".to_string());
    std::path::Path::new(&base).join(display)
}

fn x11_socket_path(display: &str) -> Option<std::path::PathBuf> {
    if !display.starts_with(':') {
        return None;
    }

    let display_num = display
        .trim_start_matches(':')
        .split('.')
        .next()
        .and_then(|num| num.parse::<i32>().ok())?;

    Some(std::path::Path::new("/tmp/.X11-unix").join(format!("X{display_num}")))
}

fn has_wayland_client_lib() -> bool {
    #[cfg(target_os = "linux")]
    {
        for lib in ["libwayland-client.so.0", "libwayland-client.so"] {
            if unsafe { libloading::Library::new(lib) }.is_ok() {
                return true;
            }
        }
        false
    }
    #[cfg(not(target_os = "linux"))]
    {
        true
    }
}

fn has_x11_client_lib() -> bool {
    #[cfg(target_os = "linux")]
    {
        for lib in ["libX11.so.6", "libX11.so"] {
            if unsafe { libloading::Library::new(lib) }.is_ok() {
                return true;
            }
        }
        false
    }
    #[cfg(not(target_os = "linux"))]
    {
        true
    }
}

#[derive(Debug)]
struct FlowEditor {
    document: FlowDocument,
    file_path: String,
    status: String,
    selected_flow: usize,
    selected_node: Option<String>,
    edge_output: Option<String>,
    edge_target: Option<String>,
    drag: Option<DragState>,
    new_node_counter: u32,
}

#[derive(Debug, Clone)]
struct DragState {
    node_id: String,
    offset: Vector,
}

#[derive(Debug, Clone, Copy)]
enum NodeTemplate {
    Input,
    Output,
    Agent,
    Decision,
    Tool,
    Merge,
    Parallel,
    Loop,
    Subflow,
}

#[derive(Debug, Clone)]
enum Message {
    FilePathChanged(String),
    LoadFile,
    SaveFile,
    NewDocument,
    SelectFlow(String),
    AddNode(NodeTemplate),
    DeleteSelectedNode,
    StartDrag { node_id: String, offset: Vector },
    DragTo(Point),
    EndDrag,
    UpdateNodeId(String),
    UpdateNodeName(String),
    UpdateNodeDescription(String),
    UpdateAgentId(String),
    UpdatePromptId(String),
    UpdateTools(String),
    UpdateParallelConverge(bool),
    UpdateDecisionStrategy(DecisionStrategy),
    UpdateSubflowId(String),
    UpdateLoopCondition(String),
    UpdateLoopMax(String),
    SelectEdgeOutput(String),
    SelectEdgeTarget(String),
    AddEdge,
    AddOutput,
    UpdateOutputLabel(usize, String),
    UpdateOutputCondition(usize, String),
    RemoveOutput(usize),
}

impl Application for FlowEditor {
    type Executor = executor::Default;
    type Theme = Theme;
    type Flags = ();
    type Message = Message;

    fn new(_flags: ()) -> (Self, Command<Message>) {
        let mut document = default_document();
        ensure_layouts(&mut document);
        (
            FlowEditor {
                document,
                file_path: "examples/flows/sample_flow.yaml".to_string(),
                status: "Ready".to_string(),
                selected_flow: 0,
                selected_node: None,
                edge_output: None,
                edge_target: None,
                drag: None,
                new_node_counter: 0,
            },
            Command::none(),
        )
    }

    fn title(&self) -> String {
        "denkwerk flow editor".to_string()
    }

    fn subscription(&self) -> Subscription<Message> {
        Subscription::none()
    }

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::FilePathChanged(path) => self.file_path = path,
            Message::LoadFile => self.load_file(),
            Message::SaveFile => self.save_file(),
            Message::NewDocument => {
                self.document = default_document();
                ensure_layouts(&mut self.document);
                self.selected_flow = 0;
                self.selected_node = None;
                self.status = "New document created".to_string();
            }
            Message::SelectFlow(flow_id) => {
                if let Some(idx) = self
                    .document
                    .flows
                    .iter()
                    .position(|flow| flow.id == flow_id)
                {
                    self.selected_flow = idx;
                    self.selected_node = None;
                }
            }
            Message::AddNode(template) => self.add_node(template),
            Message::DeleteSelectedNode => self.delete_selected(),
            Message::StartDrag { node_id, offset } => {
                self.selected_node = Some(node_id.clone());
                self.drag = Some(DragState { node_id, offset });
            }
            Message::DragTo(point) => {
                if let Some(drag) = &self.drag {
                    let node_id = drag.node_id.clone();
                    let offset = drag.offset;
                    self.move_node(&node_id, point - offset);
                }
            }
            Message::EndDrag => self.drag = None,
            Message::UpdateNodeId(id) => self.update_node_id(id),
            Message::UpdateNodeName(name) => {
                self.update_node_field(|base| base.name = some(name.clone()))
            }
            Message::UpdateNodeDescription(desc) => {
                self.update_node_field(|base| base.description = some(desc.clone()))
            }
            Message::UpdateAgentId(agent) => {
                if let Some(node) = self.selected_node_mut() {
                    match &mut node.kind {
                        FlowNodeKind::Agent { agent: a, .. } => *a = agent.clone(),
                        FlowNodeKind::Tool { tool } => *tool = agent.clone(),
                        _ => {}
                    }
                }
            }
            Message::UpdatePromptId(prompt) => {
                self.update_prompt_field(prompt);
            }
            Message::UpdateTools(list) => {
                if let Some(node) = self.selected_node_mut() {
                    if let FlowNodeKind::Agent { tools, .. } = &mut node.kind {
                        *tools = parse_csv(&list);
                    }
                }
            }
            Message::UpdateDecisionStrategy(strategy) => {
                if let Some(node) = self.selected_node_mut() {
                    if let FlowNodeKind::Decision { strategy: s, .. } = &mut node.kind {
                        *s = Some(strategy);
                    }
                }
            }
            Message::UpdateParallelConverge(value) => {
                if let Some(node) = self.selected_node_mut() {
                    if let FlowNodeKind::Parallel { converge } = &mut node.kind {
                        *converge = Some(value);
                    }
                }
            }
            Message::UpdateSubflowId(flow) => {
                if let Some(node) = self.selected_node_mut() {
                    if let FlowNodeKind::Subflow { flow: f } = &mut node.kind {
                        *f = flow;
                    }
                }
            }
            Message::UpdateLoopCondition(cond) => {
                if let Some(node) = self.selected_node_mut() {
                    if let FlowNodeKind::Loop { condition, .. } = &mut node.kind {
                        *condition = if cond.is_empty() { None } else { Some(cond) };
                    }
                }
            }
            Message::UpdateLoopMax(value) => {
                if let Ok(parsed) = value.parse::<u32>() {
                    if let Some(node) = self.selected_node_mut() {
                        if let FlowNodeKind::Loop { max_iterations, .. } = &mut node.kind {
                            *max_iterations = parsed;
                        }
                    }
                }
            }
            Message::SelectEdgeOutput(label) => self.edge_output = Some(label),
            Message::SelectEdgeTarget(target) => self.edge_target = Some(target),
            Message::AddEdge => self.add_edge(),
            Message::AddOutput => self.add_output(),
            Message::UpdateOutputLabel(index, label) => {
                self.update_output(index, |output| output.label = label.clone())
            }
            Message::UpdateOutputCondition(index, cond) => {
                self.update_output(index, |output| {
                    output.condition = if cond.is_empty() { None } else { Some(cond.clone()) }
                })
            }
            Message::RemoveOutput(index) => self.remove_output(index),
        }
        Command::none()
    }

    fn view(&self) -> Element<'_, Message> {
        let flows = self
            .document
            .flows
            .iter()
            .map(|f| f.id.clone())
            .collect::<Vec<_>>();
        let current_flow = self.flow();

        let file_controls = column![
            text("File").size(20),
            text_input("path", &self.file_path).on_input(Message::FilePathChanged),
            row![button("Load").on_press(Message::LoadFile), button("Save").on_press(Message::SaveFile)]
                .spacing(8),
            button("New document").on_press(Message::NewDocument)
        ]
        .spacing(8);

        let flow_picker = row![
            text("Flow:"),
            pick_list(flows, Some(current_flow.id.clone()), Message::SelectFlow)
        ]
        .spacing(8);

        let palette = column![
            text("Add node").size(20),
            wrapped_row(vec![
                ("Input", NodeTemplate::Input),
                ("Output", NodeTemplate::Output),
                ("Agent", NodeTemplate::Agent),
                ("Decision", NodeTemplate::Decision),
                ("Tool", NodeTemplate::Tool),
                ("Merge", NodeTemplate::Merge),
                ("Parallel", NodeTemplate::Parallel),
                ("Loop", NodeTemplate::Loop),
                ("Subflow", NodeTemplate::Subflow),
            ])
        ]
        .spacing(8);

        let left_panel = scrollable(column![file_controls, flow_picker, palette].spacing(16))
            .width(Length::Fixed(240.0));

        let canvas_view: Element<Message> = Canvas::new(GraphView {
            flow: current_flow,
            selected: self.selected_node.clone(),
        })
        .width(Length::FillPortion(3))
        .height(Length::Fill)
        .into();

        let inspector = self.inspector_view();
        let yaml_preview = self.yaml_preview();

        let right_panel = scrollable(
            column![inspector, text("YAML preview").size(20), yaml_preview].spacing(16),
        )
        .width(Length::Fixed(360.0));

        let content = row![left_panel, canvas_view, right_panel].spacing(8);

        container(column![content, text(&self.status)]).padding(8).into()
    }
}

impl FlowEditor {
    fn flow(&self) -> &denkwerk::FlowDefinition {
        &self.document.flows[self.selected_flow]
    }

    fn flow_mut(&mut self) -> &mut denkwerk::FlowDefinition {
        &mut self.document.flows[self.selected_flow]
    }

    fn load_file(&mut self) {
        match fs::read_to_string(&self.file_path) {
            Ok(content) => match FlowDocument::from_yaml_str(&content) {
                Ok(mut doc) => {
                    ensure_layouts(&mut doc);
                    self.document = doc;
                    self.selected_flow = 0;
                    self.selected_node = None;
                    self.status = "Loaded flow file".to_string();
                }
                Err(err) => self.status = format!("Parse error: {err}"),
            },
            Err(err) => self.status = format!("Read error: {err}"),
        }
    }

    fn save_file(&mut self) {
        match self.document.to_yaml_string() {
            Ok(yaml) => match fs::write(&self.file_path, yaml) {
                Ok(_) => self.status = "Saved flow file".to_string(),
                Err(err) => self.status = format!("Write error: {err}"),
            },
            Err(err) => self.status = format!("Serialize error: {err}"),
        }
    }

    fn add_node(&mut self, template: NodeTemplate) {
        let id = format!("node_{}", self.new_node_counter);
        self.new_node_counter += 1;
        let (kind, outputs) = default_node(template);
        let node = FlowNode {
            base: FlowNodeBase {
                id: id.clone(),
                name: None,
                description: None,
                inputs: vec![],
                outputs,
                layout: Some(NodeLayout { x: 80.0, y: 80.0 }),
            },
            kind,
        };
        self.flow_mut().nodes.push(node);
        self.selected_node = Some(id);
    }

    fn delete_selected(&mut self) {
        if let Some(id) = self.selected_node.take() {
            let flow = self.flow_mut();
            flow.nodes.retain(|n| n.base.id != id);
            flow.edges.retain(|e| !e.from.starts_with(&id) && e.to != id);
        }
    }

    fn move_node(&mut self, node_id: &str, position: Point) {
        if let Some(node) = self
            .flow_mut()
            .nodes
            .iter_mut()
            .find(|n| n.base.id == node_id)
        {
            node.base.layout = Some(NodeLayout {
                x: position.x.max(0.0),
                y: position.y.max(0.0),
            });
        }
    }

    fn update_node_id(&mut self, new_id: String) {
        if new_id.is_empty() {
            return;
        }
        if let Some(node) = self.selected_node_mut() {
            node.base.id = new_id.clone();
            self.selected_node = Some(new_id);
        }
    }

    fn update_node_field<F>(&mut self, mut update: F)
    where
        F: FnMut(&mut FlowNodeBase),
    {
        if let Some(node) = self.selected_node_mut() {
            update(&mut node.base);
        }
    }

    fn update_prompt_field(&mut self, prompt: String) {
        if let Some(node) = self.selected_node_mut() {
            match &mut node.kind {
                FlowNodeKind::Agent { prompt: p, .. }
                | FlowNodeKind::Decision { prompt: p, .. } => {
                    *p = if prompt.is_empty() { None } else { Some(prompt) };
                }
                _ => {}
            }
        }
    }

    fn add_edge(&mut self) {
        let (from_node, to) = match (&self.selected_node, &self.edge_target) {
            (Some(f), Some(t)) => (f.clone(), t.clone()),
            _ => return,
        };

        let label = self
            .edge_output
            .clone()
            .or_else(|| {
                self.flow()
                    .nodes
                    .iter()
                    .find(|n| n.base.id == from_node)
                    .and_then(|n| n.base.outputs.first().map(|o| o.label.clone()))
            })
            .unwrap_or_else(|| "out".to_string());

        let from = format!("{}/{}", from_node, label);
        self.flow_mut().edges.push(FlowEdge {
            from,
            to,
            condition: None,
            label: None,
        });
        self.status = "Edge added".to_string();
    }

    fn add_output(&mut self) {
        if let Some(node) = self.selected_node_mut() {
            let count = node.base.outputs.len();
            node.base.outputs.push(NodeOutput {
                label: format!("out_{count}"),
                condition: None,
            });
        }
    }

    fn update_output<F>(&mut self, index: usize, mut update: F)
    where
        F: FnMut(&mut NodeOutput),
    {
        if let Some(node) = self.selected_node_mut() {
            if let Some(output) = node.base.outputs.get_mut(index) {
                update(output);
            }
        }
    }

    fn remove_output(&mut self, index: usize) {
        if let Some(node) = self.selected_node_mut() {
            if index < node.base.outputs.len() {
                node.base.outputs.remove(index);
            }
        }
    }

    fn selected_node_mut(&mut self) -> Option<&mut FlowNode> {
        let id = self.selected_node.clone()?;
        self.flow_mut().nodes.iter_mut().find(|n| n.base.id == id)
    }

    fn inspector_view(&self) -> Element<'_, Message> {
        if let Some(selected) = &self.selected_node {
            if let Some(node) = self.flow().nodes.iter().find(|n| n.base.id == *selected) {
                let base = &node.base;
                let outputs = base.outputs.iter().enumerate().fold(
                    column![text("Outputs").size(16)].spacing(6),
                    |col, (idx, output)| {
                        col.push(
                            row![
                                text_input("label", &output.label)
                                    .on_input(move |v| Message::UpdateOutputLabel(idx, v))
                                .width(Length::FillPortion(1)),
                                text_input(
                                    "condition",
                                    output.condition.as_deref().unwrap_or("")
                                )
                                .on_input(move |v| Message::UpdateOutputCondition(idx, v))
                                .width(Length::FillPortion(1)),
                                button("X").on_press(Message::RemoveOutput(idx))
                            ]
                            .spacing(4),
                        )
                    },
                );

                let specific: Element<Message> = match &node.kind {
                    FlowNodeKind::Input {} => column![text("Input node")].into(),
                    FlowNodeKind::Output {} => column![text("Output node")].into(),
                    FlowNodeKind::Agent { agent, prompt, tools, .. } => column![
                        text_input("agent id", agent).on_input(Message::UpdateAgentId),
                        text_input("prompt id (optional)", prompt.as_deref().unwrap_or("")).on_input(Message::UpdatePromptId),
                        text_input("tools (comma separated)", &tools.join(",")).on_input(Message::UpdateTools)
                    ]
                    .spacing(6)
                    .into(),
                    FlowNodeKind::Decision { prompt, strategy } => column![
                        text_input("prompt id (optional)", prompt.as_deref().unwrap_or("")).on_input(Message::UpdatePromptId),
                        pick_list(
                            vec![DecisionStrategy::Llm, DecisionStrategy::Rule],
                            strategy.clone(),
                            Message::UpdateDecisionStrategy
                        )
                    ]
                    .spacing(6)
                    .into(),
                    FlowNodeKind::Tool { tool } => {
                        column![text_input("tool id", tool).on_input(Message::UpdateAgentId)].into()
                    }
                    FlowNodeKind::Merge {} => column![text("Merge node")].into(),
                    FlowNodeKind::Parallel { converge } => column![checkbox(
                        "Converge on completion",
                        converge.unwrap_or(true)
                    )
                    .on_toggle(Message::UpdateParallelConverge)]
                    .into(),
                    FlowNodeKind::Loop { condition, max_iterations } => column![
                        text_input("condition (optional)", condition.as_deref().unwrap_or("")).on_input(Message::UpdateLoopCondition),
                        text_input("max iterations", &max_iterations.to_string()).on_input(Message::UpdateLoopMax)
                    ]
                    .spacing(6)
                    .into(),
                    FlowNodeKind::Subflow { flow } => {
                        column![text_input("subflow id", flow).on_input(Message::UpdateSubflowId)].into()
                    }
                };

                let mut view = column![
                    text(format!("Editing {}", base.id)).size(20),
                    text_input("id", &base.id).on_input(Message::UpdateNodeId),
                    text_input("name (optional)", base.name.as_deref().unwrap_or("")).on_input(Message::UpdateNodeName),
                    text_input(
                        "description (optional)",
                        base.description.as_deref().unwrap_or("")
                    )
                    .on_input(Message::UpdateNodeDescription),
                    outputs,
                    button("Add output").on_press(Message::AddOutput),
                    specific,
                    row![
                        button("Delete node").on_press(Message::DeleteSelectedNode),
                        button("Add edge to selected").on_press(Message::AddEdge)
                    ]
                    .spacing(8)
                ]
                .spacing(8);

                // Edge creation helpers
                let node_outputs = base
                    .outputs
                    .iter()
                    .map(|o| o.label.clone())
                    .collect::<Vec<_>>();
                let node_ids = self
                    .flow()
                    .nodes
                    .iter()
                    .filter(|n| n.base.id != base.id)
                    .map(|n| n.base.id.clone())
                    .collect::<Vec<_>>();

                view = view.push(
                    column![
                        text("Edge creation").size(16),
                        pick_list(node_outputs, self.edge_output.clone(), Message::SelectEdgeOutput)
                            .placeholder("output label"),
                        pick_list(node_ids, self.edge_target.clone(), Message::SelectEdgeTarget)
                            .placeholder("target node"),
                        button("Create edge").on_press(Message::AddEdge),
                    ]
                    .spacing(6),
                );

                return view.into();
            }
        }

        container(text("Select a node to edit").horizontal_alignment(alignment::Horizontal::Center))
            .padding(16)
            .into()
    }

    fn yaml_preview(&self) -> Element<'_, Message> {
        let yaml = match self.document.to_yaml_string() {
            Ok(yaml) => yaml,
            Err(err) => format!("Error: {err}"),
        };
        scrollable(
            text(yaml)
                .size(14)
                .width(Length::Fill),
        )
        .height(Length::Fixed(220.0))
        .into()
    }
}

fn default_document() -> FlowDocument {
    FlowDocument {
        version: "0.1".to_string(),
        metadata: None,
        agents: vec![],
        tools: vec![],
        prompts: vec![],
        flows: vec![denkwerk::FlowDefinition {
            id: "main".to_string(),
            entry: "input".to_string(),
            nodes: vec![FlowNode {
                base: FlowNodeBase {
                    id: "input".to_string(),
                    name: Some("Input".to_string()),
                    description: None,
                    inputs: vec![],
                    outputs: vec![NodeOutput {
                        label: "out".to_string(),
                        condition: None,
                    }],
                    layout: Some(NodeLayout { x: 40.0, y: 80.0 }),
                },
                kind: FlowNodeKind::Input {},
            }],
            edges: vec![],
        }],
    }
}

fn default_node(template: NodeTemplate) -> (FlowNodeKind, Vec<NodeOutput>) {
    match template {
        NodeTemplate::Input => (FlowNodeKind::Input {}, vec![NodeOutput { label: "out".to_string(), condition: None }]),
        NodeTemplate::Output => (FlowNodeKind::Output {}, vec![]),
        NodeTemplate::Agent => (FlowNodeKind::Agent { agent: "agent_id".to_string(), prompt: None, tools: vec![], parameters: None }, vec![NodeOutput { label: "out".to_string(), condition: None }]),
        NodeTemplate::Decision => (FlowNodeKind::Decision { prompt: None, strategy: Some(DecisionStrategy::Llm) }, vec![
            NodeOutput { label: "yes".to_string(), condition: Some("yes".to_string()) },
            NodeOutput { label: "no".to_string(), condition: Some("no".to_string()) },
        ]),
        NodeTemplate::Tool => (FlowNodeKind::Tool { tool: "tool_id".to_string() }, vec![NodeOutput { label: "out".to_string(), condition: None }]),
        NodeTemplate::Merge => (FlowNodeKind::Merge {}, vec![NodeOutput { label: "out".to_string(), condition: None }]),
        NodeTemplate::Parallel => (FlowNodeKind::Parallel { converge: Some(true) }, vec![NodeOutput { label: "out".to_string(), condition: None }]),
        NodeTemplate::Loop => (FlowNodeKind::Loop { max_iterations: 3, condition: None }, vec![NodeOutput { label: "next".to_string(), condition: None }]),
        NodeTemplate::Subflow => (FlowNodeKind::Subflow { flow: "subflow_id".to_string() }, vec![NodeOutput { label: "out".to_string(), condition: None }]),
    }
}

fn parse_csv(input: &str) -> Vec<String> {
    input
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn ensure_layouts(document: &mut FlowDocument) {
    for flow in &mut document.flows {
        let mut x = 40.0;
        let mut y = 80.0;
        for node in &mut flow.nodes {
            if node.base.layout.is_none() {
                node.base.layout = Some(NodeLayout { x, y });
                x += NODE_WIDTH + GRID;
                if x > 900.0 {
                    x = 40.0;
                    y += NODE_HEIGHT + GRID;
                }
            }
        }
    }
}

fn wrapped_row(entries: Vec<(&str, NodeTemplate)>) -> Element<'_, Message> {
    let mut col = Column::new().spacing(6);
    let mut current = Row::new().spacing(6);
    let mut width = 0.0;
    for (label, template) in entries {
        let button = button(label).on_press(Message::AddNode(template));
        width += 70.0;
        if width > 220.0 {
            col = col.push(current);
            current = Row::new().spacing(6);
            width = 70.0;
        }
        current = current.push(button);
    }
    col.push(current).into()
}

struct GraphView<'a> {
    flow: &'a denkwerk::FlowDefinition,
    selected: Option<String>,
}

impl<'a> canvas::Program<Message> for GraphView<'a> {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry> {
        let mut frame = canvas::Frame::new(renderer, bounds.size());

        // Grid
        let grid_color = Color::from_rgba(0.6, 0.6, 0.6, 0.2);
        let mut y = 0.0;
        while y < bounds.height {
            frame.stroke(
                &canvas::Path::line(Point::new(0.0, y), Point::new(bounds.width, y)),
                canvas::Stroke {
                    style: canvas::Style::Solid(grid_color),
                    width: 1.0,
                    ..Default::default()
                },
            );
            y += GRID;
        }
        let mut x = 0.0;
        while x < bounds.width {
            frame.stroke(
                &canvas::Path::line(Point::new(x, 0.0), Point::new(x, bounds.height)),
                canvas::Stroke {
                    style: canvas::Style::Solid(grid_color),
                    width: 1.0,
                    ..Default::default()
                },
            );
            x += GRID;
        }

        // Edges
        for edge in &self.flow.edges {
            let (from, to) = parse_edge_targets(edge);
            if let (Some(from_pos), Some(to_pos)) = (node_center(self.flow, &from), node_center(self.flow, &to)) {
                frame.stroke(
                    &canvas::Path::line(from_pos, to_pos),
                    canvas::Stroke {
                        style: canvas::Style::Solid(Color::from_rgb(0.9, 0.9, 0.9)),
                        width: 2.0,
                        ..Default::default()
                    },
                );
            }
        }

        // Nodes
        for node in &self.flow.nodes {
            let Some(layout) = &node.base.layout else { continue };
            let rect = Rectangle {
                x: layout.x,
                y: layout.y,
                width: NODE_WIDTH,
                height: NODE_HEIGHT,
            };
            let is_selected = self
                .selected
                .as_ref()
                .map(|s| s == &node.base.id)
                .unwrap_or(false);
            let color = if is_selected {
                Color::from_rgb(0.18, 0.35, 0.62)
            } else {
                Color::from_rgb(0.23, 0.23, 0.26)
            };
            frame.fill_rectangle(rect.position(), rect.size(), color);
            frame.stroke(
                &canvas::Path::rectangle(rect.position(), rect.size()),
                canvas::Stroke {
                    width: 2.0,
                    style: canvas::Style::Solid(Color::WHITE),
                    ..Default::default()
                },
            );
            frame.fill_text(canvas::Text {
                content: node.base.id.clone(),
                position: Point::new(rect.x + 8.0, rect.y + 18.0),
                color: Color::WHITE,
                size: iced::Pixels(16.0),
                ..Default::default()
            });
            frame.fill_text(canvas::Text {
                content: format!("{:?}", node.kind),
                position: Point::new(rect.x + 8.0, rect.y + 38.0),
                color: Color::from_rgba(1.0, 1.0, 1.0, 0.7),
                size: iced::Pixels(12.0),
                ..Default::default()
            });
        }

        vec![frame.into_geometry()]
    }

    fn update(
        &self,
        _state: &mut Self::State,
        event: canvas::Event,
        _bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> (iced::event::Status, Option<Message>) {
        match event {
            canvas::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                if let Some(position) = cursor.position() {
                    if let Some((id, offset)) = hit_node(self.flow, position) {
                        return (
                            iced::event::Status::Captured,
                            Some(Message::StartDrag { node_id: id, offset }),
                        );
                    }
                }
            }
            canvas::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => {
                return (iced::event::Status::Captured, Some(Message::EndDrag));
            }
            canvas::Event::Mouse(mouse::Event::CursorMoved { position }) => {
                return (iced::event::Status::Captured, Some(Message::DragTo(position)));
            }
            _ => {}
        }
        (iced::event::Status::Ignored, None)
    }

    fn mouse_interaction(
        &self,
        _state: &Self::State,
        _bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> mouse::Interaction {
        if let Some(position) = cursor.position() {
            if hit_node(self.flow, position).is_some() {
                return mouse::Interaction::Grab;
            }
        }
        mouse::Interaction::default()
    }
}

fn parse_edge_targets(edge: &FlowEdge) -> (String, String) {
    let from = edge
        .from
        .split('/')
        .next()
        .unwrap_or(&edge.from)
        .to_string();
    (from, edge.to.clone())
}

fn node_center(flow: &denkwerk::FlowDefinition, id: &str) -> Option<Point> {
    flow.nodes
        .iter()
        .find(|n| n.base.id == id)
        .and_then(|node| node.base.layout.as_ref())
        .map(|layout| Point::new(layout.x + NODE_WIDTH / 2.0, layout.y + NODE_HEIGHT / 2.0))
}

fn hit_node(flow: &denkwerk::FlowDefinition, position: Point) -> Option<(String, Vector)> {
    for node in &flow.nodes {
        if let Some(layout) = &node.base.layout {
            let rect = Rectangle {
                x: layout.x,
                y: layout.y,
                width: NODE_WIDTH,
                height: NODE_HEIGHT,
            };
            if position.x >= rect.x
                && position.x <= rect.x + rect.width
                && position.y >= rect.y
                && position.y <= rect.y + rect.height
            {
                let offset = Vector::new(position.x - rect.x, position.y - rect.y);
                return Some((node.base.id.clone(), offset));
            }
        }
    }
    None
}

fn some(value: String) -> Option<String> {
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}
