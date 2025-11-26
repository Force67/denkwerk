import React, { useEffect, useMemo, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  Edge,
  Handle,
  MiniMap,
  Node,
  NodeProps,
  NodeDragHandler,
  NodeMouseHandler,
  OnConnect,
  Position,
} from "reactflow";
import "reactflow/dist/style.css";
import {
  AgentDefinition,
  DecisionStrategy,
  FlowDefinition,
  FlowDocument,
  FlowEdge,
  FlowNode,
  FlowNodeKindType,
  NodeOutput,
  PromptDefinition,
  ToolDefinition,
} from "./types";
import { toYaml } from "./utils/yaml";

const uid = (prefix: string) => `${prefix}_${Math.random().toString(16).slice(2, 6)}`;

type FlowKind = "sequential" | "handoff" | "parallel" | "group";

const flowTypeLabels: Record<FlowKind, string> = {
  sequential: "Sequential",
  handoff: "Handoff",
  parallel: "Parallel/Concurrent",
  group: "Group Chat",
};

const allowedByKind: Record<FlowKind, FlowNodeKindType[]> = {
  sequential: ["input", "agent", "decision", "tool", "subflow", "loop", "output"],
  handoff: ["input", "agent", "decision", "output"],
  parallel: ["input", "agent", "parallel", "merge", "tool", "output"],
  group: ["input", "agent", "decision", "merge", "output"],
};

const createFlowNode = (type: FlowNodeKindType, id?: string): FlowNode => {
  const base = { id: id ?? uid(type), inputs: [], outputs: [{ label: "out" }] };
  switch (type) {
    case "input":
      return { base: { ...base, outputs: [{ label: "out" }] }, kind: { type } };
    case "output":
      return { base: { ...base, inputs: [{ from: "" }], outputs: [] }, kind: { type } };
    case "agent":
      return { base, kind: { type, agent: "" } };
    case "decision":
      return { base: { ...base, outputs: [] }, kind: { type, strategy: "llm" } };
    case "tool":
      return { base, kind: { type, tool: "" } };
    case "merge":
      return { base: { ...base, inputs: [{ from: "" }, { from: "" }], outputs: [{ label: "out" }] }, kind: { type } };
    case "parallel":
      return { base: { ...base, outputs: [{ label: "out" }] }, kind: { type, converge: true } };
    case "loop":
      return { base: { ...base, outputs: [{ label: "out" }] }, kind: { type, max_iterations: 3 } };
    case "subflow":
      return { base, kind: { type, flow: "" } };
  }
};

const starterDoc: FlowDocument = {
  version: "0.1",
  metadata: {
    name: "support_router",
    description: "Route inbound chats to the right path.",
    tags: ["demo", "routing"],
  },
  agents: [
    {
      id: "support_agent",
      model: "openai/gpt-4o",
      name: "Support Agent",
      system_prompt: "prompts/support.md",
      tools: ["search_tool", "ticket_tool"],
      defaults: { temperature: 0.3, max_tokens: 512 },
    },
    {
      id: "router",
      model: "openai/gpt-4o-mini",
      description: "Routes intents based on signals.",
    },
  ],
  tools: [
    { id: "search_tool", kind: "http", spec: "tools/search.json" },
    { id: "ticket_tool", kind: "internal", function: "create_ticket" },
  ],
  prompts: [
    { id: "classify_prompt", file: "prompts/classify.md", description: "Intent classifier" },
    { id: "fallback_prompt", file: "prompts/fallback.md" },
  ],
  flows: [
    {
      id: "main",
      entry: "n_start",
      nodes: [
        { base: { id: "n_start", outputs: [{ label: "out" }] }, kind: { type: "input" } },
        {
          base: {
            id: "n_route",
            name: "Route intent",
            inputs: [{ from: "n_start/out" }],
            outputs: [
              { label: "support", condition: "intent == 'support'" },
              { label: "sales", condition: "intent == 'sales'" },
              { label: "fallback" },
            ],
          },
          kind: { type: "decision", prompt: "classify_prompt", strategy: "llm" },
        },
        {
          base: {
            id: "n_support",
            inputs: [{ from: "n_route/support" }],
            outputs: [{ label: "done" }],
          },
          kind: { type: "agent", agent: "support_agent", tools: ["search_tool"], prompt: "classify_prompt" },
        },
        {
          base: {
            id: "n_sales",
            inputs: [{ from: "n_route/sales" }],
            outputs: [{ label: "done" }],
          },
          kind: { type: "subflow", flow: "sales_flow" },
        },
        {
          base: {
            id: "n_fallback",
            inputs: [{ from: "n_route/fallback" }],
            outputs: [{ label: "done" }],
          },
          kind: { type: "agent", agent: "support_agent", prompt: "fallback_prompt" },
        },
        {
          base: {
            id: "n_merge",
            inputs: [
              { from: "n_support/done" },
              { from: "n_sales/done" },
              { from: "n_fallback/done" },
            ],
            outputs: [{ label: "out" }],
          },
          kind: { type: "merge" },
        },
        { base: { id: "n_output", inputs: [{ from: "n_merge/out" }], outputs: [] }, kind: { type: "output" } },
      ],
      edges: [
        { from: "n_start/out", to: "n_route" },
        { from: "n_route/support", to: "n_support" },
        { from: "n_route/sales", to: "n_sales" },
        { from: "n_route/fallback", to: "n_fallback" },
        { from: "n_support/done", to: "n_merge" },
        { from: "n_sales/done", to: "n_merge" },
        { from: "n_fallback/done", to: "n_merge" },
        { from: "n_merge/out", to: "n_output" },
      ],
    },
  ],
};

const App: React.FC = () => {
  const [flowDoc, setFlowDoc] = useState<FlowDocument>(starterDoc);
  const [activeFlowId, setActiveFlowId] = useState<string>(starterDoc.flows[0]?.id ?? "");
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [flowKinds, setFlowKinds] = useState<Record<string, FlowKind>>(
    Object.fromEntries(starterDoc.flows.map((f) => [f.id, "sequential"])),
  );
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; nodeId: string } | null>(null);

  const activeFlow = useMemo(
    () => flowDoc.flows.find((f) => f.id === activeFlowId) ?? flowDoc.flows[0],
    [flowDoc, activeFlowId],
  );
  const activeFlowKind = activeFlow ? flowKinds[activeFlow.id] ?? "sequential" : "sequential";

  useEffect(() => {
    if (!activeFlow && flowDoc.flows.length === 0) {
      const fallback = {
        id: "flow_1",
        entry: "start",
        nodes: [createFlowNode("input", "start"), createFlowNode("output", "finish")],
        edges: [{ from: "start/out", to: "finish" }],
      } satisfies FlowDefinition;
      setFlowDoc((prev) => ({ ...prev, flows: [fallback] }));
      setActiveFlowId(fallback.id);
      setFlowKinds((prev) => ({ ...prev, [fallback.id]: "sequential" }));
    }
  }, [activeFlow, flowDoc.flows.length]);

  const selectedNode = useMemo(
    () => activeFlow?.nodes.find((n) => n.base.id === selectedNodeId) ?? null,
    [activeFlow, selectedNodeId],
  );

  useEffect(() => {
    if (activeFlow?.nodes.length) {
      setSelectedNodeId(activeFlow.nodes[0].base.id);
    } else {
      setSelectedNodeId(null);
    }
  }, [activeFlow?.id, activeFlow?.nodes.length]);

  useEffect(() => {
    if (activeFlow && !(activeFlow.id in flowKinds)) {
      setFlowKinds((prev) => ({ ...prev, [activeFlow.id]: "sequential" }));
    }
  }, [activeFlow, flowKinds]);

  const updateFlow = (flowId: string, next: FlowDefinition) => {
    setFlowDoc((prev) => ({
      ...prev,
      flows: prev.flows.map((f) => (f.id === flowId ? next : f)),
    }));
  };

  const addNode = (type: FlowNodeKindType) => {
    if (!activeFlow) return;
    const allowed = allowedByKind[activeFlowKind];
    if (!allowed.includes(type)) return;
    const node = createFlowNode(type);
    updateFlow(activeFlow.id, { ...activeFlow, nodes: [...activeFlow.nodes, node] });
    setSelectedNodeId(node.base.id);
  };

  const updateNode = (nodeId: string, nextNode: FlowNode) => {
    if (!activeFlow) return;
    updateFlow(activeFlow.id, {
      ...activeFlow,
      nodes: activeFlow.nodes.map((n) => (n.base.id === nodeId ? nextNode : n)),
    });
  };

  const removeNode = (nodeId: string) => {
    if (!activeFlow) return;
    const remainingNodes = activeFlow.nodes.filter((n) => n.base.id !== nodeId);
    updateFlow(activeFlow.id, {
      ...activeFlow,
      nodes: remainingNodes,
      edges: activeFlow.edges.filter((e) => !e.from.startsWith(nodeId) && !e.to.startsWith(nodeId)),
    });
    setSelectedNodeId(remainingNodes[0]?.base.id ?? null);
    setContextMenu(null);
  };

  const updateEdge = (index: number, nextEdge: FlowEdge) => {
    if (!activeFlow) return;
    const edges = activeFlow.edges.map((edge, i) => (i === index ? nextEdge : edge));
    updateFlow(activeFlow.id, { ...activeFlow, edges });
    setContextMenu(null);
  };

  const removeEdge = (index: number) => {
    if (!activeFlow) return;
    updateFlow(activeFlow.id, { ...activeFlow, edges: activeFlow.edges.filter((_, i) => i !== index) });
    setContextMenu(null);
  };

  const handleConnect: OnConnect = (connection) => {
    if (!activeFlow || !connection.source || !connection.target) return;
    const sourceNode = activeFlow.nodes.find((n) => n.base.id === connection.source);
    const defaultOutput = sourceNode?.base.outputs?.[0]?.label ?? "out";
    const from = `${connection.source}/${connection.sourceHandle ?? defaultOutput}`;
    const to = connection.target;
    updateFlow(activeFlow.id, {
      ...activeFlow,
      edges: [...activeFlow.edges, { from, to }],
    });
  };

  const addFlow = () => {
    const newFlow: FlowDefinition = {
      id: uid("flow"),
      entry: "start",
      nodes: [createFlowNode("input", "start"), createFlowNode("output", "finish")],
      edges: [{ from: "start/out", to: "finish" }],
    };
    setFlowDoc((prev) => ({ ...prev, flows: [...prev.flows, newFlow] }));
    setActiveFlowId(newFlow.id);
    setFlowKinds((prev) => ({ ...prev, [newFlow.id]: "sequential" }));
  };

  const setFlowEntry = (entry: string) => {
    if (!activeFlow) return;
    updateFlow(activeFlow.id, { ...activeFlow, entry });
  };

  const addAgent = () =>
    setFlowDoc((prev) => ({
      ...prev,
      agents: [...prev.agents, { id: uid("agent"), model: "openai/gpt-4o" }],
    }));

  const updateAgent = (index: number, agent: Partial<AgentDefinition>) => {
    setFlowDoc((prev) => {
      const next = [...prev.agents];
      next[index] = { ...next[index], ...agent } as AgentDefinition;
      return { ...prev, agents: next };
    });
  };

  const removeAgent = (index: number) => {
    setFlowDoc((prev) => ({ ...prev, agents: prev.agents.filter((_, i) => i !== index) }));
  };

  const addTool = () =>
    setFlowDoc((prev) => ({ ...prev, tools: [...prev.tools, { id: uid("tool"), kind: "http" }] }));

  const updateTool = (index: number, tool: Partial<ToolDefinition>) => {
    setFlowDoc((prev) => {
      const next = [...prev.tools];
      next[index] = { ...next[index], ...tool } as ToolDefinition;
      return { ...prev, tools: next };
    });
  };

  const removeTool = (index: number) => setFlowDoc((prev) => ({ ...prev, tools: prev.tools.filter((_, i) => i !== index) }));

  const addPrompt = () =>
    setFlowDoc((prev) => ({
      ...prev,
      prompts: [...prev.prompts, { id: uid("prompt"), file: "prompts/new.md" }],
    }));

  const updatePrompt = (index: number, prompt: Partial<PromptDefinition>) => {
    setFlowDoc((prev) => {
      const next = [...prev.prompts];
      next[index] = { ...next[index], ...prompt } as PromptDefinition;
      return { ...prev, prompts: next };
    });
  };

  const removePrompt = (index: number) =>
    setFlowDoc((prev) => ({ ...prev, prompts: prev.prompts.filter((_, i) => i !== index) }));

  const yamlPreview = useMemo(() => toYaml(flowDoc), [flowDoc]);

  const downloadYaml = () => {
    const blob = new Blob([yamlPreview], { type: "text/yaml" });
    const url = URL.createObjectURL(blob);
    const anchor = window.document.createElement("a");
    anchor.href = url;
    anchor.setAttribute("download", `${flowDoc.metadata?.name || "flow"}.yaml`);
    window.document.body.appendChild(anchor);
    anchor.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true, view: window }));
    window.document.body.removeChild(anchor);
    setTimeout(() => URL.revokeObjectURL(url), 50);
    setContextMenu(null);
  };

  if (!activeFlow) return null;

  const allowedTypes = allowedByKind[activeFlowKind];
  const typeButtons: { kind: FlowNodeKindType; label: string }[] = [
    { kind: "agent", label: "Agent" },
    { kind: "decision", label: "Decision" },
    { kind: "tool", label: "Tool" },
    { kind: "merge", label: "Merge" },
    { kind: "parallel", label: "Parallel" },
    { kind: "loop", label: "Loop" },
    { kind: "output", label: "Output" },
    { kind: "subflow", label: "Subflow" },
  ];

  return (
    <div className="viewport">
      <div className="topbar">
        <div className="brand">denkwerk canvas</div>
        <div className="toolbar">
          <select
            className="flow-select"
            value={activeFlowKind}
            onChange={(e) => setFlowKinds((prev) => ({ ...prev, [activeFlow.id]: e.target.value as FlowKind }))}
          >
            {Object.entries(flowTypeLabels).map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
          <div className="divider" />
          {typeButtons.map(
            (btn) =>
              allowedTypes.includes(btn.kind) && (
                <button key={btn.kind} onClick={() => addNode(btn.kind)}>
                  + {btn.label}
                </button>
              ),
          )}
          <button className="ghost" onClick={addFlow}>
            + New flow
          </button>
        </div>
        <div className="toolbar">
          <button className="ghost" onClick={() => setDocument(starterDoc)}>
            Load sample
          </button>
          <button className="ghost" onClick={() => setDocument({ ...starterDoc, flows: [] })}>
            Reset blank
          </button>
          <button onClick={downloadYaml}>Download YAML</button>
        </div>
      </div>

      <div className="canvas-area">
        <div className="flow-wrapper" onClick={() => contextMenu && setContextMenu(null)}>
          <ReactFlow
            nodes={mapNodes(activeFlow.nodes, allowedTypes)}
            edges={mapEdges(activeFlow.edges)}
            fitView
            proOptions={{ hideAttribution: true }}
            nodeTypes={nodeTypes}
            onNodeDragStop={handleDrag(activeFlow, updateFlow)}
            onNodeClick={handleClick(setSelectedNodeId)}
            onNodeContextMenu={(event, node) => {
              event.preventDefault();
              setSelectedNodeId(node.id);
              setContextMenu({ x: event.clientX, y: event.clientY, nodeId: node.id });
            }}
            onConnect={handleConnect}
          >
            <Background color="#1f2a44" gap={24} size={2} />
            <Controls showInteractive={false} />
            <MiniMap
              nodeStrokeColor={() => "#58b1ff"}
              nodeColor={() => "#122038"}
              maskColor="rgba(10,13,22,0.6)"
            />
          </ReactFlow>
          {contextMenu && (
            <div
              className="context-menu"
              style={{ top: contextMenu.y, left: contextMenu.x }}
              onMouseLeave={() => setContextMenu(null)}
            >
              <button
                className="ghost"
                onClick={() =>
                  activeFlow &&
                  quickConnectToNew(
                    contextMenu.nodeId,
                    activeFlowKind,
                    activeFlow,
                    updateFlow,
                    setSelectedNodeId,
                    setContextMenu,
                  )
                }
              >
                New agent + edge
              </button>
              <button
                className="ghost"
                onClick={() =>
                  activeFlow && promptEdge(contextMenu.nodeId, activeFlow, updateFlow, setContextMenu)
                }
              >
                Connect to node…
              </button>
              <button className="ghost" onClick={() => removeNode(contextMenu.nodeId)}>
                Delete node
              </button>
            </div>
          )}
        </div>

        <div className="side-panel">
          <div className="panel-section">
            <div className="panel-header">
              <div>
                <div className="panel-title">Flow</div>
                <small className="muted">{activeFlow.id}</small>
              </div>
              <div className="badge small">entry</div>
            </div>
            <input
              value={activeFlow.entry}
              onChange={(e) => setFlowEntry(e.target.value)}
              className="panel-input"
              placeholder="entry node id"
            />
          </div>

          <div className="panel-section">
            <div className="panel-header">
              <div className="panel-title">Selected node</div>
              {selectedNode && <div className="badge small">{selectedNode.kind.type}</div>}
            </div>
            {selectedNode ? (
              <NodeEditor
                node={selectedNode}
                onChange={(next) => updateNode(selectedNode.base.id, next)}
                onDelete={() => removeNode(selectedNode.base.id)}
              />
            ) : (
              <div className="muted">Select a node on the canvas to edit behaviour.</div>
            )}
          </div>

          <div className="panel-section">
            <div className="panel-header">
              <div className="panel-title">Edges</div>
              <div className="badge small">{activeFlow.edges.length}</div>
            </div>
            <div className="edge-list">
              {activeFlow.edges.map((edge, idx) => (
                <div key={`${edge.from}-${idx}`} className="edge-row">
                  <input
                    value={edge.from}
                    onChange={(e) => updateEdge(idx, { ...edge, from: e.target.value })}
                    className="panel-input"
                    placeholder="from"
                  />
                  <input
                    value={edge.to}
                    onChange={(e) => updateEdge(idx, { ...edge, to: e.target.value })}
                    className="panel-input"
                    placeholder="to"
                  />
                  <button className="ghost" onClick={() => removeEdge(idx)}>
                    ×
                  </button>
                </div>
              ))}
              {activeFlow.edges.length === 0 && <div className="muted">No edges yet.</div>}
            </div>
          </div>

          <div className="panel-section">
            <div className="panel-header">
              <div className="panel-title">Definitions</div>
            </div>
            <DefinitionList
              title="Agents"
              items={flowDoc.agents}
              render={(agent, idx) => (
                <div className="stack">
                  <input
                    className="panel-input"
                    value={agent.id}
                    onChange={(e) => updateAgent(idx, { id: e.target.value })}
                    placeholder="agent id"
                  />
                  <input
                    className="panel-input"
                    value={agent.model}
                    onChange={(e) => updateAgent(idx, { model: e.target.value })}
                    placeholder="model"
                  />
                  <input
                    className="panel-input"
                    value={agent.system_prompt ?? ""}
                    onChange={(e) => updateAgent(idx, { system_prompt: e.target.value })}
                    placeholder="system prompt file"
                  />
                  <button className="ghost" onClick={() => removeAgent(idx)}>
                    Remove
                  </button>
                </div>
              )}
              onAdd={addAgent}
            />

            <DefinitionList
              title="Tools"
              items={flowDoc.tools}
              render={(tool, idx) => (
                <div className="stack">
                  <input
                    className="panel-input"
                    value={tool.id}
                    onChange={(e) => updateTool(idx, { id: e.target.value })}
                    placeholder="tool id"
                  />
                  <input
                    className="panel-input"
                    value={tool.kind}
                    onChange={(e) => updateTool(idx, { kind: e.target.value })}
                    placeholder="kind"
                  />
                  <input
                    className="panel-input"
                    value={tool.spec ?? ""}
                    onChange={(e) => updateTool(idx, { spec: e.target.value })}
                    placeholder="spec file"
                  />
                  <button className="ghost" onClick={() => removeTool(idx)}>
                    Remove
                  </button>
                </div>
              )}
              onAdd={addTool}
            />

            <DefinitionList
              title="Prompts"
              items={flowDoc.prompts}
              render={(prompt, idx) => (
                <div className="stack">
                  <input
                    className="panel-input"
                    value={prompt.id}
                    onChange={(e) => updatePrompt(idx, { id: e.target.value })}
                    placeholder="prompt id"
                  />
                  <input
                    className="panel-input"
                    value={prompt.file}
                    onChange={(e) => updatePrompt(idx, { file: e.target.value })}
                    placeholder="file"
                  />
                  <button className="ghost" onClick={() => removePrompt(idx)}>
                    Remove
                  </button>
                </div>
              )}
              onAdd={addPrompt}
            />
          </div>

          <div className="panel-section">
            <div className="panel-header">
              <div className="panel-title">YAML</div>
            </div>
            <textarea className="yaml-preview" readOnly value={yamlPreview} />
            <div className="edge-row" style={{ gridTemplateColumns: "1fr auto" }}>
              <button className="ghost" onClick={downloadYaml} title="Download .yaml file">
                Download
              </button>
              <button
                className="ghost"
                onClick={async () => {
                  try {
                    await navigator.clipboard.writeText(yamlPreview);
                  } catch (err) {
                    console.error("Copy failed", err);
                  }
                }}
              >
                Copy
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

type FlowNodeData = { label: string; kind: FlowNodeKindType; outputs: NodeOutput[] };

const mapNodes = (nodes: FlowNode[], allowed: FlowNodeKindType[]): Node<FlowNodeData>[] =>
  nodes.map((node, idx) => {
    const { background, border, color, labelPrefix } = styleForKind(node.kind.type, allowed.includes(node.kind.type));
    const label = `${labelPrefix}${node.base.name || node.base.id}`;
    return {
      id: node.base.id,
      position: node.base.layout ?? { x: 120 + (idx % 4) * 220, y: 120 + Math.floor(idx / 4) * 180 },
      data: { label, kind: node.kind.type, outputs: node.base.outputs ?? [] },
      type: "flowNode",
      style: {
        background,
        border,
        color,
        padding: 12,
        borderRadius: 12,
        fontSize: 13,
        boxShadow: "0 10px 24px rgba(0,0,0,0.35)",
        minWidth: 140,
      },
    };
  });

const styleForKind = (
  kind: FlowNodeKindType,
  enabled: boolean,
): { background: string; border: string; color: string; labelPrefix: string } => {
  const disabled = {
    background: "linear-gradient(145deg, #11131c, #0d0f16)",
    border: "1px dashed rgba(255,255,255,0.15)",
    color: "#99a0b8",
    labelPrefix: "✕ ",
  };

  if (!enabled) return disabled;

  switch (kind) {
    case "input":
      return {
        background: "linear-gradient(145deg, #103014, #0d2411)",
        border: "1px solid rgba(74, 222, 128, 0.5)",
        color: "#d6f8e1",
        labelPrefix: "● Start · ",
      };
    case "output":
      return {
        background: "linear-gradient(145deg, #2d1b3a, #201329)",
        border: "1px solid rgba(255, 149, 233, 0.5)",
        color: "#ffe1f7",
        labelPrefix: "◆ End · ",
      };
    case "agent":
      return {
        background: "linear-gradient(145deg, #0f172a, #0c1220)",
        border: "1px solid rgba(88, 177, 255, 0.5)",
        color: "#e6e9f8",
        labelPrefix: "🤖 ",
      };
    case "decision":
      return {
        background: "linear-gradient(145deg, #1e2235, #161a2b)",
        border: "1px solid rgba(255, 211, 105, 0.6)",
        color: "#ffeebe",
        labelPrefix: "✦ ",
      };
    case "tool":
      return {
        background: "linear-gradient(145deg, #162524, #10201e)",
        border: "1px solid rgba(119, 255, 214, 0.45)",
        color: "#dcfff5",
        labelPrefix: "🛠 ",
      };
    case "merge":
      return {
        background: "linear-gradient(145deg, #1d1a2c, #161325)",
        border: "1px solid rgba(168, 140, 255, 0.5)",
        color: "#e8ddff",
        labelPrefix: "⇄ ",
      };
    case "parallel":
      return {
        background: "linear-gradient(145deg, #14202d, #0f1a26)",
        border: "1px solid rgba(102, 217, 255, 0.45)",
        color: "#d8f3ff",
        labelPrefix: "⫴ ",
      };
    case "loop":
      return {
        background: "linear-gradient(145deg, #272016, #1f1a12)",
        border: "1px solid rgba(255, 199, 111, 0.45)",
        color: "#ffe9c3",
        labelPrefix: "⟳ ",
      };
    case "subflow":
      return {
        background: "linear-gradient(145deg, #1a1d2d, #121625)",
        border: "1px solid rgba(138, 212, 255, 0.45)",
        color: "#d8edff",
        labelPrefix: "↳ ",
      };
    default:
      return enabled
        ? {
            background: "linear-gradient(145deg, #0f172a, #0c1220)",
            border: "1px solid rgba(88, 177, 255, 0.4)",
            color: "#e6e9f8",
            labelPrefix: "",
          }
        : disabled;
  }
};

const mapEdges = (edges: FlowEdge[]): Edge[] =>
  edges.map((edge, idx) => {
    const [source, sourceHandle] = edge.from.split("/");
    const [target] = edge.to.split("/");
    return {
      id: `${edge.from}-${edge.to}-${idx}`,
      source,
      target,
      sourceHandle,
      label: edge.label,
      animated: false,
      style: { stroke: "#58b1ff" },
      labelBgPadding: [6, 4],
      labelStyle: { fill: "#dce6ff", fontSize: 12 },
    } as Edge;
  });

const handleDrag = (flow: FlowDefinition, updateFlow: (id: string, next: FlowDefinition) => void): NodeDragHandler =>
  (_evt, node) => {
    updateFlow(flow.id, {
      ...flow,
      nodes: flow.nodes.map((n) =>
        n.base.id === node.id ? { ...n, base: { ...n.base, layout: { ...node.position } } } : n,
      ),
    });
  };

const handleClick = (setSelectedNodeId: (id: string) => void): NodeMouseHandler => (_evt, node) => {
  setSelectedNodeId(node.id);
};

const getDefaultOutput = (flow: FlowDefinition, sourceId: string) => {
  const sourceNode = flow.nodes.find((n) => n.base.id === sourceId);
  return sourceNode?.base.outputs?.[0]?.label ?? "out";
};

const promptEdge = (
  sourceId: string,
  flow: FlowDefinition,
  updateFlow: (id: string, next: FlowDefinition) => void,
  setContextMenu: (v: null) => void,
) => {
  const target = window.prompt("Connect to node id:");
  if (!target) return;
  const from = `${sourceId}/${getDefaultOutput(flow, sourceId)}`;
  const to = target.trim();
  if (!to) return;
  updateFlow(flow.id, { ...flow, edges: [...flow.edges, { from, to }] });
  setContextMenu(null);
};

const quickConnectToNew = (
  sourceId: string,
  flowKind: FlowKind,
  flow: FlowDefinition,
  updateFlow: (id: string, next: FlowDefinition) => void,
  setSelectedNodeId: (id: string) => void,
  setContextMenu: (v: null) => void,
) => {
  if (!allowedByKind[flowKind].includes("agent")) return;
  const newNode = createFlowNode("agent");
  const from = `${sourceId}/${getDefaultOutput(flow, sourceId)}`;
  const nextFlow: FlowDefinition = {
    ...flow,
    nodes: [...flow.nodes, newNode],
    edges: [...flow.edges, { from, to: newNode.base.id }],
  };
  updateFlow(flow.id, nextFlow);
  setSelectedNodeId(newNode.base.id);
  setContextMenu(null);
};

const FlowNodeComponent: React.FC<NodeProps<FlowNodeData>> = ({ data }) => {
  const outputs = data.outputs;
  const baseOffset = 25;
  const spacing = outputs.length > 0 ? 50 / outputs.length : 0;

  return (
    <div>
      <Handle type="target" position={Position.Left} style={{ background: "rgba(255,255,255,0.8)" }} />
      <div>{data.label}</div>
      <div className="muted" style={{ fontSize: 12 }}>{data.kind}</div>
      {outputs.map((out, idx) => (
        <Handle
          key={`${out.label}-${idx}`}
          type="source"
          position={Position.Right}
          id={out.label}
          style={{
            top: baseOffset + idx * spacing,
            background: "rgba(88, 177, 255, 0.9)",
            width: 10,
            height: 10,
          }}
        />
      ))}
    </div>
  );
};

const nodeTypes = { flowNode: FlowNodeComponent };

interface NodeEditorProps {
  node: FlowNode;
  onChange: (node: FlowNode) => void;
  onDelete: () => void;
}

const NodeEditor: React.FC<NodeEditorProps> = ({ node, onChange, onDelete }) => {
  const base = node.base;
  const setOutputs = (outputs: NodeOutput[]) => onChange({ ...node, base: { ...base, outputs } });
  const outputsValue = base.outputs ?? [];

  const setInputs = (inputs: string) =>
    onChange({
      ...node,
      base: {
        ...base,
        inputs: inputs
          .split(/\n|,/)
          .map((x) => x.trim())
          .filter(Boolean)
          .map((from) => ({ from })),
      },
    });

  const updateOutput = (idx: number, next: Partial<NodeOutput>) => {
    const outputs = outputsValue.map((out, i) => (i === idx ? { ...out, ...next } : out));
    setOutputs(outputs);
  };

  const addOutput = () => setOutputs([...outputsValue, { label: "out" }]);
  const removeOutput = (idx: number) => setOutputs(outputsValue.filter((_, i) => i !== idx));

  const renderKindFields = () => {
    switch (node.kind.type) {
      case "agent":
        return (
          <div className="stack">
            <label className="muted">Agent id</label>
            <input
              className="panel-input"
              value={node.kind.agent}
              onChange={(e) => onChange({ ...node, kind: { ...node.kind, agent: e.target.value } })}
              placeholder="support_agent"
            />
            <label className="muted">Prompt</label>
            <input
              className="panel-input"
              value={node.kind.prompt ?? ""}
              onChange={(e) => onChange({ ...node, kind: { ...node.kind, prompt: e.target.value } })}
              placeholder="prompt id"
            />
            <label className="muted">Tools</label>
            <input
              className="panel-input"
              value={(node.kind.tools ?? []).join(", ")}
              onChange={(e) =>
                onChange({
                  ...node,
                  kind: {
                    ...node.kind,
                    tools: e.target.value
                      .split(",")
                      .map((x) => x.trim())
                      .filter(Boolean),
                  },
                })
              }
              placeholder="tool ids"
            />
          </div>
        );
      case "decision":
        return (
          <div className="stack">
            <label className="muted">Prompt</label>
            <input
              className="panel-input"
              value={node.kind.prompt ?? ""}
              onChange={(e) => onChange({ ...node, kind: { ...node.kind, prompt: e.target.value } })}
              placeholder="prompt id"
            />
            <label className="muted">Strategy</label>
            <select
              className="panel-input"
              value={node.kind.strategy ?? "llm"}
              onChange={(e) =>
                onChange({
                  ...node,
                  kind: { ...node.kind, strategy: e.target.value as DecisionStrategy },
                })
              }
            >
              <option value="llm">llm</option>
              <option value="rule">rule</option>
            </select>
          </div>
        );
      case "tool":
        return (
          <div className="stack">
            <label className="muted">Tool id</label>
            <input
              className="panel-input"
              value={node.kind.tool}
              onChange={(e) => onChange({ ...node, kind: { ...node.kind, tool: e.target.value } })}
              placeholder="search_tool"
            />
          </div>
        );
      case "loop":
        return (
          <div className="stack">
            <label className="muted">Max iterations</label>
            <input
              className="panel-input"
              type="number"
              value={node.kind.max_iterations ?? ""}
              onChange={(e) =>
                onChange({
                  ...node,
                  kind: { ...node.kind, max_iterations: e.target.value === "" ? undefined : Number(e.target.value) },
                })
              }
            />
            <label className="muted">Condition</label>
            <input
              className="panel-input"
              value={node.kind.condition ?? ""}
              onChange={(e) => onChange({ ...node, kind: { ...node.kind, condition: e.target.value } })}
              placeholder="e.g. retry < 3"
            />
          </div>
        );
      case "parallel":
        return (
          <label className="muted" style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <input
              type="checkbox"
              checked={node.kind.converge ?? false}
              onChange={(e) => onChange({ ...node, kind: { ...node.kind, converge: e.target.checked } })}
              style={{ width: "auto" }}
            />
            Converge before continuing
          </label>
        );
      case "subflow":
        return (
          <div className="stack">
            <label className="muted">Flow id</label>
            <input
              className="panel-input"
              value={node.kind.flow}
              onChange={(e) => onChange({ ...node, kind: { ...node.kind, flow: e.target.value } })}
              placeholder="child flow id"
            />
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="stack">
      <label className="muted">Node id</label>
      <input
        className="panel-input"
        value={base.id}
        onChange={(e) => onChange({ ...node, base: { ...base, id: e.target.value } })}
      />
      <label className="muted">Name</label>
      <input
        className="panel-input"
        value={base.name ?? ""}
        onChange={(e) => onChange({ ...node, base: { ...base, name: e.target.value } })}
        placeholder="display label"
      />
      <label className="muted">Description</label>
      <input
        className="panel-input"
        value={base.description ?? ""}
        onChange={(e) => onChange({ ...node, base: { ...base, description: e.target.value } })}
        placeholder="optional"
      />
      <label className="muted">Inputs (one per line)</label>
      <textarea
        className="panel-input"
        value={(base.inputs ?? []).map((i) => i.from).join("\n")}
        onChange={(e) => setInputs(e.target.value)}
        placeholder="node/output"
      />

      <div className="muted" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        Outputs
        <button className="ghost" onClick={addOutput}>
          +
        </button>
      </div>
      <div className="stack">
        {outputsValue.map((out, idx) => (
          <div className="edge-row" key={`${out.label}-${idx}`}>
            <input
              className="panel-input"
              value={out.label}
              onChange={(e) => updateOutput(idx, { label: e.target.value })}
              placeholder="label"
            />
            <input
              className="panel-input"
              value={out.condition ?? ""}
              onChange={(e) => updateOutput(idx, { condition: e.target.value })}
              placeholder="condition"
            />
            <button className="ghost" onClick={() => removeOutput(idx)}>
              ×
            </button>
          </div>
        ))}
        {outputsValue.length === 0 && <div className="muted">No outputs defined.</div>}
      </div>

      {renderKindFields()}

      <button className="ghost" onClick={onDelete}>
        Delete node
      </button>
    </div>
  );
};

interface DefinitionListProps<T> {
  title: string;
  items: T[];
  render: (item: T, index: number) => React.ReactNode;
  onAdd: () => void;
}

const DefinitionList = <T,>({ title, items, render, onAdd }: DefinitionListProps<T>) => (
  <div className="defs">
    <div className="panel-header">
      <div className="panel-title">{title}</div>
      <button className="ghost" onClick={onAdd}>
        +
      </button>
    </div>
    <div className="stack">
      {items.map((item, idx) => (
        <div key={idx} className="def-card">
          {render(item, idx)}
        </div>
      ))}
      {items.length === 0 && <div className="muted">No {title.toLowerCase()} defined.</div>}
    </div>
  </div>
);

export default App;
