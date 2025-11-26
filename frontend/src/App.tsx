import React, { useEffect, useMemo, useState } from "react";
import { AgentDefinition, FlowDefinition, FlowDocument, FlowEdge, FlowNode, FlowNodeKindType, PromptDefinition, ToolDefinition } from "./types";
import { toYaml } from "./utils/yaml";
import { allowedByKind, FlowKind, flowTypeLabels, typeButtons } from "./utils/flowConfig";
import { createFlowNode, uid } from "./utils/flowNodes";
import TopBar from "./components/TopBar";
import CanvasArea from "./components/CanvasArea";
import NodeEditor from "./components/NodeEditor";
import DefinitionList from "./components/DefinitionList";
import EdgesPanel from "./components/EdgesPanel";
import YamlPanel from "./components/YamlPanel";

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

  const activeFlow = useMemo(
    () => flowDoc.flows.find((f) => f.id === activeFlowId) ?? flowDoc.flows[0],
    [flowDoc, activeFlowId],
  );
  const activeFlowKind = activeFlow ? flowKinds[activeFlow.id] ?? "sequential" : "sequential";

  useEffect(() => {
    if (!activeFlow && flowDoc.flows.length === 0) {
      const fallback: FlowDefinition = {
        id: "flow_1",
        entry: "start",
        nodes: [createFlowNode("input", "start"), createFlowNode("output", "finish")],
        edges: [{ from: "start/out", to: "finish" }],
      };
      setFlowDoc((prev) => ({ ...prev, flows: [fallback] }));
      setActiveFlowId(fallback.id);
      setFlowKinds((prev) => ({ ...prev, [fallback.id]: "sequential" }));
    }
  }, [activeFlow, flowDoc.flows.length]);

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
  };

  const updateEdge = (index: number, nextEdge: FlowEdge) => {
    if (!activeFlow) return;
    const edges = activeFlow.edges.map((edge, i) => (i === index ? nextEdge : edge));
    updateFlow(activeFlow.id, { ...activeFlow, edges });
  };

  const removeEdge = (index: number) => {
    if (!activeFlow) return;
    updateFlow(activeFlow.id, { ...activeFlow, edges: activeFlow.edges.filter((_, i) => i !== index) });
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

  const removeTool = (index: number) =>
    setFlowDoc((prev) => ({ ...prev, tools: prev.tools.filter((_, i) => i !== index) }));

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
    if (typeof window === "undefined" || !window.document) return;
    const blob = new Blob([yamlPreview], { type: "text/yaml" });
    const url = URL.createObjectURL(blob);
    const anchor = window.document.createElement("a");
    anchor.href = url;
    anchor.setAttribute("download", `${flowDoc.metadata?.name || "flow"}.yaml`);
    window.document.body.appendChild(anchor);
    anchor.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true, view: window }));
    window.document.body.removeChild(anchor);
    setTimeout(() => URL.revokeObjectURL(url), 50);
  };

  const copyYaml = async () => {
    try {
      await navigator.clipboard.writeText(yamlPreview);
    } catch (err) {
      console.error("Copy failed", err);
    }
  };

  if (!activeFlow) return null;

  const allowedTypes = allowedByKind[activeFlowKind];

  return (
    <div className="viewport">
      <TopBar
        flowKind={activeFlowKind}
        allowedTypes={allowedTypes}
        typeButtons={typeButtons}
        onChangeFlowKind={(kind) => setFlowKinds((prev) => ({ ...prev, [activeFlow.id]: kind }))}
        onAddNode={addNode}
        onAddFlow={addFlow}
        onLoadSample={() => setFlowDoc(starterDoc)}
        onReset={() => setFlowDoc({ ...starterDoc, flows: [] })}
        onDownload={downloadYaml}
      />

      <div className="canvas-area">
        <CanvasArea
          flow={activeFlow}
          flowKind={activeFlowKind}
          allowedTypes={allowedTypes}
          onUpdateFlow={(next) => updateFlow(activeFlow.id, next)}
          onSelectNode={(id) => setSelectedNodeId(id)}
          onRemoveNode={removeNode}
        />

        <div className="side-panel">
          <div className="panel-section">
            <div className="panel-header">
              <div>
                <div className="panel-title">Flow</div>
                <small className="muted">{activeFlow.id}</small>
              </div>
              <div className="badge small">{flowTypeLabels[activeFlowKind]}</div>
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
              {selectedNodeId && <div className="badge small">{selectedNodeId}</div>}
            </div>
            {selectedNodeId ? (
              <NodeEditor
                node={activeFlow.nodes.find((n) => n.base.id === selectedNodeId)!}
                onChange={(next) => updateNode(selectedNodeId, next)}
                onDelete={() => removeNode(selectedNodeId)}
              />
            ) : (
              <div className="muted">Select a node on the canvas to edit behaviour.</div>
            )}
          </div>

          <EdgesPanel edges={activeFlow.edges} onChange={updateEdge} onRemove={removeEdge} />

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

          <YamlPanel yaml={yamlPreview} onDownload={downloadYaml} onCopy={copyYaml} />
        </div>
      </div>
    </div>
  );
};

export default App;
