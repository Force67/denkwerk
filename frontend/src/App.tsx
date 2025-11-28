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
import { apiClient } from "./services/apiClient";

const starterDoc: FlowDocument = {
  version: "0.1",
  metadata: {
    name: "Customer Support Triage",
    description: "Intelligent routing system for inbound customer support tickets.",
    tags: ["support", "triage", "automation"],
  },
  agents: [
    {
      id: "triage_agent",
      model: "openai/gpt-4o",
      name: "Triage Specialist",
      description: "Analyzes incoming requests to determine the category and urgency.",
      system_prompt: "prompts/triage.md",
      defaults: { temperature: 0.0 },
    },
    {
      id: "tech_support",
      model: "openai/gpt-4o",
      name: "Technical Support",
      description: "Resolves technical issues using documentation and diagnostics.",
      system_prompt: "prompts/tech_ops.md",
      tools: ["fetch_docs"],
    },
    {
      id: "billing_support",
      model: "openai/gpt-4o",
      name: "Billing Specialist",
      description: "Handles invoices, refunds, and subscription status.",
      system_prompt: "prompts/billing_policy.md",
      tools: ["check_status"],
    },
    {
      id: "general_assistant",
      model: "openai/gpt-4o-mini",
      name: "General Assistant",
      description: "Handles general inquiries and FAQs.",
      system_prompt: "prompts/general_faq.md",
    },
  ],
  tools: [
    {
      id: "fetch_docs",
      kind: "http",
      description: "Search internal technical documentation.",
      spec: "tools/docs.yaml",
    },
    {
      id: "check_status",
      kind: "function",
      description: "Check the status of a user's order or ticket.",
      function: "check_order_status",
    },
  ],
  prompts: [
    { id: "triage_prompt", file: "prompts/triage.md", description: "Classification rules" },
    { id: "tech_prompt", file: "prompts/tech_ops.md", description: "Debug guidelines" },
    { id: "billing_prompt", file: "prompts/billing_policy.md", description: "Refund policy" },
  ],
  flows: [
    {
      id: "main",
      entry: "start",
      nodes: [
        {
          base: { id: "start", name: "Inbound Ticket", outputs: [{ label: "out" }] },
          kind: { type: "input" },
        },
        {
          base: {
            id: "triage_node",
            name: "Classify Intent",
            description: "Analyze the user input to determine the correct department.",
            inputs: [{ from: "start/out" }],
            outputs: [{ label: "classified" }],
          },
          kind: { type: "agent", agent: "triage_agent", prompt: "triage_prompt" },
        },
        {
          base: {
            id: "router",
            name: "Route Department",
            inputs: [{ from: "triage_node/classified" }],
            outputs: [
              { label: "tech", condition: "category == 'technical'" },
              { label: "billing", condition: "category == 'billing'" },
              { label: "general" },
            ],
          },
          kind: { type: "decision", strategy: "llm" },
        },
        {
          base: {
            id: "tech_node",
            name: "Tech Support",
            inputs: [{ from: "router/tech" }],
            outputs: [{ label: "resolved" }],
          },
          kind: { type: "agent", agent: "tech_support", tools: ["fetch_docs"] },
        },
        {
          base: {
            id: "billing_node",
            name: "Billing Support",
            inputs: [{ from: "router/billing" }],
            outputs: [{ label: "resolved" }],
          },
          kind: { type: "agent", agent: "billing_support", tools: ["check_status"] },
        },
        {
          base: {
            id: "general_node",
            name: "General Inquiry",
            inputs: [{ from: "router/general" }],
            outputs: [{ label: "resolved" }],
          },
          kind: { type: "agent", agent: "general_assistant" },
        },
        {
          base: {
            id: "merge_node",
            name: "Consolidate",
            inputs: [
              { from: "tech_node/resolved" },
              { from: "billing_node/resolved" },
              { from: "general_node/resolved" },
            ],
            outputs: [{ label: "out" }],
          },
          kind: { type: "merge" },
        },
        {
          base: {
            id: "final_check",
            name: "Quality Assurance",
            description: "Review the response for tone and accuracy.",
            inputs: [{ from: "merge_node/out" }],
            outputs: [{ label: "final" }],
          },
          kind: { type: "agent", agent: "triage_agent", prompt: "triage_prompt" },
        },
        {
          base: { id: "end", name: "Send Response", inputs: [{ from: "final_check/final" }] },
          kind: { type: "output" },
        },
      ],
      edges: [
        { from: "start/out", to: "triage_node" },
        { from: "triage_node/classified", to: "router" },
        { from: "router/tech", to: "tech_node", label: "Technical" },
        { from: "router/billing", to: "billing_node", label: "Billing" },
        { from: "router/general", to: "general_node", label: "Other" },
        { from: "tech_node/resolved", to: "merge_node" },
        { from: "billing_node/resolved", to: "merge_node" },
        { from: "general_node/resolved", to: "merge_node" },
        { from: "merge_node/out", to: "final_check" },
        { from: "final_check/final", to: "end" },
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

  const handleRun = async () => {
    const input = window.prompt("Enter task input for the flow:", "Describe the issue...");
    if (!input) return;

    try {
      // Check/Prompt for API Key if needed
      if (!apiClient.getApiKey()) {
        const key = window.prompt("Enter OpenAI API Key (optional, if not set on server):");
        if (key) apiClient.setApiKey(key);
      }

      console.log("Executing flow...", flowDoc);
      const res = await apiClient.executeFlow(flowDoc, input);
      
      if (res.success && res.data) {
        console.log("Execution Result:", res.data);
        const output = res.data.final_output || "No output";
        alert(`Execution Successful!\n\nOutput:\n${output}`);
      } else {
        console.error("Execution failed:", res);
        alert(`Execution Failed: ${res.message || "Unknown error"}`);
      }
    } catch (err) {
      console.error("Execution error:", err);
      alert(`Error executing flow: ${err}`);
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
        onRun={handleRun}
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
                availableAgents={flowDoc.agents}
                availableTools={flowDoc.tools}
                availablePrompts={flowDoc.prompts}
                availableFlows={flowDoc.flows}
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
