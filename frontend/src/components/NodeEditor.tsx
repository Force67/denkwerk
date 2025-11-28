import React from "react";
import {
  AgentDefinition,
  DecisionStrategy,
  FlowDefinition,
  FlowNode,
  NodeOutput,
  PromptDefinition,
  ToolDefinition,
} from "../types";

interface NodeEditorProps {
  node: FlowNode;
  availableAgents: AgentDefinition[];
  availableTools: ToolDefinition[];
  availablePrompts: PromptDefinition[];
  availableFlows: FlowDefinition[];
  onChange: (node: FlowNode) => void;
  onDelete: () => void;
}

const NodeEditor: React.FC<NodeEditorProps> = ({
  node,
  availableAgents,
  availableTools,
  availablePrompts,
  availableFlows,
  onChange,
  onDelete,
}) => {
  const base = node.base;
  const outputsValue = base.outputs ?? [];

  const setOutputs = (outputs: NodeOutput[]) => onChange({ ...node, base: { ...base, outputs } });

  const updateOutput = (idx: number, next: Partial<NodeOutput>) => {
    const outputs = outputsValue.map((out, i) => (i === idx ? { ...out, ...next } : out));
    setOutputs(outputs);
  };

  const addOutput = () => setOutputs([...outputsValue, { label: "out" }]);
  const removeOutput = (idx: number) => setOutputs(outputsValue.filter((_, i) => i !== idx));

  const toggleTool = (toolId: string) => {
    if (node.kind.type !== "agent") return;
    const currentTools = node.kind.tools ?? [];
    const newTools = currentTools.includes(toolId)
      ? currentTools.filter((t) => t !== toolId)
      : [...currentTools, toolId];
    onChange({ ...node, kind: { ...node.kind, tools: newTools } });
  };

  const renderKindFields = () => {
    switch (node.kind.type) {
      case "agent":
        return (
          <div className="stack">
            <div className="form-group">
              <label className="muted">Agent</label>
              <select
                className="panel-input"
                value={node.kind.agent}
                onChange={(e) => onChange({ ...node, kind: { ...node.kind, agent: e.target.value } })}
              >
                <option value="" disabled>
                  Select an agent
                </option>
                {availableAgents.map((a) => (
                  <option key={a.id} value={a.id}>
                    {a.name || a.id}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="muted">Prompt Override (Optional)</label>
              <select
                className="panel-input"
                value={node.kind.prompt ?? ""}
                onChange={(e) =>
                  onChange({
                    ...node,
                    kind: { ...node.kind, prompt: e.target.value || undefined },
                  })
                }
              >
                <option value="">(Default system prompt)</option>
                {availablePrompts.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.id} ({p.file})
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="muted">Tools</label>
              <div className="checkbox-list">
                {availableTools.length === 0 && <div className="muted small">No tools defined.</div>}
                {availableTools.map((tool) => (
                  <label key={tool.id} className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={(node.kind.tools ?? []).includes(tool.id)}
                      onChange={() => toggleTool(tool.id)}
                    />
                    <span>{tool.id}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        );
      case "decision":
        return (
          <div className="stack">
            <div className="form-group">
              <label className="muted">Prompt</label>
              <select
                className="panel-input"
                value={node.kind.prompt ?? ""}
                onChange={(e) =>
                  onChange({
                    ...node,
                    kind: { ...node.kind, prompt: e.target.value || undefined },
                  })
                }
              >
                <option value="">Select a prompt...</option>
                {availablePrompts.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.id}
                  </option>
                ))}
              </select>
            </div>
            <div className="form-group">
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
                <option value="llm">LLM (Model decides)</option>
                <option value="rule">Rule-based (Conditions)</option>
              </select>
            </div>
          </div>
        );
      case "tool":
        return (
          <div className="form-group">
            <label className="muted">Tool</label>
            <select
              className="panel-input"
              value={node.kind.tool}
              onChange={(e) => onChange({ ...node, kind: { ...node.kind, tool: e.target.value } })}
            >
              <option value="" disabled>
                Select a tool
              </option>
              {availableTools.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.id}
                </option>
              ))}
            </select>
          </div>
        );
      case "loop":
        return (
          <div className="stack">
            <div className="form-group">
              <label className="muted">Max Iterations</label>
              <input
                className="panel-input"
                type="number"
                value={node.kind.max_iterations ?? ""}
                onChange={(e) =>
                  onChange({
                    ...node,
                    kind: {
                      ...node.kind,
                      max_iterations: e.target.value === "" ? undefined : Number(e.target.value),
                    },
                  })
                }
                placeholder="e.g. 3"
              />
            </div>
            <div className="form-group">
              <label className="muted">Break Condition</label>
              <input
                className="panel-input"
                value={node.kind.condition ?? ""}
                onChange={(e) =>
                  onChange({
                    ...node,
                    kind: { ...node.kind, condition: e.target.value },
                  })
                }
                placeholder="e.g. retry_count < 3"
              />
            </div>
          </div>
        );
      case "parallel":
        return (
          <label className="checkbox-label" style={{ marginTop: 8 }}>
            <input
              type="checkbox"
              checked={node.kind.converge ?? false}
              onChange={(e) => onChange({ ...node, kind: { ...node.kind, converge: e.target.checked } })}
            />
            <span>Wait for all branches (converge)</span>
          </label>
        );
      case "subflow":
        return (
          <div className="form-group">
            <label className="muted">Sub-flow</label>
            <select
              className="panel-input"
              value={node.kind.flow}
              onChange={(e) => onChange({ ...node, kind: { ...node.kind, flow: e.target.value } })}
            >
              <option value="" disabled>
                Select a flow
              </option>
              {availableFlows
                .filter((f) => f.id !== availableFlows.find((af) => af.nodes.some((n) => n.base.id === base.id))?.id) // Simple cycle prevention check (imperfect but helpful)
                .map((f) => (
                  <option key={f.id} value={f.id}>
                    {f.id}
                  </option>
                ))}
            </select>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="stack">
      <div className="form-group">
        <label className="muted">Node ID</label>
        <input
          className="panel-input"
          value={base.id}
          onChange={(e) => onChange({ ...node, base: { ...base, id: e.target.value } })}
        />
      </div>

      <div className="form-group">
        <label className="muted">Name</label>
        <input
          className="panel-input"
          value={base.name ?? ""}
          onChange={(e) => onChange({ ...node, base: { ...base, name: e.target.value } })}
          placeholder="Display Label"
        />
      </div>

      <div className="form-group">
        <label className="muted">Description</label>
        <textarea
          className="panel-input"
          value={base.description ?? ""}
          onChange={(e) => onChange({ ...node, base: { ...base, description: e.target.value } })}
          placeholder="What does this node do?"
          style={{ minHeight: 60 }}
        />
      </div>

      <div className="divider" style={{ width: "100%", height: 1, margin: "8px 0" }} />

      <div className="section-title muted">Configuration</div>
      {renderKindFields()}

      <div className="divider" style={{ width: "100%", height: 1, margin: "8px 0" }} />

      <div className="section-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span className="muted">Outputs</span>
        <button className="ghost small" onClick={addOutput}>
          + Add
        </button>
      </div>
      
      <div className="stack">
        {outputsValue.map((out, idx) => (
          <div className="edge-row" key={`${idx}`} style={{ alignItems: "start" }}>
            <div className="stack" style={{ flex: 1, gap: 4 }}>
              <input
                className="panel-input small"
                value={out.label}
                onChange={(e) => updateOutput(idx, { label: e.target.value })}
                placeholder="Output label"
              />
              <input
                className="panel-input small"
                value={out.condition ?? ""}
                onChange={(e) => updateOutput(idx, { condition: e.target.value })}
                placeholder="Condition (optional)"
              />
            </div>
            <button className="ghost small" onClick={() => removeOutput(idx)} title="Remove output">
              ×
            </button>
          </div>
        ))}
        {outputsValue.length === 0 && <div className="muted small">No explicit outputs.</div>}
      </div>

      <div style={{ marginTop: 24 }}>
        <button className="ghost danger" onClick={onDelete} style={{ width: "100%", borderColor: "rgba(255, 80, 80, 0.3)", color: "#ff6b6b" }}>
          Delete Node
        </button>
      </div>
    </div>
  );
};

export default NodeEditor;
