import React from "react";
import { DecisionStrategy, FlowNode, NodeOutput } from "../types";

interface NodeEditorProps {
  node: FlowNode;
  onChange: (node: FlowNode) => void;
  onDelete: () => void;
}

const NodeEditor: React.FC<NodeEditorProps> = ({ node, onChange, onDelete }) => {
  const base = node.base;
  const outputsValue = base.outputs ?? [];

  const setOutputs = (outputs: NodeOutput[]) => onChange({ ...node, base: { ...base, outputs } });
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

export default NodeEditor;
