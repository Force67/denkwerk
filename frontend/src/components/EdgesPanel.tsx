import React from "react";
import { FlowEdge } from "../types";

interface EdgesPanelProps {
  edges: FlowEdge[];
  onChange: (index: number, edge: FlowEdge) => void;
  onRemove: (index: number) => void;
}

const EdgesPanel: React.FC<EdgesPanelProps> = ({ edges, onChange, onRemove }) => (
  <div className="panel-section">
    <div className="panel-header">
      <div className="panel-title">Edges</div>
      <div className="badge small">{edges.length}</div>
    </div>
    <div className="edge-list">
      {edges.map((edge, idx) => (
        <div key={`${edge.from}-${idx}`} className="edge-row">
          <input
            value={edge.from}
            onChange={(e) => onChange(idx, { ...edge, from: e.target.value })}
            className="panel-input"
            placeholder="from"
          />
          <input
            value={edge.to}
            onChange={(e) => onChange(idx, { ...edge, to: e.target.value })}
            className="panel-input"
            placeholder="to"
          />
          <button className="ghost" onClick={() => onRemove(idx)}>
            ×
          </button>
        </div>
      ))}
      {edges.length === 0 && <div className="muted">No edges yet.</div>}
    </div>
  </div>
);

export default EdgesPanel;
