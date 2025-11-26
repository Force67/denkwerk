import React from "react";
import { FlowNodeKindType } from "../types";
import { FlowKind, flowTypeLabels } from "../utils/flowConfig";

interface TopBarProps {
  flowKind: FlowKind;
  allowedTypes: FlowNodeKindType[];
  typeButtons: { kind: FlowNodeKindType; label: string }[];
  onChangeFlowKind: (kind: FlowKind) => void;
  onAddNode: (kind: FlowNodeKindType) => void;
  onAddFlow: () => void;
  onLoadSample: () => void;
  onReset: () => void;
  onDownload: () => void;
}

const TopBar: React.FC<TopBarProps> = ({
  flowKind,
  allowedTypes,
  typeButtons,
  onChangeFlowKind,
  onAddNode,
  onAddFlow,
  onLoadSample,
  onReset,
  onDownload,
}) => (
  <div className="topbar">
    <div className="brand">denkwerk canvas</div>
    <div className="toolbar">
      <select
        className="flow-select"
        value={flowKind}
        onChange={(e) => onChangeFlowKind(e.target.value as FlowKind)}
      >
        {Object.entries(flowTypeLabels).map(([value, label]) => (
          <option key={value} value={value}>
            {label}
          </option>
        ))}
      </select>
      <div className="divider" />
      {typeButtons
        .filter((btn) => allowedTypes.includes(btn.kind))
        .map((btn) => (
          <button key={btn.kind} onClick={() => onAddNode(btn.kind)}>
            + {btn.label}
          </button>
        ))}
      <button className="ghost" onClick={onAddFlow}>
        + New flow
      </button>
    </div>
    <div className="toolbar">
      <button className="ghost" onClick={onLoadSample}>
        Load sample
      </button>
      <button className="ghost" onClick={onReset}>
        Reset blank
      </button>
      <button onClick={onDownload}>Download YAML</button>
    </div>
  </div>
);

export default TopBar;
