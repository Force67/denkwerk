import React from "react";

interface YamlPanelProps {
  yaml: string;
  onDownload: () => void;
  onCopy: () => Promise<void> | void;
}

const YamlPanel: React.FC<YamlPanelProps> = ({ yaml, onDownload, onCopy }) => (
  <div className="panel-section">
    <div className="panel-header">
      <div className="panel-title">YAML</div>
    </div>
    <textarea className="yaml-preview" readOnly value={yaml} />
    <div className="edge-row" style={{ gridTemplateColumns: "1fr auto" }}>
      <button className="ghost" onClick={onDownload} title="Download .yaml file">
        Download
      </button>
      <button className="ghost" onClick={onCopy}>
        Copy
      </button>
    </div>
  </div>
);

export default YamlPanel;
