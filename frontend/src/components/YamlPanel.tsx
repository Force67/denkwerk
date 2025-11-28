import React from "react";
import Editor from "@monaco-editor/react";

interface YamlPanelProps {
  yaml: string;
  onDownload: () => void;
  onCopy: () => Promise<void> | void;
}

const YamlPanel: React.FC<YamlPanelProps> = ({ yaml, onDownload, onCopy }) => (
  <div className="panel-section" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
    <div className="panel-header">
      <div className="panel-title">YAML Preview</div>
      <div style={{ display: "flex", gap: 8 }}>
        <button className="ghost small" onClick={onCopy} title="Copy to clipboard">
          Copy
        </button>
        <button className="ghost small" onClick={onDownload} title="Download .yaml file">
          Download
        </button>
      </div>
    </div>
    <div style={{ height: "300px", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, overflow: "hidden" }}>
      <Editor
        height="100%"
        defaultLanguage="yaml"
        value={yaml}
        theme="vs-dark"
        options={{
          readOnly: true,
          minimap: { enabled: false },
          fontSize: 12,
          lineNumbers: "off",
          scrollBeyondLastLine: false,
          fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
          padding: { top: 12, bottom: 12 },
        }}
      />
    </div>
  </div>
);

export default YamlPanel;
