import React, { useMemo, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  Edge,
  Handle,
  MiniMap,
  Node,
  NodeProps,
  OnConnect,
  Position,
} from "reactflow";
import "reactflow/dist/style.css";
import { FlowDefinition, FlowEdge, FlowNode, FlowNodeKindType, NodeOutput } from "../types";
import { allowedByKind, FlowKind } from "../utils/flowConfig";
import { createFlowNode, defaultPosition, FlowNodeData, getDefaultOutput } from "../utils/flowNodes";

interface CanvasAreaProps {
  flow: FlowDefinition;
  flowKind: FlowKind;
  allowedTypes: FlowNodeKindType[];
  onUpdateFlow: (flow: FlowDefinition) => void;
  onSelectNode: (id: string) => void;
  onRemoveNode: (id: string) => void;
}

const CanvasArea: React.FC<CanvasAreaProps> = ({
  flow,
  flowKind,
  allowedTypes,
  onUpdateFlow,
  onSelectNode,
  onRemoveNode,
}) => {
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; nodeId: string } | null>(null);

  const nodes = useMemo(() => mapNodes(flow.nodes, allowedTypes), [flow.nodes, allowedTypes]);
  const edges = useMemo(() => mapEdges(flow.edges), [flow.edges]);

  const handleConnect: OnConnect = (connection) => {
    if (!connection.source || !connection.target) return;
    const defaultOutput = getDefaultOutput(flow.nodes, connection.source);
    const from = `${connection.source}/${connection.sourceHandle ?? defaultOutput}`;
    const to = connection.target;
    onUpdateFlow({ ...flow, edges: [...flow.edges, { from, to }] });
  };

  const handleNodeDragStop: (event: React.MouseEvent, node: Node) => void = (_evt, node) => {
    onUpdateFlow({
      ...flow,
      nodes: flow.nodes.map((n) =>
        n.base.id === node.id ? { ...n, base: { ...n.base, layout: { ...node.position } } } : n,
      ),
    });
  };

  const quickConnectToNew = (sourceId: string) => {
    if (!allowedByKind[flowKind].includes("agent")) return;
    const newNode = createFlowNode("agent");
    const from = `${sourceId}/${getDefaultOutput(flow.nodes, sourceId)}`;
    const nextFlow: FlowDefinition = {
      ...flow,
      nodes: [...flow.nodes, newNode],
      edges: [...flow.edges, { from, to: newNode.base.id }],
    };
    onUpdateFlow(nextFlow);
    onSelectNode(newNode.base.id);
    setContextMenu(null);
  };

  const promptEdge = (sourceId: string) => {
    const target = window.prompt("Connect to node id:");
    if (!target) return;
    const from = `${sourceId}/${getDefaultOutput(flow.nodes, sourceId)}`;
    const to = target.trim();
    if (!to) return;
    onUpdateFlow({ ...flow, edges: [...flow.edges, { from, to }] });
    setContextMenu(null);
  };

  return (
    <div className="flow-wrapper" onClick={() => contextMenu && setContextMenu(null)}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        fitView
        proOptions={{ hideAttribution: true }}
        nodeTypes={nodeTypes}
        onNodeDragStop={handleNodeDragStop}
        onNodeClick={(_evt, node) => onSelectNode(node.id)}
        onNodeContextMenu={(event, node) => {
          event.preventDefault();
          onSelectNode(node.id);
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
          <button className="ghost" onClick={() => quickConnectToNew(contextMenu.nodeId)}>
            New agent + edge
          </button>
          <button className="ghost" onClick={() => promptEdge(contextMenu.nodeId)}>
            Connect to node…
          </button>
          <button className="ghost" onClick={() => onRemoveNode(contextMenu.nodeId)}>
            Delete node
          </button>
        </div>
      )}
    </div>
  );
};

const mapNodes = (nodes: FlowNode[], allowed: FlowNodeKindType[]): Node<FlowNodeData>[] =>
  nodes.map((node, idx) => {
    const { background, border, color, labelPrefix } = styleForKind(node.kind.type, allowed.includes(node.kind.type));
    const label = `${labelPrefix}${node.base.name || node.base.id}`;
    return {
      id: node.base.id,
      position: node.base.layout ?? defaultPosition(idx),
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
      return {
        background: "linear-gradient(145deg, #0f172a, #0c1220)",
        border: "1px solid rgba(88, 177, 255, 0.4)",
        color: "#e6e9f8",
        labelPrefix: "",
      };
  }
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

export default CanvasArea;
