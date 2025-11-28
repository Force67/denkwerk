import React, { useEffect, useState } from "react";
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
  useEdgesState,
  useNodesState,
  MarkerType,
} from "reactflow";
import "reactflow/dist/style.css";
import { FlowDefinition, FlowEdge, FlowNode, FlowNodeKindType } from "../types";
import { allowedByKind, FlowKind } from "../utils/flowConfig";
import { createFlowNode, defaultPosition, FlowNodeData, getDefaultOutput } from "../utils/flowNodes";
import {
  Bot,
  GitFork,
  Layers,
  Merge,
  PlayCircle,
  RotateCw,
  Split,
  StopCircle,
  Wrench,
} from "lucide-react";

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

  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  useEffect(() => {
    setNodes(mapNodes(flow.nodes, allowedTypes));
    setEdges(mapEdges(flow.edges, flow.nodes));
  }, [flow.nodes, flow.edges, allowedTypes, setNodes, setEdges]);

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
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
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

const stringToHue = (str: string): number => {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  return Math.abs(hash % 360);
};

const getAgentStyle = (agentId: string) => {
  const hue = stringToHue(agentId);
  return {
    background: `linear-gradient(145deg, hsl(${hue}, 40%, 20%), hsl(${hue}, 40%, 10%))`,
    border: `1px solid hsl(${hue}, 70%, 60%)`,
    color: `hsl(${hue}, 90%, 90%)`,
    stroke: `hsl(${hue}, 70%, 60%)`
  };
};

const mapNodes = (nodes: FlowNode[], allowed: FlowNodeKindType[]): Node<FlowNodeData>[] =>
  nodes.map((node, idx) => {
    let { background, border, color } = styleForKind(node.kind.type, allowed.includes(node.kind.type));

    if (node.kind.type === "agent" && allowed.includes("agent")) {
        const style = getAgentStyle(node.kind.agent);
        background = style.background;
        border = style.border;
        color = style.color;
    }

    return {
      id: node.base.id,
      position: node.base.layout ?? defaultPosition(idx),
      data: { label: node.base.name || node.base.id, kind: node.kind.type, outputs: node.base.outputs ?? [] },
      type: "flowNode",
      style: {
        background,
        border,
        color,
        padding: 0,
        borderRadius: 8,
        fontSize: 13,
        boxShadow: "0 10px 24px rgba(0,0,0,0.35)",
        minWidth: 160,
        overflow: "hidden",
      },
    };
  });

const mapEdges = (edges: FlowEdge[], nodes: FlowNode[]): Edge[] =>
  edges.map((edge, idx) => {
    const [source, sourceHandle] = edge.from.split("/");
    const [target] = edge.to.split("/");
    const sourceNode = nodes.find((n) => n.base.id === source);
    let strokeColor = getEdgeColor(sourceNode?.kind.type);

    if (sourceNode?.kind.type === "agent") {
        strokeColor = getAgentStyle(sourceNode.kind.agent).stroke;
    }

    return {
      id: `${edge.from}-${edge.to}-${idx}`,
      source,
      target,
      sourceHandle,
      label: edge.label,
      animated: false,
      style: { stroke: strokeColor, strokeWidth: 1.5 },
      labelBgPadding: [6, 4],
      labelBgBorderRadius: 4,
      labelStyle: { fill: "#dce6ff", fontSize: 11, fontWeight: 500 },
      labelBgStyle: { fill: "#0f172a", stroke: strokeColor, strokeOpacity: 0.7 },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: strokeColor,
      },
    } as Edge;
  });

const getEdgeColor = (kind?: FlowNodeKindType): string => {
  switch (kind) {
    case "input":
      return "#4ade80";
    case "output":
      return "#f472b6";
    case "agent":
      return "#60a5fa";
    case "decision":
      return "#facc15";
    case "tool":
      return "#2dd4bf";
    case "merge":
      return "#a78bfa";
    case "parallel":
      return "#38bdf8";
    case "loop":
      return "#fb923c";
    case "subflow":
      return "#818cf8";
    default:
      return "#58b1ff";
  }
};

const styleForKind = (
  kind: FlowNodeKindType,
  enabled: boolean,
): { background: string; border: string; color: string } => {
  const disabled = {
    background: "linear-gradient(145deg, #11131c, #0d0f16)",
    border: "1px dashed rgba(255,255,255,0.15)",
    color: "#99a0b8",
  };

  if (!enabled) return disabled;

  switch (kind) {
    case "input":
      return {
        background: "linear-gradient(145deg, #0f2e1b, #0a1f12)",
        border: "1px solid rgba(74, 222, 128, 0.4)",
        color: "#d6f8e1",
      };
    case "output":
      return {
        background: "linear-gradient(145deg, #2d1b3a, #201329)",
        border: "1px solid rgba(255, 149, 233, 0.4)",
        color: "#ffe1f7",
      };
    case "agent":
      return {
        background: "linear-gradient(145deg, #1e293b, #0f172a)",
        border: "1px solid rgba(96, 165, 250, 0.4)",
        color: "#e6e9f8",
      };
    case "decision":
      return {
        background: "linear-gradient(145deg, #2e2a10, #1f1c0b)",
        border: "1px solid rgba(253, 224, 71, 0.4)",
        color: "#fef9c3",
      };
    case "tool":
      return {
        background: "linear-gradient(145deg, #132e2b, #0d1f1d)",
        border: "1px solid rgba(45, 212, 191, 0.4)",
        color: "#ccfbf1",
      };
    case "merge":
      return {
        background: "linear-gradient(145deg, #1d1a2c, #161325)",
        border: "1px solid rgba(167, 139, 250, 0.4)",
        color: "#ede9fe",
      };
    case "parallel":
      return {
        background: "linear-gradient(145deg, #14202d, #0f1a26)",
        border: "1px solid rgba(56, 189, 248, 0.4)",
        color: "#e0f2fe",
      };
    case "loop":
      return {
        background: "linear-gradient(145deg, #2e1e10, #1f140b)",
        border: "1px solid rgba(251, 146, 60, 0.4)",
        color: "#ffedd5",
      };
    case "subflow":
      return {
        background: "linear-gradient(145deg, #1a1d2d, #121625)",
        border: "1px solid rgba(129, 140, 248, 0.4)",
        color: "#e0e7ff",
      };
    default:
      return {
        background: "linear-gradient(145deg, #0f172a, #0c1220)",
        border: "1px solid rgba(88, 177, 255, 0.4)",
        color: "#e6e9f8",
      };
  }
};

const getIcon = (kind: FlowNodeKindType) => {
  switch (kind) {
    case "input":
      return PlayCircle;
    case "output":
      return StopCircle;
    case "agent":
      return Bot;
    case "decision":
      return Split;
    case "tool":
      return Wrench;
    case "merge":
      return Merge;
    case "parallel":
      return GitFork; // or Layers
    case "loop":
      return RotateCw;
    case "subflow":
      return Layers;
    default:
      return Bot;
  }
};

const FlowNodeComponent: React.FC<NodeProps<FlowNodeData>> = ({ data }) => {
  const outputs = data.outputs;
  const Icon = getIcon(data.kind);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <Handle
        type="target"
        position={Position.Left}
        style={{ background: "rgba(255,255,255,0.8)", width: 8, height: 8 }}
      />
      
      <div style={{ padding: "10px 12px", borderBottom: "1px solid rgba(255,255,255,0.1)", display: "flex", alignItems: "center", gap: 8 }}>
        <Icon size={16} style={{ opacity: 0.8 }} />
        <div style={{ fontWeight: 600, fontSize: 13 }}>{data.label}</div>
      </div>

      <div style={{ padding: "8px 12px", flex: 1, minHeight: 30, display: "flex", flexDirection: "column", justifyContent: "center" }}>
        <div className="muted" style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: 0.5 }}>{data.kind}</div>
      </div>

      {outputs.map((out, idx) => {
        const top = 40 + idx * 24; // Rough estimation, better to rely on flex or dynamic calc if possible, but handles need absolute pos. 
        // Actually, React Flow handles are absolute. Let's distribute them evenly along the right side or just stack them.
        // For now, let's put them on the right side, distributed.
        // Better yet, let's render a row for each output if there are multiple, or just one generic one if unspecified.
        // Since we can't easily change the handle position dynamically based on DOM layout without custom logic, let's stick to distributed.
        const isMulti = outputs.length > 1;
        const offset = isMulti ? (idx + 1) * (100 / (outputs.length + 1)) : 50;

        return (
            <Handle
            key={`${out.label}-${idx}`}
            type="source"
            position={Position.Right}
            id={out.label}
            style={{
                top: `${offset}%`,
                background: "rgba(88, 177, 255, 0.9)",
                width: 8,
                height: 8,
            }}
            >
            {isMulti && (
                <div style={{ position: "absolute", right: 12, top: -8, fontSize: 10, color: "rgba(255,255,255,0.7)", whiteSpace: "nowrap" }}>
                    {out.label}
                </div>
            )}
            </Handle>
        );
      })}
    </div>
  );
};

const nodeTypes = { flowNode: FlowNodeComponent };

export default CanvasArea;
