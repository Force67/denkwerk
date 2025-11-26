import { FlowNode, FlowNodeKindType, NodeOutput } from "../types";

export const createFlowNode = (type: FlowNodeKindType, id?: string): FlowNode => {
  const base = { id: id ?? uid(type), inputs: [], outputs: [{ label: "out" }] };
  switch (type) {
    case "input":
      return { base: { ...base, outputs: [{ label: "out" }] }, kind: { type } };
    case "output":
      return { base: { ...base, inputs: [{ from: "" }], outputs: [] }, kind: { type } };
    case "agent":
      return { base, kind: { type, agent: "" } };
    case "decision":
      return { base: { ...base, outputs: [] }, kind: { type, strategy: "llm" } };
    case "tool":
      return { base, kind: { type, tool: "" } };
    case "merge":
      return { base: { ...base, inputs: [{ from: "" }, { from: "" }], outputs: [{ label: "out" }] }, kind: { type } };
    case "parallel":
      return { base: { ...base, outputs: [{ label: "out" }] }, kind: { type, converge: true } };
    case "loop":
      return { base: { ...base, outputs: [{ label: "out" }] }, kind: { type, max_iterations: 3 } };
    case "subflow":
      return { base, kind: { type, flow: "" } };
  }
};

export const defaultPosition = (index: number): { x: number; y: number } => ({
  x: 120 + (index % 4) * 220,
  y: 120 + Math.floor(index / 4) * 180,
});

export const getDefaultOutput = (nodes: FlowNode[], sourceId: string) => {
  const sourceNode = nodes.find((n) => n.base.id === sourceId);
  return sourceNode?.base.outputs?.[0]?.label ?? "out";
};

export const uid = (prefix: string) => `${prefix}_${Math.random().toString(16).slice(2, 6)}`;

export type FlowNodeData = { label: string; kind: FlowNodeKindType; outputs: NodeOutput[] };
