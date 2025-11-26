import { FlowNodeKindType } from "../types";

export type FlowKind = "sequential" | "handoff" | "parallel" | "group";

export const flowTypeLabels: Record<FlowKind, string> = {
  sequential: "Sequential",
  handoff: "Handoff",
  parallel: "Parallel/Concurrent",
  group: "Group Chat",
};

export const allowedByKind: Record<FlowKind, FlowNodeKindType[]> = {
  sequential: ["input", "agent", "decision", "tool", "subflow", "loop", "output"],
  handoff: ["input", "agent", "decision", "output"],
  parallel: ["input", "agent", "parallel", "merge", "tool", "output"],
  group: ["input", "agent", "decision", "merge", "output"],
};

export const typeButtons: { kind: FlowNodeKindType; label: string }[] = [
  { kind: "agent", label: "Agent" },
  { kind: "decision", label: "Decision" },
  { kind: "tool", label: "Tool" },
  { kind: "merge", label: "Merge" },
  { kind: "parallel", label: "Parallel" },
  { kind: "loop", label: "Loop" },
  { kind: "output", label: "Output" },
  { kind: "subflow", label: "Subflow" },
];
