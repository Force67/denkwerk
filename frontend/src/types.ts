export interface FlowDocument {
  version: string;
  metadata?: FlowMetadata;
  agents: AgentDefinition[];
  tools: ToolDefinition[];
  prompts: PromptDefinition[];
  flows: FlowDefinition[];
}

export interface FlowMetadata {
  name?: string;
  description?: string;
  tags?: string[];
}

export interface AgentDefinition {
  id: string;
  model: string;
  name?: string;
  description?: string;
  system_prompt?: string;
  tools?: string[];
  defaults?: CallSettings;
}

export interface ToolDefinition {
  id: string;
  kind: string;
  description?: string;
  spec?: string;
  function?: string;
}

export interface PromptDefinition {
  id: string;
  file: string;
  description?: string;
}

export interface FlowDefinition {
  id: string;
  entry: string;
  nodes: FlowNode[];
  edges: FlowEdge[];
}

export interface FlowEdge {
  from: string;
  to: string;
  condition?: string;
  label?: string;
}

export interface NodeBase {
  id: string;
  name?: string;
  description?: string;
  inputs?: NodeInput[];
  outputs?: NodeOutput[];
  layout?: NodeLayout;
}

export interface NodeInput {
  from: string;
}

export interface NodeOutput {
  label: string;
  condition?: string;
}

export interface NodeLayout {
  x: number;
  y: number;
}

export type FlowNode =
  | { base: NodeBase; kind: InputNode }
  | { base: NodeBase; kind: OutputNode }
  | { base: NodeBase; kind: AgentNode }
  | { base: NodeBase; kind: DecisionNode }
  | { base: NodeBase; kind: ToolNode }
  | { base: NodeBase; kind: MergeNode }
  | { base: NodeBase; kind: ParallelNode }
  | { base: NodeBase; kind: LoopNode }
  | { base: NodeBase; kind: SubflowNode };

export type FlowNodeKindType = FlowNode["kind"]["type"];

export interface InputNode {
  type: "input";
}

export interface OutputNode {
  type: "output";
}

export interface AgentNode {
  type: "agent";
  agent: string;
  prompt?: string;
  tools?: string[];
  parameters?: CallSettings;
}

export interface DecisionNode {
  type: "decision";
  prompt?: string;
  strategy?: DecisionStrategy;
}

export interface ToolNode {
  type: "tool";
  tool: string;
}

export interface MergeNode {
  type: "merge";
}

export interface ParallelNode {
  type: "parallel";
  converge?: boolean;
}

export interface LoopNode {
  type: "loop";
  max_iterations?: number;
  condition?: string;
}

export interface SubflowNode {
  type: "subflow";
  flow: string;
}

export type DecisionStrategy = "llm" | "rule";

export interface CallSettings {
  model?: string;
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  timeout_ms?: number;
  retry?: RetryPolicy;
}

export interface RetryPolicy {
  max: number;
  backoff_ms?: number;
}
