import { useState, useCallback, useMemo, useRef } from 'react';
import { Node, Edge, addEdge, Connection, MarkerType } from 'reactflow';
import type { FlowDocument, FlowDefinition } from '../../types';
import { generateYaml } from '../utils/yamlGenerator';

export interface FlowManagerState {
  nodes: Node[];
  edges: Edge[];
  document: FlowDocument;
  selectedNode: string | null;
  history: {
    nodes: Node[][];
    edges: Edge[][];
  };
  historyIndex: number;
}

export interface FlowManagerActions {
  onNodesChange: (nodes: Node[]) => void;
  onEdgesChange: (edges: Edge[]) => void;
  onConnect: (connection: Connection) => void;
  addNode: (type: string, position: { x: number; y: number }) => void;
  deleteNode: (nodeId: string) => void;
  updateNodeData: (nodeId: string, data: any) => void;
  selectNode: (nodeId: string | null) => void;
  undo: () => void;
  redo: () => void;
  resetFlow: () => void;
  loadFlow: (document: FlowDocument) => void;
  generateYaml: () => string;
}

export function useFlowManager(initialDocument?: FlowDocument): [FlowManagerState, FlowManagerActions] {
  const [state, setState] = useState<FlowManagerState>(() => ({
    nodes: initialDocument?.flows?.[0]?.nodes || [],
    edges: initialDocument?.flows?.[0]?.edges || [],
    document: initialDocument || getDefaultDocument(),
    selectedNode: null,
    history: {
      nodes: [initialDocument?.flows?.[0]?.nodes || []],
      edges: [initialDocument?.flows?.[0]?.edges || []],
    },
    historyIndex: 0,
  }));

  const saveToHistory = useCallback((newNodes: Node[], newEdges: Edge[]) => {
    setState(prev => {
      const newHistory = {
        nodes: prev.history.nodes.slice(0, prev.historyIndex + 1),
        edges: prev.history.edges.slice(0, prev.historyIndex + 1),
      };

      newHistory.nodes.push(newNodes);
      newHistory.edges.push(newEdges);

      return {
        ...prev,
        nodes: newNodes,
        edges: newEdges,
        history: newHistory,
        historyIndex: newHistory.nodes.length - 1,
      };
    });
  }, []);

  const onNodesChange = useCallback((nodes: Node[]) => {
    setState(prev => ({ ...prev, nodes }));
  }, []);

  const onEdgesChange = useCallback((edges: Edge[]) => {
    setState(prev => ({ ...prev, edges }));
  }, []);

  const onConnect = useCallback((connection: Connection) => {
    const newEdge = {
      ...connection,
      id: `${connection.source}-${connection.target}`,
      type: 'smoothstep',
      animated: true,
      markerEnd: {
        type: MarkerType.Arrow,
        color: '#3b82f6',
      },
    };

    setState(prev => {
      const newEdges = [...prev.edges, newEdge];
      saveToHistory(prev.nodes, newEdges);
      return {
        ...prev,
        edges: newEdges,
      };
    });
  }, [saveToHistory]);

  const addNode = useCallback((type: string, position: { x: number; y: number }) => {
    const nodeId = `${type}_${Date.now()}`;
    const newNode: Node = {
      id: nodeId,
      type,
      position,
      data: {
        label: `${type} Node`,
        description: '',
        // Default data based on node type
        ...(type === 'input' && { defaultValue: '' }),
        ...(type === 'agent' && {
          agentId: '',
          model: '',
          temperature: 0.7,
        }),
        ...(type === 'decision' && {
          strategy: 'llm',
          conditions: [],
        }),
      },
    };

    setState(prev => {
      const newNodes = [...prev.nodes, newNode];
      saveToHistory(newNodes, prev.edges);
      return {
        ...prev,
        nodes: newNodes,
      };
    });
  }, [saveToHistory]);

  const deleteNode = useCallback((nodeId: string) => {
    setState(prev => {
      const newNodes = prev.nodes.filter(node => node.id !== nodeId);
      const newEdges = prev.edges.filter(
        edge => edge.source !== nodeId && edge.target !== nodeId
      );

      saveToHistory(newNodes, newEdges);

      return {
        ...prev,
        nodes: newNodes,
        edges: newEdges,
        selectedNode: prev.selectedNode === nodeId ? null : prev.selectedNode,
      };
    });
  }, [saveToHistory]);

  const updateNodeData = useCallback((nodeId: string, data: any) => {
    setState(prev => {
      const newNodes = prev.nodes.map(node =>
        node.id === nodeId ? { ...node, data: { ...node.data, ...data } } : node
      );
      saveToHistory(newNodes, prev.edges);
      return {
        ...prev,
        nodes: newNodes,
      };
    });
  }, [saveToHistory]);

  const selectNode = useCallback((nodeId: string | null) => {
    setState(prev => ({ ...prev, selectedNode: nodeId }));
  }, []);

  const undo = useCallback(() => {
    setState(prev => {
      if (prev.historyIndex > 0) {
        const newIndex = prev.historyIndex - 1;
        return {
          ...prev,
          nodes: prev.history.nodes[newIndex],
          edges: prev.history.edges[newIndex],
          historyIndex: newIndex,
        };
      }
      return prev;
    });
  }, []);

  const redo = useCallback(() => {
    setState(prev => {
      if (prev.historyIndex < prev.history.nodes.length - 1) {
        const newIndex = prev.historyIndex + 1;
        return {
          ...prev,
          nodes: prev.history.nodes[newIndex],
          edges: prev.history.edges[newIndex],
          historyIndex: newIndex,
        };
      }
      return prev;
    });
  }, []);

  const resetFlow = useCallback(() => {
    const defaultDocument = getDefaultDocument();
    setState(prev => ({
      ...prev,
      nodes: defaultDocument.flows[0].nodes,
      edges: defaultDocument.flows[0].edges,
      document: defaultDocument,
      selectedNode: null,
      history: {
        nodes: [defaultDocument.flows[0].nodes],
        edges: [defaultDocument.flows[0].edges],
      },
      historyIndex: 0,
    }));
  }, []);

  const loadFlow = useCallback((document: FlowDocument) => {
    const flow = document.flows?.[0];
    if (flow) {
      setState(prev => ({
        ...prev,
        nodes: flow.nodes || [],
        edges: flow.edges || [],
        document,
        selectedNode: null,
        history: {
          nodes: [flow.nodes || []],
          edges: [flow.edges || []],
        },
        historyIndex: 0,
      }));
    }
  }, []);

  const generateYaml = useCallback(() => {
    return generateYaml(state.document, state.nodes, state.edges);
  }, [state.document, state.nodes, state.edges]);

  const actions: FlowManagerActions = useMemo(() => ({
    onNodesChange,
    onEdgesChange,
    onConnect,
    addNode,
    deleteNode,
    updateNodeData,
    selectNode,
    undo,
    redo,
    resetFlow,
    loadFlow,
    generateYaml,
  }), [
    onNodesChange,
    onEdgesChange,
    onConnect,
    addNode,
    deleteNode,
    updateNodeData,
    selectNode,
    undo,
    redo,
    resetFlow,
    loadFlow,
    generateYaml,
  ]);

  return [state, actions];
}

function getDefaultDocument(): FlowDocument {
  return {
    version: '0.1',
    flows: [{
      id: 'main',
      entry: 'start',
      nodes: [],
      edges: [],
    }],
  };
}