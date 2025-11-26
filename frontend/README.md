# denkwerk flow builder (frontend)

Full-screen React Flow canvas for composing denkwerk YAML specs. Drag nodes, wire connections, edit node behaviour, and manage agents/tools/prompts from the side panel.

## Quickstart

```bash
cd frontend
bun install
bun run dev
```

- Use the toolbar to add node types (agent, decision, tool, merge, output) and new flows.
- Select a flow mode (Sequential, Handoff, Parallel, Group) to constrain the node palette.
- Drag nodes on the canvas, connect them, and click to edit details in the inspector.
- Right-click a node to quick-add a connected agent or manually connect to another node.
- Manage global agents/tools/prompts definitions in the side panel; YAML preview updates live and can be downloaded.
