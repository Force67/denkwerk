import React from "react";

interface DefinitionListProps<T> {
  title: string;
  items: T[];
  render: (item: T, index: number) => React.ReactNode;
  onAdd: () => void;
}

const DefinitionList = <T,>({ title, items, render, onAdd }: DefinitionListProps<T>) => (
  <div className="defs">
    <div className="panel-header">
      <div className="panel-title">{title}</div>
      <button className="ghost" onClick={onAdd}>
        +
      </button>
    </div>
    <div className="stack">
      {items.map((item, idx) => (
        <div key={idx} className="def-card">
          {render(item, idx)}
        </div>
      ))}
      {items.length === 0 && <div className="muted">No {title.toLowerCase()} defined.</div>}
    </div>
  </div>
);

export default DefinitionList;
