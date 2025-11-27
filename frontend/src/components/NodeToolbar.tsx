import React from 'react';
import {
  FaUndo,
  FaRedo,
  FaRefresh,
  FaDownload,
  FaUpload,
  FaPlay,
} from 'react-icons/fa';

interface NodeToolbarProps {
  onUndo: () => void;
  onRedo: () => void;
  onReset: () => void;
  onSave?: () => void;
  onLoad?: () => void;
  onTest?: () => void;
  canUndo: boolean;
  canRedo: boolean;
  className?: string;
}

const NodeToolbar: React.FC<NodeToolbarProps> = ({
  onUndo,
  onRedo,
  onReset,
  onSave,
  onLoad,
  onTest,
  canUndo,
  canRedo,
  className = '',
}) => {
  const toolbarStyles = {
    display: 'flex',
    gap: '0.5rem',
    background: 'rgba(30, 41, 59, 0.95)',
    border: '1px solid rgba(148, 163, 184, 0.2)',
    borderRadius: '8px',
    padding: '0.5rem',
    backdropFilter: 'blur(10px)',
  };

  const buttonStyles = (disabled = false) => ({
    background: disabled
      ? 'rgba(71, 85, 105, 0.5)'
      : 'rgba(51, 65, 85, 0.8)',
    border: '1px solid rgba(148, 163, 184, 0.3)',
    borderRadius: '6px',
    padding: '0.5rem',
    color: disabled ? '#64748b' : '#f1f5f9',
    cursor: disabled ? 'not-allowed' : 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'all 0.2s ease',
    fontSize: '0.875rem',
    minWidth: '36px',
    height: '36px',
  });

  const iconStyles = {
    fontSize: '0.875rem',
  };

  return (
    <div style={toolbarStyles} className={`node-toolbar ${className}`}>
      {/* Undo/Redo Group */}
      <div style={{ display: 'flex', gap: '0.25rem' }}>
        <button
          onClick={onUndo}
          disabled={!canUndo}
          style={buttonStyles(!canUndo)}
          title="Undo (Ctrl+Z)"
          aria-label="Undo"
        >
          <FaUndo style={iconStyles} />
        </button>

        <button
          onClick={onRedo}
          disabled={!canRedo}
          style={buttonStyles(!canRedo)}
          title="Redo (Ctrl+Y)"
          aria-label="Redo"
        >
          <FaRedo style={iconStyles} />
        </button>
      </div>

      {/* Separator */}
      <div style={{
        width: '1px',
        background: 'rgba(148, 163, 184, 0.3)',
        margin: '0 0.25rem'
      }} />

      {/* Action Buttons */}
      <button
        onClick={onReset}
        style={buttonStyles()}
        title="Reset Flow"
        aria-label="Reset flow"
      >
        <FaRefresh style={iconStyles} />
      </button>

      {onSave && (
        <button
          onClick={onSave}
          style={buttonStyles()}
          title="Save Flow (Ctrl+S)"
          aria-label="Save flow"
        >
          <FaDownload style={iconStyles} />
        </button>
      )}

      {onLoad && (
        <button
          onClick={onLoad}
          style={buttonStyles()}
          title="Load Flow (Ctrl+O)"
          aria-label="Load flow"
        >
          <FaUpload style={iconStyles} />
        </button>
      )}

      {onTest && (
        <button
          onClick={onTest}
          style={{
            ...buttonStyles(),
            background: 'rgba(34, 197, 94, 0.8)',
            border: '1px solid rgba(34, 197, 94, 0.3)',
          }}
          title="Test Flow (Ctrl+Enter)"
          aria-label="Test flow"
        >
          <FaPlay style={iconStyles} />
        </button>
      )}
    </div>
  );
};

export default NodeToolbar;