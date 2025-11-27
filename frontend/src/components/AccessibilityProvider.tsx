import React, { createContext, useContext, useEffect, useState } from 'react';
import { createPortal } from 'react-dom';

interface AccessibilityContextType {
  announceMessage: (message: string, priority?: 'polite' | 'assertive') => void;
  keyboardShortcuts: Record<string, string>;
  isHighContrast: boolean;
  setIsHighContrast: (enabled: boolean) => void;
  reducedMotion: boolean;
  setReducedMotion: (enabled: boolean) => void;
}

const AccessibilityContext = createContext<AccessibilityContextType>({
  announceMessage: () => {},
  keyboardShortcuts: {},
  isHighContrast: false,
  setIsHighContrast: () => {},
  reducedMotion: false,
  setReducedMotion: () => {},
});

export const useAccessibility = () => useContext(AccessibilityContext);

const defaultKeyboardShortcuts: Record<string, string> = {
  'Ctrl+Z': 'Undo',
  'Ctrl+Y': 'Redo',
  'Ctrl+S': 'Save',
  'Ctrl+O': 'Load',
  'Ctrl+Enter': 'Test Flow',
  'Delete': 'Delete Selected Node',
  'Escape': 'Deselect',
  'Arrow Keys': 'Navigate Nodes',
  'Enter': 'Edit Selected Node',
  'Space': 'Add Node',
};

interface AccessibilityProviderProps {
  children: React.ReactNode;
}

export const AccessibilityProvider: React.FC<AccessibilityProviderProps> = ({ children }) => {
  const [announcements, setAnnouncements] = useState<string[]>([]);
  const [isHighContrast, setIsHighContrast] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(
    window.matchMedia('(prefers-reduced-motion: reduce)').matches
  );

  // Keyboard shortcuts handler
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Handle common shortcuts
      if (event.ctrlKey || event.metaKey) {
        switch (event.key.toLowerCase()) {
          case 'z':
            event.preventDefault();
            announceMessage('Undo action');
            break;
          case 'y':
            event.preventDefault();
            announceMessage('Redo action');
            break;
          case 's':
            event.preventDefault();
            announceMessage('Saving flow');
            break;
          case 'o':
            event.preventDefault();
            announceMessage('Loading flow');
            break;
        }
      }

      // Navigation shortcuts
      switch (event.key) {
        case 'Delete':
        case 'Backspace':
          if (event.target === document.body) {
            announceMessage('Delete selected node');
          }
          break;
        case 'Escape':
          announceMessage('Deselected node');
          break;
        case 'ArrowUp':
        case 'ArrowDown':
        case 'ArrowLeft':
        case 'ArrowRight':
          if (event.target === document.body) {
            announceMessage('Navigate flow');
          }
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Screen reader announcements
  const announceMessage = (message: string, priority: 'polite' | 'assertive' = 'polite') => {
    const id = Date.now().toString();
    setAnnouncements(prev => [...prev, message]);

    // Auto-remove after announcement
    setTimeout(() => {
      setAnnouncements(prev => prev.filter(msg => msg !== message));
    }, 1000);
  };

  // Detect system preferences
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    const handleChange = (e: MediaQueryListEvent) => setReducedMotion(e.matches);

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  const value: AccessibilityContextType = {
    announceMessage,
    keyboardShortcuts: defaultKeyboardShortcuts,
    isHighContrast,
    setIsHighContrast,
    reducedMotion,
    setReducedMotion,
  };

  return (
    <AccessibilityContext.Provider value={value}>
      {children}

      {/* Screen reader announcements */}
      {typeof document !== 'undefined' &&
        createPortal(
          <div
            aria-live="polite"
            aria-atomic="true"
            style={{
              position: 'absolute',
              left: '-10000px',
              width: '1px',
              height: '1px',
              overflow: 'hidden',
            }}
          >
            {announcements[announcements.length - 1]}
          </div>,
          document.body
        )}

      {/* High contrast styles */}
      {isHighContrast && (
        <style>
          {`
            body {
              filter: contrast(1.5) !important;
            }

            .flow-node {
              border: 2px solid white !important;
            }

            .flow-node.selected {
              outline: 3px solid yellow !important;
              outline-offset: 2px;
            }
          `}
        </style>
      )}

      {/* Reduced motion styles */}
      {reducedMotion && (
        <style>
          {`
            *,
            *::before,
            *::after {
              animation-duration: 0.01ms !important;
              animation-iteration-count: 1 !important;
              transition-duration: 0.01ms !important;
              scroll-behavior: auto !important;
            }
          `}
        </style>
      )}
    </AccessibilityContext.Provider>
  );
};