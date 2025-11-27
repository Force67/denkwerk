import React, { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface State {
  hasError: boolean;
  error?: Error;
  errorInfo?: ErrorInfo;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({ errorInfo });
    this.props.onError?.(error, errorInfo);

    // Log to error reporting service
    console.error('FlowEditor Error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="error-boundary" style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100vh',
          background: 'linear-gradient(135deg, #1e293b, #0f172a)',
          color: '#f1f5f9',
          fontFamily: 'Space Grotesk, sans-serif',
          textAlign: 'center',
          padding: '2rem',
        }}>
          <h1 style={{
            fontSize: '2rem',
            marginBottom: '1rem',
            color: '#ef4444'
          }}>
            Oops! Something went wrong
          </h1>

          <p style={{
            marginBottom: '2rem',
            opacity: 0.8,
            maxWidth: '500px'
          }}>
            The Flow Editor encountered an unexpected error.
            Your work has been auto-saved and you can refresh the page to continue.
          </p>

          <div style={{
            background: 'rgba(30, 41, 59, 0.8)',
            border: '1px solid rgba(148, 163, 184, 0.2)',
            borderRadius: '8px',
            padding: '1rem',
            marginBottom: '2rem',
            width: '100%',
            maxWidth: '600px',
            textAlign: 'left',
            fontFamily: 'DM Mono, monospace',
            fontSize: '0.875rem',
          }}>
            <h3 style={{ marginBottom: '0.5rem', color: '#f87171' }}>
              Error Details:
            </h3>
            <pre style={{
              margin: 0,
              whiteSpace: 'pre-wrap',
              color: '#fbbf24',
              maxHeight: '150px',
              overflow: 'auto'
            }}>
              {this.state.error?.message}
            </pre>

            {process.env.NODE_ENV === 'development' && this.state.errorInfo && (
              <details style={{ marginTop: '1rem' }}>
                <summary style={{ cursor: 'pointer', color: '#60a5fa' }}>
                  Stack Trace (Development)
                </summary>
                <pre style={{
                  margin: '0.5rem 0 0 0',
                  whiteSpace: 'pre-wrap',
                  color: '#94a3b8',
                  fontSize: '0.75rem',
                  maxHeight: '200px',
                  overflow: 'auto'
                }}>
                  {this.state.errorInfo.componentStack}
                </pre>
              </details>
            )}
          </div>

          <div style={{ display: 'flex', gap: '1rem' }}>
            <button
              onClick={() => window.location.reload()}
              style={{
                background: '#3b82f6',
                color: 'white',
                border: 'none',
                padding: '0.75rem 1.5rem',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '1rem',
                fontWeight: '500',
              }}
            >
              Refresh Page
            </button>

            <button
              onClick={() => this.setState({ hasError: false, error: undefined, errorInfo: undefined })}
              style={{
                background: 'rgba(30, 41, 59, 0.8)',
                color: '#f1f5f9',
                border: '1px solid rgba(148, 163, 184, 0.3)',
                padding: '0.75rem 1.5rem',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '1rem',
                fontWeight: '500',
              }}
            >
              Try Again
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}