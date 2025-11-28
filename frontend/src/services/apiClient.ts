import type { FlowDocument } from '../types';

export interface ApiResponse<T> {
  data: T;
  message?: string;
  success: boolean;
}

export interface FlowExecutionResult {
  final_output?: string;
  events: any[];
  transcript: any[];
  tool_results: any[];
  metrics?: any;
}

export interface ValidationError {
  path: string;
  message: string;
  severity: 'error' | 'warning' | 'info';
}

class ApiClient {
  private baseURL: string;
  private apiKey: string | null = null;

  constructor(baseURL: string = '/api') {
    this.baseURL = baseURL;
  }

  setApiKey(key: string) {
    this.apiKey = key;
    localStorage.setItem('denkwerk_api_key', key);
  }

  getApiKey(): string | null {
    if (!this.apiKey) {
      this.apiKey = localStorage.getItem('denkwerk_api_key');
    }
    return this.apiKey;
  }

  clearApiKey() {
    this.apiKey = null;
    localStorage.removeItem('denkwerk_api_key');
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    const url = `${this.baseURL}${endpoint}`;

    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      ...options.headers,
    };

    if (this.getApiKey()) {
      headers['Authorization'] = `Bearer ${this.getApiKey()}`;
    }

    try {
      const response = await fetch(url, {
        ...options,
        headers,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      if (error instanceof Error) {
        throw new Error(`API Error: ${error.message}`);
      }
      throw new Error('Unknown API error');
    }
  }

  // Flow Management
  async createFlow(flow: FlowDocument): Promise<ApiResponse<{ id: string }>> {
    return this.request('/flows', {
      method: 'POST',
      body: JSON.stringify(flow),
    });
  }

  async updateFlow(id: string, flow: FlowDocument): Promise<ApiResponse<FlowDocument>> {
    return this.request(`/flows/${id}`, {
      method: 'PUT',
      body: JSON.stringify(flow),
    });
  }

  async getFlow(id: string): Promise<ApiResponse<FlowDocument>> {
    return this.request(`/flows/${id}`);
  }

  async listFlows(): Promise<ApiResponse<FlowDocument[]>> {
    return this.request('/flows');
  }

  async deleteFlow(id: string): Promise<ApiResponse<void>> {
    return this.request(`/flows/${id}`, {
      method: 'DELETE',
    });
  }

  // Flow Validation
  async validateFlow(flow: FlowDocument): Promise<ApiResponse<ValidationError[]>> {
    return this.request('/flows/validate', {
      method: 'POST',
      body: JSON.stringify(flow),
    });
  }

  // Flow Execution
  async executeFlow(
    flow: FlowDocument,
    input: string,
    context?: Record<string, any>
  ): Promise<ApiResponse<FlowExecutionResult>> {
    // Map frontend FlowDocument to backend expected format (flattened nodes)
    const backendFlow = {
      ...flow,
      flows: flow.flows.map(f => ({
        ...f,
        nodes: f.nodes.map(n => ({
          ...n.base,
          ...n.kind,
        }))
      }))
    };

    return this.request('/flows/execute', {
      method: 'POST',
      body: JSON.stringify({
        flow: backendFlow,
        input,
        context,
      }),
    });
  }

  async streamFlowExecution(
    flow: FlowDocument,
    input: string,
    context?: Record<string, any>
  ): Promise<ReadableStream<Uint8Array>> {
    const url = `${this.baseURL}/flows/execute/stream`;
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(this.getApiKey() && { 'Authorization': `Bearer ${this.getApiKey()}` }),
      },
      body: JSON.stringify({
        flow,
        input,
        context,
      }),
    });

    if (!response.ok || !response.body) {
      throw new Error('Failed to start stream execution');
    }

    return response.body;
  }

  // Flow Testing
  async testFlow(
    flow: FlowDocument,
    input: string
  ): Promise<ApiResponse<{
    valid: boolean;
    errors: ValidationError[];
    warnings: ValidationError[];
  }>> {
    return this.request('/flows/test', {
      method: 'POST',
      body: JSON.stringify({ flow, input }),
    });
  }

  // Template Management
  async getTemplates(): Promise<ApiResponse<FlowDocument[]>> {
    return this.request('/templates');
  }

  async createTemplate(flow: FlowDocument, name: string, description?: string): Promise<ApiResponse<{ id: string }>> {
    return this.request('/templates', {
      method: 'POST',
      body: JSON.stringify({ flow, name, description }),
    });
  }

  // Analytics and Metrics
  async getFlowMetrics(id: string): Promise<ApiResponse<{
    total_executions: number;
    success_rate: number;
    average_duration: number;
    token_usage: number;
  }>> {
    return this.request(`/flows/${id}/metrics`);
  }

  async getExecutionHistory(id: string, limit = 10): Promise<ApiResponse<FlowExecutionResult[]>> {
    return this.request(`/flows/${id}/history?limit=${limit}`);
  }

  // Authentication
  async authenticate(username: string, password: string): Promise<ApiResponse<{ token: string }>> {
    return this.request('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    });
  }

  async refreshToken(): Promise<ApiResponse<{ token: string }>> {
    return this.request('/auth/refresh');
  }

  async logout(): Promise<ApiResponse<void>> {
    return this.request('/auth/logout', {
      method: 'POST',
    });
  }

  // File Upload/Download
  async uploadFlow(file: File): Promise<ApiResponse<FlowDocument>> {
    const formData = new FormData();
    formData.append('file', file);

    return this.request('/flows/upload', {
      method: 'POST',
      headers: {}, // Let browser set Content-Type for multipart
      body: formData,
    });
  }

  async downloadFlow(id: string, format: 'yaml' | 'json' = 'yaml'): Promise<Blob> {
    const response = await fetch(`${this.baseURL}/flows/${id}/download?format=${format}`, {
      headers: {
        ...(this.getApiKey() && { 'Authorization': `Bearer ${this.getApiKey()}` }),
      },
    });

    if (!response.ok) {
      throw new Error('Failed to download flow');
    }

    return response.blob();
  }

  // WebSocket connection for real-time updates
  createWebSocket(path: string = '/ws'): WebSocket {
    const wsURL = this.baseURL.replace(/^http/, 'ws') + path;
    const ws = new WebSocket(wsURL);

    if (this.getApiKey()) {
      ws.addEventListener('open', () => {
        ws.send(JSON.stringify({ type: 'auth', token: this.getApiKey() }));
      });
    }

    return ws;
  }
}

// Create singleton instance
export const apiClient = new ApiClient(
  process.env.NODE_ENV === 'production'
    ? 'https://api.denkwerk.com'
    : 'http://localhost:3002/api'
);

// Hook for using API client in React components
import { useCallback } from 'react';

export function useApiClient() {
  return {
    ...apiClient,
    authenticated: !!apiClient.getApiKey(),
    setApiKey: useCallback(apiClient.setApiKey.bind(apiClient), []),
    clearApiKey: useCallback(apiClient.clearApiKey.bind(apiClient), []),
  };
}