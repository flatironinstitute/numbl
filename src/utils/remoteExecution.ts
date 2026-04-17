/**
 * Remote execution service client
 */

import type { PlotInstruction } from "../graphics/types.js";

export interface RemoteExecutionFile {
  name: string;
  content: string;
}

export interface RemoteExecutionRequest {
  files: RemoteExecutionFile[];
  mainScript: string;
  optimization?: number;
  fuse?: boolean;
}

export interface RemoteExecutionResult {
  success: boolean;
  output?: string;
  error?: string;
  timedOut?: boolean;
  generatedJS?: string;
  generatedC?: string;
}

export interface RemoteServiceHealth {
  status: string;
  activeExecutions: number;
  maxConcurrentExecutions: number;
  nativeAddon?: boolean | null;
}

/**
 * Get the remote service URL from localStorage or default
 */
export const DEFAULT_REMOTE_SERVICE_URL = "http://localhost:3001";

export function getRemoteServiceUrl(): string {
  return (
    localStorage.getItem("numbl_remote_service_url") ||
    DEFAULT_REMOTE_SERVICE_URL
  );
}

/**
 * Set the remote service URL in localStorage
 */
export function setRemoteServiceUrl(url: string): void {
  localStorage.setItem("numbl_remote_service_url", url);
}

/**
 * Get whether remote execution is enabled
 */
export function isRemoteExecutionEnabled(): boolean {
  return localStorage.getItem("numbl_remote_execution_enabled") === "true";
}

/**
 * Set whether remote execution is enabled
 */
export function setRemoteExecutionEnabled(enabled: boolean): void {
  localStorage.setItem("numbl_remote_execution_enabled", String(enabled));
}

/**
 * Check if the remote service is available
 */
export async function checkRemoteServiceHealth(
  serviceUrl?: string,
  passkey?: string
): Promise<RemoteServiceHealth | null> {
  const url = serviceUrl || getRemoteServiceUrl();
  try {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (passkey) headers["Authorization"] = `Bearer ${passkey}`;
    const response = await fetch(`${url}/health`, {
      method: "GET",
      headers,
    });

    if (!response.ok) {
      return null;
    }

    return await response.json();
  } catch (error) {
    console.error("Failed to check remote service health:", error);
    return null;
  }
}

/**
 * Execute code on the remote service with real-time streaming of output and plot instructions.
 */
export async function executeRemoteStream(
  request: RemoteExecutionRequest,
  callbacks: {
    onOutput: (text: string) => void;
    onDrawnow: (plotInstructions: PlotInstruction[]) => void;
  },
  serviceUrl?: string,
  abortSignal?: AbortSignal,
  passkey?: string
): Promise<RemoteExecutionResult> {
  const url = serviceUrl || getRemoteServiceUrl();
  try {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (passkey) headers["Authorization"] = `Bearer ${passkey}`;
    const response = await fetch(`${url}/execute`, {
      method: "POST",
      headers,
      body: JSON.stringify(request),
      signal: abortSignal,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return {
        success: false,
        error: errorData.error || `Server error: ${response.status}`,
      };
    }

    // Read the SSE stream
    const reader = response.body?.getReader();
    if (!reader) {
      return { success: false, error: "No response body" };
    }

    const decoder = new TextDecoder();
    let buffer = "";
    let result: RemoteExecutionResult = { success: true };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Parse SSE events: "data: <json>\n\n"
      const events = buffer.split("\n\n");
      // Keep the last (possibly incomplete) chunk in the buffer
      buffer = events.pop() || "";

      for (const event of events) {
        const dataLine = event.trim();
        if (!dataLine.startsWith("data: ")) continue;
        const json = dataLine.slice(6);
        try {
          const msg = JSON.parse(json);
          if (msg.type === "output") {
            callbacks.onOutput(msg.text);
          } else if (msg.type === "drawnow") {
            callbacks.onDrawnow(msg.plotInstructions);
          } else if (msg.type === "done") {
            result = {
              ...result,
              generatedJS: msg.generatedJS,
              generatedC: msg.generatedC,
            };
          } else if (msg.type === "error") {
            result = {
              success: false,
              error: msg.message,
              timedOut: msg.timedOut,
            };
          } else if (msg.type === "close") {
            result = { ...result, success: msg.success };
          }
        } catch {
          // Skip unparseable events
        }
      }
    }

    return result;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      return { success: false, error: "Execution aborted" };
    }
    console.error("Remote execution failed:", error);
    return {
      success: false,
      error:
        error instanceof Error
          ? `Connection failed: ${error.message}`
          : "Unknown error occurred",
    };
  }
}
