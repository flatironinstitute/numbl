/**
 * Settings and health check for the execution server (`numbl serve`). Runs
 * go through `RemoteExecutionBackend` (src/execution/).
 */

export interface RemoteServiceHealth {
  status: string;
  activeExecutions: number;
  maxConcurrentExecutions: number;
  nativeAddon?: boolean | null;
}

const DEFAULT_REMOTE_SERVICE_URL = "http://localhost:3001";

/**
 * Get the remote service URL from localStorage or default
 */
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
