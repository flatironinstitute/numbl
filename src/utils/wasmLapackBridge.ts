/**
 * Browser (main-thread) settings for the optional WASM linear-algebra
 * accelerator bridge. Mirrors the tiny localStorage-backed pattern of
 * `remoteExecution.ts`. Web Workers cannot read localStorage, so the main
 * thread reads these and forwards the effective URL to the worker (see the
 * `set_wasm_bridge` message in numbl-worker.ts).
 */

/** Default endpoint: a single-threaded libFLAME/BLIS matmul bridge served
 *  from GitHub Pages. The base URL of a directory containing
 *  `numbl-bridge.json` and the wasm it names. */
export const DEFAULT_WASM_BRIDGE_URL =
  "https://magland.github.io/numbl-wasm-bridge/";

const URL_KEY = "numbl_wasm_bridge_url";

/** The configured endpoint base URL (or the default if unset). Persisted. */
export function getWasmBridgeUrl(): string {
  return localStorage.getItem(URL_KEY) || DEFAULT_WASM_BRIDGE_URL;
}

export function setWasmBridgeUrl(url: string): void {
  localStorage.setItem(URL_KEY, url);
}

/** Whether the accelerator is enabled. Deliberately in-memory only (NOT
 *  persisted): a page reload always resets it to the default (enabled), so a
 *  temporary disable for A/B comparison can't be accidentally left off for
 *  the next visitor/session. */
let enabled = true;

export function isWasmBridgeEnabled(): boolean {
  return enabled;
}

export function setWasmBridgeEnabled(value: boolean): void {
  enabled = value;
}

/** The URL to hand the worker: the endpoint when enabled, else null
 *  (which tells the worker to uninstall / not load). */
export function effectiveWasmBridgeUrl(): string | null {
  return isWasmBridgeEnabled() ? getWasmBridgeUrl() : null;
}
