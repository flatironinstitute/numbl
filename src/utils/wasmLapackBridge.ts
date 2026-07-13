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
const ENABLED_KEY = "numbl_wasm_bridge_enabled";

/** The configured endpoint base URL (or the default if unset). */
export function getWasmBridgeUrl(): string {
  return localStorage.getItem(URL_KEY) || DEFAULT_WASM_BRIDGE_URL;
}

export function setWasmBridgeUrl(url: string): void {
  localStorage.setItem(URL_KEY, url);
}

/** Whether the browser should auto-load the accelerator. Enabled by default
 *  (unset key ⇒ enabled); an explicit "false" opts out. */
export function isWasmBridgeEnabled(): boolean {
  return localStorage.getItem(ENABLED_KEY) !== "false";
}

export function setWasmBridgeEnabled(enabled: boolean): void {
  localStorage.setItem(ENABLED_KEY, String(enabled));
}

/** The URL to hand the worker: the endpoint when enabled, else null
 *  (which tells the worker to uninstall / not load). */
export function effectiveWasmBridgeUrl(): string | null {
  return isWasmBridgeEnabled() ? getWasmBridgeUrl() : null;
}
