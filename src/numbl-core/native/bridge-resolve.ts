/**
 * Centralized LAPACK bridge resolution with fallback and optional one-time
 * logging (enabled by the CLI under --verbose).
 *
 * Tries the native bridge first; falls back to ts-lapack (pure TypeScript).
 */

import { getLapackBridge, type LapackBridge } from "./lapack-bridge.js";
import { getTsLapackBridge } from "./ts-lapack-bridge.js";

const _logged = new Set<string>();
let _log: ((message: string) => void) | null = null;

/** Set (or clear) the sink for the per-op "using bridge" diagnostic. */
export function setBridgeLog(log: ((message: string) => void) | null): void {
  _log = log;
}

/**
 * Get the effective LAPACK bridge for a given operation.
 *
 * @param opName  Operation name for logging (e.g. "inv", "qr")
 * @param method  Optional method to check on native bridge. If native bridge
 *                lacks this method, falls back to ts-lapack.
 */
export function getEffectiveBridge(
  opName: string,
  method?: keyof LapackBridge
): LapackBridge {
  const native = getLapackBridge();
  const bridge =
    method && native
      ? native[method]
        ? native
        : getTsLapackBridge()
      : (native ?? getTsLapackBridge());
  if (_log && !_logged.has(opName)) {
    _logged.add(opName);
    const name = native ? "native LAPACK addon" : "ts-lapack (TypeScript)";
    _log(`[${opName}] using bridge: ${name}`);
  }
  return bridge;
}
