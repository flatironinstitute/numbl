/**
 * Centralized LAPACK bridge resolution with fallback and one-time logging.
 *
 * Tries the native bridge first; falls back to ts-lapack (pure TypeScript).
 */

import { getLapackBridge, type LapackBridge } from "./lapack-bridge.js";
import { getTsLapackBridge } from "./ts-lapack-bridge.js";

const _logged = new Set<string>();

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
  const active = getLapackBridge();
  const bridge =
    method && active
      ? active[method]
        ? active
        : getTsLapackBridge()
      : (active ?? getTsLapackBridge());
  if (!_logged.has(opName)) {
    _logged.add(opName);
    const name = active?.bridgeName ?? (active ? "native LAPACK addon" : "ts-lapack (TypeScript)");
    console.log(`[${opName}] using bridge: ${name}`);
  }
  return bridge;
}
