/**
 * Plugin registration entry points.
 *
 * A "plugin" here is just a function that registers one or more
 * executors with the registry. Different `--opt` modes call different
 * subsets of these. The browser bundle simply omits modules whose
 * dependencies it can't satisfy (e.g., the C-kernel plugin is
 * Node-only).
 *
 * For the initial skeleton, only the interpreter plugin exists. As
 * executors are ported, each gets its own plugin module that lands
 * here.
 */

import type { Registry } from "./registry.js";
import { interpreterExecutor } from "./interpreterExecutor.js";

/** Always-on baseline. The interpreter executor is the last-resort
 *  fallback that every mode needs. */
export function registerInterpreterPlugin(registry: Registry): void {
  registry.register(interpreterExecutor);
}
