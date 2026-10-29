/**
 * The C-JIT capability for Node hosts. `nodeCJitBackend()` is what
 * `initNodePlatform` puts in `NumblHost.cjit`; importing this module (Node
 * only) is what brings the C compile/codegen graph and the ~300 KB C
 * runtime-snippet table in, which keeps them out of the browser bundle.
 */

import { setCSnippets } from "../../jit/codegen/runtime.js";
import { C_SNIPPETS } from "../../jit/builtins/runtime/snippets.c.gen.js";
import { defaultHost } from "../../host.js";
import { cBackend } from "./cBackend.js";

/** The C backend, for `NumblHost.cjit`. The snippet table is static C
 *  source shared by every host, so it is installed process-wide. */
export function nodeCJitBackend(): object {
  setCSnippets(C_SNIPPETS);
  return cBackend;
}

/** Give the process-default host the C backend.
 *  @deprecated Use `initNodePlatform` (numbl/node), or set `host.cjit` to
 *  `nodeCJitBackend()`. */
export function registerNodeCompileC(): void {
  defaultHost().cjit = nodeCJitBackend();
}
