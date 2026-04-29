/**
 * Node-only entry that wires the C-JIT executors into the shared
 * `plugins.ts` registrar slot. Importing this module for its side
 * effect (from `cli.ts`) is what makes `--opt e3` work in the CLI.
 *
 * The browser worker bundle deliberately does not import this file,
 * keeping `compile.ts` (and its `node:fs`/`node:os`/`node:child_process`
 * imports) out of the web build's module graph.
 */

import { setCJitRegistrar } from "../plugins.js";
import { cJitLoopExecutor } from "./loopExecutor.js";
import { cJitFuseExecutor } from "./fuseExecutor.js";

setCJitRegistrar(registry => {
  registry.register(cJitLoopExecutor);
  registry.register(cJitFuseExecutor);
});
