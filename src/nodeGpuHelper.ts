/**
 * The GPU helper in Node: the browser helper (gpuHelper.ts) run in a
 * worker_threads worker, with WebGPU from Dawn (the `webgpu` npm package).
 * The package is optional and not a dependency of numbl (it is large):
 * without it, `startNodeGpuHelper` returns null. Unlike the browser helper,
 * this one can be started during a run (a Node worker starts without the
 * creating thread returning to its event loop), so it is started on first
 * use.
 */

import { Worker } from "node:worker_threads";
import { createRequire } from "node:module";
import { gpuHelperClient, helperSource, type GpuHelper } from "./gpuHelper.js";

/** Start the helper and wait for it to run; null when the `webgpu` package
 *  is not installed or the worker does not start. */
export function startNodeGpuHelper(): GpuHelper | null {
  let webgpuPath: string;
  try {
    webgpuPath = createRequire(import.meta.url).resolve("webgpu");
  } catch {
    return null;
  }
  // Give the helper the browser globals it uses: `self` (messages) and
  // `navigator.gpu`.
  // `__name`: esbuild's keepNames helper, which a function's source text
  // may refer to under tsx.
  const preamble = `
globalThis.__name = f => f;
const { parentPort } = require("node:worker_threads");
const { create, globals } = require(${JSON.stringify(webgpuPath)});
Object.assign(globalThis, globals);
Object.defineProperty(globalThis, "navigator", {
  value: { gpu: create([]) },
  configurable: true,
});
globalThis.self = globalThis;
self.postMessage = m => parentPort.postMessage(m);
parentPort.on("message", data => self.onmessage && self.onmessage({ data }));
`;
  const worker = new Worker(preamble + helperSource(), {
    eval: true,
    stdout: true,
    stderr: true,
  });
  // The helper does not keep the process alive.
  worker.unref();
  const ready = new Int32Array(new SharedArrayBuffer(4));
  worker.postMessage({ type: "init", ready: ready.buffer });
  if (Atomics.wait(ready, 0, 0, 10000) === "timed-out") {
    void worker.terminate();
    return null;
  }
  return gpuHelperClient(worker, ready, Promise.resolve());
}
