/**
 * Host capabilities for numbl's two browser workers: the IDE worker
 * (numbl-worker.ts) and the `numbl/browser` session worker
 * (browser/worker.ts). Both build their host here, so they differ only in
 * the options they pass:
 *
 * | Capability            | IDE worker                | Session worker            |
 * | --------------------- | ------------------------- | ------------------------- |
 * | WebGPU helper (`gpu`) | yes                       | yes                       |
 * | qhull geometry        | yes (`ensureQhullBackend`) | no (see below)           |
 * | wasm LAPACK provider  | when the IDE settings name one (`set_wasm_bridge`) | no |
 *
 * The session worker is bundled standalone by esbuild and started from a
 * Blob URL, so it has no asset URL to fetch qhull.wasm from, and inlining it
 * would add about 0.5 MB to every embedding page. The wasm accelerator needs
 * a hosting URL that only the IDE settings provide. Either could become a
 * `createNumblSession` option.
 */

import { addPending, createHost, type NumblHost } from "./numbl-core/host.js";
import { startGpuHelper } from "./gpuHelper.js";
import { GpuDeviceSet } from "./numbl-core/runtime/gpu/device.js";
import { defaultGpuDevices } from "./numbl-core/runtime/gpu/cpuDevice.js";
import { WebGpuDevice } from "./numbl-core/runtime/gpu/webgpuDevice.js";

export interface BrowserHostOptions {
  /** Install geometry backends into the host; `host.ready` waits for it.
   *  The IDE worker passes `ensureQhullBackend`. */
  geometry?: (host: NumblHost) => Promise<void>;
  /** Start the WebGPU helper (default true). Must happen at worker boot: a
   *  worker created during a synchronous run would not start until the run
   *  ends. */
  gpu?: boolean;
}

/** Build a browser worker's host. Await `host.ready` (or use
 *  `deferUntilReady`) before the first run. */
export function bootBrowserHost(opts: BrowserHostOptions = {}): NumblHost {
  const gpu = opts.gpu !== false ? startGpuHelper() : null;
  // The WebGPU device (when the helper finds an adapter) comes first, as
  // MATLAB's default device is the GPU; the CPU reference devices follow.
  const host = createHost({
    gpu,
    gpuDevices: new GpuDeviceSet(() => {
      const d = gpu && WebGpuDevice.probe(gpu);
      return d ? [d, ...defaultGpuDevices()] : defaultGpuDevices();
    }),
  });
  if (gpu) addPending(host, gpu.ready);
  if (opts.geometry) addPending(host, opts.geometry(host));
  return host;
}

/** Wrap a worker message handler so that messages arriving before
 *  `host.ready` are queued and then handled in order. */
export function deferUntilReady<T>(
  host: NumblHost,
  handle: (msg: T) => void
): (e: MessageEvent<T>) => void {
  let ready = false;
  const queue: T[] = [];
  void host.ready.then(() => {
    ready = true;
    for (const msg of queue.splice(0)) handle(msg);
  });
  return e => {
    if (ready) handle(e.data);
    else queue.push(e.data);
  };
}
