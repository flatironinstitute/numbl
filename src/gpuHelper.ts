/// <reference types="@webgpu/types" />
/**
 * Synchronous WebGPU access for .numbl.js functions (browser worker only).
 *
 * Scripts run synchronously in the numbl worker, but WebGPU is async. A helper
 * worker owns the GPU device; `gpu.call` is a request on the blocking channel
 * (numbl-core/runtime/blockingChannel.ts), so it waits until the helper
 * answers or Stop is pressed, and typed arrays cross through shared memory.
 *
 * The helper must be spawned at worker boot: browsers do not start a worker
 * until its creating thread returns to the event loop, so spawning it from
 * inside a run (which never yields) would deadlock.
 *
 * .numbl.js usage (the injected `gpu` global is null when unavailable):
 *   gpu.load("name", function () { return { method: async (device, args) => result }; });
 *   var result = gpu.call("name", "method", args);
 * The factory runs inside the helper, so it must be self-contained; its state
 * (e.g. GPU buffers) persists across calls and runs until the source changes,
 * when an optional dispose() method is called. Arguments and results may be
 * numbers, strings, booleans, null, arrays, plain objects and typed arrays.
 * A call abandoned by Stop still runs to completion in the helper.
 */

import {
  BlockingChannel,
  channelServerSource,
  type ChannelServer,
  type ChannelServerHandlers,
  type ChannelTarget,
} from "./numbl-core/runtime/blockingChannel.js";
import { getCurrentRuntime } from "./numbl-core/runtime/refcount.js";

export interface GpuHelper {
  load(name: string, factory: (() => unknown) | string): void;
  call(name: string, method: string, args?: unknown): unknown;
  /** Call a method without waiting for it. Calls run in the order they are
   *  made, posted or not; an error from a posted call is raised by the next
   *  `call`. */
  post(name: string, method: string, args?: unknown): void;
  /** Release the GPU device, waiting until it is gone. */
  shutdown(): void;
  /** Resolves when the helper worker has started (or failed to). A run
   *  before then would find it unstarted, so hosts await this first (it is
   *  part of the host's `ready`). */
  readonly ready: Promise<void>;
}

interface CallBody {
  name: string;
  method: string;
  args: unknown;
}

// Runs in the helper worker; serialized with toString(), so self-contained.
function helperMain(
  createChannelServer: (h: ChannelServerHandlers) => ChannelServer
) {
  const modules = new Map<
    string,
    { source: string; mod?: Record<string, unknown>; error?: string }
  >();
  let devP: Promise<GPUDevice> | null = null;
  function device(): Promise<GPUDevice> {
    if (!devP) {
      devP = navigator.gpu.requestAdapter().then(a => {
        if (!a) throw new Error("gpu: no WebGPU adapter");
        return a.requestDevice({
          requiredLimits: {
            maxStorageBufferBindingSize: a.limits.maxStorageBufferBindingSize,
            maxBufferSize: a.limits.maxBufferSize,
          },
        });
      });
    }
    return devP;
  }
  function load(name: string, source: string) {
    const prev = modules.get(name);
    if (prev && prev.source === source) return;
    // A module that keeps GPU resources across calls releases them in
    // dispose() when it is replaced by a new version.
    const dispose = prev?.mod?.dispose;
    if (typeof dispose === "function") {
      try {
        dispose();
      } catch {
        // ignore
      }
    }
    try {
      const mod = new Function("return (" + source + ")();")();
      modules.set(name, { source, mod });
    } catch (err) {
      modules.set(name, { source, error: String(err) });
    }
  }
  async function call(m: CallBody) {
    const entry = modules.get(m.name);
    if (!entry) throw new Error("gpu: module '" + m.name + "' not loaded");
    if (entry.error) throw new Error(entry.error);
    const fn = entry.mod![m.method];
    if (typeof fn !== "function")
      throw new Error("gpu: '" + m.name + "' has no method " + m.method);
    return await fn(await device(), m.args);
  }
  // Calls run one at a time, in the order they arrive, so a posted call
  // (an upload, a kernel) is done before a later request reads its result.
  let queue: Promise<unknown> = Promise.resolve();
  function enqueue<T>(f: () => Promise<T>): Promise<T> {
    const p = queue.then(f);
    queue = p.catch(() => undefined);
    return p;
  }
  // The first error of a posted call, raised by the next request.
  let postError: string | null = null;
  const message = (err: unknown) =>
    String((err as Error)?.stack || (err as Error)?.message || err);
  const server = createChannelServer({
    post(cmd, body) {
      if (cmd === "load") {
        const b = body as { name: string; source: string };
        enqueue(async () => load(b.name, b.source));
      } else if (cmd === "call") {
        enqueue(() => call(body as CallBody)).catch(err => {
          if (postError === null) postError = message(err);
        });
      }
    },
    request(cmd, body, reply) {
      if (cmd === "shutdown") {
        // Release the modules' resources and the device, and wait until
        // the device is gone.
        enqueue(async () => {
          for (const entry of modules.values()) {
            const dispose = entry.mod?.dispose;
            if (typeof dispose === "function") dispose();
          }
          if (devP) {
            const d = await devP.catch(() => null);
            if (d) {
              d.destroy();
              await d.lost;
            }
          }
        }).then(
          () => reply.ok(null),
          err => reply.error(message(err))
        );
        return;
      }
      if (cmd !== "call") {
        reply.error("gpu: unknown request " + cmd);
        return;
      }
      enqueue(() => {
        if (postError !== null) {
          const e = postError;
          postError = null;
          throw new Error(e);
        }
        return call(body as CallBody);
      }).then(
        result => reply.ok(result),
        err => reply.error(message(err))
      );
    },
  });
  self.onmessage = (e: MessageEvent) => {
    const m = e.data;
    if (m && m.type === "init") {
      const ready = new Int32Array(m.ready);
      Atomics.store(ready, 0, 1);
      Atomics.notify(ready, 0);
      self.postMessage({ type: "ready" });
    } else {
      server.handle(m);
    }
  };
}

/** The helper's source: `helperMain` started with the channel server.
 *  Runs where `self` receives messages and `navigator.gpu` is WebGPU (a
 *  browser worker, or a Node worker that provides both). */
export function helperSource(): string {
  return `(${helperMain.toString()})(${channelServerSource()});\n`;
}

/**
 * Spawn the GPU helper. Call at worker boot, before any run. Returns null when
 * WebGPU or SharedArrayBuffer (cross-origin isolation) is unavailable.
 */
export function startGpuHelper(): GpuHelper | null {
  if (
    typeof navigator === "undefined" ||
    !(navigator as Navigator & { gpu?: unknown }).gpu ||
    typeof SharedArrayBuffer === "undefined" ||
    typeof Worker === "undefined"
  ) {
    return null;
  }
  const url = URL.createObjectURL(
    new Blob([helperSource()], { type: "text/javascript" })
  );
  const worker = new Worker(url);
  const ready = new Int32Array(new SharedArrayBuffer(4));
  const started = new Promise<void>(resolve => {
    worker.addEventListener("message", e => {
      if ((e.data as { type?: string } | null)?.type === "ready") resolve();
    });
    worker.addEventListener("error", () => resolve());
  });
  worker.postMessage({ type: "init", ready: ready.buffer });
  return gpuHelperClient(worker, ready, started);
}

/** The client side of a helper started with `helperSource()` and sent
 *  `{ type: "init", ready }`: `ready[0]` becomes 1 once it runs. */
export function gpuHelperClient(
  worker: ChannelTarget,
  ready: Int32Array,
  started: Promise<void>
): GpuHelper {
  // The running code's channel, so Stop interrupts a call; a private one
  // outside a run.
  let fallback: BlockingChannel | null = null;
  function channel(): BlockingChannel {
    const rt = getCurrentRuntime() as {
      blocking?: BlockingChannel | null;
    } | null;
    return rt?.blocking ?? (fallback ??= new BlockingChannel(null));
  }

  return {
    ready: started,
    load(name, factory) {
      const source =
        typeof factory === "function" ? factory.toString() : factory;
      channel().post(worker, "load", { name, source });
    },
    call(name, method, args) {
      if (Atomics.load(ready, 0) !== 1) {
        throw new Error(
          "gpu: helper worker has not started (await host.ready before running)"
        );
      }
      return channel().request(worker, "call", { name, method, args });
    },
    post(name, method, args) {
      channel().post(worker, "call", { name, method, args });
    },
    shutdown() {
      if (Atomics.load(ready, 0) === 1) channel().request(worker, "shutdown");
    },
  };
}
