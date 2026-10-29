/**
 * Host capabilities: what the platform running numbl provides (file I/O,
 * system info, native libraries, numeric providers, geometry backends, the
 * C-JIT, a GPU helper, the gpuArray devices). A run gets its host from `ExecOptions.host` (an
 * `ExecSession` passes its own), else the process-default host, and exposes
 * it as `rt.host`; helpers without a runtime handle use `currentHost()`.
 *
 * Node hosts come from `initNodePlatform` (src/node-platform.ts), browser
 * workers from `bootBrowserHost` (src/browserHost.ts). The process-default
 * host is what the older process-wide setters (`registerProvider`,
 * `setDelaunayBackend`, `setJsGpuHelper`, `registerNodeCompileC`) edit, so
 * code that never passes a host keeps working.
 *
 * Kept a leaf (type imports aside, only the provider stack, refcount and the
 * GPU device set), so that the runtime can import it.
 */

import type { FileIOAdapter } from "./fileIOAdapter.js";
import type { SystemAdapter } from "./systemAdapter.js";
import type { NativeBridge } from "./workspace/types.js";
import { ProviderStack, defaultProviderStack } from "./native/lapack-bridge.js";
import { getCurrentRuntime } from "./runtime/refcount.js";
import { GpuDeviceSet } from "./runtime/gpu/device.js";
import { defaultGpuDevices } from "./runtime/gpu/cpuDevice.js";

/**
 * Compute a Delaunay triangulation of `points` (an array of dim-dimensional
 * tuples). Returns an array of simplices, each a list of `dim+1` 0-based input
 * point indices.
 */
export type DelaunayBackend = (points: number[][], dim: number) => number[][];

/**
 * Compute the convex hull of `points` (an array of dim-dimensional tuples).
 * Returns an array of simplicial facets, each a list of `dim` 0-based input
 * point indices (edges in 2-D, triangles in 3-D).
 */
export type ConvexHullBackend = (points: number[][], dim: number) => number[][];

export interface GeometryBackends {
  delaunay?: DelaunayBackend;
  convexHull?: ConvexHullBackend;
}

export interface NumblHost {
  /** Default file I/O for runs that do not pass `ExecOptions.fileIO`. */
  fileIO?: FileIOAdapter;
  /** Default system adapter for runs that do not pass `ExecOptions.system`. */
  system?: SystemAdapter;
  /** koffi bridge for native shared libraries (`.numbl.js` `// native:`)
   *  and the C-JIT. Node only. */
  nativeBridge?: NativeBridge;
  /** The numeric provider stack behind `getLapackBridge()`. */
  readonly numeric: ProviderStack;
  /** Backends for delaunay/convhull (qhull); builtins error without them. */
  readonly geometry: GeometryBackends;
  /** The C backend for optimization "2" (a `JitBackend`, opaque here to keep
   *  this module a leaf); needs `nativeBridge` too. Node only. */
  cjit?: object;
  /** The `gpu` object passed to `.numbl.js` files (see src/gpuHelper.ts);
   *  null or undefined when there is no WebGPU. */
  gpu?: unknown;
  /** The devices behind gpuArray and which one is selected. */
  readonly gpuDevices: GpuDeviceSet;
  /** Resolves once the capabilities that load asynchronously (qhull, the
   *  GPU helper, a wasm accelerator) are installed or have failed to load.
   *  Hosts await it before the first run. */
  ready: Promise<void>;
}

export interface NumblHostInit {
  fileIO?: FileIOAdapter;
  system?: SystemAdapter;
  nativeBridge?: NativeBridge;
  /** Default: a fresh stack with only the built-in inline provider. */
  numeric?: ProviderStack;
  geometry?: GeometryBackends;
  cjit?: object;
  gpu?: unknown;
  /** Default: the CPU reference devices (`defaultGpuDevices`). */
  gpuDevices?: GpuDeviceSet;
  ready?: Promise<void>;
}

/** A new host. Nothing is inherited from the process default: a field left
 *  out is a capability the host does not have. */
export function createHost(init: NumblHostInit = {}): NumblHost {
  return {
    ...init,
    numeric: init.numeric ?? new ProviderStack(),
    geometry: { ...init.geometry },
    gpuDevices: init.gpuDevices ?? new GpuDeviceSet(defaultGpuDevices()),
    ready: init.ready ?? Promise.resolve(),
  };
}

let _defaultHost: NumblHost | null = null;

/** The process-default host, used by runs that are not given one. */
export function defaultHost(): NumblHost {
  return (_defaultHost ??= createHost({ numeric: defaultProviderStack }));
}

/** The host of the running code: the active runtime's, else the process
 *  default. For helpers without a runtime handle; builtins use `ctx.rt.host`. */
export function currentHost(): NumblHost {
  const rt = getCurrentRuntime() as { host?: NumblHost } | null;
  return rt?.host ?? defaultHost();
}

/** Add `p` to what `host.ready` waits for. */
export function addPending(host: NumblHost, p: Promise<unknown>): void {
  host.ready = Promise.all([host.ready, p.catch(() => {})]).then(() => {});
}
