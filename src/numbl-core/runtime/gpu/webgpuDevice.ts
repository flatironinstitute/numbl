/**
 * The WebGPU device: gpuArray data in WebGPU buffers, owned by the GPU
 * helper (a browser worker, or a Node worker with Dawn) through its
 * gpuArray module (`webgpuModule.ts`). WebGPU computes in single precision
 * only, so `supportsDouble` is false. Single and logical arrays are stored
 * as f32; a double array can still arise as an operation's result (the
 * index output of `max`), and is kept in host memory so that it keeps its
 * precision.
 *
 * Uploads and frees are posted without waiting; a download waits for the
 * helper. Buffers are freed when their gpuArray is garbage collected.
 */

import { RuntimeTensor, type ElementClass } from "../types.js";
import { allocFloat64Array } from "../alloc.js";
import type { GpuDevice } from "./device.js";
import { gpuArrayModule } from "./webgpuModule.js";

/** What the device needs of a GPU helper (`GpuHelper` in src/gpuHelper.ts). */
export interface GpuHelperLike {
  load(name: string, factory: (() => unknown) | string): void;
  call(name: string, method: string, args?: unknown): unknown;
  post(name: string, method: string, args?: unknown): void;
}

const MODULE = "numbl:gpuarray";

/** A buffer: device buffers by id, or host data for a double array. */
type WebGpuBuffer =
  | { id: number; imagId: number; n: number }
  | { host: Float64Array; hostImag: Float64Array | undefined };

export class WebGpuDevice implements GpuDevice {
  readonly supportsDouble = false;
  generation = 0;
  private nextId = 1;
  private readonly freed = new FinalizationRegistry<number[]>(ids =>
    this.helper.post(MODULE, "free", { ids })
  );

  private constructor(
    private readonly helper: GpuHelperLike,
    readonly name: string,
    readonly totalMemory: number
  ) {}

  /** The device of a helper whose WebGPU has an adapter, else null. */
  static probe(helper: GpuHelperLike): WebGpuDevice | null {
    try {
      // A bundler may make the function's source refer to `__name`.
      helper.load(
        MODULE,
        `function () { const __name = f => f; return (${gpuArrayModule.toString()})(); }`
      );
      const info = helper.call(MODULE, "info") as {
        name: string;
        maxBufferSize: number;
      };
      return new WebGpuDevice(helper, info.name, info.maxBufferSize);
    } catch {
      return null;
    }
  }

  upload(t: RuntimeTensor): WebGpuBuffer {
    if (t.cls === "double")
      return {
        host: allocFloat64Array(t.data),
        hostImag: t.imag && allocFloat64Array(t.imag),
      };
    const id = this.send(t.data);
    const imagId = t.imag ? this.send(t.imag) : 0;
    const b = { id, imagId, n: t.data.length };
    this.freed.register(b, imagId ? [id, imagId] : [id]);
    return b;
  }

  download(
    buffer: unknown,
    shape: number[],
    cls: ElementClass,
    isComplex: boolean
  ): RuntimeTensor {
    const b = buffer as WebGpuBuffer;
    if ("host" in b)
      return new RuntimeTensor(
        allocFloat64Array(b.host),
        shape.slice(),
        isComplex && b.hostImag ? allocFloat64Array(b.hostImag) : undefined,
        cls
      );
    const re = this.fetch(b.id, b.n);
    const im = isComplex && b.imagId ? this.fetch(b.imagId, b.n) : undefined;
    return new RuntimeTensor(re, shape.slice(), im, cls);
  }

  synchronize(): void {
    this.helper.call(MODULE, "synchronize");
  }

  private send(data: Float64Array): number {
    const id = this.nextId++;
    this.helper.post(MODULE, "upload", { id, data: Float32Array.from(data) });
    return id;
  }

  private fetch(id: number, n: number): Float64Array {
    const f = this.helper.call(MODULE, "download", { id, n }) as Float32Array;
    const out = allocFloat64Array(n);
    out.set(f);
    return out;
  }
}
