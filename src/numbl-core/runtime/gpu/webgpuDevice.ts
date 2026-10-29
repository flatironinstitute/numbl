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
import type {
  GpuDevice,
  GpuKernels,
  KernelOperand,
  KernelSelection,
} from "./device.js";
import type { FusedExpr } from "./deviceTypes.js";
import { gpuArrayModule } from "./webgpuModule.js";

/** What the device needs of a GPU helper (`GpuHelper` in src/gpuHelper.ts). */
export interface GpuHelperLike {
  load(name: string, factory: (() => unknown) | string): void;
  call(name: string, method: string, args?: unknown): unknown;
  post(name: string, method: string, args?: unknown): void;
}

const MODULE = "numbl:gpuarray";

/** `owners` of a buffer that has been freed. */
const FREED = -1;

/** A buffer: device buffers by id, or host data for a double array. */
type WebGpuBuffer =
  | DeviceBuffer
  | { host: Float64Array; hostImag: Float64Array | undefined };

/** A buffer in device memory, and how many gpuArrays use it. */
type DeviceBuffer = { id: number; imagId: number; n: number; owners: number };

export class WebGpuDevice implements GpuDevice, GpuKernels {
  readonly supportsDouble = false;
  readonly kernels: GpuKernels = this;
  generation = 0;
  private nextId = 1;
  /** Frees the buffers of gpuArrays that were never counted (see
   *  `release`) when they are garbage collected. */
  private readonly freed = new FinalizationRegistry<{
    ids: number[];
    bytes: number;
  }>(f => {
    this.bytes -= f.bytes;
    this.free(f.ids);
  });
  private bytes = 0;

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
    const b = { id, imagId, n: t.data.length, owners: 0 };
    this.track(b);
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

  // ── Kernels ─────────────────────────────────────────────────────────

  holds(buffer: unknown): boolean {
    return !("host" in (buffer as WebGpuBuffer));
  }

  elementwise(
    op: string,
    inputs: KernelOperand[],
    n: number,
    dims?: number[]
  ): DeviceBuffer {
    const out = this.result(n);
    this.helper.post(
      MODULE,
      "elementwise",
      this.ewArgs(out, op, inputs, n, dims)
    );
    return out;
  }

  fused(expr: FusedExpr): DeviceBuffer {
    const out = this.result(expr.n);
    this.helper.post(MODULE, "fused", {
      out: out.id,
      n: expr.n,
      inputs: expr.inputs.map(x => ({
        id: (x.buffer as DeviceBuffer).id,
        n: x.n,
      })),
      scalars: expr.scalars,
      nodes: expr.nodes,
    });
    return out;
  }

  elementwiseChecked(
    op: string,
    inputs: KernelOperand[],
    n: number,
    dims?: number[]
  ): { buffer: DeviceBuffer; bad: boolean } {
    const out = this.result(n);
    const r = this.helper.call(MODULE, "elementwise", {
      ...this.ewArgs(out, op, inputs, n, dims),
      check: true,
    }) as { bad: boolean };
    return { buffer: out, bad: r.bad };
  }

  private ewArgs(
    out: DeviceBuffer,
    op: string,
    inputs: KernelOperand[],
    n: number,
    dims?: number[]
  ) {
    return {
      out: out.id,
      op,
      n,
      dims,
      inputs: inputs.map(x =>
        "scalar" in x
          ? { scalar: x.scalar }
          : { id: (x.buffer as DeviceBuffer).id, n: x.n, strides: x.strides }
      ),
    };
  }

  take(src: unknown, sel: KernelSelection): DeviceBuffer {
    const out = this.result(sel.n);
    this.helper.post(MODULE, "take", {
      out: out.id,
      src: (src as DeviceBuffer).id,
      ...sel,
    });
    return out;
  }

  put(
    src: unknown,
    total: number,
    sel: KernelSelection,
    rhs: KernelOperand
  ): DeviceBuffer {
    const out = this.result(total);
    this.helper.post(MODULE, "put", {
      out: out.id,
      src: (src as DeviceBuffer).id,
      total,
      ...sel,
      rhs:
        "scalar" in rhs
          ? { scalar: rhs.scalar }
          : { id: (rhs.buffer as DeviceBuffer).id, n: rhs.n },
    });
    return out;
  }

  reduce(
    op: string,
    src: unknown,
    inner: number,
    len: number,
    outer: number
  ): DeviceBuffer {
    const out = this.result(inner * outer);
    this.helper.post(MODULE, "reduce", {
      out: out.id,
      op,
      src: (src as DeviceBuffer).id,
      inner,
      len,
      outer,
    });
    return out;
  }

  reduceWithIndex(
    op: string,
    src: unknown,
    inner: number,
    len: number,
    outer: number
  ): { value: DeviceBuffer; index: DeviceBuffer } {
    const value = this.result(inner * outer);
    const index = this.result(inner * outer);
    this.helper.post(MODULE, "reduce", {
      out: value.id,
      outIdx: index.id,
      op,
      src: (src as DeviceBuffer).id,
      inner,
      len,
      outer,
    });
    return { value, index };
  }

  scan(
    op: string,
    src: unknown,
    inner: number,
    len: number,
    outer: number
  ): DeviceBuffer {
    const out = this.result(inner * len * outer);
    this.helper.post(MODULE, "scan", {
      out: out.id,
      op,
      src: (src as DeviceBuffer).id,
      inner,
      len,
      outer,
    });
    return out;
  }

  concat(
    total: number,
    outer: number,
    stride: number,
    parts: { src: unknown; block: number; offset: number }[]
  ): DeviceBuffer {
    const out = this.result(total);
    this.helper.post(MODULE, "concat", {
      out: out.id,
      total,
      outer,
      stride,
      parts: parts.map(p => ({
        src: (p.src as DeviceBuffer).id,
        block: p.block,
        offset: p.offset,
      })),
    });
    return out;
  }

  matmul(
    a: unknown,
    b: unknown,
    m: number,
    k: number,
    n: number
  ): DeviceBuffer {
    const out = this.result(m * n);
    this.helper.post(MODULE, "matmul", {
      out: out.id,
      a: (a as DeviceBuffer).id,
      b: (b as DeviceBuffer).id,
      m,
      k,
      n,
    });
    return out;
  }

  transpose(src: unknown, m: number, n: number): DeviceBuffer {
    const out = this.result(m * n);
    this.helper.post(MODULE, "transpose", {
      out: out.id,
      src: (src as DeviceBuffer).id,
      m,
      n,
    });
    return out;
  }

  /** A new buffer id for a kernel's result of `n` elements. */
  private result(n: number): DeviceBuffer {
    const b = { id: this.nextId++, imagId: 0, n, owners: 0 };
    this.track(b);
    return b;
  }

  private track(b: DeviceBuffer): void {
    const ids = b.imagId ? [b.id, b.imagId] : [b.id];
    const bytes = ids.length * Math.max(4, b.n * 4);
    this.bytes += bytes;
    this.freed.register(b, { ids, bytes }, b);
  }

  private free(ids: number[]): void {
    this.helper.post(MODULE, "free", { ids });
  }

  retain(buffer: unknown): void {
    const b = buffer as WebGpuBuffer;
    if (!("host" in b)) b.owners++;
  }

  release(buffer: unknown): void {
    const b = buffer as WebGpuBuffer;
    if ("host" in b || b.owners === FREED || --b.owners > 0) return;
    // Freed in order after the work already posted that reads it.
    this.freed.unregister(b);
    b.owners = FREED;
    const ids = b.imagId ? [b.id, b.imagId] : [b.id];
    this.bytes -= ids.length * Math.max(4, b.n * 4);
    this.free(ids);
  }

  usedMemory(): number {
    return this.bytes;
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
