/**
 * The interface of a GPU device (see device.ts). A leaf module, so that
 * types.ts (whose gpuArray value refers to its device) can import it
 * without an import cycle.
 */

import type { ElementClass } from "../elementClass.js";

/** Array data in host memory, as a device takes and returns it (a
 *  `RuntimeTensor` is one). */
export interface HostArray {
  data: Float64Array;
  imag: Float64Array | undefined;
  shape: number[];
  cls: ElementClass;
}

export interface GpuDevice {
  /** `gpuDevice().Name`. */
  readonly name: string;
  /** Whether the device computes in double precision. On a device that
   *  does not, creating a double gpuArray is an error. */
  readonly supportsDouble: boolean;
  /** `gpuDevice().TotalMemory`, in bytes. */
  readonly totalMemory: number;
  /** Incremented when the device is reset: arrays made in an earlier
   *  generation no longer exist. */
  generation: number;
  /** A device copy of a host tensor's elements (real or complex). */
  upload(t: HostArray): unknown;
  /** A buffer's elements, in fresh host storage. */
  download(
    buffer: unknown,
    shape: number[],
    cls: ElementClass,
    isComplex: boolean
  ): HostArray;
  /** Wait for queued work to finish (`wait(D)`). */
  synchronize?(): void;
  /** Kernels the device runs natively (see runtime/gpu/kernels.ts); an
   *  operation without one runs on the host. */
  kernels?: GpuKernels;
}

/** A kernel operand: a device buffer of `n` elements (1 broadcasts), or a
 *  scalar. With broadcasting, `strides` gives the buffer's stride along each
 *  of the result's dimensions (0 where the operand has size 1). */
export type KernelOperand =
  | { buffer: unknown; n: number; strides?: number[] }
  | { scalar: number };

/** Linear offsets into column-major data: `start + k*step`, or
 *  `offsets[k]`, for k < n. */
export interface KernelSelection {
  n: number;
  start: number;
  step: number;
  offsets: Uint32Array | null;
}

/** Kernels over real f32 data. Each returns the result's buffer; the
 *  caller has checked the operands and shapes. */
export interface GpuKernels {
  /** Whether `buffer` is in device memory and can be a kernel operand. */
  holds(buffer: unknown): boolean;
  /** An element-wise operation (`plus`, `lt`, `sin`, ...) over `n`
   *  elements; with `dims` (at most 4), the operands are broadcast to the
   *  result's dimensions by their `strides`. */
  elementwise(
    op: string,
    inputs: KernelOperand[],
    n: number,
    dims?: number[]
  ): unknown;
  /** `elementwise` for an operation with a real domain (`sqrt`, `log`,
   *  `power`, ...), waiting for it: also whether an element was outside
   *  the domain (its real result is NaN). */
  elementwiseChecked(
    op: string,
    inputs: KernelOperand[],
    n: number,
    dims?: number[]
  ): { buffer: unknown; bad: boolean };
  /** The selected elements of `src`. */
  take(src: unknown, sel: KernelSelection): unknown;
  /** A copy of `src` (`total` elements) with the selected elements set to
   *  `rhs` (a buffer of `sel.n` elements or 1, or a scalar). */
  put(
    src: unknown,
    total: number,
    sel: KernelSelection,
    rhs: KernelOperand
  ): unknown;
  /** `sum`, `prod`, `max` or `min` (NaN skipped), `any` or `all` over the
   *  middle dimension of `src` viewed as [inner, len, outer]. */
  reduce(
    op: string,
    src: unknown,
    inner: number,
    len: number,
    outer: number
  ): unknown;
  /** `max` or `min` over the middle dimension, with the 0-based position
   *  of the first extreme value along it (as f32: positions below 2^24). */
  reduceWithIndex(
    op: string,
    src: unknown,
    inner: number,
    len: number,
    outer: number
  ): { value: unknown; index: unknown };
  /** `cumsum` or `cumprod` over the middle dimension. */
  scan(
    op: string,
    src: unknown,
    inner: number,
    len: number,
    outer: number
  ): unknown;
  /** Concatenation into a new buffer of `total` elements: each part puts
   *  `block` contiguous elements per slab (`outer` slabs) at `offset`
   *  within the slab of `stride` elements. */
  concat(
    total: number,
    outer: number,
    stride: number,
    parts: { src: unknown; block: number; offset: number }[]
  ): unknown;
  /** a (m x k) * b (k x n). */
  matmul(a: unknown, b: unknown, m: number, k: number, n: number): unknown;
  /** The transpose of src (m x n). */
  transpose(src: unknown, m: number, n: number): unknown;
}
