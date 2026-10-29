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
 *  scalar. */
export type KernelOperand = { buffer: unknown; n: number } | { scalar: number };

/** Kernels over real f32 data. Each returns the result's buffer; the
 *  caller has checked the operands and shapes. */
export interface GpuKernels {
  /** Whether `buffer` is in device memory and can be a kernel operand. */
  holds(buffer: unknown): boolean;
  /** An element-wise operation (`plus`, `lt`, `sin`, ...) over `n`
   *  elements. */
  elementwise(op: string, inputs: KernelOperand[], n: number): unknown;
  /** `sum`, `prod`, `max` or `min` (NaN skipped) over the middle dimension
   *  of `src` viewed as [inner, len, outer]. */
  reduce(
    op: string,
    src: unknown,
    inner: number,
    len: number,
    outer: number
  ): unknown;
  /** a (m x k) * b (k x n). */
  matmul(a: unknown, b: unknown, m: number, k: number, n: number): unknown;
  /** The transpose of src (m x n). */
  transpose(src: unknown, m: number, n: number): unknown;
}
