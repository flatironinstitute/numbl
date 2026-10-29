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
}
