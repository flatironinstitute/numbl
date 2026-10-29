/**
 * The CPU reference device: a GPU device whose memory is host memory and
 * which computes every operation with the host builtins. It makes gpuArray
 * code run, and testable, without a GPU, and is the reference that a real
 * device's results are compared against. A single-only instance behaves
 * like a device without double precision.
 */

import type { ElementClass, RuntimeTensor } from "../types.js";
import { RuntimeTensor as Tensor } from "../types.js";
import { allocFloat64Array } from "../alloc.js";
import type { GpuDevice } from "./device.js";

interface CpuBuffer {
  data: Float64Array;
  imag: Float64Array | undefined;
}

export class CpuReferenceDevice implements GpuDevice {
  readonly name: string;
  readonly totalMemory = 8 * 2 ** 30;
  generation = 0;

  constructor(readonly supportsDouble: boolean) {
    this.name = supportsDouble
      ? "numbl CPU reference"
      : "numbl CPU reference (single only)";
  }

  upload(t: RuntimeTensor): CpuBuffer {
    return {
      data: allocFloat64Array(t.data),
      imag: t.imag && allocFloat64Array(t.imag),
    };
  }

  download(
    buffer: unknown,
    shape: number[],
    cls: ElementClass,
    isComplex: boolean
  ): RuntimeTensor {
    const b = buffer as CpuBuffer;
    return new Tensor(
      allocFloat64Array(b.data),
      shape.slice(),
      isComplex && b.imag ? allocFloat64Array(b.imag) : undefined,
      cls
    );
  }
}

/** The devices a host has when it is given none: the CPU reference device
 *  with double precision, and one without. */
export function defaultGpuDevices(): GpuDevice[] {
  return [new CpuReferenceDevice(true), new CpuReferenceDevice(false)];
}
