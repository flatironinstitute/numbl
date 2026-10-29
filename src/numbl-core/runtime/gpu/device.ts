/**
 * GPU devices: the compute devices behind gpuArray.
 *
 * A device stores array data in its own memory (`upload` / `download`) and
 * says whether it computes in double precision. The host lists its devices
 * (`NumblHost.gpuDevices`); one at a time is selected, as in MATLAB. Every
 * builtin runs on a gpuArray (see `builtin.ts`): a device without a kernel
 * for an operation has its operands downloaded, computed on the host and the
 * results uploaded, so a device only needs `upload` and `download` to be
 * complete.
 */

import { RuntimeError } from "../error.js";
import type { GpuDevice } from "./deviceTypes.js";

export type { GpuDevice, HostArray } from "./deviceTypes.js";

/** A host's GPU devices and which one is selected. */
export class GpuDeviceSet {
  /** The selected device's position in `devices`, or -1 for none. */
  private selected = -1;

  private list: GpuDevice[] | null = null;

  /** `devices`, or a function that lists them on first use (a host whose
   *  WebGPU device is found by starting the GPU helper). */
  constructor(private readonly source: GpuDevice[] | (() => GpuDevice[])) {}

  get devices(): GpuDevice[] {
    return (this.list ??=
      typeof this.source === "function" ? this.source() : this.source);
  }

  /** The selected device, selecting the first one when none is (as
   *  MATLAB selects the default device on first use). */
  current(): GpuDevice {
    if (this.selected < 0) {
      if (this.devices.length === 0) throw noDeviceError();
      this.selected = 0;
    }
    return this.devices[this.selected];
  }

  /** 1-based index of the selected device, or 0 for none. */
  selectedIndex(): number {
    return this.selected + 1;
  }

  /** `gpuDevice(index)`: select a device and reset it, so the arrays on it
   *  (and on the previously selected device) no longer exist. */
  select(index: number): GpuDevice {
    if (!Number.isInteger(index) || index < 1 || index > this.devices.length)
      throw new RuntimeError(
        `Invalid GPU device index ${index}. The index must be an integer between 1 and ${this.devices.length}.`,
        null,
        { identifier: "parallel:gpu:device:InvalidIndex" }
      );
    if (this.selected >= 0) this.devices[this.selected].generation++;
    this.selected = index - 1;
    const d = this.devices[this.selected];
    d.generation++;
    return d;
  }

  /** `gpuDevice([])`: deselect the device, clearing its arrays. */
  deselect(): void {
    if (this.selected >= 0) this.devices[this.selected].generation++;
    this.selected = -1;
  }
}

export function noDeviceError(): RuntimeError {
  return new RuntimeError("Unable to find a supported GPU device.", null, {
    identifier: "parallel:gpu:device:NoCUDADevice",
  });
}
