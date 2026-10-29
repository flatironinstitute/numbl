/**
 * gpuArray values: moving data to a device and back.
 */

import {
  type RuntimeValue,
  RuntimeGpuArray,
  RuntimeTensor,
  RuntimeComplexNumber,
  isRuntimeSparseMatrix,
} from "../types.js";
import { RuntimeError } from "../error.js";
import { kstr } from "../kinds.js";
import { currentHost } from "../../host.js";
import type { GpuDevice } from "./device.js";

/** The selected device of the running code's host. */
export function currentGpuDevice(): GpuDevice {
  return currentHost().gpuDevices.current();
}

/** Whether `v` is a gpuArray. */
export function isGpuArray(v: unknown): v is RuntimeGpuArray {
  return v instanceof RuntimeGpuArray;
}

/** The first gpuArray among `args`, or null. */
export function firstGpuArg(args: readonly unknown[]): RuntimeGpuArray | null {
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a instanceof RuntimeGpuArray) return a;
  }
  return null;
}

export function doubleNotSupportedError(): RuntimeError {
  return new RuntimeError(
    "The selected GPU device does not support double precision. Convert the data to single first, for example gpuArray(single(X)).",
    null,
    { identifier: "numbl:gpu:doubleNotSupported" }
  );
}

/** `v` (a numeric or logical host value) as a gpuArray on `device`. An
 *  `explicit` creation (gpuArray(X), zeros(n, 'gpuArray'), double(G)) of a
 *  double array on a device without double precision is an error; the
 *  results of operations are not checked (an index output of `max` is
 *  double). */
export function toGpuArray(
  v: RuntimeValue,
  device: GpuDevice,
  explicit: boolean
): RuntimeGpuArray {
  let t: RuntimeTensor;
  if (typeof v === "number") t = new RuntimeTensor(Float64Array.of(v), [1, 1]);
  else if (typeof v === "boolean")
    t = new RuntimeTensor(
      Float64Array.of(v ? 1 : 0),
      [1, 1],
      undefined,
      "logical"
    );
  else if (v instanceof RuntimeComplexNumber)
    t = new RuntimeTensor(Float64Array.of(v.re), [1, 1], Float64Array.of(v.im));
  else if (v instanceof RuntimeTensor) t = v;
  else if (isRuntimeSparseMatrix(v))
    throw new RuntimeError("Sparse gpuArrays are not supported.", null, {
      identifier: "numbl:gpu:sparseNotSupported",
    });
  else
    throw new RuntimeError(
      `Unable to create a gpuArray from a ${kstr(v)}: the data must be numeric or logical.`,
      null,
      { identifier: "parallel:gpu:array:InvalidData" }
    );
  if (explicit && t.cls === "double" && !device.supportsDouble)
    throw doubleNotSupportedError();
  const shape = t.shape.length >= 2 ? t.shape.slice() : [1, t.data.length];
  return new RuntimeGpuArray(
    device,
    device.upload(t),
    shape,
    t.cls,
    t.imag !== undefined,
    device.generation
  );
}

/** Raise MATLAB's error when `g`'s device has been reset since it was made. */
export function checkExists(g: RuntimeGpuArray): void {
  if (g.generation !== g.device.generation)
    throw new RuntimeError("The data no longer exists on the device.", null, {
      identifier: "parallel:gpu:array:InvalidData",
    });
}

/** Whether `g` still exists on its device (`existsOnGPU`). */
export function existsOnDevice(g: RuntimeGpuArray): boolean {
  return g.generation === g.device.generation;
}

/** The host tensor holding a gpuArray's elements (a fresh copy). */
export function gatherTensor(g: RuntimeGpuArray): RuntimeTensor {
  checkExists(g);
  return g.device.download(g.buffer, g.shape, g.cls, g.isComplex);
}

/** `gather(g)`: the host value, a 1x1 double or logical array as a scalar
 *  (a single 1x1 array stays a tensor). */
export function gatherValue(g: RuntimeGpuArray): RuntimeValue {
  const t = gatherTensor(g);
  if (t.data.length !== 1 || t.shape.length !== 2 || t.cls === "single")
    return t;
  if (t.cls === "logical") return t.data[0] !== 0;
  const im = t.imag ? t.imag[0] : 0;
  return t.imag ? new RuntimeComplexNumber(t.data[0], im) : t.data[0];
}

/** A builtin's output when it ran with a gpuArray argument: numeric and
 *  logical values go to `device`, others (text, cells, ...) stay on the
 *  host. */
export function outputToGpu(v: RuntimeValue, device: GpuDevice): RuntimeValue {
  if (
    typeof v === "number" ||
    typeof v === "boolean" ||
    v instanceof RuntimeComplexNumber ||
    v instanceof RuntimeTensor
  )
    return toGpuArray(v, device, false);
  return v;
}
