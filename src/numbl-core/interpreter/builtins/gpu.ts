/**
 * gpuArray builtins: gpuArray, gather, the type queries, gpuDevice and the
 * device functions. Every other builtin runs on gpuArray arguments through
 * the call wrapper (`builtin.ts`): their data is gathered, the builtin runs
 * and its numeric outputs go back to the device. See runtime/gpu/.
 */

import {
  type RuntimeValue,
  type RuntimeClassInstance,
  isRuntimeChar,
  isRuntimeString,
  isRuntimeTensor,
} from "../../runtime/types.js";
import { RTV } from "../../runtime/constructors.js";
import { RuntimeError } from "../../runtime/error.js";
import { toString } from "../../runtime/convert.js";
import { valueClass, isaValue, isRealValue } from "../../runtime/kinds.js";
import { currentHost } from "../../host.js";
import type { GpuDevice } from "../../runtime/gpu/device.js";
import {
  currentGpuDevice,
  doubleNotSupportedError,
  existsOnDevice,
  gatherValue,
  isGpuArray,
  toGpuArray,
} from "../../runtime/gpu/gpuArray.js";
import {
  DOMAIN_OPS,
  complexResultError,
  concatKernel,
  dotKernel,
  elementwiseKernel,
  normKernel,
  repmatKernel,
  scanKernel,
  mtimesKernel,
  reduceKernel,
  transposeKernel,
} from "../../runtime/gpu/kernels.js";
import { builtin, classMethod } from "./builtin.js";
import { builtinError } from "./args.js";
import { getIBuiltin } from "../../runtime/builtinRegistry.js";

const GPU = { gpu: true } as const;

builtin({
  name: "gpuArray",
  help: {
    signatures: ["G = gpuArray(X)"],
    description:
      "Copy the numeric or logical array X to the selected GPU device. A gpuArray input is returned unchanged. On a device without double precision, X must not be double (convert it with single first).",
  },
  nargin: 1,
  accepts: GPU,
  apply: args => {
    const x = args[0];
    if (isGpuArray(x)) return x;
    return toGpuArray(x, currentGpuDevice(), true);
  },
});

builtin({
  name: "gather",
  help: {
    signatures: ["X = gather(G)", "[X1, ..., Xn] = gather(G1, ..., Gn)"],
    description:
      "Copy gpuArray data to the host. Other inputs are returned unchanged.",
  },
  nargin: [1, Infinity],
  nargout: [0, Infinity],
  accepts: { gpu: true, sparse: true },
  apply: (args, nargout) => {
    if (nargout > args.length)
      throw new RuntimeError("Too many output arguments.", null, {
        identifier: "MATLAB:TooManyOutputs",
      });
    const out = args.map(a => (isGpuArray(a) ? gatherValue(a) : a));
    return nargout <= 1 ? out[0] : out;
  },
});

builtin({
  name: "isgpuarray",
  help: {
    signatures: ["tf = isgpuarray(X)"],
    description: "True if X is a gpuArray.",
  },
  nargin: 1,
  accepts: GPU,
  apply: args => isGpuArray(args[0]),
});

builtin({
  name: "existsOnGPU",
  help: {
    signatures: ["tf = existsOnGPU(G)"],
    description:
      "True if the gpuArray G still exists on its device (resetting or deselecting the device clears its arrays).",
  },
  nargin: 1,
  accepts: GPU,
  apply: args => {
    const g = args[0];
    if (!isGpuArray(g))
      throw builtinError(
        "MATLAB:UndefinedFunction",
        `Undefined function 'existsOnGPU' for input arguments of type '${valueClass(g)}'.`
      );
    return existsOnDevice(g);
  },
});

/** The class of a value's elements: a gpuArray's underlying class, else
 *  the value's class. */
function underlyingClass(v: RuntimeValue): string {
  return isGpuArray(v) ? v.cls : valueClass(v);
}

builtin({
  name: "underlyingType",
  help: {
    signatures: ["T = underlyingType(X)"],
    description:
      "The class of the data in X: for a gpuArray, the class of its elements ('double', 'single' or 'logical'); otherwise class(X).",
  },
  nargin: 1,
  accepts: { gpu: true, sparse: true },
  apply: args => RTV.char(underlyingClass(args[0])),
});

builtin({
  name: "isUnderlyingType",
  help: {
    signatures: ["tf = isUnderlyingType(X, T)"],
    description:
      "True if the data in X is of class T, or T is a category ('numeric', 'float', 'integer') containing it.",
  },
  nargin: 2,
  accepts: { gpu: true, sparse: true },
  apply: args => {
    const t = args[1];
    if (!(isRuntimeChar(t) || isRuntimeString(t)))
      throw builtinError(
        "MATLAB:isUnderlyingType:invalidType",
        "Type name must be a character vector or string scalar."
      );
    const x = args[0];
    const name = toString(t);
    return isaValue(isGpuArray(x) ? sampleOf(x) : x, name);
  },
});

/** A host scalar of a gpuArray's underlying class (for class tests),
 *  without gathering its data. */
function sampleOf(g: { cls: string }): RuntimeValue {
  return g.cls === "logical"
    ? true
    : g.cls === "single"
      ? RTV.tensor(Float64Array.of(0), [1, 1], undefined, "single")
      : 0;
}

// ── Conversions that create a double gpuArray ───────────────────────────
// double(G) and cast(G, 'double') are explicit creations: on a device
// without double precision they raise the error that gpuArray(X) raises.

classMethod("gpuArray", {
  name: "double",
  nargin: 1,
  accepts: GPU,
  apply: args => {
    const g = args[0];
    if (!isGpuArray(g)) return getIBuiltin("double")!.call([g], 1);
    if (!g.device.supportsDouble) throw doubleNotSupportedError();
    const v = getIBuiltin("double")!.call([gatherValue(g)], 1) as RuntimeValue;
    return toGpuArray(v, g.device, true);
  },
});

classMethod("gpuArray", {
  name: "cast",
  nargin: [2, 3],
  accepts: GPU,
  apply: args => {
    const g = args.find(isGpuArray)!;
    const host = args.map(a => (isGpuArray(a) ? gatherValue(a) : a));
    const v = getIBuiltin("cast")!.call(host, 1) as RuntimeValue;
    // cast(G, ...) stays on the device; cast(X, 'like', G) moves X there.
    if (!isGpuArray(args[0]) && !(args.length === 3 && isGpuArray(args[2])))
      return v;
    return toGpuArray(v, g.device, true);
  },
});

// ── Devices ─────────────────────────────────────────────────────────────

const DEVICE_CLASS = "parallel.gpu.CUDADevice";

/** The `gpuDevice` object of a device: a snapshot of its properties. */
function deviceObject(d: GpuDevice, index: number): RuntimeClassInstance {
  const set = currentHost().gpuDevices;
  const props: [string, RuntimeValue][] = [
    ["Name", RTV.char(d.name)],
    ["Index", index],
    ["ComputeCapability", RTV.char("")],
    ["SupportsDouble", d.supportsDouble],
    ["SingleDoubleRatio", d.supportsDouble ? 1 : Infinity],
    ["TotalMemory", d.totalMemory],
    ["AvailableMemory", d.totalMemory],
    ["ComputeMode", RTV.char("Default")],
    ["DeviceAvailable", true],
    ["DeviceSupported", true],
    ["DeviceSelected", set.selectedIndex() === index],
  ];
  return RTV.classInstance(
    DEVICE_CLASS,
    props.map(p => p[0]),
    true,
    new Map(props)
  );
}

/** The device a `gpuDevice` object describes. */
function deviceOf(v: RuntimeValue): GpuDevice {
  const set = currentHost().gpuDevices;
  const idx =
    typeof v === "object" && v !== null && "fields" in v
      ? (v as RuntimeClassInstance).fields.get("Index")
      : undefined;
  if (typeof idx !== "number" || !set.devices[idx - 1])
    throw new RuntimeError("Expected a GPU device object.");
  return set.devices[idx - 1];
}

builtin({
  name: "gpuDevice",
  help: {
    signatures: ["D = gpuDevice", "D = gpuDevice(index)", "gpuDevice([])"],
    description:
      "The selected GPU device (selecting the first one if none is). gpuDevice(index) selects device index and resets it, clearing its arrays; gpuDevice([]) deselects the device. numbl's devices without a GPU are CPU reference devices: device 1 computes in double precision, device 2 only in single (SupportsDouble is false).",
  },
  nargin: [0, 1],
  apply: (args, nargout) => {
    const set = currentHost().gpuDevices;
    if (args.length === 0) {
      const d = set.current();
      return deviceObject(d, set.selectedIndex());
    }
    const a = args[0];
    if (isRuntimeTensor(a) && a.data.length === 0) {
      set.deselect();
      return nargout > 0 ? RTV.tensor(new Float64Array(0), [0, 0]) : undefined;
    }
    const idx = typeof a === "number" ? a : NaN;
    const d = set.select(idx);
    return deviceObject(d, idx);
  },
});

builtin({
  name: "gpuDeviceCount",
  help: {
    signatures: ["n = gpuDeviceCount"],
    description: "The number of GPU devices.",
  },
  nargin: [0, 1],
  nargout: 2,
  apply: (_args, nargout) => {
    const n = currentHost().gpuDevices.devices.length;
    if (nargout < 2) return n;
    return [
      n,
      RTV.tensor(
        Float64Array.from({ length: n }, (_, i) => i + 1),
        [1, n]
      ),
    ];
  },
});

builtin({
  name: "canUseGPU",
  help: {
    signatures: ["tf = canUseGPU"],
    description: "True if a GPU device is available for gpuArray.",
  },
  nargin: 0,
  apply: () => currentHost().gpuDevices.devices.length > 0,
});

classMethod(DEVICE_CLASS, {
  name: "reset",
  nargin: 1,
  nargout: 0,
  apply: args => {
    deviceOf(args[0]).generation++;
  },
});

classMethod(DEVICE_CLASS, {
  name: "wait",
  nargin: 1,
  nargout: 0,
  apply: args => {
    deviceOf(args[0]).synchronize?.();
  },
});

// ── Kernels ─────────────────────────────────────────────────────────────
// Operations a device may run natively (`GpuDevice.kernels`). Each method
// tries the device's kernel and otherwise calls the builtin, which runs the
// operation on the host.

/** Register a gpuArray method `name` that tries `kernel` first. */
function kernelMethod(
  name: string,
  kernel: (
    args: RuntimeValue[],
    nargout: number
  ) => RuntimeValue | RuntimeValue[] | null
): void {
  classMethod("gpuArray", {
    name,
    nargout: [0, Infinity],
    accepts: GPU,
    apply: (args, nargout) =>
      kernel(args, nargout) ?? getIBuiltin(name)!.call(args, nargout),
  });
}

for (const op of [
  "plus",
  "minus",
  "times",
  "rdivide",
  "ldivide",
  "lt",
  "gt",
  "le",
  "ge",
  "eq",
  "ne",
  "and",
  "or",
  "uminus",
  "not",
  "abs",
  "exp",
  "sin",
  "cos",
  "tan",
  "sinh",
  "cosh",
  "tanh",
  "floor",
  "ceil",
  "fix",
  "round",
  "sign",
  "isnan",
  "isinf",
  "isfinite",
  "mod",
  "rem",
  "atan2",
  "hypot",
])
  kernelMethod(op, (args, nargout) =>
    nargout <= 1 ? elementwiseKernel(op, args) : null
  );

// Functions whose real result needs the input in a domain (sqrt of a
// negative number is complex): on real gpuArray data, MATLAB raises an
// error rather than return a complex result, on every device.
for (const op of DOMAIN_OPS)
  classMethod("gpuArray", {
    name: op,
    nargout: [0, Infinity],
    accepts: GPU,
    apply: (args, nargout) => {
      const k = nargout <= 1 ? elementwiseKernel(op, args) : null;
      if (k) return k;
      const r = getIBuiltin(op)!.call(args, nargout) as RuntimeValue;
      if (isGpuArray(r) && r.isComplex && !args.some(isComplexValue))
        throw complexResultError(op);
      return r;
    },
  });

/** Whether a value has complex data. */
function isComplexValue(v: RuntimeValue): boolean {
  return isGpuArray(v) ? v.isComplex : !isRealValue(v);
}

for (const op of ["max", "min"] as const)
  kernelMethod(op, (args, nargout) =>
    args.length === 2 && nargout <= 1
      ? elementwiseKernel(op, args)
      : reduceKernel(op, args, nargout)
  );

for (const op of ["sum", "prod", "mean", "any", "all"] as const)
  kernelMethod(op, (args, nargout) => reduceKernel(op, args, nargout));

for (const op of ["cumsum", "cumprod"] as const)
  kernelMethod(op, (args, nargout) =>
    nargout <= 1 ? scanKernel(op, args) : null
  );

kernelMethod("dot", (args, nargout) => (nargout <= 1 ? dotKernel(args) : null));
kernelMethod("norm", (args, nargout) =>
  nargout <= 1 ? normKernel(args) : null
);
kernelMethod("repmat", (args, nargout) =>
  nargout <= 1 ? repmatKernel(args) : null
);
kernelMethod("horzcat", args => concatKernel(1, args));
kernelMethod("vertcat", args => concatKernel(0, args));
kernelMethod("cat", args =>
  typeof args[0] === "number" && Number.isInteger(args[0]) && args[0] >= 1
    ? concatKernel(args[0] - 1, args.slice(1))
    : null
);

kernelMethod("mtimes", args => mtimesKernel(args));
kernelMethod("transpose", args => transposeKernel(args));
kernelMethod("ctranspose", args => transposeKernel(args));
