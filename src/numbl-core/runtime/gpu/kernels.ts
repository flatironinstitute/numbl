/**
 * Running operations on a device's kernels (`GpuDevice.kernels`).
 *
 * Each function returns the result as a gpuArray, or null when the
 * operation cannot run on the device's kernels: the device has none, an
 * operand is not real f32 data on it (a complex array, a double array, a
 * host array), the shapes need more than scalar expansion, or the result
 * would be double. The caller then runs the builtin, which computes on the
 * host (see `builtin.ts`). The result classes follow MATLAB: comparisons
 * and logical operators give logical, arithmetic with a single operand
 * gives single.
 */

import {
  type ElementClass,
  type RuntimeValue,
  RuntimeGpuArray,
  RuntimeTensor,
  isRuntimeChar,
  isRuntimeString,
} from "../types.js";
import type {
  GpuDevice,
  GpuKernels,
  KernelOperand,
  KernelSelection,
} from "./device.js";
import { checkExists, firstGpuArg } from "./gpuArray.js";
import { RuntimeError } from "../error.js";
import { planIndex, planStore, type Selection } from "../indexPlan.js";

interface Arg {
  op: KernelOperand;
  cls: ElementClass;
  /** The gpuArray's shape; null for a host scalar. */
  shape: number[] | null;
  n: number;
}

function kernelsOf(args: RuntimeValue[]): {
  device: GpuDevice;
  k: GpuKernels;
} | null {
  const g = firstGpuArg(args);
  const k = g?.device.kernels;
  return g && k ? { device: g.device, k } : null;
}

/** `v` as a kernel operand on `device`, or null. */
function kernelArg(
  v: RuntimeValue,
  device: GpuDevice,
  k: GpuKernels
): Arg | null {
  if (v instanceof RuntimeGpuArray) {
    if (v.device !== device || v.isComplex || v.cls === "double") return null;
    if (!k.holds(v.buffer)) return null;
    checkExists(v);
    const n = numelOf(v.shape);
    return { op: { buffer: v.buffer, n }, cls: v.cls, shape: v.shape, n };
  }
  if (typeof v === "number")
    return { op: { scalar: v }, cls: "double", shape: null, n: 1 };
  if (typeof v === "boolean")
    return { op: { scalar: v ? 1 : 0 }, cls: "logical", shape: null, n: 1 };
  if (
    v instanceof RuntimeTensor &&
    v.data.length === 1 &&
    !v.imag &&
    v.cls !== "double"
  )
    return { op: { scalar: v.data[0] }, cls: v.cls, shape: null, n: 1 };
  return null;
}

function numelOf(shape: number[]): number {
  let n = 1;
  for (const d of shape) n *= d;
  return n;
}

function sameShape(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((d, i) => d === b[i]);
}

function result(
  device: GpuDevice,
  buffer: unknown,
  shape: number[],
  cls: ElementClass
): RuntimeGpuArray {
  return new RuntimeGpuArray(
    device,
    buffer,
    shape,
    cls,
    false,
    device.generation
  );
}

/** Operations whose result is logical. */
const LOGICAL_RESULT = new Set([
  "lt",
  "gt",
  "le",
  "ge",
  "eq",
  "ne",
  "and",
  "or",
  "not",
]);

/** An element-wise operation (`plus`, `lt`, `sin`, ...) on one or two
 *  operands, each a gpuArray or a scalar, the arrays of one shape or one
 *  element. */
export function elementwiseKernel(
  op: string,
  args: RuntimeValue[]
): RuntimeValue | null {
  const dk = kernelsOf(args);
  if (!dk) return null;
  const xs: Arg[] = [];
  for (const a of args) {
    const x = kernelArg(a, dk.device, dk.k);
    if (!x) return null;
    xs.push(x);
  }
  // The result has the shape of the operands that are not scalars, or
  // their implicit expansion.
  let shape: number[] | null = null;
  let broadcast = false;
  for (const x of xs)
    if (x.shape && x.n !== 1) {
      if (shape && !sameShape(shape, x.shape)) {
        shape = expandShapes(shape, x.shape);
        if (!shape) return null;
        broadcast = true;
      } else shape ??= x.shape;
    }
  shape ??= xs.find(x => x.shape)!.shape!;
  let cls: ElementClass;
  if (LOGICAL_RESULT.has(op)) cls = "logical";
  else if (xs.some(x => x.cls === "single")) cls = "single";
  else return null; // a double result (logical + 1, abs of a logical, ...)
  let dims: number[] | undefined;
  let ops = xs.map(x => x.op);
  if (broadcast) {
    if (shape.length > 4) return null;
    dims = [...shape, 1, 1, 1].slice(0, 4);
    ops = xs.map(x =>
      "scalar" in x.op
        ? x.op
        : { ...x.op, strides: expandStrides(x.shape!, dims!) }
    );
  }
  const n = numelOf(shape);
  let buffer: unknown;
  if (DOMAIN_OPS.has(op)) {
    const r = dk.k.elementwiseChecked(op, ops, n, dims);
    if (r.bad) throw complexResultError(op);
    buffer = r.buffer;
  } else buffer = dk.k.elementwise(op, ops, n, dims);
  return result(dk.device, buffer, shape.slice(), cls);
}

/** The implicit expansion of two shapes (each dimension equal or 1), or
 *  null when they are incompatible. */
function expandShapes(a: number[], b: number[]): number[] | null {
  const n = Math.max(a.length, b.length);
  const out: number[] = [];
  for (let d = 0; d < n; d++) {
    const x = d < a.length ? a[d] : 1;
    const y = d < b.length ? b[d] : 1;
    if (x !== y && x !== 1 && y !== 1) return null;
    out.push(x === 1 ? y : x);
  }
  return out;
}

/** An operand's strides over the result's dimensions `dims`: its
 *  column-major strides, 0 where it has size 1. */
function expandStrides(shape: number[], dims: number[]): number[] {
  const out: number[] = [];
  let stride = 1;
  for (let d = 0; d < dims.length; d++) {
    const s = d < shape.length ? shape[d] : 1;
    out.push(s === 1 ? 0 : stride);
    stride *= s;
  }
  return out;
}

/** The operations whose real input must lie in a domain for a real
 *  result. */
export const DOMAIN_OPS: ReadonlySet<string> = new Set([
  "sqrt",
  "log",
  "log2",
  "log10",
  "log1p",
  "asin",
  "acos",
  "atanh",
  "acosh",
  "power",
]);

/** MATLAB's error when an operation on real gpuArray data needs a complex
 *  result. */
export function complexResultError(op: string): RuntimeError {
  const name = op.toUpperCase();
  const use = op === "power" ? "POWER(COMPLEX(X), Y)" : `${name}(COMPLEX(X))`;
  return new RuntimeError(
    `${name}: needs to return a complex result, but this is not supported for real input X on the GPU. Use ${use}.`
  );
}

/** `G(indices)` for a real gpuArray, gathered on the device. */
export function indexKernel(
  g: RuntimeGpuArray,
  indices: RuntimeValue[]
): RuntimeValue | null {
  const k = g.device.kernels;
  if (!k) return null;
  const x = kernelArg(g, g.device, k);
  if (!x) return null;
  const plan = planIndex(g.shape, x.n, indices);
  const src = (x.op as { buffer: unknown }).buffer;
  // All elements in order (`G(:)`): the same data in a new shape.
  const buffer =
    plan.offsets === null &&
    plan.start === 0 &&
    plan.step === 1 &&
    plan.n === x.n
      ? src
      : k.take(src, selection(plan));
  return result(g.device, buffer, plan.shape.slice(), g.cls);
}

/** `G(indices) = rhs` for a real gpuArray, on the device; null when the
 *  assignment grows `G` or converts the values (into a logical array). */
export function storeKernel(
  g: RuntimeGpuArray,
  indices: RuntimeValue[],
  rhs: RuntimeValue
): RuntimeValue | null {
  const k = g.device.kernels;
  if (!k) return null;
  const x = kernelArg(g, g.device, k);
  const r = kernelArg(rhs, g.device, k);
  if (!x || !r) return null;
  if (g.cls === "logical" && r.cls !== "logical") return null;
  const plan = planStore(g.shape, x.n, indices, r.n, r.shape ?? [1, 1]);
  if (plan.grow) return null;
  const buffer = k.put(
    (x.op as { buffer: unknown }).buffer,
    x.n,
    selection(plan),
    r.op
  );
  return result(g.device, buffer, g.shape.slice(), g.cls);
}

function selection(sel: Selection): KernelSelection {
  if (sel.offsets === null)
    return { n: sel.n, start: sel.start, step: sel.step, offsets: null };
  const offsets = new Uint32Array(sel.n);
  for (let i = 0; i < sel.n; i++) offsets[i] = sel.offsets[i] - sel.bias;
  return { n: sel.n, start: 0, step: 0, offsets };
}

const isText = (v: RuntimeValue) => isRuntimeChar(v) || isRuntimeString(v);

/** `sum(G)`, `sum(G, dim)`, `sum(G, 'all')`, and likewise `prod`, `mean`,
 *  and one-output `max(G)`, `max(G, [], dim)`, `min`. */
export function reduceKernel(
  op: "sum" | "prod" | "mean" | "max" | "min",
  args: RuntimeValue[],
  nargout: number
): RuntimeValue | null {
  if (nargout > 1) return null;
  const dk = kernelsOf(args);
  if (!dk || !(args[0] instanceof RuntimeGpuArray)) return null;
  const x = kernelArg(args[0], dk.device, dk.k);
  if (!x || x.n === 0) return null;
  const minmax = op === "max" || op === "min";
  // sum/prod/mean of a logical array are double.
  if (x.cls !== "single" && !(minmax && x.cls === "logical")) return null;
  let rest = args.slice(1);
  if (minmax) {
    if (rest.length === 0) {
      // max(G)
    } else if (
      rest[0] instanceof RuntimeTensor &&
      rest[0].data.length === 0 &&
      rest.length === 2
    )
      rest = rest.slice(1);
    else return null;
  }
  const shape = x.shape!;
  let dim: number | "all";
  if (rest.length === 0) {
    const d = shape.findIndex(s => s !== 1);
    dim = d < 0 ? 0 : d;
  } else if (rest.length === 1) {
    const r = rest[0];
    if (isText(r) && String(isRuntimeChar(r) ? r.value : r) === "all")
      dim = "all";
    else if (typeof r === "number" && Number.isInteger(r) && r >= 1) {
      if (r > shape.length) return null;
      dim = r - 1;
    } else return null;
  } else return null;

  let inner: number, len: number, outer: number, outShape: number[];
  if (dim === "all") {
    [inner, len, outer, outShape] = [1, x.n, 1, [1, 1]];
  } else {
    inner = numelOf(shape.slice(0, dim));
    len = shape[dim];
    outer = numelOf(shape.slice(dim + 1));
    outShape = shape.slice();
    outShape[dim] = 1;
  }
  const kop = op === "mean" ? "sum" : op;
  let buf = dk.k.reduce(
    kop,
    (x.op as { buffer: unknown }).buffer,
    inner,
    len,
    outer
  );
  const nOut = inner * outer;
  if (op === "mean")
    buf = dk.k.elementwise(
      "times",
      [{ buffer: buf, n: nOut }, { scalar: 1 / len }],
      nOut
    );
  while (outShape.length > 2 && outShape[outShape.length - 1] === 1)
    outShape.pop();
  return result(dk.device, buf, outShape, x.cls);
}

/** `A * B` for two-dimensional single arrays (a scalar operand is
 *  element-wise). */
export function mtimesKernel(args: RuntimeValue[]): RuntimeValue | null {
  const dk = kernelsOf(args);
  if (!dk || args.length !== 2) return null;
  const a = kernelArg(args[0], dk.device, dk.k);
  const b = kernelArg(args[1], dk.device, dk.k);
  if (!a || !b) return null;
  if (a.n === 1 || b.n === 1) return elementwiseKernel("times", args);
  const [sa, sb] = [a.shape!, b.shape!];
  if (sa.length !== 2 || sb.length !== 2 || sa[1] !== sb[0]) return null;
  if (a.cls !== "single" && b.cls !== "single") return null;
  const buf = dk.k.matmul(
    (a.op as { buffer: unknown }).buffer,
    (b.op as { buffer: unknown }).buffer,
    sa[0],
    sa[1],
    sb[1]
  );
  return result(dk.device, buf, [sa[0], sb[1]], "single");
}

/** The transpose of a real two-dimensional gpuArray (for real data, `'`
 *  and `.'` agree). */
export function transposeKernel(args: RuntimeValue[]): RuntimeValue | null {
  const dk = kernelsOf(args);
  if (!dk || args.length !== 1) return null;
  const x = kernelArg(args[0], dk.device, dk.k);
  if (!x || !x.shape || x.shape.length !== 2) return null;
  const [m, n] = x.shape;
  const buf =
    m === 1 || n === 1
      ? (x.op as { buffer: unknown }).buffer
      : dk.k.transpose((x.op as { buffer: unknown }).buffer, m, n);
  return result(dk.device, buf, [n, m], x.cls);
}
