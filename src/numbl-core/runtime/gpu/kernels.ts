/**
 * Running operations on a device's kernels (`GpuDevice.kernels`).
 *
 * Each function returns the result as a gpuArray, or null when the
 * operation cannot run on the device's kernels: the device has none, an
 * operand is not f32 data on it (a double array, a host array), the shapes
 * are incompatible, or the result would be double. The caller then runs
 * the builtin, which computes on the host (see `builtin.ts`). The result
 * classes follow MATLAB: comparisons and logical operators give logical,
 * arithmetic with a single operand gives single.
 *
 * A complex array is processed as its real and imaginary parts, each a
 * real buffer (`GpuKernels.parts`), except by the complex element-wise
 * kernel and the FFT. As in MATLAB, a complex result stays complex when
 * its imaginary parts are all zero.
 */

import {
  type ElementClass,
  type RuntimeValue,
  RuntimeComplexNumber,
  RuntimeGpuArray,
  RuntimeTensor,
  isRuntimeChar,
  isRuntimeString,
} from "../types.js";
import type {
  ComplexOperand,
  GpuDevice,
  GpuKernels,
  KernelOperand,
  KernelSelection,
} from "./device.js";
import { checkExists, firstGpuArg, toGpuArray } from "./gpuArray.js";
import { fuse } from "./fusion.js";
import { getCurrentRuntime } from "../refcount.js";
import { singleTensor } from "../single.js";
import { RuntimeError } from "../error.js";
import { planIndex, planStore, type Selection } from "../indexPlan.js";

interface Arg {
  op: KernelOperand;
  /** The gpuArray, for an array operand. */
  gpu?: RuntimeGpuArray;
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
    // A lazy array is on the device; reading its buffer would compute it.
    if (!v.lazy && !k.holds(v.buffer)) return null;
    checkExists(v);
    const n = numelOf(v.shape);
    return {
      // The buffer is read (computing a lazy array) only when needed.
      get op() {
        return { buffer: v.buffer, n };
      },
      gpu: v,
      cls: v.cls,
      shape: v.shape,
      n,
    };
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

/** Whether gpuArray `g` (real or complex) is f32 data that `k`'s kernels
 *  take; raises the error for an array that no longer exists. */
function onDevice(g: RuntimeGpuArray, k: GpuKernels): boolean {
  if (g.device.kernels !== k || g.cls === "double") return false;
  if (!g.lazy && !k.holds(g.buffer)) return false;
  checkExists(g);
  return true;
}

/** The real and imaginary parts of gpuArray `g` (null for real data). */
function planes(
  g: RuntimeGpuArray,
  k: GpuKernels
): { re: unknown; im: unknown | null } {
  return g.isComplex ? k.parts(g.buffer) : { re: g.buffer, im: null };
}

interface ComplexArg {
  op: ComplexOperand;
  gpu?: RuntimeGpuArray;
  cls: ElementClass;
  shape: number[] | null;
  n: number;
}

/** `v` (real or complex) as a complex kernel operand on `device`, or
 *  null. */
function complexArg(
  v: RuntimeValue,
  device: GpuDevice,
  k: GpuKernels
): ComplexArg | null {
  if (v instanceof RuntimeGpuArray) {
    if (v.device !== device || !onDevice(v, k)) return null;
    const n = numelOf(v.shape);
    return {
      get op() {
        return { ...planes(v, k), n };
      },
      gpu: v,
      cls: v.cls,
      shape: v.shape,
      n,
    };
  }
  const scalar = (re: number, im: number, cls: ElementClass): ComplexArg => ({
    op: { scalar: [re, im] },
    cls,
    shape: null,
    n: 1,
  });
  if (typeof v === "number") return scalar(v, 0, "double");
  if (typeof v === "boolean") return scalar(v ? 1 : 0, 0, "logical");
  if (v instanceof RuntimeComplexNumber) return scalar(v.re, v.im, "double");
  if (v instanceof RuntimeTensor && v.data.length === 1 && v.cls !== "double")
    return scalar(v.data[0], v.imag ? v.imag[0] : 0, v.cls);
  return null;
}

/** Whether `v` has complex data. */
function isComplexArg(v: RuntimeValue): boolean {
  return v instanceof RuntimeGpuArray
    ? v.isComplex
    : v instanceof RuntimeComplexNumber ||
        (v instanceof RuntimeTensor && v.imag !== undefined);
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
  cls: ElementClass,
  isComplex = false
): RuntimeGpuArray {
  return new RuntimeGpuArray(
    device,
    buffer,
    shape,
    cls,
    isComplex,
    device.generation
  );
}

/** A complex result from its parts (an `im` of null is all zeros). */
function complexResult(
  device: GpuDevice,
  k: GpuKernels,
  re: unknown,
  im: unknown | null,
  shape: number[],
  cls: ElementClass
): RuntimeGpuArray {
  return result(
    device,
    k.complex(re, im ?? zeros(k, numelOf(shape))),
    shape,
    cls,
    true
  );
}

/** A new real buffer of `n` zeros. */
function zeros(k: GpuKernels, n: number): unknown {
  return k.elementwise("times", [{ scalar: 0 }, { scalar: 0 }], n);
}

/** Run `f` on each part of `g` (real and imaginary) and make the result,
 *  complex when `g` is. */
function perPart(
  g: RuntimeGpuArray,
  k: GpuKernels,
  shape: number[],
  f: (part: unknown) => unknown
): RuntimeGpuArray {
  const p = planes(g, k);
  const re = f(p.re);
  if (p.im === null) return result(g.device, re, shape, g.cls);
  return complexResult(g.device, k, re, f(p.im), shape, g.cls);
}

/** Operations whose result is logical. */
const LOGICAL_RESULT = new Set([
  "isnan",
  "isinf",
  "isfinite",
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
  if (args.some(isComplexArg)) return complexElementwiseKernel(op, args);
  const dk = kernelsOf(args);
  if (!dk) return null;
  const xs: Arg[] = [];
  for (const a of args) {
    const x = kernelArg(a, dk.device, dk.k);
    if (!x) return null;
    xs.push(x);
  }
  const rs = resultShape(xs);
  if (!rs) return null;
  const { shape, broadcast } = rs;
  let cls: ElementClass;
  if (LOGICAL_RESULT.has(op)) cls = "logical";
  else if (xs.some(x => x.cls === "single")) cls = "single";
  else return null; // a double result (logical + 1, abs of a logical, ...)
  // On a device that fuses, the result is a lazy array (see fusion.ts),
  // inside a statement (whose end computes what remains).
  if (
    dk.k.fused &&
    !broadcast &&
    !DOMAIN_OPS.has(op) &&
    getCurrentRuntime()?.currentScope
  )
    return fuse(
      dk.device,
      op,
      xs.map(x =>
        x.gpu ? { gpu: x.gpu } : { scalar: (x.op as { scalar: number }).scalar }
      ),
      numelOf(shape),
      shape.slice(),
      cls
    );
  let dims: number[] | undefined;
  if (broadcast && shape.length > 4) return null;
  let ops = xs.map(x => x.op);
  if (broadcast) {
    dims = [...shape, 1, 1, 1].slice(0, 4);
    ops = ops.map((o, i) =>
      "scalar" in o ? o : { ...o, strides: expandStrides(xs[i].shape!, dims!) }
    );
  }
  const n = numelOf(shape);
  let buffer: unknown;
  if (DOMAIN_OPS.has(op)) {
    const r = dk.k.elementwiseChecked(op, ops, n, dims);
    if (r.bad) {
      dk.device.release?.(r.buffer);
      throw complexResultError(op);
    }
    buffer = r.buffer;
  } else buffer = dk.k.elementwise(op, ops, n, dims);
  return result(dk.device, buffer, shape.slice(), cls);
}

/** The shape of an element-wise result: that of the operands that are
 *  not scalars, or their implicit expansion (`broadcast`); null when the
 *  shapes are incompatible. */
function resultShape(
  xs: { shape: number[] | null; n: number }[]
): { shape: number[]; broadcast: boolean } | null {
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
  return { shape, broadcast };
}

/** Complex element-wise operations, and whether the result is complex. */
const COMPLEX_OPS: Record<string, boolean> = {
  plus: true,
  minus: true,
  times: true,
  rdivide: true,
  ldivide: true,
  conjtimes: true,
  uminus: true,
  conj: true,
  complex: true,
  exp: true,
  log: true,
  sqrt: true,
  power: true,
  sin: true,
  cos: true,
  real: false,
  imag: false,
  abs: false,
  angle: false,
  eq: false,
  ne: false,
  isnan: false,
  isinf: false,
  isfinite: false,
};

/** An element-wise operation with a complex operand, or one that makes a
 *  complex value (`complex`) or takes its parts (`real`, `angle`, ...). */
export function complexElementwiseKernel(
  op: string,
  args: RuntimeValue[]
): RuntimeValue | null {
  const cplx = COMPLEX_OPS[op];
  if (cplx === undefined) return null;
  const dk = kernelsOf(args);
  if (!dk) return null;
  const xs: ComplexArg[] = [];
  for (const a of args) {
    const x = complexArg(a, dk.device, dk.k);
    if (!x) return null;
    xs.push(x);
  }
  const rs = resultShape(xs);
  if (!rs) return null;
  const { shape, broadcast } = rs;
  let cls: ElementClass;
  if (LOGICAL_RESULT.has(op)) cls = "logical";
  else if (xs.some(x => x.cls === "single")) cls = "single";
  else return null;
  if (broadcast && shape.length > 4) return null;
  let dims: number[] | undefined;
  let ops = xs.map(x => x.op);
  if (broadcast) {
    dims = [...shape, 1, 1, 1].slice(0, 4);
    ops = ops.map((o, i) =>
      "scalar" in o ? o : { ...o, strides: expandStrides(xs[i].shape!, dims!) }
    );
  }
  // complex(A) has a zero imaginary part.
  if (op === "complex" && ops.length === 1) ops.push({ scalar: [0, 0] });
  const buffer = dk.k.celementwise(op, ops, numelOf(shape), cplx, dims);
  return result(dk.device, buffer, shape.slice(), cls, cplx);
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

/** `G(indices)` for a gpuArray, gathered on the device. */
export function indexKernel(
  g: RuntimeGpuArray,
  indices: RuntimeValue[]
): RuntimeValue | null {
  const k = g.device.kernels;
  if (!k || !onDevice(g, k)) return null;
  const n = numelOf(g.shape);
  const plan = planIndex(g.shape, n, indices, true);
  const shape = plan.shape.slice();
  // All elements in order (`G(:)`): the same data in a new shape.
  if (
    plan.offsets === null &&
    plan.start === 0 &&
    plan.step === 1 &&
    plan.n === n
  )
    return result(g.device, g.buffer, shape, g.cls, g.isComplex);
  const sel = selection(plan);
  return perPart(g, k, shape, src => k.take(src, sel));
}

/** `G(indices) = rhs` for a gpuArray, on the device; null when the
 *  assignment grows `G` or converts the values (into a logical array). */
export function storeKernel(
  g: RuntimeGpuArray,
  indices: RuntimeValue[],
  rhs: RuntimeValue
): RuntimeValue | null {
  const k = g.device.kernels;
  if (!k || !onDevice(g, k)) return null;
  const r = complexArg(rhs, g.device, k);
  if (!r) return null;
  if (g.cls === "logical" && r.cls !== "logical") return null;
  const cplx = g.isComplex || isComplexArg(rhs);
  if (g.cls === "logical" && cplx) return null;
  const n = numelOf(g.shape);
  const plan = planStore(g.shape, n, indices, r.n, r.shape ?? [1, 1], true);
  if (plan.grow) return null;
  const sel = selection(plan);
  const shape = g.shape.slice();
  const src = planes(g, k);
  const ro = r.op;
  const rre: KernelOperand =
    "scalar" in ro ? { scalar: ro.scalar[0] } : { buffer: ro.re, n: ro.n };
  const re = k.put(src.re, n, sel, rre);
  if (!cplx) return result(g.device, re, shape, g.cls);
  const rim: KernelOperand =
    "scalar" in ro
      ? { scalar: ro.scalar[1] }
      : ro.im === null
        ? { scalar: 0 }
        : { buffer: ro.im, n: ro.n };
  // A real G becomes complex: its imaginary part is zero elsewhere.
  const zero = src.im === null ? zeros(k, n) : null;
  const im = k.put(src.im ?? zero, n, sel, rim);
  if (zero !== null) g.device.release?.(zero);
  return complexResult(g.device, k, re, im, shape, g.cls);
}

function selection(sel: Selection): KernelSelection {
  if (sel.grid)
    return {
      n: sel.n,
      start: sel.start,
      step: 0,
      offsets: null,
      grid: sel.grid,
    };
  if (sel.offsets === null)
    return { n: sel.n, start: sel.start, step: sel.step, offsets: null };
  const offsets = new Uint32Array(sel.n);
  for (let i = 0; i < sel.n; i++) offsets[i] = sel.offsets[i] - sel.bias;
  return { n: sel.n, start: 0, step: 0, offsets };
}

const isText = (v: RuntimeValue) => isRuntimeChar(v) || isRuntimeString(v);

/** The dimension argument of a reduction: the first non-singleton
 *  dimension when absent, a 0-based dimension, `"all"`, or null for a form
 *  the kernels do not take. */
function reductionDim(
  shape: number[],
  rest: RuntimeValue[]
): number | "all" | null {
  if (rest.length === 0) {
    const d = shape.findIndex(s => s !== 1);
    return d < 0 ? 0 : d;
  }
  if (rest.length !== 1) return null;
  const r = rest[0];
  if (isText(r) && String(isRuntimeChar(r) ? r.value : r) === "all")
    return "all";
  if (typeof r === "number" && Number.isInteger(r) && r >= 1)
    return r > shape.length ? null : r - 1;
  return null;
}

/** [inner, len, outer] of `shape` around `dim`, and the shape of a
 *  reduction over it. */
function aroundDim(
  shape: number[],
  dim: number | "all"
): { inner: number; len: number; outer: number; outShape: number[] } {
  if (dim === "all")
    return { inner: 1, len: numelOf(shape), outer: 1, outShape: [1, 1] };
  const outShape = shape.slice();
  outShape[dim] = 1;
  while (outShape.length > 2 && outShape[outShape.length - 1] === 1)
    outShape.pop();
  return {
    inner: numelOf(shape.slice(0, dim)),
    len: shape[dim],
    outer: numelOf(shape.slice(dim + 1)),
    outShape,
  };
}

const bufferOf = (x: Arg) => (x.op as { buffer: unknown }).buffer;

/** `sum(G)`, `sum(G, dim)`, `sum(G, 'all')`, and likewise `prod`, `mean`,
 *  `any`, `all`, and `max(G)`, `max(G, [], dim)`, `min` (with the index
 *  output too). */
export function reduceKernel(
  op: "sum" | "prod" | "mean" | "max" | "min" | "any" | "all",
  args: RuntimeValue[],
  nargout: number
): RuntimeValue | RuntimeValue[] | null {
  const minmax = op === "max" || op === "min";
  if (nargout > (minmax ? 2 : 1)) return null;
  const dk = kernelsOf(args);
  if (!dk || !(args[0] instanceof RuntimeGpuArray)) return null;
  if (args[0].isComplex) return complexReduce(op, args, nargout);
  const x = kernelArg(args[0], dk.device, dk.k);
  if (!x || x.n === 0) return null;
  const logicalResult = op === "any" || op === "all";
  // sum/prod/mean of a logical array are double.
  if (x.cls !== "single" && !minmax && !logicalResult) return null;
  let rest = args.slice(1);
  if (minmax && rest.length > 0) {
    if (
      !(rest[0] instanceof RuntimeTensor) ||
      rest[0].data.length !== 0 ||
      rest.length !== 2
    )
      return null;
    rest = rest.slice(1);
  }
  const dim = reductionDim(x.shape!, rest);
  if (dim === null) return null;
  const { inner, len, outer, outShape } = aroundDim(x.shape!, dim);
  const nOut = inner * outer;
  const cls: ElementClass = logicalResult ? "logical" : x.cls;
  if (minmax && nargout === 2) {
    // Positions are carried as f32, exact below 2^24.
    if (len >= 2 ** 24) return null;
    const r = dk.k.reduceWithIndex(op, bufferOf(x), inner, len, outer);
    const index = dk.k.elementwise(
      "plus",
      [{ buffer: r.index, n: nOut }, { scalar: 1 }],
      nOut
    );
    dk.device.release?.(r.index);
    return [
      result(dk.device, r.value, outShape, cls),
      result(dk.device, index, outShape.slice(), "double"),
    ];
  }
  let buf = dk.k.reduce(
    op === "mean" ? "sum" : op,
    bufferOf(x),
    inner,
    len,
    outer
  );
  if (op === "mean") {
    const sum = buf;
    buf = dk.k.elementwise(
      "times",
      [{ buffer: sum, n: nOut }, { scalar: 1 / len }],
      nOut
    );
    dk.device.release?.(sum);
  }
  return result(dk.device, buf, outShape, cls);
}

/** `sum` and `mean` of a complex gpuArray, by parts. */
function complexReduce(
  op: string,
  args: RuntimeValue[],
  nargout: number
): RuntimeValue | null {
  const g = args[0] as RuntimeGpuArray;
  const k = g.device.kernels!;
  if ((op !== "sum" && op !== "mean") || nargout > 1) return null;
  if (!onDevice(g, k) || g.cls !== "single") return null;
  const n = numelOf(g.shape);
  if (n === 0) return null;
  const dim = reductionDim(g.shape, args.slice(1));
  if (dim === null) return null;
  const { inner, len, outer, outShape } = aroundDim(g.shape, dim);
  const nOut = inner * outer;
  return perPart(g, k, outShape, part => {
    const sum = k.reduce("sum", part, inner, len, outer);
    if (op === "sum") return sum;
    const mean = k.elementwise(
      "times",
      [{ buffer: sum, n: nOut }, { scalar: 1 / len }],
      nOut
    );
    g.device.release?.(sum);
    return mean;
  });
}

/** `cumsum(G)`, `cumsum(G, dim)`, and likewise `cumprod` (of real data). */
export function scanKernel(
  op: "cumsum" | "cumprod",
  args: RuntimeValue[]
): RuntimeValue | null {
  const dk = kernelsOf(args);
  const g = args[0];
  if (!dk || !(g instanceof RuntimeGpuArray)) return null;
  if (!onDevice(g, dk.k) || g.cls !== "single") return null;
  if (numelOf(g.shape) === 0 || (g.isComplex && op !== "cumsum")) return null;
  const dim = reductionDim(g.shape, args.slice(1));
  if (dim === null || dim === "all") return null;
  const { inner, len, outer } = aroundDim(g.shape, dim);
  return perPart(g, dk.k, g.shape.slice(), part =>
    dk.k.scan(op, part, inner, len, outer)
  );
}

/** `dot(a, b)` of two vectors with as many elements. */
export function dotKernel(args: RuntimeValue[]): RuntimeValue | null {
  const dk = kernelsOf(args);
  if (!dk || args.length !== 2) return null;
  if (args.some(isComplexArg)) return complexDot(args, dk.device, dk.k);
  const a = kernelArg(args[0], dk.device, dk.k);
  const b = kernelArg(args[1], dk.device, dk.k);
  if (!a || !b || !a.shape || !b.shape || a.n !== b.n || a.n === 0) return null;
  if (!isVectorShape(a.shape) || !isVectorShape(b.shape)) return null;
  if (a.cls !== "single" && b.cls !== "single") return null;
  const prod = dk.k.elementwise("times", [a.op, b.op], a.n);
  const buf = dk.k.reduce("sum", prod, 1, a.n, 1);
  dk.device.release?.(prod);
  return result(dk.device, buf, [1, 1], "single");
}

/** `dot(a, b) = sum(conj(a) .* b)` with a complex vector. */
function complexDot(
  args: RuntimeValue[],
  device: GpuDevice,
  k: GpuKernels
): RuntimeValue | null {
  const [a, b] = args;
  if (!(a instanceof RuntimeGpuArray) || !(b instanceof RuntimeGpuArray))
    return null;
  if (!onDevice(a, k) || !onDevice(b, k)) return null;
  if (a.cls !== "single" && b.cls !== "single") return null;
  const n = numelOf(a.shape);
  if (n === 0 || n !== numelOf(b.shape)) return null;
  if (!isVectorShape(a.shape) || !isVectorShape(b.shape)) return null;
  const prod = k.celementwise(
    "conjtimes",
    [
      { ...planes(a, k), n },
      { ...planes(b, k), n },
    ],
    n,
    true
  );
  const p = k.parts(prod);
  const re = k.reduce("sum", p.re, 1, n, 1);
  const im = k.reduce("sum", p.im, 1, n, 1);
  device.release?.(prod);
  return complexResult(device, k, re, im, [1, 1], "single");
}

/** `norm(v)`, `norm(v, p)` for a vector and p = 1, 2, Inf or 'fro'. */
export function normKernel(args: RuntimeValue[]): RuntimeValue | null {
  const dk = kernelsOf(args);
  if (!dk || args.length > 2 || !(args[0] instanceof RuntimeGpuArray))
    return null;
  const p = args[1];
  const kind =
    p === undefined || p === 2 || (isText(p) && textOf(p) === "fro")
      ? 2
      : p === 1
        ? 1
        : p === Infinity
          ? Infinity
          : null;
  if (kind === null) return null;
  const g = args[0];
  const release = (b: unknown) => dk.device.release?.(b);
  let x: Arg | null;
  // The norm of a complex vector is that of its absolute values.
  let abs: unknown = null;
  if (g.isComplex) {
    const n = numelOf(g.shape);
    if (!onDevice(g, dk.k) || g.cls !== "single" || n === 0) return null;
    if (!isVectorShape(g.shape)) return null;
    abs = dk.k.celementwise("abs", [{ ...planes(g, dk.k), n }], n, false);
    x = { op: { buffer: abs, n }, cls: "single", shape: g.shape, n };
  } else x = kernelArg(g, dk.device, dk.k);
  if (!x || !x.shape || x.n === 0 || x.cls !== "single") return null;
  if (!isVectorShape(x.shape)) return null;
  const n = x.n;
  let buf: unknown;
  if (kind === 2) {
    const sq = dk.k.elementwise("times", [x.op, x.op], n);
    const s = dk.k.reduce("sum", sq, 1, n, 1);
    buf = dk.k.elementwise("sqrt", [{ buffer: s, n: 1 }], 1);
    release(sq);
    release(s);
  } else {
    const a = dk.k.elementwise("abs", [x.op], n);
    buf = dk.k.reduce(kind === 1 ? "sum" : "max", a, 1, n, 1);
    release(a);
  }
  if (abs !== null) release(abs);
  return result(dk.device, buf, [1, 1], "single");
}

function isVectorShape(shape: number[]): boolean {
  return shape.length === 2 && (shape[0] === 1 || shape[1] === 1);
}

function textOf(v: RuntimeValue): string {
  return String(isRuntimeChar(v) ? v.value : v);
}

/** Concatenation along 0-based `dim` of gpuArrays and host numeric arrays
 *  (uploaded in the result's class). */
export function concatKernel(
  dim: number,
  parts: RuntimeValue[]
): RuntimeValue | null {
  const dk = kernelsOf(parts);
  if (!dk) return null;
  // MATLAB drops 0x0 operands.
  const kept = parts.filter(
    v =>
      !(
        (v instanceof RuntimeTensor || v instanceof RuntimeGpuArray) &&
        v.shape.length === 2 &&
        v.shape[0] === 0 &&
        v.shape[1] === 0
      )
  );
  const hosts: { cls: ElementClass; real: boolean }[] = kept.map(v =>
    v instanceof RuntimeGpuArray
      ? { cls: v.cls, real: !v.isComplex }
      : typeof v === "number"
        ? { cls: "double", real: true }
        : typeof v === "boolean"
          ? { cls: "logical", real: true }
          : v instanceof RuntimeTensor
            ? { cls: v.cls, real: !v.imag }
            : { cls: "double", real: false }
  );
  const cplx = hosts.some(h => !h.real);
  const cls: ElementClass = hosts.every(h => h.cls === "logical")
    ? "logical"
    : hosts.some(h => h.cls === "single")
      ? "single"
      : "double";
  if (cls === "double") return null;
  const arrays: RuntimeGpuArray[] = [];
  for (const v of kept) {
    let g: RuntimeGpuArray;
    if (!(v instanceof RuntimeGpuArray)) {
      const t =
        v instanceof RuntimeTensor
          ? v
          : v instanceof RuntimeComplexNumber
            ? new RuntimeTensor(
                Float64Array.of(v.re),
                [1, 1],
                Float64Array.of(v.im)
              )
            : new RuntimeTensor(
                Float64Array.of(
                  typeof v === "boolean" ? (v ? 1 : 0) : (v as number)
                ),
                [1, 1]
              );
      g = toGpuArray(
        cls === "single" ? singleTensor(t.data, t.shape, t.imag) : t,
        dk.device,
        false,
        "concat"
      );
    } else if (v.cls !== cls && cls === "single" && v.cls !== "logical")
      return null;
    else g = v;
    if (!onDevice(g, dk.k)) return null;
    arrays.push(g);
  }
  if (arrays.length === 0) return null;
  const nd = Math.max(dim + 1, ...arrays.map(a => a.shape.length));
  const shapes = arrays.map(a => {
    const s = a.shape.slice();
    while (s.length < nd) s.push(1);
    return s;
  });
  const ref = shapes[0];
  for (const s of shapes)
    for (let d = 0; d < nd; d++) if (d !== dim && s[d] !== ref[d]) return null;
  const inner = numelOf(ref.slice(0, dim));
  const outer = numelOf(ref.slice(dim + 1));
  const total = shapes.reduce((acc, s) => acc + s[dim], 0);
  const outShape = ref.slice();
  outShape[dim] = total;
  const blocks = shapes.map(s => inner * s[dim]);
  const offsets = blocks.map((_, i) =>
    blocks.slice(0, i).reduce((a, b) => a + b, 0)
  );
  const concat = (srcs: unknown[]) =>
    dk.k.concat(
      numelOf(outShape),
      outer,
      inner * total,
      srcs.map((src, i) => ({ src, block: blocks[i], offset: offsets[i] }))
    );
  const ps = arrays.map(a => planes(a, dk.k));
  const re = concat(ps.map(p => p.re));
  while (outShape.length > 2 && outShape[outShape.length - 1] === 1)
    outShape.pop();
  if (!cplx) return result(dk.device, re, outShape, cls);
  // A real part has a zero imaginary part.
  const temps: unknown[] = [];
  const im = concat(
    ps.map((p, i) => {
      if (p.im !== null) return p.im;
      const z = zeros(dk.k, blocks[i] * outer);
      temps.push(z);
      return z;
    })
  );
  for (const z of temps) dk.device.release?.(z);
  return complexResult(dk.device, dk.k, re, im, outShape, cls);
}

/** `repmat(G, m, n, ...)` or `repmat(G, [m n ...])`, gathered on the
 *  device with offsets computed on the host. */
export function repmatKernel(args: RuntimeValue[]): RuntimeValue | null {
  const dk = kernelsOf(args);
  const g = args[0];
  if (!dk || !(g instanceof RuntimeGpuArray) || !onDevice(g, dk.k)) return null;
  const reps: number[] = [];
  for (const a of args.slice(1)) {
    if (typeof a === "number") reps.push(a);
    else if (a instanceof RuntimeTensor && !a.imag && a.cls !== "single")
      reps.push(...a.data);
    else return null;
  }
  if (args.length === 2 && reps.length === 1) reps.push(reps[0]);
  if (reps.length < 2 || reps.some(r => !Number.isInteger(r) || r < 0))
    return null;
  const nd = Math.max(reps.length, g.shape.length);
  const src = [...g.shape, ...Array(nd - g.shape.length).fill(1)];
  const rp = [...reps, ...Array(nd - reps.length).fill(1)];
  const outShape = src.map((s, d) => s * rp[d]);
  const n = numelOf(outShape);
  const offsets = new Uint32Array(n);
  const sub = new Array(nd).fill(0);
  for (let k = 0; k < n; k++) {
    let off = 0;
    let stride = 1;
    for (let d = 0; d < nd; d++) {
      off += (sub[d] % src[d]) * stride;
      stride *= src[d];
    }
    offsets[k] = off;
    for (let d = 0; d < nd; d++) {
      if (++sub[d] < outShape[d]) break;
      sub[d] = 0;
    }
  }
  while (outShape.length > 2 && outShape[outShape.length - 1] === 1)
    outShape.pop();
  return perPart(g, dk.k, outShape, part =>
    dk.k.take(part, { n, start: 0, step: 0, offsets })
  );
}

/** `A * B` for two-dimensional single arrays, real or complex (a scalar
 *  operand is element-wise). */
export function mtimesKernel(args: RuntimeValue[]): RuntimeValue | null {
  const dk = kernelsOf(args);
  if (!dk || args.length !== 2) return null;
  const a = complexArg(args[0], dk.device, dk.k);
  const b = complexArg(args[1], dk.device, dk.k);
  if (!a || !b) return null;
  if (a.n === 1 || b.n === 1) return elementwiseKernel("times", args);
  const [sa, sb] = [a.shape!, b.shape!];
  if (sa.length !== 2 || sb.length !== 2 || sa[1] !== sb[0]) return null;
  if (a.cls !== "single" && b.cls !== "single") return null;
  const k = dk.k;
  const mm = (x: unknown, y: unknown) => k.matmul(x, y, sa[0], sa[1], sb[1]);
  const pa = planes(a.gpu!, k);
  const pb = planes(b.gpu!, k);
  const shape = [sa[0], sb[1]];
  const re = mm(pa.re, pb.re);
  if (pa.im === null && pb.im === null)
    return result(dk.device, re, shape, "single");
  if (pa.im === null || pb.im === null) {
    const im = pa.im === null ? mm(pa.re, pb.im) : mm(pa.im, pb.re);
    return complexResult(dk.device, k, re, im, shape, "single");
  }
  // (ar + i ai)(br + i bi) = (ar br - ai bi) + i (ar bi + ai br)
  const n = shape[0] * shape[1];
  const ii = mm(pa.im, pb.im);
  const ri = mm(pa.re, pb.im);
  const ir = mm(pa.im, pb.re);
  const op = (o: string, x: unknown, y: unknown) =>
    k.elementwise(
      o,
      [
        { buffer: x, n },
        { buffer: y, n },
      ],
      n
    );
  const outRe = op("minus", re, ii);
  const outIm = op("plus", ri, ir);
  for (const t of [re, ii, ri, ir]) dk.device.release?.(t);
  return complexResult(dk.device, k, outRe, outIm, shape, "single");
}

/** The transpose of a two-dimensional gpuArray; with `conj`, the complex
 *  conjugate transpose. */
export function transposeKernel(
  args: RuntimeValue[],
  conj = false
): RuntimeValue | null {
  const dk = kernelsOf(args);
  const g = args[0];
  if (!dk || args.length !== 1 || !(g instanceof RuntimeGpuArray)) return null;
  if (!onDevice(g, dk.k) || g.shape.length !== 2) return null;
  const k = dk.k;
  const [m, n] = g.shape;
  const vector = m === 1 || n === 1;
  // A vector's transpose has the same data in the other shape.
  if (vector && !(conj && g.isComplex))
    return result(dk.device, g.buffer, [n, m], g.cls, g.isComplex);
  const p = planes(g, k);
  const t = (part: unknown) =>
    vector
      ? k.elementwise(
          "times",
          [{ buffer: part, n: m * n }, { scalar: 1 }],
          m * n
        )
      : k.transpose(part, m, n);
  const re = t(p.re);
  if (p.im === null) return result(dk.device, re, [n, m], g.cls);
  let im = t(p.im);
  if (conj) {
    const neg = k.elementwise("uminus", [{ buffer: im, n: m * n }], m * n);
    dk.device.release?.(im);
    im = neg;
  }
  return complexResult(dk.device, k, re, im, [n, m], g.cls);
}

/** `fft(X)`, `fft(X, n)`, `fft(X, n, dim)` (`n` may be []), and likewise
 *  `ifft`, which also takes a final 'symmetric' or 'nonsymmetric'. As in
 *  MATLAB, the result is complex, except for `ifft(..., 'symmetric')`. */
export function fftKernel(
  args: RuntimeValue[],
  inverse: boolean
): RuntimeValue | null {
  let rest = args;
  let symmetric = false;
  const last = args[args.length - 1];
  if (inverse && args.length >= 2 && isText(last)) {
    const f = textOf(last).toLowerCase();
    if (f !== "symmetric" && f !== "nonsymmetric") return null;
    symmetric = f === "symmetric";
    rest = args.slice(0, -1);
  }
  const g = rest[0];
  const k = g instanceof RuntimeGpuArray ? g.device.kernels : undefined;
  if (rest.length > 3 || !k || !(g instanceof RuntimeGpuArray)) return null;
  if (!onDevice(g, k) || g.cls !== "single") return null;
  const nArg = sizeArg(rest[1]);
  const dimArg = sizeArg(rest[2]);
  if (nArg === null || dimArg === null) return null;
  const shape = g.shape;
  let dim: number;
  if (dimArg !== undefined) {
    if (dimArg < 1 || dimArg > shape.length) return null;
    dim = dimArg - 1;
  } else {
    dim = shape.findIndex(s => s !== 1);
    if (dim < 0) dim = 0;
  }
  const len = shape[dim];
  const n = nArg ?? len;
  if (len === 0 || n < 1) return null;
  const { inner, outer } = aroundDim(shape, dim);
  const buf = k.fft(
    g.buffer,
    g.isComplex,
    inner,
    len,
    outer,
    n,
    inverse,
    symmetric
  );
  if (buf === null) return null;
  const outShape = shape.slice();
  outShape[dim] = n;
  return result(g.device, buf, outShape, "single", !symmetric);
}

// ── Masks ─────────────────────────────────────────────────────────────
// A logical index, `find` and `nnz` select the elements of a 0/1 mask. Its
// running count (a `cumsum` scan, f32, so exact below 2^24 elements) gives
// each selected element its position; a result whose size is the count
// reads the count back from the device.

const MASK_LIMIT = 2 ** 24;

/** The 0/1 mask of real gpuArray `g`'s nonzero elements (NaN is nonzero),
 *  and whether it is a new buffer for the caller to release. */
function maskOf(
  g: RuntimeGpuArray,
  k: GpuKernels,
  n: number
): { mask: unknown; owned: boolean } {
  if (g.cls === "logical") return { mask: g.buffer, owned: false };
  const mask = k.elementwise("ne", [{ buffer: g.buffer, n }, { scalar: 0 }], n);
  return { mask, owned: true };
}

/** The inclusive running count of `mask` and its total. */
function maskPositions(
  k: GpuKernels,
  mask: unknown,
  n: number
): { pos: unknown; count: number } {
  const pos = k.scan("cumsum", mask, 1, n, 1);
  return { pos, count: k.readAt!(pos, n - 1) };
}

/** Whether `m` is a logical gpuArray mask over the `n` elements of `g`
 *  that the mask kernels take. */
function isMaskFor(
  m: RuntimeValue,
  g: RuntimeGpuArray,
  k: GpuKernels,
  n: number
): m is RuntimeGpuArray {
  return (
    m instanceof RuntimeGpuArray &&
    m.cls === "logical" &&
    m.device === g.device &&
    !m.isComplex &&
    onDevice(m, k) &&
    numelOf(m.shape) === n &&
    n > 0 &&
    n < MASK_LIMIT
  );
}

/** `G(M)` for a logical gpuArray `M` with as many elements as `G`. */
export function maskIndexKernel(
  g: RuntimeGpuArray,
  indices: RuntimeValue[]
): RuntimeValue | null {
  const k = g.device.kernels;
  if (!k?.compact || indices.length !== 1 || !onDevice(g, k)) return null;
  const n = numelOf(g.shape);
  const m = indices[0];
  if (!isMaskFor(m, g, k, n)) return null;
  const mask = m.buffer;
  const { pos, count } = maskPositions(k, mask, n);
  // A row vector gives a row; anything else a column.
  const row = g.shape.length === 2 && g.shape[0] === 1;
  const shape = row ? [1, count] : [count, 1];
  const r = perPart(
    g,
    k,
    shape,
    part => k.compact!(part, mask, pos, n, count) as unknown
  );
  g.device.release?.(pos);
  return r;
}

/** `G(M) = rhs` for a logical gpuArray `M` with as many elements as real
 *  `G`, and a real scalar or array `rhs`. */
export function maskStoreKernel(
  g: RuntimeGpuArray,
  indices: RuntimeValue[],
  rhs: RuntimeValue
): RuntimeValue | null {
  const k = g.device.kernels;
  if (!k?.expand || indices.length !== 1 || !onDevice(g, k) || g.isComplex)
    return null;
  const n = numelOf(g.shape);
  const m = indices[0];
  if (!isMaskFor(m, g, k, n)) return null;
  const r = kernelArg(rhs, g.device, k);
  if (!r || (g.cls === "logical" && r.cls !== "logical")) return null;
  const mask = m.buffer;
  if (r.n === 1)
    return result(
      g.device,
      k.expand(g.buffer, mask, null, n, r.op),
      g.shape.slice(),
      g.cls
    );
  // An array: one element per selected position, checked on the host.
  const { pos, count } = maskPositions(k, mask, n);
  const out = count === r.n ? k.expand(g.buffer, mask, pos, n, r.op) : null;
  g.device.release?.(pos);
  return out === null ? null : result(g.device, out, g.shape.slice(), g.cls);
}

/** `find(X)`, `find(X, k)`, `find(X, k, 'first')` and `[r, c] = find(...)`
 *  of a real gpuArray: double indices (f32 on the device, exact below
 *  2^24). An empty result runs on the host, which gives its shape. */
export function findKernel(
  args: RuntimeValue[],
  nargout: number
): RuntimeValue | RuntimeValue[] | null {
  const g = args[0];
  if (!(g instanceof RuntimeGpuArray) || nargout > 2 || args.length > 3)
    return null;
  const k = g.device.kernels;
  if (!k?.compact || !onDevice(g, k) || g.isComplex) return null;
  let limit = Infinity;
  if (args.length >= 2) {
    const lim = args[1];
    if (typeof lim !== "number" || !Number.isInteger(lim) || lim < 1)
      return null;
    limit = lim;
  }
  if (args.length === 3 && textOf(args[2]).toLowerCase() !== "first")
    return null;
  const n = numelOf(g.shape);
  if (n < 2 || n >= MASK_LIMIT) return null;
  const { mask, owned } = maskOf(g, k, n);
  const { pos, count: total } = maskPositions(k, mask, n);
  const count = Math.min(total, limit);
  let out: RuntimeValue | RuntimeValue[] | null = null;
  if (count > 0) {
    const row = g.shape.length === 2 && g.shape[0] === 1;
    const shape = row ? [1, count] : [count, 1];
    if (nargout <= 1) {
      const idx = k.compact(null, mask, pos, n, count);
      out = result(g.device, idx, shape, "double");
    } else {
      const [r, c] = k.compact(null, mask, pos, n, count, g.shape[0]) as [
        unknown,
        unknown,
      ];
      out = [
        result(g.device, r, shape.slice(), "double"),
        result(g.device, c, shape, "double"),
      ];
    }
  }
  g.device.release?.(pos);
  if (owned) g.device.release?.(mask);
  return out;
}

/** `nnz(X)` of a real gpuArray: a double scalar (exact below 2^24). */
export function nnzKernel(args: RuntimeValue[]): RuntimeValue | null {
  const g = args[0];
  if (args.length !== 1 || !(g instanceof RuntimeGpuArray)) return null;
  const k = g.device.kernels;
  if (!k || !onDevice(g, k) || g.isComplex) return null;
  const n = numelOf(g.shape);
  if (n === 0 || n >= MASK_LIMIT) return null;
  const { mask, owned } = maskOf(g, k, n);
  const sum = k.reduce("sum", mask, 1, n, 1);
  if (owned) g.device.release?.(mask);
  return result(g.device, sum, [1, 1], "double");
}

/** `single(G)` or `logical(G)` on the device: the array itself when it
 *  has the class, and the same buffer for `single(L)` of a logical array
 *  (its 0/1 values are f32 data). Other conversions run on the host. */
export function castKernel(
  cls: "single" | "logical",
  args: RuntimeValue[]
): RuntimeValue | null {
  const g = args[0];
  if (args.length !== 1 || !(g instanceof RuntimeGpuArray)) return null;
  const k = g.device.kernels;
  if (!k || !onDevice(g, k)) return null;
  if (g.cls === cls) return g;
  if (cls === "single" && g.cls === "logical")
    return result(g.device, g.buffer, g.shape.slice(), "single");
  return null;
}

/** `reshape(G, m, n, ...)` or `reshape(G, [m n ...])`, with at most one
 *  `[]`: the same buffer in the new shape. An invalid size runs on the
 *  host, which raises the error. */
export function reshapeKernel(args: RuntimeValue[]): RuntimeValue | null {
  const g = args[0];
  if (args.length < 2 || !(g instanceof RuntimeGpuArray)) return null;
  const k = g.device.kernels;
  if (!k || !onDevice(g, k)) return null;
  let dims: (number | null)[];
  const v = args[1];
  if (args.length === 2) {
    if (!(v instanceof RuntimeTensor) || v.imag || v.data.length < 2)
      return null;
    dims = Array.from(v.data);
  } else {
    dims = [];
    for (const a of args.slice(1)) {
      if (typeof a === "number") dims.push(a);
      else if (a instanceof RuntimeTensor && a.data.length === 0)
        dims.push(null);
      else return null;
    }
  }
  const n = numelOf(g.shape);
  let known = 1;
  let free = -1;
  for (let d = 0; d < dims.length; d++) {
    const x = dims[d];
    if (x === null) {
      if (free >= 0) return null;
      free = d;
    } else if (!Number.isInteger(x) || x < 0) return null;
    else known *= x;
  }
  if (free >= 0) {
    if (known === 0 || n % known !== 0) return null;
    dims[free] = n / known;
  } else if (known !== n) return null;
  const shape = dims as number[];
  while (shape.length > 2 && shape[shape.length - 1] === 1) shape.pop();
  return result(g.device, g.buffer, shape, g.cls, g.isComplex);
}

/** The element count of a value (1 for a scalar), or -1 for a value that
 *  is not numeric or logical. */
function numelOfValue(v: RuntimeValue): number {
  if (typeof v === "number" || typeof v === "boolean") return 1;
  if (v instanceof RuntimeComplexNumber) return 1;
  if (v instanceof RuntimeGpuArray) return numelOf(v.shape);
  if (v instanceof RuntimeTensor) return v.data.length;
  return -1;
}

/** `A / s` and `s \ A` with a scalar `s`: element-wise division. */
export function scalarDivideKernel(
  op: "mrdivide" | "mldivide",
  args: RuntimeValue[]
): RuntimeValue | null {
  if (args.length !== 2) return null;
  return op === "mrdivide"
    ? numelOfValue(args[1]) === 1
      ? elementwiseKernel("rdivide", args)
      : null
    : numelOfValue(args[0]) === 1
      ? elementwiseKernel("ldivide", args)
      : null;
}

/** A length or dimension argument: undefined when absent or [], a
 *  positive integer, or null for anything else. */
function sizeArg(v: RuntimeValue | undefined): number | undefined | null {
  if (v === undefined) return undefined;
  if (v instanceof RuntimeTensor && v.data.length === 0) return undefined;
  if (typeof v === "number" && Number.isInteger(v) && v >= 1) return v;
  return null;
}
