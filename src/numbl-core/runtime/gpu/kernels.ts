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

/** `cumsum(G)`, `cumsum(G, dim)`, and likewise `cumprod`. */
export function scanKernel(
  op: "cumsum" | "cumprod",
  args: RuntimeValue[]
): RuntimeValue | null {
  const dk = kernelsOf(args);
  if (!dk || !(args[0] instanceof RuntimeGpuArray)) return null;
  const x = kernelArg(args[0], dk.device, dk.k);
  if (!x || x.n === 0 || x.cls !== "single") return null;
  const dim = reductionDim(x.shape!, args.slice(1));
  if (dim === null || dim === "all") return null;
  const { inner, len, outer } = aroundDim(x.shape!, dim);
  const buf = dk.k.scan(op, bufferOf(x), inner, len, outer);
  return result(dk.device, buf, x.shape!.slice(), "single");
}

/** `dot(a, b)` of two vectors with as many elements. */
export function dotKernel(args: RuntimeValue[]): RuntimeValue | null {
  const dk = kernelsOf(args);
  if (!dk || args.length !== 2) return null;
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

/** `norm(v)`, `norm(v, p)` for a vector and p = 1, 2, Inf or 'fro'. */
export function normKernel(args: RuntimeValue[]): RuntimeValue | null {
  const dk = kernelsOf(args);
  if (!dk || args.length > 2 || !(args[0] instanceof RuntimeGpuArray))
    return null;
  const x = kernelArg(args[0], dk.device, dk.k);
  if (!x || !x.shape || x.n === 0 || x.cls !== "single") return null;
  if (!isVectorShape(x.shape)) return null;
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
  const n = x.n;
  const release = (b: unknown) => dk.device.release?.(b);
  let buf: unknown;
  if (kind === 2) {
    const sq = dk.k.elementwise("times", [x.op, x.op], n);
    const s = dk.k.reduce("sum", sq, 1, n, 1);
    buf = dk.k.elementwise("sqrt", [{ buffer: s, n: 1 }], 1);
    release(sq);
    release(s);
  } else {
    const abs = dk.k.elementwise("abs", [x.op], n);
    buf = dk.k.reduce(kind === 1 ? "sum" : "max", abs, 1, n, 1);
    release(abs);
  }
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
  if (hosts.some(h => !h.real)) return null;
  const cls: ElementClass = hosts.every(h => h.cls === "logical")
    ? "logical"
    : hosts.some(h => h.cls === "single")
      ? "single"
      : "double";
  if (cls === "double") return null;
  const args: Arg[] = [];
  for (const v of kept) {
    let g: RuntimeValue = v;
    if (!(v instanceof RuntimeGpuArray)) {
      const t =
        v instanceof RuntimeTensor
          ? v
          : new RuntimeTensor(
              Float64Array.of(
                typeof v === "boolean" ? (v ? 1 : 0) : (v as number)
              ),
              [1, 1]
            );
      g = toGpuArray(
        cls === "single" ? singleTensor(t.data, t.shape) : t,
        dk.device,
        false
      );
    } else if (v.cls !== cls && cls === "single" && v.cls !== "logical")
      return null;
    const x = kernelArg(g, dk.device, dk.k);
    if (!x) return null;
    args.push(x);
  }
  if (args.length === 0) return null;
  const nd = Math.max(dim + 1, ...args.map(a => a.shape!.length));
  const shapes = args.map(a => {
    const s = a.shape!.slice();
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
  let offset = 0;
  const specs = args.map((a, i) => {
    const block = inner * shapes[i][dim];
    const spec = { src: bufferOf(a), block, offset };
    offset += block;
    return spec;
  });
  const buf = dk.k.concat(numelOf(outShape), outer, inner * total, specs);
  while (outShape.length > 2 && outShape[outShape.length - 1] === 1)
    outShape.pop();
  return result(dk.device, buf, outShape, cls);
}

/** `repmat(G, m, n, ...)` or `repmat(G, [m n ...])`, gathered on the
 *  device with offsets computed on the host. */
export function repmatKernel(args: RuntimeValue[]): RuntimeValue | null {
  const dk = kernelsOf(args);
  if (!dk || !(args[0] instanceof RuntimeGpuArray)) return null;
  const x = kernelArg(args[0], dk.device, dk.k);
  if (!x || !x.shape) return null;
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
  const nd = Math.max(reps.length, x.shape.length);
  const src = [...x.shape, ...Array(nd - x.shape.length).fill(1)];
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
  const buf = dk.k.take(bufferOf(x), { n, start: 0, step: 0, offsets });
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
