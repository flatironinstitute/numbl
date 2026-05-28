/**
 * Value adapter for the mtoc2 C-JIT path.
 *
 * Mirrors `valueAdapter.ts` (JS path) but converts numbl
 * `RuntimeValue` to / from the C ABI mtoc2 expects across the koffi
 * FFI boundary.
 *
 * **Copy budget** (see CLAUDE memo `feedback-no-silent-bails.md`
 * and the user's "no extra copies" guideline):
 *   - Scalars: zero copies; passed by value in koffi's args array.
 *   - Tensors: one copy in, one copy out. mtoc2's call convention
 *     requires the callee own its tensor params (it frees them at
 *     scope exit), so the input data must be copied into a
 *     `koffi.alloc`'d malloc-backed buffer that mtoc2 can free.
 *     Reused / shared buffer would either (a) get freed under
 *     numbl's still-live Float64Array, or (b) require extending
 *     mtoc2 with borrowed-param semantics — deferred.
 *
 *   - Output tensors arrive as `mtoc2_tensor_t` with mtoc2_alloc'd
 *     `real`/`imag` pointers. Caller (this adapter) copies the data
 *     into a fresh `Float64Array`, then calls `free()` on the
 *     pointers (`koffi`-bound `free` from libc) to release mtoc2's
 *     buffers.
 */

import {
  RuntimeTensor,
  isRuntimeTensor,
  isRuntimeNumber,
  isRuntimeLogical,
  type RuntimeValue,
} from "../../runtime/types.js";
import type { Type, NumericType } from "../../mtoc2/index.js";
import { isMultiElement } from "../../mtoc2/index.js";
import type { KoffiLib } from "./compileC.js";
import { MTOC2_MAX_NDIM } from "./typeAdapterC.js";

/** Loose koffi typing — same shape as in typeAdapterC.ts. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type KoffiAny = any;
interface Koffi {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  alloc(type: string, count: number): any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  decode(ptr: unknown, type: string, count: number): any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  encode(buf: any, type: string, value: any): void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  address(buf: any): any;
}

/** A function that frees the `mtoc2_alloc`'d buffer at the given
 *  pointer. Resolved via koffi against libc's `free`. The .so we
 *  dlopen pulls in libc anyway, so binding through the same lib is
 *  zero-cost. */
type FreeFn = (ptr: unknown) => void;

interface MarshalCtx {
  readonly koffi: Koffi;
  readonly free: FreeFn;
}

/** Build a marshaling context from a koffi lib handle. The host
 *  passes the lib it got from `compileAndLoadC`; we resolve `free`
 *  against that same lib (libc symbols are reachable through any
 *  dlopen'd .so). */
export function makeCMarshalCtx(koffi: Koffi, lib: KoffiLib): MarshalCtx {
  // `mtoc2_jit_free` is a tiny wrapper around libc's `free` that
  // mtoc2's emit always exports from a JIT-mode .so (see
  // `mtoc2/src/codegen/emit.ts` exposeSpec branch). Binding through
  // the .so itself avoids any cross-platform libc loader dance.
  const free = lib.func("void mtoc2_jit_free(void *)") as FreeFn;
  return { koffi, free };
}

/** Marshaled-input bookkeeping returned by `marshalInputs`. The
 *  caller is responsible for invoking `release()` once the C call
 *  has returned (success or throw). `release` does nothing for
 *  scalar inputs (no allocation); for tensor inputs it's also a
 *  no-op: mtoc2's spec body frees the buffers at its scope exit.
 *  Kept as an explicit handle so a future change to lifetime
 *  semantics has a single place to update. */
export interface MarshaledInputs {
  /** The values to splat into the koffi call, in order. */
  readonly args: unknown[];
  release(): void;
}

/** Convert a numbl `RuntimeValue` to the matching koffi-callable
 *  arg for a parameter of mtoc2 `Type`. */
export function marshalInputs(
  ctx: MarshalCtx,
  paramTypes: ReadonlyArray<Type>,
  values: ReadonlyArray<RuntimeValue | undefined>
): MarshaledInputs | null {
  const args: unknown[] = [];
  for (let i = 0; i < paramTypes.length; i++) {
    const ty = paramTypes[i];
    const v = values[i];
    if (v === undefined) return null;
    const arg = marshalOneInput(ctx, ty, v);
    if (arg === undefined) return null;
    args.push(arg);
  }
  return {
    args,
    release: () => {
      // mtoc2 owns + frees tensor input buffers at scope exit; we
      // have nothing to do here today. Hook left in place so a
      // future borrowed-param mode can free our own koffi.alloc
      // buffers from this site.
    },
  };
}

function marshalOneInput(ctx: MarshalCtx, ty: Type, v: RuntimeValue): unknown {
  if (ty.kind === "Numeric") {
    const nty = ty as NumericType;
    if (isMultiElement(nty)) {
      if (!isRuntimeTensor(v)) return undefined;
      return marshalTensorInput(ctx, v);
    }
    // Scalar — mtoc2's C ABI passes every single-element numeric
    // (real, logical) as `double` (see typeAdapterC.ts's
    // numericToCDecl). Coerce booleans to 0/1 so koffi marshals a
    // proper double.
    if (isRuntimeNumber(v)) return v;
    if (isRuntimeLogical(v)) return v ? 1 : 0;
    return undefined;
  }
  return undefined;
}

/** Build a koffi-allocated `mtoc2_tensor_t` whose `real`/`imag`
 *  pointers point to fresh malloc-backed buffers carrying a copy of
 *  the numbl tensor's data. mtoc2's spec body owns and frees these
 *  buffers at its scope exit. */
function marshalTensorInput(ctx: MarshalCtx, v: RuntimeTensor): unknown {
  const n = v.data.length;
  // `koffi.alloc("double", n)` returns a koffi buffer that's also
  // an iterable view backed by malloc'd memory. `encode` copies the
  // Float64Array into it. The C side sees the same byte layout.
  const realBuf = ctx.koffi.alloc("double", Math.max(n, 1));
  ctx.koffi.encode(realBuf, "double", v.data);
  let imagBuf: unknown = null;
  if (v.imag !== undefined) {
    imagBuf = ctx.koffi.alloc("double", Math.max(n, 1));
    ctx.koffi.encode(imagBuf, "double", v.imag);
  }
  // Build the struct as a plain JS object — koffi marshals into the
  // C layout based on the `mtoc2_tensor_t` struct registration.
  const shape = [...v.shape];
  while (shape.length < 2) shape.push(1); // mtoc2 minimum ndim is 2.
  const dims = new Array(MTOC2_MAX_NDIM).fill(0);
  for (let k = 0; k < shape.length && k < MTOC2_MAX_NDIM; k++) {
    dims[k] = shape[k];
  }
  return {
    real: realBuf,
    imag: imagBuf,
    ndim: shape.length,
    dims,
  };
}

/** Convert a returned C tensor struct (decoded by koffi into a JS
 *  object with `{real, imag, ndim, dims}`) to a numbl RuntimeTensor.
 *  Reads `n = product(dims[0..ndim])` doubles from each lane, then
 *  frees mtoc2's buffers.
 *
 *  For `Numeric` outputs that aren't multi-element (scalars), the
 *  caller passes the bare value through `unmarshalScalarOutput`
 *  instead — koffi unboxes them automatically.
 */
export function unmarshalTensorOutput(
  ctx: MarshalCtx,
  ret: KoffiAny,
  ty: NumericType
): RuntimeTensor {
  const ndim = ret.ndim as number;
  const dimsArr = ret.dims as number[];
  const shape = dimsArr.slice(0, ndim);
  let n = 1;
  for (const d of shape) n *= d;
  const realPtr = ret.real;
  const imagPtr = ret.imag;
  const realArr = ctx.koffi.decode(realPtr, "double", Math.max(n, 1));
  const realData = new Float64Array(n);
  for (let i = 0; i < n; i++) realData[i] = realArr[i];
  let imagData: Float64Array | undefined;
  if (ty.isComplex && imagPtr) {
    const imagArr = ctx.koffi.decode(imagPtr, "double", Math.max(n, 1));
    imagData = new Float64Array(n);
    for (let i = 0; i < n; i++) imagData[i] = imagArr[i];
  }
  // Release mtoc2's buffers. mtoc2 alloc'd them with the same
  // `malloc` that libc's `free` consumes (see runtime/tensor/alloc.h).
  ctx.free(realPtr);
  if (imagPtr) ctx.free(imagPtr);
  return new RuntimeTensor(realData, shape, imagData);
}

/** Convert a bare scalar return value (`double`) to a numbl
 *  RuntimeValue. Every scalar mtoc2-emitted spec returns `double`
 *  regardless of element kind (see typeAdapterC.ts), so koffi hands
 *  us a JS number; logical outputs coerce to `Boolean` here. */
export function unmarshalScalarOutput(
  v: unknown,
  ty: NumericType
): RuntimeValue {
  if (ty.elem === "logical") return Boolean(v);
  return v as number;
}
