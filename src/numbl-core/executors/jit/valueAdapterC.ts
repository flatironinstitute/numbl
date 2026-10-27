/**
 * Value adapter for the mtoc2 C-JIT path.
 *
 * Mirrors `valueAdapter.ts` (JS path) but converts numbl
 * `RuntimeValue` to / from the C ABI mtoc2 expects across the koffi
 * FFI boundary.
 *
 * **Copy budget**:
 *   - Scalars: zero copies; passed by value in koffi's args array.
 *   - Tensors: one copy in, one copy out. The callee owns (and frees at
 *     scope exit) its tensor params, so input data is copied into
 *     buffers from the `.so`'s own `mtoc2_jit_malloc`. Output tensors
 *     arrive with callee-allocated `real`/`imag`; they are copied into
 *     fresh `Float64Array`s and released with `mtoc2_jit_free`. Both
 *     copies are bulk `koffi.encode` / `koffi.decode` calls.
 */

import {
  RuntimeTensor,
  RuntimeComplexNumber,
  isRuntimeTensor,
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeComplexNumber,
  type RuntimeValue,
} from "../../runtime/types.js";
import type { Type, NumericType } from "../../jit/index.js";
import { isMultiElement } from "../../jit/index.js";
import type { KoffiLib } from "./compileC.js";
import { JIT_MAX_NDIM } from "./typeAdapterC.js";

/** Loose koffi typing — same shape as in typeAdapterC.ts. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type KoffiAny = any;
interface Koffi {
  encode(ptr: unknown, type: string, value: Float64Array, len: number): void;
  decode(ptr: unknown, type: string, len: number): Float64Array;
}

/** A function that frees the `mtoc2_alloc`'d buffer at the given
 *  pointer. Resolved via koffi against libc's `free`. The .so we
 *  dlopen pulls in libc anyway, so binding through the same lib is
 *  zero-cost. */
type FreeFn = (ptr: unknown) => void;
type MallocFn = (bytes: number) => unknown;

interface MarshalCtx {
  readonly koffi: Koffi;
  readonly malloc: MallocFn;
  readonly free: FreeFn;
}

/** koffi callback type for the C `void (*)(const char *, long)` host
 *  output hook. Created once (koffi rejects re-declaring a named
 *  proto) and reused across runs. */
let hostWriteCbType: unknown | null = null;
function hostWriteCallbackType(koffi: KoffiAny): unknown {
  if (hostWriteCbType === null) {
    // `void *` (not `const char *`): koffi auto-decodes a char* callback
    // param to a NUL-terminated JS string, which both ignores `len` and
    // mis-handles our non-NUL-terminated byte slices. A raw pointer lets
    // us decode exactly `len` bytes. ABI-identical to the C side's
    // `const char *`.
    const proto = koffi.proto(
      "void mtoc2_host_write_cb(void *bytes, long len)"
    );
    hostWriteCbType = koffi.pointer(proto);
  }
  return hostWriteCbType;
}

/** Output sink of the C call in flight. One koffi callback is registered
 *  per process and forwards here, so binding a `.so` costs no per-call
 *  registration. */
let hostWriteSink: ((s: string) => void) | null = null;
let hostWriteCb: unknown = null;
function hostWriteCallback(koffi: KoffiAny): unknown {
  if (hostWriteCb === null) {
    const decoder = new TextDecoder();
    hostWriteCb = koffi.register((bytes: unknown, len: number | bigint) => {
      const n = Number(len);
      if (n <= 0 || hostWriteSink === null) return;
      hostWriteSink(decoder.decode(new Uint8Array(koffi.view(bytes, n))));
    }, hostWriteCallbackType(koffi));
  }
  return hostWriteCb;
}

export interface HostWriteBinding {
  /** Run `fn` (the C call) with the `.so`'s output routed to `write`. */
  run<T>(write: (s: string) => void, fn: () => T): T;
}

/** Bind numbl's output sink into an emitted C `.so` so program output
 *  (disp / fprintf / …) routes through the host instead of libc stdout
 *  (see `runtime/io/host_output.h`). Done once per loaded `.so`.
 *
 *  When the spec emitted no output, `mtoc2_set_host_write` is absent
 *  from the `.so`; binding then no-ops (the C code never writes). */
export function bindHostWrite(
  koffi: KoffiAny,
  lib: KoffiLib
): HostWriteBinding {
  let setHostWrite: (cb: unknown) => void;
  try {
    setHostWrite = lib.func("void mtoc2_set_host_write(void *)") as (
      cb: unknown
    ) => void;
  } catch {
    return { run: (_write, fn) => fn() };
  }
  setHostWrite(hostWriteCallback(koffi));
  return {
    run(write, fn) {
      const prev = hostWriteSink;
      hostWriteSink = write;
      try {
        return fn();
      } finally {
        hostWriteSink = prev;
      }
    },
  };
}

/** Bind the emitted `.so`'s grow-bail flag accessors. A runtime
 *  indexed-store grow (`v(k) = x`, k past the runtime extent) sets a
 *  flag in the `.so` and `longjmp`s back to the host-entry function's
 *  guard (see `grow_bail.h`), which returns normally. The host calls
 *  `reset()` before each invocation and `bailed()` after; when
 *  `bailed()` is true it discards the outputs and bails to the
 *  interpreter (which has full MATLAB grow semantics).
 *
 *  When the spec emitted no scalar indexed store, `mtoc2_grow_bail_*`
 *  are absent from the `.so`; binding then no-ops (`bailed()` is always
 *  false — the code can't grow). */
export function bindGrowBail(lib: KoffiLib): {
  reset(): void;
  bailed(): boolean;
} {
  let reset: () => void;
  let check: () => number;
  try {
    reset = lib.func("void mtoc2_grow_bail_reset(void)") as () => void;
    check = lib.func("int mtoc2_grow_bail_check(void)") as () => number;
  } catch {
    return { reset() {}, bailed: () => false };
  }
  return { reset, bailed: () => check() !== 0 };
}

/** Build a marshaling context from a koffi lib handle. `mtoc2_jit_malloc`
 *  / `mtoc2_jit_free` are exported by every JIT `.so` (see
 *  `jit/codegen/emit.ts`), so host-side buffers use the same allocator
 *  the emitted code frees with. */
export function makeCMarshalCtx(koffi: Koffi, lib: KoffiLib): MarshalCtx {
  const malloc = lib.func("void *mtoc2_jit_malloc(long)") as MallocFn;
  const free = lib.func("void mtoc2_jit_free(void *)") as FreeFn;
  return { koffi, malloc, free };
}

/** Marshaled-input bookkeeping returned by `marshalInputs`. The callee
 *  takes ownership of tensor buffers once it is called; `release()`
 *  frees them for a call that never happened (marshaling or koffi
 *  argument-conversion failure). */
export interface MarshaledInputs {
  /** The values to splat into the koffi call, in order. */
  readonly args: unknown[];
  release(): void;
}

/** Convert numbl `RuntimeValue`s to koffi-callable args for parameters of
 *  the given compiler `Type`s, or null if a value doesn't fit its type. */
export function marshalInputs(
  ctx: MarshalCtx,
  paramTypes: ReadonlyArray<Type>,
  values: ReadonlyArray<RuntimeValue | undefined>
): MarshaledInputs | null {
  const args: unknown[] = [];
  const owned: unknown[] = [];
  const release = () => {
    for (const p of owned) ctx.free(p);
    owned.length = 0;
  };
  for (let i = 0; i < paramTypes.length; i++) {
    const ty = paramTypes[i];
    const v = values[i];
    const arg =
      v === undefined ? undefined : marshalOneInput(ctx, ty, v, owned);
    if (arg === undefined) {
      release();
      return null;
    }
    args.push(arg);
  }
  return { args, release };
}

function marshalOneInput(
  ctx: MarshalCtx,
  ty: Type,
  v: RuntimeValue,
  owned: unknown[]
): unknown {
  if (ty.kind === "Numeric") {
    const nty = ty as NumericType;
    if (isMultiElement(nty)) {
      if (!isRuntimeTensor(v)) return undefined;
      // A real-typed param can't carry an imaginary part.
      if (!nty.isComplex && v.imag !== undefined) return undefined;
      return marshalTensorInput(ctx, v, nty.isComplex, owned);
    }
    // Complex scalar — the C ABI is `double _Complex`, marshaled via
    // the ABI-identical `{re, im}` koffi struct (see typeAdapterC.ts).
    // A real value flowing into a complex param rides `im = 0`.
    if (nty.isComplex) {
      if (isRuntimeComplexNumber(v)) return { re: v.re, im: v.im };
      if (isRuntimeNumber(v)) return { re: v, im: 0 };
      if (isRuntimeLogical(v)) return { re: v ? 1 : 0, im: 0 };
      // A 1×1 tensor compiled as a scalar param: unwrap to its element
      // (the compiler collapses [1,1] to a scalar; without this the
      // tensor value has no scalar slot → undefined → NaN).
      if (isRuntimeTensor(v) && v.data.length === 1) {
        return { re: v.data[0], im: v.imag?.[0] ?? 0 };
      }
      return undefined;
    }
    // Scalar — mtoc2's C ABI passes every single-element real numeric
    // (real, logical) as `double` (see typeAdapterC.ts's
    // numericToCDecl). Coerce booleans to 0/1 so koffi marshals a
    // proper double.
    if (isRuntimeNumber(v)) return v;
    if (isRuntimeLogical(v)) return v ? 1 : 0;
    // A 1×1 tensor compiled as a scalar param: unwrap to its element.
    if (isRuntimeTensor(v) && v.data.length === 1) return v.data[0];
    return undefined;
  }
  return undefined;
}

/** Copy `src` into a fresh `mtoc2_jit_malloc` buffer (or zero-fill
 *  `n` doubles when `src` is undefined). */
function copyIn(
  ctx: MarshalCtx,
  src: Float64Array | undefined,
  n: number,
  owned: unknown[]
): unknown {
  const ptr = ctx.malloc(Math.max(n, 1) * 8);
  if (ptr === null) throw new Error("mtoc2_jit_malloc: out of memory");
  owned.push(ptr);
  if (n > 0) {
    ctx.koffi.encode(ptr, "double", src ?? new Float64Array(n), n);
  }
  return ptr;
}

/** Build an `mtoc2_tensor_t` whose `real`/`imag` point to fresh buffers
 *  holding a copy of the numbl tensor's data. A complex-typed param
 *  always gets an `imag` lane (zeros for a real value), since the
 *  emitted code reads it unconditionally. */
function marshalTensorInput(
  ctx: MarshalCtx,
  v: RuntimeTensor,
  isComplex: boolean,
  owned: unknown[]
): unknown {
  const shape = [...v.shape];
  while (shape.length < 2) shape.push(1); // mtoc2 minimum ndim is 2.
  if (shape.length > JIT_MAX_NDIM) return undefined;
  const n = v.data.length;
  const real = copyIn(ctx, v.data, n, owned);
  const imag = isComplex ? copyIn(ctx, v.imag, n, owned) : null;
  const dims = new Array(JIT_MAX_NDIM).fill(0);
  for (let k = 0; k < shape.length; k++) dims[k] = shape[k];
  return { real, imag, ndim: shape.length, dims };
}

/** Copy `n` doubles out of a callee-allocated buffer. */
function copyOut(ctx: MarshalCtx, ptr: unknown, n: number): Float64Array {
  if (n === 0) return new Float64Array(0);
  // A `double` decode with a length copies into a fresh Float64Array.
  return ctx.koffi.decode(ptr, "double", n);
}

/** Convert a returned C tensor struct (decoded by koffi into a JS object
 *  with `{real, imag, ndim, dims}`) to a numbl RuntimeTensor, then free
 *  the callee's buffers. Scalar outputs go through
 *  `unmarshalScalarOutput` instead. */
export function unmarshalTensorOutput(
  ctx: MarshalCtx,
  ret: KoffiAny,
  ty: NumericType
): RuntimeTensor {
  const ndim = ret.ndim as number;
  const shape = (ret.dims as number[]).slice(0, ndim).map(Number);
  let n = 1;
  for (const d of shape) n *= d;
  const realPtr = ret.real;
  const imagPtr = ret.imag;
  const realData = copyOut(ctx, realPtr, n);
  const imagData =
    ty.isComplex && imagPtr ? copyOut(ctx, imagPtr, n) : undefined;
  if (realPtr) ctx.free(realPtr);
  if (imagPtr) ctx.free(imagPtr);
  return new RuntimeTensor(
    realData,
    shape,
    imagData,
    ty.elem === "logical" ? true : undefined
  );
}

/** Convert a bare scalar return value to a numbl RuntimeValue.
 *
 *  Real / logical scalars ride the `double` ABI, so koffi hands us a
 *  JS number (logical coerces to `Boolean`). Complex scalars ride the
 *  `mtoc2_cscalar_t` struct, so koffi hands us a `{re, im}` object,
 *  which becomes a `RuntimeComplexNumber`. */
export function unmarshalScalarOutput(
  v: unknown,
  ty: NumericType
): RuntimeValue {
  if (ty.isComplex) {
    const c = v as { re: number; im: number };
    return new RuntimeComplexNumber(c.re, c.im);
  }
  if (ty.elem === "logical") return Boolean(v);
  return v as number;
}
