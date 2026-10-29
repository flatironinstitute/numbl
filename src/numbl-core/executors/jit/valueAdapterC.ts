/**
 * Value adapter for the C-JIT path.
 *
 * Mirrors `valueAdapter.ts` (JS path) but converts numbl
 * `RuntimeValue` to / from the C ABI the JIT expects across the koffi
 * FFI boundary.
 *
 * **Copy budget**:
 *   - Scalars: zero copies; passed by value in koffi's args array.
 *   - Tensors: one copy in, one copy out. The callee owns (and frees at
 *     scope exit) its tensor params, so input data is copied into
 *     buffers from the `.so`'s own `jit_malloc`. Output tensors
 *     arrive with callee-allocated `real`/`imag`; they are copied into
 *     fresh `Float64Array`s and released with `jit_free`. Both
 *     copies are bulk `koffi.encode` / `koffi.decode` calls.
 */

import {
  RuntimeTensor,
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
import { complexScalarOut, narrowedImag } from "./valueAdapter.js";
import { getLapackBridge } from "../../native/lapack-bridge.js";
import { MATMUL_INLINE_MAX_WORK } from "../../native/inline-matmul.js";

/** Loose koffi typing — same shape as in typeAdapterC.ts. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type KoffiAny = any;
interface Koffi {
  encode(ptr: unknown, type: string, value: Float64Array, len: number): void;
  decode(ptr: unknown, type: string, len: number): Float64Array;
}

/** A function that frees the `jit_alloc`'d buffer at the given
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
    const proto = koffi.proto("void jit_host_write_cb(void *bytes, long len)");
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
 *  When the spec emitted no output, `jit_set_host_write` is absent
 *  from the `.so`; binding then no-ops (the C code never writes). */
export function bindHostWrite(
  koffi: KoffiAny,
  lib: KoffiLib
): HostWriteBinding {
  let setHostWrite: (cb: unknown) => void;
  try {
    setHostWrite = lib.func("void jit_set_host_write(void *)") as (
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

/** The per-process koffi callbacks behind `jit_set_host_numeric`: views
 *  over the C buffers, run through the LAPACK provider stack, and copy the
 *  result into the C output. Return 0, or 1 when the provider threw. */
let hostNumericCbs: [unknown, unknown] | null = null;
function hostNumericCallbacks(koffi: KoffiAny): [unknown, unknown] {
  if (hostNumericCbs !== null) return hostNumericCbs;
  const f64 = (p: unknown, n: number): Float64Array =>
    new Float64Array(koffi.view(p, n * 8));
  const realProto = koffi.proto(
    "int jit_host_matmul_cb(void *a, long m, long k, void *b, long n, void *out)"
  );
  const complexProto = koffi.proto(
    "int jit_host_matmul_complex_cb(void *ar, void *ai, long m, long k, " +
      "void *br, void *bi, long n, void *out_re, void *out_im)"
  );
  const real = koffi.register(
    (
      a: unknown,
      m: number | bigint,
      k: number | bigint,
      b: unknown,
      n: number | bigint,
      out: unknown
    ): number => {
      try {
        const [M, K, N] = [Number(m), Number(k), Number(n)];
        const r = getLapackBridge().matmul(
          f64(a, M * K),
          M,
          K,
          f64(b, K * N),
          N
        );
        f64(out, M * N).set(r);
        return 0;
      } catch {
        return 1;
      }
    },
    koffi.pointer(realProto)
  );
  const complex = koffi.register(
    (
      ar: unknown,
      ai: unknown,
      m: number | bigint,
      k: number | bigint,
      br: unknown,
      bi: unknown,
      n: number | bigint,
      outRe: unknown,
      outIm: unknown
    ): number => {
      try {
        const [M, K, N] = [Number(m), Number(k), Number(n)];
        const r = getLapackBridge().matmulComplex(
          f64(ar, M * K),
          ai === null ? new Float64Array(M * K) : f64(ai, M * K),
          M,
          K,
          f64(br, K * N),
          bi === null ? new Float64Array(K * N) : f64(bi, K * N),
          N
        );
        f64(outRe, M * N).set(r.re);
        const im = f64(outIm, M * N);
        if (r.im) im.set(r.im);
        else im.fill(0);
        return 0;
      } catch {
        return 1;
      }
    },
    koffi.pointer(complexProto)
  );
  hostNumericCbs = [real, complex];
  return hostNumericCbs;
}

/** Bind the provider-stack matmul into an emitted `.so` (see
 *  `runtime/tensor_ops/host_numeric.h`), so its large products match the
 *  interpreter. Done once per loaded `.so`; a no-op when the spec has no
 *  matrix product. */
export function bindHostNumeric(koffi: KoffiAny, lib: KoffiLib): void {
  let set: (re: unknown, cx: unknown, inlineMax: number) => void;
  try {
    set = lib.func("void jit_set_host_numeric(void *, void *, double)");
  } catch {
    return;
  }
  const [real, complex] = hostNumericCallbacks(koffi);
  set(real, complex, MATMUL_INLINE_MAX_WORK);
}

/** The raise status a call left behind (see `runtime/system/raise.h`). */
export interface CRaiseStatus {
  /** 1 = bail, 2 = error, 3 = cancel. */
  readonly kind: number;
  readonly id: string;
  /** null: the JIT leaves the error's wording to the interpreter. */
  readonly msg: string | null;
  /** Statement location code (`locCode`), negative when unknown. */
  readonly loc: number;
}

/** Bind the emitted `.so`'s raise-status accessors. Every `.so` exports
 *  them: the host-entry wrapper clears the status before each call, and a
 *  `jit_raise` sets it before `longjmp`ing back to the wrapper's guard.
 *  `status()` is null after a call that returned normally. */
export function bindRaise(lib: KoffiLib): { status(): CRaiseStatus | null } {
  const kind = lib.func("int jit_raise_kind(void)") as () => number;
  const id = lib.func("const char *jit_raise_id(void)") as () => string;
  const msg = lib.func("const char *jit_raise_msg(void)") as () =>
    | string
    | null;
  const loc = lib.func("long long jit_raise_loc(void)") as () =>
    | number
    | bigint;
  return {
    status() {
      const k = kind();
      if (k === 0) return null;
      return { kind: k, id: id() ?? "", msg: msg(), loc: Number(loc()) };
    },
  };
}

/** The cancel flag installed in each loaded `.so`, by path (one `dlopen`
 *  handle, and so one `jit__cancel`, per path in a process). */
const installedCancel = new Map<string, Int32Array | null>();

/** Bind the emitted `.so`'s `jit_set_cancel` (see `runtime/system/raise.h`).
 *  `install(flag)` points the compiled cancel polls at `flag`, a view of a
 *  SharedArrayBuffer word (koffi passes the backing store, which never
 *  moves, so the pointer stays valid while the runtime holds the view), or
 *  at a constant zero for null. It calls into C only when the flag
 *  changes. */
export function bindCancel(
  lib: KoffiLib,
  libPath: string
): { install(flag: Int32Array | null): void } {
  const set = lib.func("void jit_set_cancel(int32_t *)") as (
    flag: Int32Array | null
  ) => void;
  return {
    install(flag) {
      if (installedCancel.get(libPath) === flag) return;
      set(flag);
      installedCancel.set(libPath, flag);
    },
  };
}

/** Build a marshaling context from a koffi lib handle. `jit_malloc`
 *  / `jit_free` are exported by every JIT `.so` (see
 *  `jit/codegen/emit.ts`), so host-side buffers use the same allocator
 *  the emitted code frees with. */
export function makeCMarshalCtx(koffi: Koffi, lib: KoffiLib): MarshalCtx {
  const malloc = lib.func("void *jit_malloc(long)") as MallocFn;
  const free = lib.func("void jit_free(void *)") as FreeFn;
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
    // Scalar — the JIT's C ABI passes every single-element real numeric
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

/** Copy `src` into a fresh `jit_malloc` buffer (or zero-fill
 *  `n` doubles when `src` is undefined). */
function copyIn(
  ctx: MarshalCtx,
  src: Float64Array | undefined,
  n: number,
  owned: unknown[]
): unknown {
  const ptr = ctx.malloc(Math.max(n, 1) * 8);
  if (ptr === null) throw new Error("jit_malloc: out of memory");
  owned.push(ptr);
  if (n > 0) {
    ctx.koffi.encode(ptr, "double", src ?? new Float64Array(n), n);
  }
  return ptr;
}

/** Build an `jit_tensor_t` whose `real`/`imag` point to fresh buffers
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
  while (shape.length < 2) shape.push(1); // minimum ndim is 2.
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
    ty.isComplex && imagPtr
      ? narrowedImag(copyOut(ctx, imagPtr, n))
      : undefined;
  if (realPtr) ctx.free(realPtr);
  if (imagPtr) ctx.free(imagPtr);
  return new RuntimeTensor(
    realData,
    shape,
    imagData,
    ty.elem === "logical" ? "logical" : "double"
  );
}

/** Convert a bare scalar return value to a numbl RuntimeValue.
 *
 *  Real / logical scalars ride the `double` ABI, so koffi hands us a
 *  JS number (logical coerces to `Boolean`). Complex scalars ride the
 *  `jit_cscalar_t` struct, so koffi hands us a `{re, im}` object,
 *  which becomes a `RuntimeComplexNumber` (or a real number when its
 *  imaginary part is zero, see `narrowedImag`). */
export function unmarshalScalarOutput(
  v: unknown,
  ty: NumericType
): RuntimeValue {
  if (ty.isComplex) {
    const c = v as { re: number; im: number };
    return complexScalarOut(c.re, c.im);
  }
  if (ty.elem === "logical") return Boolean(v);
  return v as number;
}
