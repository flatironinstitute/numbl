# C-JIT: migrate from N-API to koffi

## Problem

The C-JIT currently compiles each specialization to a `.node` native addon,
loaded via `createRequire`. Each `.node` registers with Node's module system
via `NAPI_MODULE_INIT`. This causes two issues:

1. **Exit hangs.** Node's module teardown blocks on libuv thread-pool
   cleanup. With 5+ loaded `.node` modules the process hangs on exit
   ~20-30% of the time — `process.exit(0)` doesn't return promptly.

2. **Handle-scope crashes.** `napi_value` handles are only valid within
   their enclosing handle scope (one N-API callback). Persisting them
   across invocations via `static __thread` causes use-after-free
   segfaults. We patched this by making scratch slots per-invocation,
   but the architecture is fragile — any future optimization that tries
   to persist a JS reference across calls will hit the same wall.

3. **Bloated generated C.** Every `.node` embeds a 500-line tensor-helper
   preamble that marshals `napi_value` → `double*` → libnumbl_ops call →
   `napi_value`. This is the largest part of each generated file.

## Solution: koffi

[koffi](https://koffi.dev) is a pure-FFI library. It loads `.so` files
via `dlopen`/`dlsym` directly — no Node module registration, no handle
scopes, no `NAPI_MODULE_INIT`. The C functions work with raw C types
(`double*`, `int64_t`, etc.) and koffi handles the JS↔C marshalling.

Key capability: koffi passes `Float64Array` directly as `double*` to C
functions. No boxing, no N-API typed-array extraction.

## Architecture after migration

### JS side (cJitInstall.ts)

Before calling the C function, the JS wrapper extracts raw data from
RuntimeTensor objects:

```js
// For each tensor param:
const x_data = x.data;         // Float64Array — koffi passes as double*
const x_len = x.data.length;   // number → int64_t

// Call the raw C function via koffi:
const result = jitFn(x_data, x_len, y_data, y_len, ...);

// For tensor return: result is { data: Float64Array, len: int }
// Wrap back into RuntimeTensor:
return RTV.tensor(result_data, shape);
```

Buffer reuse (the `dest` / `_rc === 1` pattern) moves entirely to JS:

```js
// JS-side buffer reuse, same logic as jitHelpersTensor.ts:
let dest_data = (dest && dest._rc === 1 && dest.data.length === n)
  ? dest.data : new Float64Array(n);
// Pass dest_data to C; C writes into it directly.
jitFn(dest_data, x_data, x_len, ...);
// Wrap: if dest_data === dest.data, reuse the wrapper object.
```

### C side (generated code)

Pure C with no N-API. Functions take raw pointers and lengths:

```c
#include <math.h>
#include "numbl_ops.h"

void jit_loop_for(
    double *v_r_data, int64_t v_r_len,
    const double *v_x_data, int64_t v_x_len,
    const double *v_y_data, int64_t v_y_len,
    double v_trials,
    double *out_k)
{
    double v_k = 0.0;
    for (double t = 1.0; t <= v_trials; t += 1.0) {
        v_k = t;
        // r = x + y  — write directly into v_r_data
        numbl_real_binary_elemwise(NUMBL_REAL_BIN_ADD,
            (size_t)v_r_len, v_x_data, v_y_data, v_r_data);
        // r = r .* x
        numbl_real_binary_elemwise(NUMBL_REAL_BIN_MUL,
            (size_t)v_r_len, v_r_data, v_x_data, v_r_data);
    }
    *out_k = v_k;
}
```

No `napi_env`, no `napi_value`, no helper preamble. Each op is a direct
`numbl_real_*` call. Scalar outputs use out-pointers. Tensor I/O is
caller-allocated `double*` buffers.

### Loading (.so instead of .node)

```js
import koffi from "koffi";

const lib = koffi.load("/path/to/jit_loop.so");
const jitFn = lib.func(
  "void jit_loop_for(double*, int64_t, double*, int64_t, double*, int64_t, double, _Out_ double*)"
);
```

No `NAPI_MODULE_INIT`. No module registration. No exit hooks.

### Compile step

Same `cc -O2 -shared -fPIC ...` invocation, but:

- Output is `.so` not `.node` (just a rename; same ELF format).
- No N-API headers needed (`-I<node-gyp-headers>` dropped).
- No shim file — the user function IS the export.
- Link flags: just `-L... -l:numbl_ops.a -lm`.

The `node-gyp install` pre-install step is no longer needed.

## Files to change

| File              | Change                                                                                                                                                                                                                                                                                                                                                                                                |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cCompile.ts`     | Replace `createRequire` load with `koffi.load()`. Drop node-gyp header discovery. Output `.so` not `.node`. Signature declaration via koffi type strings.                                                                                                                                                                                                                                             |
| `cJitHelpers.ts`  | **Delete entirely.** The 500-line N-API helper block is no longer needed. Buffer allocation/reuse moves to JS.                                                                                                                                                                                                                                                                                        |
| `cNapiShim.ts`    | **Delete entirely.** koffi IS the shim.                                                                                                                                                                                                                                                                                                                                                               |
| `jitCodegenC.ts`  | Major simplification. Tensor params become `(double *data, int64_t len)` pairs. Tensor ops become direct `numbl_real_*` calls with pointer args. No `napi_env`, no `numbl_jit_t*` helpers. Scratch slots become plain `double*` locals (malloc/free within the function, or caller-allocated). Multi-output: scalar out-pointers (`double *out_k`). Tensor output: caller-allocated buffer passed in. |
| `cFeasibility.ts` | Mostly unchanged. Drop the `needsEnv` concept.                                                                                                                                                                                                                                                                                                                                                        |
| `cJitInstall.ts`  | Major rewrite. JS-side tensor marshalling: extract `.data` + `.length` from RuntimeTensor, pass to koffi. Buffer reuse logic (currently in cJitHelpers.ts C code) moves here to JS. Multi-output: read scalar out-pointers after the call. Tensor return: wrap the pre-allocated Float64Array back into RuntimeTensor.                                                                                |
| `cJitBackend.ts`  | Interface unchanged (still `tryCompile → callable`).                                                                                                                                                                                                                                                                                                                                                  |
| `package.json`    | Add `koffi` dependency.                                                                                                                                                                                                                                                                                                                                                                               |

## Buffer management strategy

**Key design decision:** who allocates tensor output buffers?

Option A — **JS allocates, C writes in-place.** JS creates or reuses a
`Float64Array`, passes its backing pointer to C. C writes results
directly into it. No C-side malloc. This is simplest and matches how
the JS-JIT works today (`reuseRealBuffer` / `uninitFloat64`).

Option B — C allocates via `malloc`, JS wraps via `koffi.decode`. Adds
complexity (who frees?) and prevents buffer reuse.

**Recommendation: Option A.** It mirrors the JS-JIT exactly and avoids
all lifetime issues.

For intermediate tensors within a C function (e.g., `r = x + y; r = r .* x`
where the first `r` is consumed by the second op): allocate scratch
buffers on the C side via `alloca` or stack arrays for small sizes, or
`malloc`/`free` pairs scoped to the function body. These never escape
to JS.

## Multi-output return strategy

Scalar outputs: `double *out_k` out-pointer, read by JS after the call.

Tensor outputs: caller-allocated `Float64Array` passed as `double*`.
JS pre-allocates based on input shapes (element-wise ops preserve shape).

Multi-output example:

```c
void jit_loop(const double *x, int64_t x_len,
              double *out_r, /* tensor output, pre-allocated */
              double *out_k  /* scalar output */) { ... }
```

JS side:

```js
const out_r = reuseOrAlloc(dest_r, n); // Float64Array
const out_k = new Float64Array(1); // scalar via 1-element array
jitFn(x.data, x.data.length, out_r, out_k);
const k = out_k[0];
const r = wrapTensor(out_r, shape);
```

## Migration plan

1. **Add koffi dependency.** `npm install koffi`.
2. **Prototype on one scalar function.** Change `cCompile.ts` to output
   `.so` and load via koffi. Get a scalar `f(a, b) = a*a + b` working
   end-to-end without N-API.
3. **Extend to tensor functions.** Change codegen to emit `double*` params.
   Move buffer reuse to JS side in `cJitInstall.ts`.
4. **Delete `cJitHelpers.ts` and `cNapiShim.ts`.**
5. **Multi-output.** Out-pointers for scalars, pre-allocated buffers for
   tensors.
6. **Loop C-JIT.** Same as function C-JIT — same codegen, same loading.
7. **Verify.** All existing tests + bench + stress test 100x.
8. **Remove node-gyp header dependency** from `getCEnv()` and CI.

## What stays the same

- Cache directory (`~/.cache/numbl/c-jit/`).
- Content-addressed hash (source + compiler + platform + libnumbl_ops).
- `libnumbl_ops.a` static archive (linked into each `.so`).
- `cFeasibility.ts` whitelist approach.
- `binding.gyp` target for building `numbl_ops.a`.
- The overall `tryCompile → callable` backend interface.

## Risks

- **koffi as a dependency.** It's a native addon itself (prebuilt
  binaries for major platforms). If it fails to install on some
  platform, C-JIT is unavailable (same as today when `cc` is missing).
- **FFI overhead per call.** koffi is fast but adds ~50-100ns per call
  vs direct N-API. For tensor ops on N=2M elements this is negligible
  (<0.01% of kernel time). For scalar-only hot loops it could matter —
  but those are already fast in the JS-JIT path.
- **Intermediate buffer allocation.** Moving from "C helpers allocate
  JS ArrayBuffers" to "JS pre-allocates, C writes" changes who owns
  the intermediate. For chained ops (`r = x+y; r = r.*x; r = exp(r)`)
  the JS wrapper needs to manage a small pool of scratch buffers. This
  is ~20 lines of JS, same pattern as `jitHelpersTensor.ts` today.
