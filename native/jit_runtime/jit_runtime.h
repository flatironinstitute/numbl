/**
 * numbl_jit_runtime — helpers the C-JIT emitter calls from generated code.
 *
 * Kept in a static archive (jit_runtime.a) separate from numbl_ops.a:
 * the `ops/` directory is reserved for op-code-dispatched kernels that
 * achieve parity with the TS ops layer. Anything the JIT emitter calls
 * *directly* from emitted C (bounds-checked index reads, MATLAB-semantics
 * math helpers, tic/toc, reduction shims, ...) lives here instead.
 *
 * When adding a new helper, bump NUMBL_JIT_RT_VERSION below AND
 * NUMBL_JIT_RT_REQUIRED_VERSION in jitCodegenC.ts. The generated C asserts
 * `NUMBL_JIT_RT_VERSION >= N`, so a stale archive fails the per-JIT
 * compile step with a clear "rebuild the addon" message instead of a
 * cryptic linker error.
 */

#ifndef NUMBL_JIT_RUNTIME_H
#define NUMBL_JIT_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Version ──────────────────────────────────────────────────────────── */

/**
 * Version log:
 *   1 — initial: idx1r, mod, sign, reduce_flat, tic/toc/monotonic_time.
 */
#define NUMBL_JIT_RT_VERSION 1

/** Returns NUMBL_JIT_RT_VERSION baked into the compiled archive. */
int numbl_jit_rt_version(void);

/* ── Scalar linear Index read ─────────────────────────────────────────── */

/**
 * 1-based MATLAB linear Index read on a real-valued tensor buffer.
 *
 *   `i` — 1-based double index (truncation-to-zero via int64 cast). The
 *         emitter wraps non-integer indices with `round()` beforehand to
 *         match the JS-JIT's `Math.round`-then-truncate sequence.
 *   `err_flag` — set to 1.0 on OOB (and returns 0.0) so the C function
 *                finishes without a native crash. The caller must zero
 *                the flag before each koffi call; the JS wrapper checks
 *                it after and throws "Index exceeds array bounds".
 */
double numbl_idx1r(const double* data, size_t len, double i, double* err_flag);

/* ── Scalar math helpers with MATLAB semantics ────────────────────────── */

/** MATLAB `mod(a, b)`: result has the sign of `b`; `mod(a, 0) == a`. */
double numbl_mod(double a, double b);

/** Three-valued sign: -1, 0, or 1. */
double numbl_sign(double x);

/* ── Reduction wrapper ────────────────────────────────────────────────── */

/**
 * Thin wrapper around numbl_real_flat_reduce (from numbl_ops.a) that
 * returns the scalar directly, so generated C doesn't need a local
 * `double out; … ; return out;` dance inline.
 */
double numbl_reduce_flat(int op, const double* data, int64_t len);

/* ── Timers (tic/toc) ─────────────────────────────────────────────────── */

/**
 * Monotonic wall-clock seconds. Exported by every JIT .so so the JS
 * wrapper can cross-reference the C clock domain when bridging tic/toc
 * state between JS and C.
 */
double numbl_monotonic_time(void);

/** Capture the current monotonic time into *state, return it. */
double numbl_tic(double* state);

/** Elapsed seconds since *state. */
double numbl_toc(const double* state);

#ifdef __cplusplus
}
#endif

#endif /* NUMBL_JIT_RUNTIME_H */
