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
 *   2 — set1r_h (scalar linear Index write with soft-bail on OOB).
 *   3 — idx2r / idx3r / set2r_h / set3r_h (multi-index Index read/write).
 *   4 — setRange1r_h / setCol2r_h / copyRange1r (range/col slice r/w).
 */
#define NUMBL_JIT_RT_VERSION 4

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

/**
 * 1-based MATLAB linear Index write on a real-valued tensor buffer.
 *
 * On OOB, writes `2.0` to *err_flag (the "growth-needed" code the JS
 * wrapper translates to a JitBailToInterpreter) and returns without
 * writing, mirroring the JS-JIT's `set1r_h` helper. The interpreter
 * then re-runs the call with proper tensor-growth semantics. As with
 * the read path, the caller must zero the flag before each call.
 */
void numbl_set1r_h(double* data, size_t len, double i, double v, double* err_flag);

/* ── Multi-index Index read (2D, 3D) ──────────────────────────────────── */

/**
 * 1-based MATLAB 2D Index read on a real-valued tensor buffer.
 *
 * Column-major: returns data[(j-1)*d0 + (i-1)]. `d0` is the row count.
 * The derived column count is `len / d0`; both dimensions are bounds-
 * checked independently. Emitter wraps non-integer indices with
 * `round()` to match the JS-JIT's `Math.round`-then-truncate sequence.
 * On OOB, sets *err_flag = 1.0 and returns 0.0 (hard bounds error).
 */
double numbl_idx2r(const double* data, size_t len, size_t d0,
                   double i, double j, double* err_flag);

/**
 * 1-based MATLAB 3D Index read on a real-valued tensor buffer.
 *
 * Column-major: returns data[((k-1)*d1 + (j-1))*d0 + (i-1)]. `d0` is
 * the row count, `d1` the column count. The derived page count is
 * `len / (d0 * d1)`; each dimension is bounds-checked independently.
 */
double numbl_idx3r(const double* data, size_t len, size_t d0, size_t d1,
                   double i, double j, double k, double* err_flag);

/* ── Multi-index Index write (2D, 3D), soft-bail on OOB ───────────────── */

/**
 * 2D Index write. On OOB (along either dim) writes `2.0` to *err_flag
 * (the "growth-needed" code the JS wrapper translates to a
 * JitBailToInterpreter) and returns without writing.
 */
void numbl_set2r_h(double* data, size_t len, size_t d0,
                   double i, double j, double v, double* err_flag);

/** 3D Index write with the same soft-bail convention as set2r_h. */
void numbl_set3r_h(double* data, size_t len, size_t d0, size_t d1,
                   double i, double j, double k, double v,
                   double* err_flag);

/* ── Range / column slice read and write ──────────────────────────────── */

/**
 * Range → range write: `dst(dStart:dEnd) = src(sStart:sEnd)` on real
 * tensors, 1-based MATLAB indices. Sizes must match: on mismatch,
 * writes `3.0` to *err_flag and returns (the JS wrapper translates this
 * into "Unable to perform assignment because the indices on the left
 * side are not compatible with the size of the right side."). Bounds
 * violations on either range write `1.0` and return. Uses memmove so
 * overlapping dst == src ranges are handled correctly.
 */
void numbl_setRange1r_h(double* dstData, size_t dstLen,
                        double dstStart, double dstEnd,
                        const double* srcData, size_t srcLen,
                        double srcStart, double srcEnd,
                        double* err_flag);

/**
 * Column-vector write into column `col` of a 2-D tensor `dst` with row
 * count `dstRows`. `srcLen` must equal `dstRows` (length mismatch →
 * *err_flag = 3.0). When `col` exceeds the current column count, the
 * MATLAB interpreter would grow dst — we soft-bail (*err_flag = 2.0) so
 * the caller's stale hoisted aliases don't corrupt the growth path.
 */
void numbl_setCol2r_h(double* dstData, size_t dstRows, size_t dstLen,
                      double col,
                      const double* srcData, size_t srcLen,
                      double* err_flag);

/**
 * Range read: copy `src(start:end)` into caller-provided `dstData`. The
 * caller must ensure `dstData` has capacity >= `end - start + 1`. OOB
 * on `src` writes `1.0` to *err_flag and returns without writing.
 * `start > end` is a valid empty range (no write, no error).
 */
void numbl_copyRange1r(const double* srcData, size_t srcLen,
                       double start, double end,
                       double* dstData,
                       double* err_flag);

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
