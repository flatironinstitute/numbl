/**
 * Per-module C helper boilerplate prepended to every C-JIT'd function that
 * touches tensors.
 *
 * Mirrors the JS-side helpers in [jitHelpersTensor.ts](../jitHelpersTensor.ts)
 * so the C-JIT and JS-JIT make identical decisions for every tensor op:
 *
 *   - Read tensor inputs via N-API (`obj.data` Float64Array, `obj.shape`,
 *     `obj._rc`) — equivalent of `isTensor(v) && v.data instanceof Float64Array`.
 *   - Reuse `dest`'s buffer if it's a unique-rc real tensor of matching length
 *     (the equivalent of `reuseRealBuffer(dest, n)`).
 *   - Otherwise allocate a fresh `Float64Array` (the equivalent of
 *     `uninitFloat64(n)`).
 *   - Call into `libnumbl_ops` directly (the same C code the N-API addon
 *     wraps for JS-JIT).
 *   - Wrap the result in a `RuntimeTensor` (the equivalent of
 *     `finalizeReal(out, shape, dest, isLogical)`), reusing the existing
 *     wrapper object when buffer reuse hit.
 *
 * On any unsupported input (complex tensor, mismatched shapes that JS would
 * broadcast, etc.) the helper throws the bail-sentinel JS error; the JIT
 * wrapper catches it and converts to `JitBailToInterpreter` so the
 * interpreter retries the call through the slow path.
 *
 * The block is *embedded* into each generated `.c` file (rather than
 * lived in a separately-linked .o) so each cached `.node` is fully
 * self-contained and the cache hash captures any helper source changes
 * via the existing source-hash key. Per-call cost is one `static`
 * function call, no DSO crossing.
 */

export const C_JIT_BAIL_SENTINEL = "C-JIT bail to interpreter";

/**
 * Returns the C source block to prepend to every tensor-touching .node
 * source file. Includes <node_api.h>, <math.h>, <stdint.h>, "numbl_ops.h",
 * the bail sentinel, the tensor-read / buffer-reuse / wrapper-finalize
 * helpers, and one small helper per supported tensor op. The op helpers
 * mirror `tAdd / tSub / tMul / tDiv / tEq / tNeq / tLt / tLe / tGt / tGe /
 * tNeg / tUnary / tSum / tProd / tMax / tMin / tMean` from the JS side.
 *
 * Helpers are emitted unconditionally — every tensor-touching .node gets
 * the full set. Their static-keyword scoping means unused ones are dead-
 * stripped by the C compiler, so binary size stays small. This avoids
 * having to thread "which helpers are needed" through the codegen and
 * matches the JS-side approach where `$h.*` exports are also "always
 * present, take what you need."
 */
export function cJitTensorHelpersSource(): string {
  return HELPER_SOURCE;
}

const HELPER_SOURCE = String.raw`
#include <node_api.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "numbl_ops.h"

#define NUMBL_JIT_BAIL_SENTINEL "${C_JIT_BAIL_SENTINEL}"

/* ── Bail-out helper ──────────────────────────────────────────────────────
 * Throws a JS Error whose message is the bail sentinel. The JIT wrapper
 * (cJitInstall.ts) catches that exact message and converts to
 * JitBailToInterpreter so the interpreter retries through the slow path.
 * Returns NULL so callers can return numbl_jit_bail(env) on errors.
 */
static napi_value numbl_jit_bail(napi_env env) {
  napi_throw_error(env, NULL, NUMBL_JIT_BAIL_SENTINEL);
  return NULL;
}

/* ── Tensor-read helpers ──────────────────────────────────────────────────
 * Inspect a JS RuntimeTensor (shape: { kind: tensor, data: Float64Array,
 * shape: number[], _rc: number, _isLogical?: boolean }) and pull out:
 *
 *   data      — pointer to the Float64Array contents
 *   n         — element count (data.length)
 *   shape_val — the .shape JS value (kept around so we can clone it for
 *               the result without re-reading)
 *
 * Returns 1 on success, 0 if the value isn't a real Float64-backed
 * RuntimeTensor (caller bails). This is the N-API equivalent of
 *   isTensor(v) && v.data instanceof Float64Array && !v.imag
 */
typedef struct {
  napi_value obj;            /* the original JS object */
  double *data;              /* Float64Array contents */
  size_t n;                  /* data.length */
  napi_value shape_val;      /* obj.shape */
  int rc;                    /* obj._rc */
  int is_complex;            /* 1 if obj.imag is present */
} numbl_jit_tensor_view_t;

static int numbl_jit_read_tensor(napi_env env, napi_value v,
                                 numbl_jit_tensor_view_t *out) {
  napi_status status;
  napi_valuetype t;
  status = napi_typeof(env, v, &t);
  if (status != napi_ok || t != napi_object) return 0;

  /* obj.data must be a Float64Array */
  napi_value data_val;
  status = napi_get_named_property(env, v, "data", &data_val);
  if (status != napi_ok) return 0;
  bool is_typed;
  status = napi_is_typedarray(env, data_val, &is_typed);
  if (status != napi_ok || !is_typed) return 0;
  napi_typedarray_type tt;
  size_t n_elems;
  void *raw;
  status = napi_get_typedarray_info(env, data_val, &tt, &n_elems, &raw, NULL, NULL);
  if (status != napi_ok || tt != napi_float64_array) return 0;

  /* obj.imag (optional, but presence ⇒ complex ⇒ caller bails) */
  napi_value imag_val;
  status = napi_get_named_property(env, v, "imag", &imag_val);
  if (status != napi_ok) return 0;
  napi_valuetype imag_t;
  status = napi_typeof(env, imag_val, &imag_t);
  if (status != napi_ok) return 0;
  int is_complex = (imag_t != napi_undefined && imag_t != napi_null) ? 1 : 0;

  /* obj.shape — array of numbers, kept as napi_value for cloning later */
  napi_value shape_val;
  status = napi_get_named_property(env, v, "shape", &shape_val);
  if (status != napi_ok) return 0;

  /* obj._rc — number; missing ⇒ treat as shared (rc>1) so we never
   * accidentally clobber a buffer. */
  napi_value rc_val;
  status = napi_get_named_property(env, v, "_rc", &rc_val);
  if (status != napi_ok) return 0;
  napi_valuetype rc_t;
  status = napi_typeof(env, rc_val, &rc_t);
  if (status != napi_ok) return 0;
  int rc = 2;  /* default: shared */
  if (rc_t == napi_number) {
    int32_t rci = 0;
    status = napi_get_value_int32(env, rc_val, &rci);
    if (status == napi_ok) rc = rci;
  }

  out->obj = v;
  out->data = (double *)raw;
  out->n = n_elems;
  out->shape_val = shape_val;
  out->rc = rc;
  out->is_complex = is_complex;
  return 1;
}

/* Box a C double as a JS number (napi_value). Used by the codegen to
 * feed scalar sub-expressions into tensor helpers that expect napi_value.
 * On allocation failure, throws via bail and returns NULL. */
static napi_value numbl_jit_box_double(napi_env env, double v) {
  napi_value out;
  if (napi_create_double(env, v, &out) != napi_ok) return numbl_jit_bail(env);
  return out;
}

/* Read a JS scalar (number or boolean) → double. Returns 1 on success. */
static int numbl_jit_read_scalar(napi_env env, napi_value v, double *out) {
  napi_status status;
  napi_value n;
  status = napi_coerce_to_number(env, v, &n);
  if (status != napi_ok) return 0;
  return napi_get_value_double(env, n, out) == napi_ok;
}

/* ── Output buffer (re)allocation ─────────────────────────────────────────
 * If dest is a unique-rc real-Float64 RuntimeTensor of length n, return
 * its data pointer (reuse). Else return NULL — caller allocates fresh.
 *
 * Mirrors jitHelpersTensor.ts:reuseRealBuffer.
 */
static double *numbl_jit_reuse_real_buffer(napi_env env,
                                           napi_value dest, size_t n) {
  if (!dest) return NULL;
  napi_status status;
  napi_valuetype t;
  status = napi_typeof(env, dest, &t);
  if (status != napi_ok || t != napi_object) return NULL;
  numbl_jit_tensor_view_t v;
  if (!numbl_jit_read_tensor(env, dest, &v)) return NULL;
  if (v.is_complex) return NULL;
  if (v.rc != 1) return NULL;
  if (v.n != n) return NULL;
  return v.data;
}

/* Allocate a fresh Float64Array of length n.
 *
 * Returns the underlying double pointer plus the JS typed-array value
 * (out_typed) and the underlying ArrayBuffer (unused but returned for
 * future zero-copy plays).
 *
 * On allocation failure, throws via napi and returns NULL.
 */
static double *numbl_jit_alloc_real_buffer(napi_env env, size_t n,
                                           napi_value *out_typed) {
  napi_status status;
  napi_value ab;
  void *ab_data;
  status = napi_create_arraybuffer(env, n * sizeof(double), &ab_data, &ab);
  if (status != napi_ok) { numbl_jit_bail(env); return NULL; }
  napi_value typed;
  status = napi_create_typedarray(env, napi_float64_array, n, ab, 0, &typed);
  if (status != napi_ok) { numbl_jit_bail(env); return NULL; }
  *out_typed = typed;
  return (double *)ab_data;
}

/* Build a fresh RuntimeTensor wrapper around (typed, shape).
 * Mirrors makeTensor + the isLogical decoration.
 */
static napi_value numbl_jit_make_tensor(napi_env env, napi_value typed,
                                        napi_value shape_val, int is_logical) {
  napi_status status;
  napi_value obj;
  status = napi_create_object(env, &obj);
  if (status != napi_ok) return numbl_jit_bail(env);
  napi_value kind_str;
  status = napi_create_string_utf8(env, "tensor", NAPI_AUTO_LENGTH, &kind_str);
  if (status != napi_ok) return numbl_jit_bail(env);
  napi_set_named_property(env, obj, "kind", kind_str);
  napi_set_named_property(env, obj, "data", typed);
  napi_set_named_property(env, obj, "shape", shape_val);
  napi_value rc_one;
  status = napi_create_int32(env, 1, &rc_one);
  if (status != napi_ok) return numbl_jit_bail(env);
  napi_set_named_property(env, obj, "_rc", rc_one);
  if (is_logical) {
    napi_value tru;
    napi_get_boolean(env, true, &tru);
    napi_set_named_property(env, obj, "_isLogical", tru);
  }
  return obj;
}

/* finalizeReal: if dest is the same RuntimeTensor whose buffer was
 * reused (out == dest.data), mutate dest's shape/imag/_isLogical and
 * return dest. Else build a fresh wrapper around out_typed.
 *
 * To detect buffer reuse without re-reading the typed array, callers pass
 * reused_dest=1 if they took the reuse path. This mirrors the JS-side
 * pointer-equality check (dest as RuntimeTensor).data === out.
 */
static napi_value numbl_jit_finalize_real(napi_env env,
                                          napi_value dest,
                                          napi_value out_typed,
                                          napi_value shape_val,
                                          int reused_dest,
                                          int is_logical) {
  if (reused_dest) {
    /* Mutate dest.shape, dest.imag = undefined, dest._isLogical. */
    napi_set_named_property(env, dest, "shape", shape_val);
    napi_value undef;
    napi_get_undefined(env, &undef);
    napi_set_named_property(env, dest, "imag", undef);
    if (is_logical) {
      napi_value tru;
      napi_get_boolean(env, true, &tru);
      napi_set_named_property(env, dest, "_isLogical", tru);
    } else {
      napi_set_named_property(env, dest, "_isLogical", undef);
    }
    return dest;
  }
  return numbl_jit_make_tensor(env, out_typed, shape_val, is_logical);
}

/* ── Output-buffer setup combinator ───────────────────────────────────────
 * Common pattern: try reuse, else alloc. Sets *reused_out to 1 if reuse
 * hit. On alloc failure, returns NULL (and bail is already thrown).
 */
static double *numbl_jit_get_out_buffer(napi_env env, napi_value dest, size_t n,
                                        napi_value *out_typed,
                                        int *reused_out) {
  double *reuse = numbl_jit_reuse_real_buffer(env, dest, n);
  if (reuse) {
    *reused_out = 1;
    return reuse;
  }
  *reused_out = 0;
  return numbl_jit_alloc_real_buffer(env, n, out_typed);
}

/* ── Tensor binary ops (real) ─────────────────────────────────────────────
 * Each helper mirrors jitHelpersTensor.ts:fastBinaryOp for one opcode.
 * Three fast paths: tensor-tensor, tensor-scalar, scalar-tensor. Anything
 * else (complex, mismatched length) bails to the interpreter.
 */
static napi_value numbl_jit_t_binary_real(napi_env env, napi_value dest,
                                          napi_value a, napi_value b,
                                          int op) {
  numbl_jit_tensor_view_t av, bv;
  int aIsT = numbl_jit_read_tensor(env, a, &av);
  int bIsT = numbl_jit_read_tensor(env, b, &bv);

  /* tensor – tensor */
  if (aIsT && bIsT) {
    if (av.is_complex || bv.is_complex) return numbl_jit_bail(env);
    if (av.n != bv.n) return numbl_jit_bail(env);
    napi_value out_typed = NULL;
    int reused = 0;
    double *out = numbl_jit_get_out_buffer(env, dest, av.n, &out_typed, &reused);
    if (!out) return NULL;
    numbl_real_binary_elemwise(op, av.n, av.data, bv.data, out);
    return numbl_jit_finalize_real(env, dest,
                                   reused ? (napi_value)NULL : out_typed,
                                   av.shape_val, reused, 0);
  }
  /* tensor – scalar */
  if (aIsT) {
    if (av.is_complex) return numbl_jit_bail(env);
    double bs;
    if (!numbl_jit_read_scalar(env, b, &bs)) return numbl_jit_bail(env);
    napi_value out_typed = NULL;
    int reused = 0;
    double *out = numbl_jit_get_out_buffer(env, dest, av.n, &out_typed, &reused);
    if (!out) return NULL;
    numbl_real_scalar_binary_elemwise(op, av.n, bs, av.data,
                                      /*scalar_on_left=*/0, out);
    return numbl_jit_finalize_real(env, dest,
                                   reused ? (napi_value)NULL : out_typed,
                                   av.shape_val, reused, 0);
  }
  /* scalar – tensor */
  if (bIsT) {
    if (bv.is_complex) return numbl_jit_bail(env);
    double as;
    if (!numbl_jit_read_scalar(env, a, &as)) return numbl_jit_bail(env);
    napi_value out_typed = NULL;
    int reused = 0;
    double *out = numbl_jit_get_out_buffer(env, dest, bv.n, &out_typed, &reused);
    if (!out) return NULL;
    numbl_real_scalar_binary_elemwise(op, bv.n, as, bv.data,
                                      /*scalar_on_left=*/1, out);
    return numbl_jit_finalize_real(env, dest,
                                   reused ? (napi_value)NULL : out_typed,
                                   bv.shape_val, reused, 0);
  }
  return numbl_jit_bail(env);
}

static napi_value numbl_jit_tAdd(napi_env env, napi_value d, napi_value a, napi_value b) {
  return numbl_jit_t_binary_real(env, d, a, b, NUMBL_REAL_BIN_ADD);
}
static napi_value numbl_jit_tSub(napi_env env, napi_value d, napi_value a, napi_value b) {
  return numbl_jit_t_binary_real(env, d, a, b, NUMBL_REAL_BIN_SUB);
}
static napi_value numbl_jit_tMul(napi_env env, napi_value d, napi_value a, napi_value b) {
  return numbl_jit_t_binary_real(env, d, a, b, NUMBL_REAL_BIN_MUL);
}
static napi_value numbl_jit_tDiv(napi_env env, napi_value d, napi_value a, napi_value b) {
  return numbl_jit_t_binary_real(env, d, a, b, NUMBL_REAL_BIN_DIV);
}

/* ── Tensor compare ops (real) ────────────────────────────────────────────
 * Same shape as binary, but result is logical (set _isLogical=true) and
 * uses numbl_real_comparison.
 */
static napi_value numbl_jit_t_compare_real(napi_env env, napi_value dest,
                                           napi_value a, napi_value b, int op) {
  numbl_jit_tensor_view_t av, bv;
  int aIsT = numbl_jit_read_tensor(env, a, &av);
  int bIsT = numbl_jit_read_tensor(env, b, &bv);
  if (aIsT && bIsT) {
    if (av.is_complex || bv.is_complex) return numbl_jit_bail(env);
    if (av.n != bv.n) return numbl_jit_bail(env);
    napi_value out_typed = NULL;
    int reused = 0;
    double *out = numbl_jit_get_out_buffer(env, dest, av.n, &out_typed, &reused);
    if (!out) return NULL;
    numbl_real_comparison(op, av.n, av.data, bv.data, out);
    return numbl_jit_finalize_real(env, dest,
                                   reused ? (napi_value)NULL : out_typed,
                                   av.shape_val, reused, 1);
  }
  if (aIsT) {
    if (av.is_complex) return numbl_jit_bail(env);
    double bs;
    if (!numbl_jit_read_scalar(env, b, &bs)) return numbl_jit_bail(env);
    napi_value out_typed = NULL;
    int reused = 0;
    double *out = numbl_jit_get_out_buffer(env, dest, av.n, &out_typed, &reused);
    if (!out) return NULL;
    numbl_real_scalar_comparison(op, av.n, bs, av.data,
                                 /*scalar_on_left=*/0, out);
    return numbl_jit_finalize_real(env, dest,
                                   reused ? (napi_value)NULL : out_typed,
                                   av.shape_val, reused, 1);
  }
  if (bIsT) {
    if (bv.is_complex) return numbl_jit_bail(env);
    double as;
    if (!numbl_jit_read_scalar(env, a, &as)) return numbl_jit_bail(env);
    napi_value out_typed = NULL;
    int reused = 0;
    double *out = numbl_jit_get_out_buffer(env, dest, bv.n, &out_typed, &reused);
    if (!out) return NULL;
    numbl_real_scalar_comparison(op, bv.n, as, bv.data,
                                 /*scalar_on_left=*/1, out);
    return numbl_jit_finalize_real(env, dest,
                                   reused ? (napi_value)NULL : out_typed,
                                   bv.shape_val, reused, 1);
  }
  return numbl_jit_bail(env);
}

static napi_value numbl_jit_tEq(napi_env env, napi_value d, napi_value a, napi_value b) {
  return numbl_jit_t_compare_real(env, d, a, b, NUMBL_CMP_EQ);
}
static napi_value numbl_jit_tNeq(napi_env env, napi_value d, napi_value a, napi_value b) {
  return numbl_jit_t_compare_real(env, d, a, b, NUMBL_CMP_NE);
}
static napi_value numbl_jit_tLt(napi_env env, napi_value d, napi_value a, napi_value b) {
  return numbl_jit_t_compare_real(env, d, a, b, NUMBL_CMP_LT);
}
static napi_value numbl_jit_tLe(napi_env env, napi_value d, napi_value a, napi_value b) {
  return numbl_jit_t_compare_real(env, d, a, b, NUMBL_CMP_LE);
}
static napi_value numbl_jit_tGt(napi_env env, napi_value d, napi_value a, napi_value b) {
  return numbl_jit_t_compare_real(env, d, a, b, NUMBL_CMP_GT);
}
static napi_value numbl_jit_tGe(napi_env env, napi_value d, napi_value a, napi_value b) {
  return numbl_jit_t_compare_real(env, d, a, b, NUMBL_CMP_GE);
}

/* ── Tensor unary (real) ─────────────────────────────────────────────────
 * Negate, plus the libnumbl_ops unary kernels (exp/sin/cos/.../sign).
 * Domain-restricted unaries (sqrt/log/log2/log10/asin/acos) are excluded
 * — feasibility doesn't allow them — same reason as scalar Phase 1.
 */
static napi_value numbl_jit_t_unary_real(napi_env env, napi_value dest,
                                         napi_value a, int op) {
  numbl_jit_tensor_view_t av;
  if (!numbl_jit_read_tensor(env, a, &av)) return numbl_jit_bail(env);
  if (av.is_complex) return numbl_jit_bail(env);
  napi_value out_typed = NULL;
  int reused = 0;
  double *out = numbl_jit_get_out_buffer(env, dest, av.n, &out_typed, &reused);
  if (!out) return NULL;
  numbl_real_unary_elemwise(op, av.n, av.data, out);
  return numbl_jit_finalize_real(env, dest,
                                 reused ? (napi_value)NULL : out_typed,
                                 av.shape_val, reused, 0);
}

/* tNeg via "scalar_binary MUL by -1": cleanest mirror of JS-JIT's
 * approach (which special-cases unary minus in jitHelpersTensor). */
static napi_value numbl_jit_tNeg(napi_env env, napi_value dest, napi_value a) {
  numbl_jit_tensor_view_t av;
  if (!numbl_jit_read_tensor(env, a, &av)) return numbl_jit_bail(env);
  if (av.is_complex) return numbl_jit_bail(env);
  napi_value out_typed = NULL;
  int reused = 0;
  double *out = numbl_jit_get_out_buffer(env, dest, av.n, &out_typed, &reused);
  if (!out) return NULL;
  numbl_real_scalar_binary_elemwise(NUMBL_REAL_BIN_MUL, av.n, -1.0, av.data,
                                    /*scalar_on_left=*/1, out);
  return numbl_jit_finalize_real(env, dest,
                                 reused ? (napi_value)NULL : out_typed,
                                 av.shape_val, reused, 0);
}

/* Per-op tiny wrappers for the supported tensor unaries. Codegen picks
 * the one matching the MATLAB builtin name. */
static napi_value numbl_jit_tExp  (napi_env e, napi_value d, napi_value a) { return numbl_jit_t_unary_real(e, d, a, NUMBL_UNARY_EXP); }
static napi_value numbl_jit_tAbs  (napi_env e, napi_value d, napi_value a) { return numbl_jit_t_unary_real(e, d, a, NUMBL_UNARY_ABS); }
static napi_value numbl_jit_tFloor(napi_env e, napi_value d, napi_value a) { return numbl_jit_t_unary_real(e, d, a, NUMBL_UNARY_FLOOR); }
static napi_value numbl_jit_tCeil (napi_env e, napi_value d, napi_value a) { return numbl_jit_t_unary_real(e, d, a, NUMBL_UNARY_CEIL); }
static napi_value numbl_jit_tRound(napi_env e, napi_value d, napi_value a) { return numbl_jit_t_unary_real(e, d, a, NUMBL_UNARY_ROUND); }
static napi_value numbl_jit_tFix  (napi_env e, napi_value d, napi_value a) { return numbl_jit_t_unary_real(e, d, a, NUMBL_UNARY_TRUNC); }
static napi_value numbl_jit_tSin  (napi_env e, napi_value d, napi_value a) { return numbl_jit_t_unary_real(e, d, a, NUMBL_UNARY_SIN); }
static napi_value numbl_jit_tCos  (napi_env e, napi_value d, napi_value a) { return numbl_jit_t_unary_real(e, d, a, NUMBL_UNARY_COS); }
static napi_value numbl_jit_tTan  (napi_env e, napi_value d, napi_value a) { return numbl_jit_t_unary_real(e, d, a, NUMBL_UNARY_TAN); }
static napi_value numbl_jit_tAtan (napi_env e, napi_value d, napi_value a) { return numbl_jit_t_unary_real(e, d, a, NUMBL_UNARY_ATAN); }
static napi_value numbl_jit_tSinh (napi_env e, napi_value d, napi_value a) { return numbl_jit_t_unary_real(e, d, a, NUMBL_UNARY_SINH); }
static napi_value numbl_jit_tCosh (napi_env e, napi_value d, napi_value a) { return numbl_jit_t_unary_real(e, d, a, NUMBL_UNARY_COSH); }
static napi_value numbl_jit_tTanh (napi_env e, napi_value d, napi_value a) { return numbl_jit_t_unary_real(e, d, a, NUMBL_UNARY_TANH); }
static napi_value numbl_jit_tSign (napi_env e, napi_value d, napi_value a) { return numbl_jit_t_unary_real(e, d, a, NUMBL_UNARY_SIGN); }

/* ── Reductions (tensor → scalar number) ──────────────────────────────────
 * Mirror tSum: vector-only, real-only, returns a JS number. Returns the
 * bail sentinel for matrices and complex (matches JS-side return-undefined
 * in tSum which causes JS-JIT to defer to the interpreter).
 *
 * Shape vector check: ndim==1, OR ndim==2 with dim 0 or dim 1 == 1, OR
 * ndim==0 (zero-d scalar tensor).
 */
static int numbl_jit_shape_is_vector(napi_env env, napi_value shape_val) {
  napi_status status;
  uint32_t len;
  status = napi_get_array_length(env, shape_val, &len);
  if (status != napi_ok) return 0;
  if (len == 0) return 1;        /* 0-d tensor (treated as scalar by JS) */
  if (len == 1) return 1;        /* 1-d vector */
  if (len == 2) {
    napi_value d0v, d1v;
    status = napi_get_element(env, shape_val, 0, &d0v);
    if (status != napi_ok) return 0;
    status = napi_get_element(env, shape_val, 1, &d1v);
    if (status != napi_ok) return 0;
    int32_t d0 = 0, d1 = 0;
    napi_get_value_int32(env, d0v, &d0);
    napi_get_value_int32(env, d1v, &d1);
    if (d0 == 1 || d1 == 1) return 1;
  }
  return 0;
}

static napi_value numbl_jit_make_double(napi_env env, double v) {
  napi_value out;
  if (napi_create_double(env, v, &out) != napi_ok) return numbl_jit_bail(env);
  return out;
}

static napi_value numbl_jit_t_reduce_real(napi_env env, napi_value v, int op) {
  /* Scalar passthrough: number → number; boolean → 0/1. */
  napi_status status;
  napi_valuetype t;
  status = napi_typeof(env, v, &t);
  if (status != napi_ok) return numbl_jit_bail(env);
  if (t == napi_number) return v;
  if (t == napi_boolean) {
    bool bv;
    napi_get_value_bool(env, v, &bv);
    return numbl_jit_make_double(env, bv ? 1.0 : 0.0);
  }
  numbl_jit_tensor_view_t av;
  if (!numbl_jit_read_tensor(env, v, &av)) return numbl_jit_bail(env);
  if (av.is_complex) return numbl_jit_bail(env);
  if (!numbl_jit_shape_is_vector(env, av.shape_val)) return numbl_jit_bail(env);
  double out = 0.0;
  numbl_real_flat_reduce(op, av.n, av.data, &out);
  return numbl_jit_make_double(env, out);
}

static napi_value numbl_jit_tSum  (napi_env e, napi_value v) { return numbl_jit_t_reduce_real(e, v, NUMBL_REDUCE_SUM); }
static napi_value numbl_jit_tProd (napi_env e, napi_value v) { return numbl_jit_t_reduce_real(e, v, NUMBL_REDUCE_PROD); }
static napi_value numbl_jit_tMax  (napi_env e, napi_value v) { return numbl_jit_t_reduce_real(e, v, NUMBL_REDUCE_MAX); }
static napi_value numbl_jit_tMin  (napi_env e, napi_value v) { return numbl_jit_t_reduce_real(e, v, NUMBL_REDUCE_MIN); }
static napi_value numbl_jit_tMean (napi_env e, napi_value v) { return numbl_jit_t_reduce_real(e, v, NUMBL_REDUCE_MEAN); }
static napi_value numbl_jit_tAny  (napi_env e, napi_value v) { return numbl_jit_t_reduce_real(e, v, NUMBL_REDUCE_ANY); }
static napi_value numbl_jit_tAll  (napi_env e, napi_value v) { return numbl_jit_t_reduce_real(e, v, NUMBL_REDUCE_ALL); }
`;
