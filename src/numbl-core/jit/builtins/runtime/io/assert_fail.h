/* @snippet name=jit_assert_scalar */
/* JIT runtime helper: scalar `assert(cond)` / `assert(cond, msg)` with a
 * literal message. Tests `cond` for MATLAB truthiness (non-zero and not
 * NaN) and raises a MATLAB error (`raise.h`) on failure.
 *
 * The codegen passes the interpreter's wording: "Assertion failed." with
 * identifier `MATLAB:assertion:failed` for the 1-arg form, the message
 * taken literally (identifier "") for the 2-arg form. Tensor conds are
 * declined at lowering.
 */

static void jit_assert_scalar(double cond, const char *id, const char *msg) {
  /* `cond != cond` filters NaN — numbl's `isNaN(v)` check. */
  if (cond != 0.0 && cond == cond) return;
  jit_raise_error(id, msg);
}
