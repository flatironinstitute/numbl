/* @snippet name=jit_assert_scalar_fmt deps=jit_error_fmt */
/* JIT runtime helpers: `assert(cond, msg)` with a non-literal message and
 * `assert(cond, [id,] fmt, args...)`: raise a MATLAB error (`raise.h`)
 * when `cond` is not MATLAB-truthy (non-zero and not NaN).
 *
 *   jit_assert_text        message taken literally (the 2-arg form)
 *   jit_assert_scalar_fmt  message formatted with the shared format engine
 *                          (spec set / escapes / arg cycling as `fprintf`)
 *
 * `id` is the error identifier ("" for none), resolved at lowering.
 */

static void jit_assert_text(double cond, const char *id, jit_text_view_t msg) {
  if (cond != 0.0 && cond == cond) return;
  jit_error_text(id, msg);
}

static void jit_assert_scalar_fmt(double cond, const char *id,
                                  jit_text_view_t fmt, int nargs,
                                  const jit_fprintf_arg_t *args) {
  if (cond != 0.0 && cond == cond) return;
  jit_error_fmt(id, fmt, nargs, args);
}
