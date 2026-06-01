/* mtoc runtime helper: consume-and-replace tensor assignment.
 *
 * Frees `*lhs`'s current backing buffers and moves `rhs`'s buffers
 * into `*lhs`. The codegen invariant is that every tensor C-expression
 * on an Assign RHS produces a freshly-owned tensor — a literal, a
 * `mtoc2_tensor_copy(...)`, or an elementwise-result built by
 * `mtoc2_tensor_alloc(...)` — so the move (rather than a deep copy
 * here) is sound: there is no caller-side reference to the rhs after
 * the call. After the move `rhs.real`/`rhs.imag` hold the old `*lhs`
 * pointers, but rhs is a temporary in the C sense (it's a value
 * passed by copy) and is discarded immediately.
 *
 * Both frees are unconditional — see `mtoc2_tensor_free` for why
 * that's safe for both real and complex tensors.
 */

#include <stdlib.h>

static void mtoc2_tensor_assign(mtoc2_tensor_t *lhs, mtoc2_tensor_t rhs) {
  free(lhs->real);
  free(lhs->imag);
  *lhs = rhs;
}
