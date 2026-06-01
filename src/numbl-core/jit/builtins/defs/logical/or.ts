/**
 * `or(a, b)` — the function form of `|`. For scalar arguments,
 * mtoc2 routes this through the same scaffold as `||` (`oror`): the
 * result type is scalar logical, the codegen uses C's `||` operator,
 * and the only semantic difference (short-circuit on `||` vs.
 * eager-evaluate on `|` / `or`) is moot — by the time the function
 * is called, both arguments have already been computed by the
 * call-site machinery, so the C-level short-circuit is harmless.
 * Tensor / elementwise behavior is deferred.
 */

import { defineShortCircuit } from "./_shortcircuit.js";

export const orBuiltin = defineShortCircuit("or", "|", "or");
