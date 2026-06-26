/**
 * rand — uniform random scalar in [0, 1).
 *
 * Only the scalar form `rand()` (zero args) is JIT-compiled. It emits a call
 * to the host helper `$h.rand` (bound to `globalThis.$rand` by the emit
 * wrapper), which draws from numbl's process-global PRNG — the same stream the
 * interpreter's `rand` uses, so `rng(seed)` controls both and a function that
 * mixes JIT'd and interpreted `rand` stays on one sequence.
 *
 * Matrix forms (`rand(n)`, `rand(m,n)`, `rand([m n])`) and multi-output are
 * declined with `UnsupportedConstruct`, so the enclosing unit falls back to the
 * interpreter (which handles every form). There is no `emitC`: under C-JIT
 * (`--opt 2`) there is no JS host to call, so a `rand`-containing unit falls
 * through to JS-JIT, which this enables.
 */
import { UnsupportedConstruct } from "../../../lowering/errors.js";
import { scalarDouble } from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import { rngRandom } from "../../../../helpers/prng.js";

export const rand: Builtin = {
  name: "rand",
  transfer(argTypes, nargout) {
    if (nargout > 1) {
      throw new UnsupportedConstruct(
        `'rand' does not support multi-output (nargout=${nargout})`
      );
    }
    if (argTypes.length !== 0) {
      // Sizing/seeding forms (rand(n), rand(m,n), rand('seed',…), …) are not
      // JIT'd yet; decline so the interpreter handles them.
      throw new UnsupportedConstruct(
        `JS-JIT 'rand' supports only the scalar form rand() so far ` +
          `(got ${argTypes.length} arg(s)); matrix/seed forms run in the interpreter`
      );
    }
    return [scalarDouble("nonneg")];
  },
  emitJs() {
    // `$rand` is bound to the host PRNG helper by the emit wrapper.
    return "$rand()";
  },
  call() {
    return [rngRandom()];
  },
};
