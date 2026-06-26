/**
 * Build the `$h` helpers object that mtoc2's emitted spec receives
 * when invoked via the numbl JIT bridge.
 *
 *  - `write(s)` — output sink. Routed to `rt.output(...)` so `disp` /
 *    `fprintf` / `warning` etc. land in numbl's output stream.
 *  - `plotDispatch(name, args)` — host hook for plot builtins.
 *    mtoc2's emitted `mtoc2_plot_dispatch(...)` calls this in
 *    preference to the standalone-mode JSON-on-stdout wire format.
 *    We translate args from mtoc2's emit shape into `RuntimeValue`
 *    via `jitToNumbl` and call numbl's `dispatchPlotBuiltin`,
 *    pushing the resulting instruction onto `rt.plotInstructions` —
 *    so `colorbar('off')` / `pcolor(...)` / etc. produce the same
 *    plot-instruction stream as the interpreter would.
 *
 * The unknown-name fallthrough (when `dispatchPlotBuiltin` returns
 * false) is a programmer error: every name mtoc2's plot builtin
 * registration covers is in `PLOT_ALL_NAMES`, which is sourced from
 * the same list `dispatchPlotBuiltin` switches on. Throw so the
 * mismatch is loud rather than silently dropping a side effect.
 */

import type { Runtime } from "../../runtime/runtime.js";
import type { RuntimeValue } from "../../runtime/types.js";
import { dispatchPlotBuiltin } from "../../runtime/plotBuiltinDispatch.js";
import { rngRandom } from "../../helpers/prng.js";
import { jitToNumbl } from "./valueAdapter.js";

export interface JitHostHelpers {
  write: (s: string) => void;
  plotDispatch: (name: string, args: unknown[]) => void;
  /** One uniform double in [0, 1). Backed by numbl's process-global PRNG
   *  (the same `rngRandom` the interpreter uses), so JIT'd and interpreted
   *  `rand` draw from one shared, `rng(seed)`-controllable stream. */
  rand: () => number;
}

export function buildHostHelpers(rt: Runtime): JitHostHelpers {
  return {
    write: (s: string) => rt.output(s),
    rand: () => rngRandom(),
    plotDispatch: (name: string, args: unknown[]) => {
      const runtimeArgs: RuntimeValue[] = args.map(a => jitToNumbl(a));
      const handled = dispatchPlotBuiltin(
        name,
        runtimeArgs,
        rt.plotInstructions,
        rt
      );
      if (!handled) {
        throw new Error(
          `mtoc2 plot bridge: dispatchPlotBuiltin rejected '${name}' — ` +
            `mtoc2's plot builtin set is out of sync with numbl's dispatch table`
        );
      }
    },
  };
}
