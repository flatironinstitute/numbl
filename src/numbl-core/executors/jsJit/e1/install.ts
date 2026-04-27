/**
 * Node-only install shim for the e1 (experimental) kernel pipeline.
 *
 * Side-effect import from `cli.ts`. Replaces the `compileKernel` stub on
 * the module-level `jitHelpers` object with a real implementation that
 * shells out to `cc` via `compile.ts` and loads the result through koffi.
 *
 * Registration is idempotent — re-importing this module in tests won't
 * re-install. The kernel cache on `jitHelpers.$kernels` is shared across
 * all specializations in the process so the same fused chain used from
 * two different JIT'd functions compiles only once.
 */

import { jitHelpers } from "../js/jitHelpers.js";
import { compileAndLoad, cJitOpenmpAvailable } from "../../../jit/c/compile.js";
import { setOpenmpAvailableGetter } from "../../../jit/openmpFlag.js";

let _installed = false;

function install(): void {
  if (_installed) return;
  _installed = true;

  // Swap the browser-safe OpenMP-flag stub for the real Node probe. The
  // e1 codegen (scalarFnKernel.ts) reads this via `isOpenmpAvailable()`
  // — see openmpFlag.ts for why the indirection exists.
  setOpenmpAvailableGetter(() => cJitOpenmpAvailable());

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const h = jitHelpers as any;

  // Sanity: the kernels cache is a plain object populated lazily per
  // hash. The generated JS uses `$h.$kernels[hash] ??= $h.compileKernel(...)`.
  if (!h.$kernels) h.$kernels = {};

  h.compileKernel = (cSource: string, koffiSig: string) => {
    // Parse the function name out of the koffi signature — everything
    // between the last space before the `(` and the `(` itself. We
    // only need it for the compileAndLoad caller-id / error messages.
    const parenIdx = koffiSig.indexOf("(");
    const headerBeforeParen = koffiSig.slice(0, parenIdx).trim();
    const fnName = headerBeforeParen.split(/\s+/).pop() ?? "kernel";

    // Link with -fopenmp when the compiler supports it so `#pragma omp
    // parallel for` actually spawns threads. Without this, the pragma is
    // silently reduced to a serial loop. Mirrors the unconditional link
    // in c/install.ts — cost is negligible for kernels that don't use
    // the pragma (the runtime is only brought in if a parallel region
    // runs).
    const ompLink = cJitOpenmpAvailable();
    const loaded = compileAndLoad(
      cSource,
      koffiSig,
      fnName,
      msg => process.stderr.write(`[e1] ${msg}\n`),
      ompLink ? ["-fopenmp"] : undefined
    );
    if (!loaded) {
      // Fall back: the helper returns a function that throws at call
      // time. The generated JS's size-dispatch branch only reaches us
      // when N >= threshold; a failure here means the JS fallback
      // path wouldn't have been taken, so surface the error loudly.
      return () => {
        throw new Error(`--opt e1: C kernel compile/load failed for ${fnName}`);
      };
    }
    return loaded.fn as (...args: unknown[]) => unknown;
  };
}

install();
