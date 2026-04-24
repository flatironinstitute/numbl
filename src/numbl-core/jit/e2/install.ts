/**
 * e2 — Node-only install hook.
 *
 * Sets the module-level `e2CompileFn` to the real `compileAndLoad`
 * driver from `c/compile.ts`. The browser bundle never imports this
 * file, so `e2CompileFn` stays at the throwing stub and any attempt
 * to use `--opt e2` from the web fails with a clear message.
 *
 * Idempotent: re-importing in tests doesn't re-install.
 */

import { compileAndLoad, cJitOpenmpAvailable } from "../c/compile.js";
import { setE2CompileFn } from "./compileFn.js";

let _installed = false;

if (!_installed) {
  _installed = true;
  setE2CompileFn((cSource, koffiSig, kernelName, log) => {
    // Link with -fopenmp when the compiler supports it so e2's
    // `--par` `#pragma omp parallel for` actually spawns threads.
    // Without this, the pragma silently degrades to a serial loop.
    // Cost is negligible for kernels that don't use the pragma —
    // the runtime is only brought in if a parallel region runs.
    const ompLink = cJitOpenmpAvailable();
    const loaded = compileAndLoad(
      cSource,
      koffiSig,
      kernelName,
      log,
      ompLink ? ["-fopenmp"] : undefined
    );
    if (!loaded) return null;
    return loaded.fn;
  });
}
