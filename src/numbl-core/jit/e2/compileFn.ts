/**
 * e2 — browser-safe indirection for the C compile driver.
 *
 * The driver in `c/compile.ts` is Node-only (it shells out to `cc` and
 * loads via koffi). The browser bundle includes the e2 modules but
 * NOT this driver — `setE2CompileFn` from `e2/install.ts` (Node only)
 * swaps in the real implementation. In the browser, the stub throws.
 */

export type E2CompileFn = (
  cSource: string,
  koffiSig: string,
  kernelName: string,
  log?: (msg: string) => void
) => ((...args: unknown[]) => unknown) | null;

let _compileFn: E2CompileFn = (_src, sig, name) => {
  throw new Error(
    "--opt e2: C-kernel compilation unavailable (Node-only). " +
      `kernel=${name}, sig=${sig}`
  );
};

export function setE2CompileFn(fn: E2CompileFn): void {
  _compileFn = fn;
}

export function getE2CompileFn(): E2CompileFn {
  return _compileFn;
}

/** Minimum element count of the largest tensor input before we'll
 *  consider compiling an e2 kernel. Below this, koffi overhead dwarfs
 *  the work and falling through to the interpreter is faster.
 *  Overridable via `NUMBL_E2_MIN_ELEMS`. */
export function e2MinElems(): number {
  return parseInt(process.env.NUMBL_E2_MIN_ELEMS || "", 10) || 1000;
}
