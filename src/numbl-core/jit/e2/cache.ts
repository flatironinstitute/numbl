/**
 * e2 — per-AST-node compiled-kernel cache.
 *
 * Each AST `Expr` (the RHS of an `Assign` we've seen at least once) maps
 * to a per-signature cache: the same expression visited with a different
 * runtime type signature produces a different specialization. The
 * signature includes input names, scalar-vs-tensor, complex-or-not, and
 * the LHS name (since the kernel hard-codes which output to write).
 *
 * The Map is keyed by the AST node identity, not by source text — two
 * identical-looking `r = r .* y` statements at different file:line
 * positions get separate cache entries, so a recompile from a
 * different call-site doesn't poison earlier ones.
 *
 * The cache holds either an `E2CacheEntry` or the `E2_BAILED` sentinel
 * indicating that classification or compilation failed for this
 * signature; the sentinel prevents re-attempting the same hopeless
 * lowering on every invocation.
 */

import type { Stmt, BinaryOperation } from "../../parser/types.js";

export type CompiledKernelFn = (...args: unknown[]) => unknown;

export const E2_BAILED: unique symbol = Symbol("E2_BAILED");

export interface E2ReductionInfo {
  /** Reduction op name (sum / prod / max / min / mean / any / all). */
  reduceName: string;
  /** Accumulator variable name in env. */
  accName: string;
  /** When true, the source pattern was `acc = acc OP reduce(...)`;
   *  the driver applies the same OP to combine the kernel's scalar
   *  output with the existing env value of `acc`. When false, the
   *  source pattern was `acc = reduce(...)` and the kernel output is
   *  written directly. */
  hasAccumulate: boolean;
  /** Only meaningful when `hasAccumulate` is true. */
  accOp?: BinaryOperation;
}

export interface E2CacheEntry {
  fn: CompiledKernelFn;
  /** Regular env tensor input names — order matches kernel signature. */
  tensorNames: string[];
  /** Chain LHS names that need `in_<name>` (between tensors and scalars). */
  inputLhsNames: string[];
  /** Ordered scalar input names. */
  scalarNames: string[];
  /** Chain LHS names that materialize via `out_<name>` (escape names). */
  escapeLhsNames: string[];
  /** Number of chain assigns this entry encodes (0 for a standalone
   *  reduction kernel, 1 for a single-assign chain kernel, >=2 for
   *  multi-stmt chains). */
  chainLength: number;
  /** Set when the kernel produces a trailing scalar reduction output.
   *  The driver allocates a `Float64Array(1)` for `out_acc`, calls the
   *  kernel, then combines the result with `env[accName]` per the
   *  `accOp` and `hasAccumulate` fields. */
  reduction?: E2ReductionInfo;
}

// Chain cache: keyed on the FIRST AST Stmt of the chain. Different
// signatures (chain length, input types) hash to different entries.
const stmtCache = new WeakMap<
  Stmt,
  Map<string, E2CacheEntry | typeof E2_BAILED>
>();

export function chainCacheGet(
  firstStmt: Stmt,
  sig: string
): E2CacheEntry | typeof E2_BAILED | undefined {
  return stmtCache.get(firstStmt)?.get(sig);
}

export function chainCacheSet(
  firstStmt: Stmt,
  sig: string,
  entry: E2CacheEntry | typeof E2_BAILED
): void {
  let m = stmtCache.get(firstStmt);
  if (!m) {
    m = new Map();
    stmtCache.set(firstStmt, m);
  }
  m.set(sig, entry);
}
