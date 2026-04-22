/**
 * Thrown under `--check-c-jit-parity` when the C-JIT declines to compile
 * a function/loop whose IR the JS-JIT would have accepted. The message
 * names the construct (from the feasibility checker) and the call site,
 * giving us an actionable punch list of features to implement in the
 * C-JIT so parity with JS-JIT holds.
 */
export class CJitParityError extends Error {
  constructor(
    message: string,
    public readonly reason: string,
    public readonly kind: "infeasible" | "env"
  ) {
    super(message);
    this.name = "CJitParityError";
  }
}

/**
 * Build a one-line parity-error message with the offending construct,
 * the call site (function/loop name + file:line), and the arg-type
 * signature that triggered the specialization.
 */
export function formatCJitParityMessage(opts: {
  kind: "infeasible" | "env";
  reason: string;
  reasonLine?: number;
  siteLabel: string; // e.g. "fn foo" or "for-loop"
  file: string;
  callSiteLine: number;
  argsDesc: string; // e.g. "x: number, A: tensor[100,100] real"
}): string {
  const where = opts.reasonLine
    ? `${opts.file}:${opts.reasonLine}`
    : `${opts.file}:${opts.callSiteLine}`;
  const head = opts.kind === "env" ? "C-JIT unavailable" : "C-JIT parity gap";
  return `${head}: ${opts.reason} [${opts.siteLabel} @ ${where}; args: (${opts.argsDesc})]`;
}
