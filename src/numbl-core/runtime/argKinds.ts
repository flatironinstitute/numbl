/**
 * The argument-kind policy of builtins.
 *
 * Some value kinds are storage variants of an ordinary array that most
 * builtins have no code for (a sparse matrix; later, say, an array held
 * on a device). A builtin declares the variants it handles in its spec's
 * `accepts`; for any other, the kind's default is applied after the arity
 * check and before `apply` runs (in the call wrapper `builtin()` builds,
 * which `invokeBuiltin`, `invokeClassMethod` and every other caller go
 * through): a sparse matrix is converted to a dense tensor. A
 * builtin can instead have the kind rejected with MATLAB's error. So a
 * builtin without a declaration never sees these kinds, and adding a kind
 * means adding a rule here, not editing every builtin.
 */

import type { RuntimeSparseMatrix, RuntimeValue } from "./types.js";
import type { ValueKind } from "./kinds.js";
import { RuntimeError } from "./error.js";
import { sparseToDense } from "../helpers/sparse-arithmetic.js";

/** How a builtin takes arguments of one kind:
 *  - `true`: as they are (the builtin handles the kind itself);
 *  - `"convert"`: converted by the kind's default (the default);
 *  - `"error"`, or `{ identifier, message }`: rejected with an error. */
export type KindAcceptance =
  | true
  | "convert"
  | "error"
  | { readonly identifier: string; readonly message: string };

/** A builtin's declaration of how it takes the non-dense kinds. */
export interface ArgKindPolicy {
  /** Sparse matrices. Default: converted to dense. */
  sparse?: KindAcceptance;
}

type Action = null | ((v: RuntimeValue) => RuntimeValue);

/** A builtin's policy resolved per value kind: for each kind with a rule,
 *  pass the value through (`null`) or a function that converts it or
 *  throws. Kinds with no rule are absent. */
export type ResolvedArgPolicy = { readonly [K in ValueKind]?: Action };

interface KindRule {
  /** The key in `ArgKindPolicy`. */
  policy: keyof ArgKindPolicy;
  /** The default conversion, to a kind every builtin handles. */
  convert(v: RuntimeValue): RuntimeValue;
  /** The error for `"error"`. */
  error(builtinName: string): RuntimeError;
}

/** One rule per value kind the policy applies to, keyed by `kind`. A kind
 *  with no rule (the dense values) passes through untouched. */
const RULES: { readonly [K in ValueKind]?: KindRule } = {
  sparse_matrix: {
    policy: "sparse",
    convert: v => sparseToDense(v as RuntimeSparseMatrix),
    error: name =>
      new RuntimeError("Function is not defined for sparse inputs.", null, {
        identifier: `MATLAB:${name}:sparseNotSupported`,
      }),
  },
};

/** Resolve a builtin's declared policy (undefined: every default). Done
 *  once, when the builtin is defined. */
export function resolveArgPolicy(
  builtinName: string,
  accepts: ArgKindPolicy | undefined
): ResolvedArgPolicy {
  const out: { [K in ValueKind]?: Action } = {};
  for (const kind of Object.keys(RULES) as ValueKind[]) {
    const rule = RULES[kind]!;
    const a = accepts?.[rule.policy] ?? "convert";
    if (a === true) out[kind] = null;
    else if (a === "convert") out[kind] = rule.convert;
    else {
      const err =
        a === "error"
          ? () => rule.error(builtinName)
          : () =>
              new RuntimeError(a.message, null, { identifier: a.identifier });
      out[kind] = () => {
        throw err();
      };
    }
  }
  return out;
}

/** Apply `policy` to a call's arguments. One property lookup per
 *  argument; `args` itself is returned unless an argument is converted,
 *  so dense arguments are never copied. */
export function applyArgPolicy(
  policy: ResolvedArgPolicy,
  args: RuntimeValue[]
): RuntimeValue[] {
  let out = args;
  for (let i = 0; i < args.length; i++) {
    const v = args[i];
    if (typeof v !== "object") continue;
    const action = policy[v.kind];
    if (!action) continue;
    if (out === args) out = args.slice();
    out[i] = action(v);
  }
  return out;
}
