/**
 * Interpreter special builtins: functions that need access to interpreter
 * context (env, callerEnv, etc.) and run before normal function resolution.
 */

import {
  isRuntimeNumber,
  isRuntimeTensor,
  isRuntimeChar,
  isRuntimeString,
  isRuntimeCell,
  isRuntimeClassInstance,
  isRuntimeFunction,
} from "../runtime/types.js";
import { RTV } from "../runtime/constructors.js";
import { ensureRuntimeValue } from "../runtime/runtimeHelpers.js";
import { RuntimeError } from "../runtime/error.js";
import { getIBuiltin } from "./builtins/index.js";
import { toNumber, toString } from "../runtime/convert.js";
import type { Environment } from "./types.js";
import type { Runtime } from "../runtime/runtime.js";

// ── Types ────────────────────────────────────────────────────────────────

export interface InterpreterContext {
  env: Environment;
  setEnv: (env: Environment) => void;
  callerEnv: Environment | undefined;
  workspaceEnv: Environment | undefined;
  evalInLocalScope: (codeArg: unknown) => unknown;
  callFunction: (name: string, args: unknown[], nargout: number) => unknown;
  rt: Runtime;
}

export const FALL_THROUGH: unique symbol = Symbol("FALL_THROUGH");

export type InterpreterSpecialBuiltinHandler = (
  ctx: InterpreterContext,
  args: unknown[],
  nargout: number
) => unknown | typeof FALL_THROUGH;

// ── Registry ─────────────────────────────────────────────────────────────

const registry = new Map<string, InterpreterSpecialBuiltinHandler>();

function register(
  name: string,
  handler: InterpreterSpecialBuiltinHandler
): void {
  registry.set(name, handler);
}

export function getInterpreterSpecialBuiltin(
  name: string
): InterpreterSpecialBuiltinHandler | undefined {
  return registry.get(name);
}

// ── Handlers ─────────────────────────────────────────────────────────────

register("eval", (ctx, args) => {
  if (args.length !== 1) return FALL_THROUGH;
  return ctx.evalInLocalScope(args[0]);
});

register("evalin", (ctx, args) => {
  if (args.length < 2) return FALL_THROUGH;
  const scope = toString(ensureRuntimeValue(args[0]));
  const targetEnv =
    scope === "caller"
      ? ctx.callerEnv
      : scope === "workspace"
        ? ctx.workspaceEnv
        : undefined;
  if (targetEnv) {
    const code = toString(ensureRuntimeValue(args[1]));
    // Simple variable name — just look it up
    if (/^[a-zA-Z_]\w*$/.test(code)) {
      const val = targetEnv.get(code);
      if (val !== undefined) return val;
      if (args.length >= 3) return ensureRuntimeValue(args[2]);
      throw new RuntimeError(
        `Variable '${code}' does not exist in ${scope} scope`
      );
    }
    // Otherwise evaluate as code in target scope
    const savedEnv = ctx.env;
    ctx.setEnv(targetEnv);
    try {
      return ctx.evalInLocalScope(args[1]);
    } finally {
      ctx.setEnv(savedEnv);
    }
  }
  return ctx.evalInLocalScope(args[1]);
});

register("assignin", (ctx, args) => {
  if (args.length < 3) return FALL_THROUGH;
  const scope = toString(ensureRuntimeValue(args[0]));
  const varName = toString(ensureRuntimeValue(args[1]));
  const val = ensureRuntimeValue(args[2]);
  const targetEnv =
    scope === "caller"
      ? ctx.callerEnv
      : scope === "workspace"
        ? ctx.workspaceEnv
        : undefined;
  if (targetEnv) {
    targetEnv.set(varName, val);
  } else {
    ctx.env.set(varName, val);
  }
  return undefined;
});

register("feval", (ctx, args, nargout) => {
  if (args.length < 1) return FALL_THROUGH;
  const first = ensureRuntimeValue(args[0]);
  if (isRuntimeFunction(first)) {
    return ctx.rt.index(first, args.slice(1), nargout);
  }
  // Class instance: fall through to normal resolution
  if (isRuntimeClassInstance(first)) {
    return FALL_THROUGH;
  }
  const funcName = toString(first);
  return ctx.callFunction(funcName, args.slice(1), nargout);
});

register("exist", (ctx, args) => {
  if (args.length < 2) return FALL_THROUGH;
  const nameArg = toString(ensureRuntimeValue(args[0]));
  const typeArg = toString(ensureRuntimeValue(args[1]));
  if (typeArg === "var") {
    return ctx.env.has(nameArg) ? 1 : 0;
  }
  if (typeArg === "builtin") {
    return ctx.rt.builtins[nameArg] || getIBuiltin(nameArg) ? 5 : 0;
  }
  return FALL_THROUGH;
});

register("who", (ctx, args, nargout) => {
  const getters: Record<string, () => unknown> = {};
  for (const varName of ctx.env.localNames()) {
    if (varName.startsWith("$")) continue;
    const n = varName;
    getters[n] = () => ctx.env.get(n);
  }
  return ctx.rt.who(nargout, getters, args);
});

register("whos", (ctx, args, nargout) => {
  const getters: Record<string, () => unknown> = {};
  for (const varName of ctx.env.localNames()) {
    if (varName.startsWith("$")) continue;
    const n = varName;
    getters[n] = () => ctx.env.get(n);
  }
  return ctx.rt.whos(nargout, getters, args);
});

register("isa", (ctx, args) => {
  if (args.length !== 2) return FALL_THROUGH;
  return ctx.rt.isa(args[0], args[1]);
});

register("__inferred_type_str", (_ctx, args) => {
  if (args.length !== 1) return FALL_THROUGH;
  const rv = ensureRuntimeValue(args[0]);
  if (isRuntimeNumber(rv)) return RTV.string("Number");
  if (isRuntimeTensor(rv)) return RTV.string("Tensor");
  if (isRuntimeCell(rv)) return RTV.string("Cell");
  if (isRuntimeClassInstance(rv))
    return RTV.string(`ClassInstance(${rv.className})`);
  if (isRuntimeChar(rv)) return RTV.string("Char");
  if (isRuntimeString(rv)) return RTV.string("String");
  if (isRuntimeFunction(rv)) return RTV.string("Function");
  if (typeof rv === "boolean") return RTV.string("Boolean");
  return RTV.string("Unknown");
});

register("nargin", (ctx, args) => {
  if (args.length !== 0) return FALL_THROUGH;
  const v = ctx.env.get("$nargin");
  return v !== undefined ? v : 0;
});

register("nargout", (ctx, args) => {
  if (args.length !== 0) return FALL_THROUGH;
  const v = ctx.env.get("$nargout");
  return v !== undefined ? v : 0;
});

register("narginchk", (ctx, args) => {
  if (args.length !== 2) return FALL_THROUGH;
  const narginVal = ctx.env.get("$nargin");
  const n = typeof narginVal === "number" ? narginVal : 0;
  const lo = toNumber(ensureRuntimeValue(args[0]));
  const hi = toNumber(ensureRuntimeValue(args[1]));
  if (n < lo) throw new RuntimeError("Not enough input arguments.");
  if (n > hi) throw new RuntimeError("Too many input arguments.");
  return undefined;
});

register("nargoutchk", (ctx, args) => {
  if (args.length !== 2) return FALL_THROUGH;
  const nargoutVal = ctx.env.get("$nargout");
  const n = typeof nargoutVal === "number" ? nargoutVal : 0;
  const lo = toNumber(ensureRuntimeValue(args[0]));
  const hi = toNumber(ensureRuntimeValue(args[1]));
  if (n < lo) throw new RuntimeError("Not enough output arguments.");
  if (n > hi) throw new RuntimeError("Too many output arguments.");
  return undefined;
});
