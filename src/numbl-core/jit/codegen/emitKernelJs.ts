/**
 * JS rendering of an `ElemwiseKernel` (built by the kernel pass,
 * `lowering/elemwiseKernels.ts`): one loop over the operands' data arrays
 * that allocates only the result, or, for a reducing kernel, folds the
 * elements into an accumulator in linear order (the order of the
 * `jit_<op>_all` helpers). The JS fusion policy admits only operations
 * whose per-element result IEEE arithmetic fixes, so the loop computes the
 * bits the per-op helpers (and the host kernels they use) would.
 */

import type { ElemwiseKernel, IRExpr } from "../lowering/ir.js";
import { isNumeric } from "../lowering/types.js";
import { getBuiltin } from "../builtins/index.js";
import { requireEmitJs, type InlineSnippet } from "../builtins/registry.js";
import { FUSED_SHAPE_EQ_SNIPPET } from "./emitKernelC.js";

/** What the kernel emitter needs from `emitJs.ts`. */
export interface KernelJsContext {
  /** A leaf (`Var` / `NumLit`) or a whole expression, emitted normally. */
  emitExpr(e: IRExpr): string;
  useRuntime(spec: string | InlineSnippet): void;
}

function emitBodyJs(e: IRExpr, ctx: KernelJsContext): string {
  const emitCall = (name: string, args: IRExpr[]): string => {
    const b = getBuiltin(name);
    if (!b) throw new Error(`emitKernelJs internal: no builtin '${name}'`);
    return requireEmitJs(b)({
      argsJs: args.map(a => emitBodyJs(a, ctx)),
      argTypes: args.map(a => a.ty),
      nargout: 1,
      useRuntime: ctx.useRuntime,
    });
  };
  switch (e.kind) {
    case "NumLit":
    case "Var":
      return ctx.emitExpr(e);
    case "KernelElem":
      return `_jit_ex${e.input}[_jit_ei]`;
    case "Binary":
      return emitCall(e.builtin, [e.left, e.right]);
    case "Unary":
      return emitCall(e.builtin, [e.operand]);
    case "Call":
      return emitCall(e.name, e.args);
    default:
      throw new Error(`emitKernelJs internal: unexpected node '${e.kind}'`);
  }
}

/** Statements assigning kernel `k` to `target` (a tensor, or the scalar of
 *  a reducing kernel). `storeScalar` turns the reduced value into the
 *  store statement. */
export function emitKernelJs(
  k: ElemwiseKernel,
  target: string,
  indent: string,
  ctx: KernelJsContext,
  storeScalar: (value: string) => string
): string {
  const ind = indent + "  ";
  const lines: string[] = [`${indent}{`];
  const tensors = k.inputs.map((v, i) => `_jit_et${i} = ${ctx.emitExpr(v)}`);
  lines.push(`${ind}const ${tensors.join(", ")};`);
  const body: string[] = [];
  const data = k.inputs.map((_, i) => `_jit_ex${i} = _jit_et${i}.data`);
  body.push(`const ${data.join(", ")};`);
  const slot = emitBodyJs(k.body, ctx);
  if (k.reduce === undefined) {
    ctx.useRuntime("jit_tensor_alloc_nd");
    body.push(
      `const _jit_er = jit_tensor_alloc_nd(_jit_et0.shape.length, _jit_et0.shape);`,
      `const _jit_eo = _jit_er.data;`,
      `for (let _jit_ei = 0; _jit_ei < _jit_eo.length; _jit_ei++) _jit_eo[_jit_ei] = ${slot};`
    );
    if (isNumeric(k.ty) && k.ty.elem === "logical") {
      body.push(`_jit_er.isLogical = true;`);
    }
    body.push(`${target} = _jit_er;`);
  } else {
    const op = k.reduce.op;
    body.push(
      `const _jit_en = _jit_ex0.length;`,
      `let _jit_eacc = ${op === "prod" ? 1 : 0};`,
      `for (let _jit_ei = 0; _jit_ei < _jit_en; _jit_ei++) _jit_eacc ${op === "prod" ? "*=" : "+="} ${slot};`,
      storeScalar(op === "mean" ? "_jit_eacc / _jit_en" : "_jit_eacc")
    );
  }
  if (k.guard) {
    ctx.useRuntime(FUSED_SHAPE_EQ_SNIPPET);
    const cond = k.inputs
      .slice(1)
      .map((_, i) => `jit_fused_shape_eq(_jit_et${i + 1}, _jit_et0)`)
      .join(" && ");
    const fallback =
      k.reduce === undefined
        ? `${target} = ${ctx.emitExpr(k.tree)};`
        : storeScalar(ctx.emitExpr(k.tree));
    lines.push(`${ind}if (${cond}) {`);
    lines.push(...body.map(l => `${ind}  ${l}`));
    lines.push(`${ind}} else {`, `${ind}  ${fallback}`, `${ind}}`);
  } else {
    lines.push(...body.map(l => `${ind}${l}`));
  }
  lines.push(`${indent}}`);
  return lines.join("\n");
}
