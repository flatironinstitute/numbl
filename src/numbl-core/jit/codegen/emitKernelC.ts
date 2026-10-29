/**
 * C rendering of an `ElemwiseKernel` (built by the kernel pass,
 * `lowering/elemwiseKernels.ts`). An element-wise kernel becomes one
 * loop that allocates only the result:
 *
 *     // c = a .* b + d;     emits:
 *     {
 *       jit_tensor_t _r = jit_tensor_alloc_nd(a.ndim, a.dims);
 *       long _jit_n = 1; for (int _jit_i = 0; _jit_i < _r.ndim; _jit_i++) _jit_n *= _r.dims[_jit_i];
 *       JIT_OMP_PARFOR_N
 *       for (long _jit_i = 0; _jit_i < _jit_n; _jit_i++) _r.real[_jit_i] = ((a.real[_jit_i] * b.real[_jit_i]) + d.real[_jit_i]);
 *       jit_tensor_assign(&c, _r);
 *     }
 *
 * A reducing kernel folds the elements into a scalar accumulator in
 * linear order, the order of the `jit_<op>_all` helpers, so it computes
 * the same bits. The loop locals use the reserved `_jit_` prefix so they
 * cannot shadow a user variable named `n` or `i` that the body reads.
 */

import type { ElemwiseKernel, IRExpr } from "../lowering/ir.js";
import { getBuiltin } from "../builtins/index.js";
import { requireEmitC } from "../builtins/registry.js";
import {
  makeEmitUseRuntime,
  useRuntimeByName,
  useRuntimeInline,
  type InlineSnippet,
  type RuntimeState,
} from "./runtime.js";
import { formatDouble } from "./cHelpers.js";

/** Runtime shape-equality probe guarding a kernel whose operand extents
 *  are not statically proven equal (`ElemwiseKernel.guard`). */
export const FUSED_SHAPE_EQ_SNIPPET: InlineSnippet = {
  name: "jit_fused_shape_eq",
  deps: ["jit_tensor_t"],
  code: `static int jit_fused_shape_eq(jit_tensor_t a, jit_tensor_t b) {
  if (a.ndim != b.ndim) return 0;
  for (int i = 0; i < a.ndim; i++)
    if (a.dims[i] != b.dims[i]) return 0;
  return 1;
}`,
  jsCode: `function jit_fused_shape_eq(a, b) {
  const x = a.shape;
  const y = b.shape;
  if (x.length !== y.length) return false;
  for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false;
  return true;
}`,
};

/** The per-element C expression of a kernel body at index `_jit_i`.
 *  Builtins contribute their scalar `emitC` form. */
function emitBodyC(e: IRExpr, k: ElemwiseKernel, state: RuntimeState): string {
  const emitCall = (name: string, args: IRExpr[]): string => {
    const b = getBuiltin(name);
    if (!b) throw new Error(`emitKernelC internal: no builtin '${name}'`);
    return requireEmitC(b)({
      argsC: args.map(a => emitBodyC(a, k, state)),
      argTypes: args.map(a => a.ty),
      nargout: 1,
      useRuntime: makeEmitUseRuntime(state),
    });
  };
  switch (e.kind) {
    case "NumLit":
      return formatDouble(e.value);
    case "Var":
      return e.cName;
    case "KernelElem":
      return `${k.inputs[e.input].cName}.real[_jit_i]`;
    case "Binary":
      return emitCall(e.builtin, [e.left, e.right]);
    case "Unary":
      return emitCall(e.builtin, [e.operand]);
    case "Call":
      return emitCall(e.name, e.args);
    default:
      throw new Error(`emitKernelC internal: unexpected node '${e.kind}'`);
  }
}

/** The runtime shape check of a guarded kernel: every other operand has
 *  the shape source's extents. */
function guardCond(k: ElemwiseKernel, state: RuntimeState): string {
  useRuntimeInline(state, FUSED_SHAPE_EQ_SNIPPET);
  const src = k.inputs[0].cName;
  return k.inputs
    .slice(1)
    .map(v => `jit_fused_shape_eq(${v.cName}, ${src})`)
    .join(" && ");
}

/** Statement assigning an element-wise kernel to the tensor `target`.
 *  `fallbackStmt` (required iff `k.guard`) evaluates the kernel's source
 *  expression through the broadcasting helpers, which broadcast
 *  compatible shapes and raise on incompatible ones. */
export function emitElemwiseKernelAssign(
  k: ElemwiseKernel,
  target: string,
  indent: string,
  state: RuntimeState,
  fallbackStmt?: string
): string {
  // `JIT_OMP_PARFOR_N` comes with the tensor typedef.
  useRuntimeByName(state, "jit_tensor_alloc_nd");
  useRuntimeByName(state, "jit_tensor_assign");
  const src = k.inputs[0].cName;
  const slot = emitBodyC(k.body, k, state);
  const block = (ind: string): string[] => [
    `${ind}{`,
    `${ind}  jit_tensor_t _r = jit_tensor_alloc_nd(${src}.ndim, ${src}.dims);`,
    `${ind}  long _jit_n = 1;`,
    `${ind}  for (int _jit_i = 0; _jit_i < _r.ndim; _jit_i++) _jit_n *= _r.dims[_jit_i];`,
    `${ind}  JIT_OMP_PARFOR_N`,
    `${ind}  for (long _jit_i = 0; _jit_i < _jit_n; _jit_i++) _r.real[_jit_i] = ${slot};`,
    `${ind}  jit_tensor_assign(&${target}, _r);`,
    `${ind}}`,
  ];
  return wrapGuard(k, block, indent, state, fallbackStmt);
}

/** Statement folding a reducing kernel into `_jit_acc` and then running
 *  `store(value)`, the statement that stores the scalar result. */
export function emitReduceKernel(
  k: ElemwiseKernel,
  store: (value: string) => string,
  indent: string,
  state: RuntimeState,
  fallbackStmt?: string
): string {
  const op = k.reduce!.op;
  const src = k.inputs[0].cName;
  const slot = emitBodyC(k.body, k, state);
  const init = op === "prod" ? "1.0" : "0.0";
  const step = op === "prod" ? "*=" : "+=";
  const value =
    op === "mean" ? "((double)_jit_acc / (double)_jit_n)" : "_jit_acc";
  const block = (ind: string): string[] => [
    `${ind}{`,
    `${ind}  long _jit_n = 1;`,
    `${ind}  for (int _jit_i = 0; _jit_i < ${src}.ndim; _jit_i++) _jit_n *= ${src}.dims[_jit_i];`,
    `${ind}  double _jit_acc = ${init};`,
    `${ind}  for (long _jit_i = 0; _jit_i < _jit_n; _jit_i++) _jit_acc ${step} ${slot};`,
    `${ind}  ${store(value)}`,
    `${ind}}`,
  ];
  return wrapGuard(k, block, indent, state, fallbackStmt);
}

function wrapGuard(
  k: ElemwiseKernel,
  block: (ind: string) => string[],
  indent: string,
  state: RuntimeState,
  fallbackStmt: string | undefined
): string {
  if (!k.guard) return block(indent).join("\n");
  if (fallbackStmt === undefined) {
    throw new Error("emitKernelC internal: guarded kernel without fallback");
  }
  return [
    `${indent}if (${guardCond(k, state)}) {`,
    ...block(indent + "  "),
    `${indent}} else {`,
    `${indent}  ${fallbackStmt}`,
    `${indent}}`,
  ].join("\n");
}
