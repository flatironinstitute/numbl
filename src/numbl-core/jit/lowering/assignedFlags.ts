/**
 * Assigned flags for the C backend's synthetic loop / top-level entries.
 *
 * A synthetic wrapper's outputs include names the body assigns only on a
 * path that may not run (the loop variable and body locals of a zero-trip
 * `for`). C predeclares every output to 0, so writing such an output back
 * would bind a value the interpreter and JS-JIT leave undefined. For each
 * output that is neither a param nor definitely assigned, this pass adds a
 * scalar flag output set to 1 right after every assignment to it; the host
 * skips the writeback when the flag is still 0.
 */

import type { Assign, IRFunc, IRStmt } from "./ir.js";
import { scalarDouble } from "./types.js";

export interface AssignedFlag {
  /** Index of the guarded output in `fn.outputs`. */
  readonly output: number;
  /** Index of its flag output in `fn.outputs`. */
  readonly flag: number;
}

/** Return a copy of `fn` with a flag output per output in `maybeUnassigned`
 *  (or `fn` itself when there is none), plus the output/flag pairing. */
export function addAssignedFlags(
  fn: IRFunc,
  maybeUnassigned: ReadonlySet<string>
): { fn: IRFunc; flags: AssignedFlag[] } {
  const flagCName = new Map<string, string>();
  const flags: AssignedFlag[] = [];
  const outputs = [...fn.outputs];
  const cOutputs = [...fn.cOutputs];
  const outputTypes = [...fn.outputTypes];
  fn.outputs.forEach((name, i) => {
    if (!maybeUnassigned.has(name) || flagCName.has(name)) return;
    const cName = `_mtoc2_asg${flags.length}`;
    flagCName.set(name, cName);
    flags.push({ output: i, flag: outputs.length });
    outputs.push(cName);
    cOutputs.push(cName);
    outputTypes.push(scalarDouble("nonneg"));
  });
  if (flags.length === 0) return { fn, flags };

  const setFlag = (name: string, s: IRStmt): Assign[] => {
    const cName = flagCName.get(name);
    if (cName === undefined) return [];
    return [
      {
        kind: "Assign",
        name: cName,
        cName,
        ty: scalarDouble("nonneg"),
        expr: {
          kind: "NumLit",
          value: 1,
          ty: scalarDouble("positive", 1),
          span: s.span,
        },
        span: s.span,
      },
    ];
  };

  const rewrite = (stmts: IRStmt[]): IRStmt[] => {
    const out: IRStmt[] = [];
    for (const s of stmts) {
      switch (s.kind) {
        case "Assign":
          out.push(s, ...setFlag(s.name, s));
          break;
        case "MultiAssignCall":
          out.push(s);
          for (const o of s.outputs) {
            if (o.binding) out.push(...setFlag(o.binding.name, s));
          }
          break;
        case "IndexStore":
        case "IndexSliceStore":
        case "CellIndexStore":
        case "MemberStore":
          out.push(s, ...setFlag(s.base.name, s));
          break;
        case "If":
          out.push({
            ...s,
            thenBody: rewrite(s.thenBody),
            elseBody: rewrite(s.elseBody),
          });
          break;
        case "While":
          out.push({ ...s, body: rewrite(s.body) });
          break;
        case "For":
          out.push({
            ...s,
            body: [...setFlag(s.varName, s), ...rewrite(s.body)],
          });
          break;
        default:
          out.push(s);
      }
    }
    return out;
  };

  return {
    fn: { ...fn, outputs, cOutputs, outputTypes, body: rewrite(fn.body) },
    flags,
  };
}
