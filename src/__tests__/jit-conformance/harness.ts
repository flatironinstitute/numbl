/**
 * Per-builtin conformance harness: runs each case of a builtin's table at
 * `--opt 0` (the reference) and at a JIT level, in process, and compares
 * the results exactly.
 *
 * Each case becomes a tiny kernel function
 *
 *     function [y1, ...] = k3(a1, ...)
 *     %!numbl:assert_jit c
 *     [y1, ...] = name(a1, 2, ...);
 *     end
 *
 * called from a script whose top level stays on the interpreter (it ends
 * in `return`), so the call executor compiles the kernel on its own. The
 * `assert_jit c` directive is a no-op in compiled code; reaching it on the
 * interpreter (a decline, a bail or a re-run) raises, so a case that fell
 * back to the interpreter is reported as declined and never as a pass.
 *
 * At `--opt 2` the cases are batched into one kernel per builtin to
 * amortize the C compile; a batch that declines or errors is split in
 * halves, down to single cases (`runBatched`).
 */

import { writeFileSync } from "node:fs";
import { executeCode } from "../../numbl-core/executeCode.js";
import type { OptLevel } from "../../numbl-core/executors/plugins.js";
import type { NativeBridge } from "../../numbl-core/workspace/types.js";
import type { RuntimeValue } from "../../numbl-core/runtime/index.js";

/** An argument: a MATLAB expression passed as a parameter (the JIT sees
 *  its type, not its value), or `{ lit }`, spliced into the kernel source
 *  as a constant (dims, flags, option strings). */
export type Arg = string | { lit: string };

export interface Case {
  args: Arg[];
  /** Outputs requested (default 1). */
  nargout?: number;
  /** Kernel body template replacing `[y1, ...] = name(args);`: `$1`, `$2`
   *  are the arguments and `y1`, `y2` the outputs. */
  body?: string;
  /** A script statement run on the interpreter before the call (e.g.
   *  `rng(3)`). Such a case is never batched. */
  pre?: string;
}

export type Outcome =
  | { kind: "value"; values: string[]; output: string }
  | { kind: "error"; error: string; output: string }
  | { kind: "declined"; reason: string };

const ASSERT_TAG = "%!numbl:assert_jit";

/** Stable id of a case, for reports and the known-failure list. */
export function caseId(name: string, c: Case): string {
  const args = c.args.map(a => (typeof a === "string" ? a : a.lit));
  const shown =
    c.body === undefined
      ? `${name}(${args.join(", ")})`
      : `${c.body.replace(/\$(\d+)/g, (_, k) => args[Number(k) - 1])}`;
  const n = c.nargout ?? 1;
  return (n === 1 ? "" : `[${n} out] `) + (c.pre ? `${c.pre}; ` : "") + shown;
}

// ── Script generation ──────────────────────────────────────────────────

interface Rendered {
  /** Parameter names (`a<i>_<k>`) in kernel order, with their script
   *  variable (`p<i>_<k>`). */
  params: string[];
  /** Body statements, outputs renamed to `y<i>_<j>`. */
  body: string;
  outs: string[];
}

function render(name: string, c: Case, i: number): Rendered {
  const params: string[] = [];
  const argText = c.args.map((a, k) => {
    if (typeof a !== "string") return a.lit;
    params.push(`a${i}_${k + 1}`);
    return `a${i}_${k + 1}`;
  });
  const n = c.nargout ?? 1;
  const outs = Array.from({ length: n }, (_, j) => `y${i}_${j + 1}`);
  let body: string;
  if (c.body !== undefined) {
    body = c.body
      .replace(/\$(\d+)/g, (_, k) => argText[Number(k) - 1])
      .replace(/\by(\d+)\b/g, (_, j) => `y${i}_${j}`);
  } else {
    const lhs = n === 1 ? outs[0] : `[${outs.join(", ")}]`;
    body = `${lhs} = ${name}(${argText.join(", ")});`;
  }
  return { params, body, outs };
}

function scriptParams(c: Case, i: number): string[] {
  const ps: string[] = [];
  c.args.forEach((a, k) => {
    if (typeof a === "string") ps.push(`p${i}_${k + 1}`);
  });
  return ps;
}

function prelude(cases: Case[]): string {
  const lines: string[] = [];
  cases.forEach((c, i) =>
    c.args.forEach((a, k) => {
      if (typeof a === "string") lines.push(`p${i}_${k + 1} = ${a};`);
    })
  );
  return lines.join("\n");
}

const outVars = (r: Rendered) => r.outs.map(o => "o" + o.slice(1));

/** The script-side call of one case's own kernel `k<i>`. */
function singleCall(c: Case, i: number, r: Rendered): string {
  return [
    `disp('@@${i}');`,
    c.pre ? `${c.pre};` : "",
    `try`,
    `  [${outVars(r).join(", ")}] = k${i}(${scriptParams(c, i).join(", ")});`,
    `catch err`,
    `  e${i} = [err.identifier '|' err.message];`,
    `end`,
  ].join("\n");
}

function kernel(fname: string, rs: Rendered[], markers: number[] | null) {
  const outs = rs.flatMap(r => r.outs);
  const params = rs.flatMap(r => r.params);
  const body = rs.map((r, k) =>
    markers ? `disp('@@${markers[k]}');\n${r.body}` : r.body
  );
  return [
    `function [${outs.join(", ")}] = ${fname}(${params.join(", ")})`,
    `${ASSERT_TAG} c`,
    ...body,
    `end`,
  ].join("\n");
}

// ── Running ────────────────────────────────────────────────────────────

function run(
  source: string,
  opt: OptLevel,
  nativeBridge: NativeBridge | undefined
): { vars: Record<string, RuntimeValue>; output: string } {
  // Debug aid: the last generated script, to rerun with the CLI.
  if (process.env.NUMBL_CONFORMANCE_DUMP)
    writeFileSync(process.env.NUMBL_CONFORMANCE_DUMP, source);
  const out: string[] = [];
  const r = executeCode(
    source,
    { optimization: opt, onOutput: t => out.push(t) },
    [],
    "conformance.m",
    undefined,
    nativeBridge
  );
  return { vars: r.variableValues, output: out.join("") };
}

/** Last output segment printed after each `@@<i>` marker (a marker need
 *  not start a line: the previous case may not end its output with one). */
function segments(output: string): Map<number, string> {
  const segs = new Map<number, string>();
  const parts = output.split(/@@(\d+)\n/);
  for (let k = 1; k < parts.length; k += 2)
    segs.set(Number(parts[k]), parts[k + 1]);
  return segs;
}

function outcomes(
  cases: Case[],
  rendered: Rendered[],
  vars: Record<string, RuntimeValue>,
  output: string
): Outcome[] {
  const segs = segments(output);
  return cases.map((_, i) => {
    const out = segs.get(i) ?? "";
    const e = vars[`e${i}`];
    if (e !== undefined) {
      const msg = canon(e);
      const at = msg.indexOf(ASSERT_TAG);
      if (at >= 0) {
        const why = msg.indexOf("Most recent JIT decline");
        return {
          kind: "declined",
          reason:
            why < 0
              ? "unknown"
              : msg
                  .slice(why + 24)
                  .replace(/ \(Run with --opt 0 to silence\.\)"$/, "")
                  .slice(0, 200),
        };
      }
      return { kind: "error", error: msg, output: out };
    }
    const values = outVars(rendered[i]).map(v =>
      v in vars ? canon(vars[v]) : "<unset>"
    );
    return { kind: "value", values, output: out };
  });
}

/** Run every case with its own kernel. */
export function runSingle(
  name: string,
  cases: Case[],
  opt: OptLevel,
  nativeBridge?: NativeBridge
): Outcome[] {
  const rendered = cases.map((c, i) => render(name, c, i));
  const source = [
    prelude(cases),
    ...cases.map((c, i) => singleCall(c, i, rendered[i])),
    "return",
    ...rendered.map((r, i) => kernel(`k${i}`, [r], null)),
  ].join("\n");
  const { vars, output } = run(source, opt, nativeBridge);
  return outcomes(cases, rendered, vars, output);
}

/** Run the cases in `batchable` together, bisecting a batch that fails,
 *  and the rest with their own kernels. */
export function runBatched(
  name: string,
  cases: Case[],
  batchable: boolean[],
  opt: OptLevel,
  nativeBridge?: NativeBridge
): Outcome[] {
  const rendered = cases.map((c, i) => render(name, c, i));
  const kernels: string[] = [];
  const gen = (ids: number[]): string => {
    if (ids.length === 1) {
      const i = ids[0];
      kernels.push(kernel(`k${i}`, [rendered[i]], null));
      return singleCall(cases[i], i, rendered[i]);
    }
    const fname = `kb${ids[0]}_${ids[ids.length - 1]}_${ids.length}`;
    const rs = ids.map(i => rendered[i]);
    kernels.push(kernel(fname, rs, ids));
    const outs = rs.flatMap(outVars);
    const params = ids.flatMap(i => scriptParams(cases[i], i));
    const mid = ids.length >> 1;
    return [
      `try`,
      `  [${outs.join(", ")}] = ${fname}(${params.join(", ")});`,
      `catch err`,
      gen(ids.slice(0, mid)),
      gen(ids.slice(mid)),
      `end`,
    ].join("\n");
  };
  const batch = cases.map((_, i) => i).filter(i => batchable[i]);
  const rest = cases.map((_, i) => i).filter(i => !batchable[i]);
  const calls = [
    ...(batch.length > 0 ? [gen(batch)] : []),
    ...rest.map(i => gen([i])),
  ];
  const source = [prelude(cases), ...calls, "return", ...kernels].join("\n");
  const { vars, output } = run(source, opt, nativeBridge);
  return outcomes(cases, rendered, vars, output);
}

// ── Canonical value form ───────────────────────────────────────────────

/** Exact text of a double: shortest round-trip form, which distinguishes
 *  every value; NaN is one value whatever its payload, and -0 is not 0. */
function num(x: number): string {
  if (Object.is(x, -0)) return "-0";
  return String(x);
}

function shapeText(shape: number[]): string {
  const s = shape.length < 2 ? [1, ...shape] : shape.slice();
  while (s.length > 2 && s[s.length - 1] === 1) s.pop();
  return s.join("x");
}

/** A string that two values share iff they are the same MATLAB value:
 *  class, shape, complexness and every element. A 1x1 tensor and a scalar
 *  of the same class are the same value. */
export function canon(v: RuntimeValue): string {
  if (typeof v === "number") return `double[1x1]:${num(v)}`;
  if (typeof v === "boolean") return `logical[1x1]:${v ? 1 : 0}`;
  if (typeof v === "string") return `string:${JSON.stringify(v)}`;
  switch (v.kind) {
    case "tensor": {
      const cls = v.cls;
      const re = Array.from(v.data, num).join(",");
      const im = v.imag ? ` im:${Array.from(v.imag, num).join(",")}` : "";
      return `${cls}[${shapeText(v.shape)}]${v.imag ? "c" : ""}:${re}${im}`;
    }
    case "complex_number":
      return `double[1x1]c:${num(v.re)} im:${num(v.im)}`;
    case "char":
      return `char[${shapeText(v.shape ?? [1, v.value.length])}]:${JSON.stringify(v.value)}`;
    case "cell":
      return `cell[${shapeText(v.shape)}]{${v.data.map(canon).join("; ")}}`;
    case "struct":
      return `struct{${[...v.fields].map(([k, f]) => `${k}=${canon(f)}`).join("; ")}}`;
    default:
      return `${v.kind}`;
  }
}

// ── Known divergence classes ───────────────────────────────────────────

/** A way a JIT result is known to differ from the interpreter's (see
 *  `KNOWN_DIVERGENCES` in cases.ts). */
export type DivergenceKind =
  /** A complex result whose imaginary parts are all zero, where the
   *  interpreter (and MATLAB) narrow it to real. */
  | { kind: "zero-imag" }
  /** A double result where the interpreter (and MATLAB) keep logical. */
  | { kind: "logical-class" }
  /** Elements that differ in the last few bits or in the sign of a zero:
   *  a different floating-point algorithm or libm. */
  | { kind: "ulp"; ulps: number; ids?: string[] }
  /** Any difference, for the listed case ids only. */
  | { kind: "case"; ids: string[] };

interface Numeric {
  cls: string;
  shape: string;
  re: number[];
  im: number[] | null;
}

function parseNumeric(v: string): Numeric | null {
  const m = /^(double|logical)\[([\dx]+)\](c?):(.*)$/s.exec(v);
  if (!m) return null;
  const nums = (s: string) => (s === "" ? [] : s.split(",").map(Number));
  const [re, im] = m[3] ? m[4].split(" im:") : [m[4], null];
  return {
    cls: m[1],
    shape: m[2],
    re: nums(re),
    im: im === null ? null : nums(im),
  };
}

const f64 = new Float64Array(1);
const i64 = new BigInt64Array(f64.buffer);
/** Position of `x` among the doubles (±0 both map to 0). */
function ordinal(x: number): bigint {
  f64[0] = x;
  const b = i64[0];
  return b < 0n ? -(b & 0x7fffffffffffffffn) : b;
}

function withinUlps(a: number[], b: number[], ulps: number): boolean {
  return (
    a.length === b.length &&
    a.every((x, k) => {
      const y = b[k];
      if (Number.isNaN(x) || Number.isNaN(y))
        return Number.isNaN(x) && Number.isNaN(y);
      const d = ordinal(x) - ordinal(y);
      return (d < 0n ? -d : d) <= BigInt(ulps);
    })
  );
}

const sameNums = (a: number[], b: number[]) =>
  a.length === b.length &&
  a.every(
    (x, k) => Object.is(x, b[k]) || (Number.isNaN(x) && Number.isNaN(b[k]))
  );

/** Whether `got` differs from `ref` only in the way `d` describes. */
export function explains(
  d: DivergenceKind,
  id: string,
  ref: Outcome,
  got: Outcome
): boolean {
  if (d.kind === "case") return d.ids.includes(id);
  if (d.kind === "ulp" && d.ids && !d.ids.includes(id)) return false;
  if (ref.kind !== "value" || got.kind !== "value") return false;
  if (ref.output !== got.output || ref.values.length !== got.values.length)
    return false;
  return ref.values.every((rv, k) => {
    const gv = got.values[k];
    if (rv === gv) return true;
    const r = parseNumeric(rv);
    const g = parseNumeric(gv);
    if (!r || !g || r.shape !== g.shape) return false;
    switch (d.kind) {
      case "zero-imag":
        return (
          r.cls === g.cls &&
          r.im === null &&
          g.im !== null &&
          g.im.every(x => x === 0) &&
          sameNums(r.re, g.re)
        );
      case "logical-class":
        return (
          r.cls === "logical" &&
          g.cls === "double" &&
          r.im === null &&
          g.im === null &&
          sameNums(r.re, g.re)
        );
      case "ulp":
        return (
          r.cls === g.cls &&
          (r.im === null) === (g.im === null) &&
          withinUlps(r.re, g.re, d.ulps) &&
          (r.im === null || withinUlps(r.im, g.im!, d.ulps))
        );
    }
  });
}
