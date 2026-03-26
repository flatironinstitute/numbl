/**
 * String builtins that return cell arrays or have complex output modes:
 * strsplit, strjoin, regexp, regexpi, regexprep, symvar.
 */

import { FloatXArray, isRuntimeCell } from "../../runtime/types.js";
import type { RuntimeValue } from "../../runtime/types.js";
import { RTV, RuntimeError } from "../../runtime/index.js";
import { toString } from "../../runtime/convert.js";
import { toNumber } from "../../runtime/convert.js";
import type { JitType } from "../jit/jitTypes.js";
import { registerIBuiltin } from "./types.js";
import { parseMFile, type Stmt, type Expr } from "../../parser/index.js";
import { BUILTIN_CONSTANTS } from "../../lowering/constants.js";
import { getIBuiltin } from "./types.js";

// ── strsplit ────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "strsplit",
  resolve: argTypes => {
    if (argTypes.length < 1 || argTypes.length > 2) return null;
    return {
      outputTypes: [{ kind: "unknown" }],
      apply: args => {
        const s = toString(args[0]);
        let parts: string[];
        if (args.length < 2) {
          const trimmed = s.trim();
          parts = trimmed.length === 0 ? [""] : trimmed.split(/[ \f\n\r\t\v]+/);
        } else {
          let delims: string[];
          if (isRuntimeCell(args[1])) {
            delims = args[1].data.map(d => toString(d));
          } else {
            delims = [toString(args[1])];
          }
          const escaped = delims
            .map(d => d.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
            .join("|");
          parts = s.split(new RegExp("(?:" + escaped + ")+"));
        }
        return RTV.cell(
          parts.map(p => RTV.string(p)),
          [1, parts.length]
        );
      },
    };
  },
});

// ── strjoin ─────────────────────────────────────────────────────────────

registerIBuiltin({
  name: "strjoin",
  resolve: argTypes => {
    if (argTypes.length < 1 || argTypes.length > 2) return null;
    return {
      outputTypes: [{ kind: "string" }],
      apply: args => {
        if (!isRuntimeCell(args[0]))
          throw new RuntimeError(
            "strjoin: first argument must be a cell array"
          );
        const elements = args[0].data.map(v => toString(v));
        const delim = args.length >= 2 ? toString(args[1]) : " ";
        return RTV.string(elements.join(delim));
      },
    };
  },
});

// ── regexp/regexpi shared implementation ────────────────────────────────

function regexpImpl(
  caseSensitive: boolean,
  name: string,
  args: RuntimeValue[],
  nargout: number
): RuntimeValue | RuntimeValue[] {
  if (args.length < 2)
    throw new RuntimeError(`${name} requires at least 2 arguments`);
  const str = toString(args[0]);
  const pat = toString(args[1]);

  let matchOnce = false;
  const outModes: string[] = [];
  for (let i = 2; i < args.length; i++) {
    const opt = toString(args[i]).toLowerCase();
    if (opt === "once") matchOnce = true;
    else outModes.push(opt);
  }
  if (outModes.length === 0 && nargout <= 1) outModes.push("start");
  if (outModes.length === 0) {
    outModes.push("start", "end", "tokenextents", "match", "tokens", "names");
  }

  const flags = caseSensitive ? "g" : "gi";
  const re = new RegExp(pat, flags);
  const starts: number[] = [];
  const ends: number[] = [];
  const matches: string[] = [];
  const tokensList: string[][] = [];
  let m: RegExpExecArray | null;
  while ((m = re.exec(str)) !== null) {
    starts.push(m.index + 1);
    ends.push(m.index + m[0].length);
    matches.push(m[0]);
    const toks: string[] = [];
    for (let g = 1; g < m.length; g++) toks.push(m[g] ?? "");
    tokensList.push(toks);
    if (matchOnce) break;
    if (m[0].length === 0) re.lastIndex++;
  }

  function buildOutput(mode: string): RuntimeValue {
    if (mode === "start") {
      if (matchOnce)
        return starts.length > 0
          ? RTV.num(starts[0])
          : RTV.tensor(new FloatXArray(0), [1, 0]);
      return RTV.tensor(new FloatXArray(starts), [1, starts.length]);
    }
    if (mode === "end") {
      if (matchOnce)
        return ends.length > 0
          ? RTV.num(ends[0])
          : RTV.tensor(new FloatXArray(0), [1, 0]);
      return RTV.tensor(new FloatXArray(ends), [1, ends.length]);
    }
    if (mode === "match") {
      if (matchOnce)
        return matches.length > 0 ? RTV.char(matches[0]) : RTV.char("");
      return RTV.cell(
        matches.map(s => RTV.char(s)),
        [1, matches.length]
      );
    }
    if (mode === "tokens") {
      if (matchOnce) {
        if (tokensList.length === 0) return RTV.cell([], [1, 0]);
        return RTV.cell(
          tokensList[0].map(s => RTV.char(s)),
          [1, tokensList[0].length]
        );
      }
      return RTV.cell(
        tokensList.map(toks =>
          RTV.cell(
            toks.map(s => RTV.char(s)),
            [1, toks.length]
          )
        ),
        [1, tokensList.length]
      );
    }
    if (mode === "names") {
      if (starts.length === 0) return RTV.tensor(new FloatXArray(0), [0, 0]);
      const namedRe = new RegExp(pat, flags);
      const namedMatches: Record<string, string>[] = [];
      let nm: RegExpExecArray | null;
      while ((nm = namedRe.exec(str)) !== null) {
        namedMatches.push(
          Object.fromEntries(
            Object.entries(nm.groups ?? {}).map(([k, v]) => [k, v ?? ""])
          )
        );
        if (matchOnce) break;
        if (nm[0].length === 0) namedRe.lastIndex++;
      }
      if (matchOnce) {
        const fields: Record<string, RuntimeValue> = {};
        for (const [k, v] of Object.entries(namedMatches[0] ?? {})) {
          fields[k] = RTV.char(v);
        }
        return RTV.struct(fields);
      }
      if (namedMatches.length === 0)
        return RTV.tensor(new FloatXArray(0), [0, 0]);
      const keys = Object.keys(namedMatches[0]);
      const elements = namedMatches.map(nm2 => {
        const fields: Record<string, RuntimeValue> = {};
        for (const k of keys) fields[k] = RTV.char(nm2[k] ?? "");
        return RTV.struct(fields);
      });
      return RTV.structArray(keys, elements);
    }
    // tokenextents
    if (matchOnce) return RTV.tensor(new FloatXArray(0), [1, 0]);
    return RTV.tensor(new FloatXArray(0), [1, 0]);
  }

  if (outModes.length === 1) return buildOutput(outModes[0]);
  return outModes.map(m => buildOutput(m)).slice(0, nargout);
}

registerIBuiltin({
  name: "regexp",
  resolve: (argTypes, nargout) => {
    if (argTypes.length < 2) return null;
    const outTypes: JitType[] = [];
    for (let i = 0; i < Math.max(nargout, 1); i++)
      outTypes.push({ kind: "unknown" });
    return {
      outputTypes: outTypes,
      apply: (args, nargout) => regexpImpl(true, "regexp", args, nargout),
    };
  },
});

registerIBuiltin({
  name: "regexpi",
  resolve: (argTypes, nargout) => {
    if (argTypes.length < 2) return null;
    const outTypes: JitType[] = [];
    for (let i = 0; i < Math.max(nargout, 1); i++)
      outTypes.push({ kind: "unknown" });
    return {
      outputTypes: outTypes,
      apply: (args, nargout) => regexpImpl(false, "regexpi", args, nargout),
    };
  },
});

// ── symvar ──────────────────────────────────────────────────────────────

function walkExpr(e: Expr, found: Set<string>): void {
  switch (e.type) {
    case "Ident":
      found.add(e.name);
      break;
    case "Binary":
      walkExpr(e.left, found);
      walkExpr(e.right, found);
      break;
    case "Unary":
      walkExpr(e.operand, found);
      break;
    case "FuncCall":
      found.add(e.name);
      for (const arg of e.args) walkExpr(arg, found);
      break;
    case "Index":
      walkExpr(e.base, found);
      for (const idx of e.indices) walkExpr(idx, found);
      break;
    case "Member":
      walkExpr(e.base, found);
      break;
    case "MemberDynamic":
      walkExpr(e.base, found);
      walkExpr(e.nameExpr, found);
      break;
    case "MethodCall":
      walkExpr(e.base, found);
      for (const arg of e.args) walkExpr(arg, found);
      break;
    case "AnonFunc":
      break;
    case "Tensor":
    case "Cell":
      for (const row of e.rows) for (const el of row) walkExpr(el, found);
      break;
    case "ClassInstantiation":
      for (const arg of e.args) walkExpr(arg, found);
      break;
    case "SuperMethodCall":
      for (const arg of e.args) walkExpr(arg, found);
      break;
    default:
      break;
  }
}

function walkStmt(s: Stmt, found: Set<string>): void {
  switch (s.type) {
    case "Assign":
      walkExpr(s.expr, found);
      break;
    case "ExprStmt":
      walkExpr(s.expr, found);
      break;
    default:
      break;
  }
}

registerIBuiltin({
  name: "symvar",
  resolve: argTypes => {
    if (argTypes.length < 1 || argTypes.length > 2) return null;
    return {
      outputTypes: [{ kind: "unknown" }],
      apply: args => {
        const expr = toString(args[0]);
        let ast;
        try {
          ast = parseMFile(`__symvar_out__ = ${expr};`, "__symvar__.m");
        } catch {
          return RTV.cell([], [1, 0]);
        }
        const found = new Set<string>();
        for (const stmt of ast.body) walkStmt(stmt, found);
        found.delete("__symvar_out__");
        for (const name of found) {
          if (BUILTIN_CONSTANTS.has(name) || getIBuiltin(name)) {
            found.delete(name);
          }
        }
        const sorted = [...found].sort((a, b) => {
          const aUpper = a[0] >= "A" && a[0] <= "Z";
          const bUpper = b[0] >= "A" && b[0] <= "Z";
          if (aUpper && !bUpper) return -1;
          if (!aUpper && bUpper) return 1;
          return a < b ? -1 : a > b ? 1 : 0;
        });
        let result = sorted;
        if (args.length === 2) {
          const n = Math.round(toNumber(args[1]));
          const byDist = [...sorted].sort((a, b) => {
            const distA = Math.abs(a.charCodeAt(0) - "x".charCodeAt(0));
            const distB = Math.abs(b.charCodeAt(0) - "x".charCodeAt(0));
            if (distA !== distB) return distA - distB;
            return a < b ? -1 : a > b ? 1 : 0;
          });
          const selected = new Set(byDist.slice(0, n));
          result = sorted.filter(v => selected.has(v));
        }
        return RTV.cell(
          result.map(v => RTV.char(v)),
          [1, result.length]
        );
      },
    };
  },
});
