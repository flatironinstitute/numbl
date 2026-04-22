/**
 * Hybrid JS/C-JIT compilation: when the outer body falls back to JS-JIT,
 * still try C-JIT at narrower scopes and splice the native handles in.
 * Two shapes:
 *
 *   - `compileHybridCallees`: walks `generatedIRBodies` (one entry per
 *     user-function specialization reached by the outer) and tries
 *     C-JIT on each standalone. Successes get installed on
 *     `rt.jitHelpers[$cjit_<jitName>]` and their cached JS source is
 *     rewritten to `var <jitName> = $h.$cjit_<jitName>;` so the outer
 *     JS transparently calls native code at the callee boundary.
 *
 *   - `compileHybridLoops`: walks the outer's top-level JitStmt body,
 *     finds `For`/`While` stmts whose body is C-feasible, extracts each
 *     as a synthetic specialization (live-in vars → params, live-out
 *     vars → outputs), compiles it to native, and replaces the loop
 *     stmt in the outer's IR with a `UserCallWriteback` that invokes
 *     the handle and writes back outputs into the outer's locals.
 *
 * Preconditions at call time:
 *   - `interp.optimization >= 2`
 *   - `interp.rt.jitHelpers` is populated (i.e. past executeCode init)
 *   - the shared `generatedFns` / `generatedIRBodies` maps come from the
 *     outer specialization's `LoweringResult`
 *
 * Calling convention matches: `$h.callUser($rt, name, fn, ...args)`
 * invokes `fn(...args)`, and the args are the JIT's emitted JS forms —
 * raw JS numbers for scalars, RuntimeTensor for tensors, etc. — same
 * shape the C-JIT wrapper expects since it was compiled with matching
 * `argTypes`.
 */

import type { Interpreter } from "../../interpreter/interpreter.js";
import type { FunctionDef } from "../../interpreter/types.js";
import type { GeneratedFn } from "../jitLower.js";
import type { JitExpr, JitStmt, JitType } from "../jitTypes.js";
import type { TypeEnv } from "../jitLowerTypes.js";
import { jitTypeKey } from "../jitTypes.js";
import { getCJitBackend } from "./registry.js";
import { walkExprNodes, walkStmts, walkStmtExprs } from "./visit.js";

/**
 * True if the callee's body does enough work per call to amortize the
 * JS→C N-API crossing cost (~hundreds of ns per call with buffer setup).
 *
 * Heuristic: the body (or any nested body) must contain at least one
 * For or While loop. Tiny straight-line callees like `y = x*x` called
 * inside a 10M-iteration outer loop lose catastrophically to FFI
 * overhead; requiring a loop in the callee makes the native body do
 * batch work per call.
 */
function bodyWorthCrossing(body: JitStmt[]): boolean {
  for (const s of body) {
    if (s.tag === "For" || s.tag === "While") return true;
    if (s.tag === "If") {
      if (bodyWorthCrossing(s.thenBody)) return true;
      for (const eib of s.elseifBlocks) {
        if (bodyWorthCrossing(eib.body)) return true;
      }
      if (s.elseBody && bodyWorthCrossing(s.elseBody)) return true;
    }
  }
  return false;
}

export function compileHybridCallees(
  interp: Interpreter,
  generatedIRBodies: Map<string, GeneratedFn>,
  generatedFns: Map<string, string>
): void {
  if (interp.optimization < 2) return;
  const backend = getCJitBackend();
  if (!backend) return;
  const helpers = interp.rt.jitHelpers;
  if (!helpers) return;

  for (const [jitName, gf] of generatedIRBodies) {
    const helperKey = `$cjit_${jitName}`;
    if (helpers[helperKey]) {
      generatedFns.set(jitName, `var ${jitName} = $h.${helperKey};`);
      continue;
    }
    if (!bodyWorthCrossing(gf.body)) continue;

    const res = backend.tryCompile(
      interp,
      gf.fn,
      gf.body,
      gf.outputNames,
      gf.localVars,
      gf.outputTypes[0] ?? null,
      gf.outputTypes,
      gf.argTypes,
      gf.nargout,
      generatedIRBodies
    );
    if (!res.ok) continue;
    helpers[helperKey] = res.fn;
    generatedFns.set(jitName, `var ${jitName} = $h.${helperKey};`);
  }
}

// ── Hybrid-loop extraction ──────────────────────────────────────────────

interface VarRefs {
  reads: Set<string>;
  writes: Set<string>;
  /** Early-exit control flow: Return / Break / Continue in the body.
   *  If true, the loop isn't safe to extract as its own function because
   *  a Return would change semantics (outer fn exits, not the loop). */
  hasControl: boolean;
  /** Set to true on any JitStmt / JitExpr kind the walker doesn't handle.
   *  When true, we conservatively skip the extraction to avoid missing a
   *  read/write that would cause stale-var writebacks or type errors. */
  hasUnknown: boolean;
}

function emptyRefs(): VarRefs {
  return {
    reads: new Set(),
    writes: new Set(),
    hasControl: false,
    hasUnknown: false,
  };
}

/** Walk an expression subtree and accumulate reads into `out`. Uses the
 *  shared `walkExprNodes` for the recursion; the per-node callback
 *  handles the read-adding logic for `Var`, `RangeSliceRead`,
 *  `MemberRead`, and `StructArrayMemberRead` (which refer to vars via
 *  non-expression fields that the generic walker can't see), and marks
 *  `hasUnknown` on any tag the hybrid pass doesn't recognize. */
function walkExpr(e: JitExpr, out: VarRefs): void {
  walkExprNodes(e, node => {
    switch (node.tag) {
      case "Var":
        out.reads.add(node.name);
        return;
      case "RangeSliceRead":
      case "MemberRead":
        out.reads.add(node.baseName);
        return;
      case "StructArrayMemberRead":
        out.reads.add(node.structVarName);
        return;
      case "NumberLiteral":
      case "ImagLiteral":
      case "StringLiteral":
      case "Binary":
      case "Unary":
      case "Call":
      case "UserCall":
      case "FuncHandleCall":
      case "UserDispatchCall":
      case "Index":
      case "TensorLiteral":
      case "VConcatGrow":
        return;
      default:
        out.hasUnknown = true;
        return;
    }
  });
}

/** Walk a statement subtree, accumulating reads/writes/control-flow
 *  flags into `out`. Uses `walkStmts` for descent into nested bodies
 *  and `walkStmtExprs` for each stmt's top-level expressions, then
 *  `walkExpr` for the expression contents. */
function walkStmt(s: JitStmt, out: VarRefs): void {
  walkStmts([s], st => {
    switch (st.tag) {
      case "Assign":
        out.writes.add(st.name);
        break;
      case "AssignIndex":
      case "AssignIndexPage3d":
      case "AssignMember":
        out.writes.add(st.baseName);
        out.reads.add(st.baseName);
        break;
      case "AssignIndexRange":
      case "AssignIndexCol":
        out.writes.add(st.baseName);
        out.reads.add(st.baseName);
        out.reads.add(st.srcBaseName);
        break;
      case "For":
        out.writes.add(st.varName);
        break;
      case "MultiAssign":
        for (const n of st.names) if (n !== null) out.writes.add(n);
        break;
      case "UserCallWriteback":
        for (const o of st.outputs) out.writes.add(o);
        break;
      case "Return":
      case "Break":
      case "Continue":
        out.hasControl = true;
        break;
      case "ExprStmt":
      case "If":
      case "While":
      case "SetLoc":
      case "AssertCJit":
        break;
      default:
        out.hasUnknown = true;
        break;
    }
    walkStmtExprs(st, e => walkExpr(e, out));
  });
}

/**
 * Walk the body in source order and return the set of variable names
 * that are read BEFORE being unconditionally written on that iteration
 * — i.e. genuine loop-carried reads that require a live-in.
 *
 * Only a plain top-level `Assign` (no `.name` self-read) counts as an
 * unconditional pre-write. Conditional branches, index writes, and
 * nested loops are not pre-writes. This is a sound under-approximation:
 * false-negative pre-writes mean a var is treated as live-in (extra
 * arg but still correct); false-positive pre-writes would drop a real
 * live-in and break semantics, so we're careful not to claim a write
 * is unconditional unless we're sure.
 */
function genuineLoopReads(stmts: JitStmt[]): Set<string> {
  const preWritten = new Set<string>();
  const genuine = new Set<string>();
  for (const s of stmts) {
    const r = emptyRefs();
    walkStmt(s, r);
    for (const n of r.reads) {
      if (!preWritten.has(n)) genuine.add(n);
    }
    if (s.tag === "Assign" && !r.reads.has(s.name)) {
      preWritten.add(s.name);
    }
  }
  return genuine;
}

function hashStr(s: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = (h * 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, "0");
}

/**
 * Extract top-level For/While loops from the outer's IR and try to
 * compile each to native. On success the loop is replaced in-place by a
 * `UserCallWriteback` stmt.
 */
export function compileHybridLoops(
  interp: Interpreter,
  outerBody: JitStmt[],
  outerEndEnv: TypeEnv,
  outerOutputNames: string[],
  generatedIRBodies: Map<string, GeneratedFn>,
  generatedFns: Map<string, string>
): void {
  if (interp.optimization < 2) return;
  const backend = getCJitBackend();
  if (!backend) return;
  const helpers = interp.rt.jitHelpers;
  if (!helpers) return;

  // Precompute reads of every suffix of the outer body so we can check
  // live-out in O(n) instead of O(n^2).
  const suffixReads: Set<string>[] = new Array(outerBody.length + 1);
  suffixReads[outerBody.length] = new Set(outerOutputNames);
  for (let i = outerBody.length - 1; i >= 0; i--) {
    const r = emptyRefs();
    walkStmt(outerBody[i], r);
    suffixReads[i] = new Set(suffixReads[i + 1]);
    for (const n of r.reads) suffixReads[i].add(n);
  }

  // Precompute, for each outer stmt index, the set of variables that
  // are definitely defined (have a real value) at that point. A var is
  // defined if it's an outer param (no outer write ever touches it) or
  // the previous outer stmt was a plain `Assign` that unconditionally
  // wrote it. Conditional / control-flow writes don't count.
  //
  // The hybrid-loop extraction consults this: a variable can only be
  // a live-in if it's defined at the extraction point — otherwise the
  // call site would pass `undefined`.
  const outerWrites = new Set<string>();
  for (const s of outerBody) {
    const r = emptyRefs();
    walkStmt(s, r);
    for (const n of r.writes) outerWrites.add(n);
  }
  const definedBefore: Set<string>[] = new Array(outerBody.length + 1);
  const paramsDef = new Set<string>();
  for (const n of outerEndEnv.keys()) {
    if (!outerWrites.has(n)) paramsDef.add(n);
  }
  definedBefore[0] = paramsDef;
  for (let i = 0; i < outerBody.length; i++) {
    const s = outerBody[i];
    const next = new Set(definedBefore[i]);
    if (s.tag === "Assign") {
      const r = emptyRefs();
      walkExpr(s.expr, r);
      if (!r.reads.has(s.name)) next.add(s.name);
    }
    definedBefore[i + 1] = next;
  }

  for (let i = 0; i < outerBody.length; i++) {
    const stmt = outerBody[i];
    if (stmt.tag !== "For" && stmt.tag !== "While") continue;

    const refs = emptyRefs();
    walkStmt(stmt, refs);
    if (refs.hasControl || refs.hasUnknown) continue;

    // Live-out: loop var + any write that's read after the loop (or is
    // an outer fn output). Vars that are locally-scoped to the loop
    // (not referenced anywhere after) stay inside the synthetic fn.
    const later = suffixReads[i + 1];
    const liveOutNames: string[] = [];
    if (stmt.tag === "For") liveOutNames.push(stmt.varName);
    for (const n of refs.writes) {
      if (liveOutNames.includes(n)) continue;
      if (stmt.tag === "For" && n === stmt.varName) continue;
      if (later.has(n)) liveOutNames.push(n);
    }
    if (liveOutNames.length === 0) continue;

    const liveOutTypes: JitType[] = liveOutNames.map(
      n => outerEndEnv.get(n) ?? { kind: "unknown" as const }
    );
    if (liveOutTypes.some(t => t.kind === "unknown")) continue;

    // Live-in: vars genuinely read before any pre-write on this
    // iteration, plus vars written by the loop that are ALSO liveOut
    // (the loop may only conditionally write them, so the pre-loop
    // value must flow through). Exclude the loop var. Write-only
    // locals that aren't observed after the loop are NOT live-in —
    // they're fresh per iteration.
    const loopBody =
      stmt.tag === "For" || stmt.tag === "While" ? stmt.body : [];
    const genuineReads = genuineLoopReads(loopBody);
    const defined = definedBefore[i];
    const liveInNames: string[] = [];
    const liveInTypes: JitType[] = [];
    const addLiveIn = (name: string): void => {
      if (stmt.tag === "For" && name === stmt.varName) return;
      if (liveInNames.includes(name)) return;
      // Must be defined at the extraction point — otherwise the call
      // site would pass undefined.
      if (!defined.has(name)) return;
      const t = outerEndEnv.get(name);
      if (!t) return;
      liveInNames.push(name);
      liveInTypes.push(t);
    };
    // Also include reads that come from the loop's own header (start /
    // step / end / While cond) — these fire before the body, so any
    // body-level pre-write doesn't apply.
    if (stmt.tag === "For") {
      const hdr = emptyRefs();
      walkExpr(stmt.start, hdr);
      if (stmt.step) walkExpr(stmt.step, hdr);
      walkExpr(stmt.end, hdr);
      for (const n of hdr.reads) addLiveIn(n);
    } else {
      const hdr = emptyRefs();
      walkExpr(stmt.cond, hdr);
      for (const n of hdr.reads) addLiveIn(n);
    }
    for (const n of genuineReads) addLiveIn(n);
    for (const n of refs.writes) {
      if (liveOutNames.includes(n)) addLiveIn(n);
    }

    // Locals: any write that isn't a param. Outputs that are not also
    // params still need a local declaration in the synthetic fn (the C
    // codegen emits `double v_x = 0.0; ... *v_x_out = v_x;` — the local
    // is the storage, the out-param is the writeback slot).
    const localVars = new Set<string>();
    for (const n of refs.writes) {
      if (liveInNames.includes(n)) continue;
      localVars.add(n);
    }

    const typeKey = liveInNames
      .map((n, k) => `${n}:${jitTypeKey(liveInTypes[k])}`)
      .join(",");
    const outKey = liveOutNames
      .map((n, k) => `${n}:${jitTypeKey(liveOutTypes[k])}`)
      .join(",");
    const identity = `${interp.currentFile}:hybrid_${stmt.tag}:${i}:${typeKey}->${outKey}`;
    const jitName = `$jit_hybrid_${stmt.tag}_${i}_${hashStr(identity)}`;
    const helperKey = `$cjit_${jitName}`;

    if (helpers[helperKey]) {
      installReplacement();
      continue;
    }

    const syntheticFn: FunctionDef = {
      name: `$hybrid_${stmt.tag}`,
      params: liveInNames,
      outputs: liveOutNames,
      body: [],
    };

    const res = backend.tryCompile(
      interp,
      syntheticFn,
      [stmt],
      liveOutNames,
      localVars,
      liveOutTypes[0] ?? null,
      liveOutTypes,
      liveInTypes,
      liveOutNames.length,
      generatedIRBodies
    );
    if (!res.ok) continue;
    helpers[helperKey] = res.fn;
    installReplacement();

    function installReplacement(): void {
      generatedFns.set(jitName, `var ${jitName} = $h.${helperKey};`);
      const args: JitExpr[] = liveInNames.map((n, k) => ({
        tag: "Var",
        name: n,
        jitType: liveInTypes[k],
      }));
      outerBody[i] = {
        tag: "UserCallWriteback",
        outputs: liveOutNames,
        jitName,
        name: syntheticFn.name,
        args,
        outputTypes: liveOutTypes,
      };
    }
  }
}
