/**
 * e2 whole-loop C JIT.
 *
 * For a `for varName = lo:hi <body> end` where the body fits a supported
 * shape, emit a single C function that runs all n iterations and call it
 * once, instead of walking the AST on every iteration.
 *
 * Without this path, `--opt e2` pays ~70–100 ns per iter on a trivial
 * `s = s + i` just for AST dispatch; a compiled C loop runs it in <1 ns.
 *
 * Current supported body shapes (all may mix in one loop):
 *   - scalar assign           `s = s + sin(i) * cos(i) + sqrt(i*0.01)`
 *   - scalar indexed read     `s = s + x(i)`                   (real tensor x)
 *   - scalar indexed write    `y(i) = sin(i*0.01)`             (preallocated y)
 *   - tensor local (elemwise) `c = a.*b + i*0.001`             (per-element
 *                                                               expression is
 *                                                               inlined into any
 *                                                               consuming sum();
 *                                                               last-iter value
 *                                                               is also written
 *                                                               back to the env
 *                                                               for MATLAB
 *                                                               post-loop
 *                                                               visibility)
 *   - reductions              `s = s + sum(c)`                 (c is a
 *                                                               tensor_local —
 *                                                               chained
 *                                                               tensor_locals
 *                                                               fuse through)
 *
 * Not supported (falls through to the interpreter / other JIT paths):
 *   - non-`lo:hi` loop shapes (stepped ranges, `for i = v`)
 *   - complex or logical tensor inputs
 *   - matrix-matrix / matrix-vector multiplication
 *   - bsxfun / broadcast across shapes
 *   - function-handle calls, user-function calls
 *   - control flow inside the body (if / while / return)
 *   - multi-dimensional tensor access
 */
import type { Expr, Stmt } from "../../parser/types.js";
import type { Interpreter } from "../../interpreter/interpreter.js";
import type { JitExpr, JitType } from "../../jit/jitTypes.js";
import { lowerAstToJitExpr, E2LowerError } from "./astToJitExpr.js";
import { getE2CompileFn } from "./compileFn.js";
import { analyzeForLoop } from "../../jit/jitLoopAnalysis.js";
import { isRuntimeTensor, type RuntimeTensor } from "../../runtime/types.js";
import { unshare } from "../jsJit/js/jitHelpersTensor.js";
import {
  emitLoopKernel,
  LOOP_SCALAR_BUILTINS,
  type BodyStmt,
  type Reduction,
} from "./loopKernelEmit.js";

/** Reduction builtins that fuse into a scalar accumulator inside the
 *  loop body. These take a single tensor argument and return a scalar;
 *  they are NEVER env references, so `readBeforeWrite` analysis and
 *  env-type classification must skip them. */
const LOOP_REDUCTION_BUILTINS: ReadonlySet<string> = new Set(["sum"]);

/** Per-stmt compile cache: maps the For-stmt AST node identity to the
 *  already-compiled C function + its calling-convention metadata. A
 *  `"bailed"` marker means we've determined this stmt can't be compiled
 *  and shouldn't be re-analyzed on every iteration. */
interface CachedLoop {
  fn: (...args: unknown[]) => unknown;
  /** Scalar env vars read but not written in the body. Passed by value. */
  scalarInputVars: string[];
  /** Tensor env vars READ but never written in the body. `const double *`. */
  tensorInputVars: string[];
  /** Tensor env vars WRITTEN (element-wise) in the body. `double *`.
   *  Must exist in env with enough capacity before the loop runs. */
  tensorInoutVars: string[];
  /** Scalar env vars written (possibly also read) in the body. Passed
   *  as `double *` and written back to env after the kernel returns. */
  inoutVars: string[];
  /** Pure loop-local tensor assignments. The kernel writes each one's
   *  last-iteration value to a caller-allocated `double *` buffer; the
   *  JS side wraps that buffer into a RuntimeTensor and writes it to
   *  env so MATLAB post-loop visibility is preserved. Size/shape is
   *  copied from `lengthTensor` (an env tensor referenced by name). */
  tensorLocalOutputs: { name: string; lengthTensor: string }[];
}

const LOOP_CACHE = new WeakMap<Stmt, CachedLoop | "bailed">();

/**
 * Attempt to compile and execute a for-loop as one C kernel under
 * `--opt e2`. Returns true on success, false to fall back to the regular
 * interpreter path (the caller will run the loop normally).
 */
export function tryE2Loop(
  interp: Interpreter,
  stmt: Stmt & { type: "For" }
): boolean {
  if (interp.experimental !== "e2") return false;

  const DBG = process.env.NUMBL_E2_LOOP_DEBUG === "1";
  const dbg = (msg: string): void => {
    if (DBG) process.stderr.write(`[e2-loop] ${msg}\n`);
  };
  /** Soft bail: don't cache. Use for runtime-dependent checks (env
   *  types, range values) that might succeed on a later call. */
  const bailSoft = (reason: string): false => {
    dbg(`bail: ${reason}`);
    return false;
  };
  /** Hard bail: cache the "bailed" marker so we don't redo analysis
   *  every iteration. Use for structural issues (unsupported shape,
   *  type combinations the emitter rejects) that won't change. */
  const bailHard = (reason: string): false => {
    dbg(`bail (cached): ${reason}`);
    LOOP_CACHE.set(stmt, "bailed");
    return false;
  };

  dbg(`try ${stmt.span?.file}:${stmt.span?.start}`);

  const cached = LOOP_CACHE.get(stmt);
  if (cached === "bailed") {
    dbg("bailed (cached)");
    return false;
  }

  // ── 1. Validate loop shape (structural — hard-bail candidates,
  //      but we don't cache here: on first visit these checks are
  //      cheap and `cached === "bailed"` short-circuits future visits
  //      once any subsequent phase hard-bails). ─────────────────────
  if (stmt.expr.type !== "Range") {
    return bailSoft(`not Range (${stmt.expr.type})`);
  }
  if (stmt.expr.step !== null) return bailSoft("non-null step");
  if (stmt.body.length === 0) return bailSoft("empty body");
  for (const s of stmt.body) {
    if (s.type !== "Assign" && s.type !== "AssignLValue") {
      return bailSoft(`body has ${s.type}`);
    }
    if (s.type === "AssignLValue") {
      // Only `ident(scalar_idx) = ...` (single-index tensor write) —
      // everything else (cell index, member, multi-dim index) bails.
      const lv = s.lvalue;
      if (
        lv.type !== "Index" ||
        lv.base.type !== "Ident" ||
        lv.indices.length !== 1
      ) {
        return bailSoft("unsupported AssignLValue form");
      }
    }
  }

  // ── 2. Input/output classification. Loop variable is internal. ──
  const analysis = analyzeForLoop(stmt);
  if (analysis.hasReturn) return bailSoft("body has return");
  const readInBody = new Set(analysis.inputs);
  const writtenInBody = new Set(analysis.outputs);
  readInBody.delete(stmt.varName);
  writtenInBody.delete(stmt.varName);

  // Tensor targets of AssignLValue writes need `double *` (non-const)
  // in the kernel signature. `analyzeForLoop` doesn't register indexed
  // writes as outputs — add them manually so later checks see the
  // tensor as a read-before-write (its initial data is needed, and
  // the buffer is mutated in-place).
  const tensorWriteTargets = new Set<string>();
  for (const s of stmt.body) {
    if (s.type !== "AssignLValue") continue;
    const lv = s.lvalue;
    if (lv.type === "Index" && lv.base.type === "Ident") {
      tensorWriteTargets.add(lv.base.name);
      writtenInBody.add(lv.base.name);
    }
  }

  // ── 3. Compute "read before write" — names whose initial env value
  //      actually matters inside the kernel. Names written before read
  //      are pure loop-locals; their initial env value is never used,
  //      so we don't require them to exist. ──────────────────────────
  const readBeforeWrite = computeReadBeforeWrite(stmt.body, stmt.varName);

  // ── 4. Build initial envTypes. Each read-before-written name must
  //      resolve to a scalar number or a real 1-D double tensor; every
  //      other shape bails. ────────────────────────────────────────
  const envTypes = new Map<string, JitType>();
  envTypes.set(stmt.varName, { kind: "number" });
  const tensorInputSet = new Set<string>();
  for (const name of readBeforeWrite) {
    // `collectExprNames` can't distinguish env references from builtin
    // names — filter out known scalar builtins and reduction builtins
    // here so we don't require them to exist in env.
    if (name in LOOP_SCALAR_BUILTINS) continue;
    if (LOOP_REDUCTION_BUILTINS.has(name)) continue;
    const val = interp.env.get(name);
    if (val === undefined) return bailSoft(`env.${name} is undefined`);
    if (typeof val === "number") {
      envTypes.set(name, { kind: "number" });
      continue;
    }
    if (isRuntimeTensor(val)) {
      if (!(val.data instanceof Float64Array) || val.imag) {
        return bailSoft(`env.${name} is non-real-double tensor`);
      }
      envTypes.set(name, { kind: "tensor", isComplex: false });
      tensorInputSet.add(name);
      continue;
    }
    return bailSoft(`env.${name} is ${typeof val}`);
  }
  // Tensor-write targets must also exist as real-double tensors so we
  // know the destination buffer is real-typed.
  for (const name of tensorWriteTargets) {
    const val = interp.env.get(name);
    if (
      val === undefined ||
      !isRuntimeTensor(val) ||
      !(val.data instanceof Float64Array) ||
      val.imag
    ) {
      return bailSoft(`tensor-write target ${name} missing or non-real-double`);
    }
    envTypes.set(name, { kind: "tensor", isComplex: false });
  }
  // Names written-only in body (i.e. pure loop-locals) deliberately
  // don't get a default type here — their type is decided by the
  // lowered RHS of their first write (a tensor-valued RHS produces a
  // `tensor_local` destination; a scalar RHS produces a plain local).

  // ── 5. Lower each body stmt to a BodyStmt for codegen. This also
  //      rewrites `sum(<tensor_local>)` into synthetic idents + a
  //      Reduction record for the emitter. ─────────────────────────
  const bodySpecs: BodyStmt[] = [];
  const tensorLocals = new Map<
    string,
    { elemExpr: JitExpr; lengthTensor: string }
  >();
  const lowerOpts = { resolveFuncCallAsTensorIndex: true };
  const lowerEnvTypes = new Map(envTypes);
  let synthCounter = 0;
  const nextSynthName = (): string => `__sum_${synthCounter++}`;

  for (const s of stmt.body) {
    if (s.type === "Assign") {
      const reductions: Reduction[] = [];
      const rewrittenRhs = rewriteReductionsInAst(
        s.expr,
        tensorLocals,
        reductions,
        nextSynthName
      );
      for (const r of reductions) {
        lowerEnvTypes.set(r.synthName, { kind: "number" });
      }

      let rhs: JitExpr;
      try {
        rhs = lowerAstToJitExpr(rewrittenRhs, lowerEnvTypes, lowerOpts);
      } catch (e) {
        if (e instanceof E2LowerError) {
          return bailHard(`lower ${s.name}: ${e.message}`);
        }
        throw e;
      }

      if (rhs.jitType.kind === "tensor") {
        if (rhs.jitType.isComplex || rhs.jitType.isLogical) {
          return bailHard(`${s.name}: tensor rhs is complex/logical`);
        }
        if (reductions.length > 0) {
          // A tensor-valued rhs with `sum()` inside would require
          // materializing the intermediate; we don't support that.
          return bailHard(`${s.name}: tensor rhs contains reduction refs`);
        }
        if (tensorLocals.has(s.name)) {
          // Multiple writes to the same tensor_local name would cause
          // a self-reference in the emitter's innerFt substitution
          // (the second write's elemExpr refs a name bound to itself).
          return bailHard(`${s.name}: duplicate tensor_local assignment`);
        }
        if (tensorInputSet.has(s.name)) {
          // The name is read from env BEFORE this assignment (so it's a
          // tensor input) AND re-assigned in the body. Cross-iter dep:
          // later body references would need the new value, but the
          // kernel always reads input buffers from their pre-loop state.
          return bailHard(
            `${s.name}: input tensor reassigned in body (cross-iter dependency)`
          );
        }
        // All input tensors in an elemwise expression share length at
        // runtime (MATLAB semantics), so any referenced one works. A
        // tensor_local ref resolves transitively to a real input tensor.
        const lengthTensor = findFirstInputTensor(
          rhs,
          tensorInputSet,
          tensorLocals
        );
        if (!lengthTensor) {
          return bailHard(
            `${s.name}: tensor rhs has no input tensor (can't determine length)`
          );
        }
        tensorLocals.set(s.name, { elemExpr: rhs, lengthTensor });
        bodySpecs.push({
          kind: "tensor_local",
          name: s.name,
          elemExpr: rhs,
          lengthTensor,
        });
        lowerEnvTypes.set(s.name, { kind: "tensor", isComplex: false });
      } else if (rhs.jitType.kind === "number") {
        bodySpecs.push({
          kind: "scalar_assign",
          name: s.name,
          rhs,
          reductions,
        });
        lowerEnvTypes.set(s.name, { kind: "number" });
      } else {
        return bailHard(
          `${s.name}: rhs unsupported type (${rhs.jitType.kind})`
        );
      }
      continue;
    }

    if (s.type !== "AssignLValue") continue; // unreachable (shape-validated)
    // Already shape-validated above as `Index{base: Ident, indices: [i]}`.
    const lv = s.lvalue as {
      type: "Index";
      base: { type: "Ident"; name: string };
      indices: Expr[];
    };
    const targetName = lv.base.name;
    let idxRhs: JitExpr;
    let rhs: JitExpr;
    try {
      idxRhs = lowerAstToJitExpr(lv.indices[0], lowerEnvTypes, lowerOpts);
      rhs = lowerAstToJitExpr(s.expr, lowerEnvTypes, lowerOpts);
    } catch (e) {
      if (e instanceof E2LowerError) {
        return bailHard(`lower ${targetName}: ${e.message}`);
      }
      throw e;
    }
    if (idxRhs.jitType.kind !== "number" && idxRhs.jitType.kind !== "boolean") {
      return bailHard(`${targetName}(..): index not scalar number`);
    }
    if (rhs.jitType.kind !== "number") {
      return bailHard(
        `${targetName}(..): rhs not scalar (${rhs.jitType.kind})`
      );
    }
    bodySpecs.push({ kind: "tensor_write", name: targetName, idxRhs, rhs });
  }

  // Safety: a tensor_local must NEVER appear as a plain Var in any
  // later scalar/tensor-write RHS. The only supported use is inside
  // `sum(...)`, which was already rewritten to a synthetic ident.
  // This also catches tensor_locals that would escape the loop.
  for (const b of bodySpecs) {
    const exprs: JitExpr[] = [];
    if (b.kind === "scalar_assign") exprs.push(b.rhs);
    else if (b.kind === "tensor_write") {
      exprs.push(b.idxRhs);
      exprs.push(b.rhs);
    } else continue;
    for (const e of exprs) {
      const offender = findTensorLocalRef(e, tensorLocals);
      if (offender) {
        return bailHard(`tensor_local '${offender}' used outside sum()`);
      }
    }
  }

  // A tensor_local's elemExpr may only reference tensor_locals declared
  // EARLIER in body order. References to self or to later-declared
  // tensor_locals would create cycles in the emitter's innerFt
  // substitution (which expands tensor_local Vars recursively).
  const earlierTLs = new Set<string>();
  for (const b of bodySpecs) {
    if (b.kind !== "tensor_local") {
      continue;
    }
    const refs = new Set<string>();
    collectVarNames(b.elemExpr, refs);
    for (const r of refs) {
      if (!tensorLocals.has(r)) continue;
      if (!earlierTLs.has(r)) {
        return bailHard(
          `tensor_local '${b.name}' refs tensor_local '${r}' which is not earlier in body`
        );
      }
    }
    earlierTLs.add(b.name);
  }

  // ── 6. Evaluate range endpoints. Must be exact integers. ────────
  let loNum: number;
  let hiNum: number;
  try {
    loNum = toIntExactOrNaN(interp.evalExpr(stmt.expr.start));
    hiNum = toIntExactOrNaN(interp.evalExpr(stmt.expr.end));
  } catch {
    return bailSoft("range eval threw");
  }
  if (!Number.isFinite(loNum) || !Number.isFinite(hiNum)) {
    return bailSoft(`range non-integer (lo=${loNum} hi=${hiNum})`);
  }

  // ── 7. Classify body-referenced names into kernel-param buckets:
  //        scalarInputVars:  scalar reads, pass by value
  //        tensorInputVars:  tensor reads only, `const double *`
  //        tensorInoutVars:  tensor writes (via y(i)=), `double *`
  //        inoutVars:        scalar writes, `double *` (read back)
  //      Only include names actually referenced in the lowered body —
  //      `analyzeForLoop` also picks up range-expr names that we've
  //      already evaluated above. ─────────────────────────────────
  const bodyRefs = new Set<string>();
  for (const b of bodySpecs) {
    if (b.kind === "scalar_assign") {
      collectVarNames(b.rhs, bodyRefs);
    } else if (b.kind === "tensor_write") {
      collectVarNames(b.idxRhs, bodyRefs);
      collectVarNames(b.rhs, bodyRefs);
      bodyRefs.add(b.name); // the write target itself
    } else {
      collectVarNames(b.elemExpr, bodyRefs);
    }
  }
  const tensorInoutVars = [...tensorWriteTargets];
  // Scalar inoutVars are written names that aren't tensor write-targets
  // and aren't tensor_locals (the latter have no runtime scalar value).
  const inoutVars = [...writtenInBody].filter(
    n => !tensorWriteTargets.has(n) && !tensorLocals.has(n)
  );
  const tensorInputVars: string[] = [];
  const scalarInputVars: string[] = [];
  for (const n of readInBody) {
    if (writtenInBody.has(n)) continue;
    if (!bodyRefs.has(n)) continue;
    if (tensorInputSet.has(n)) tensorInputVars.push(n);
    else scalarInputVars.push(n);
  }

  dbg(
    `match: lo=${loNum} hi=${hiNum} ` +
      `inout=[${inoutVars.join(",")}] ` +
      `scalarIn=[${scalarInputVars.join(",")}] ` +
      `tensorIn=[${tensorInputVars.join(",")}] ` +
      `tensorInout=[${tensorInoutVars.join(",")}]`
  );

  // ── 8. Compile (or reuse cached compile). Source is deterministic
  //      in the shape of body + variable names, so compile.ts dedupes
  //      across structurally-identical loops. ──────────────────────
  let entry: CachedLoop | undefined = cached;
  if (!entry) {
    const { cSource, kernelName, koffiSig } = emitLoopKernel(
      scalarInputVars,
      tensorInputVars,
      tensorInoutVars,
      inoutVars,
      stmt.varName,
      bodySpecs
    );
    let fn;
    try {
      fn = getE2CompileFn()(cSource, koffiSig, kernelName, msg =>
        process.stderr.write(`[e2] ${msg}\n`)
      );
    } catch {
      return bailHard("compile threw");
    }
    if (!fn) return bailHard("compile returned null");
    const tensorLocalOutputs: { name: string; lengthTensor: string }[] = [];
    for (const b of bodySpecs) {
      if (b.kind === "tensor_local") {
        tensorLocalOutputs.push({ name: b.name, lengthTensor: b.lengthTensor });
      }
    }
    entry = {
      fn,
      scalarInputVars,
      tensorInputVars,
      tensorInoutVars,
      inoutVars,
      tensorLocalOutputs,
    };
    LOOP_CACHE.set(stmt, entry);
    interp.onCCompile?.(
      `e2 loop kernel: for ${stmt.varName} = ...: (scalarIn=${scalarInputVars.join(",")}, tensorIn=${tensorInputVars.join(",")}, tensorInout=${tensorInoutVars.join(",")}, inout=${inoutVars.join(",")}, tensorLocals=${tensorLocalOutputs.map(t => t.name).join(",")})`,
      cSource
    );
  }

  // ── 9. Marshal and call. ────────────────────────────────────────
  const args: unknown[] = [loNum, hiNum];
  for (const name of entry.scalarInputVars) {
    args.push(interp.env.get(name) as number);
  }
  for (const name of entry.tensorInputVars) {
    const t = interp.env.get(name) as RuntimeTensor;
    args.push(t.data);
    args.push(t.data.length);
  }
  for (const name of entry.tensorInoutVars) {
    // The kernel mutates this buffer in-place; if it's shared (rc > 1)
    // we must copy first so we don't corrupt the caller's tensor.
    // `unshare` is a no-op at rc == 1.
    const t = unshare(interp.env.get(name));
    interp.env.set(name, t);
    args.push(t.data);
    args.push(t.data.length);
  }
  const inoutBufs: Float64Array[] = [];
  for (const name of entry.inoutVars) {
    const init = interp.env.get(name);
    const buf = new Float64Array(1);
    buf[0] = typeof init === "number" ? init : 0;
    inoutBufs.push(buf);
    args.push(buf);
  }
  // One output buffer per tensor_local, sized to match its
  // length-input-tensor. The kernel fills it on the last iteration.
  const tlBufs: Float64Array[] = [];
  for (const tl of entry.tensorLocalOutputs) {
    const lenT = interp.env.get(tl.lengthTensor) as RuntimeTensor;
    const buf = new Float64Array(lenT.data.length);
    tlBufs.push(buf);
    args.push(buf);
  }
  entry.fn(...args);

  // ── 10. Write inout + tensor_local results back to env. ────────
  for (let i = 0; i < entry.inoutVars.length; i++) {
    interp.env.set(entry.inoutVars[i], inoutBufs[i][0]);
  }
  // Only write tensor_locals back if the loop actually iterated — a
  // zero-iteration loop never touched the output buffer, so pre-loop
  // env values should be preserved.
  if (loNum <= hiNum) {
    for (let i = 0; i < entry.tensorLocalOutputs.length; i++) {
      const tl = entry.tensorLocalOutputs[i];
      const lenT = interp.env.get(tl.lengthTensor) as RuntimeTensor;
      const out: RuntimeTensor = {
        kind: "tensor",
        data: tlBufs[i],
        shape: [...lenT.shape],
        _rc: 1,
      };
      interp.env.set(tl.name, out);
    }
    // The loop variable's post-loop value in MATLAB is the last
    // iterated value (hiNum); zero-iter leaves it unchanged.
    interp.env.set(stmt.varName, hiNum);
  }

  return true;
}

// ── Helpers ──────────────────────────────────────────────────────

/** Walk the body stmts in order and return the set of names that are
 *  read before they are first written. Those names' initial env values
 *  are actually observed; names not in this set are pure loop-locals
 *  (written first, so initial value never matters). The loop variable
 *  is never included — it's always fresh each iteration. */
function computeReadBeforeWrite(body: Stmt[], loopVar: string): Set<string> {
  const written = new Set<string>();
  const readBeforeWrite = new Set<string>();
  const addRef = (name: string): void => {
    if (name === loopVar) return;
    if (!written.has(name)) readBeforeWrite.add(name);
  };
  for (const s of body) {
    if (s.type === "Assign") {
      const refs = new Set<string>();
      collectExprNames(s.expr, refs);
      for (const r of refs) addRef(r);
      written.add(s.name);
    } else if (s.type === "AssignLValue") {
      const refs = new Set<string>();
      collectExprNames(s.expr, refs);
      const lv = s.lvalue;
      if (lv.type === "Index") {
        // `y(i) = ...` reads i (and maybe others via the index expr).
        for (const idx of lv.indices) collectExprNames(idx, refs);
      }
      for (const r of refs) addRef(r);
      if (lv.type === "Index" && lv.base.type === "Ident") {
        // A tensor-element write mutates the base tensor, so treat it
        // as "written" — subsequent references don't need env lookup.
        written.add(lv.base.name);
      }
    }
  }
  return readBeforeWrite;
}

/** Walk an AST `Expr` (parser type) and collect Ident names referenced.
 *  Used for read-before-write analysis before lowering — we can't use
 *  the lowered JitExpr walk because lowering itself requires types for
 *  every referenced name. */
function collectExprNames(expr: Expr, out: Set<string>): void {
  switch (expr.type) {
    case "Ident":
      out.add(expr.name);
      return;
    case "Binary":
      collectExprNames(expr.left, out);
      collectExprNames(expr.right, out);
      return;
    case "Unary":
      collectExprNames(expr.operand, out);
      return;
    case "FuncCall":
      // The name could be a builtin OR a tensor index (MATLAB syntax
      // overload) — treat it as a potential env ref; caller filters
      // out known builtin names.
      out.add(expr.name);
      for (const a of expr.args) collectExprNames(a, out);
      return;
    case "Index":
      collectExprNames(expr.base, out);
      for (const a of expr.indices) collectExprNames(a, out);
      return;
    default:
      return;
  }
}

/** Walk an AST Expr tree; for every `FuncCall{name:"sum", args:[Ident x]}`
 *  where `x` is a tensor_local declared earlier in the body, replace the
 *  FuncCall with a fresh synthetic Ident, append a `Reduction` record,
 *  and return the (structurally-copied) tree.
 *
 *  Future reductions (prod/max/min/...) plug in via the same helper. */
function rewriteReductionsInAst(
  expr: Expr,
  tensorLocals: ReadonlyMap<string, unknown>,
  out: Reduction[],
  nextSynthName: () => string
): Expr {
  const rec = (e: Expr): Expr =>
    rewriteReductionsInAst(e, tensorLocals, out, nextSynthName);
  if (
    expr.type === "FuncCall" &&
    LOOP_REDUCTION_BUILTINS.has(expr.name) &&
    expr.args.length === 1
  ) {
    const arg = expr.args[0];
    if (arg.type === "Ident" && tensorLocals.has(arg.name)) {
      const synthName = nextSynthName();
      out.push({
        synthName,
        tensorLocal: arg.name,
        op: expr.name as "sum",
      });
      return { type: "Ident", name: synthName, span: expr.span };
    }
  }
  switch (expr.type) {
    case "Binary":
      return { ...expr, left: rec(expr.left), right: rec(expr.right) };
    case "Unary":
      return { ...expr, operand: rec(expr.operand) };
    case "FuncCall":
      return { ...expr, args: expr.args.map(rec) };
    case "Index":
      return { ...expr, base: rec(expr.base), indices: expr.indices.map(rec) };
    default:
      return expr;
  }
}

/** Return an input-tensor name whose length suffices to size an inner
 *  reduction loop over this expression. All input tensors in an
 *  elemwise expression must have matching length at runtime (MATLAB
 *  semantics), so any one works. A tensor_local reference resolves to
 *  that local's own lengthTensor (chained tensor_locals inherit through
 *  to a real input tensor). Returns null if the expression neither
 *  reads nor transitively reaches an input tensor. */
function findFirstInputTensor(
  expr: JitExpr,
  tensorInputSet: ReadonlySet<string>,
  tensorLocals: ReadonlyMap<string, { elemExpr: JitExpr; lengthTensor: string }>
): string | null {
  const rec = (e: JitExpr): string | null =>
    findFirstInputTensor(e, tensorInputSet, tensorLocals);
  switch (expr.tag) {
    case "Var":
      if (tensorInputSet.has(expr.name)) return expr.name;
      return tensorLocals.get(expr.name)?.lengthTensor ?? null;
    case "Binary":
      return rec(expr.left) ?? rec(expr.right);
    case "Unary":
      return rec(expr.operand);
    case "Call":
      for (const a of expr.args) {
        const f = rec(a);
        if (f) return f;
      }
      return null;
    case "Index":
      return rec(expr.base);
    default:
      return null;
  }
}

/** Walk a JitExpr; if any Var references a tensor_local, return its
 *  name (to report in the bail message). Null means the expression is
 *  clean — only input tensors and scalars. */
function findTensorLocalRef(
  expr: JitExpr,
  tensorLocals: ReadonlyMap<string, unknown>
): string | null {
  const rec = (e: JitExpr): string | null =>
    findTensorLocalRef(e, tensorLocals);
  switch (expr.tag) {
    case "Var":
      return tensorLocals.has(expr.name) ? expr.name : null;
    case "Binary":
      return rec(expr.left) ?? rec(expr.right);
    case "Unary":
      return rec(expr.operand);
    case "Call":
      for (const a of expr.args) {
        const f = rec(a);
        if (f) return f;
      }
      return null;
    case "Index":
      return (
        rec(expr.base) ??
        expr.indices.reduce<string | null>((acc, i) => acc ?? rec(i), null)
      );
    default:
      return null;
  }
}

/** Walk a JitExpr and collect all names referenced via Var nodes. Used
 *  to drop names that only appear in the range expression (already
 *  evaluated) from the kernel's input list. */
function collectVarNames(expr: JitExpr, out: Set<string>): void {
  switch (expr.tag) {
    case "Var":
      out.add(expr.name);
      return;
    case "Binary":
      collectVarNames(expr.left, out);
      collectVarNames(expr.right, out);
      return;
    case "Unary":
      collectVarNames(expr.operand, out);
      return;
    case "Call":
      for (const a of expr.args) collectVarNames(a, out);
      return;
    case "Index":
      collectVarNames(expr.base, out);
      for (const i of expr.indices) collectVarNames(i, out);
      return;
    default:
      // Other shapes aren't produced by lowerAstToJitExpr for the
      // scalar-loop body (TensorLiteral / UserCall / etc. are outside
      // the supported subset).
      return;
  }
}

/** Return `v` if it's an integer (representable exactly), otherwise NaN
 *  so the caller bails. Range-form without step iterates integers in
 *  MATLAB, and we only emit integer-counter loops. */
function toIntExactOrNaN(v: unknown): number {
  if (typeof v !== "number") return NaN;
  if (!Number.isFinite(v) || !Number.isInteger(v)) return NaN;
  return v;
}
