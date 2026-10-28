/**
 * Simplified runtime providing the $rt interface for generated JS.
 *
 * This is a focused rewrite of runtime/runtime.ts (~3300 lines) that provides
 * exactly the methods the codegen emits, without class support, namespace
 * dispatch, WebGPU, subsref/subsasgn overloads, or property accessors.
 *
 * It reuses existing runtime value types, operations, indexing, and builtins.
 *
 * Operator functions are in runtimeOperators.ts, dispatch/registration in
 * runtimeDispatch.ts, and helper types/utilities in runtimeHelpers.ts.
 */

import {
  type RuntimeValue,
  type RuntimeLogical,
  type RuntimeTensor,
  type RuntimeStruct,
  type RuntimeString,
  type RuntimeFunction,
  RTV,
  toNumber,
  toBool,
  displayValue,
  RuntimeError,
  CancellationError,
  type CallFrame,
} from "../runtime/index.js";
import { incref, decref, RefScope } from "./refcount.js";
import {
  isRuntimeNumber,
  isRuntimeTensor,
  isRuntimeClassInstance,
  isRuntimeClassInstanceArray,
  isRuntimeCell,
  isRuntimeLogical,
  isRuntimeSparseMatrix,
  RuntimeChar,
  RuntimeCell,
  RuntimeClassInstanceArray,
  type RuntimeComplexNumber,
  type RuntimeClassInstance,
} from "../runtime/types.js";
import { BinaryOperation } from "../parser/index.js";
import { getConstant } from "../helpers/constants.js";
import { globToRegExp } from "../helpers/glob.js";
import { COLON_SENTINEL, END_SENTINEL } from "./sentinels.js";
import type { PlotInstruction } from "../plotting/types.js";
import type { ExecOptions } from "../executeCode.js";
import type { FileIOAdapter } from "../fileIOAdapter.js";
import { type FileTable, fileIOView, fileTableFor } from "../fileTable.js";
import type { SystemAdapter } from "../systemAdapter.js";
import { ensureRuntimeValue } from "./runtimeHelpers.js";
import { errorToMException } from "./mexception.js";
import {
  uplus as _uplus,
  uminus as _uminus,
  transpose as _transpose,
  ctranspose as _ctranspose,
  not as _not,
  binop as _binop,
  range as _range,
  forIter as _forIter,
  emptyTensor as _emptyTensor,
  emptyStruct as _emptyStruct,
  makeCell as _makeCell,
  opHorzcat,
  opVertcat,
  switchMatch as _switchMatch,
} from "./runtimeOperators.js";
import {
  getFuncHandle as _getFuncHandle,
  makeUserFuncHandle as _makeUserFuncHandle,
  isa as _isa,
  callSuperConstructor as _callSuperConstructor,
  createClassInstance as _createClassInstance,
  dispatch as _dispatch,
  methodDispatch as _methodDispatch,
  callBuiltin as _callBuiltin,
  callClassMethod as _callClassMethod,
  numblClass as _numblClass,
} from "./runtimeDispatch.js";
import {
  index as _index,
  indexCell as _indexCell,
  indexStore as _indexStore,
  indexCellStore as _indexCellStore,
  multiOutputCellAssign as _multiOutputCellAssign,
} from "./runtimeIndexing.js";
import { setWarningSink } from "../helpers/warn.js";
import {
  getMember as _getMember,
  getMemberDynamic as _getMemberDynamic,
  getMemberOrEmpty as _getMemberOrEmpty,
  setMemberReturn as _setMemberReturn,
  setMemberDynamicReturn as _setMemberDynamicReturn,
  subsrefCall as _subsrefCall,
  subsasgnCall as _subsasgnCall,
  memberChainAssign as _memberChainAssign,
} from "./runtimeMemberAccess.js";
import { syncSleep } from "./syncChannel.js";
import {
  figuresReducer,
  initialFiguresState,
} from "../plotting/figuresReducer.js";
import { computeAxisLimits } from "../plotting/axisLimits.js";
import { isRuntimeChar, isRuntimeString, kstr } from "./types.js";
import { allocFloat64Array } from "./alloc.js";
import { valueShape } from "./utils.js";

/** One `beginOutputCapture()` frame: the buffered output, plus the plot
 *  state to restore if the attempt is discarded. */
interface CaptureFrame {
  text: string[];
  plotLength: number;
  holdState: boolean;
  currentFigureHandle: number;
  maxFigureHandle: number;
  tiledLayoutState: Runtime["tiledLayoutState"];
}

// ── Runtime class ────────────────────────────────────────────────────

export class Runtime {
  outputLines: string[] = [];
  /** Stack of in-progress capture frames. While non-empty, `output()`
   *  appends to the top frame's buffer instead of emitting, so a
   *  speculative JIT attempt's side effects can be committed on success
   *  or discarded on bail (preventing duplicated output when the
   *  interpreter re-runs the same code). Plot instructions are pushed
   *  directly, so a frame records the plot state to roll back to. */
  private outputCaptureStack: CaptureFrame[] = [];
  plotInstructions: PlotInstruction[] = [];
  variableValues: Record<string, RuntimeValue> = {};
  holdState = false;
  /** Figure-handle tracking for MATLAB-style `figure` allocation: `figure`
   *  with no argument creates a new figure (`maxFigureHandle + 1`). */
  currentFigureHandle = 0;
  maxFigureHandle = 0;
  /** Monotonic id source for graphics handles whose trace can be live-updated
   *  via `set` / `update_trace` (e.g. lines returned by `line`). */
  graphicsIdCounter = 1;
  /** Reverse channel for `uihtml` components (HTML → MATLAB). Maps a component
   *  id (the `uihtml` instruction's `id`) to the callback handles registered
   *  for it. When non-empty after a run, the host keeps this runtime alive so
   *  iframe events can re-enter the interpreter and invoke these handles.
   *  Handles are incref'd while stored (see registeruihtmlcallback). */
  uihtmlCallbacks: Map<
    string,
    { HTMLEventReceived?: RuntimeValue; DataChanged?: RuntimeValue }
  > = new Map();
  /** Hook invoked by `sendEventToHTMLSource(src, name, data)` to push an event
   *  from MATLAB back to a uihtml component's page. `dataJson` is the data
   *  already `jsonencode`d. Set from ExecOptions; the host forwards it to the
   *  iframe. */
  onHtmlSourceEvent?: (compId: string, name: string, dataJson: string) => void;

  // tiledlayout/nexttile state. Reset by tiledlayout, advanced by nexttile.
  // mode: "fixed" uses the rows/cols verbatim; "flow"/"vertical"/"horizontal"
  // grow the grid as tiles are added.
  tiledLayoutState: {
    rows: number;
    cols: number;
    mode: "fixed" | "flow" | "vertical" | "horizontal";
    count: number;
  } | null = null;

  // Cooperative cancellation
  private cancelFlag: Int32Array | null = null;

  // Line tracking
  public $line = 0;
  public $file: string | null = null;
  public $callStack: CallFrame[] = [];

  // onCleanup scope stack: each function entry pushes an empty array;
  // onCleanup() appends to the top; function exit pops and runs handlers LIFO.
  public _cleanupScopes: RuntimeFunction[][] = [];

  public pushCleanupScope(): void {
    this._cleanupScopes.push([]);
  }

  public registerCleanup(fn: RuntimeFunction): void {
    const top = this._cleanupScopes[this._cleanupScopes.length - 1];
    if (top) top.push(fn);
  }

  public popAndRunCleanups(callFn: (fn: RuntimeFunction) => void): void {
    const handlers = this._cleanupScopes.pop();
    if (!handlers) return;
    for (let i = handlers.length - 1; i >= 0; i--) {
      try {
        callFn(handlers[i]);
      } catch {
        // Cleanup errors are silently ignored (MATLAB behavior)
      }
    }
  }

  // Sentinel values
  public COLON = COLON_SENTINEL;
  public END = END_SENTINEL;

  // Constructor helpers
  public RTV = RTV;

  // Persistent variable storage: funcJsId → varName → value
  // Public so the alias sweep can traverse it as a root.
  public persistentStore = new Map<string, Map<string, RuntimeValue>>();

  // Global variable storage: varName → value (shared across all functions)
  // Public object so generated code can read/write as $rt.$g["name"]
  public $g: Record<string, RuntimeValue> = {};

  /** Active per-statement transient scope. New refcounted values are
   *  auto-adopted into this scope (rc 0→1) on construction so they
   *  survive long enough to be bound somewhere; on scope drain at end
   *  of statement, anything still held only by the scope is decref'd
   *  to 0 and destroyed. Null when no statement is in flight. */
  public currentScope: RefScope | null = null;

  /** When true, `decref` on a zero count throws (loud at the
   *  underflow site rather than silently leaking). On by default — the
   *  refcount-driven COW system relies on accurate refcounts to make
   *  correct mutate-vs-copy decisions. */
  public strictRefcount: boolean = true;

  // Accessor guard: prevents recursive getter/setter/subsref calls
  public activeAccessors = new Set<string>();

  // Callbacks into the compiler for class information (set by executeCode).
  // resolveClassMethod compiles and returns a class method on demand.
  public resolveClassMethod:
    | ((
        className: string,
        methodName: string
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      ) => ((...args: any[]) => any) | null)
    | null = null;
  // Ancestor class names in method-resolution order (for isa / superclasses).
  public getClassAncestors: ((className: string) => string[]) | null = null;
  // Cache for resolved class methods: "className.methodName" → fn | null
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  public classMethodCache = new Map<string, ((...args: any[]) => any) | null>();

  /** Callback for addpath/rmpath — mutates search paths and rebuilds function index. */
  public onPathChange:
    | ((
        action: "add" | "remove",
        dir: string,
        position: "begin" | "end"
      ) => void)
    | null = null;

  /** Callback invoked after cd() to update the implicit cwd search path. */
  public onCwdChange: ((newCwd: string) => void) | null = null;

  /** Reference to the active search paths (set by executeCode). */
  public searchPaths: string[] = [];

  /** Warning identifiers turned off with warning('off', id); 'all' covers
   *  identifiers without their own entry. */
  public warningsOff = new Set<string>();

  /** Counter for uihtml component ids. */
  public uihtmlSeq = 0;

  // File I/O adapter (injected from host environment), viewed so each call
  // first flushes `files`.
  public fileIO?: FileIOAdapter;

  // Open file identifiers (fopen/fread/...), shared across runs that use
  // the same adapter.
  public files?: FileTable;

  // System adapter (injected from host environment)
  public system?: SystemAdapter;

  // Custom builtins (execution-specific overrides).
  // These take priority over IBuiltins.
  public customBuiltins: Record<
    string,
    (nargout: number, args: unknown[]) => unknown
  > = {};

  // The calling scope that builtins declared with `scope: true` see (an
  // interpreter `CallerScope`, opaque here so the runtime does not depend
  // on the interpreter). Set by Interpreter.installRuntimeCallbacks().
  public callerScope: (() => object) | null = null;

  // Resolves a function called by name (`dispatch`) from the running code's
  // scope; with `targetClassName`, the method of that class. Set by
  // Interpreter.installRuntimeCallbacks(). Returns a callable taking
  // (nargout, ...args), or null for a builtin or an unknown name.
  public resolveNamedCall:
    | ((
        name: string,
        args: unknown[],
        targetClassName?: string
      ) => ((nargout: number, ...args: unknown[]) => unknown) | null)
    | null = null;

  // Per-function JIT-compiled code for debugging: jsId → generated JS
  public jitFunctionCode = new Map<string, string>();

  // Always-on counter: how many times dispatchUnknown was called per function name
  public dispatchUnknownCounts = new Map<string, number>();

  // Profiling: stack-based self-time profiler for disjoint timings
  public profilingEnabled = false;
  private profileStack: { key: string; startTime: number }[] = [];
  private profileAccum = new Map<string, number>();
  private profileCounts = new Map<string, number>();

  // Profiling: interpreted loops with >1000 iterations (keyed by "file:line")
  public hotLoops = new Map<
    string,
    {
      file: string;
      line: number;
      kind: "for" | "while";
      iterations: number;
      callCount: number;
      totalTimeMs: number;
    }
  >();

  constructor(
    private options: ExecOptions,
    private initialVariableValues?: Record<string, RuntimeValue>
  ) {
    this.profilingEnabled = !!options.profile;
    if (options.fileIO) {
      this.fileIO = fileIOView(options.fileIO);
      this.files = fileTableFor(options.fileIO);
    }
    this.system = options.system;
    this.onHtmlSourceEvent = options.onHtmlSourceEvent;
    if (options.initialHoldState) {
      this.holdState = options.initialHoldState;
    }
    if (options.initialFigureHandles) {
      this.currentFigureHandle = options.initialFigureHandles.current;
      this.maxFigureHandle = options.initialFigureHandles.max;
    }
    if (options.cancelSAB) {
      this.cancelFlag = new Int32Array(options.cancelSAB);
    }
    setWarningSink(msg => this.output("Warning: " + msg + "\n"));
  }

  /** Throw CancellationError if the cancel flag has been set. */
  checkCancel(): void {
    if (this.cancelFlag && Atomics.load(this.cancelFlag, 0) !== 0) {
      throw new CancellationError();
    }
  }

  /** Run `fn` with a fresh `RefScope` as `this.currentScope`. On return,
   *  every value adopted into the scope is decref'd. Used by `execStmt`
   *  to bound the lifetime of expression transients to one statement. */
  public withScope<T>(fn: () => T): T {
    const prev = this.currentScope;
    const scope = new RefScope();
    this.currentScope = scope;
    try {
      return fn();
    } finally {
      this.currentScope = prev;
      scope.drain(this);
    }
  }

  // ── Builtin initialization ──────────────────────────────────────────

  /** Register the implicit default figure (handle 1) the first time a graphics
   *  op runs without an explicit `figure`, so a later no-arg `figure` allocates
   *  a new handle instead of reusing 1 (MATLAB semantics). */
  ensureCurrentFigure(): void {
    if (!this.currentFigureHandle) {
      this.currentFigureHandle = 1;
      this.maxFigureHandle = Math.max(this.maxFigureHandle, 1);
    }
  }

  public profileEnter(key: string): void {
    if (!this.profilingEnabled) return;
    const now = performance.now();
    const stack = this.profileStack;
    if (stack.length > 0) {
      const top = stack[stack.length - 1];
      this.profileAccum.set(
        top.key,
        (this.profileAccum.get(top.key) ?? 0) + (now - top.startTime)
      );
    }
    stack.push({ key, startTime: now });
  }

  public profileLeave(): void {
    if (!this.profilingEnabled) return;
    const now = performance.now();
    const frame = this.profileStack.pop()!;
    this.profileAccum.set(
      frame.key,
      (this.profileAccum.get(frame.key) ?? 0) + (now - frame.startTime)
    );
    this.profileCounts.set(
      frame.key,
      (this.profileCounts.get(frame.key) ?? 0) + 1
    );
    if (this.profileStack.length > 0) {
      this.profileStack[this.profileStack.length - 1].startTime = now;
    }
  }

  public getBuiltinProfile(): Record<
    string,
    {
      fallback: { totalTimeMs: number; callCount: number };
      interp: { totalTimeMs: number; callCount: number };
      jit: { totalTimeMs: number; callCount: number };
    }
  > {
    const zero = () => ({ totalTimeMs: 0, callCount: 0 });
    const result: Record<
      string,
      {
        fallback: { totalTimeMs: number; callCount: number };
        interp: { totalTimeMs: number; callCount: number };
        jit: { totalTimeMs: number; callCount: number };
      }
    > = {};
    const prefix = "builtin:";
    for (const [key, time] of this.profileAccum) {
      if (!key.startsWith(prefix)) continue;
      const rest = key.slice(prefix.length);
      let source: "fallback" | "interp" | "jit";
      let name: string;
      if (rest.startsWith("fallback:")) {
        source = "fallback";
        name = rest.slice(9);
      } else if (rest.startsWith("interp:")) {
        source = "interp";
        name = rest.slice(7);
      } else if (rest.startsWith("jit:")) {
        source = "jit";
        name = rest.slice(4);
      } else {
        continue;
      }
      const count = this.profileCounts.get(key) ?? 0;
      if (count > 0) {
        if (!result[name])
          result[name] = { fallback: zero(), interp: zero(), jit: zero() };
        result[name][source] = { totalTimeMs: time, callCount: count };
      }
    }
    return result;
  }

  public getDispatchProfile(): Record<
    string,
    { totalTimeMs: number; callCount: number }
  > {
    const result: Record<string, { totalTimeMs: number; callCount: number }> =
      {};
    const prefix = "dispatch:";
    for (const [key, time] of this.profileAccum) {
      if (key.startsWith(prefix)) {
        const name = key.slice(prefix.length);
        const count = this.profileCounts.get(key) ?? 0;
        if (count > 0) {
          result[name] = { totalTimeMs: time, callCount: count };
        }
      }
    }
    return result;
  }

  public getJitCompileTimeMs(): number {
    return this.profileAccum.get("jit") ?? 0;
  }

  public getDispatchUnknownCounts(): Record<string, number> {
    const result: Record<string, number> = {};
    for (const [name, count] of this.dispatchUnknownCounts) {
      result[name] = count;
    }
    return result;
  }

  // ── Output ──────────────────────────────────────────────────────────

  output(text: string): void {
    const depth = this.outputCaptureStack.length;
    if (depth > 0) {
      this.outputCaptureStack[depth - 1].text.push(text);
      return;
    }
    this.outputLines.push(text);
    this.options.onOutput?.(text);
  }

  /** Begin buffering output for a speculative execution attempt.
   *  Pair with exactly one `commitOutputCapture()` or
   *  `discardOutputCapture()`. */
  beginOutputCapture(): void {
    const t = this.tiledLayoutState;
    this.outputCaptureStack.push({
      text: [],
      plotLength: this.plotInstructions.length,
      holdState: this.holdState,
      currentFigureHandle: this.currentFigureHandle,
      maxFigureHandle: this.maxFigureHandle,
      tiledLayoutState: t && { ...t },
    });
  }

  /** Emit the buffered output (to the parent capture frame if nested,
   *  else to the real sink) — call when the attempt succeeded. */
  commitOutputCapture(): void {
    const frame = this.outputCaptureStack.pop();
    if (frame === undefined) return;
    for (const text of frame.text) this.output(text);
  }

  /** Drop the buffered output and roll back the plot instructions and
   *  plot state the attempt produced — call when the attempt bailed, so
   *  a subsequent re-run is the only thing that emits. */
  discardOutputCapture(): void {
    const frame = this.outputCaptureStack.pop();
    if (frame === undefined) return;
    if (this.plotInstructions.length > frame.plotLength) {
      this.plotInstructions.length = frame.plotLength;
    }
    this.holdState = frame.holdState;
    this.currentFigureHandle = frame.currentFigureHandle;
    this.maxFigureHandle = frame.maxFigureHandle;
    this.tiledLayoutState = frame.tiledLayoutState;
  }

  // ── Error handling ──────────────────────────────────────────────────

  public error(message: string): RuntimeError {
    const err = new RuntimeError(message);
    if (this.$line > 0) err.line = this.$line;
    return err;
  }

  public annotateError(e: unknown): void {
    if (e instanceof RuntimeError && e.callStack === null) {
      e.callStack = [...this.$callStack];
      if (e.file === null && this.$file !== null) e.file = this.$file;
      if (e.line === null && this.$line > 0) e.line = this.$line;
    }
  }

  /** The MException `catch` binds for an error, with the stack of where
   *  it was raised. */
  public wrapError(e: unknown): RuntimeClassInstance {
    this.annotateError(e);
    return errorToMException(e);
  }

  // ── Call stack ──────────────────────────────────────────────────────

  public pushCallFrame(name: string): void {
    if (this.$callStack.length >= 500) {
      throw new RuntimeError("Maximum call stack depth exceeded");
    }
    this.$callStack.push({
      name,
      callerFile: this.$file,
      callerLine: this.$line,
    });
  }

  public popCallFrame(): void {
    this.$callStack.pop();
  }

  // ── Type conversion ─────────────────────────────────────────────────

  public toBool(v: unknown): boolean {
    if (typeof v === "number") return v !== 0;
    if (typeof v === "boolean") return v;
    return toBool(ensureRuntimeValue(v));
  }

  public toNumber(v: unknown): number {
    if (typeof v === "number") return v;
    return toNumber(ensureRuntimeValue(v));
  }

  public lg(v: boolean): RuntimeLogical {
    return RTV.logical(v);
  }

  /** Scalar power that returns complex for negative base with fractional exponent */
  public pow(a: number, b: number): number | RuntimeComplexNumber {
    const r = Math.pow(a, b);
    if (isNaN(r) && !isNaN(a) && !isNaN(b)) {
      // Negative base with fractional exponent → complex result
      const absA = Math.abs(a);
      const theta = a < 0 ? Math.PI : 0;
      const lnR = Math.log(absA);
      const newRe = b * lnR;
      const newIm = b * theta;
      const expR = Math.exp(newRe);
      const re = expR * Math.cos(newIm);
      const im = expR * Math.sin(newIm);
      if (Math.abs(im) < 1e-15) return re;
      return RTV.complex(re, im);
    }
    return r;
  }

  // ── Display ─────────────────────────────────────────────────────────

  public get displayResults(): boolean {
    return !!this.options.displayResults;
  }

  public displayResult(v: unknown): void {
    if (!this.options.displayResults) return;
    if (v === undefined || v === null) return;
    const mv = ensureRuntimeValue(v);
    const text = displayValue(mv);
    this.output("ans =\n" + text + "\n\n");
  }

  public displayAssign(name: string, v: unknown): void {
    if (!this.options.displayResults) return;
    if (v === undefined || v === null) return;
    const mv = ensureRuntimeValue(v);
    const text = displayValue(mv);
    this.output(name + " =\n" + text + "\n\n");
  }

  // ── Constants ───────────────────────────────────────────────────────

  public getConstant(name: string): RuntimeValue {
    const c = getConstant(name);
    if (c === undefined) throw new RuntimeError(`Undefined constant: ${name}`);
    return c;
  }

  // ── String literals ─────────────────────────────────────────────────

  public makeChar(raw: string): RuntimeChar {
    let s = raw.slice(1, raw.length - 1);
    s = s.replaceAll("''", "'");
    return RTV.char(s);
  }

  public makeString(raw: string): RuntimeString {
    let s = raw.slice(1, raw.length - 1);
    s = s.replaceAll('""', '"');
    return RTV.string(s);
  }

  // ── Variables ───────────────────────────────────────────────────────

  public getInitialVariableValue(name: string): RuntimeValue | undefined {
    return this.initialVariableValues
      ? this.initialVariableValues[name]
      : undefined;
  }

  public setVariableValue(name: string, value: RuntimeValue | undefined): void {
    if (value === undefined) return;
    this.variableValues[name] = ensureRuntimeValue(value);
  }

  // ── who() / whos() ─────────────────────────────────────────────────

  /** Collect defined variable names from getters and apply argument filters. */
  private filterVarNames(
    getters: Record<string, () => unknown>,
    args: unknown[]
  ): { names: string[]; values: Map<string, unknown> } {
    const values = new Map<string, unknown>();
    for (const [name, getter] of Object.entries(getters)) {
      const v = getter();
      if (v !== undefined) values.set(name, v);
    }
    let names = [...values.keys()].sort();

    const margs = args.map(a => ensureRuntimeValue(a));
    if (margs.length > 0) {
      const strArgs = margs.map(a => {
        if (isRuntimeChar(a)) return a.value;
        if (isRuntimeString(a)) return a;
        if (typeof a === "string") return a;
        return String(a);
      });

      if (strArgs[0] === "-regexp") {
        const regexps = strArgs.slice(1).map(s => new RegExp(s));
        names = names.filter(name => regexps.some(re => re.test(name)));
      } else {
        const patterns = strArgs.map(s => globToRegExp(s));
        names = names.filter(name => patterns.some(re => re.test(name)));
      }
    }
    return { names, values };
  }

  /** Estimate bytes for a runtime value. */
  private static varBytes(v: unknown): number {
    const rv = ensureRuntimeValue(v);
    if (isRuntimeNumber(rv)) return 8;
    if (isRuntimeLogical(rv)) return 1;
    if (isRuntimeTensor(rv)) {
      const n = rv.data.length;
      return n * 8 + (rv.imag ? n * 8 : 0);
    }
    if (isRuntimeSparseMatrix(rv)) {
      const nnz = rv.pr.length;
      // ir (int32) + pr (float64) per nnz, jc (int32) per column+1, optional pi
      return nnz * 4 + nnz * 8 + (rv.n + 1) * 4 + (rv.pi ? nnz * 8 : 0);
    }
    if (isRuntimeCell(rv)) return rv.data.length * 8;
    if (isRuntimeChar(rv)) return rv.value.length * 2;
    if (isRuntimeString(rv)) return rv.length * 2;
    return 0;
  }

  /** Check if a runtime value is complex. */
  private static varIsComplex(rv: RuntimeValue): boolean {
    if (isRuntimeTensor(rv)) return !!rv.imag;
    if (isRuntimeSparseMatrix(rv)) return !!rv.pi;
    return (
      typeof rv === "object" &&
      rv !== null &&
      "kind" in rv &&
      rv.kind === "complex_number"
    );
  }

  /** Check if a runtime value is sparse. */
  private static varIsSparse(rv: RuntimeValue): boolean {
    return isRuntimeSparseMatrix(rv);
  }

  public who(
    nargout: number,
    getters: Record<string, () => unknown>,
    args: unknown[]
  ): unknown {
    const { names } = this.filterVarNames(getters, args);

    if (nargout >= 1) {
      return _makeCell(
        names.map(n => RTV.char(n)),
        [names.length, 1]
      );
    }
    if (names.length > 0) {
      this.output("Your variables are:\n\n");
      for (const name of names) {
        this.output(name + "  ");
      }
      this.output("\n\n");
    }
    return 0;
  }

  /** Build info record for a single variable. */
  private static varInfo(
    name: string,
    v: unknown
  ): {
    name: string;
    shape: number[];
    bytes: number;
    cls: string;
    complex: boolean;
    sparse: boolean;
  } {
    const rv = ensureRuntimeValue(v);
    return {
      name,
      shape: valueShape(ensureRuntimeValue(v)),
      bytes: Runtime.varBytes(v),
      cls: _numblClass(rv),
      complex: Runtime.varIsComplex(rv),
      sparse: Runtime.varIsSparse(rv),
    };
  }

  private static whosStruct(info: {
    name: string;
    shape: number[];
    bytes: number;
    cls: string;
    complex: boolean;
    sparse: boolean;
  }): RuntimeStruct {
    return RTV.struct(
      new Map<string, RuntimeValue>([
        ["name", RTV.char(info.name)],
        [
          "size",
          RTV.tensor(allocFloat64Array(info.shape), [1, info.shape.length]),
        ],
        ["bytes", RTV.num(info.bytes)],
        ["class", RTV.char(info.cls)],
        ["global", RTV.logical(false)],
        ["sparse", RTV.logical(info.sparse)],
        ["complex", RTV.logical(info.complex)],
        [
          "nesting",
          RTV.struct(
            new Map<string, RuntimeValue>([
              ["function", RTV.char("")],
              ["level", RTV.num(0)],
            ])
          ),
        ],
        ["persistent", RTV.logical(false)],
      ])
    );
  }

  public whos(
    nargout: number,
    getters: Record<string, () => unknown>,
    args: unknown[]
  ): unknown {
    const { names, values } = this.filterVarNames(getters, args);
    const infos = names.map(name => Runtime.varInfo(name, values.get(name)!));

    if (nargout >= 1) {
      if (infos.length === 0) {
        return Runtime.whosStruct({
          name: "",
          shape: [0, 0],
          bytes: 0,
          cls: "",
          complex: false,
          sparse: false,
        });
      }
      const structs = infos.map(info => Runtime.whosStruct(info));
      if (structs.length === 1) return structs[0];
      return RTV.structArray([...structs[0].fields.keys()], structs);
    }

    // Display mode
    if (infos.length === 0) return 0;

    const rows = infos.map(info => {
      const attrs: string[] = [];
      if (info.complex) attrs.push("complex");
      if (info.sparse) attrs.push("sparse");
      return {
        ...info,
        sizeStr: info.shape.join("x"),
        attrsStr: attrs.join(", "),
      };
    });

    const nameW = Math.max(4, ...rows.map(r => r.name.length));
    const sizeW = Math.max(4, ...rows.map(r => r.sizeStr.length));
    const bytesW = Math.max(5, ...rows.map(r => String(r.bytes).length));
    const classW = Math.max(5, ...rows.map(r => r.cls.length));
    const hasAttrs = rows.some(r => r.attrsStr.length > 0);
    const attrsHeader = hasAttrs ? "  Attributes" : "";

    this.output(
      `  ${"Name".padEnd(nameW)}  ${"Size".padStart(sizeW)}  ${"Bytes".padStart(bytesW)}  ${"Class".padEnd(classW)}${attrsHeader}\n\n`
    );
    for (const r of rows) {
      const attrsPart = hasAttrs ? `  ${r.attrsStr}` : "";
      this.output(
        `  ${r.name.padEnd(nameW)}  ${r.sizeStr.padStart(sizeW)}  ${String(r.bytes).padStart(bytesW)}  ${r.cls.padEnd(classW)}${attrsPart}\n`
      );
    }
    this.output("\n");
    return 0;
  }

  // ── Persistent variables ────────────────────────────────────────────

  public getPersistent(funcId: string, varName: string): RuntimeValue {
    const val = this.persistentStore.get(funcId)?.get(varName);
    if (val === undefined) return RTV.tensor(allocFloat64Array(0), [0, 0]);
    return val;
  }

  public setPersistent(
    funcId: string,
    varName: string,
    value: RuntimeValue
  ): void {
    let funcMap = this.persistentStore.get(funcId);
    if (!funcMap) {
      funcMap = new Map();
      this.persistentStore.set(funcId, funcMap);
    }
    const old = funcMap.get(varName);
    incref(value);
    funcMap.set(varName, value);
    if (old !== undefined) decref(this, old);
  }

  // ── Thin wrappers to runtimeOperators ─────────────────────────────

  public uplus(v: unknown): RuntimeValue {
    return _uplus(v);
  }
  public uminus(v: unknown): unknown {
    if (typeof v !== "number") {
      const mv = ensureRuntimeValue(v);
      if (isRuntimeClassInstance(mv)) return this.dispatch("uminus", 1, [v]);
    }
    return _uminus(v);
  }
  public transpose(v: unknown): unknown {
    if (typeof v !== "number") {
      const mv = ensureRuntimeValue(v);
      if (isRuntimeClassInstance(mv) || isRuntimeClassInstanceArray(mv))
        return this.dispatch("transpose", 1, [v]);
    }
    return _transpose(v);
  }
  public ctranspose(v: unknown): unknown {
    if (typeof v !== "number") {
      const mv = ensureRuntimeValue(v);
      if (isRuntimeClassInstance(mv) || isRuntimeClassInstanceArray(mv))
        return this.dispatch("ctranspose", 1, [v]);
    }
    return _ctranspose(v);
  }
  public not(v: unknown): RuntimeLogical | RuntimeTensor {
    return _not(v);
  }
  private static readonly binopMethodMap: Record<string, string> = {
    [BinaryOperation.Add]: "plus",
    [BinaryOperation.Sub]: "minus",
    [BinaryOperation.Mul]: "mtimes",
    [BinaryOperation.Div]: "mrdivide",
    [BinaryOperation.Pow]: "mpower",
    [BinaryOperation.LeftDiv]: "mldivide",
    [BinaryOperation.ElemMul]: "times",
    [BinaryOperation.ElemDiv]: "rdivide",
    [BinaryOperation.ElemPow]: "power",
    [BinaryOperation.ElemLeftDiv]: "ldivide",
    [BinaryOperation.Equal]: "eq",
    [BinaryOperation.NotEqual]: "ne",
    [BinaryOperation.Less]: "lt",
    [BinaryOperation.LessEqual]: "le",
    [BinaryOperation.Greater]: "gt",
    [BinaryOperation.GreaterEqual]: "ge",
    [BinaryOperation.BitAnd]: "and",
    [BinaryOperation.BitOr]: "or",
  };
  public binop(op: string, a: unknown, b: unknown): unknown {
    // Class operator dispatch: if either operand is a class instance,
    // try to call the overloaded operator method
    if (typeof a !== "number" || typeof b !== "number") {
      const ma = typeof a === "number" ? null : ensureRuntimeValue(a);
      const mb = typeof b === "number" ? null : ensureRuntimeValue(b);
      if (
        (ma && isRuntimeClassInstance(ma)) ||
        (mb && isRuntimeClassInstance(mb))
      ) {
        const methodName = Runtime.binopMethodMap[op];
        if (methodName) {
          return this.dispatch(methodName, 1, [a, b]);
        }
      }
    }
    return _binop(op, a, b);
  }
  public range(start: unknown, step: unknown, end: unknown): unknown {
    return _range(start, step, end);
  }
  public forIter(v: unknown): unknown[] {
    return _forIter(v);
  }
  public emptyTensor(): RuntimeTensor {
    return _emptyTensor();
  }
  public emptyStruct(): RuntimeStruct {
    return _emptyStruct();
  }
  public makeCell(elems: unknown[], shape: number[]): RuntimeCell {
    return _makeCell(elems, shape);
  }
  /**
   * Build a struct from trailing name-value pair arguments.
   * @param nvArgs  The rest-parameter array of trailing name-value pairs
   * @param defaults  An object mapping field names to default values
   */
  public buildNameValueStruct(
    nvArgs: unknown[],
    defaults: Record<string, unknown>
  ): RuntimeStruct {
    const fields = new Map<string, RuntimeValue>();
    // Apply defaults first
    for (const [k, v] of Object.entries(defaults)) {
      fields.set(k, ensureRuntimeValue(v));
    }
    // Override with passed name-value pairs
    for (let i = 0; i < nvArgs.length; i += 2) {
      const rv = ensureRuntimeValue(nvArgs[i]);
      const key =
        isRuntimeChar(rv) || isRuntimeString(rv) ? rstr(rv) : String(rv);
      const val = ensureRuntimeValue(nvArgs[i + 1]);
      fields.set(key, val);
    }
    return RTV.struct(fields);
  }
  public horzcat(elems: unknown[]): unknown {
    // Flatten first, then check for class instances
    const flat: unknown[] = [];
    for (const e of elems) {
      if (Array.isArray(e)) flat.push(...e);
      else flat.push(e);
    }
    // Single element: just return it
    if (flat.length === 1) return flat[0];
    const mvals = flat.map(e =>
      typeof e === "number" ? null : ensureRuntimeValue(e)
    );
    if (
      mvals.some(
        v => v && (isRuntimeClassInstance(v) || isRuntimeClassInstanceArray(v))
      )
    ) {
      // If the class overloads horzcat, dispatch to it
      if (this._classHasMethod(mvals, "horzcat")) {
        return this.dispatch("horzcat", 1, flat);
      }
      // Otherwise create a default object array
      return defaultClassInstanceHorzcat(flat);
    }
    return opHorzcat(elems);
  }
  public vertcat(rows: unknown[]): unknown {
    // Single element: just return it (vertcat(x) == x)
    if (rows.length === 1) return rows[0];
    const mvals = rows.map(r =>
      typeof r === "number" ? null : ensureRuntimeValue(r)
    );
    if (
      mvals.some(
        v => v && (isRuntimeClassInstance(v) || isRuntimeClassInstanceArray(v))
      )
    ) {
      // If the class overloads vertcat, dispatch to it
      if (this._classHasMethod(mvals, "vertcat")) {
        return this.dispatch("vertcat", 1, rows);
      }
      // Otherwise create a default object array
      return defaultClassInstanceVertcat(rows);
    }
    return opVertcat(rows);
  }
  /** Check if any class instance in the values has a given method. */
  private _classHasMethod(
    mvals: (RuntimeValue | null)[],
    methodName: string
  ): boolean {
    if (!this.resolveClassMethod) return false;
    for (const v of mvals) {
      if (v && (isRuntimeClassInstance(v) || isRuntimeClassInstanceArray(v))) {
        if (this.resolveClassMethod(v.className, methodName) !== null) {
          return true;
        }
      }
    }
    return false;
  }
  public switchMatch(control: unknown, caseVal: unknown): boolean {
    return _switchMatch(control, caseVal);
  }

  // ── Thin wrappers to runtimeDispatch ──────────────────────────────

  public getFuncHandle(name: string): RuntimeFunction {
    return _getFuncHandle(name);
  }
  public makeUserFuncHandle(
    jsFn: (...args: unknown[]) => unknown,
    nargin?: number
  ): RuntimeFunction {
    return _makeUserFuncHandle(jsFn, nargin);
  }
  public isa(value: unknown, classNameArg: unknown): RuntimeLogical {
    return _isa(this, value, classNameArg);
  }
  public callSuperConstructor(
    target: unknown,
    superInstance: unknown
  ): unknown {
    return _callSuperConstructor(target, superInstance, this);
  }
  public createClassInstance(
    className: string,
    propertyNames: string[],
    defaults?: Record<string, unknown>,
    isHandleClass = false
  ): RuntimeValue {
    return _createClassInstance(
      this,
      className,
      propertyNames,
      defaults,
      isHandleClass
    );
  }
  /**
   * Resolve a class method by name, using cache + JIT compilation.
   * Returns the compiled method function or null if it doesn't exist.
   */
  public cachedResolveClassMethod(
    className: string,
    methodName: string
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ): ((...args: any[]) => any) | null {
    const cacheKey = `${className}.${methodName}`;
    if (this.classMethodCache.has(cacheKey)) {
      return this.classMethodCache.get(cacheKey)!;
    }
    if (!this.resolveClassMethod) return null;
    const fn = this.resolveClassMethod(className, methodName);
    this.classMethodCache.set(cacheKey, fn);
    return fn;
  }

  /** Ancestor class names of a class, or [] if unknown. */
  public classAncestors(className: string): string[] {
    return this.getClassAncestors?.(className) ?? [];
  }

  /** Get the class name from a runtime value (for dynamic static method dispatch). */
  public getClassName(value: unknown): string {
    const mv = ensureRuntimeValue(value);
    if (isRuntimeClassInstance(mv)) return mv.className;
    throw new RuntimeError("Expected a class instance");
  }
  public dispatch(name: string, nargout: number, args: unknown[]): unknown {
    return _dispatch(this, name, nargout, args);
  }
  public methodDispatch(
    name: string,
    nargout: number,
    args: unknown[]
  ): unknown {
    return _methodDispatch(this, name, nargout, args);
  }
  public callBuiltin(name: string, nargout: number, args: unknown[]): unknown {
    return _callBuiltin(this, name, nargout, args);
  }
  public callClassMethod(
    className: string,
    methodName: string,
    nargout: number,
    args: unknown[]
  ): unknown {
    return _callClassMethod(this, className, methodName, nargout, args);
  }

  // ── Indexing ────────────────────────────────────────────────────────

  public index(
    base: unknown,
    indices: unknown[],
    nargout: number = 1,
    skipSubsref: boolean | string = false
  ): unknown {
    return _index(this, base, indices, nargout, skipSubsref);
  }

  public indexCell(base: unknown, indices: unknown[]): unknown {
    return _indexCell(this, base, indices);
  }

  /** Normalize a cell indexing result into an array for CSL spreading.
   *  If the result is already an array (multi-element indexing), return it as-is.
   *  If it's a single value (scalar indexing), wrap it in an array. */
  public asCsl(value: unknown): unknown[] {
    return Array.isArray(value) ? value : [value];
  }

  /** Like index() but returns an empty struct instead of throwing.
   *  Used for chained lvalue assignment: V(i).field = rhs where V may be uninitialized. */
  public indexOrEmpty(base: unknown, indices: unknown[]): unknown {
    if (base === undefined || base === null) return _emptyStruct();
    try {
      return _index(this, base, indices, 1);
    } catch {
      return _emptyStruct();
    }
  }

  /** Like indexOrEmpty() but uses built-in indexing (bypasses subsref) for
   *  class instances.  For scalar class instances, V(1) returns V itself.
   *  Used in V(i).field = rhs patterns where the () part should always use
   *  built-in array indexing, not the class's overloaded subsref. */
  public builtinIndexOrEmpty(base: unknown, indices: unknown[]): unknown {
    if (base === undefined || base === null) return _emptyStruct();
    try {
      return _index(this, base, indices, 1, true);
    } catch {
      return _emptyStruct();
    }
  }

  /** Like indexCell() but returns 0 instead of throwing.
   *  Used for chained lvalue assignment: data{i}.field = rhs where data may be uninitialized. */
  public indexCellOrEmpty(base: unknown, indices: unknown[]): unknown {
    try {
      return _indexCell(this, base, indices);
    } catch {
      return 0;
    }
  }

  public indexStore(
    base: unknown,
    indices: unknown[],
    rhs: unknown,
    skipSubsasgn = false,
    storeBack = false
  ): unknown {
    return _indexStore(this, base, indices, rhs, skipSubsasgn, storeBack);
  }

  /** Built-in index store for V(i).field = rhs patterns.
   *  For class instances, the element was already modified via setMemberReturn
   *  so we just return the new value.  For other types, delegate to indexStore. */
  public builtinIndexStore(
    base: unknown,
    indices: unknown[],
    value: unknown
  ): unknown {
    const mv = ensureRuntimeValue(base);
    if (isRuntimeClassInstance(mv)) {
      return value;
    }
    return _indexStore(this, base, indices, value, false, true);
  }

  /** Check if a value is a class instance of the given class name.
   *  Used by codegen for runtime skipSubsasgn checks on Unknown-type variables. */
  public isClassInstance(value: unknown, className: string): boolean {
    const mv = ensureRuntimeValue(value);
    return isRuntimeClassInstance(mv) && mv.className === className;
  }

  public indexCellStore(
    base: unknown,
    indices: unknown[],
    rhs: unknown
  ): unknown {
    return _indexCellStore(base, indices, rhs, this);
  }

  public multiOutputCellAssign(
    base: unknown,
    indices: unknown,
    results: unknown[]
  ): unknown {
    return _multiOutputCellAssign(base, indices, results, this);
  }

  // ── Member access ───────────────────────────────────────────────────

  public getMember(base: unknown, name: string, nargout = 1): unknown {
    return _getMember(this, base, name, nargout);
  }

  public getMemberDynamic(base: unknown, nameExpr: unknown): RuntimeValue {
    return _getMemberDynamic(this, base, nameExpr);
  }

  public getMemberOrEmpty(base: unknown, name: string): RuntimeValue {
    return _getMemberOrEmpty(base, name);
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  public setMember(_base: unknown, _name: string, _rhs: unknown): void {
    throw this.error("Cannot assign to member of temporary expression");
  }

  public setMemberReturn(base: unknown, name: string, rhs: unknown): unknown {
    return _setMemberReturn(this, base, name, rhs);
  }

  public setMemberDynamicReturn(
    base: unknown,
    nameExpr: unknown,
    rhs: unknown
  ): RuntimeValue {
    return _setMemberDynamicReturn(this, base, nameExpr, rhs);
  }

  public subsrefCall(base: unknown, names: string[]): unknown {
    return _subsrefCall(this, base, names);
  }

  public subsasgnCall(base: unknown, names: string[], rhs: unknown): unknown {
    return _subsasgnCall(this, base, names, rhs);
  }

  public memberChainAssign(
    base: unknown,
    names: string[],
    rhs: unknown
  ): unknown {
    return _memberChainAssign(this, base, names, rhs);
  }

  // ── narginchk / nargoutchk ──────────────────────────────────────────

  public narginchk(
    actualNargin: unknown,
    minArgs: unknown,
    maxArgs: unknown
  ): void {
    const n =
      typeof actualNargin === "number"
        ? actualNargin
        : toNumber(actualNargin as RuntimeValue);
    const lo =
      typeof minArgs === "number" ? minArgs : toNumber(minArgs as RuntimeValue);
    const hi =
      typeof maxArgs === "number" ? maxArgs : toNumber(maxArgs as RuntimeValue);
    if (n < lo) throw new RuntimeError("Not enough input arguments.");
    if (n > hi) throw new RuntimeError("Too many input arguments.");
  }

  public nargoutchk(
    actualNargout: unknown,
    minArgs: unknown,
    maxArgs: unknown
  ): void {
    const n =
      typeof actualNargout === "number"
        ? actualNargout
        : toNumber(actualNargout as RuntimeValue);
    const lo =
      typeof minArgs === "number" ? minArgs : toNumber(minArgs as RuntimeValue);
    const hi =
      typeof maxArgs === "number" ? maxArgs : toNumber(maxArgs as RuntimeValue);
    if (n < lo) throw new RuntimeError("Not enough output arguments.");
    if (n > hi) throw new RuntimeError("Too many output arguments.");
  }

  // ── Plot helpers (state-bearing) ────────────────────────────────────

  /** Read the hold state, for `ishold()` queries. */
  public ishold(): RuntimeValue {
    return RTV.logical(this.holdState);
  }

  /** Current axis limits of the active axes, as MATLAB's `axis` query
   *  returns them: `[xmin xmax ymin ymax]` (2-D) or with `zmin zmax`
   *  appended (3-D). Reduces the accumulated plot instructions into a figure
   *  state and reads the current axes' explicit/data-derived limits. */
  public currentAxisLimits(): number[] {
    let state = initialFiguresState;
    for (const instr of this.plotInstructions) {
      state = figuresReducer(state, instr);
    }
    const fig = state.figs[state.currentHandle];
    if (!fig) return [0, 1, 0, 1];
    const axes = fig.axes[fig.currentAxesIndex];
    if (!axes) return [0, 1, 0, 1];
    return computeAxisLimits(axes);
  }

  // ── Drawnow / Pause ─────────────────────────────────────────────────

  public drawnow(): void {
    if (this.options.onDrawnow && this.plotInstructions.length > 0) {
      this.options.onDrawnow([...this.plotInstructions]);
      this.plotInstructions.length = 0;
    }
  }

  public pause(seconds: unknown): void {
    const s =
      typeof seconds === "number"
        ? seconds
        : toNumber(ensureRuntimeValue(seconds));
    syncSleep(s * 1000);
  }

  public readInput(prompt: string): string {
    if (!this.options.onInput) {
      throw new RuntimeError("input() is not available in this environment");
    }
    return this.options.onInput(prompt);
  }
}

export const rstr = (s: RuntimeString | RuntimeChar): string => {
  if (isRuntimeString(s)) return s;
  if (isRuntimeChar(s)) return s.value;
  throw new RuntimeError(`Expected string or char, got ${kstr(s)}`);
};

// ── Default class instance array concatenation ─────────────────────────

interface ObjArrayParts {
  rows: number;
  cols: number;
  elements: RuntimeClassInstance[]; // column-major
  className: string;
}

/** View an operand as 2-D object-array parts (column-major). Scalars are
 *  1×1; numeric `[]` is treated as an ignorable empty. */
function asObjArrayParts(item: unknown): ObjArrayParts | null {
  const rv = ensureRuntimeValue(item);
  if (isRuntimeClassInstance(rv))
    return { rows: 1, cols: 1, elements: [rv], className: rv.className };
  if (isRuntimeClassInstanceArray(rv))
    return {
      rows: rv.shape[0],
      cols: rv.shape[1],
      elements: rv.elements,
      className: rv.className,
    };
  // Empty numeric [] is dropped from concatenation (MATLAB semantics).
  if (isRuntimeTensor(rv) && rv.data.length === 0) return null;
  throw new RuntimeError(`Cannot concatenate ${kstr(rv)} with class instances`);
}

/** Extract column `j` (0-based) of a column-major operand. */
function objColumn(p: ObjArrayParts, j: number): RuntimeClassInstance[] {
  return p.elements.slice(j * p.rows, j * p.rows + p.rows);
}

/** Default horzcat for class instances: concatenates operands along columns
 *  (dim 2). Rows must agree. Column-major storage means appending columns is
 *  simply appending element lists. */
function defaultClassInstanceHorzcat(
  items: unknown[]
): RuntimeClassInstance | RuntimeClassInstanceArray {
  const parts = items
    .map(asObjArrayParts)
    .filter((p): p is ObjArrayParts => p !== null && p.elements.length > 0);
  if (parts.length === 0)
    throw new RuntimeError("Cannot concatenate empty class instances");
  const rows = parts[0].rows;
  let cols = 0;
  const elements: RuntimeClassInstance[] = [];
  for (const p of parts) {
    if (p.rows !== rows)
      throw new RuntimeError(
        "Dimensions of arrays being concatenated are not consistent"
      );
    elements.push(...p.elements);
    cols += p.cols;
  }
  if (elements.length === 1) return elements[0];
  return new RuntimeClassInstanceArray(parts[0].className, elements, [
    rows,
    cols,
  ]);
}

/** Default vertcat for class instances: concatenates operands along rows
 *  (dim 1). Columns must agree. Column-major storage means each output column
 *  interleaves that column from every operand top to bottom. */
function defaultClassInstanceVertcat(
  rows: unknown[]
): RuntimeClassInstance | RuntimeClassInstanceArray {
  const parts = rows
    .map(asObjArrayParts)
    .filter((p): p is ObjArrayParts => p !== null && p.elements.length > 0);
  if (parts.length === 0)
    throw new RuntimeError("Cannot concatenate empty class instances");
  const cols = parts[0].cols;
  let totalRows = 0;
  for (const p of parts) {
    if (p.cols !== cols)
      throw new RuntimeError(
        "Dimensions of arrays being concatenated are not consistent"
      );
    totalRows += p.rows;
  }
  const elements: RuntimeClassInstance[] = [];
  for (let j = 0; j < cols; j++) {
    for (const p of parts) elements.push(...objColumn(p, j));
  }
  if (elements.length === 1) return elements[0];
  return new RuntimeClassInstanceArray(parts[0].className, elements, [
    totalRows,
    cols,
  ]);
}
